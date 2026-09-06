import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM

/// Does the memory system deliver more than the GPU alone pulls when a second
/// engine streams at the same time?
///
/// The GPU reaches 230 to 440 GB/s of the M4 Max's 546 GB/s on decode-shaped
/// weight streams. Whether the remaining fabric bandwidth is reachable by the
/// ANE or the CPU concurrently -- or whether the GPU's figure is already the
/// contention ceiling -- decides if any engine split can add bandwidth at all.
///
/// Three engines, each looping over its own multi-GB working set: the GPU on a
/// chain of bf16 GEMVs (one eval per chain), the ANE on one fp16 1x1-conv
/// program with a 1.27 GB weight (the lm_head shape) through the full
/// makeInput/predict/readOutput cycle, and the CPU summing a 2 GB byte buffer
/// on four threads. Alone, then in every pair, then all three, each for a fixed
/// window; the report is each engine's achieved GB/s inside the window and the
/// sum.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test -c release \
///       --force-resolved-versions --filter MemoryHeadroom
final class MemoryHeadroomTests: XCTestCase {
    func testConcurrentEngineBandwidth() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "needs a GPU")

        // GPU working set: 24 distinct bf16 [10240, 2560] weights, 1.26 GB.
        let k = 2560, n = 10240, distinct = 24
        let x = MLXRandom.normal([1, k]).asType(.bfloat16)
        var weights = [MLXArray]()
        for _ in 0 ..< distinct {
            let w = MLXRandom.normal([n, k]).asType(.bfloat16)
            eval(w)
            weights.append(w)
        }
        eval(x)
        let gpuBytesPerChain = Double(n * k * 2 * distinct)
        func gpuChain() -> MLXArray {
            var acc = MLXArray(Float(0)).asType(.bfloat16)
            for w in weights { acc = acc + matmul(x, w.T).sum() }
            return acc
        }
        eval(gpuChain())

        // ANE working set: one fp16 program with a [248320, 2560] weight.
        let aneN = 248320
        let aneWeight = MLXRandom.normal([aneN, k])
        eval(aneWeight)
        let projection = try Qwen4ExpANEProjection(weight: aneWeight, sequenceLength: 1)
        let aneBytesPerCall = Double(aneN * k * 2)
        let aneX = MLXRandom.normal([1, k]).asType(.bfloat16)
        eval(aneX)
        do {
            let p = try projection.makeInput(aneX)
            _ = try projection.predict(p)
            _ = projection.readOutput(p)
        }

        // CPU working set: 2 GB of bytes, summed on four threads.
        let cpuBytes = 2 << 30
        let cpuBuffer = UnsafeMutableRawPointer.allocate(byteCount: cpuBytes, alignment: 16384)
        memset(cpuBuffer, 1, cpuBytes)
        defer { cpuBuffer.deallocate() }
        func cpuPass() {
            let threads = 4
            let slice = cpuBytes / threads
            DispatchQueue.concurrentPerform(iterations: threads) { t in
                let base = cpuBuffer.advanced(by: t * slice).assumingMemoryBound(to: UInt64.self)
                var acc: UInt64 = 0
                for i in stride(from: 0, to: slice / 8, by: 4) {
                    acc &+= base[i] &+ base[i + 1] &+ base[i + 2] &+ base[i + 3]
                }
                if acc == 42 { print("") }
            }
        }
        cpuPass()

        let window = 4.0
        func run(gpu: Bool, ane: Bool, cpu: Bool) -> (Double, Double, Double) {
            let deadline = CFAbsoluteTimeGetCurrent() + window
            var gpuBytes = 0.0, aneBytes = 0.0, cpuBytesDone = 0.0
            let group = DispatchGroup()
            let lock = NSLock()
            if ane {
                group.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    var local = 0.0
                    while CFAbsoluteTimeGetCurrent() < deadline {
                        guard let p = try? projection.makeInput(aneX), (try? projection.predict(p)) != nil
                        else { break }
                        _ = projection.readOutput(p)
                        local += aneBytesPerCall
                    }
                    lock.lock(); aneBytes = local; lock.unlock()
                    group.leave()
                }
            }
            if cpu {
                group.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    var local = 0.0
                    while CFAbsoluteTimeGetCurrent() < deadline {
                        cpuPass()
                        local += Double(cpuBytes)
                    }
                    lock.lock(); cpuBytesDone = local; lock.unlock()
                    group.leave()
                }
            }
            if gpu {
                while CFAbsoluteTimeGetCurrent() < deadline {
                    eval(gpuChain())
                    gpuBytes += gpuBytesPerChain
                }
            }
            group.wait()
            return (gpuBytes / window / 1e9, aneBytes / window / 1e9, cpuBytesDone / window / 1e9)
        }

        print("[headroom] \(window) s windows; GB/s achieved by each engine inside the window")
        let modes: [(String, Bool, Bool, Bool)] = [
            ("gpu", true, false, false), ("ane", false, true, false), ("cpu", false, false, true),
            ("gpu+ane", true, true, false), ("gpu+cpu", true, false, true), ("ane+cpu", false, true, true),
            ("gpu+ane+cpu", true, true, true), ("gpu (again)", true, false, false),
        ]
        for (name, g, a, c) in modes {
            let (gb, ab, cb) = run(gpu: g, ane: a, cpu: c)
            print(String(
                format: "  %-12@ gpu %6.1f  ane %6.1f  cpu %6.1f   sum %6.1f GB/s",
                name as NSString, gb, ab, cb, gb + ab + cb))
        }
    }
}
