import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXFastCore

/// What memory bandwidth does one decode row actually achieve, per weight
/// family, on this machine?
///
/// On disk only the routed experts are quantized; every dense tensor is bf16,
/// and the dense tensors are 85 percent of the bytes a decode token touches
/// (8.78 GB of 10.29). The per-layer instrument puts a linear-attention layer
/// at 1.18 ms against a ~0.42 ms byte floor at 450 GB/s. Whether that 2.8x is
/// a bandwidth-inefficient bf16 gemv or launch overhead decides what to fix,
/// so this measures the gemv's achieved GB/s at the real decode shapes, with
/// 36 distinct weight matrices per arm so nothing is served from cache and
/// ONE eval per arm so the isolated-eval floor does not contaminate it.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test -c release \
///       --force-resolved-versions --filter DecodeBandwidth
final class DecodeBandwidthTests: XCTestCase {
    func testAchievedBandwidthAtOneRow() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "needs a GPU")

        let layers = 36
        let hidden = 2560
        // in_proj_qkv of a gated-delta layer: the single largest dense weight
        // the decode path touches, 36 times per token.
        let out = 10240
        let x = MLXRandom.normal([1, hidden]).asType(.bfloat16)
        eval(x)

        func bench(_ label: String, bytesPerLayer: Double, build: () -> [() -> MLXArray]) {
            let fns = build()
            // Warm the kernel families and the pages, then time a whole chain
            // of 36 layers under one eval, five times, keeping the best.
            func run() -> MLXArray {
                var acc = MLXArray(Float(0)).asType(.bfloat16)
                for f in fns { acc = acc + f().sum() }
                return acc
            }
            eval(run())
            var best = Double.greatestFiniteMagnitude
            for _ in 0 ..< 5 {
                let t0 = DispatchTime.now().uptimeNanoseconds
                eval(run())
                best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
            }
            let bytes = bytesPerLayer * Double(layers)
            print(String(
                format: "  %-34@ %6.2f ms / %d layers  %7.1f MB each  -> %6.1f GB/s   %.3f ms per layer",
                label as NSString, best * 1000, layers, bytesPerLayer / 1e6, bytes / best / 1e9,
                best * 1000 / Double(layers)))
        }

        print("[decode-bw] one row, \(layers) distinct weights per arm, one eval per chain")

        // bf16 dense, as shipped
        bench("bf16 gemv [1,2560]x[2560,10240]", bytesPerLayer: Double(hidden * out * 2)) {
            (0 ..< layers).map { _ in
                let w = MLXRandom.normal([out, hidden]).asType(.bfloat16)
                eval(w)
                return { matmul(x, w.T) }
            }
        }

        // 4-bit affine g32, what upstream's quant_predicate would produce
        for (bits, gs) in [(4, 32), (4, 64), (8, 32)] {
            let wBytes = Double(hidden * out) * Double(bits) / 8
            let sBytes = Double(hidden * out / gs) * 2 * 2  // fp16 scale + bias per group
            bench("q\(bits) g\(gs) qmv same shape", bytesPerLayer: wBytes + sBytes) {
                (0 ..< layers).map { _ in
                    let w = MLXRandom.normal([out, hidden]).asType(.float16)
                    let (wq, s, b) = quantized(w, groupSize: gs, bits: bits)
                    eval(wq, s, b!)
                    let xh = x.asType(.float16)
                    return {
                        quantizedMatmul(
                            xh, wq, scales: s, biases: b!, transpose: true, groupSize: gs, bits: bits)
                    }
                }
            }
        }

        // pure copy: the ceiling this machine will give any kernel
        bench("memcpy-class (a + 0) on 52 MB", bytesPerLayer: Double(hidden * out * 2) * 2) {
            (0 ..< layers).map { _ in
                let w = MLXRandom.normal([out, hidden]).asType(.bfloat16)
                eval(w)
                return { w + 0 }
            }
        }
    }
}
