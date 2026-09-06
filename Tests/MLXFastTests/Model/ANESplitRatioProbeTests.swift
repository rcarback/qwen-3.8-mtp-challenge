import CoreML
import Foundation
import IOSurface
import MLX
import Testing

@testable import MLXFastModel

/// Re-measures `r`, the ANE-to-GPU throughput ratio, with the two things the
/// 2026-09-04 measurement (`ANEExpertGemmRatioTests`, commit 10350f84) did not
/// have: surface-backed zero-copy I/O on the ANE arm (that run paid a Core ML
/// input and output copy per call, 6.5 MB at S=1024) and a GPU arm at the
/// PRODUCTION quantization of each projection (that run used dense fp16 on the
/// GPU, which understates the GPU). It also sweeps the ANE weight form (fp16
/// against int4 palette with a per-channel scale) and, at one shape, the
/// spatial layout of the activation, because the conv is position-wise and
/// `[1,K,H,W]` with `H*W=S` computes the same projection as `[1,K,1,S]`.
///
/// Packages come from `gen_r_pkgs.py`; every package has a sidecar with the
/// input, the effective fp16 weight and an fp32 reference, and every shape has
/// a `<shape>.w_f16.bin` plus `<shape>.shape.json` naming the GPU form.
///
///   MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_NO_SANDBOX=1 \
///     MLXFAST_ANE_R_DIR=<dir> swift test -c release \
///     --force-resolved-versions --filter ANESplitRatioProbeTests
@Suite(.serialized)
struct ANESplitRatioProbeTests {
    static var enabled: Bool {
        let env = ProcessInfo.processInfo.environment
        return env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" && env["MLXFAST_ANE_R_DIR"] != nil
    }

    private final class Box<T>: @unchecked Sendable {
        var value: T?
        var error: Error?
    }

    private func awaitSync<T>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
        let sema = DispatchSemaphore(value: 0)
        let box = Box<T>()
        Task.detached {
            do { box.value = try await body() } catch { box.error = error }
            sema.signal()
        }
        sema.wait()
        if let e = box.error { throw e }
        guard let v = box.value else {
            throw NSError(domain: "ANESplitRatioProbe", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "async call produced neither value nor error"])
        }
        return v
    }

    private func makeSurface(bytes: Int) -> IOSurface {
        let alloc = max(65536, (bytes + 65535) & ~65535)
        let props: NSDictionary = [
            kIOSurfaceWidth: alloc, kIOSurfaceHeight: 1, kIOSurfaceBytesPerElement: 1,
            kIOSurfaceBytesPerRow: alloc, kIOSurfaceAllocSize: alloc, kIOSurfacePixelFormat: 0,
        ]
        return IOSurfaceCreate(props as CFDictionary)!
    }

    private func median(_ f: () throws -> Void, warm: Int, n: Int) rethrows -> Double {
        for _ in 0 ..< warm { try f() }
        var t: [Double] = []
        for _ in 0 ..< n {
            let s = DispatchTime.now().uptimeNanoseconds
            try f()
            t.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6)
        }
        t.sort()
        return t[n / 2]
    }

    private struct ANEResult { let ms: Double; let rel: Float; let meanRel: Float }

    /// Loads one package on the ANE, fills the input surface from the sidecar,
    /// checks the output against the fp32 reference and times `prediction`.
    private func runANE(pkg: URL, side: URL) throws -> ANEResult {
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: side.appendingPathComponent("meta.json"))) as! [String: Any]
        let O = meta["O"] as! Int, K = meta["K"] as! Int, S = meta["S"] as! Int
        let H = meta["H"] as? Int ?? 1, W = meta["W"] as? Int ?? S
        precondition(H * W == S)
        let x = try Data(contentsOf: side.appendingPathComponent("x_f16.bin"))
        let yref = try Data(contentsOf: side.appendingPathComponent("y_ref_f32.bin"))
        precondition(x.count == S * K * 2 && yref.count == S * O * 4, "sidecar sizes mismatch for \(pkg.lastPathComponent)")

        let compiled = try awaitSync { try await MLModel.compileModel(at: pkg) }
        let model = try awaitSync {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndNeuralEngine
            return try await MLModel.load(contentsOf: compiled, configuration: cfg)
        }

        let inSurf = makeSurface(bytes: K * S * 2)
        let outSurf = makeSurface(bytes: O * S * 2)
        inSurf.lock(options: [], seed: nil)
        x.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            let sp = src.bindMemory(to: UInt16.self)
            let dp = inSurf.baseAddress.bindMemory(to: UInt16.self, capacity: K * S)
            for s in 0 ..< S { for k in 0 ..< K { dp[k * S + s] = sp[s * K + k] } }
        }
        inSurf.unlock(options: [], seed: nil)
        let inMA = try MLMultiArray(
            dataPointer: inSurf.baseAddress, shape: [1, K, H, W].map { NSNumber(value: $0) },
            dataType: .float16, strides: [K * S, S, W, 1].map { NSNumber(value: $0) })
        let outMA = try MLMultiArray(
            dataPointer: outSurf.baseAddress, shape: [1, O, H, W].map { NSNumber(value: $0) },
            dataType: .float16, strides: [O * S, S, W, 1].map { NSNumber(value: $0) })
        let inName = model.modelDescription.inputDescriptionsByName.keys.first!
        let outName = model.modelDescription.outputDescriptionsByName.keys.first!
        let provider = try MLDictionaryFeatureProvider(dictionary: [inName: MLFeatureValue(multiArray: inMA)])
        let opts = MLPredictionOptions()
        opts.outputBackings = [outName: outMA]
        _ = try model.prediction(from: provider, options: opts)

        var maxAbs: Float = 0, refMax: Float = 0, sumAbs: Float = 0, sumRef: Float = 0
        outSurf.lock(options: [], seed: nil)
        let op = outSurf.baseAddress.bindMemory(to: Float16.self, capacity: O * S)
        yref.withUnsafeBytes { (rb: UnsafeRawBufferPointer) in
            let rp = rb.bindMemory(to: Float.self)
            for s in 0 ..< S {
                for o in 0 ..< O {
                    let d = abs(Float(op[o * S + s]) - rp[s * O + o])
                    maxAbs = max(maxAbs, d); refMax = max(refMax, abs(rp[s * O + o]))
                    sumAbs += d; sumRef += abs(rp[s * O + o])
                }
            }
        }
        outSurf.unlock(options: [], seed: nil)
        let ms = try median({ _ = try model.prediction(from: provider, options: opts) }, warm: 10, n: 40)
        return ANEResult(ms: ms, rel: maxAbs / max(refMax, 1e-9), meanRel: sumAbs / max(sumRef, 1e-9))
    }

    @Test("r re-measured: zero-copy ANE at fp16 and int4 vs production-quantized GPU", .enabled(if: enabled))
    func splitRatio() throws {
        try #require(ANERuntime.available())
        guard #available(macOS 15.0, *) else { return }
        let dir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["MLXFAST_ANE_R_DIR"]!)
        let shapes = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasSuffix(".shape.json") }
            .map { $0.lastPathComponent.replacingOccurrences(of: ".shape.json", with: "") }
            .sorted()
        let onlyShape = ProcessInfo.processInfo.environment["MLXFAST_ANE_R_SHAPE"]
        var rows: [String] = []
        for shape in shapes where onlyShape == nil || shape.contains(onlyShape!) {
            let sj = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent(shape + ".shape.json"))) as! [String: Any]
            let O = sj["O"] as! Int, K = sj["K"] as! Int, bits = sj["gpu_bits"] as! Int, group = sj["gpu_group"] as! Int
            // GPU arm: the SAME tensor quantized at the projection's production form.
            let wBytes = try Data(contentsOf: dir.appendingPathComponent(shape + ".w_f16.bin"))
            let w = wBytes.withUnsafeBytes { MLXArray($0.bindMemory(to: Float16.self), [O, K]) }
            let (wq, sc, bi) = quantized(w, groupSize: group, bits: bits)
            eval(wq, sc, bi)

            let pkgs = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "mlpackage" && $0.lastPathComponent.hasPrefix(shape + "__") }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            for pkg in pkgs {
                let name = pkg.deletingPathExtension().lastPathComponent
                let side = dir.appendingPathComponent(name + ".probe")
                let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: side.appendingPathComponent("meta.json"))) as! [String: Any]
                let S = meta["S"] as! Int
                let xBytes = try Data(contentsOf: side.appendingPathComponent("x_f16.bin"))
                let x = xBytes.withUnsafeBytes { MLXArray($0.bindMemory(to: Float16.self), [S, K]) }
                eval(x)
                let gpuMs = median({
                    let y = quantizedMatmul(x, wq, scales: sc, biases: bi, transpose: true, groupSize: group, bits: bits)
                    eval(y)
                }, warm: 10, n: 40)
                let flops = 2.0 * Double(O) * Double(K) * Double(S)
                var line = String(format: "%@  S=%d  GPU q%d g%d %.3f ms (%.2f TF/s)", name, S, bits, group, gpuMs, flops / gpuMs / 1e9)
                do {
                    let a = try runANE(pkg: pkg, side: side)
                    let r = gpuMs / a.ms
                    line += String(format: " | ANE %.3f ms (%.2f TF/s) rel %.4f meanRel %.4f | r=%.3f f*=%.3f ceiling=%.2fx",
                                   a.ms, flops / a.ms / 1e9, a.rel, a.meanRel, r, r / (1 + r), 1 + r)
                } catch {
                    line += " | ANE FAILED: \(error)"
                }
                print("[r2] " + line)
                rows.append(line)
            }
        }
        print("[r2] ===== SUMMARY =====")
        for r in rows { print("[r2] " + r) }
    }
}
