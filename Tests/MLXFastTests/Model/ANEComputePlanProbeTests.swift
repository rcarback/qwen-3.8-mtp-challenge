import CoreML
import Foundation
import IOSurface
import Testing

@testable import MLXFastModel

/// Asks Core ML's compute plan which operations of a set of `.mlpackage`
/// probes the ANE takes, then runs each package on the ANE against a numpy
/// reference and times it against the CPU-only load of the same package.
///
/// The packages are built by `gen_probe_pkgs.py` (coremltools 9, iOS18
/// opset): the same fp32 weight stored as fp16, int8 per-channel affine,
/// int8/int4 blockwise (the MLX affine group form, with an fp16 offset), int4
/// palettes (per-tensor, per-64-rows, int8 codebook with a per-channel scale,
/// and fp16 LUT with a runtime per-channel `mul`), plus one program of
/// miscellaneous ops for eligibility only. Each weight package has a sidecar
/// `<name>.probe/` with `w_eff_f16.bin`, `x_f16.bin` and `y_ref_f32.bin`.
///
/// Why this exists: the in-memory `_ANEInMemoryModel` compiler rejects the
/// iOS18 blockwise op with a byte-identical blob. Whether the `.mlpackage`
/// compile path places it on the ANE decides whether the ANE can hold the
/// exact q4 group-32 / q4 group-64 / q8 group-32 tensors the GPU holds.
///
///   MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_NO_SANDBOX=1 \
///     MLXFAST_ANE_PKG_DIR=<dir> swift test -c release \
///     --force-resolved-versions --filter ANEComputePlanProbeTests
@Suite(.serialized)
struct ANEComputePlanProbeTests {
    static var enabled: Bool {
        let env = ProcessInfo.processInfo.environment
        return env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" && env["MLXFAST_ANE_PKG_DIR"] != nil
    }

    private final class Box<T>: @unchecked Sendable {
        var value: T?
        var error: Error?
    }

    /// Blocks on an async Core ML call from a synchronous test, off the
    /// caller's actor, matching the sibling probes.
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
            throw NSError(domain: "ANEComputePlanProbe", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "async call produced neither value nor error"])
        }
        return v
    }

    private func deviceName(_ d: MLComputeDevice) -> String {
        switch d {
        case .cpu: return "cpu"
        case .gpu: return "gpu"
        case .neuralEngine: return "ane"
        @unknown default: return "?"
        }
    }

    private func makeSurface(bytes: Int) -> IOSurface {
        let alloc = max(65536, (bytes + 65535) & ~65535)
        let props: NSDictionary = [
            kIOSurfaceWidth: alloc, kIOSurfaceHeight: 1, kIOSurfaceBytesPerElement: 1,
            kIOSurfaceBytesPerRow: alloc, kIOSurfaceAllocSize: alloc, kIOSurfacePixelFormat: 0,
        ]
        return IOSurfaceCreate(props as CFDictionary)!
    }

    private func readBytes(_ url: URL) throws -> Data { try Data(contentsOf: url) }

    /// Prints one line per operation of the program with the plan's preferred
    /// and supported devices. Returns the set of operator names the ANE is the
    /// preferred device for.
    @available(macOS 15.0, *)
    private func printPlan(compiled: URL, label: String) throws -> (ane: [String], nonAne: [String]) {
        // The configuration is built inside the detached task: MLModelConfiguration
        // is not Sendable, and the units value is.
        let plan = try awaitSync {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .all
            return try await MLComputePlan.load(contentsOf: compiled, configuration: cfg)
        }
        var ane: [String] = []
        var nonAne: [String] = []
        guard case .program(let program) = plan.modelStructure else {
            print("[plan] \(label): not an ML program")
            return ([], [])
        }
        for (fname, fn) in program.functions.sorted(by: { $0.key < $1.key }) {
            for op in fn.block.operations {
                let outs = op.outputs.map(\.name).joined(separator: ",")
                if let u = plan.deviceUsage(for: op) {
                    let sup = u.supported.map(deviceName).joined(separator: "/")
                    let pref = deviceName(u.preferred)
                    print("[plan] \(label) \(fname) \(op.operatorName)(\(outs)) preferred=\(pref) supported=\(sup)")
                    if pref == "ane" { ane.append(op.operatorName) } else { nonAne.append("\(op.operatorName):\(pref)") }
                } else {
                    // const ops carry no device usage; skip silently.
                    if op.operatorName != "const" {
                        print("[plan] \(label) \(fname) \(op.operatorName)(\(outs)) no-device-usage")
                    }
                }
            }
        }
        return (ane, nonAne)
    }

    private func load(compiled: URL, units: MLComputeUnits) throws -> MLModel {
        try awaitSync {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = units
            return try await MLModel.load(contentsOf: compiled, configuration: cfg)
        }
    }

    /// Runs one weight-form package on `units` with surface-backed I/O.
    /// Returns (maxAbs, relToRefMax, meanAbsRel, medianMs).
    private func runProbe(model: MLModel, side: URL, O: Int, K: Int, S: Int, iters: Int)
        throws -> (maxAbs: Float, rel: Float, meanRel: Float, ms: Double)
    {
        let x = try readBytes(side.appendingPathComponent("x_f16.bin"))       // [S,K] fp16
        // A timing-only probe (bead i6v's whole-layer program) ships no reference.
        let refURL = side.appendingPathComponent("y_ref_f32.bin")
        let yref = FileManager.default.fileExists(atPath: refURL.path) ? try readBytes(refURL) : Data()
        precondition(x.count == S * K * 2 && (yref.isEmpty || yref.count == S * O * 4), "sidecar sizes mismatch")

        let inSurf = makeSurface(bytes: K * S * 2)
        let outSurf = makeSurface(bytes: O * S * 2)
        // x[S,K] -> input surface [1,K,1,S]: transpose on the CPU once.
        inSurf.lock(options: [], seed: nil)
        x.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            let sp = src.bindMemory(to: UInt16.self)
            let dp = inSurf.baseAddress.bindMemory(to: UInt16.self, capacity: K * S)
            for s in 0 ..< S { for k in 0 ..< K { dp[k * S + s] = sp[s * K + k] } }
        }
        inSurf.unlock(options: [], seed: nil)

        let inMA = try MLMultiArray(
            dataPointer: inSurf.baseAddress, shape: [1, K, 1, S].map { NSNumber(value: $0) },
            dataType: .float16, strides: [K * S, S, S, 1].map { NSNumber(value: $0) })
        let outMA = try MLMultiArray(
            dataPointer: outSurf.baseAddress, shape: [1, O, 1, S].map { NSNumber(value: $0) },
            dataType: .float16, strides: [O * S, S, S, 1].map { NSNumber(value: $0) })
        let inName = model.modelDescription.inputDescriptionsByName.keys.first!
        let outNames = model.modelDescription.outputDescriptionsByName.keys.sorted()
        let outName = outNames.contains("y") ? "y" : outNames.first!
        let provider = try MLDictionaryFeatureProvider(dictionary: [inName: MLFeatureValue(multiArray: inMA)])
        let opts = MLPredictionOptions()
        opts.outputBackings = [outName: outMA]

        _ = try model.prediction(from: provider, options: opts)

        // Numerics against the fp32 numpy reference y_ref[s,o].
        var maxAbs: Float = 0, refMax: Float = 0, sumAbs: Float = 0, sumRef: Float = 0
        outSurf.lock(options: [], seed: nil)
        let op = outSurf.baseAddress.bindMemory(to: Float16.self, capacity: O * S)
        if !yref.isEmpty { yref.withUnsafeBytes { (rb: UnsafeRawBufferPointer) in
            let rp = rb.bindMemory(to: Float.self)
            for s in 0 ..< S {
                for o in 0 ..< O {
                    let got = Float(op[o * S + s])
                    let ref = rp[s * O + o]
                    let d = abs(got - ref)
                    maxAbs = max(maxAbs, d); refMax = max(refMax, abs(ref))
                    sumAbs += d; sumRef += abs(ref)
                }
            }
        } }
        outSurf.unlock(options: [], seed: nil)

        for _ in 0 ..< 5 { _ = try model.prediction(from: provider, options: opts) }
        var t: [Double] = []
        for _ in 0 ..< iters {
            let s = DispatchTime.now().uptimeNanoseconds
            _ = try model.prediction(from: provider, options: opts)
            t.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6)
        }
        t.sort()
        return (maxAbs, maxAbs / max(refMax, 1e-9), sumAbs / max(sumRef, 1e-9), t[t.count / 2])
    }

    @Test("compute plan, numerics and timing for every ANE probe package", .enabled(if: enabled))
    func computePlanProbe() throws {
        try #require(ANERuntime.available())
        guard #available(macOS 15.0, *) else { return }
        let dir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["MLXFAST_ANE_PKG_DIR"]!)
        let pkgs = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "mlpackage" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        print("[plan] \(pkgs.count) packages under \(dir.path)")
        var summary: [String] = []
        for pkg in pkgs {
            let name = pkg.deletingPathExtension().lastPathComponent
            let compiled: URL
            do {
                compiled = try awaitSync { try await MLModel.compileModel(at: pkg) }
            } catch {
                print("[plan] \(name): COMPILE FAILED \(error)")
                summary.append("\(name): compile failed")
                continue
            }
            let (ane, nonAne) = try printPlan(compiled: compiled, label: name)
            let side = dir.appendingPathComponent(name + ".probe")
            guard FileManager.default.fileExists(atPath: side.appendingPathComponent("meta.json").path) else {
                summary.append("\(name): plan only; ane=[\(ane.joined(separator: " "))] other=[\(nonAne.joined(separator: " "))]")
                continue
            }
            let meta = try JSONSerialization.jsonObject(with: readBytes(side.appendingPathComponent("meta.json"))) as! [String: Any]
            let O = meta["O"] as! Int, K = meta["K"] as! Int, S = meta["S"] as! Int
            var line = "\(name): plan ane=[\(ane.joined(separator: " "))] other=[\(nonAne.joined(separator: " "))]"
            do {
                let mANE = try load(compiled: compiled, units: .cpuAndNeuralEngine)
                let a = try runProbe(model: mANE, side: side, O: O, K: K, S: S, iters: 30)
                line += String(format: " | ANE-config: maxAbs %.4f rel %.4f meanRel %.4f %.3f ms", a.maxAbs, a.rel, a.meanRel, a.ms)
            } catch {
                line += " | ANE-config load/run FAILED: \(error)"
            }
            do {
                let mCPU = try load(compiled: compiled, units: .cpuOnly)
                let c = try runProbe(model: mCPU, side: side, O: O, K: K, S: S, iters: 8)
                line += String(format: " | CPU-only: maxAbs %.4f rel %.4f %.3f ms", c.maxAbs, c.rel, c.ms)
            } catch {
                line += " | CPU-only load/run FAILED: \(error)"
            }
            print("[probe] " + line)
            summary.append(line)
        }
        print("[probe] ===== SUMMARY =====")
        for s in summary { print("[probe] " + s) }
    }
}
