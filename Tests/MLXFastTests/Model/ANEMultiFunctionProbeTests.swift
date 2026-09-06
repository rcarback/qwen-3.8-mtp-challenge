import CoreML
import Foundation
import IOSurface
import MLX
import MLXRandom
import Testing

@testable import MLXFastModel

/// The follow-up to `ANEProcedureBankProbeTests`: the bare-text procedure bank
/// loads N functions in one program but only `main` dispatches, because the
/// in-memory descriptor carries no model description. This suite supplies the
/// missing half -- a MULTIFUNCTION Core ML `Model` proto whose
/// `ModelDescription.functions` list declares every function -- and asks
/// whether each function then runs on the ANE when selected by
/// `MLModelConfiguration.functionName`.
///
/// Findings on this M4 Max, macOS 26.5.2 (2026-09-06):
///
///   - The IN-MEMORY spec path is a dead end for multifunction. `MLModelAsset`
///     `(specification:)` takes a `.mlmodel` blob and rejects a multifunction
///     description: "This MLModel doesn't support the multi-function
///     description sytnax" (default load) and "functionName must be nil unless
///     the model type is ML Program" (by-name load), at both the CoreML8 and
///     CoreML9 opsets. A top-level input/output alongside the functions list
///     is a hard error: "Multi-function model must not use top level input
///     feature description." So the description must be functions-only, and
///     the in-memory blob cannot carry it. `multifunctionSweep` records this.
///
///   - The `.mlpackage` path WORKS. Writing the same functions-only spec into
///     a `.mlpackage`, compiling it with `MLModel.compileModel(at:)`, and
///     loading each function by `MLModelConfiguration.functionName` runs every
///     function on the ANE with its own weights: `main` computed w0 and
///     `proc1` computed w1, each at the fp16 floor. `multifunctionMLPackage`
///     asserts it. This closes the dispatch half of the 126-program
///     workaround: N fixed shapes pack into one program (one of the 126
///     slots) and every one is dispatchable.
///
/// The caveat is that this path runs through `MLModel.prediction`
/// (MLMultiArray copies), not the zero-copy IOSurface + `procedureIndex`
/// direct dispatch. That overhead amortizes over a compute-bound prefill
/// (S >= 128); it does not help decode, where the ANE loses at one row anyway.
///
/// Needs the real ANE. Run with:
///   MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
///     swift test -c release --force-resolved-versions \
///     --filter ANEMultiFunctionProbeTests
@Suite(.serialized)
struct ANEMultiFunctionProbeTests {
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    /// Boxes the async `MLModel.load` outcome across the `Task.detached`
    /// isolation boundary, exactly as `ANEGemm.init` does.
    private final class LoadBox: @unchecked Sendable {
        var model: MLModel?
        var error: Error?
    }

    private final class LoadInputs: @unchecked Sendable {
        let asset: MLModelAsset
        let cfg: MLModelConfiguration
        init(asset: MLModelAsset, cfg: MLModelConfiguration) {
            self.asset = asset
            self.cfg = cfg
        }
    }

    /// Compiles `spec` in memory and loads the model for one function name on
    /// the ANE. `MLModel.load` is async-only in this SDK; block on it with a
    /// semaphore, off the caller's actor, matching `ANEGemm`.
    private func load(spec: Data, functionName: String?) throws -> MLModel {
        let asset = try MLModelAsset(specification: spec)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        if let functionName {
            // Multifunction selection is macOS 15+; the package targets 14, so
            // guard the assignment. The ranked box runs a far newer macOS.
            guard #available(macOS 15.0, *) else {
                throw NSError(domain: "ANEMultiFunctionProbe", code: 3,
                              userInfo: [NSLocalizedDescriptionKey: "multifunction needs macOS 15+"])
            }
            cfg.functionName = functionName
        }
        let sema = DispatchSemaphore(value: 0)
        let box = LoadBox()
        let inputs = LoadInputs(asset: asset, cfg: cfg)
        Task.detached {
            do { box.model = try await MLModel.load(asset: inputs.asset, configuration: inputs.cfg) }
            catch { box.error = error }
            sema.signal()
        }
        sema.wait()
        if let e = box.error { throw e }
        guard let m = box.model else {
            throw NSError(domain: "ANEMultiFunctionProbe", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "load produced neither model nor error"])
        }
        return m
    }

    /// Runs function-selected `model` on `x [S, in]` and returns `[S, out]`.
    private func predict(_ model: MLModel, x: MLXArray) throws -> MLXArray {
        let input = try mlxToMultiArray_1C1S(x.asType(.float16))
        let provider = try MLDictionaryFeatureProvider(dictionary: ["a": MLFeatureValue(multiArray: input)])
        let out = try model.prediction(from: provider)
        guard let ya = out.featureValue(for: "y")?.multiArrayValue else {
            throw NSError(domain: "ANEMultiFunctionProbe", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "no output feature 'y'"])
        }
        return multiArray_1C1S_toMLX(ya)
    }

    private func maxAbsError(_ got: MLXArray, x: MLXArray, w: MLXArray) -> Float {
        let ref = matmul(x.asType(.float32), w.asType(.float32).transposed(1, 0))
        let d = MLX.abs(got.asType(.float32) - ref).max()
        eval(d)
        return d.item(Float.self)
    }

    private func randomWeight(_ out: Int, _ inn: Int, seed: UInt64) -> MLXArray {
        MLXRandom.seed(seed)
        let w = MLXRandom.normal([out, inn]).asType(.float16)
        eval(w)
        return w
    }

    /// Two distinct-weight convs declared as `main` and `proc1` in one
    /// multifunction program. Each is loaded by name and run; the diagonal
    /// (function i scored against its own weights) must sit at the fp16 floor
    /// while the off-diagonal is ~1. `proc1` landing on the diagonal is the
    /// result the bare-text bank could not reach.
    /// Writes a minimal `.mlpackage` wrapping `spec` (weights are inline
    /// consts, so no external `weights/weight.bin` is needed) and returns its
    /// URL. Multifunction descriptions are only honoured by the `.mlpackage`
    /// path; the in-memory `MLModelAsset(specification:)` blob rejects them.
    private func writeMLPackage(spec: Data, dir: URL) throws -> URL {
        let pkg = dir.appendingPathComponent("bank.mlpackage")
        let modelDir = pkg.appendingPathComponent("Data/com.apple.CoreML")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try spec.write(to: modelDir.appendingPathComponent("model.mlmodel"))
        let id = "1A1A1A1A-1A1A-1A1A-1A1A-1A1A1A1A1A1A"
        let manifest = """
        {
          "fileFormatVersion": "1.0.0",
          "itemInfoEntries": {
            "\(id)": {
              "author": "com.apple.CoreML",
              "description": "CoreML Model Specification",
              "name": "model.mlmodel",
              "path": "com.apple.CoreML/model.mlmodel"
            }
          },
          "rootModelIdentifier": "\(id)"
        }
        """
        try Data(manifest.utf8).write(to: pkg.appendingPathComponent("Manifest.json"))
        return pkg
    }

    /// Compiles a `.mlpackage` to `.mlmodelc` and loads one function by name,
    /// blocking on the async APIs the way `ANEGemm` does.
    @available(macOS 15.0, *)
    private func loadFromPackage(_ pkg: URL, functionName: String) throws -> MLModel {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        cfg.functionName = functionName
        let sema = DispatchSemaphore(value: 0)
        let box = LoadBox()
        Task.detached {
            do {
                let compiled = try await MLModel.compileModel(at: pkg)
                box.model = try await MLModel.load(contentsOf: compiled, configuration: cfg)
            } catch { box.error = error }
            sema.signal()
        }
        sema.wait()
        if let e = box.error { throw e }
        guard let m = box.model else {
            throw NSError(domain: "ANEMultiFunctionProbe", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "package load produced neither model nor error"])
        }
        return m
    }

    /// The on-disk multifunction path: write a `.mlpackage`, compile it, load
    /// `main` and `proc1` each by name, and check each ran its own weights.
    /// `proc1` on its diagonal is the result the bare-text bank and the
    /// in-memory spec path could not reach.
    @Test("multifunction mlpackage: every declared function dispatches", .enabled(if: enabled))
    func multifunctionMLPackage() throws {
        try #require(ANERuntime.available())
        guard #available(macOS 15.0, *) else { return }
        let inn = 64, out = 64, seq = 128
        let w0 = randomWeight(out, inn, seed: 1)
        let w1 = randomWeight(out, inn, seed: 2)
        let x = { () -> MLXArray in MLXRandom.seed(9); let v = MLXRandom.normal([seq, inn]).asType(.float16); eval(v); return v }()
        let spec = buildMultiFunctionConvSpec(procs: [
            ANEMultiFunctionProc(name: "main", inputDim: inn, outputDim: out, sequenceLength: seq, weight: f16Bytes(w0)),
            ANEMultiFunctionProc(name: "proc1", inputDim: inn, outputDim: out, sequenceLength: seq, weight: f16Bytes(w1)),
        ])
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("mf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let pkg = try writeMLPackage(spec: spec, dir: tmp)

        for name in ["main", "proc1"] {
            do {
                let model = try loadFromPackage(pkg, functionName: name)
                let y = try predict(model, x: x)
                let e0 = maxAbsError(y, x: x, w: w0)
                let e1 = maxAbsError(y, x: x, w: w1)
                let ran = e0 < e1 ? "w0 (main)" : "w1 (proc1)"
                print("[mf-pkg] \(name): OK — vs w0=\(String(format: "%.4f", e0)) w1=\(String(format: "%.4f", e1)) -> ran \(ran)")
                if name == "proc1" {
                    #expect(e1 < 0.1, "proc1 diverged from its own weights by \(e1) — function not dispatched")
                    #expect(e1 < e0, "proc1 matched w0 better than its own w1")
                }
            } catch {
                print("[mf-pkg] \(name): FAILED — \(error)")
                Issue.record("\(name) did not dispatch via mlpackage: \(error)")
            }
        }
    }

    /// p84 feasibility: can a multifunction program compiled to `.mlmodelc` be
    /// loaded as a private `_ANEModel` whose procedures are individually
    /// addressable, the prerequisite for zero-copy `procedureIndex` dispatch?
    /// `_ANEModel` exposes `+modelAtURL:key:`, `-procedureInfoForProcedureIndex:`
    /// and `-programHandle`. This probe loads the compiled model and reads
    /// procedure info for indices 0 and 1; both non-nil means the direct path
    /// can see every packed function, and the zero-copy bridge is buildable.
    @Test("p84: compiled multifunction model exposes per-procedure info", .enabled(if: enabled))
    func compiledModelProcedureInfo() throws {
        try #require(ANERuntime.available())
        guard #available(macOS 15.0, *) else { return }
        guard let aneModelClass = ANERuntime.cls("_ANEModel") else {
            Issue.record("_ANEModel class not present")
            return
        }
        let inn = 64, out = 64, seq = 128
        let spec = buildMultiFunctionConvSpec(procs: [
            ANEMultiFunctionProc(name: "main", inputDim: inn, outputDim: out, sequenceLength: seq, weight: f16Bytes(randomWeight(out, inn, seed: 1))),
            ANEMultiFunctionProc(name: "proc1", inputDim: inn, outputDim: out, sequenceLength: seq, weight: f16Bytes(randomWeight(out, inn, seed: 2))),
        ])
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("mf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let pkg = try writeMLPackage(spec: spec, dir: tmp)

        // Compile the .mlpackage to a .mlmodelc URL, blocking on the async API.
        let box = LoadBox()  // reuse the box just for its error/URL carriage
        let urlBox = { () -> URL? in
            final class UB: @unchecked Sendable { var url: URL?; var err: Error? }
            let ub = UB()
            let sema = DispatchSemaphore(value: 0)
            Task.detached {
                do { ub.url = try await MLModel.compileModel(at: pkg) } catch { ub.err = error }
                sema.signal()
            }
            sema.wait()
            if let e = ub.err { box.error = e }
            return ub.url
        }()
        guard let compiled = urlBox else {
            Issue.record("compileModel failed: \(box.error.map { "\($0)" } ?? "nil url")")
            return
        }
        print("[p84] compiled -> \(compiled.lastPathComponent)")
        if let items = try? FileManager.default.contentsOfDirectory(atPath: compiled.path).sorted() {
            print("[p84] .mlmodelc contents: \(items.joined(separator: ", "))")
        }

        // `+modelAtURL:key:` — class factory, autoreleased (+0).
        let msgSend = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
        typealias ModelAtURL = @convention(c) (AnyObject?, Selector, AnyObject?, AnyObject?) -> Unmanaged<AnyObject>?
        let modelAtURL = unsafeBitCast(msgSend, to: ModelAtURL.self)
        let sel = Selector(("modelAtURL:key:"))
        guard let modelU = modelAtURL(aneModelClass, sel, compiled as NSURL, nil) else {
            Issue.record("modelAtURL:key: returned nil")
            return
        }
        let model = modelU.takeUnretainedValue()
        print("[p84] _ANEModel loaded: \(type(of: model))")

        typealias ProcInfo = @convention(c) (AnyObject?, Selector, Int) -> Unmanaged<AnyObject>?
        let procInfo = unsafeBitCast(msgSend, to: ProcInfo.self)
        let piSel = Selector(("procedureInfoForProcedureIndex:"))
        func readProcedures(_ tag: String) {
            for idx in 0 ... 1 {
                let info = procInfo(model, piSel, idx)?.takeUnretainedValue()
                print("[p84] \(tag) procedureInfo(\(idx)): \(info.map { "\($0)" } ?? "nil")")
            }
            let handle = ANERuntime.sendUInt64(model, Selector(("programHandle")))
            print("[p84] \(tag) programHandle: 0x\(String(handle, radix: 16))")
        }
        readProcedures("pre-load")

        // Load onto the ANE via a fresh _ANEClient, which should populate the
        // program handle and the procedure table.
        guard let clientClass = ANERuntime.cls("_ANEClient") else {
            Issue.record("_ANEClient class not present"); return
        }
        // Plain alloc/init returns nil; the client is a shared connection.
        guard let client = ANERuntime.send(clientClass, Selector(("sharedConnection"))) else {
            Issue.record("_ANEClient sharedConnection nil"); return
        }
        print("[p84] _ANEClient: \(type(of: client))")

        typealias LoadModel = @convention(c) (AnyObject?, Selector, AnyObject?, AnyObject?, Int, UnsafeMutablePointer<Unmanaged<NSError>?>?) -> ObjCBool
        let loadModel = unsafeBitCast(msgSend, to: LoadModel.self)
        let loadSel = Selector(("loadModel:options:qos:error:"))
        var errU: Unmanaged<NSError>?
        let ok = withUnsafeMutablePointer(to: &errU) { p in
            loadModel(client, loadSel, model, NSDictionary(), 0x15, p).boolValue
        }
        if ok {
            print("[p84] _ANEClient loadModel: OK")
        } else {
            print("[p84] _ANEClient loadModel FAILED: \(errU?.takeUnretainedValue().localizedDescription ?? "unknown")")
        }
        readProcedures("post-load")
    }

    private func makeSurface(bytes: Int) -> IOSurface {
        let alloc = max(65536, (bytes + 65535) & ~65535)
        let props: NSDictionary = [
            kIOSurfaceWidth: alloc, kIOSurfaceHeight: 1, kIOSurfaceBytesPerElement: 1,
            kIOSurfaceBytesPerRow: alloc, kIOSurfaceAllocSize: alloc, kIOSurfacePixelFormat: 0,
        ]
        return IOSurfaceCreate(props as CFDictionary)!
    }

    /// The corrected e2l/p84 measurement: the multifunction bank dispatched
    /// through `MLModel.prediction` with ZERO-COPY-ish I/O -- a persistent
    /// IOSurface-backed input MLMultiArray and `MLPredictionOptions.outputBackings`
    /// writing into a surface-backed output wrapped straight into MLX. This
    /// tests whether the ANE compute win (0.598 ms vs 0.689 ms quantized GPU)
    /// survives once the copying `mlxToMultiArray_1C1S` helper is removed. The
    /// round trip is split into stage (MLX activation -> input surface),
    /// predict, and read (output surface -> MLX), each timed, versus the
    /// quantized GPU matmul that reads the MLX buffer with no staging.
    @Test("zero-copy surface-backed multifunction dispatch vs GPU", .enabled(if: enabled))
    func realShapeDispatchCostSurface() throws {
        try #require(ANERuntime.available())
        guard #available(macOS 15.0, *) else { return }
        let inn = 2560, out = 10240, seq = 128
        let w0 = randomWeight(out, inn, seed: 1)
        let w1 = randomWeight(out, inn, seed: 2)
        let x = { () -> MLXArray in MLXRandom.seed(9); let v = MLXRandom.normal([seq, inn]).asType(.float16); eval(v); return v }()
        let spec = buildMultiFunctionConvSpec(procs: [
            ANEMultiFunctionProc(name: "main", inputDim: inn, outputDim: out, sequenceLength: seq, weight: f16Bytes(w0)),
            ANEMultiFunctionProc(name: "proc1", inputDim: inn, outputDim: out, sequenceLength: seq, weight: f16Bytes(w1)),
        ])
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("mf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let pkg = try writeMLPackage(spec: spec, dir: tmp)
        let model = try loadFromPackage(pkg, functionName: "proc1")

        // Persistent surface-backed input [1,inn,1,seq] and output [1,out,1,seq],
        // contiguous fp16 (CoreML handles the ANE's internal padding on this path).
        let inSurf = makeSurface(bytes: inn * seq * 2)
        let outSurf = makeSurface(bytes: out * seq * 2)
        let inMA = try MLMultiArray(
            dataPointer: inSurf.baseAddress, shape: [1, inn, 1, seq].map { NSNumber(value: $0) },
            dataType: .float16, strides: [inn * seq, seq, seq, 1].map { NSNumber(value: $0) })
        let outMA = try MLMultiArray(
            dataPointer: outSurf.baseAddress, shape: [1, out, 1, seq].map { NSNumber(value: $0) },
            dataType: .float16, strides: [out * seq, seq, seq, 1].map { NSNumber(value: $0) })
        let provider = try MLDictionaryFeatureProvider(dictionary: ["a": MLFeatureValue(multiArray: inMA)])
        let opts = MLPredictionOptions()
        opts.outputBackings = ["y": outMA]

        // stage: MLX activation [seq,inn] -> [inn,seq] fp16 bytes into the input surface.
        func stage() {
            let xt = contiguous(x.transposed(1, 0).asType(.float16))
            eval(xt)
            let d = xt.asData().data
            inSurf.lock(options: [], seed: nil)
            d.withUnsafeBytes { _ = memcpy(inSurf.baseAddress, $0.baseAddress!, inn * seq * 2) }
            inSurf.unlock(options: [], seed: nil)
        }
        // read: output surface -> MLX [seq,out].
        func readOut() -> MLXArray {
            let m = MLXArray(rawPointer: outSurf.baseAddress, [out, seq], dtype: .float16, finalizer: { [outSurf] in _ = outSurf })
            return contiguous(m.transposed(1, 0))
        }

        // Correctness: stage, predict, read; compare to x @ w1.T.
        stage()
        _ = try model.prediction(from: provider, options: opts)
        let y = readOut()
        let e1 = maxAbsError(y, x: x, w: w1)
        print("[mf-zc] correctness maxAbs vs w1 = \(String(format: "%.4f", e1))")

        func med(_ f: () -> Void, _ n: Int = 50) -> Double {
            for _ in 0 ..< 10 { f() }
            var t: [Double] = []
            for _ in 0 ..< n { let s = DispatchTime.now().uptimeNanoseconds; f(); t.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6) }
            t.sort(); return t[n / 2]
        }
        // Attribute the stage cost: the GPU transpose+sync that produces the
        // ANE's [inn,seq] input, versus the memcpy into the surface. The
        // transpose+eval is the GPU->ANE handoff proper; the memcpy is small.
        let transposeEvalMs = med { let xt = contiguous(x.transposed(1, 0).asType(.float16)); eval(xt) }
        let asDataMs = med { let xt = contiguous(x.transposed(1, 0).asType(.float16)); eval(xt); _ = xt.asData().data }
        let stageMs = med { stage() }
        let predictMs = med { _ = try? model.prediction(from: provider, options: opts) }
        let readMs = med { _ = readOut() }
        let e2eMs = med { stage(); _ = try? model.prediction(from: provider, options: opts); _ = readOut() }

        // GPU q4 baseline, reads the MLX buffer directly (no staging).
        let (wq, sc, bi) = quantized(w1, groupSize: 64, bits: 4)
        eval(wq, sc, bi)
        let qMs = med { let r = quantizedMatmul(x, wq, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 4); eval(r) }

        print(String(format: "[mf-zc] stage %.3f (transpose+eval %.3f, +asData %.3f) | predict %.3f | read %.3f | end-to-end %.3f ms",
                     stageMs, transposeEvalMs, asDataMs, predictMs, readMs, e2eMs))
        print(String(format: "[mf-zc] GPU q4 %.3f ms | end-to-end ANE / q4 = %.2f | predict-only / q4 = %.2f", qMs, e2eMs / qMs, predictMs / qMs))
    }

    /// The pivotal measurement for beads e2l and p84: warm per-function
    /// dispatch cost of a banked multifunction program at a REAL dense-lane
    /// shape, through `MLModel.prediction`, versus the GPU matmul it would
    /// replace. The bank cuts program COUNT, not dispatch COUNT, so this cost
    /// is paid once per layer per projection regardless of banking. If it far
    /// exceeds the GPU matmul, the MLModel.prediction path cannot carry a
    /// full-coverage dense lane and e2l needs the zero-copy bridge (p84)
    /// before it is worth wiring; if it is close, e2l can ship on this path.
    ///
    /// `in_proj_qkv` [10240, 2560] at S=128 is the largest real dense-lane
    /// projection (52 MB fp16 per layer). Two functions keep the inline-const
    /// spec under the 2 GB protobuf limit while still measuring per-call cost.
    @Test("multifunction dispatch cost at real shape vs GPU", .enabled(if: enabled))
    func realShapeDispatchCost() throws {
        try #require(ANERuntime.available())
        guard #available(macOS 15.0, *) else { return }
        let inn = 2560, out = 10240, seq = 128
        let w0 = randomWeight(out, inn, seed: 1)
        let w1 = randomWeight(out, inn, seed: 2)
        let x = { () -> MLXArray in MLXRandom.seed(9); let v = MLXRandom.normal([seq, inn]).asType(.float16); eval(v); return v }()

        let spec = buildMultiFunctionConvSpec(procs: [
            ANEMultiFunctionProc(name: "main", inputDim: inn, outputDim: out, sequenceLength: seq, weight: f16Bytes(w0)),
            ANEMultiFunctionProc(name: "proc1", inputDim: inn, outputDim: out, sequenceLength: seq, weight: f16Bytes(w1)),
        ])
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("mf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let pkg = try writeMLPackage(spec: spec, dir: tmp)

        let model = try loadFromPackage(pkg, functionName: "proc1")
        // Correctness first: this function must run its own weights.
        let y = try predict(model, x: x)
        let e1 = maxAbsError(y, x: x, w: w1)
        #expect(e1 < 0.5, "proc1 at real shape diverged from its own weights by \(e1)")

        // Warm, then time N predictions. Stage the MLMultiArray input once and
        // reuse it, so the timed region is prediction only (the copy cost the
        // zero-copy bridge would remove is measured separately below).
        let input = try mlxToMultiArray_1C1S(x.asType(.float16))
        let provider = try MLDictionaryFeatureProvider(dictionary: ["a": MLFeatureValue(multiArray: input)])
        for _ in 0 ..< 10 { _ = try model.prediction(from: provider) }
        let iters = 50
        var aneTimes: [Double] = []
        for _ in 0 ..< iters {
            let t0 = DispatchTime.now().uptimeNanoseconds
            _ = try model.prediction(from: provider)
            aneTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
        }
        aneTimes.sort()
        let aneMs = aneTimes[iters / 2]

        // GPU matmul [128,2560] @ [10240,2560].T, warm, same iteration count.
        let xg = x
        let wg = w1
        for _ in 0 ..< 10 { let r = matmul(xg, wg.transposed(1, 0)); eval(r) }
        var gpuTimes: [Double] = []
        for _ in 0 ..< iters {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let r = matmul(xg, wg.transposed(1, 0))
            eval(r)
            gpuTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
        }
        gpuTimes.sort()
        let gpuMs = gpuTimes[iters / 2]

        // The REAL production baseline: 4-bit group-64 affine quantized matmul,
        // the representation the dense projections actually run in. This is
        // faster than the dense bf16 arm above and is the honest comparison.
        let (wq, scales, biases) = quantized(wg, groupSize: 64, bits: 4)
        eval(wq, scales, biases)
        for _ in 0 ..< 10 {
            let r = quantizedMatmul(xg, wq, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4)
            eval(r)
        }
        var qTimes: [Double] = []
        for _ in 0 ..< iters {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let r = quantizedMatmul(xg, wq, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4)
            eval(r)
            qTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
        }
        qTimes.sort()
        let qMs = qTimes[iters / 2]

        // The MLMultiArray copy cost the zero-copy bridge (p84) would remove.
        for _ in 0 ..< 10 { _ = try mlxToMultiArray_1C1S(x.asType(.float16)) }
        var copyTimes: [Double] = []
        for _ in 0 ..< iters {
            let t0 = DispatchTime.now().uptimeNanoseconds
            _ = try mlxToMultiArray_1C1S(x.asType(.float16))
            copyTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
        }
        copyTimes.sort()

        print(String(format: "[mf-cost] in_proj_qkv [10240x2560] S=128: ANE predict %.3f ms | GPU bf16 %.3f ms | GPU q4g64 %.3f ms | input-copy %.3f ms",
                     aneMs, gpuMs, qMs, copyTimes[iters / 2]))
        print(String(format: "[mf-cost] ratios ANE/bf16 %.2f | ANE/q4 %.2f | (ANE+copy)/q4 %.2f",
                     aneMs / gpuMs, aneMs / qMs, (aneMs + copyTimes[iters / 2]) / qMs))
        print("[mf-cost] one prefill forward has 64 layers; per-projection x64 = "
            + String(format: "%.1f ms ANE vs %.1f ms GPU-q4", aneMs * 64, qMs * 64))
    }

    /// Attempts one (spec-variant, load-mode) combination and prints the
    /// max-abs error of the result against BOTH weight sets, or the error.
    private func attempt(label: String, spec: Data, functionName: String?, x: MLXArray, w0: MLXArray, w1: MLXArray) {
        do {
            let model = try load(spec: spec, functionName: functionName)
            let y = try predict(model, x: x)
            let e0 = maxAbsError(y, x: x, w: w0)
            let e1 = maxAbsError(y, x: x, w: w1)
            let ran = e0 < e1 ? "w0 (main)" : "w1 (proc1)"
            print("[mf-probe] \(label): OK — vs w0=\(String(format: "%.4f", e0)) w1=\(String(format: "%.4f", e1)) -> ran \(ran)")
        } catch {
            print("[mf-probe] \(label): FAILED — \(error)")
        }
    }

    /// Sweeps two spec variants (functions-only description, and functions
    /// plus top-level default I/O) against three load modes (default function,
    /// and each function by name). One build, every combination, so the
    /// working shape is found without blind 4-minute rebuild iterations. The
    /// combination where the `proc1`-by-name row "ran w1" is the fix: the
    /// multifunction descriptor makes the packed non-main function
    /// dispatchable.
    @Test("multifunction MLModel: which spec+load makes proc1 dispatch", .enabled(if: enabled))
    func multifunctionSweep() throws {
        try #require(ANERuntime.available())
        let inn = 64, out = 64, seq = 128
        let w0 = randomWeight(out, inn, seed: 1)
        let w1 = randomWeight(out, inn, seed: 2)
        let x = { () -> MLXArray in MLXRandom.seed(9); let v = MLXRandom.normal([seq, inn]).asType(.float16); eval(v); return v }()
        let procs = [
            ANEMultiFunctionProc(name: "main", inputDim: inn, outputDim: out, sequenceLength: seq, weight: f16Bytes(w0)),
            ANEMultiFunctionProc(name: "proc1", inputDim: inn, outputDim: out, sequenceLength: seq, weight: f16Bytes(w1)),
        ]
        let variants: [(String, Data)] = [
            ("CoreML8", buildMultiFunctionConvSpec(procs: procs, opset: "CoreML8")),
            ("CoreML9", buildMultiFunctionConvSpec(procs: procs, opset: "CoreML9")),
        ]
        for (vName, spec) in variants {
            attempt(label: "\(vName) default", spec: spec, functionName: nil, x: x, w0: w0, w1: w1)
            attempt(label: "\(vName) name=main", spec: spec, functionName: "main", x: x, w0: w0, w1: w1)
            attempt(label: "\(vName) name=proc1", spec: spec, functionName: "proc1", x: x, w0: w0, w1: w1)
        }
    }
}
