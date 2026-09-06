import CoreML
import Foundation
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
