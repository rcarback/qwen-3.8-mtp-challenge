// Reusable, warm ANE GEMM primitive -- lifts the Task 1 PoC's inline
// per-call ANE matmul into a type that compiles its Core ML program once at
// init and reuses it for every `callAsFunction`. See
// `.superpowers/sdd/2026-08-30-ane-gpu-concurrent-offload/task-2-brief.md`.
import CoreML
import Foundation
import MLX

/// Cross-thread result box for the one-time async `MLModel.load` join in
/// `ANEGemm.init` -- see the comment there.
private final class LoadResult: @unchecked Sendable {
    var model: MLModel?
    var error: Error?
}

/// Carries the non-`Sendable` `MLModelAsset`/`MLModelConfiguration` into the
/// `Task.detached` body in `ANEGemm.init`. Both are read-only by the time
/// they are handed off (the asset was just compiled from an immutable spec,
/// the configuration's `computeUnits` was set once before this box is
/// constructed and never touched again), so the single detached read is safe
/// despite neither type conforming to `Sendable`.
private final class LoadInputs: @unchecked Sendable {
    let asset: MLModelAsset
    let configuration: MLModelConfiguration
    init(asset: MLModelAsset, configuration: MLModelConfiguration) {
        self.asset = asset
        self.configuration = configuration
    }
}

/// One fp16 projection GEMM (`x[S,in] @ weight[out,in].T -> [S,out]`) run on
/// the ANE via a single-op Core ML MIL program (a 1x1 `conv`, see
/// `buildConvMatmul`). The Core ML model is compiled once in `init` --
/// `MLModelAsset(specification:)` + `MLModel.load(asset:configuration:)` is
/// Apple's own in-memory compiler path (no SIP-off, no entitlement forging,
/// no trustcache edits) -- and reused warm for every subsequent call.
public final class ANEGemm {
    public let out: Int
    public let inn: Int
    private let sequenceLength: Int
    private let model: MLModel

    /// weight: [out, in] fp16 (already dequantized). Compiles the ANE model
    /// once, for exactly the given `sequenceLength`; `callAsFunction` only
    /// accepts activations of that length (the MIL program's `S` dimension is
    /// baked in at compile time).
    public init(weight: MLXArray, sequenceLength: Int) throws {
        precondition(weight.ndim == 2, "ANEGemm weight must be rank-2 [out, in], got shape \(weight.shape)")
        out = weight.shape[0]
        inn = weight.shape[1]
        self.sequenceLength = sequenceLength

        let wBytes = f16Bytes(weight)
        let spec = buildConvMatmul(K: inn, F: out, S: sequenceLength, weight: wBytes)
        let asset = try MLModelAsset(specification: spec)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine

        // `MLModel.load(asset:configuration:)` is async-only in this Core ML
        // SDK (no synchronous overload). `init` is a sync throws signature
        // per the brief's interface (later tasks assume it), and load
        // happens exactly once here, outside every timed path, so block on
        // the async load with a semaphore rather than making `init` async.
        // `LoadResult` boxes the outcome so it can cross the `Task`'s
        // isolation boundary; a `DispatchSemaphore` join (like the
        // `MLBox`/`ResultBox` pattern in `ANEChannelSplitPoCTests`) makes the
        // handoff happens-before/-after safe without a lock.
        //
        // `Task.detached` (not a plain `Task {}`) so this never inherits the
        // caller's actor -- a plain `Task {}` created from a `@MainActor`
        // context would inherit main-actor isolation, and `sema.wait()`
        // pinning that executor while the detached-from-main-actor `await`
        // needs it back would deadlock. `asset`/`cfg` are not `Sendable`, so
        // they cross into the detached body via the `LoadInputs` box instead
        // of being captured directly.
        let sema = DispatchSemaphore(value: 0)
        let result = LoadResult()
        let inputs = LoadInputs(asset: asset, configuration: cfg)
        Task.detached {
            do {
                result.model = try await MLModel.load(asset: inputs.asset, configuration: inputs.configuration)
            } catch {
                result.error = error
            }
            sema.signal()
        }
        sema.wait()
        if let loadError = result.error { throw loadError }
        guard let loaded = result.model else {
            throw NSError(domain: "ANEGemm", code: 1,
                           userInfo: [NSLocalizedDescriptionKey: "MLModel.load produced neither a model nor an error"])
        }
        model = loaded
    }

    /// x: [S, in] fp16 -> [S, out] fp16, computed on the ANE. `S` must equal
    /// the `sequenceLength` this instance was compiled for.
    public func callAsFunction(_ x: MLXArray) throws -> MLXArray {
        precondition(x.ndim == 2 && x.shape[0] == sequenceLength && x.shape[1] == inn,
                     "ANEGemm expected input shape [\(sequenceLength), \(inn)], got \(x.shape)")
        return try autoreleasepool {
            let xa = try mlxToMultiArray_1C1S(x)
            let input = try MLDictionaryFeatureProvider(dictionary: ["a": MLFeatureValue(multiArray: xa)])
            let out = try model.prediction(from: input)
            guard let ya = out.featureValue(for: "y")?.multiArrayValue else {
                throw NSError(domain: "ANEGemm", code: 2,
                               userInfo: [NSLocalizedDescriptionKey: "no ANE output 'y'"])
            }
            return multiArray_1C1S_toMLX(ya)
        }
    }
}
