// LOCAL M4 FORK ONLY. Core ML public API, no private frameworks.
//
// The dense tower's fused SwiGLU-down ANE prefix as a multifunction
// `.mlpackage` per bucket, every layer a function, dispatched through
// `MLModel.prediction` with surface-backed I/O. This is the path that (a)
// dispatches every function of one loaded program, so 64 layers cost one
// program slot per bucket instead of 64 against the 126-program wall, and
// (b) carries int4 palettes with one codebook per weight row, which the
// in-memory compiler does not accept. The packages are built offline by
// `gen_fused_bank.py` from the q4 group-64 safetensors; the runtime here only
// compiles (once, cached beside the package) and loads them.
import CoreML
import Foundation
import IOSurface
import MLX

/// One function of a bank package, loaded on the ANE, with its persistent
/// input and output surfaces. `makeInput` / `predict` / `readOutput` mirror
/// `ANEFusedMLP` so `ANEFusedSplitMLP` can hold either leg.
public final class ANEFusedMLPBankFunction: @unchecked Sendable {
    public let hidden: Int
    public let sequenceLength: Int
    public let layer: Int
    private let model: MLModel
    private let inSurf: IOSurface
    private let outSurf: IOSurface
    private let provider: MLDictionaryFeatureProvider
    private let options: MLPredictionOptions

    /// A handle for the three-phase call; the surfaces are owned by the
    /// function and reused, so a layer's prefill must consume one output
    /// before the next call overwrites it (the split path does).
    public final class Prepared: @unchecked Sendable {}

    fileprivate init(model: MLModel, hidden: Int, sequenceLength: Int, layer: Int) throws {
        self.model = model
        self.hidden = hidden
        self.sequenceLength = sequenceLength
        self.layer = layer
        func surface(bytes: Int) -> IOSurface {
            let alloc = max(65536, (bytes + 65535) & ~65535)
            let props: NSDictionary = [
                kIOSurfaceWidth: alloc, kIOSurfaceHeight: 1, kIOSurfaceBytesPerElement: 1,
                kIOSurfaceBytesPerRow: alloc, kIOSurfaceAllocSize: alloc, kIOSurfacePixelFormat: 0,
            ]
            return IOSurfaceCreate(props as CFDictionary)!
        }
        inSurf = surface(bytes: hidden * sequenceLength * 2)
        outSurf = surface(bytes: hidden * sequenceLength * 2)
        let shape = [1, hidden, 1, sequenceLength].map { NSNumber(value: $0) }
        let strides = [hidden * sequenceLength, sequenceLength, sequenceLength, 1].map { NSNumber(value: $0) }
        let inMA = try MLMultiArray(dataPointer: inSurf.baseAddress, shape: shape, dataType: .float16, strides: strides)
        let outMA = try MLMultiArray(dataPointer: outSurf.baseAddress, shape: shape, dataType: .float16, strides: strides)
        let inName = model.modelDescription.inputDescriptionsByName.keys.first ?? "x"
        let outName = model.modelDescription.outputDescriptionsByName.keys.first ?? "y"
        provider = try MLDictionaryFeatureProvider(dictionary: [inName: MLFeatureValue(multiArray: inMA)])
        options = MLPredictionOptions()
        options.outputBackings = [outName: outMA]
    }

    /// CALLER THREAD ONLY (MLX). `x` is `[sequenceLength, hidden]`, any float
    /// dtype; it is transposed to the channel-major surface layout here.
    public func makeInput(_ x: MLXArray) throws -> Prepared {
        precondition(x.ndim == 2 && x.dim(0) == sequenceLength && x.dim(1) == hidden,
                     "ANEFusedMLPBankFunction.makeInput expected [\(sequenceLength), \(hidden)], got \(x.shape)")
        let xt = contiguous(x.transposed(1, 0).asType(.float16))
        eval(xt)
        let d = xt.asData().data
        inSurf.lock(options: [], seed: nil)
        d.withUnsafeBytes { _ = memcpy(inSurf.baseAddress, $0.baseAddress!, hidden * sequenceLength * 2) }
        inSurf.unlock(options: [], seed: nil)
        return Prepared()
    }

    /// BACKGROUND SAFE. The synchronous Core ML prediction; no MLX.
    public func predict(_ p: Prepared) throws {
        _ = try model.prediction(from: provider, options: options)
    }

    /// CALLER THREAD ONLY (MLX). Copies the output surface out as
    /// `[sequenceLength, hidden]` fp16 so the surface is free for the next call.
    public func readOutput(_ p: Prepared) -> MLXArray {
        let m = MLXArray(rawPointer: outSurf.baseAddress, [hidden, sequenceLength], dtype: .float16,
                         finalizer: { [outSurf] in _ = outSurf })
        let y = contiguous(m.transposed(1, 0))
        eval(y)
        return y
    }
}

/// The bank directory (`MLX_ANE_BANK_DIR`): `S<bucket>.mlpackage` plus its
/// `S<bucket>.json` metadata from `gen_fused_bank.py`. Compiled once per
/// process into `S<bucket>.mlmodelc` beside the package.
public enum ANEFusedMLPBank {
    public static let directory: URL? = {
        guard let c = getenv("MLX_ANE_BANK_DIR") else { return nil }
        let p = String(cString: c)
        return p.isEmpty ? nil : URL(fileURLWithPath: p)
    }()

    private final class Box<T>: @unchecked Sendable { var value: T?; var error: Error? }
    private static let lock = NSLock()
    nonisolated(unsafe) private static var compiledByBucket: [Int: URL] = [:]
    nonisolated(unsafe) private static var metaByBucket: [Int: (fraction: Double, f: [Int: Int])] = [:]

    private static func awaitSync<T>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
        let sema = DispatchSemaphore(value: 0)
        let box = Box<T>()
        Task.detached {
            do { box.value = try await body() } catch { box.error = error }
            sema.signal()
        }
        sema.wait()
        if let e = box.error { throw e }
        guard let v = box.value else {
            throw NSError(domain: "ANEFusedMLPBank", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "async call produced neither value nor error"])
        }
        return v
    }

    /// The compiled model for `bucket`, compiling the package on first use and
    /// keeping the `.mlmodelc` beside it for later processes.
    private static func compiled(bucket: Int) throws -> URL {
        guard let dir = directory else {
            throw NSError(domain: "ANEFusedMLPBank", code: 2, userInfo: [NSLocalizedDescriptionKey: "MLX_ANE_BANK_DIR unset"])
        }
        lock.lock()
        defer { lock.unlock() }
        if let c = compiledByBucket[bucket] { return c }
        let pkg = dir.appendingPathComponent("S\(bucket).mlpackage")
        let cached = dir.appendingPathComponent("S\(bucket).mlmodelc")
        let fm = FileManager.default
        guard fm.fileExists(atPath: pkg.path) else {
            throw NSError(domain: "ANEFusedMLPBank", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "no bank package for bucket \(bucket) at \(pkg.path)"])
        }
        if !fm.fileExists(atPath: cached.path) {
            let tmp = try awaitSync { try await MLModel.compileModel(at: pkg) }
            try fm.moveItem(at: tmp, to: cached)
        }
        // Metadata: the prefix width F per layer, to check against the split's own F.
        let metaURL = dir.appendingPathComponent("S\(bucket).json")
        if let data = try? Data(contentsOf: metaURL),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let layers = obj["layers"] as? [[String: Any]]
        {
            var f: [Int: Int] = [:]
            for l in layers { if let n = l["layer"] as? Int, let ff = l["F"] as? Int { f[n] = ff } }
            metaByBucket[bucket] = ((obj["fraction"] as? Double) ?? 0, f)
        }
        compiledByBucket[bucket] = cached
        return cached
    }

    /// The prefix width the bank holds for `layer` at `bucket`, if known.
    public static func prefixChannels(bucket: Int, layer: Int) -> Int? {
        lock.lock()
        defer { lock.unlock() }
        return metaByBucket[bucket]?.f[layer]
    }

    /// Loads function `layer<layer>` of the bucket's package on the ANE.
    public static func function(bucket: Int, layer: Int, hidden: Int) throws -> ANEFusedMLPBankFunction {
        let url = try compiled(bucket: bucket)
        guard #available(macOS 15.0, *) else {
            throw NSError(domain: "ANEFusedMLPBank", code: 4, userInfo: [NSLocalizedDescriptionKey: "multifunction needs macOS 15+"])
        }
        let name = "layer\(layer)"
        let model = try awaitSync {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndNeuralEngine
            cfg.functionName = name
            return try await MLModel.load(contentsOf: url, configuration: cfg)
        }
        return try ANEFusedMLPBankFunction(model: model, hidden: hidden, sequenceLength: bucket, layer: layer)
    }
}
