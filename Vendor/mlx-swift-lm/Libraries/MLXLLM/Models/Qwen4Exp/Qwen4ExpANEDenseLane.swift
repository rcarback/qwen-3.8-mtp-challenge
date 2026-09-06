// LOCAL M4 FORK ONLY. Private Apple frameworks; breaks on macOS updates.
// Qwen3.8-Flash-Next (qwen4_exp) ANE dense lane: fixed-shape fp16 programs for
// the per-token dense projections, used during prefill only, gated by
// MLX_ANE_DIRECT=1 at the call sites. The routed experts never touch the ANE
// (no dynamic gather, one program per shape, per-process program limit).
import CryptoKit
import Foundation
import MLX
import MLXNN

enum Qwen4ExpANELane {
    static let enabled: Bool = ProcessInfo.processInfo.environment["MLX_ANE_DIRECT"] == "1"
    /// Micro-batch length of every ANE program; a prefill chunk is pipelined as
    /// micro-batches of this many tokens.
    static let microBatch: Int =
        Int(ProcessInfo.processInfo.environment["MLX_QWEN4EXP_ANE_MICROBATCH"] ?? "") ?? 256
    static let minSeq: Int = Int(ProcessInfo.processInfo.environment["MLX_ANE_MIN_SEQ"] ?? "") ?? 128
    static let log: Bool = ProcessInfo.processInfo.environment["MLX_ANE_LOG"] == "1"
    /// `MLX_ANE_WEIGHT_FORM=fp16|int8|int4`, the storage form of every ANE
    /// program this tower builds. Shared with the dense tower's switch.
    static let weightForm: ANEWeightForm = ANESplitConfig.weightForm

    /// The dense `[out, in]` weight of a projection, dequantizing a
    /// `QuantizedLinear` (whose `.weight` is the packed form, for q8
    /// `dim(1) = in/4`) through MLX's own `dequantized` so the ANE program is
    /// built from the same values the GPU computes with.
    static func denseWeight(_ module: Module) -> MLXArray? {
        if let q = module as? QuantizedLinear {
            guard let biases = q.biases else { return nil }
            let w = dequantized(q.weight, scales: q.scales, biases: biases, groupSize: q.groupSize, bits: q.bits)
            eval(w)
            return w
        }
        if let l = module as? Linear { return l.weight }
        return nil
    }

    /// The lane engages only when a prefill holds at least two full micro-batches.
    static func armed(sequenceLength: Int) -> Bool {
        enabled && sequenceLength >= 2 * microBatch && sequenceLength >= minSeq
    }
}

/// One fixed-shape fp16 ANE program computing `x[S, in] @ w[out, in]^T` as a
/// 1x1 conv, through the in-memory compile+load path `ANEInMemoryModel` and
/// the IOSurface dispatch `ANEDirectDispatch`.
final class Qwen4ExpANEProjection {
    let out: Int
    let inn: Int
    let sequenceLength: Int
    private let model: ANEInMemoryModel

    /// `weight` is the DENSE `[out, in]` projection (a `QuantizedLinear` is
    /// dequantized first by `Qwen4ExpANELane.denseWeight`). `form` selects the
    /// storage the ANE program keeps it in; see `ANEWeightForm`.
    init(weight: MLXArray, sequenceLength: Int, form: ANEWeightForm = Qwen4ExpANELane.weightForm) throws {
        precondition(weight.ndim == 2, "Qwen4ExpANEProjection weight must be [out, in], got \(weight.shape)")
        out = weight.dim(0)
        inn = weight.dim(1)
        self.sequenceLength = sequenceLength
        // Same identity for the same weights in every process: the ANE daemon's
        // compiled-program cache hits instead of growing (see ANEFusedMLP).
        func tag(_ blob: Data) -> String {
            "sha256:" + SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        }
        let m: ANEInMemoryModel
        switch form {
        case .fp16:
            let blob = buildConvWeightBlob(f16Bytes(weight))
            let milText = buildConvMILText(
                inputDim: inn, outputDim: out, sequenceLength: sequenceLength, programTag: tag(blob))
            m = try ANEInMemoryModel(milText: milText, weightBlob: blob, weightFileName: "weight_data.bin")
        case .int8:
            let q = ANEWeightQuant.int8PerChannel(weight)
            let (blob, o) = buildMultiWeightBlob(chunks: [q.data, q.scale])
            let milText = buildConvMILTextForm(
                inputDim: inn, outputDim: out, sequenceLength: sequenceLength,
                weight: .int8(data: o[0], scale: o[1]), programTag: tag(blob))
            m = try ANEInMemoryModel(milText: milText, weightBlob: blob, weightFileName: "weight.bin")
        case .int4:
            let codebook = ANEWeightQuant.fitCodebook([weight])
            let q = ANEWeightQuant.int4Palette(weight, codebook: codebook)
            let (blob, o) = buildMultiWeightBlob(chunks: [q.indices, q.lut, q.scale], chunkTypes: [3, 1, 1])
            let milText = buildConvMILTextForm(
                inputDim: inn, outputDim: out, sequenceLength: sequenceLength,
                weight: .int4(indices: o[0], lut: o[1], scale: o[2]), programTag: tag(blob))
            m = try ANEInMemoryModel(milText: milText, weightBlob: blob, weightFileName: "weight.bin")
        }
        try m.compile()
        try m.load()
        model = m
    }

    deinit { model.unload() }

    /// CALLER THREAD ONLY (MLX): stage `x[S, in]` into the ANE input surface.
    func makeInput(_ x: MLXArray) throws -> ANEDirectDispatch.Prepared {
        try ANEDirectDispatch.prepare(
            model: model, x: x.asType(.float16), inputDim: inn, outputDim: out, sequenceLength: sequenceLength)
    }

    /// BACKGROUND-SAFE: runs the program, no MLX.
    func predict(_ p: ANEDirectDispatch.Prepared) throws { try ANEDirectDispatch.evaluate(p) }

    /// CALLER THREAD ONLY (MLX): read `[S, out]` fp16.
    func readOutput(_ p: ANEDirectDispatch.Prepared) -> MLXArray { ANEDirectDispatch.readZeroCopy(p) }

    /// Convenience for the caller thread: stage, run, read.
    func callAsFunction(_ x: MLXArray) throws -> MLXArray {
        let p = try makeInput(x)
        try predict(p)
        return readOutput(p)
    }
}

/// Per-module cache of ANE projections keyed by micro-batch length. A failed
/// build is remembered so the GPU path is taken without retrying every forward.
final class Qwen4ExpANEProjectionCache {
    private var programs: [Int: Qwen4ExpANEProjection] = [:]
    private var failed: Set<Int> = []
    let label: String

    init(label: String) { self.label = label }

    func program(weight: () -> MLXArray, sequenceLength: Int) -> Qwen4ExpANEProjection? {
        if let p = programs[sequenceLength] { return p }
        if failed.contains(sequenceLength) { return nil }
        do {
            let p = try Qwen4ExpANEProjection(weight: weight(), sequenceLength: sequenceLength)
            programs[sequenceLength] = p
            if Qwen4ExpANELane.log {
                fputs("[qwen4exp-ane] built \(label) [\(p.out)x\(p.inn)] S=\(sequenceLength)\n", stderr)
            }
            return p
        } catch {
            failed.insert(sequenceLength)
            fputs("[qwen4exp-ane] BUILD FAILED \(label) S=\(sequenceLength): \(error); GPU path stays\n", stderr)
            return nil
        }
    }
}

enum Qwen4ExpANESplitError: Error, CustomStringConvertible {
    case degenerateSplit(f: Int, out: Int)

    var description: String {
        switch self {
        case let .degenerateSplit(f, out):
            return "Qwen4ExpANESplitProjection: degenerate split f=\(f) out=\(out) (need 0 < f < out)"
        }
    }
}

/// One output-channel split of a dense `[out, in]` projection. The ANE owns
/// rows `[0, F)` as an fp16 1x1 conv program; the GPU owns rows `[F, out)` as a
/// bf16 `matmul`. The two partials concatenate along the last axis, which
/// reproduces the full projection's row order exactly.
///
/// The program is compiled at one fixed row count `sequenceLength` (a bucket).
/// `run` accepts any `T <= sequenceLength` and pads only the ANE leg; the GPU
/// leg always runs on the real `T` rows, so no GPU work is thrown away.
final class Qwen4ExpANESplitProjection {
    /// Rows of the PHYSICAL weight this program splits.
    let out: Int
    let inn: Int
    /// ANE row count. Always a multiple of 64, always `> 0`, always `< out`.
    let f: Int
    /// Compiled row count (the bucket).
    let sequenceLength: Int
    /// fp16 weight bytes this program holds resident.
    let programBytes: Int

    /// `nil` under `MLX_QWEN4EXP_ANE_FP16_GPU=1` (ablate): no ANE program is
    /// built at all and `run` computes the prefix rows as a GPU fp16 matmul.
    private let prefix: Qwen4ExpANEProjection?
    /// `weight[0 ..< f, 0...]` cast to fp16, evaluated once at build. Kept
    /// resident (not just handed to `Qwen4ExpANEProjection`) so the ablation
    /// matmul and the `MLX_QWEN4EXP_ANE_VERIFY=1` reference both have it.
    private let prefixWeightFP16: MLXArray
    /// `weight[f ..< out, 0...]`, evaluated once at build.
    private let suffixWeight: MLXArray
    /// `MLX_QWEN4EXP_ANE_FP16_GPU=1`: run the prefix as a GPU fp16 matmul
    /// instead of on the ANE. See spec 8.5.
    private let ablate: Bool
    /// `MLX_QWEN4EXP_ANE_VERIFY=1`: log the offloaded result against the
    /// pure-GPU reference it substitutes for.
    private let verify: Bool
    /// Only populated under `MLX_QWEN4EXP_ANE_VERIFY=1`: the unsliced,
    /// original-dtype weight, so `run` can compute that pure-GPU reference.
    private let referenceWeight: MLXArray?

    /// `weight` is the PHYSICAL `[out, in]` dense projection weight, any float
    /// dtype. `logicalOut` is the whole phase-1 output row count the fraction is
    /// a share of; the gated-delta lane passes `convDim + valueDim` while
    /// splitting only `in_proj_qkv`, and the attention lane passes
    /// `weight.dim(0)`. Throws when `F` is degenerate or when the ANE program
    /// cannot be built; the caller must then stay on the GPU.
    init(weight: MLXArray, logicalOut: Int, fraction: Double, sequenceLength: Int) throws {
        precondition(weight.ndim == 2, "Qwen4ExpANESplitProjection weight must be [out, in]")
        out = weight.dim(0)
        inn = weight.dim(1)
        self.sequenceLength = sequenceLength
        f = Qwen4ExpANEFused.prefixChannels(
            logicalOut: logicalOut, physicalOut: out, fraction: fraction)
        // The guard is against the PHYSICAL row count. `prefixChannels` clamps
        // to `out`, so a large fraction lands exactly on `out` and would leave
        // the GPU leg with zero rows: `weight[out ..< out]` is a `[0, in]`
        // array, the matmul yields `[T, 0]`, and the concat silently returns a
        // short row that no downstream shape check catches.
        guard f > 0, f < out else { throw Qwen4ExpANESplitError.degenerateSplit(f: f, out: out) }
        programBytes = f * inn * 2
        ablate = Qwen4ExpANEFused.fp16GpuAblate
        verify = Qwen4ExpANEFused.verify

        let prefixSlice = weight[0 ..< f, 0...].asType(.float16)
        eval(prefixSlice)
        prefixWeightFP16 = prefixSlice
        prefix = ablate ? nil : try Qwen4ExpANEProjection(weight: prefixSlice, sequenceLength: sequenceLength)

        let suffix = weight[f ..< out, 0...]
        eval(suffix)
        suffixWeight = suffix
        precondition(suffixWeight.dim(0) == out - f)

        if verify {
            eval(weight)
            referenceWeight = weight
        } else {
            referenceWeight = nil
        }
    }

    /// `x`: `[T, in]`, `T <= sequenceLength`, any float dtype.
    /// Returns `([T, out], extra)`.
    ///
    /// `gpuExtra` is additional GPU work to run inside the same concurrency
    /// window, on the caller thread, before the join. The gated-delta lane
    /// passes `in_proj_z` there so its rows never leave the GPU and no fused
    /// weight is ever materialized. It must not touch the ANE. `run` evaluates
    /// the suffix and every array `gpuExtra` returns, together, inside the GPU
    /// closure.
    func run(_ x: MLXArray, gpuExtra: (MLXArray) throws -> [MLXArray]) throws -> (MLXArray, [MLXArray]) {
        let tokens = x.dim(0)
        precondition(
            x.ndim == 2 && x.dim(1) == inn,
            "Qwen4ExpANESplitProjection.run expected x [T, \(inn)], got \(x.shape)")
        precondition(tokens <= sequenceLength, "Qwen4ExpANESplitProjection.run: T=\(tokens) > compiled \(sequenceLength)")

        let y: MLXArray
        let extraResult: [MLXArray]

        if ablate {
            // GPU-fp16 ablation (MLX_QWEN4EXP_ANE_FP16_GPU=1, diagnostic): the
            // SAME F prefix rows the ANE would compute, as a plain GPU fp16
            // matmul. No ANE dispatch, no padding, no private frameworks
            // touched. A divergence that survives this path is the fp16
            // REPRESENTATION, not ANE-specific (silu table / radix-4 /
            // surface padding). See spec 8.5.
            let x16 = x.asType(.float16)
            let anePart = matmul(x16, prefixWeightFP16.transposed(1, 0))
            let suffix = matmul(x, suffixWeight.transposed(1, 0))
            let extra = try gpuExtra(x)
            eval([anePart, suffix] + extra)
            y = concatenated([anePart.asType(suffix.dtype), suffix], axis: -1)
            extraResult = extra
        } else {
            guard let prefix else {
                preconditionFailure("Qwen4ExpANESplitProjection: ablate=false but no ANE program was built")
            }
            // CALLER THREAD. Pad the ANE leg only, in the SOURCE dtype. Do not
            // cast here: `Qwen4ExpANEProjection.makeInput` casts to fp16 and
            // `ANEDirectDispatch.prepare` casts again. Padding in fp16 would be
            // a third cast and would materialize a full `[bucket, in]` fp16
            // temp per layer per forward. `ANEFusedSplitMLP.padForANE` pads in
            // the source dtype for the same reason.
            let padded =
                tokens == sequenceLength
                ? x
                : concatenated([x, MLXArray.zeros([sequenceLength - tokens, inn], dtype: x.dtype)], axis: 0)

            // BARRIER 1: `prepare` calls `eval(xT)` internally, on this thread.
            let prepared = try prefix.makeInput(padded)

            let (_, gpu) = try ConcurrentEngines.run(
                ane: { try prefix.predict(prepared) },  // BACKGROUND, no MLX
                gpu: { () throws -> (MLXArray, [MLXArray]) in  // CALLER THREAD
                    let suffix = matmul(x, self.suffixWeight.transposed(1, 0))
                    let extra = try gpuExtra(x)
                    eval([suffix] + extra)  // BARRIER 2, mandatory
                    return (suffix, extra)
                })

            // CALLER THREAD. Read the ANE surface back, drop the padded rows,
            // join. `read` gathers eagerly, so the output surface is free once
            // `prepared` drops and no third barrier is needed. See spec 0.5.
            let aneFull = readBack(prepared)  // [sequenceLength, F] fp16
            let anePart = tokens == sequenceLength ? aneFull : aneFull[0 ..< tokens, 0...]
            precondition(gpu.0.dim(1) == out - f)
            y = concatenated([anePart.asType(gpu.0.dtype), gpu.0], axis: -1)  // [T, out]
            extraResult = gpu.1
            if Qwen4ExpANEFused.zeroCopyReadback { eval(y) }  // BARRIER 3, zero-copy only
        }

        if verify, let referenceWeight {
            // MLX_QWEN4EXP_ANE_VERIFY=1 (diagnostic): the pure-GPU result this
            // call substitutes for, using the ORIGINAL unsplit weight (not the
            // fp16 prefix slice), logged as a per-call max-abs error. Doubles
            // the matmul work of the offloaded piece; never use for timing.
            let ref = matmul(x, referenceWeight.transposed(1, 0).asType(x.dtype))
            let diff = MLX.abs(y.asType(.float32) - ref.asType(.float32))
            let maxAbs = diff.max()
            let maxRef = MLX.abs(ref.asType(.float32)).max()
            eval(maxAbs, maxRef)
            aneLog(String(
                format: "split verify out=%d F=%d T=%d: maxAbs=%.4f max|ref|=%.2f",
                out, f, tokens, maxAbs.item(Float.self), maxRef.item(Float.self)))
        }

        return (y, extraResult)
    }

    /// `run` with no extra GPU work. Passes `{ _ in [] }`.
    func callAsFunction(_ x: MLXArray) throws -> MLXArray {
        try run(x, gpuExtra: { _ in [] }).0
    }

    /// `read` by default; `readZeroCopy` under `MLX_QWEN4EXP_ANE_ZEROCOPY=1`.
    private func readBack(_ p: ANEDirectDispatch.Prepared) -> MLXArray {
        Qwen4ExpANEFused.zeroCopyReadback
            ? ANEDirectDispatch.readZeroCopy(p) : ANEDirectDispatch.read(p)
    }
}

/// Per-module cache of split programs keyed by
/// `Qwen4ExpANEFused.bucket(tokens)`. A bucket that failed to build, or that the
/// program budget refused, is remembered so the failure is paid once.
///
/// Guarded by an `NSLock`, matching `ANESplitMLPCache`. The Qwen4Exp prefill
/// runs layers sequentially on one caller thread, so the lock only guards the
/// dictionary against incidental races, but the two caches in this tree should
/// not differ on that by accident.
final class Qwen4ExpANESplitProjectionCache: @unchecked Sendable {
    private let lock = NSLock()
    private var programs: [Int: Qwen4ExpANESplitProjection] = [:]
    /// Buckets already tried and found unbuildable (or budget-refused), so the
    /// failure is paid once instead of on every forward.
    private var failed: Set<Int> = []
    let label: String

    init(label: String) { self.label = label }

    /// Returns a program for this token count, or nil meaning "use the GPU
    /// path". `weight` is only called on a cache miss.
    func program(forTokens tokens: Int, logicalOut: Int, weight: () -> MLXArray)
        -> Qwen4ExpANESplitProjection?
    {
        let key = Qwen4ExpANEFused.bucket(tokens)
        lock.lock()
        defer { lock.unlock() }
        if let p = programs[key] { return p }
        if failed.contains(key) { return nil }
        let w = weight()
        let f = Qwen4ExpANEFused.prefixChannels(
            logicalOut: logicalOut, physicalOut: w.dim(0),
            fraction: Qwen4ExpANEFused.splitFraction)
        guard f > 0, f < w.dim(0) else {
            failed.insert(key)
            fputs("[qwen4exp-ane] SPLIT DEGENERATE \(label) F=\(f) out=\(w.dim(0)); GPU path stays\n", stderr)
            return nil
        }
        let bytes = f * w.dim(1) * 2
        guard Qwen4ExpANEFused.reserveProgram(bytes: bytes, label: label) else {
            failed.insert(key)
            return nil  // reserveProgram already logged
        }
        do {
            let p = try Qwen4ExpANESplitProjection(
                weight: w, logicalOut: logicalOut,
                fraction: Qwen4ExpANEFused.splitFraction, sequenceLength: key)
            programs[key] = p
            if Qwen4ExpANEFused.log {
                fputs("[qwen4exp-ane] split \(label) out=\(p.out) F=\(p.f) bucket=\(key)\n", stderr)
            }
            return p
        } catch {
            // Give the reservation back. A build that failed after reserving
            // would otherwise hold budget for a program that does not exist,
            // and 48 layers times two buckets can retire the count limit
            // against zero resident programs.
            Qwen4ExpANEFused.releaseProgram(bytes: bytes)
            failed.insert(key)
            fputs("[qwen4exp-ane] SPLIT BUILD FAILED \(label) bucket=\(key): \(error); GPU path stays\n", stderr)
            return nil
        }
    }
}
