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

    init(weight: MLXArray, sequenceLength: Int) throws {
        precondition(weight.ndim == 2, "Qwen4ExpANEProjection weight must be [out, in], got \(weight.shape)")
        out = weight.dim(0)
        inn = weight.dim(1)
        self.sequenceLength = sequenceLength
        let blob = buildConvWeightBlob(f16Bytes(weight))
        // Same identity for the same weights in every process: the ANE daemon's
        // compiled-program cache hits instead of growing (see ANEFusedMLP).
        let programTag = "sha256:" + SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        let milText = buildConvMILText(
            inputDim: inn, outputDim: out, sequenceLength: sequenceLength, programTag: programTag)
        let m = try ANEInMemoryModel(milText: milText, weightBlob: blob, weightFileName: "weight_data.bin")
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
