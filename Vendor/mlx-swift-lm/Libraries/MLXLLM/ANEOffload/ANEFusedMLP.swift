// LOCAL M4 FORK ONLY. Private Apple frameworks; breaks on macOS updates.
// Never wired into the ranked forward; gated by MLXFAST_ANE_DIRECT=1 at call sites.
//
// Task B (ANE IOSurface / procedure-bank plan): the fused SwiGLU-down MLP as
// a SINGLE ANE program -- gate-conv, up-conv, silu, mul, down-conv all in
// one dispatch, no per-projection barrier. Builds the fused MIL text and
// multi-weight blob (`ANEMILBuilder.buildSwiGLUDownMILText` /
// `buildMultiWeightBlob`), compiles+loads it once at init
// (`ANEInMemoryModel`), and runs it through Task A's IOSurface dispatch
// (`ANEDirectDispatch.runConv`, whose input/output-dim + sequenceLength
// interface is generic over what op the loaded program actually computes --
// reused as-is, no duplicated padding/scatter-gather logic). See
// `.superpowers/sdd/2026-08-31-ane-iosurface-procedure-bank/task-B-brief.md`.
import CryptoKit
import Foundation
import MLX

public final class ANEFusedMLP {
    let hidden: Int
    let innerFraction: Int
    let sequenceLength: Int
    private let model: ANEInMemoryModel

    /// `gate`/`up`: `[innerFraction, hidden]`. `down`: `[hidden, innerFraction]`.
    /// Computes the ANE fraction's full contribution through the down
    /// projection: `down @ (silu(gate @ xT) * (up @ xT))` for `x=[S,hidden]`.
    public init(
        hidden: Int, innerFraction: Int, sequenceLength: Int,
        gate: MLXArray, up: MLXArray, down: MLXArray,
        activation: ANEActivation = ANESplitConfig.activation,
        weightForm: ANEWeightForm = ANESplitConfig.weightForm
    ) throws {
        precondition(gate.shape == [innerFraction, hidden],
                     "ANEFusedMLP: gate expected [\(innerFraction), \(hidden)], got \(gate.shape)")
        precondition(up.shape == [innerFraction, hidden],
                     "ANEFusedMLP: up expected [\(innerFraction), \(hidden)], got \(up.shape)")
        precondition(down.shape == [hidden, innerFraction],
                     "ANEFusedMLP: down expected [\(hidden), \(innerFraction)], got \(down.shape)")
        self.hidden = hidden
        self.innerFraction = innerFraction
        self.sequenceLength = sequenceLength

        // The program identity the ANE runtime derives covers the MIL text
        // only, so the tag stamped into `buildInfo` is what separates one
        // layer's program from another's (see `ANEInMemoryModel`). A random
        // tag did that, but it also made every process's 64 programs new to
        // the ANE daemon's compiled-program cache: ~100 MB each, never hit
        // again, ~6 GB of root-owned cache per hybrid process. Hashing the
        // weight blob instead gives the same program the same identity in
        // every process, so a rebuild of an unchanged layer at the same
        // shape and activation is a cache hit: no recompile, no growth.
        // Shape and activation are already in the text the hash sits in.
        func tag(_ blob: Data) -> String {
            "sha256:" + SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        }
        let blob: Data
        let milText: String
        switch weightForm {
        case .fp16:
            let (b, offsets) = buildMultiWeightBlob(chunks: [f16Bytes(gate), f16Bytes(up), f16Bytes(down)])
            blob = b
            milText = buildSwiGLUDownMILText(
                inputDim: hidden, hiddenDim: innerFraction, outputDim: hidden, sequenceLength: sequenceLength,
                gateOffset: offsets[0], upOffset: offsets[1], downOffset: offsets[2],
                activation: activation, programTag: tag(b)
            )
        case .int8:
            let g = ANEWeightQuant.int8PerChannel(gate)
            let u = ANEWeightQuant.int8PerChannel(up)
            let dn = ANEWeightQuant.int8PerChannel(down)
            let (b, o) = buildMultiWeightBlob(chunks: [g.data, u.data, dn.data, g.scale, u.scale, dn.scale])
            blob = b
            milText = buildSwiGLUDownMILTextCompressed(
                inputDim: hidden, hiddenDim: innerFraction, outputDim: hidden, sequenceLength: sequenceLength,
                weights: .int8(gate: o[0], up: o[1], down: o[2], gateScale: o[3], upScale: o[4], downScale: o[5]),
                activation: activation, programTag: tag(b)
            )
        case .int4:
            let codebook = ANEWeightQuant.fitCodebook([gate, up, down])
            let g = ANEWeightQuant.int4Palette(gate, codebook: codebook)
            let u = ANEWeightQuant.int4Palette(up, codebook: codebook)
            let dn = ANEWeightQuant.int4Palette(down, codebook: codebook)
            let (b, o) = buildMultiWeightBlob(
                chunks: [g.indices, u.indices, dn.indices, g.lut, g.scale, u.scale, dn.scale],
                chunkTypes: [3, 3, 3, 1, 1, 1, 1])
            blob = b
            milText = buildSwiGLUDownMILTextCompressed(
                inputDim: hidden, hiddenDim: innerFraction, outputDim: hidden, sequenceLength: sequenceLength,
                weights: .int4(gateIdx: o[0], upIdx: o[1], downIdx: o[2], lut: o[3], gateScale: o[4], upScale: o[5], downScale: o[6]),
                activation: activation, programTag: tag(b)
            )
        }
        let m = try ANEInMemoryModel(milText: milText, weightBlob: blob, weightFileName: "weight.bin")
        try m.compile()
        try m.load()
        model = m
    }

    deinit {
        model.unload()
    }

    /// `x`: `[S, hidden]` fp16. Returns `[S, hidden]` fp16. Convenience that
    /// chains `makeInput` -> `predict` -> `readOutput` on the calling
    /// thread. For concurrent ANE+GPU use (Task D1), call the three parts
    /// separately instead -- see their doc comments below.
    public func callAsFunction(_ x: MLXArray) throws -> MLXArray {
        try ANEDirectDispatch.runConv(model: model, x: x, inputDim: hidden, outputDim: hidden, sequenceLength: sequenceLength)
    }

    /// CALLER THREAD ONLY (MLX). Prepares the ANE input/output IOSurfaces
    /// and the `_ANERequest` for `x[S,hidden]` fp16. Only `predict` on the
    /// returned handle may run off this thread.
    func makeInput(_ x: MLXArray) throws -> ANEDirectDispatch.Prepared {
        try ANEDirectDispatch.prepare(model: model, x: x, inputDim: hidden, outputDim: hidden, sequenceLength: sequenceLength)
    }

    /// BACKGROUND-SAFE. Runs the fused ANE program via the blocking ObjC
    /// evaluate -- no MLX. Safe to run on `ConcurrentEngines.run`'s
    /// background queue concurrently with GPU MLX work on the caller
    /// thread.
    func predict(_ prepared: ANEDirectDispatch.Prepared) throws {
        try ANEDirectDispatch.evaluate(prepared)
    }

    /// CALLER THREAD ONLY (MLX). Reads the ANE output surface into an
    /// `[S, hidden]` fp16 `MLXArray`. Call only after `predict` has
    /// completed for this handle.
    func readOutput(_ prepared: ANEDirectDispatch.Prepared) -> MLXArray {
        ANEDirectDispatch.readZeroCopy(prepared)
    }
}
