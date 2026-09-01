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
    public init(hidden: Int, innerFraction: Int, sequenceLength: Int, gate: MLXArray, up: MLXArray, down: MLXArray) throws {
        precondition(gate.shape == [innerFraction, hidden],
                     "ANEFusedMLP: gate expected [\(innerFraction), \(hidden)], got \(gate.shape)")
        precondition(up.shape == [innerFraction, hidden],
                     "ANEFusedMLP: up expected [\(innerFraction), \(hidden)], got \(up.shape)")
        precondition(down.shape == [hidden, innerFraction],
                     "ANEFusedMLP: down expected [\(hidden), \(innerFraction)], got \(down.shape)")
        self.hidden = hidden
        self.innerFraction = innerFraction
        self.sequenceLength = sequenceLength

        let (blob, offsets) = buildMultiWeightBlob(chunks: [f16Bytes(gate), f16Bytes(up), f16Bytes(down)])
        let milText = buildSwiGLUDownMILText(
            inputDim: hidden, hiddenDim: innerFraction, outputDim: hidden, sequenceLength: sequenceLength,
            gateOffset: offsets[0], upOffset: offsets[1], downOffset: offsets[2]
        )
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
