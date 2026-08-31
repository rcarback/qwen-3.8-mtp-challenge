import Foundation
import MLX
import MLXFastModel
import MLXNN
import MLXRandom
import Testing

/// Task 6 (ANE+GPU concurrent offload plan): "Approach B / channel-split" --
/// each of `gate`/`up`/`down` splits its OUTPUT channels between the ANE
/// (fp16 prefix, via `ANEGemm`) and the GPU (4-bit `quantizedMM` suffix),
/// run concurrently, then concatenated. This is a CORRECTNESS-ONLY test;
/// timing is deferred to Task 7 (A-vs-B comparison, run on a user-pinged
/// idle box). See
/// `.superpowers/sdd/2026-08-30-ane-gpu-concurrent-offload/task-6-brief.md`.
@Suite(.serialized)
struct ChannelSplitMLPTests {
    /// Quantizes a random bf16 weight to the shipped 4-bit affine group-64
    /// form and returns the (wq, scales, biases) triple `quantizedMM` and
    /// `ChannelSplitMLP` both expect.
    ///
    /// Weights are scaled by `1/sqrt(inn)` so that `x @ w.T` over a
    /// std-normal `x` lands each output element around unit variance --
    /// matching the well-conditioned post-RMSNorm activations a real
    /// transformer MLP sees. Unscaled N(0,1) weights blow this shape up
    /// through the wide contractions (`CoarseOffloadMLPTests` measured
    /// maxAbsDiff=16384.0 with unscaled weights, a value with no
    /// interpretable relationship to fp16/4-bit noise); scaling first makes
    /// the absolute tolerance below actually mean something.
    private static func quantizedWeight(out: Int, inn: Int, seed: UInt64) -> (MLXArray, MLXArray, MLXArray) {
        MLXRandom.seed(seed)
        let scale = Float(1.0 / Double(inn).squareRoot())
        let w = (MLXRandom.normal([out, inn]) * scale).asType(.bfloat16)
        let (wq, scales, biases0) = quantized(w, groupSize: 64, bits: 4)
        let biases = biases0 ?? scales
        eval(wq, scales, biases)
        return (wq, scales, biases)
    }

    /// All-GPU reference MLP: `down(silu(gate(x)) * up(x))`, every
    /// projection via the real 4-bit affine group-64 `quantizedMM` path --
    /// the same weights `ChannelSplitMLP` consumes, so the only difference
    /// is that a fraction of each projection's output channels run through
    /// the ANE at fp16 instead of through `quantizedMM` at 4-bit.
    private static func referenceForward(
        _ x: MLXArray,
        gateWq: MLXArray, gateScales: MLXArray, gateBiases: MLXArray,
        upWq: MLXArray, upScales: MLXArray, upBiases: MLXArray,
        downWq: MLXArray, downScales: MLXArray, downBiases: MLXArray
    ) -> MLXArray {
        let gate = quantizedMM(x, gateWq, scales: gateScales, biases: gateBiases,
                                transpose: true, groupSize: 64, bits: 4)
        let up = quantizedMM(x, upWq, scales: upScales, biases: upBiases,
                              transpose: true, groupSize: 64, bits: 4)
        let h = (silu(gate) * up).asType(.bfloat16)
        let y = quantizedMM(h, downWq, scales: downScales, biases: downBiases,
                             transpose: true, groupSize: 64, bits: 4)
        eval(y)
        return y
    }

    private struct Weights {
        let gateWq: MLXArray, gateScales: MLXArray, gateBiases: MLXArray
        let upWq: MLXArray, upScales: MLXArray, upBiases: MLXArray
        let downWq: MLXArray, downScales: MLXArray, downBiases: MLXArray
    }

    private static func makeWeights(hidden: Int, inter: Int, seed: UInt64) -> Weights {
        let (gateWq, gateScales, gateBiases) = quantizedWeight(out: inter, inn: hidden, seed: seed)
        let (upWq, upScales, upBiases) = quantizedWeight(out: inter, inn: hidden, seed: seed + 1)
        let (downWq, downScales, downBiases) = quantizedWeight(out: hidden, inn: inter, seed: seed + 2)
        return Weights(
            gateWq: gateWq, gateScales: gateScales, gateBiases: gateBiases,
            upWq: upWq, upScales: upScales, upBiases: upBiases,
            downWq: downWq, downScales: downScales, downBiases: downBiases)
    }

    /// Tolerance on the final `[S,hidden]` max-abs error against the
    /// all-GPU reference, measured directly against THIS implementation --
    /// not borrowed from `CoarseOffloadMLPTests`. Reasoning by "offloads
    /// less" is wrong here: gate/up/down are equal-FLOP projections,
    /// `aneFraction=0.4` puts 40% of the output channels of ALL THREE
    /// through fp16 ANE, and fp16 error on `gate`/`up` leaks through the
    /// elementwise `silu(gate)*up` into `down`'s contraction. A live run at
    /// the ranked shape (`aneFraction=0.4`, S=512 and S=500, same
    /// `1/sqrt(inn)` weight scaling as `CoarseOffloadMLPTests`) measured
    /// maxAbsDiff=0.015625 on both S. This bound is ~6x that: 0.015625*6 =
    /// 0.09375 -- generous margin while still well below the
    /// unscaled-weights failure mode of maxAbsDiff in the thousands (which
    /// would flag a real layout/axis/dtype bug).
    private static let tolerance: Float = 0.09375

    /// Bound on the mean-abs error, alongside the max-abs bound above. A
    /// max-only gate would pass a hypothetical bug that adds a small
    /// *uniform* bias to every element (well under the max bound, but a real
    /// correctness bug); mean-abs catches that. The same live run measured
    /// meanAbsDiff~0.00084-0.00085 at S=512/S=500; ~12x margin.
    private static let meanTolerance: Float = 0.01

    private func assertMatchesReference(S: Int, aneFraction: Double, seed: UInt64, tol: Float = tolerance) throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        let hidden = 5120, inter = 17408
        let weights = Self.makeWeights(hidden: hidden, inter: inter, seed: seed)

        MLXRandom.seed(seed + 3)
        let x = MLXRandom.normal([S, hidden]).asType(.bfloat16)
        eval(x)

        let want = Self.referenceForward(
            x,
            gateWq: weights.gateWq, gateScales: weights.gateScales, gateBiases: weights.gateBiases,
            upWq: weights.upWq, upScales: weights.upScales, upBiases: weights.upBiases,
            downWq: weights.downWq, downScales: weights.downScales, downBiases: weights.downBiases)

        let mlp = try ChannelSplitMLP(
            gateW: weights.gateWq, gateScales: weights.gateScales, gateBiases: weights.gateBiases,
            upW: weights.upWq, upScales: weights.upScales, upBiases: weights.upBiases,
            downW: weights.downWq, downScales: weights.downScales, downBiases: weights.downBiases,
            sequenceLength: S, aneFraction: aneFraction)
        let got = try mlp(x)
        eval(got)

        #expect(got.shape == want.shape, "shape mismatch: got \(String(describing: got.shape)), want \(String(describing: want.shape))")
        let absDiff = abs(got.asType(.float32) - want.asType(.float32))
        let maxAbsDiff = absDiff.max().item(Float.self)
        let meanAbsDiff = absDiff.mean().item(Float.self)
        print("ChannelSplitMLPTests S=\(S) aneFraction=\(aneFraction) seed=\(seed): maxAbsDiff=\(maxAbsDiff) meanAbsDiff=\(meanAbsDiff)")
        #expect(maxAbsDiff < tol, "ChannelSplitMLP diverged from all-GPU reference: maxAbsErr=\(maxAbsDiff), aneFraction=\(aneFraction)")
        #expect(meanAbsDiff < Self.meanTolerance, "ChannelSplitMLP diverged from all-GPU reference: meanAbsErr=\(meanAbsDiff), aneFraction=\(aneFraction)")
    }

    @Test("ChannelSplitMLP(aneFraction=0.4) matches all-GPU reference at the ranked prefill shape (S=512)")
    func matchesReferenceS512() throws {
        try assertMatchesReference(S: 512, aneFraction: 0.4, seed: 10)
    }

    @Test("ChannelSplitMLP(aneFraction=0.4) matches all-GPU reference at a non-32-multiple S=500")
    func matchesReferenceS500() throws {
        try assertMatchesReference(S: 500, aneFraction: 0.4, seed: 20)
    }

    /// aneFraction=0.0 must take the pure-GPU path on every projection (F=0
    /// everywhere, no ANE involvement at all), so the result must be
    /// bit-identical to the all-GPU reference -- both compute the identical
    /// `quantizedMM` calls on the identical weights. A live run measured
    /// maxAbsDiff=0.0, confirming the exact-equality expectation, so this
    /// asserts `== 0` rather than an arbitrary small tolerance. NEVER weaken
    /// this back to a nonzero tolerance -- a nonzero residual here would mean
    /// the ANE path is silently touched at aneFraction=0.0.
    @Test("ChannelSplitMLP(aneFraction=0.0) is identical to all-GPU reference")
    func zeroFractionMatchesReferenceExactly() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        let hidden = 5120, inter = 17408, S = 512
        let weights = Self.makeWeights(hidden: hidden, inter: inter, seed: 30)

        MLXRandom.seed(33)
        let x = MLXRandom.normal([S, hidden]).asType(.bfloat16)
        eval(x)

        let want = Self.referenceForward(
            x,
            gateWq: weights.gateWq, gateScales: weights.gateScales, gateBiases: weights.gateBiases,
            upWq: weights.upWq, upScales: weights.upScales, upBiases: weights.upBiases,
            downWq: weights.downWq, downScales: weights.downScales, downBiases: weights.downBiases)

        let mlp = try ChannelSplitMLP(
            gateW: weights.gateWq, gateScales: weights.gateScales, gateBiases: weights.gateBiases,
            upW: weights.upWq, upScales: weights.upScales, upBiases: weights.upBiases,
            downW: weights.downWq, downScales: weights.downScales, downBiases: weights.downBiases,
            sequenceLength: S, aneFraction: 0.0)
        let got = try mlp(x)
        eval(got)

        let maxAbsDiff = (abs(got.asType(.float32) - want.asType(.float32)).max()).item(Float.self)
        print("ChannelSplitMLPTests aneFraction=0.0: maxAbsDiff=\(maxAbsDiff)")
        #expect(maxAbsDiff == 0, "aneFraction=0.0 must be pure-GPU and bit-identical to the reference: maxAbsErr=\(maxAbsDiff)")
    }

    /// aneFraction outside [0,1] must clamp rather than crash or produce
    /// garbage: -0.5 clamps to 0.0 (pure GPU, matches reference tightly),
    /// 1.5 clamps to 1.0 (pure ANE on every projection, fp16 tolerance).
    @Test("ChannelSplitMLP clamps aneFraction outside [0,1]")
    func clampsFractionOutsideUnitRange() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        try assertMatchesReference(S: 512, aneFraction: -0.5, seed: 40, tol: 1e-3)
        try assertMatchesReference(S: 512, aneFraction: 1.5, seed: 50, tol: Self.tolerance)
    }

    /// Mixed-edge combination: at `aneFraction=0.996`, `down` (out=5120,
    /// F=round(0.996*5120/64)*64=5120) hits its F==out pure-ANE edge while
    /// gate/up (out=17408, F=round(0.996*17408/64)*64=17344) still have a
    /// nonzero GPU suffix (17408-17344=64 channels). The three projections
    /// are NOT all on the same edge/interior case at once anywhere else this
    /// suite exercises, so this specifically covers `callAsFunction` combining
    /// a pure-ANE `down` result with interior-split `gate`/`up` results.
    @Test("ChannelSplitMLP(aneFraction=0.996) exercises down==pure-ANE while gate/up retain a GPU suffix")
    func mixedEdgeDownPureANEGateUpInterior() throws {
        try assertMatchesReference(S: 512, aneFraction: 0.996, seed: 60, tol: Self.tolerance)
    }

    /// Locks the frozen 4-bit envelope property `ChannelSplitMLP` depends on:
    /// row-slicing a shipped 4-bit affine group-64 operand along the output
    /// (row) axis and running `quantizedMM` on the slice must be bit-identical
    /// to running `quantizedMM` on the full operand and then slicing the
    /// output columns. This is what makes `ChannelSplitMLP`'s GPU suffix (a
    /// row slice of `wq`/`scales`/`biases`, never dequantized-then-requantized)
    /// safe -- `ChannelSplitMLPTests`'s MLP-level tolerance tests cannot
    /// isolate this property from other sources of fp16/4-bit noise, since
    /// they always compare against a full-precision-per-projection reference.
    @Test("row-slicing a 4-bit quantized operand is bit-identical to slicing quantizedMM's output")
    func rowSliceOfQuantizedOperandIsLosslessEnvelope() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        let out = 256, inn = 128, S = 8, f = 128
        let (wq, scales, biases) = Self.quantizedWeight(out: out, inn: inn, seed: 70)

        MLXRandom.seed(71)
        let x = MLXRandom.normal([S, inn]).asType(.bfloat16)
        eval(x)

        let fullOut = quantizedMM(x, wq, scales: scales, biases: biases,
                                   transpose: true, groupSize: 64, bits: 4)
        eval(fullOut)
        let wantSuffix = fullOut[0..., f...]

        let sliceWq = wq[f ..< out, 0...]
        let sliceScales = scales[f ..< out, 0...]
        let sliceBiases = biases[f ..< out, 0...]
        eval(sliceWq, sliceScales, sliceBiases)
        let gotSuffix = quantizedMM(x, sliceWq, scales: sliceScales, biases: sliceBiases,
                                     transpose: true, groupSize: 64, bits: 4)
        eval(gotSuffix)

        let maxAbsDiff = (abs(gotSuffix.asType(.float32) - wantSuffix.asType(.float32)).max()).item(Float.self)
        print("rowSliceOfQuantizedOperandIsLosslessEnvelope: maxAbsDiff=\(maxAbsDiff)")
        #expect(maxAbsDiff == 0, "row-sliced quantizedMM must be bit-identical to slicing the full output: maxAbsErr=\(maxAbsDiff)")
    }
}
