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
    /// all-GPU reference. `CoarseOffloadMLPTests` (which offloads the
    /// *entire* `up` projection to fp16 ANE) measured maxAbsDiff=0.015625
    /// with the same `1/sqrt(inn)` weight scaling and set tolerance=0.1
    /// (~6x margin). Here at aneFraction=0.4 only 40% of the output
    /// channels of EACH of gate/up/down run through fp16 -- a smaller
    /// fraction of a smaller fraction of the compute than the coarse case,
    /// so the coarse bound is not tighter than what this split can produce;
    /// reusing it keeps the same interpretable margin (still far below the
    /// unscaled-weights failure mode of maxAbsDiff in the thousands, which
    /// would flag a real layout/axis/dtype bug).
    private static let tolerance: Float = 0.1

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
        let maxAbsDiff = (abs(got.asType(.float32) - want.asType(.float32)).max()).item(Float.self)
        #expect(maxAbsDiff < tol, "ChannelSplitMLP diverged from all-GPU reference: maxAbsErr=\(maxAbsDiff), aneFraction=\(aneFraction)")
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
    /// everywhere), so the result should be identical (bit-for-bit modulo
    /// the harmless bf16 round-trip cast `projSplit` applies uniformly) to
    /// the all-GPU reference -- no ANE involvement at all.
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
        #expect(maxAbsDiff < 1e-3, "aneFraction=0.0 should be pure-GPU and match the reference near-exactly: maxAbsErr=\(maxAbsDiff)")
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
}
