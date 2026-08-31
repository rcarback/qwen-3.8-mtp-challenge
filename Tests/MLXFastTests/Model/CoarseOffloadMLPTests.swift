import Foundation
import MLX
import MLXFastModel
import MLXNN
import MLXRandom
import Testing

/// Task 5 (ANE+GPU concurrent offload plan): "Approach A / coarse" -- the
/// whole `up` projection runs on the ANE (fp16) concurrent with the whole
/// `gate` projection on the GPU (4-bit `quantizedMM`), then `down` runs on
/// the GPU. This is a CORRECTNESS-ONLY test; timing is deferred to Task 7
/// (A-vs-B comparison, run on a user-pinged idle box). See
/// `.superpowers/sdd/2026-08-30-ane-gpu-concurrent-offload/task-5-brief.md`.
@Suite(.serialized)
struct CoarseOffloadMLPTests {
    /// Quantizes a random bf16 weight to the shipped 4-bit affine group-64
    /// form and returns the (wq, scales, biases) triple `quantizedMM` and
    /// `CoarseOffloadMLP` both expect.
    ///
    /// Weights are scaled by `1/sqrt(inn)` so that `x @ w.T` over a
    /// std-normal `x` lands each output element around unit variance --
    /// matching the well-conditioned post-RMSNorm activations a real
    /// transformer MLP sees. Unscaled N(0,1) weights (both projections'
    /// natural draw) blow this shape up: a 5120-wide contraction already
    /// puts values around std~72, `silu(gate)*up` squares that to ~5000,
    /// and a second 17408-wide contraction in `down` compounds it further to
    /// ~10^5-10^6 -- at that scale a fixed absolute tolerance is meaningless
    /// (confirmed: an early run of this test with unscaled weights measured
    /// maxAbsDiff=16384.0, a value with no interpretable relationship to
    /// fp16/4-bit noise). Scaling first makes the absolute tolerance below
    /// actually mean something.
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
    /// the same weights `CoarseOffloadMLP` consumes, so the only difference
    /// is that its `up` runs through the ANE at fp16 instead of through
    /// `quantizedMM` at 4-bit.
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

    /// Tolerance on the final `[S,hidden]` max-abs error, empirically
    /// measured (not a literal-picked-to-pass value): with weights scaled by
    /// `1/sqrt(inn)` (see `quantizedWeight` above, which keeps activations
    /// at the ~unit-variance scale real post-RMSNorm hidden states have),
    /// both S=512 and S=500 measured maxAbsDiff=0.015625 (meanAbsDiff
    /// ~0.00135). That is the fp16-vs-4bit noise on a single `up`
    /// projection surviving the elementwise `silu(gate)*up` and the
    /// `down` contraction over inter=17408 terms. This bound gives ~6x
    /// margin over that measurement -- generous, but still tight enough to
    /// catch a real correctness bug (a wrong layout, dtype, or dropped
    /// row would produce errors orders of magnitude larger, as the
    /// unscaled-weights run of this same test did: maxAbsDiff=16384.0).
    private static let tolerance: Float = 0.1

    /// Bound on the mean-abs error, alongside the max-abs bound above. A
    /// max-only gate would pass a hypothetical bug that adds a small
    /// *uniform* bias to every element (well under 0.1 max, but a real
    /// correctness bug); mean-abs catches that. Measured meanAbsDiff at
    /// S=512/S=500 was ~0.00135 (see `tolerance`'s comment); ~6x margin.
    private static let meanTolerance: Float = 0.01

    /// Shared body: `CoarseOffloadMLP(x)` must match `referenceForward` (the
    /// all-GPU MLP with the same weights) within fp16-vs-4bit noise on the
    /// `up` projection. See `tolerance` above for how the bound was set.
    private func assertMatchesReference(S: Int, seed: UInt64) throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        let hidden = 5120, inter = 17408
        let (gateWq, gateScales, gateBiases) = Self.quantizedWeight(out: inter, inn: hidden, seed: seed)
        let (upWq, upScales, upBiases) = Self.quantizedWeight(out: inter, inn: hidden, seed: seed + 1)
        let (downWq, downScales, downBiases) = Self.quantizedWeight(out: hidden, inn: inter, seed: seed + 2)

        MLXRandom.seed(seed + 3)
        let x = MLXRandom.normal([S, hidden]).asType(.bfloat16)
        eval(x)

        let want = Self.referenceForward(
            x,
            gateWq: gateWq, gateScales: gateScales, gateBiases: gateBiases,
            upWq: upWq, upScales: upScales, upBiases: upBiases,
            downWq: downWq, downScales: downScales, downBiases: downBiases)

        let mlp = try CoarseOffloadMLP(
            gateW: gateWq, gateScales: gateScales, gateBiases: gateBiases,
            upW: upWq, upScales: upScales, upBiases: upBiases,
            downW: downWq, downScales: downScales, downBiases: downBiases,
            sequenceLength: S)
        let got = try mlp(x)
        eval(got)

        #expect(got.shape == want.shape, "shape mismatch: got \(String(describing: got.shape)), want \(String(describing: want.shape))")
        let absDiff = abs(got.asType(.float32) - want.asType(.float32))
        let maxAbsDiff = absDiff.max().item(Float.self)
        let meanAbsDiff = absDiff.mean().item(Float.self)
        print("CoarseOffloadMLPTests S=\(S) seed=\(seed): maxAbsDiff=\(maxAbsDiff) meanAbsDiff=\(meanAbsDiff)")
        #expect(maxAbsDiff < Self.tolerance, "CoarseOffloadMLP diverged from all-GPU reference: maxAbsErr=\(maxAbsDiff)")
        #expect(meanAbsDiff < Self.meanTolerance, "CoarseOffloadMLP diverged from all-GPU reference: meanAbsErr=\(meanAbsDiff)")
    }

    @Test("CoarseOffloadMLP matches all-GPU reference at the ranked prefill shape (S=512)")
    func matchesReferenceS512() throws {
        try assertMatchesReference(S: 512, seed: 10)
    }

    @Test("CoarseOffloadMLP matches all-GPU reference at a non-32-multiple S=500")
    func matchesReferenceS500() throws {
        try assertMatchesReference(S: 500, seed: 20)
    }
}
