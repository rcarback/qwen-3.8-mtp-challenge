import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXFastModel

/// Task C (ANE IOSurface / procedure-bank plan): ANE-fraction-split MLP
/// (Task B's fused ANE MLP for the first `F` intermediate channels + a
/// native 4-bit GPU partial for the rest) validated at REAL Qwen
/// dimensions (hidden=5120, inter=17408). Step 1 is the STOP-GATE: a
/// real-size fused-program compile smoke test, run before anything else,
/// because a fused ANE program at this size had never been compiled
/// before this task. See
/// `.superpowers/sdd/2026-08-31-ane-iosurface-procedure-bank/task-C-brief.md`.
@Suite(.serialized)
struct ANEFusedSplitMLPTests {
    private static let hidden = 5_120
    private static let inter = 17_408

    // MARK: - Step 1: STOP-GATE -- real-size fused compile smoke test

    /// Builds JUST `ANEFusedMLP` at real Qwen size (hidden=5120, F=2176,
    /// S=512) with random fp16 weights and runs it once. If this does not
    /// compile/load/run cleanly, the whole split is moot -- this must be
    /// green before Step 2's split test is written.
    @Test("STOP-GATE: ANEFusedMLP compiles and runs at real Qwen size (hidden=5120, F=2176, S=512)")
    func stopGateRealSizeFusedCompile() throws {
        try #require(ANERuntime.available())
        let hidden = Self.hidden
        let f = 2_176 // round(0.125 * 17408 / 64) * 64
        let s = 512

        let gate = (MLXRandom.normal([f, hidden]) / Float(hidden).squareRoot()).asType(.float16)
        let up = (MLXRandom.normal([f, hidden]) / Float(hidden).squareRoot()).asType(.float16)
        let down = (MLXRandom.normal([hidden, f]) / Float(f).squareRoot()).asType(.float16)
        let x = MLXRandom.normal([s, hidden]).asType(.float16)
        eval(gate, up, down, x)

        let start = Date()
        let mlp = try ANEFusedMLP(hidden: hidden, innerFraction: f, sequenceLength: s, gate: gate, up: up, down: down)
        let compileTime = Date().timeIntervalSince(start)
        print("STOP-GATE real-size ANEFusedMLP compile+load time: \(compileTime)s")

        let y = try mlp(x)
        eval(y)
        #expect(y.shape == [s, hidden])

        let yf = y.asType(.float32)
        #expect(yf.abs().max().item(Float.self).isFinite)

        let xf = x.asType(.float32)
        let g = matmul(xf, gate.asType(.float32).transposed(1, 0))
        let u = matmul(xf, up.asType(.float32).transposed(1, 0))
        let act = silu(g) * u
        let expected = matmul(act, down.asType(.float32).transposed(1, 0))
        eval(expected)

        let diff = MLX.abs(yf - expected)
        eval(diff)
        let maxAbs = diff.max().item(Float.self)
        let meanAbs = diff.mean().item(Float.self)
        print("STOP-GATE real-size ANEFusedMLP error: maxAbs=\(maxAbs) meanAbs=\(meanAbs)")
        #expect(maxAbs < 0.1, "maxAbs=\(maxAbs)")
        #expect(meanAbs < 0.01, "meanAbs=\(meanAbs)")
    }

    // MARK: - Step 2: split == all-GPU reference

    private static func quantizedWeight(out: Int, inn: Int, seed: UInt64) -> (MLXArray, MLXArray, MLXArray) {
        MLXRandom.seed(seed)
        let scale = Float(1.0 / Double(inn).squareRoot())
        let w = (MLXRandom.normal([out, inn]) * scale).asType(.bfloat16)
        let (wq, scales, biases0) = quantized(w, groupSize: 64, bits: 4)
        let biases = biases0 ?? scales
        eval(wq, scales, biases)
        return (wq, scales, biases)
    }

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

    /// Bound on max-abs error vs. the all-GPU reference at `aneFraction=0.125`
    /// (F=2176 of gate/up/down's shared intermediate channels run fp16 on
    /// the ANE; the rest run native 4-bit `quantizedMM`), at the ranked
    /// prefill shape S=512. The task brief's starting bound is
    /// `(maxAbs<1.0, meanAbs<0.05)`; a live run measured
    /// `maxAbsDiff=0.015625, meanAbsDiff=0.0017330822`
    /// (`ChannelSplitMLPTests` measured the identical `maxAbsDiff=0.015625`
    /// at its own real-size shape with a similar fraction of fp16-vs-4bit
    /// channels, so this is the representative regime, not a fluke). This
    /// bound is ~6x the measured max and ~12x the measured mean --
    /// consistent margin with `ChannelSplitMLPTests`' own tolerance choice
    /// -- generous for run-to-run noise while still well below the
    /// unscaled-weights failure mode (maxAbsDiff in the thousands) that
    /// would flag a real layout/axis/dtype bug.
    private static let tolerance: Float = 0.09375
    private static let meanTolerance: Float = 0.02

    @Test("ANEFusedSplitMLP(aneFraction=0.125) matches all-GPU reference at real Qwen size (S=512)")
    func splitMatchesReferenceAtEighthFraction() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        try #require(ANERuntime.available())

        let hidden = Self.hidden, inter = Self.inter, s = 512
        let weights = Self.makeWeights(hidden: hidden, inter: inter, seed: 100)

        MLXRandom.seed(103)
        let x = MLXRandom.normal([s, hidden]).asType(.bfloat16)
        eval(x)

        let want = Self.referenceForward(
            x,
            gateWq: weights.gateWq, gateScales: weights.gateScales, gateBiases: weights.gateBiases,
            upWq: weights.upWq, upScales: weights.upScales, upBiases: weights.upBiases,
            downWq: weights.downWq, downScales: weights.downScales, downBiases: weights.downBiases)

        let mlp = try ANEFusedSplitMLP(
            gateW: weights.gateWq, gateScales: weights.gateScales, gateBiases: weights.gateBiases,
            upW: weights.upWq, upScales: weights.upScales, upBiases: weights.upBiases,
            downW: weights.downWq, downScales: weights.downScales, downBiases: weights.downBiases,
            hidden: hidden, inter: inter, sequenceLength: s, aneFraction: 0.125)
        let got = try mlp(x)
        eval(got)

        #expect(got.shape == want.shape, "shape mismatch: got \(String(describing: got.shape)), want \(String(describing: want.shape))")
        let absDiff = abs(got.asType(.float32) - want.asType(.float32))
        let maxAbsDiff = absDiff.max().item(Float.self)
        let meanAbsDiff = absDiff.mean().item(Float.self)
        print("ANEFusedSplitMLPTests aneFraction=0.125 S=\(s): maxAbsDiff=\(maxAbsDiff) meanAbsDiff=\(meanAbsDiff)")
        #expect(maxAbsDiff < Self.tolerance, "maxAbsErr=\(maxAbsDiff)")
        #expect(meanAbsDiff < Self.meanTolerance, "meanAbsErr=\(meanAbsDiff)")
    }

    /// `aneFraction=0.0` must take the pure-GPU path (F==0, no ANE
    /// involvement), so the result must be bit-identical to the all-GPU
    /// reference -- both compute the identical `quantizedMM` calls on the
    /// identical weights.
    @Test("ANEFusedSplitMLP(aneFraction=0.0) is identical to all-GPU reference")
    func zeroFractionMatchesReferenceExactly() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        let hidden = Self.hidden, inter = Self.inter, s = 512
        let weights = Self.makeWeights(hidden: hidden, inter: inter, seed: 200)

        MLXRandom.seed(203)
        let x = MLXRandom.normal([s, hidden]).asType(.bfloat16)
        eval(x)

        let want = Self.referenceForward(
            x,
            gateWq: weights.gateWq, gateScales: weights.gateScales, gateBiases: weights.gateBiases,
            upWq: weights.upWq, upScales: weights.upScales, upBiases: weights.upBiases,
            downWq: weights.downWq, downScales: weights.downScales, downBiases: weights.downBiases)

        let mlp = try ANEFusedSplitMLP(
            gateW: weights.gateWq, gateScales: weights.gateScales, gateBiases: weights.gateBiases,
            upW: weights.upWq, upScales: weights.upScales, upBiases: weights.upBiases,
            downW: weights.downWq, downScales: weights.downScales, downBiases: weights.downBiases,
            hidden: hidden, inter: inter, sequenceLength: s, aneFraction: 0.0)
        let got = try mlp(x)
        eval(got)

        let maxAbsDiff = (abs(got.asType(.float32) - want.asType(.float32)).max()).item(Float.self)
        print("ANEFusedSplitMLPTests aneFraction=0.0: maxAbsDiff=\(maxAbsDiff)")
        #expect(maxAbsDiff == 0, "aneFraction=0.0 must be pure-GPU and bit-identical to the reference: maxAbsErr=\(maxAbsDiff)")
    }

    // MARK: - Task D1: concurrent forward == sequential forward

    /// `callAsFunction`'s `0<F<inter` path now overlaps the ANE `predict`
    /// (background thread, MLX-free) with the GPU partial (caller thread)
    /// via `ConcurrentEngines.run`. This proves the overlap introduces no
    /// numeric change vs. running the identical two partials sequentially
    /// (`sequentialCallAsFunctionForTesting`, same math, no
    /// `ConcurrentEngines`) -- same math, just overlapped, so the tolerance
    /// is tight (1e-4), unlike the ANE-vs-GPU-reference tolerance above.
    @Test("ANEFusedSplitMLP concurrent forward matches the sequential forward at real Qwen size (S=512)")
    func concurrentForwardMatchesSequentialForward() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        try #require(ANERuntime.available())

        let hidden = Self.hidden, inter = Self.inter, s = 512
        let weights = Self.makeWeights(hidden: hidden, inter: inter, seed: 300)

        MLXRandom.seed(303)
        let x = MLXRandom.normal([s, hidden]).asType(.bfloat16)
        eval(x)

        let mlp = try ANEFusedSplitMLP(
            gateW: weights.gateWq, gateScales: weights.gateScales, gateBiases: weights.gateBiases,
            upW: weights.upWq, upScales: weights.upScales, upBiases: weights.upBiases,
            downW: weights.downWq, downScales: weights.downScales, downBiases: weights.downBiases,
            hidden: hidden, inter: inter, sequenceLength: s, aneFraction: 0.125)

        let concurrent = try mlp(x)
        eval(concurrent)
        let sequential = try mlp.sequentialCallAsFunctionForTesting(x)
        eval(sequential)

        #expect(concurrent.shape == sequential.shape)
        let diff = abs(concurrent.asType(.float32) - sequential.asType(.float32))
        eval(diff)
        let maxAbsDiff = diff.max().item(Float.self)
        print("ANEFusedSplitMLPTests concurrent-vs-sequential: maxAbsDiff=\(maxAbsDiff)")
        #expect(maxAbsDiff < 1e-4, "concurrent forward diverged from sequential forward: maxAbsErr=\(maxAbsDiff)")
    }
}
