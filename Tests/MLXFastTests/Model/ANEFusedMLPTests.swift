import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXFastModel

/// Task B (ANE IOSurface / procedure-bank plan): proves the fused
/// SwiGLU-down MLP -- gate-conv, up-conv, silu, mul, down-conv as ONE ANE
/// program -- runs on the Neural Engine and matches an fp32 MLX reference.
/// See `.superpowers/sdd/2026-08-31-ane-iosurface-procedure-bank/task-B-brief.md`.
@Suite(.serialized)
struct ANEFusedMLPTests {
    /// Builds random gate/up/down weights, runs `ANEFusedMLP`, and checks
    /// the result against the fp32 MLX reference
    /// `down @ (silu(gate @ xT) * (up @ xT))` within tolerance. The task
    /// brief's starting bound is `(maxAbs<1.0, meanAbs<0.05)`; measured at
    /// `hidden=512, F=256` this settles at `maxAbs~0.02, meanAbs~0.004`
    /// (five chained fp16 ops -- two convs, silu, mul, one more conv --
    /// vs. `ANEDirectDispatchTests`' single conv), so the gate here is
    /// tightened to `(0.1, 0.01)`: still 2-5x headroom over the measured
    /// error for run-to-run noise, but tight enough to catch a real
    /// regression instead of only a gross wrong-op bug.
    /// Weights are scaled by `1/sqrt(fan_in)` (standard init scale, not
    /// unit-variance `N(0,1)`) so the three chained matmuls (gate/up over
    /// `hidden`, down over `F`) keep activations near O(1) instead of
    /// compounding into O(sqrt(hidden)*sqrt(F)) magnitudes -- at raw
    /// `N(0,1)` weights this reference's own values run into the
    /// thousands, at which point an absolute tolerance of `(1.0, 0.05)` is
    /// meaningless even though the *relative* error is the same fp16
    /// accumulation noise. Real MLP weights are scaled this way too, so
    /// this is the representative regime, not a loosened test.
    private func assertMatchesReference(hidden: Int, F: Int, S: Int) throws {
        try #require(ANERuntime.available())
        let gate = (MLXRandom.normal([F, hidden]) / Float(hidden).squareRoot()).asType(.float16)
        let up = (MLXRandom.normal([F, hidden]) / Float(hidden).squareRoot()).asType(.float16)
        let down = (MLXRandom.normal([hidden, F]) / Float(F).squareRoot()).asType(.float16)
        let x = MLXRandom.normal([S, hidden]).asType(.float16)
        eval(gate, up, down, x)

        let mlp = try ANEFusedMLP(hidden: hidden, innerFraction: F, sequenceLength: S, gate: gate, up: up, down: down)
        let y = try mlp(x)
        #expect(y.shape == [S, hidden])

        let xf = x.asType(.float32)
        let g = matmul(xf, gate.asType(.float32).transposed(1, 0)) // [S,F]
        let u = matmul(xf, up.asType(.float32).transposed(1, 0)) // [S,F]
        let act = silu(g) * u // [S,F]
        let expected = matmul(act, down.asType(.float32).transposed(1, 0)) // [S,hidden]
        eval(expected)

        let diff = MLX.abs(y.asType(.float32) - expected)
        eval(diff)
        let maxAbs = diff.max().item(Float.self)
        let meanAbs = diff.mean().item(Float.self)
        #expect(maxAbs < 0.1, "maxAbs=\(maxAbs)")
        #expect(meanAbs < 0.01, "meanAbs=\(meanAbs)")
    }

    @Test("fused SwiGLU-down matches reference at S=32 (the shape that compiles)")
    func fusedMatchesReferenceS32() throws {
        try assertMatchesReference(hidden: 512, F: 256, S: 32)
    }

    @Test("fused SwiGLU-down matches reference at S=1 (decode width)")
    func fusedMatchesReferenceS1() throws {
        try assertMatchesReference(hidden: 512, F: 256, S: 1)
    }
}
