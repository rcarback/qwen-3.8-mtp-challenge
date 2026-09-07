import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM  // `qwen35EpsScalar` is internal; see GatedDeltaScanCostTests

/// Width-1 forward stream: what the host builds per decode step, and the
/// byte-equality receipts for each launch reduction that removes part of it.
@Suite(.serialized)
struct QwenForwardStreamTests {

    /// Characterization, not a goal: compiled decode cannot take this model's
    /// cache shape, and this records why so the question is settled in code.
    ///
    /// `Qwen35TextModel.newCache` hands back a `MambaCache` for each of the 48
    /// linear-attention layers and a `KVCacheSimple` for each of the 16
    /// full-attention layers (Qwen35.swift:5348-5355).
    /// `CompiledDecode.setupCompiledDecode` refuses any layer that is neither
    /// `KVCacheSimple` nor `RotatingKVCache` (CompiledDecode.swift:132-138),
    /// and `eligible` accepts only the promoted `Compilable*` types
    /// (CompiledDecode.swift:50-55). There is no compilable `ArraysCache`.
    @Test("compiled decode rejects the Qwen cache shape")
    func compiledDecodeRejectsTheQwenCacheShape() {
        let qwenShaped: [KVCache] = [MambaCache(), KVCacheSimple()]
        #expect(!CompiledDecode.eligible(qwenShaped))

        // The specific layer that blocks it, isolated from its neighbour.
        let recurrent: any KVCache = MambaCache()
        #expect(!(recurrent is KVCacheSimple))
        #expect(!(recurrent is RotatingKVCache))
        #expect(!CompiledDecode.eligible([recurrent]))

        // The full-attention layer alone is also rejected, because `eligible`
        // wants the PROMOTED type, not the promotable one.
        #expect(!CompiledDecode.eligible([KVCacheSimple()]))
    }

    /// Receipt for Task 2: the compiled g/beta helper equals the eager
    /// expression bit for bit at every width the decode path uses.
    @Test("compiled g and beta equal the eager expression")
    func compiledGBetaIsByteExact() {
        let (trials, bad, firstBad) = qwen35VerifyCompiledGBeta()
        #expect(trials == 40)
        #expect(bad == 0, "first mismatching trial: \(firstBad)")

        let control = qwen35CompiledGBetaNegativeControl()
        #expect(control.gMoved, "the comparison cannot detect a changed g")
        #expect(control.betaHeld, "beta must not depend on a")
    }

    /// Receipt for Task 3: the fused post-norm pair equals the module form
    /// bit for bit, including at width 1 where the module used to run.
    @Test("fused gated post-norm equals the module form")
    func gatedPostNormIsByteExact() {
        let (trials, bad, firstBad) = qwen35VerifyGatedPostNorm()
        #expect(trials == 32)
        #expect(bad == 0, "first mismatching trial: \(firstBad)")
        #expect(
            qwen35GatedPostNormNegativeControl(),
            "the comparison cannot detect a changed gate")
    }

    /// Receipt for Task 4: the epsilon scalar is built once per distinct
    /// value, and the memoized array carries the same value the fresh one did.
    @Test("fused residual norm epsilon scalar is memoized")
    func epsScalarIsMemoized() {
        let eps: Float = 1.0e-6
        let before = qwen35EpsScalarMisses
        let first = qwen35EpsScalar(eps)
        let afterFirst = qwen35EpsScalarMisses
        let second = qwen35EpsScalar(eps)
        let afterSecond = qwen35EpsScalarMisses
        #expect(afterFirst - before == 1)
        #expect(afterSecond - afterFirst == 0)

        eval(first, second)
        #expect(MLX.all(MLX.equal(first, second)).item(Bool.self))
        #expect(MLX.all(MLX.equal(first, MLXArray(eps))).item(Bool.self))

        // A different value is a different entry, not a silent alias.
        let other = qwen35EpsScalar(1.0e-5)
        eval(other)
        #expect(qwen35EpsScalarMisses - afterSecond == 1)
        #expect(!MLX.all(MLX.equal(first, other)).item(Bool.self))
    }

    /// Receipt for Task 5: the packed prework kernel equals the eager chain
    /// bit for bit at widths 1 through 9, and the control proves the
    /// comparison can see the conv-state store the generalization adds.
    @Test("packed gated-delta prework equals the eager chain")
    func packedPreworkIsByteExact() {
        let (trials, bad, firstBad) = qwen35VerifyPackedPrework()
        #expect(trials == 24)
        #expect(bad == 0, "first mismatching trial: \(firstBad)")

        let control = qwen35PackedPreworkNegativeControl()
        #expect(
            control.sensitive,
            "conv-state row 1 must reach the next conv state at width 1")
        #expect(
            control.insensitive,
            "conv-state row 0 must fall out of the window at width 1")
    }
}
