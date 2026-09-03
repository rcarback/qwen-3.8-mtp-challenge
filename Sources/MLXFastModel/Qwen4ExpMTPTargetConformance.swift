// LOCAL FORK ONLY. Qwen3.8-Flash-Next (qwen4_exp) served through the MTP
// block session.
//
// The "hidden" currency on this model is the WIDE hyper-connection residual
// (`hc_count * hidden_size`): `callWithHidden` returns it, the native head
// consumes and re-emits it (llama.cpp PR 27836 `graph_mtp`), and the session
// chains draft steps on it. The trunk's final mixer is the only "norm", so
// `applyFinalNorm` is the identity and the lm-head helpers collapse a wide row
// with the mixer before projecting.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

extension Qwen4ExpModel: Qwen36MTPTarget {
    public var hasMTPHead: Bool { mtp != nil }

    public var decoderLayerCount: Int { configuration.textConfig.hiddenLayers }

    public func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        let logits = self(input.tokens, cache: cache)
        return (logits, lastWideResidual!)
    }

    public func callWithHiddenNormedAndLayers(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int, layerIDs: [Int]
    ) -> Qwen35ForwardOutput {
        precondition(layerIDs.isEmpty, "Qwen4ExpModel publishes no intermediate layer outputs")
        let (logits, wide) = callWithHidden(input: input, cache: cache, nConfirmed: nConfirmed)
        return Qwen35ForwardOutput(logits: logits, hidden: wide, normed: nil, layerHidden: nil)
    }

    public func installKVRotation(enabled: Bool, seed: UInt64) {
        // No quantized KV cache on this model; nothing to rotate.
    }

    public func applyEmbedding(_ ids: MLXArray) -> MLXArray {
        embed(ids)
    }

    public var externalProposalHead: (any Qwen35ProposalHead)? { nil }

    public func replayRecurrentPrefix(cache: [any KVCache], committedRows: Int) -> Bool {
        false
    }

    public func mtpForwardWithHidden(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> (MLXArray, MLXArray) {
        let (collapsed, wide) = mtpStep(wide: hidden, nextTokenIds: nextTokenIds, cache: cache)
        return (projectToVocab(collapsed), wide)
    }

    public func mtpHeadHiddenForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray {
        mtpStep(wide: hidden, nextTokenIds: nextTokenIds, cache: cache).wide
    }

    public func mtpHeadLastHiddenWithKVOnlyHistory(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray? {
        nil
    }

    /// Wide rows are collapsed by the mixer first; already-collapsed rows project directly.
    private func vocabLogits(_ x: MLXArray) -> MLXArray {
        x.dim(-1) == wideWidth ? projectToVocab(collapseWide(x)) : projectToVocab(x)
    }

    public func applyLMHead(_ x: MLXArray) -> MLXArray {
        vocabLogits(x)
    }

    public func applyDraftLMHead(_ x: MLXArray) -> MLXArray {
        vocabLogits(x)
    }

    public func mapDraftTokenIds(_ ids: MLXArray) -> MLXArray {
        ids
    }

    public func draftTokenID(_ x: MLXArray) -> MLXArray {
        argMax(vocabLogits(x), axis: -1).asType(.int32).reshaped(1, 1)
    }

    public func makeMTPCache() -> [any KVCache] {
        makeMTPHeadCache()
    }

    /// Identity: the head normalises the wide residual itself (`pre_fc_norm_hidden`),
    /// and the trunk's mixer is applied inside `vocabLogits` where a projection needs it.
    public func applyFinalNorm(_ x: MLXArray) -> MLXArray {
        x
    }
}
