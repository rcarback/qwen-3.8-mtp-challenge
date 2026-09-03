// LOCAL FORK ONLY. Qwen3.8-Flash-Next (qwen4_exp) served through the MTP
// block session. Until the native MTP head lands (plan Task 13) the model
// reports `hasMTPHead == false` and the session runs headless; every
// head-side member below is unreachable in that mode and traps if called.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

extension Qwen4ExpModel: Qwen36MTPTarget {
    public var hasMTPHead: Bool { false }

    public var decoderLayerCount: Int { configuration.textConfig.hiddenLayers }

    /// Backbone forward returning `(logits, pre-mixer wide residual)`. The wide
    /// residual is the head's input, so it plays the "pre-norm hidden" role.
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
        fatalError("Qwen4ExpModel has no MTP head attached")
    }

    public func mtpHeadHiddenForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray {
        fatalError("Qwen4ExpModel has no MTP head attached")
    }

    public func mtpHeadLastHiddenWithKVOnlyHistory(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray? {
        nil
    }

    public func applyLMHead(_ x: MLXArray) -> MLXArray {
        projectToVocab(x)
    }

    public func applyDraftLMHead(_ x: MLXArray) -> MLXArray {
        projectToVocab(x)
    }

    public func mapDraftTokenIds(_ ids: MLXArray) -> MLXArray {
        ids
    }

    public func draftTokenID(_ x: MLXArray) -> MLXArray {
        argMax(projectToVocab(x), axis: -1).asType(.int32).reshaped(1, 1)
    }

    public func makeMTPCache() -> [any KVCache] {
        []
    }

    /// The wide residual is collapsed by the mixer, which doubles as the final
    /// norm; applying the mixer here reconciles the pre-mixer hidden with the
    /// post-norm rows the session expects.
    public func applyFinalNorm(_ x: MLXArray) -> MLXArray {
        collapseWide(x)
    }
}
