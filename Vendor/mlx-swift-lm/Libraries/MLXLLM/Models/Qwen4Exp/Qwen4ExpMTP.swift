// Qwen3.8-Flash-Next (qwen4_exp) native multi-token-prediction head. Local
// fork port; the math follows llama.cpp PR 27836 (`graph_mtp` in
// src/models/qwen4exp.cpp): the head folds the next token's embedding into the
// trunk's wide hyper-connection residual, runs one trunk-style block over it,
// exports the post-block wide residual for the next draft step, and collapses
// with its own mixer before the trunk's lm_head.
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

public final class Qwen4ExpMTPHead: Module {
    let hc: Int
    let d: Int
    let rope: Qwen4ExpRotary

    @ModuleInfo(key: "fc_embedding") var fcEmbedding: Linear
    @ModuleInfo(key: "fc_hidden") var fcHidden: Linear
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: Qwen4ExpRMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: Qwen4ExpRMSNorm
    @ModuleInfo(key: "layers") var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var mixer: Qwen4ExpGatedResidual

    init(_ args: Qwen4ExpTextConfiguration) {
        hc = args.hcCount
        d = args.hiddenSize
        rope = Qwen4ExpRotary(dims: args.rotaryDims, base: args.ropeTheta)
        _fcEmbedding.wrappedValue = Linear(d, d, bias: false)
        _fcHidden.wrappedValue = Linear(d, d, bias: false)
        _preFcNormEmbedding.wrappedValue = Qwen4ExpRMSNorm(dimensions: d, eps: args.rmsNormEps)
        _preFcNormHidden.wrappedValue = Qwen4ExpRMSNorm(dimensions: hc * d, groupSize: d, eps: args.rmsNormEps)
        // One full-attention block; the checkpoint's MTP layer has no PLE and its
        // attention runs dense (the indexer weights load but the mask is ignored
        // below the budget, which a draft window never exceeds).
        var headArgs = args
        headArgs.hiddenLayers = args.mtpNumHiddenLayers
        headArgs.layerTypes = Array(repeating: "full_attention", count: args.mtpNumHiddenLayers)
        headArgs.pleLayerIds = []
        _layers.wrappedValue = (0 ..< args.mtpNumHiddenLayers).map { Qwen4ExpDecoderLayer(headArgs, layerIdx: $0) }
        _mixer.wrappedValue = Qwen4ExpGatedResidual(args, combine: false)
        super.init()
    }

    /// One draft step.
    /// - `wide`: the trunk's pre-mixer wide residual for these positions (`[B, S, hcDim]`),
    ///   or the head's own exported residual from the previous draft step.
    /// - `tokenEmbedding`: the embedding of the token that follows each position (`[B, S, hidden]`).
    /// Returns the collapsed hidden rows (`[B, S, hidden]`, feed to `lm_head`) and the
    /// post-block wide residual (`[B, S, hcDim]`, the next step's `wide`).
    func forward(wide: MLXArray, tokenEmbedding: MLXArray, cache: [KVCache]) -> (hidden: MLXArray, wide: MLXArray) {
        let B = wide.dim(0)
        let S = wide.dim(1)
        let hNorm = preFcNormHidden(wide).reshaped(B, S, hc, d)
        let eNorm = preFcNormEmbedding(tokenEmbedding).expandedDimensions(axis: 2)  // [B, S, 1, d]
        // fc_embedding @ e + fc_hidden @ h, per stream; the embedding is shared across streams.
        var h = (fcEmbedding(eNorm) + fcHidden(hNorm)).reshaped(B, S, hc * d)
        let ids = MLXArray.zeros([B, S], dtype: .int32)  // no PLE in the head: ids are unused
        for (i, layer) in layers.enumerated() {
            let mask = createAttentionMask(h: h, cache: [cache[i]], returnArray: false)
            h = layer(h, rope: rope, mask: mask, ssmMask: nil, cache: cache[i], ids: ids, prevContext: nil)
        }
        return (mixer.mix(h).mixed, h)
    }

    func newCache() -> [KVCache] {
        layers.map { _ in Qwen4ExpAttnCache() }
    }
}
