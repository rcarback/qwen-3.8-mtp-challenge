// Qwen3.8-Flash-Next (qwen4_exp) decoder layer, text model, and outer model.
// Local fork port; reference: mlx-lm PR 1788 (mlx_lm/models/qwen4_exp.py).
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

final class Qwen4ExpDecoderLayer: Module {
    let isLinear: Bool
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen4ExpAttention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen4ExpGatedDeltaNet?
    @ModuleInfo(key: "mlp") var mlp: Qwen4ExpSparseMoeBlock
    @ModuleInfo(key: "ple") var ple: Qwen4ExpPLELayer?
    @ModuleInfo(key: "attn_hyper_connection") var attnHC: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var mlpHC: Qwen4ExpGatedResidual

    init(_ args: Qwen4ExpTextConfiguration, layerIdx: Int) {
        isLinear = args.isLinear(layer: layerIdx)
        if isLinear {
            _linearAttn.wrappedValue = Qwen4ExpGatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen4ExpAttention(args)
        }
        _mlp.wrappedValue = Qwen4ExpSparseMoeBlock(args)
        _ple.wrappedValue = args.pleLayerIndices.contains(layerIdx) ? Qwen4ExpPLELayer(args) : nil
        _attnHC.wrappedValue = Qwen4ExpGatedResidual(args, combine: true)
        _mlpHC.wrappedValue = Qwen4ExpGatedResidual(args, combine: true)
        super.init()
    }

    func callAsFunction(
        _ hIn: MLXArray, rope: Qwen4ExpRotary, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?, cache: KVCache?, ids: MLXArray, prevContext: MLXArray?
    ) -> MLXArray {
        var h = hIn
        if let ple, let prevContext {
            h = h + ple(hidden: h, ids: ids, prevContext: prevContext, cache: cache as? ArraysCache)
        }
        let (x1, inject1) = attnHC.mix(h)
        let branch1: MLXArray
        if let linearAttn {
            branch1 = linearAttn(x1, mask: ssmMask, cache: cache as? ArraysCache)
        } else {
            branch1 = selfAttn!(x1, rope: rope, mask: mask, cache: cache as? Qwen4ExpAttnCache)
        }
        h = attnHC.combine(h, branch: branch1, inject: inject1!)
        let (x2, inject2) = mlpHC.mix(h)
        return mlpHC.combine(h, branch: mlp(x2), inject: inject2!)
    }
}

public final class Qwen4ExpTextModel: Module {
    let args: Qwen4ExpTextConfiguration
    let rope: Qwen4ExpRotary
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var mixer: Qwen4ExpGatedResidual

    init(_ args: Qwen4ExpTextConfiguration) {
        self.args = args
        rope = Qwen4ExpRotary(dims: args.rotaryDims, base: args.ropeTheta)
        _embedTokens.wrappedValue = Embedding(embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        _layers.wrappedValue = (0 ..< args.hiddenLayers).map { Qwen4ExpDecoderLayer(args, layerIdx: $0) }
        _mixer.wrappedValue = Qwen4ExpGatedResidual(args, combine: false)
        super.init()
    }

    /// Returns the mixer output (`[B, S, hidden]`, the final "norm") and the
    /// wide residual before the mixer (`[B, S, hcDim]`, the MTP head's input).
    func forward(_ ids: MLXArray, cache: [KVCache]?) -> (hidden: MLXArray, wide: MLXArray) {
        var h = tiled(embedTokens(ids), repetitions: [1, 1, args.hcCount])
        let caches: [KVCache?] = cache.map { $0.map { Optional($0) } } ?? Array(repeating: nil, count: layers.count)
        let firstAttn = layers.firstIndex { !$0.isLinear }
        let attnCache: KVCache? = firstAttn.flatMap { caches[$0] }
        let mask = createAttentionMask(h: h, cache: attnCache.map { [$0] }, returnArray: false)

        // n-gram context: the last (ngramSize - 1) ids before this call, EOS-padded at the start.
        var prevContext: MLXArray? = nil
        if let pleIdx = args.pleLayerIndices.first {
            let ctxLen = args.ngramSize - 1
            let pc = caches[pleIdx] as? ArraysCache
            let prev =
                pc?[3]
                ?? broadcast(
                    MLXArray(Int32(args.eosTokenId)).reshaped(1, 1), to: [ids.dim(0), ctxLen])
            prevContext = prev
            if let pc {
                let history = concatenated([prev, ids.asType(.int32)], axis: 1)
                pc[3] = history[0..., (history.dim(1) - ctxLen)...]
            }
        }
        for (i, layer) in layers.enumerated() {
            h = layer(
                h, rope: rope, mask: mask, ssmMask: nil, cache: caches[i], ids: ids, prevContext: prevContext)
        }
        return (mixer.mix(h).mixed, h)
    }
}

public class Qwen4ExpModel: Module, LLMModel, KVCacheDimensionProvider {
    public let configuration: Qwen4ExpConfiguration
    public var kvHeads: [Int]
    @ModuleInfo(key: "model") var model: Qwen4ExpTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    /// Wide residual from the most recent forward (MTP head input; Task 13).
    public var lastWideResidual: MLXArray?

    public var loraLayers: [Module] { model.layers }

    /// Token embedding (the MTP head borrows it). Public for the serve conformance.
    public func embed(_ ids: MLXArray) -> MLXArray { model.embedTokens(ids) }

    /// The vocabulary projection applied to post-mixer hidden rows.
    public func projectToVocab(_ x: MLXArray) -> MLXArray { lmHead(x) }

    /// Collapse a wide residual `[.., hcDim]` with the final mixer (the model's "norm").
    public func collapseWide(_ x: MLXArray) -> MLXArray { model.mixer.mix(x).mixed }

    public init(_ configuration: Qwen4ExpConfiguration) {
        self.configuration = configuration
        let t = configuration.textConfig
        kvHeads = t.layerTypes.map { $0 == "full_attention" ? t.kvHeads : 0 }
        _model.wrappedValue = Qwen4ExpTextModel(t)
        _lmHead.wrappedValue = Linear(t.hiddenSize, t.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let (hidden, wide) = model.forward(inputs, cache: cache)
        lastWideResidual = wide
        return lmHead(hidden)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let t = configuration.textConfig
        return (0 ..< t.hiddenLayers).map { i -> KVCache in
            if !t.isLinear(layer: i) { return Qwen4ExpAttnCache() }
            return t.pleLayerIndices.contains(i) ? ArraysCache(size: 4) : MambaCache()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        let keepMTP = configuration.textConfig.mtpNumHiddenLayers > 0
        for (rawKey, value) in weights {
            var key = rawKey
            if key.hasPrefix("model.visual.") || key.hasPrefix("visual.") || key.hasPrefix("vision_tower.") {
                continue
            }
            if key.hasPrefix("mtp.") || key.hasPrefix("model.mtp.") {
                if !keepMTP { continue }
                if key.hasPrefix("model.mtp.") { key = String(key.dropFirst("model.".count)) }
            }
            if key.hasPrefix("model.language_model.") {
                key = "model." + key.dropFirst("model.language_model.".count)
            } else if key.hasPrefix("language_model.") {
                key = "model." + key.dropFirst("language_model.".count)
            }

            if key.hasSuffix("mlp.experts.gate_up_proj") {
                let base = String(key.dropLast("experts.gate_up_proj".count))
                let mid = value.dim(-2) / 2
                out[base + "switch_mlp.gate_proj.weight"] = value[.ellipsis, 0 ..< mid, 0...]
                out[base + "switch_mlp.up_proj.weight"] = value[.ellipsis, mid..., 0...]
                continue
            }
            if key.hasSuffix("mlp.experts.down_proj") {
                out[String(key.dropLast("experts.down_proj".count)) + "switch_mlp.down_proj.weight"] = value
                continue
            }
            var v = value
            // torch (C,1,K) -> mlx (C,K,1); idempotent on an already converted weight
            if key.hasSuffix("conv1d.weight"), v.ndim == 3, v.dim(1) == 1 {
                v = v.transposed(0, 2, 1)
            }
            out[key] = v
        }
        return out
    }
}
