// Copyright © 2026 Eigen Labs.
//
// Port of omlx commit 696d90a:
//   patches/mlx_lm_mtp/qwen35_model.py  (MTPDecoderLayer, MTPModule)
//   patches/mlx_lm_mtp/__init__.py        (is_mtp_active / set_mtp_active)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Module-level MTP flag

/// Controls whether Qwen3.5/3.6 model inits attach the MTP head.
/// Set to `true` before calling `MLXLLM.load(...)` when MTP should be active.
/// Mirrors omlx `is_mtp_active()` / `set_mtp_active()` from
/// patches/mlx_lm_mtp/__init__.py.
public nonisolated(unsafe) var _qwen35MTPEnabled: Bool = false

/// A proposal head that is not the checkpoint's own MTP module.
///
/// The track's 2026-08-14 contract makes the head weights competitive surface,
/// and a declared head need not share the native head's architecture. This
/// protocol is deliberately empty: the backbone's only interest in such a head
/// is that one is attached, so that `hasMTPHead` reports the truth. Everything
/// about how it drafts belongs to the session that drives it.
public protocol Qwen35ProposalHead: AnyObject {}

/// A declared proposal head, hidden from the backbone's parameter walk.
///
/// THE BOX IS LOAD-BEARING, not decoration. `Module` discovers its children by
/// reflecting over stored properties and keeping every value that IS a
/// `Module`, whether or not it carries `@ModuleInfo`. A declared head stored
/// bare on `Qwen35TextModel` therefore joins the backbone's own parameter tree,
/// and the loader's `update(parameters:verify: [.all])` then demands checkpoint
/// keys for a head whose weights live in a different tree entirely -- the
/// failure reads `keyNotFound(["externalProposalHead", ...])`. Reflection does
/// not descend into a value that is not itself a `Module`, so wrapping the head
/// in this box keeps it out of the walk while leaving it reachable.
public final class Qwen35ProposalHeadBox {
    public let head: any Qwen35ProposalHead
    public init(_ head: any Qwen35ProposalHead) { self.head = head }
}

/// The declared proposal head to attach at the next model init, or nil for the
/// checkpoint's own MTP module.
///
/// Same idiom and same lifetime discipline as `_qwen35MTPEnabled` above: the
/// model factory builds the model, so an attachment decision that has to be
/// made before `init` runs has nowhere else to live. Set it around the load and
/// clear it afterwards.
public nonisolated(unsafe) var _qwen35ExternalProposalHead:
    Qwen35ProposalHeadBox?

/// E85 arm gate. `MLX_E85_FUSED_EMBED=0` restores the eager
/// `embedTokens(ids)` before the dual-norm concat.
///
/// The `MLX_` prefix is load-bearing: the trusted worker's environment
/// sanitizer drops `MLXFAST_*`, so an `MLXFAST_`-spelled gate would never
/// reach the process that runs the scored round, and both arms of an A/B
/// would silently measure the same code.
let qwen35FusedEmbedConcatEnabled: Bool =
    ProcessInfo.processInfo.environment["MLX_E85_FUSED_EMBED"] != "0"

/// Proposal-only derived quantization of the MTP head, in bits.
///
/// Unset resolves to `4`. `MLX_QWEN_MTP_HEAD_QUANT` accepts `8` to select
/// 8-bit instead, or `0` (or any other value) to disable the derivation and
/// leave the head bfloat16 exactly as loaded. The 4-bit default follows the
/// 2026-09-07 campaign (`docs/perf/clean-branch-2026-09-07.md`): 4 bits
/// matched or beat 8 bits on every prompt with all tokens matched, and both
/// beat the bfloat16 head. The derivation only applies to a head whose
/// projection is not already quantized. The `MLX_` prefix is required for
/// the same reason as the gate above.
///
/// WHAT THIS CHANGES AND WHAT IT CANNOT. It changes which tokens the head
/// PROPOSES. It cannot change which tokens are EMITTED: the target verify
/// decides every one of those, and the accept walk is untouched. So the whole
/// effect of a bad choice here is a lower accept rate, which the agreement
/// harness measures directly.
let qwen35HeadProposalQuantizationBits: Int? = {
    guard let raw = ProcessInfo.processInfo
        .environment["MLX_QWEN_MTP_HEAD_QUANT"]
    else { return 4 }
    guard let bits = Int(raw), bits == 4 || bits == 8 else { return nil }
    return bits
}()

// MARK: - MTPDecoderLayer

/// Full-attention transformer layer used inside the Qwen3.5/3.6 MTP head.
/// Unlike `Qwen35DecoderLayer`, this always uses full attention (never SSM/linear).
/// MoE config is honoured when `num_experts > 0`.
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPDecoderLayer
final class Qwen35MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration) {
        _selfAttn.wrappedValue = Qwen35Attention(args)
        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            // Same fused gate/up MLP as the backbone layers; here the linears
            // stay bf16 and the fuse takes the plain-weight path. Head side —
            // proposal-only, no exactness constraint.
            _mlp.wrappedValue = Qwen35FusedMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        // omlx: MTPDecoderLayer.__call__
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        // The backbone's decoder layer has fused this residual+norm boundary
        // since `qwen35FusedResidualRMSNorm` landed; the head layer was left on
        // the eager pair. Same kernel, same bf16/5120 guard, same
        // bf16-round-before-square argument, so the values are bit-identical to
        // `h = x + r; postAttentionLayerNorm(h)` — one launch and one host graph
        // node instead of two, paid once per PROPOSED token (draftCount times a
        // round) rather than once per layer.
        if x.dtype == .bfloat16, r.dtype == .bfloat16, x.dim(-1) == 5120 {
            let (h, postAttnNorm) = qwen35FusedResidualRMSNorm(
                x: x, r: r,
                weight: postAttentionLayerNorm.weight,
                eps: postAttentionLayerNorm.eps)
            return h + (mlp as! UnaryLayer)(postAttnNorm)
        }
        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }

    /// Populate this layer's K/V history without computing a dead layer
    /// output. Only valid when no later MTP layer consumes that output.
    func appendHistoryKV(_ x: MLXArray, cache: any KVCache) {
        selfAttn.appendHistoryKV(inputLayerNorm(x), cache: cache)
    }
}

// MARK: - MTPModule

/// Multi-Token Prediction head for Qwen3.5/3.6.
///
/// Fuses the backbone's pre-norm hidden state at position t with the embedding of
/// the sampled main token (t+1) to predict the draft token at (t+2).
///
/// Architecture (port of PR #990):
/// ```
/// pre_fc_norm_hidden:    RMSNorm(hidden_size)
/// pre_fc_norm_embedding: RMSNorm(hidden_size)
/// fc:                    Linear(hidden_size * 2 → hidden_size, bias: false)
/// layers:                [MTPDecoderLayer]  × mtp_num_hidden_layers
/// norm:                  RMSNorm(hidden_size)
/// ```
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPModule
final class Qwen35MTPModule: Module {
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    // `layers` uses the default ModuleInfo key derived from the property name.
    let layers: [Qwen35MTPDecoderLayer]
    let norm: RMSNorm

    /// Derived quantized twin of `fc`, built at first forward when the gate is
    /// on. Deliberately not `@ModuleInfo`: it is derived at runtime, not
    /// checkpoint state, and joining the parameter walk would make the loader
    /// demand checkpoint keys for it.
    private var _quantizedFC: QuantizedLinear?
    private var _proposalQuantizationApplied = false

    init(_ args: Qwen35TextConfiguration) {
        _preFcNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFcNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        self.layers = (0 ..< args.mtpNumHiddenLayers).map { _ in
            Qwen35MTPDecoderLayer(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    /// Dual RMSNorm written straight into the `[e | h]` layout `fc` consumes.
    /// Same arithmetic as `qwen35DualRMSNorm` + `concatenated([e, h], -1)`;
    /// the extra concat launch is gone. Proposal-only.
    ///
    /// The embedding table is affine 4-bit group-64, so `embedTokens(ids)` is
    /// three gathers plus a dequantize. Those four intermediates exist only to
    /// carry one row into a kernel that reads it twice, so the fused variant
    /// reads the packed row in place and the eager embed never runs.
    private func preFcConcat(
        nextTokenIds: MLXArray, embedTokens: Embedding, hidden: MLXArray
    ) -> MLXArray {
        if qwen35FusedEmbedConcatEnabled,
           let quantized = embedTokens as? QuantizedEmbedding,
           quantized.mode == .affine, quantized.bits == 4,
           quantized.groupSize == 64,
           let zeroPoints = quantized.biases,
           hidden.dtype == .bfloat16, hidden.dim(-1) == 5120,
           quantized.weight.dtype == .uint32,
           quantized.weight.dim(1) * 8 == hidden.dim(-1),
           quantized.scales.dtype == .bfloat16,
           quantized.scales.dim(1) * 64 == hidden.dim(-1),
           zeroPoints.dtype == .bfloat16,
           zeroPoints.shape == quantized.scales.shape,
           nextTokenIds.dtype == .int32,
           nextTokenIds.ndim == 2, nextTokenIds.dim(0) == 1,
           nextTokenIds.strides.last == 1,
           nextTokenIds.dim(1) * hidden.dim(-1) == hidden.size,
           preFcNormEmbedding.eps == preFcNormHidden.eps
        {
            return qwen35EmbedDualRMSNormConcat(
                ids: nextTokenIds,
                embedWeight: quantized.weight,
                embedScales: quantized.scales,
                embedBiases: zeroPoints,
                b: hidden,
                aWeight: preFcNormEmbedding.weight,
                bWeight: preFcNormHidden.weight,
                eps: preFcNormEmbedding.eps)
        }

        let embeds = embedTokens(nextTokenIds)
        if embeds.dtype == .bfloat16, hidden.dtype == .bfloat16,
           embeds.dim(-1) == 5120, hidden.dim(-1) == 5120,
           embeds.shape == hidden.shape,
           preFcNormEmbedding.eps == preFcNormHidden.eps
        {
            return qwen35DualRMSNormConcat(
                a: embeds, b: hidden,
                aWeight: preFcNormEmbedding.weight,
                bWeight: preFcNormHidden.weight,
                eps: preFcNormEmbedding.eps)
        }
        return concatenated(
            [preFcNormEmbedding(embeds), preFcNormHidden(hidden)], axis: -1)
    }

    /// Install the derived proposal-only quantization once, if the gate is set.
    private func applyProposalQuantizationIfNeeded() {
        guard !_proposalQuantizationApplied else { return }
        _proposalQuantizationApplied = true
        guard let bits = qwen35HeadProposalQuantizationBits else { return }
        if !(fc is QuantizedLinear) {
            _quantizedFC = QuantizedLinear(
                fc, groupSize: 64, bits: bits, mode: .affine)
        }
        for layer in layers {
            (layer.mlp as? Qwen35FusedMLP)?.proposalQuantizationBits = bits
        }
    }

    /// `fc`, through the derived twin when one exists.
    private func applyFC(_ x: MLXArray) -> MLXArray {
        if let quantized = _quantizedFC { return quantized(x) }
        return fc(x)
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // omlx: MTPModule.__call__
        // 1. Embed next-token ids and fuse with normed hidden state.
        applyProposalQuantizationIfNeeded()
        var fused = applyFC(
            preFcConcat(
                nextTokenIds: nextTokenIds, embedTokens: embedTokens,
                hidden: hidden))

        // 2. Compute attention mask from the first cache entry (or nil if empty).
        let firstCache: (any KVCache)? = cache.first
        let mask = createAttentionMask(h: fused, cache: firstCache)

        // 3. Run each MTPDecoderLayer.
        for (i, layer) in layers.enumerated() {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            fused = layer(fused, mask: mask, cache: c)
        }

        // 4. Return pre-lm_head hidden (norm applied; lm_head is in TextModel).
        return norm(fused)
    }

    /// Run one proposal flush while omitting leading-row outputs that have no
    /// consumer. Every supplied row still participates in the fusion stage and
    /// contributes K/V state; only the final row needs a full decoder output.
    /// Multi-layer heads fail closed before mutating cache state.
    func lastHiddenWithKVOnlyHistory(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray? {
        guard layers.count == 1, cache.count == 1,
              hidden.dim(1) > 1,
              nextTokenIds.dim(1) == hidden.dim(1)
        else { return nil }

        applyProposalQuantizationIfNeeded()
        let fused = applyFC(
            preFcConcat(
                nextTokenIds: nextTokenIds, embedTokens: embedTokens,
                hidden: hidden))
        let historyCount = fused.dim(1) - 1

        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        return norm(layers[0](current, mask: mask, cache: cache[0]))
    }

}
