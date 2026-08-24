// Copyright (c) Layr Labs, Inc.
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXFastCore
import MLXLMCommon
import MLXNN

/// A DFlash2 block-diffusion drafter, usable as this track's declared MTP head.
///
/// The published drafter `z-lab/Qwen3.8-27B-DFlash2` is built for this exact
/// backbone: `num_target_layers` 64, vocab 248320, hidden 5120, intermediate
/// 17408. It differs from the pinned native-MTP head in three ways, and the
/// third is the one that shapes this file:
///
/// - depth: 5 sliding-attention layers rather than 1;
/// - conditioning: hidden states from target layers 5, 19, 33, 47 and 61,
///   concatenated into a 25600-wide input, rather than the final hidden state;
/// - proposal: the WHOLE block in one forward, then a selector traces one path
///   through per-position candidates, rather than one token per forward.
///
/// That last property is why a heavier head is affordable here. The session's
/// cost model for the autoregressive head is `T(d) = V + d*H`; a block-parallel
/// head costs `V + H`, flat in depth, so head bytes are paid once per round.
///
/// Ported from the upstream MLX reference (`dflash/model_mlx.py`). Greedy only:
/// this track decodes at temperature 0, so the sampling branches of the
/// reference selector are deliberately absent rather than stubbed.
///
/// MEASURED, upstream Python path, two prose prompts, 160 decode tokens:
/// 1.41x serial at block 4 with an affine 4-bit group-64 head. Accept rate is
/// flat from BF16 down to 4-bit, and the smallest head decodes fastest, so the
/// 1.01 GiB artifact is the optimum rather than a concession to the 2 GiB
/// manifest cap. See `docs/dflash2-head-port-plan.md`.
public struct Qwen38DFlash2Configuration: Sendable, Equatable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var vocabSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var maxPositionEmbeddings: Int
    public var slidingWindow: Int
    public var numTargetLayers: Int
    public var targetLayerIDs: [Int]
    public var blockSize: Int
    public var maskTokenID: Int
    public var convKernelSize: Int
    public var convGroupSize: Int
    public var selectorRank: Int
    public var selectorTopK: Int
    /// The reference sets `is_causal: false` for this drafter, so the proposal
    /// block attends to itself in full and only the context half is masked.
    public var isCausal: Bool

    /// Width of the concatenated target-hidden input `fc` consumes.
    public var contextWidth: Int { targetLayerIDs.count * hiddenSize }

    public init(fromJSON root: [String: Any]) throws {
        func int(_ key: String, in object: [String: Any]) throws -> Int {
            guard let value = object[key] as? Int else {
                throw MLXFastError.invalidInput(
                    "DFlash2 config is missing integer \(key)")
            }
            return value
        }
        guard let dflash = root["dflash_config"] as? [String: Any] else {
            throw MLXFastError.invalidInput(
                "DFlash2 config is missing dflash_config")
        }
        guard let architectures = root["architectures"] as? [String],
              architectures.contains("DFlash2DraftModel")
        else {
            throw MLXFastError.invalidInput(
                "not a DFlash2 drafter: architectures must name DFlash2DraftModel")
        }
        self.hiddenSize = try int("hidden_size", in: root)
        self.numHiddenLayers = try int("num_hidden_layers", in: root)
        self.numAttentionHeads = try int("num_attention_heads", in: root)
        self.numKeyValueHeads = try int("num_key_value_heads", in: root)
        self.headDim = try int("head_dim", in: root)
        self.intermediateSize = try int("intermediate_size", in: root)
        self.vocabSize = try int("vocab_size", in: root)
        self.maxPositionEmbeddings = try int("max_position_embeddings", in: root)
        self.numTargetLayers = try int("num_target_layers", in: root)
        self.slidingWindow = try int("sliding_window", in: root)
        self.rmsNormEps = Float((root["rms_norm_eps"] as? Double) ?? 1e-6)
        let rope = (root["rope_parameters"] as? [String: Any]) ?? [:]
        self.ropeTheta = Float((rope["rope_theta"] as? Double)
            ?? (rope["rope_theta"] as? Int).map(Double.init)
            ?? 10_000.0)
        guard let targetLayerIDs = dflash["target_layer_ids"] as? [Int],
              !targetLayerIDs.isEmpty,
              Set(targetLayerIDs).count == targetLayerIDs.count
        else {
            throw MLXFastError.invalidInput(
                "DFlash2 target_layer_ids must be a non-empty unique list")
        }
        self.targetLayerIDs = targetLayerIDs
        self.blockSize = try int("block_size", in: dflash)
        self.maskTokenID = try int("mask_token_id", in: dflash)
        self.convKernelSize = try int("conv_kernel_size", in: dflash)
        self.convGroupSize = try int("conv_group_size", in: dflash)
        self.selectorRank = try int("selector_rank", in: dflash)
        self.selectorTopK = try int("selector_top_k", in: dflash)
        self.isCausal = (root["is_causal"] as? Bool) ?? false

        guard targetLayerIDs.allSatisfy({ $0 >= 0 && $0 < numTargetLayers }) else {
            throw MLXFastError.invalidInput(
                "DFlash2 target_layer_ids fall outside 0..<\(numTargetLayers)")
        }
        guard hiddenSize % convGroupSize == 0 else {
            throw MLXFastError.invalidInput(
                "DFlash2 hidden_size must divide by conv_group_size")
        }
        guard maskTokenID >= 0, maskTokenID < vocabSize else {
            throw MLXFastError.invalidInput(
                "DFlash2 mask_token_id is outside the vocabulary")
        }
        guard blockSize >= 2, selectorTopK >= 1, selectorRank >= 1 else {
            throw MLXFastError.invalidInput(
                "DFlash2 block_size, selector_rank and selector_top_k must be positive")
        }
    }
}

/// Causal convolution over channel groups whose kernel is produced per
/// position, added to a static base kernel.
///
/// The reference calls the two taps `prepare` and `finish`: `prepare` convolves
/// the block input and also emits the dynamic kernel that `finish` applies to
/// the block output. One `kernel_projection` produces both, which is why its
/// output width is `2 * kernel_size * groups`.
final class Qwen38DFlash2GroupedConv: Module {
    @ModuleInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    let kernelSize: Int
    let groupSize: Int
    let groups: Int

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        self.kernelSize = kernelSize
        self.groupSize = groupSize
        self.groups = hiddenSize / groupSize
        // Two taps, `kernelSize` offsets, one weight per channel.
        _baseKernel.wrappedValue = MLXArray.zeros([2, kernelSize, hiddenSize])
        _kernelProjection.wrappedValue = Linear(
            hiddenSize, 2 * kernelSize * groups, bias: false)
        super.init()
    }

    /// One tap. `dynamic` is `[B, L, kernelSize, groups]`, `base` is
    /// `[kernelSize, hiddenSize]`.
    private func convolve(
        _ hidden: MLXArray, dynamic: MLXArray, base: MLXArray
    ) -> MLXArray {
        let (b, l) = (hidden.dim(0), hidden.dim(1))
        let blocks = hidden.reshaped(b, l, groups, groupSize)
        let dynamic = dynamic.reshaped(b, l, kernelSize, groups, 1)
        var output = MLXArray.zeros(like: blocks)
        for offset in 0 ..< kernelSize {
            // Shift right by `offset` positions, zero-filling the head. The
            // convolution is causal, so position t reads t-offset.
            let values: MLXArray
            if offset == 0 {
                values = blocks
            } else {
                let head = MLXArray.zeros(
                    [b, offset, groups, groupSize], dtype: blocks.dtype)
                values = concatenated(
                    [head, blocks[0..., ..<(l - offset), 0..., 0...]], axis: 1)
            }
            let kernel = base[offset]
                .reshaped(1, 1, groups, groupSize)
                .asType(hidden.dtype)
            output = output + kernel * values
            output = output + dynamic[0..., 0..., offset, 0..., 0...] * values
        }
        return output.reshaped(hidden.shape)
    }

    /// Returns the convolved input and the dynamic kernel `finish` will use.
    func prepare(_ hidden: MLXArray) -> (MLXArray, MLXArray) {
        let (b, l) = (hidden.dim(0), hidden.dim(1))
        let dynamic = kernelProjection(hidden)
            .reshaped(b, l, 2, kernelSize, groups)
        return (
            convolve(
                hidden,
                dynamic: dynamic[0..., 0..., 0, 0..., 0...],
                base: baseKernel[0]),
            dynamic[0..., 0..., 1, 0..., 0...]
        )
    }

    func finish(_ hidden: MLXArray, dynamic: MLXArray) -> MLXArray {
        convolve(hidden, dynamic: dynamic, base: baseKernel[1])
    }
}

final class Qwen38DFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

/// Attention over injected target context plus the proposal block.
///
/// The drafter holds no context KV of its own. Each round it projects the
/// target's hidden states into keys and values, appends the proposal block's
/// own keys and values, and attends across both. That is the KV injection the
/// upstream write-up describes: the drafter never models the context from
/// scratch and spends its capacity on the next block.
final class Qwen38DFlash2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let config: Qwen38DFlash2Configuration
    let scale: Float

    init(_ config: Qwen38DFlash2Configuration) {
        self.config = config
        self.scale = pow(Float(config.headDim), -0.5)
        let qDim = config.numAttentionHeads * config.headDim
        let kvDim = config.numKeyValueHeads * config.headDim
        _qProj.wrappedValue = Linear(config.hiddenSize, qDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        _oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(
            dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(
            dimensions: config.headDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        context: MLXArray,
        rope: RoPE,
        cache: KVCache
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var context = context
        var contextLength = context.dim(1)

        // Every layer is sliding on this drafter. Drop context the window
        // cannot reach and advance the cache so RoPE offsets stay absolute.
        let keepContext = config.slidingWindow - 1
        if contextLength > keepContext {
            let skip = contextLength - keepContext
            context = context[0..., skip..., 0...]
            contextLength = context.dim(1)
            // Advance the cache past the dropped rows so RoPE offsets stay
            // absolute. Only `BaseKVCache` exposes a settable offset; the
            // protocol's is get-only.
            if let cache = cache as? BaseKVCache {
                cache.offset += skip
            }
        }

        let heads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let baseOffset = cache.offset

        var queries = qProj(x)
        var proposalKeys = kProj(x)
        var proposalValues = vProj(x)
        var contextKeys = kProj(context)
        var contextValues = vProj(context)

        queries = qNorm(queries.reshaped(b, l, heads, -1)).transposed(0, 2, 1, 3)
        contextKeys = kNorm(contextKeys.reshaped(b, contextLength, kvHeads, -1))
            .transposed(0, 2, 1, 3)
        contextValues = contextValues.reshaped(b, contextLength, kvHeads, -1)
            .transposed(0, 2, 1, 3)
        proposalKeys = kNorm(proposalKeys.reshaped(b, l, kvHeads, -1))
            .transposed(0, 2, 1, 3)
        proposalValues = proposalValues.reshaped(b, l, kvHeads, -1)
            .transposed(0, 2, 1, 3)

        // The proposal block sits immediately after the injected context.
        queries = rope(queries, offset: baseOffset + contextLength)
        contextKeys = rope(contextKeys, offset: baseOffset)
        proposalKeys = rope(proposalKeys, offset: baseOffset + contextLength)

        let (cachedKeys, cachedValues) = cache.update(
            keys: contextKeys, values: contextValues)
        let cachedLength = cachedKeys.dim(2)
        let keys = concatenated([cachedKeys, proposalKeys], axis: 2)
        let values = concatenated([cachedValues, proposalValues], axis: 2)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .array(blockMask(
                blockLength: l, contextLength: cachedLength))
        ).transposed(0, 2, 1, 3).reshaped(b, l, -1)
        return oProj(output)
    }

    /// Context is visible inside the sliding window; the proposal block is
    /// visible to itself in full unless the drafter declares itself causal.
    ///
    /// This is the mask the reference builds inline. Kept as a named function
    /// because the two halves have different rules and inlining them reads as
    /// one condition when it is two.
    private func blockMask(blockLength l: Int, contextLength: Int) -> MLXArray {
        let query = MLXArray(Int32(contextLength)) + MLXArray(0 ..< l)
            .reshaped(l, 1)
        let key = MLXArray(0 ..< (contextLength + l)).reshaped(1, contextLength + l)
        let inWindow = logicalAnd(
            key .< MLXArray(Int32(contextLength)),
            (query - key) .< MLXArray(Int32(config.slidingWindow)))
        var block = key .>= MLXArray(Int32(contextLength))
        if config.isCausal {
            block = logicalAnd(block, key .<= query)
        }
        return logicalOr(inWindow, block)
    }
}

final class Qwen38DFlash2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen38DFlash2Attention
    @ModuleInfo var mlp: Qwen38DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: Qwen38DFlash2GroupedConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: Qwen38DFlash2GroupedConv

    init(_ config: Qwen38DFlash2Configuration) {
        _selfAttn.wrappedValue = Qwen38DFlash2Attention(config)
        _mlp.wrappedValue = Qwen38DFlash2MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attentionConv.wrappedValue = Qwen38DFlash2GroupedConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.convKernelSize,
            groupSize: config.convGroupSize)
        _mlpConv.wrappedValue = Qwen38DFlash2GroupedConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.convKernelSize,
            groupSize: config.convGroupSize)
        super.init()
    }

    /// Both sublayers are wrapped by a convolution: the input tap runs before
    /// the sublayer and hands its dynamic kernel to the output tap. This is
    /// what keeps a block-parallel draft from decaying toward the end of the
    /// block, where no autoregressive signal is available.
    func callAsFunction(
        _ x: MLXArray,
        context: MLXArray,
        rope: RoPE,
        cache: KVCache
    ) -> MLXArray {
        var residual = x
        var (h, kernel) = attentionConv.prepare(inputLayerNorm(x))
        h = residual + attentionConv.finish(
            selfAttn(h, context: context, rope: rope, cache: cache),
            dynamic: kernel)

        residual = h
        let (m, mlpKernel) = mlpConv.prepare(postAttentionLayerNorm(h))
        return residual + mlpConv.finish(mlp(m), dynamic: mlpKernel)
    }
}

/// Traces one coherent path through per-position candidate sets.
///
/// The block head emits `selectorTopK` plausible tokens at every position
/// independently. Taking each position's argmax would produce an incoherent
/// sequence, because position t+1's best token depends on what position t
/// actually chose. The selector scores each candidate with a unary logit term
/// plus a low-rank bilinear edge term against the chosen predecessor, then
/// walks the block left to right.
final class Qwen38DFlash2CandidateSelector: Module {
    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    let topK: Int

    init(_ config: Qwen38DFlash2Configuration) {
        self.topK = config.selectorTopK
        _predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.selectorRank)
        _successorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.selectorRank)
        _hiddenProjection.wrappedValue = Linear(
            config.hiddenSize, config.selectorRank, bias: false)
        super.init()
    }

    /// Greedy path only. Returns `(path, candidates)` where `path` is
    /// `[B, L]` chosen ids and `candidates` is `[B, L, topK]`.
    func select(
        hidden: MLXArray, logits: MLXArray, anchorIDs: MLXArray
    ) -> (MLXArray, MLXArray) {
        let vocab = logits.dim(-1)
        let candidates = argPartition(logits, kth: vocab - topK, axis: -1)[
            0..., 0..., (vocab - topK)...]
        let unary = takeAlong(logits, candidates, axis: -1)
        let projected = hiddenProjection(hidden)

        var predecessor = anchorIDs
        var path = [MLXArray]()
        path.reserveCapacity(hidden.dim(1))
        for position in 0 ..< hidden.dim(1) {
            let positionCandidates = candidates[0..., position, 0...]
            let edges = sum(
                predecessorCodebook(predecessor)[0..., .newAxis, 0...]
                    * projected[0..., position, .newAxis, 0...]
                    * successorCodebook(positionCandidates),
                axis: -1)
            let scores = unary[0..., position, 0...] + edges
            let selected = argMax(scores, axis: -1)
            predecessor = takeAlong(
                positionCandidates, selected[0..., .newAxis], axis: -1)[0..., 0]
            path.append(predecessor)
        }
        return (stacked(path, axis: 1), candidates)
    }
}

public final class Qwen38DFlash2Head: Module, @unchecked Sendable {
    public let config: Qwen38DFlash2Configuration

    @ModuleInfo(key: "fc") var contextProjection: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [Qwen38DFlash2DecoderLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var candidateSelector:
        Qwen38DFlash2CandidateSelector

    private let rope: RoPE

    /// The shared RoPE, exposed so a parity check can step the layer stack by
    /// hand and localize a mismatch to one layer.
    var ropeForTesting: RoPE { rope }
    private var targetEmbed: ((MLXArray) -> MLXArray)?
    private var targetLMHead: ((MLXArray) -> MLXArray)?

    public init(config: Qwen38DFlash2Configuration) {
        self.config = config
        _contextProjection.wrappedValue = Linear(
            config.contextWidth, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            Qwen38DFlash2DecoderLayer(config)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _candidateSelector.wrappedValue =
            Qwen38DFlash2CandidateSelector(config)
        self.rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeTheta)
        super.init()
    }

    /// The drafter owns no embedding table and no vocabulary projection: it
    /// borrows the target's, exactly as the pinned native head does. That is
    /// also why the published artifact carries neither.
    public func bind(
        embed: @escaping (MLXArray) -> MLXArray,
        lmHead: @escaping (MLXArray) -> MLXArray
    ) {
        self.targetEmbed = embed
        self.targetLMHead = lmHead
    }

    public func unbind() {
        self.targetEmbed = nil
        self.targetLMHead = nil
    }

    public func makeCache() -> [KVCache] {
        (0 ..< config.numHiddenLayers).map { _ in
            RotatingKVCache(maxSize: config.slidingWindow - 1, keep: 0)
        }
    }

    /// `targetHidden` is the concatenation of the target's hidden states at
    /// `config.targetLayerIDs`, shaped `[B, S, contextWidth]`.
    ///
    /// `logitsStart` drops the leading rows of the block before the vocabulary
    /// projection. The caller passes 1: position 0 of the block is the anchor,
    /// a token the target has already committed, so drafting it again would
    /// waste a row and shift the whole proposal by one.
    func hiddenStates(
        inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [KVCache],
        logitsStart: Int = 0
    ) throws -> MLXArray {
        guard let targetEmbed else {
            throw MLXFastError.invalidInput(
                "the DFlash2 head must be bound to a target before drafting")
        }
        guard targetHidden.dim(-1) == config.contextWidth else {
            throw MLXFastError.invalidInput(
                "DFlash2 target hidden width \(targetHidden.dim(-1)) does not "
                    + "match \(config.targetLayerIDs.count) target layers x "
                    + "\(config.hiddenSize)")
        }
        guard logitsStart >= 0, logitsStart < inputs.dim(1) else {
            throw MLXFastError.invalidInput(
                "DFlash2 logitsStart \(logitsStart) is outside the block")
        }
        var h = targetEmbed(inputs)
        let context = hiddenNorm(contextProjection(targetHidden))
        for (layer, layerCache) in zip(layers, cache) {
            h = layer(h, context: context, rope: rope, cache: layerCache)
        }
        if logitsStart > 0 {
            h = h[0..., logitsStart..., 0...]
        }
        return norm(h)
    }

    /// Load a published DFlash2 drafter tree.
    ///
    /// Accepts both the BF16 artifact and an affine-quantized derivative. The
    /// artifact this track ships is affine 4-bit group-64 at 1.01 GiB, chosen
    /// because accept rate is flat from BF16 down to 4-bit while decode speed
    /// is not -- the head is weight-stream-bound, so its bytes are pure cost.
    public static func load(from directory: URL) throws -> Qwen38DFlash2Head {
        let configURL = directory.appendingPathComponent("config.json")
        let raw = try Data(contentsOf: configURL)
        guard let root = try JSONSerialization.jsonObject(with: raw)
            as? [String: Any]
        else {
            throw MLXFastError.invalidInput(
                "DFlash2 config.json must be a JSON object")
        }
        let config = try Qwen38DFlash2Configuration(fromJSON: root)
        let head = Qwen38DFlash2Head(config: config)

        var weights = [String: MLXArray]()
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        for url in contents where url.pathExtension == "safetensors" {
            for (key, value) in try loadArrays(url: url) {
                weights[key] = value
            }
        }
        guard !weights.isEmpty else {
            throw MLXFastError.invalidInput(
                "no safetensors found in \(directory.path)")
        }

        // The selector codebooks ship as bare arrays rather than as an
        // Embedding's `.weight`. Upstream renames them at load for the same
        // reason: the module expects the standard parameter name.
        for name in ["predecessor_codebook", "successor_codebook"] {
            let stored = "candidate_selector.\(name)"
            if let value = weights.removeValue(forKey: stored) {
                weights["\(stored).weight"] = value
            }
        }

        // An affine-quantized tree carries `.scales`, and the module has to be
        // quantized to the same scheme before its parameters will accept them.
        // The scheme is read from the config rather than reverse-engineered
        // from tensor widths: a wrong guess would load silently and draft
        // garbage.
        if weights["fc.scales"] != nil {
            guard let block = root["quantization"] as? [String: Any],
                  let groupSize = block["group_size"] as? Int,
                  let bits = block["bits"] as? Int,
                  (block["mode"] as? String) ?? "affine" == "affine"
            else {
                throw MLXFastError.invalidInput(
                    "the DFlash2 head is quantized but its config.json does "
                        + "not declare an affine quantization block")
            }
            quantize(model: head, groupSize: groupSize, bits: bits)
        }

        try head.update(
            parameters: ModuleParameters.unflattened(weights), verify: [.all])
        // MLX's `eval`: force the lazy graph so load cost is paid here rather
        // than inside the first drafting round. Not code evaluation.
        eval(head)
        return head
    }

    /// Propose a block. `inputs` is the mask-token block whose first entry is
    /// the anchor (the target's last committed token); the returned path is the
    /// drafted continuation, one token shorter than the block.
    public func propose(
        inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [KVCache],
        logitsStart: Int = 1
    ) throws -> MLXArray {
        guard let targetLMHead else {
            throw MLXFastError.invalidInput(
                "the DFlash2 head must be bound to a target before drafting")
        }
        let hidden = try hiddenStates(
            inputs: inputs,
            targetHidden: targetHidden,
            cache: cache,
            logitsStart: logitsStart)
        let (path, _) = candidateSelector.select(
            hidden: hidden,
            logits: targetLMHead(hidden),
            anchorIDs: inputs[0..., 0])
        return path
    }
}
