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

    /// Phase 0: PLE injection (at the PLE layer) and the attention-side read gate.
    /// Returns the updated wide residual, the mixed input, and the write gate.
    func preMix(_ hIn: MLXArray, ids: MLXArray, prevContext: MLXArray?, cache: KVCache?)
        -> (h: MLXArray, x: MLXArray, inject: MLXArray)
    {
        var h = hIn
        if let ple, let prevContext {
            h = h + ple(hidden: h, ids: ids, prevContext: prevContext, cache: cache as? ArraysCache)
        }
        let (x, inject) = attnHC.mix(h)
        return (h, x, inject!)
    }

    /// Phase 1 on the GPU: the projection the ANE lane can also produce
    /// (`[in_proj_qkv; in_proj_z]` for GDN layers, `q_proj` for attention layers).
    func gpuProjection(_ x: MLXArray) -> MLXArray {
        if let linearAttn {
            let (mixedQKV, z) = linearAttn.inProjection(x)
            return concatenated([mixedQKV, z], axis: -1)
        }
        return selfAttn!.qProjection(x)
    }

    /// The ANE program for phase 1 at `sequenceLength`, if the lane is on and the build succeeded.
    func aneProjection(sequenceLength: Int) -> Qwen4ExpANEProjection? {
        if let linearAttn { return linearAttn.aneInProjection(sequenceLength: sequenceLength) }
        return selfAttn!.aneQProjection(sequenceLength: sequenceLength)
    }

    /// Phase 2: the rest of the layer from a phase-1 projection.
    func finish(
        _ h: MLXArray, x: MLXArray, inject: MLXArray, projection: MLXArray,
        rope: Qwen4ExpRotary, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let branch: MLXArray
        if let linearAttn {
            let (mixedQKV, z) = linearAttn.splitFused(projection)
            branch = linearAttn.finish(x, mixedQKV: mixedQKV, z: z, mask: nil, cache: cache as? ArraysCache)
        } else {
            branch = selfAttn!.finish(x, qg: projection, rope: rope, mask: mask, cache: cache as? Qwen4ExpAttnCache)
        }
        let h1 = attnHC.combine(h, branch: branch, inject: inject)
        let (x2, inject2) = mlpHC.mix(h1)
        return mlpHC.combine(h1, branch: mlp(x2), inject: inject2!)
    }

    func callAsFunction(
        _ hIn: MLXArray, rope: Qwen4ExpRotary, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?, cache: KVCache?, ids: MLXArray, prevContext: MLXArray?
    ) -> MLXArray {
        let (h, x, inject) = preMix(hIn, ids: ids, prevContext: prevContext, cache: cache)
        let branch: MLXArray
        if let linearAttn {
            branch = linearAttn(x, mask: ssmMask, cache: cache as? ArraysCache)
        } else {
            branch = selfAttn!(x, rope: rope, mask: mask, cache: cache as? Qwen4ExpAttnCache)
        }
        let h1 = attnHC.combine(h, branch: branch, inject: inject)
        let (x2, inject2) = mlpHC.mix(h1)
        return mlpHC.combine(h1, branch: mlp(x2), inject: inject2!)
    }
}

public final class Qwen4ExpTextModel: Module {
    let args: Qwen4ExpTextConfiguration
    let rope: Qwen4ExpRotary
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var mixer: Qwen4ExpGatedResidual

    /// Test and diagnostic hook: run the micro-batched prefill loop at this
    /// length with GPU projections (no ANE), so the pipeline structure is
    /// exercised without the lane. Tests assign it directly; the environment
    /// variable is the out-of-process hook, which is what lets the
    /// micro-batching cost be measured apart from the ANE lane's own cost.
    nonisolated(unsafe) static var forcedMicroBatch: Int? =
        Int(ProcessInfo.processInfo.environment["MLX_QWEN4EXP_FORCE_MICROBATCH"] ?? "")

    init(_ args: Qwen4ExpTextConfiguration) {
        self.args = args
        rope = Qwen4ExpRotary(dims: args.rotaryDims, base: args.ropeTheta)
        _embedTokens.wrappedValue = Embedding(embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        _layers.wrappedValue = (0 ..< args.hiddenLayers).map { Qwen4ExpDecoderLayer(args, layerIdx: $0) }
        _mixer.wrappedValue = Qwen4ExpGatedResidual(args, combine: false)
        super.init()
    }

    /// The n-gram context for `ids` (the last `ngramSize - 1` ids before this
    /// call, EOS-padded at the start) and the context the NEXT call starts from.
    private func nextPrevContext(ids: MLXArray, pleCache pc: ArraysCache?, prev: MLXArray?)
        -> (context: MLXArray, next: MLXArray)
    {
        let ctxLen = args.ngramSize - 1
        let context =
            prev ?? pc?[3]
            ?? broadcast(MLXArray(Int32(args.eosTokenId)).reshaped(1, 1), to: [ids.dim(0), ctxLen])
        let history = concatenated([context, ids.asType(.int32)], axis: 1)
        return (context, history[0..., (history.dim(1) - ctxLen)...])
    }

    /// Returns the mixer output (`[B, S, hidden]`, the final "norm") and the
    /// wide residual before the mixer (`[B, S, hcDim]`, the MTP head's input).
    func forward(_ ids: MLXArray, cache: [KVCache]?) -> (hidden: MLXArray, wide: MLXArray) {
        let S = ids.dim(1)
        if ids.dim(0) == 1 {
            if let forced = Self.forcedMicroBatch, S >= 2 * forced {
                return forwardMicroBatched(ids, cache: cache, microBatch: forced, useANE: false)
            }
            if Qwen4ExpANEFused.microBatchEnabled, Qwen4ExpANELane.armed(sequenceLength: S) {
                return forwardMicroBatched(
                    ids, cache: cache, microBatch: Qwen4ExpANELane.microBatch, useANE: true)
            }
        }
        // Every other mode, including `split`, `shared` and `both`, runs the
        // plain forward below unchanged. The fused lanes join inside the
        // modules, so the graph shape here is the same one the pure GPU path
        // builds.
        var h = tiled(embedTokens(ids), repetitions: [1, 1, args.hcCount])
        let caches: [KVCache?] = cache.map { $0.map { Optional($0) } } ?? Array(repeating: nil, count: layers.count)
        let firstAttn = layers.firstIndex { !$0.isLinear }
        let attnCache: KVCache? = firstAttn.flatMap { caches[$0] }
        let mask = createAttentionMask(h: h, cache: attnCache.map { [$0] }, returnArray: false)

        var prevContext: MLXArray? = nil
        if let pleIdx = args.pleLayerIndices.first {
            let pc = caches[pleIdx] as? ArraysCache
            let (context, next) = nextPrevContext(ids: ids, pleCache: pc, prev: nil)
            prevContext = context
            pc?[3] = next
        }
        for (i, layer) in layers.enumerated() {
            h = layer(
                h, rope: rope, mask: mask, ssmMask: nil, cache: caches[i], ids: ids, prevContext: prevContext)
        }
        return (mixer.mix(h).mixed, h)
    }

    /// Prefill as micro-batches of `microBatch` tokens (plus a shorter tail),
    /// layer-major. Inside each layer the phase-1 projection of the NEXT
    /// micro-batch runs on the ANE while the GPU finishes the current one; a
    /// micro-batch the ANE cannot serve (no program, or the tail) projects on
    /// the GPU. Recurrent and attention state advance in token order, so the
    /// numerics differ from the plain path only by the fp16 ANE projection.
    func forwardMicroBatched(_ ids: MLXArray, cache: [KVCache]?, microBatch m: Int, useANE: Bool)
        -> (hidden: MLXArray, wide: MLXArray)
    {
        let S = ids.dim(1)
        let caches: [KVCache?] = cache.map { $0.map { Optional($0) } } ?? Array(repeating: nil, count: layers.count)
        var bounds = [(Int, Int)]()
        var start = 0
        while start < S {
            let end = min(start + m, S)
            bounds.append((start, end))
            start = end
        }
        let n = bounds.count
        let segIds = bounds.map { ids[0..., $0.0 ..< $0.1] }
        var hs = segIds.map { tiled(embedTokens($0), repetitions: [1, 1, args.hcCount]) }

        var contexts = [MLXArray?](repeating: nil, count: n)
        if let pleIdx = args.pleLayerIndices.first {
            let pc = caches[pleIdx] as? ArraysCache
            var prev: MLXArray? = nil
            for i in 0 ..< n {
                let (context, next) = nextPrevContext(ids: segIds[i], pleCache: pc, prev: prev)
                contexts[i] = context
                prev = next
            }
            pc?[3] = prev
        }

        for (li, layer) in layers.enumerated() {
            let c = caches[li]
            var mixes = [(x: MLXArray, inject: MLXArray)]()
            for i in 0 ..< n {
                let (h, x, inject) = layer.preMix(hs[i], ids: segIds[i], prevContext: contexts[i], cache: c)
                hs[i] = h
                mixes.append((x, inject))
            }
            let program = useANE ? layer.aneProjection(sequenceLength: m) : nil
            func aneServes(_ i: Int) -> Bool { program != nil && bounds[i].1 - bounds[i].0 == m }
            func readANE(_ p: Qwen4ExpANEProjection, _ prepared: ANEDirectDispatch.Prepared, dtype: DType)
                -> MLXArray
            {
                p.readOutput(prepared).asType(dtype).reshaped(1, m, -1)
            }

            var projections = [MLXArray?](repeating: nil, count: n)
            if aneServes(0), let p = program, let prepared = try? p.makeInput(mixes[0].x.reshaped(m, -1)),
                (try? p.predict(prepared)) != nil
            {
                projections[0] = readANE(p, prepared, dtype: hs[0].dtype)
            } else {
                projections[0] = layer.gpuProjection(mixes[0].x)
            }

            for i in 0 ..< n {
                let next = i + 1
                // Staging (makeInput) must stay on the calling thread: it builds
                // its own transpose/contiguous/cast graph and calls MLX eval on
                // it internally (ANEDirectDispatch.prepare), so it cannot run
                // inside the ane closure alongside the gpu closure's MLX graph
                // construction. Only predict, which touches no MLX state, is
                // safe to overlap.
                var nextPrepared: ANEDirectDispatch.Prepared? = nil
                if next < n, aneServes(next), let p = program {
                    nextPrepared = try? p.makeInput(mixes[next].x.reshaped(m, -1))
                }
                let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                    layer.isLinear
                    ? .none : createAttentionMask(h: mixes[i].x, cache: c.map { [$0] }, returnArray: false)
                var out: MLXArray? = nil
                var aneFailed = false
                do {
                    let (_, o) = try ConcurrentEngines.run(
                        ane: {
                            if let np = nextPrepared, let p = program { try p.predict(np) }
                        },
                        gpu: { () -> MLXArray in
                            layer.finish(
                                hs[i], x: mixes[i].x, inject: mixes[i].inject, projection: projections[i]!,
                                rope: rope, mask: mask, cache: c)
                        })
                    out = o
                } catch {
                    aneFailed = true
                    if Qwen4ExpANELane.log {
                        fputs("[qwen4exp-ane] predict failed at layer \(li): \(error)\n", stderr)
                    }
                }
                if out == nil {
                    out = layer.finish(
                        hs[i], x: mixes[i].x, inject: mixes[i].inject, projection: projections[i]!,
                        rope: rope, mask: mask, cache: c)
                }
                hs[i] = out!
                if next < n {
                    if !aneFailed, let np = nextPrepared, let p = program {
                        projections[next] = readANE(p, np, dtype: hs[i].dtype)
                    } else {
                        projections[next] = layer.gpuProjection(mixes[next].x)
                    }
                }
            }
            // ONE barrier per layer, not one per micro-batch. The old code ran
            // eval() inside the gpu closure, forcing 48 x n hard syncs for a
            // prefill that the plain path evaluates in a single graph.
            eval(hs)
        }
        let wide = concatenated(hs, axis: 1)
        return (mixer.mix(wide).mixed, wide)
    }
}

public class Qwen4ExpModel: Module, LLMModel, KVCacheDimensionProvider {
    public let configuration: Qwen4ExpConfiguration
    public var kvHeads: [Int]
    @ModuleInfo(key: "model") var model: Qwen4ExpTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    /// The native multi-token-prediction head, present when the runtime config
    /// declares `mtp_num_hidden_layers > 0` and the `mtp.*` tensors load.
    @ModuleInfo(key: "mtp") public var mtp: Qwen4ExpMTPHead?
    /// Wide residual from the most recent forward (the MTP head's input).
    public var lastWideResidual: MLXArray?

    /// Width of the hyper-connection residual (`hc_count * hidden_size`).
    public var wideWidth: Int { configuration.textConfig.hcDim }

    /// One MTP draft step on wide residual rows and the embeddings of the tokens
    /// that follow them. Returns collapsed rows for the vocabulary projection and
    /// the head's post-block wide residual for the next step.
    public func mtpStep(wide: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]) -> (hidden: MLXArray, wide: MLXArray) {
        guard let mtp else { fatalError("Qwen4ExpModel has no MTP head attached") }
        return mtp.forward(wide: wide, tokenEmbedding: embed(nextTokenIds), cache: cache)
    }

    public func makeMTPHeadCache() -> [KVCache] { mtp?.newCache() ?? [] }

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
        _mtp.wrappedValue = t.mtpNumHiddenLayers > 0 ? Qwen4ExpMTPHead(t) : nil
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
