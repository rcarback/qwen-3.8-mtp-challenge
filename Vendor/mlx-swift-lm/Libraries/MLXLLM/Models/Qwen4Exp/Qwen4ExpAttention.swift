// Qwen3.8-Flash-Next (qwen4_exp) QSA sparse attention. Local fork port;
// reference: mlx-lm PR 1788 (mlx_lm/models/qwen4_exp.py).
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

/// `KVCacheSimple` plus the indexer's raw per-token keys. Rows are valid up to
/// `offset`, so `trim` (which only moves the offset) needs no override; stale
/// rows past the offset are overwritten by the next write.
public final class Qwen4ExpAttnCache: KVCacheSimple {
    var indexerKeys: MLXArray?

    /// Write `k` (`[B, S, dim]`) at rows `[offset ..< offset + S]` and return rows `[0 ..< offset + S]`.
    /// Call BEFORE `update(keys:values:)` for the same tokens.
    func writeIndexerKeys(_ k: MLXArray) -> MLXArray {
        let B = k.dim(0)
        let S = k.dim(1)
        let dim = k.dim(2)
        let needed = offset + S
        if indexerKeys == nil || indexerKeys!.dim(1) < needed {
            let capacity = ((needed + step - 1) / step) * step
            let fresh = MLXArray.zeros([B, capacity, dim], dtype: k.dtype)
            if let old = indexerKeys, offset > 0 {
                fresh[0..., 0 ..< offset, 0...] = old[0..., 0 ..< offset, 0...]
            }
            indexerKeys = fresh
        }
        indexerKeys![0..., offset ..< needed, 0...] = k
        return indexerKeys![0..., 0 ..< needed, 0...]
    }

    public override var state: [MLXArray] {
        get {
            var s = super.state
            if let idx = indexerKeys, offset > 0 {
                s.append(idx[0..., 0 ..< offset, 0...])
            }
            return s
        }
        set {
            var v = newValue
            if v.count == 3 {
                indexerKeys = v.removeLast()
            } else {
                indexerKeys = nil
            }
            super.state = v
        }
    }

    public override func copy() -> any KVCache {
        let new = Qwen4ExpAttnCache()
        new.state = state.map { $0[.ellipsis] }
        new.offset = offset
        return new
    }
}

/// Partial RoPE tables: `cos/sin` for `dims` features from integer positions.
/// Text-only mRoPE with identical positions per section reduces to plain RoPE.
final class Qwen4ExpRotary {
    let dims: Int
    let invFreq: MLXArray

    init(dims: Int, base: Float) {
        self.dims = dims
        let exponents = MLXArray(stride(from: 0, to: dims, by: 2).map { Float($0) }) / Float(dims)
        invFreq = pow(MLXArray(base), -exponents)
    }

    /// `positions` `[B, T]` -> `cos, sin` `[B, T, dims]` float32.
    func cosSin(positions: MLXArray) -> (MLXArray, MLXArray) {
        let freqs = positions.asType(.float32).expandedDimensions(axis: -1) * invFreq
        let emb = concatenated([freqs, freqs], axis: -1)
        return (cos(emb), sin(emb))
    }
}

/// Rotate the first `d = cos.dim(-1)` features of `x`; `cos/sin` broadcast against `x`.
func qwen4ExpApplyPartialRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let d = cos.dim(-1)
    let c = cos.asType(x.dtype)
    let s = sin.asType(x.dtype)
    let xr = x[.ellipsis, 0 ..< d]
    let half = d / 2
    let x1 = xr[.ellipsis, 0 ..< half]
    let x2 = xr[.ellipsis, half ..< d]
    let rot = concatenated([-x2, x1], axis: -1)
    let rotated = xr * c + rot * s
    return x.dim(-1) > d ? concatenated([rotated, x[.ellipsis, d...]], axis: -1) : rotated
}

func qwen4ExpPositions(offset: Int, count: Int) -> MLXArray {
    MLXArray((0 ..< count).map { Int32(offset + $0) }).reshaped(1, count)
}

/// Qwen Sparse Attention indexer: scores mean-pooled key blocks against each
/// query and keeps the best `budget / compress` complete blocks plus the query's
/// own partial block. Returns nil while every visible token fits the budget.
final class Qwen4ExpQSAIndexer: Module {
    let nHeads: Int
    let kvHeads: Int
    let headDim: Int
    let budget: Int
    let compress: Int
    let blockTopK: Int

    @ModuleInfo(key: "index_qk_proj") var qkProj: Linear
    @ModuleInfo(key: "q_layernorm") var qNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_layernorm") var kNorm: Qwen4ExpRMSNorm

    init(_ args: Qwen4ExpTextConfiguration) {
        nHeads = args.indexerNHeads
        kvHeads = args.indexerKVHeads
        headDim = args.indexerHeadDim
        budget = args.indexerBudget
        compress = args.indexerCompressRatio
        blockTopK = budget / compress
        _qkProj.wrappedValue = Linear(args.hiddenSize, (nHeads + kvHeads) * headDim, bias: false)
        _qNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        super.init()
    }

    func keepMask(_ x: MLXArray, rope: Qwen4ExpRotary, cache: Qwen4ExpAttnCache?, offset: Int) -> MLXArray? {
        let B = x.dim(0)
        let S = x.dim(1)
        let qk = qkProj(x)
        let split = nHeads * headDim
        var q = qk[.ellipsis, 0 ..< split].reshaped(B, S, nHeads, headDim)
        var rawK = qk[.ellipsis, split...].reshaped(B, S, headDim)
        if let cache { rawK = cache.writeIndexerKeys(rawK) }
        let kvLen = rawK.dim(1)
        if kvLen <= budget { return nil }

        let nBlocks = kvLen / compress
        var pooled = rawK[0..., 0 ..< (nBlocks * compress), 0...].reshaped(B, nBlocks, compress, headDim)
        pooled = kNorm(pooled.asType(.float32).mean(axis: 2).asType(rawK.dtype))
        let blockStarts = MLXArray((0 ..< nBlocks).map { Int32($0 * compress) }).reshaped(1, nBlocks)
        let (ck, sk) = rope.cosSin(positions: blockStarts)
        pooled = qwen4ExpApplyPartialRope(pooled, cos: ck, sin: sk)  // [B, nBlocks, dim]

        let qPos = qwen4ExpPositions(offset: offset, count: S)  // [1, S]
        let (cq, sq) = rope.cosSin(positions: qPos)
        q = qNorm(q)
        q = qwen4ExpApplyPartialRope(q, cos: cq.expandedDimensions(axis: 2), sin: sq.expandedDimensions(axis: 2))

        // scores[b, s, n] = sum_h relu(q[b,s,h] . pooled[b,n]) / sqrt(dim)
        let qf = q.asType(.float32).reshaped(B, S * nHeads, headDim)
        var scores = matmul(qf, pooled.asType(.float32).transposed(0, 2, 1)).reshaped(B, S, nHeads, nBlocks)
        scores = maximum(scores, MLXArray(Float(0))).sum(axis: 2) / Float(headDim).squareRoot()  // [B, S, nBlocks]

        // only blocks entirely in the query's past are candidates; integer
        // arithmetic stays on the host because MLX `/` promotes ints to float
        let nComplete = MLXArray((0 ..< S).map { Int32((offset + $0 + 1) / compress) }).reshaped(1, S)
        let blockIds = MLXArray((0 ..< nBlocks).map { Int32($0) }).reshaped(1, 1, nBlocks)
        let visible = blockIds .< nComplete.expandedDimensions(axis: -1)  // [1, S, nBlocks]
        scores = MLX.where(visible, scores, MLXArray(-Float.infinity))

        // top-k by threshold: keep every visible block whose score reaches the k-th best.
        // Ties at the threshold keep a superset of the reference's arbitrary pick.
        let k = min(blockTopK, nBlocks)
        let kth = nBlocks - k
        let topIdx = MLX.argPartition(scores, kth: kth, axis: -1)[.ellipsis, kth...]
        let threshold = MLX.takeAlong(scores, topIdx, axis: -1).min(axis: -1, keepDims: true)
        let keepBlock = visible .&& (scores .>= threshold)  // [B, S, nBlocks]
        var keep = repeated(keepBlock, count: compress, axis: -1)  // [B, S, nBlocks*compress]
        let rest = kvLen - nBlocks * compress
        if rest > 0 {
            keep = concatenated([keep, MLXArray.zeros([keep.dim(0), S, rest], dtype: .bool)], axis: -1)
        }

        // the query's own partial block, up to and including itself
        let ownStart = nComplete * Int32(compress)  // [1, S]
        let tokens = MLXArray((0 ..< kvLen).map { Int32($0) }).reshaped(1, 1, kvLen)
        let kvPos = MLXArray(((kvLen - S) ..< kvLen).map { Int32($0) }).reshaped(1, S, 1)
        let own = (tokens .>= ownStart.expandedDimensions(axis: -1)) .&& (tokens .<= kvPos)
        return (keep .|| own).expandedDimensions(axis: 1)  // [B, 1, S, kvLen]
    }
}

/// Full attention with the QSA keep mask. Split in two phases so the ANE dense
/// lane can run `q_proj` (query and output gate) of the next micro-batch while
/// the GPU finishes this one: `qProjection` then `finish`.
final class Qwen4ExpAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "indexer") var indexer: Qwen4ExpQSAIndexer

    private let aneQProj = Qwen4ExpANEProjectionCache(label: "self_attn.q_proj")

    init(_ args: Qwen4ExpTextConfiguration) {
        nHeads = args.attentionHeads
        nKVHeads = args.kvHeads
        headDim = args.headDim
        scale = pow(Float(headDim), -0.5)
        let d = args.hiddenSize
        _qProj.wrappedValue = Linear(d, nHeads * headDim * 2, bias: false)  // query and output gate
        _kProj.wrappedValue = Linear(d, nKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(d, nKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(nHeads * headDim, d, bias: false)
        _qNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _indexer.wrappedValue = Qwen4ExpQSAIndexer(args)
        super.init()
    }

    /// GPU `q_proj`: `[B, S, nHeads * 2 * headDim]` (query then gate per head).
    func qProjection(_ x: MLXArray) -> MLXArray { qProj(x) }

    func aneQProjection(sequenceLength: Int) -> Qwen4ExpANEProjection? {
        guard Qwen4ExpANELane.enabled else { return nil }
        return aneQProj.program(weight: { qProj.weight }, sequenceLength: sequenceLength)
    }

    func finish(
        _ x: MLXArray, qg qgFlat: MLXArray, rope: Qwen4ExpRotary,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: Qwen4ExpAttnCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        let offset = cache?.offset ?? 0
        let sparse = indexer.keepMask(x, rope: rope, cache: cache, offset: offset)

        let qg = MLX.split(qgFlat.reshaped(B, S, nHeads, 2 * headDim), parts: 2, axis: -1)
        let gate = qg[1].reshaped(B, S, nHeads * headDim)
        var q = qNorm(qg[0]).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(B, S, nKVHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, S, nKVHeads, headDim).transposed(0, 2, 1, 3)

        let (c, s) = rope.cosSin(positions: qwen4ExpPositions(offset: offset, count: S))
        q = qwen4ExpApplyPartialRope(q, cos: c.expandedDimensions(axis: 1), sin: s.expandedDimensions(axis: 1))
        k = qwen4ExpApplyPartialRope(k, cos: c.expandedDimensions(axis: 1), sin: s.expandedDimensions(axis: 1))

        var effective = mask
        if let sparse {
            let kvLen = offset + S
            let causal = createCausalMask(n: S, offset: offset)  // [S, kvLen] bool
            let combined: MLXArray
            switch mask {
            case .none, .causal: combined = causal .&& sparse
            case .array(let m): combined = m .&& sparse
            case .arrays(let ms): combined = (ms.first ?? causal) .&& sparse
            }
            precondition(combined.dim(-1) == kvLen)
            effective = .array(combined)
        }
        let out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: effective)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, S, nHeads * headDim) * sigmoid(gate))
    }

    func callAsFunction(
        _ x: MLXArray, rope: Qwen4ExpRotary,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: Qwen4ExpAttnCache?
    ) -> MLXArray {
        finish(x, qg: qProjection(x), rope: rope, mask: mask, cache: cache)
    }
}
