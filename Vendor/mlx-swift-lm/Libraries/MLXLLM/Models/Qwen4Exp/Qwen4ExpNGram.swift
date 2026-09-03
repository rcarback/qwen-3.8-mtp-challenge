// Qwen3.8-Flash-Next (qwen4_exp) n-gram hashing, memory-mapped table store,
// and the per-layer embedding (PLE) block. Local fork port; references:
// mlx-lm PR 1788 (mlx_lm/models/qwen4_exp.py) and llama.cpp PR 27742.
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

/// Process-wide knobs the factory cannot pass to a model initialiser.
public enum Qwen4ExpRuntime {
    /// Root of the transformed weights tree. The n-gram table store opens
    /// `<root>/<ngram_table.directory>` on first use. Set by the worker and by
    /// the generate verb before `LLMModelFactory.load`.
    nonisolated(unsafe) public static var weightsDirectory: URL?
}

/// Host-side n-gram hashing. Mirrors the reference exactly: 64-bit wrap-around
/// products, XOR mix, Python-semantics modulo. Pure Swift, no MLX.
struct Qwen4ExpNGramHasher {
    let multipliers: [Int64]
    let sizes: [Int64]
    let offsets: [Int64]
    let eos: Int64
    let ngramSize: Int
    let headsPerNgram: Int

    static func pythonMod(_ a: Int64, _ m: Int64) -> Int64 {
        let r = a % m
        return (r != 0 && ((r < 0) != (m < 0))) ? r + m : r
    }

    /// `tokens[0] * m[0] ^ tokens[1] * m[1] ^ ...` with int64 wrap-around.
    static func mix(_ tokens: [Int64], _ m: [Int64]) -> Int64 {
        var acc = tokens[0] &* m[0]
        for p in 1 ..< tokens.count { acc ^= tokens[p] &* m[p] }
        return acc
    }

    /// `history` = (ngramSize - 1) context tokens followed by the new tokens.
    /// Returns, per NEW token, the global row id of every hash head.
    func gids(history: [Int64]) -> [[Int64]] {
        let ctx = ngramSize - 1
        let T = history.count
        // shifted[s][t] = token s positions back, EOS if that crosses an EOS boundary.
        var shifted = [[Int64]](repeating: [Int64](repeating: eos, count: T), count: ngramSize)
        var lastEOS = -1
        for t in 0 ..< T {
            let inSegment = t - (lastEOS + 1)
            for s in 0 ..< ngramSize {
                let src = t - s
                shifted[s][t] = (s == 0 || (inSegment >= s && src >= 0)) ? history[src] : eos
            }
            if history[t] == eos { lastEOS = t }
        }
        var rows = [[Int64]]()
        rows.reserveCapacity(max(0, T - ctx))
        for t in ctx ..< T {
            var row = [Int64]()
            for ngram in 2 ... ngramSize {
                let lo = (ngram - 2) * headsPerNgram
                let toks = (0 ..< ngram).map { shifted[$0][t] }
                let mixed = Self.mix(toks, multipliers)
                for h in lo ..< (lo + headsPerNgram) {
                    row.append(Self.pythonMod(mixed, sizes[h]) + offsets[h])
                }
            }
            rows.append(row)
        }
        return rows
    }
}

enum Qwen4ExpNGramError: Error, CustomStringConvertible {
    case badShard(String)
    case noRoot
    var description: String {
        switch self {
        case .badShard(let p):
            return "qwen4exp n-gram shard is not a single bf16 [rows, dim] tensor named weight: \(p)"
        case .noRoot:
            return "qwen4exp: Qwen4ExpRuntime.weightsDirectory is unset; the n-gram table cannot be opened"
        }
    }
}

/// Memory-mapped n-gram embedding table: `shards` files of `[rowsPerShard, dim]`
/// bf16 rows. The 97.7 GiB table is never loaded as MLX arrays; rows are
/// copied out of the mapped files and the page cache owns residency.
final class Qwen4ExpNGramTable {
    private let maps: [Data]  // one mapped file per shard
    private let dataOffsets: [Int]  // byte offset of the tensor payload in each file
    let rowsPerShard: Int
    let dim: Int
    let bytesPerRow: Int

    init(directory: URL, spec: Qwen4ExpNGramTableSpec) throws {
        precondition(spec.dtype == "bfloat16", "n-gram table must be bfloat16")
        rowsPerShard = spec.rowsPerShard
        dim = spec.dim
        bytesPerRow = dim * 2
        var maps = [Data]()
        var offsets = [Int]()
        for s in 0 ..< spec.shards {
            let url = directory.appendingPathComponent(String(format: "shard_%03d.safetensors", s))
            let data = try Data(contentsOf: url, options: [.alwaysMapped])
            guard data.count >= 8 else { throw Qwen4ExpNGramError.badShard(url.path) }
            let headerLength = Int(
                data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian })
            let header =
                try JSONSerialization.jsonObject(with: data.subdata(in: 8 ..< (8 + headerLength)))
                as? [String: Any]
            guard let info = header?["weight"] as? [String: Any],
                let dtype = info["dtype"] as? String, dtype == "BF16",
                let shape = info["shape"] as? [Int], shape == [spec.rowsPerShard, spec.dim],
                let range = info["data_offsets"] as? [Int], range.count == 2
            else { throw Qwen4ExpNGramError.badShard(url.path) }
            maps.append(data)
            offsets.append(8 + headerLength + range[0])
        }
        self.maps = maps
        self.dataOffsets = offsets
    }

    /// `gids[t]` lists one global row id per head. Returns `[T, heads * dim]` float16.
    func gather(_ gids: [[Int64]]) -> MLXArray {
        let T = gids.count
        let heads = gids.first?.count ?? 0
        var out = [Float16](repeating: 0, count: T * heads * dim)
        out.withUnsafeMutableBufferPointer { dst in
            var cursor = 0
            for row in gids {
                for gid in row {
                    let shard = Int(gid) / rowsPerShard
                    let r = Int(gid) % rowsPerShard
                    let start = dataOffsets[shard] + r * bytesPerRow
                    maps[shard].withUnsafeBytes { src in
                        let base = src.baseAddress!.advanced(by: start)
                        for c in 0 ..< dim {
                            // bf16 -> f32 is a 16-bit shift; f32 -> f16 rounds.
                            let bits = UInt32(base.loadUnaligned(fromByteOffset: c * 2, as: UInt16.self)) << 16
                            dst[cursor + c] = Float16(Float(bitPattern: bits))
                        }
                    }
                    cursor += dim
                }
            }
        }
        return MLXArray(out, [T, heads * dim])
    }
}

/// Holds the checkpoint's hash constants as parameters so `update(parameters:)`
/// fills them; the hasher and the table are built lazily from their values.
final class Qwen4ExpNGramEmbedding: Module {
    @ParameterInfo(key: "layer_multipliers") var layerMultipliers: MLXArray
    @ParameterInfo(key: "ngram_heads_vocab_sizes") var headVocabSizes: MLXArray
    @ParameterInfo(key: "ngram_heads_offsets") var headOffsets: MLXArray
    let args: Qwen4ExpTextConfiguration
    private var hasher: Qwen4ExpNGramHasher?
    private var table: Qwen4ExpNGramTable?

    init(_ args: Qwen4ExpTextConfiguration) {
        self.args = args
        _layerMultipliers.wrappedValue = MLXArray.zeros([args.ngramSize], dtype: .int64)
        _headVocabSizes.wrappedValue = MLXArray.zeros([args.ngramHeads], dtype: .int64)
        _headOffsets.wrappedValue = MLXArray.zeros([args.ngramHeads], dtype: .int64)
        super.init()
    }

    private func resolve() -> (Qwen4ExpNGramHasher, Qwen4ExpNGramTable) {
        if let hasher, let table { return (hasher, table) }
        let h = Qwen4ExpNGramHasher(
            multipliers: layerMultipliers.asArray(Int64.self),
            sizes: headVocabSizes.asArray(Int64.self),
            offsets: headOffsets.asArray(Int64.self),
            eos: Int64(args.eosTokenId),
            ngramSize: args.ngramSize,
            headsPerNgram: args.headsPerNgram)
        precondition(
            h.multipliers.allSatisfy { $0 != 0 },
            "n-gram multipliers were not loaded from the checkpoint")
        guard let spec = args.ngramTable else {
            fatalError("qwen4exp config.json has no ngram_table block")
        }
        guard let root = Qwen4ExpRuntime.weightsDirectory else {
            fatalError(Qwen4ExpNGramError.noRoot.description)
        }
        let t: Qwen4ExpNGramTable
        do {
            t = try Qwen4ExpNGramTable(directory: root.appendingPathComponent(spec.directory), spec: spec)
        } catch {
            fatalError("\(error)")
        }
        hasher = h
        table = t
        return (h, t)
    }

    /// `ids` `[B, S]`, `prevContext` `[B, ngramSize-1]` -> `[B, S, pleEmbedDim]` float16.
    func callAsFunction(ids: MLXArray, prevContext: MLXArray) -> MLXArray {
        let (h, t) = resolve()
        let B = ids.dim(0)
        let S = ids.dim(1)
        let idRows = ids.asType(.int64).asArray(Int64.self)
        let ctxRows = prevContext.asType(.int64).asArray(Int64.self)
        let ctx = args.ngramSize - 1
        var batches = [MLXArray]()
        for b in 0 ..< B {
            let history =
                Array(ctxRows[(b * ctx) ..< ((b + 1) * ctx)]) + Array(idRows[(b * S) ..< ((b + 1) * S)])
            batches.append(t.gather(h.gids(history: history)))
        }
        return stacked(batches, axis: 0)
    }
}

/// Per-layer embedding block at the PLE layer: gates the n-gram value by its
/// agreement with the wide residual, then adds a dilated depthwise short conv.
final class Qwen4ExpPLELayer: Module {
    let d: Int
    let hc: Int
    let dilation: Int
    let stateLen: Int

    @ModuleInfo(key: "ple_embedding") var embedding: Qwen4ExpNGramEmbedding
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "norm_key") var normKey: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_query") var normQuery: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_conv") var normConv: Qwen4ExpRMSNorm
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d

    init(_ args: Qwen4ExpTextConfiguration) {
        d = args.hiddenSize
        hc = args.hcCount
        dilation = args.ngramSize
        let k = args.pleConvKernelSize
        stateLen = (k - 1) * dilation
        let hcDim = d * hc
        _embedding.wrappedValue = Qwen4ExpNGramEmbedding(args)
        _keyProj.wrappedValue = Linear(args.pleEmbedDim, hcDim, bias: false)
        _valueProj.wrappedValue = Linear(args.pleEmbedDim, d, bias: false)
        _normKey.wrappedValue = Qwen4ExpRMSNorm(dimensions: hcDim, groupSize: d, eps: args.rmsNormEps)
        _normQuery.wrappedValue = Qwen4ExpRMSNorm(dimensions: hcDim, groupSize: d, eps: args.rmsNormEps)
        _normConv.wrappedValue = Qwen4ExpRMSNorm(dimensions: hcDim, groupSize: d, eps: args.rmsNormEps)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: hcDim, outputChannels: hcDim, kernelSize: k,
            stride: 1, padding: 0, dilation: dilation, groups: hcDim, bias: false)
        super.init()
    }

    private func shortConv(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let B = x.dim(0)
        let state = cache?[2] ?? MLXArray.zeros([B, stateLen, x.dim(-1)], dtype: x.dtype)
        let full = concatenated([state, x], axis: 1)
        if let cache { cache[2] = full[0..., (full.dim(1) - stateLen)..., 0...] }
        return silu(conv1d(full))
    }

    func callAsFunction(hidden: MLXArray, ids: MLXArray, prevContext: MLXArray, cache: ArraysCache?)
        -> MLXArray
    {
        let emb = embedding(ids: ids, prevContext: prevContext).asType(hidden.dtype)
        let lead = Array(hidden.shape.dropLast())
        let key = normKey(keyProj(emb)).reshaped(lead + [hc, d])
        let value = valueProj(emb)
        let query = normQuery(hidden).reshaped(lead + [hc, d])
        var gate = (key * query).sum(axis: -1, keepDims: true) / Float(d).squareRoot()
        gate = sqrt(maximum(abs(gate), 1e-6)) * sign(gate)
        let gated = (sigmoid(gate) * value.expandedDimensions(axis: -2)).reshaped(lead + [hc * d])
        return gated + shortConv(normConv(gated), cache: cache)
    }
}
