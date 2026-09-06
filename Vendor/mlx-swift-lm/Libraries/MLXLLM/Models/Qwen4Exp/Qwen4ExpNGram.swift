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

/// E2M1 magnitudes by 3-bit magnitude code, with a sign bit above them.
/// Duplicated from `NGramTableQuantize`: the converter lives in
/// MLXFastTransform and this in MLXLLM, and a constant table is cheaper to
/// mirror than a module dependency.
let qwen4ExpE2M1: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

/// Decodes one OCP FP8 E4M3 byte: bias 7, no infinities, 0x7F and 0xFF NaN.
func qwen4ExpE4M3ToFloat(_ b: UInt8) -> Float {
    let sign: Float = (b & 0x80) != 0 ? -1 : 1
    let exp = Int((b >> 3) & 0x0F)
    let man = Int(b & 0x07)
    if exp == 0 { return sign * Float(man) * 0.001953125 }
    if exp == 15 && man == 7 { return .nan }
    return sign * (1 + Float(man) / 8) * exp2(Float(exp - 7))
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

    /// Counters for one gather window. The n-gram path had no instrumentation
    /// at all, so its cost was inferred from the table's 102.4 GB size rather
    /// than measured. Only 3.58 MB is gathered per 700-token forward, and the
    /// gather runs once (PLE is at layer 2 only, of 48), so the size is a poor
    /// proxy and this counts the quantities that actually vary.
    struct Stats {
        var calls = 0
        var rows = 0
        var nanos: UInt64 = 0
        var distinctPages = 0
    }

    /// Off unless `MLX_QWEN4EXP_NGRAM_STATS=1`. Page accounting walks a Set per
    /// gather, which is real work, so it must not run on an unmeasured path.
    static let statsEnabled: Bool =
        ProcessInfo.processInfo.environment["MLX_QWEN4EXP_NGRAM_STATS"] == "1"

    final class StatsBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value = Stats()
        func reset() { lock.lock(); value = Stats(); lock.unlock() }
        func snapshot() -> Stats { lock.lock(); defer { lock.unlock() }; return value }
        func record(rows: Int, nanos: UInt64, pages: Int) {
            lock.lock()
            value.calls += 1
            value.rows += rows
            value.nanos += nanos
            value.distinctPages += pages
            lock.unlock()
        }
    }
    static let stats = StatsBox()

    /// Bits per element: 16 (bf16, the on-disk default), 8, or 4 (offline
    /// per-row affine codes from `NGramTableQuantize`). A quantized row is one
    /// `[scale f16][bias f16][codes]` record, so `bytesPerRow` describes every
    /// encoding and no separate scale or bias regions exist.
    private let bits: Int

    /// Sentinel width for NVFP4: four bits per value like int4, but spent as
    /// E2M1 floats under a per-16 E4M3 block scale.
    static let nvfp4Bits = 40

    /// Batched readahead before the gather reads anything. On by default;
    /// set MLX_QWEN4EXP_NGRAM_PREFETCH=0 to measure without it.
    static let prefetchEnabled: Bool =
        ProcessInfo.processInfo.environment["MLX_QWEN4EXP_NGRAM_PREFETCH"] != "0"

    /// Issue the readahead two layers early, on a background thread, rather
    /// than at the top of the gather. Set MLX_QWEN4EXP_NGRAM_AHEAD=0 to
    /// measure without it.
    static let aheadEnabled: Bool =
        ProcessInfo.processInfo.environment["MLX_QWEN4EXP_NGRAM_AHEAD"] != "0"

    /// Issues the readahead on a background thread and returns at once.
    ///
    /// The synchronous `prefetch` runs at the top of the gather, so the read
    /// loop still waits on the first faults. The gids are known before layer 0
    /// and the PLE does not run until layer 2, so issuing there gives the
    /// faults two layers of unrelated compute to resolve under. At prefill
    /// widths a layer is roughly 148 ms of a 7100 ms forward, so two layers is
    /// about 296 ms of cover against a gather that costs about 858 ms.
    func prefetchAhead(_ gids: [[Int64]]) {
        guard Self.prefetchEnabled, Self.aheadEnabled else { return }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.prefetch(gids)
        }
    }

    /// Tells the kernel which pages the gather is about to read, in ascending
    /// file order, before reading any of them.
    ///
    /// The gather reads thousands of rows scattered across 320 million. No two
    /// rows share a 16 KiB page, so a naive read takes one cold fault per row
    /// and each fault stalls the thread until the storage returns. Every gid is
    /// known before the first read, so the faults do not have to be serial:
    /// one madvise(MADV_WILLNEED) per coalesced run lets the kernel issue the
    /// reads together and the loop then walks pages that are already arriving.
    private func prefetch(_ gids: [[Int64]]) {
        guard Self.prefetchEnabled, !gids.isEmpty else { return }
        let pageSize = 16384
        var perShard = [Int: [Int]]()
        for row in gids {
            for gid in row {
                let shard = Int(gid) / rowsPerShard
                let r = Int(gid) % rowsPerShard
                perShard[shard, default: []].append(dataOffsets[shard] + r * bytesPerRow)
            }
        }
        for (shard, offsets) in perShard {
            let sorted = offsets.sorted()
            maps[shard].withUnsafeBytes { src in
                guard let base = src.baseAddress else { return }
                var i = 0
                while i < sorted.count {
                    let lo = (sorted[i] / pageSize) * pageSize
                    var hi = ((sorted[i] + bytesPerRow + pageSize - 1) / pageSize) * pageSize
                    var j = i + 1
                    // Merge runs that already touch the same or the next page,
                    // so one syscall covers a cluster instead of a row.
                    while j < sorted.count, (sorted[j] / pageSize) * pageSize <= hi {
                        hi = max(hi, ((sorted[j] + bytesPerRow + pageSize - 1) / pageSize) * pageSize)
                        j += 1
                    }
                    madvise(
                        UnsafeMutableRawPointer(mutating: base.advanced(by: lo)), hi - lo,
                        MADV_WILLNEED)
                    i = j
                }
            }
        }
    }

    /// Writes a synthetic bf16 shard set and returns its directory, so a table
    /// can be opened over it directly or re-encoded by `NGramTableQuantize`.
    static func fixtureDirectory(rowsPerShard: Int, dim: Int, shards: Int) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ngram-fixture-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for s in 0 ..< shards {
            var payload = [UInt16](repeating: 0, count: rowsPerShard * dim)
            for i in 0 ..< payload.count {
                // bf16 bit pattern for a small distinct value per position.
                payload[i] = UInt16(truncatingIfNeeded: 0x3C00 &+ (i % 97))
            }
            let body = payload.withUnsafeBufferPointer { Data(buffer: $0) }
            let header = try JSONSerialization.data(withJSONObject: [
                "weight": [
                    "dtype": "BF16", "shape": [rowsPerShard, dim],
                    "data_offsets": [0, body.count],
                ]
            ])
            var file = Data()
            withUnsafeBytes(of: UInt64(header.count).littleEndian) { file.append(contentsOf: $0) }
            file.append(header)
            file.append(body)
            try file.write(to: dir.appendingPathComponent(String(format: "shard_%03d.safetensors", s)))
        }
        return dir
    }

    /// Synthetic table backed by in-memory safetensors blobs, so the gather can
    /// be tested without the 102.4 GB checkpoint.
    static func inMemoryFixture(rowsPerShard: Int, dim: Int, shards: Int) throws
        -> Qwen4ExpNGramTable
    {
        let dir = try fixtureDirectory(rowsPerShard: rowsPerShard, dim: dim, shards: shards)
        let spec = Qwen4ExpNGramTableSpec(
            directory: dir.lastPathComponent, shards: shards, rowsPerShard: rowsPerShard,
            dim: dim, dtype: "bfloat16")
        return try Qwen4ExpNGramTable(directory: dir, spec: spec)
    }

    /// Opens the table at whatever encoding its shards actually carry.
    ///
    /// The directory describes itself: a bf16 shard names one `BF16` tensor,
    /// and a shard written by `NGramTableQuantize` names a `U8` `weight`
    /// beside `scales` and `biases`. Reading that beats a config key or an
    /// environment variable, because the encoding cannot then disagree with
    /// the bytes on disk. Per-shard validation in the designated initializer
    /// still runs afterwards, so a directory holding a mixture still fails.
    convenience init(directory: URL, spec: Qwen4ExpNGramTableSpec) throws {
        try self.init(
            directory: directory, spec: spec,
            bits: Self.detectBits(directory: directory, spec: spec))
    }

    /// Reads shard 0's header and reports the encoding width: 16, 8 or 4.
    static func detectBits(directory: URL, spec: Qwen4ExpNGramTableSpec) throws -> Int {
        let url = directory.appendingPathComponent("shard_000.safetensors")
        let data = try Data(contentsOf: url, options: [.alwaysMapped])
        guard data.count >= 8 else { throw Qwen4ExpNGramError.badShard(url.path) }
        let headerLength = Int(
            data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian })
        guard 8 + headerLength <= data.count else { throw Qwen4ExpNGramError.badShard(url.path) }
        let header =
            try JSONSerialization.jsonObject(with: data.subdata(in: 8 ..< (8 + headerLength)))
            as? [String: Any]
        guard let info = header?["weight"] as? [String: Any],
            let dtype = info["dtype"] as? String,
            let shape = info["shape"] as? [Int], shape.count == 2
        else { throw Qwen4ExpNGramError.badShard(url.path) }
        if dtype == "BF16" { return 16 }
        guard dtype == "U8" else { throw Qwen4ExpNGramError.badShard(url.path) }
        // A quantized record is [scale f16][bias f16][codes], so the row width
        // alone names the width: 4 + dim for int8, 4 + dim/2 for int4.
        if shape[1] == 4 + spec.dim { return 8 }
        if shape[1] == 4 + spec.dim / 2 { return 4 }
        // NVFP4 adds one E4M3 byte per 16-value block ahead of the codes.
        if shape[1] == 4 + spec.dim / 16 + spec.dim / 2 { return nvfp4Bits }
        throw Qwen4ExpNGramError.badShard(url.path)
    }

    /// `bits` selects the on-disk row encoding: 16 reads the checkpoint's bf16
    /// shards directly; 8 and 4 read `NGramTableQuantize` output, one `U8`
    /// tensor whose rows are `[scale f16][bias f16][codes]` records.
    init(directory: URL, spec: Qwen4ExpNGramTableSpec, bits: Int) throws {
        precondition(
            bits == 16 || bits == 8 || bits == 4 || bits == Self.nvfp4Bits,
            "n-gram table bits must be 16, 8, 4 or nvfp4")
        precondition(
            bits != 16 || spec.dtype == "bfloat16", "bf16 n-gram table must declare dtype bfloat16")
        self.bits = bits
        rowsPerShard = spec.rowsPerShard
        dim = spec.dim
        // One contiguous record per row in every encoding, so one stride and
        // one page fault per row rather than three.
        bytesPerRow =
            bits == 16
            ? dim * 2
            : (bits == Self.nvfp4Bits
                ? 4 + dim / 16 + dim / 2 : 4 + (bits == 8 ? dim : dim / 2))
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
                let dtype = info["dtype"] as? String,
                let shape = info["shape"] as? [Int],
                let range = info["data_offsets"] as? [Int], range.count == 2
            else { throw Qwen4ExpNGramError.badShard(url.path) }
            if bits == 16 {
                guard dtype == "BF16", shape == [spec.rowsPerShard, spec.dim]
                else { throw Qwen4ExpNGramError.badShard(url.path) }
            } else {
                guard dtype == "U8", shape == [spec.rowsPerShard, bytesPerRow]
                else { throw Qwen4ExpNGramError.badShard(url.path) }
            }
            maps.append(data)
            offsets.append(8 + headerLength + range[0])
        }
        self.maps = maps
        self.dataOffsets = offsets
    }

    /// `gids[t]` lists one global row id per head. Returns `[T, heads * dim]` float16.
    func gather(_ gids: [[Int64]]) -> MLXArray {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let T = gids.count
        let heads = gids.first?.count ?? 0
        var pages = Set<Int>()
        if Self.statsEnabled {
            pages.reserveCapacity(T * heads)
            for row in gids {
                for gid in row {
                    let shard = Int(gid) / rowsPerShard
                    let r = Int(gid) % rowsPerShard
                    // Exact for every encoding now: one record per row means
                    // one start offset. The previous split layout counted only
                    // the weight page and undercounted the quantized paths by
                    // about 3x, which hid why they were slower than bf16.
                    let start = dataOffsets[shard] + r * bytesPerRow
                    pages.insert(shard << 40 | (start / 16384))
                }
            }
        }
        prefetch(gids)
        var out = [Float16](repeating: 0, count: T * heads * dim)
        if bits == 16 {
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
                                // Named `raw`, not `bits`: `bits` is this type's
                                // encoding width (16, 8 or 4) and shadowing it
                                // here would give one name two meanings.
                                let raw =
                                    UInt32(base.loadUnaligned(fromByteOffset: c * 2, as: UInt16.self))
                                    << 16
                                dst[cursor + c] = Float16(Float(bitPattern: raw))
                            }
                        }
                        cursor += dim
                    }
                }
            }
        } else {
            // Quantized rows carry a per-row f16 scale and bias; reconstruct on
            // the CPU into f16 directly. There is no MLX dtype for a per-row
            // affine code with this layout, so the GPU cast trick above does
            // not apply here.
            out.withUnsafeMutableBufferPointer { dstBuf in
                var cursor = 0
                for row in gids {
                    for gid in row {
                        let shard = Int(gid) / rowsPerShard
                        let r = Int(gid) % rowsPerShard
                        maps[shard].withUnsafeBytes { src in
                            let s = src.baseAddress!
                            // One record: scale, bias, then the codes.
                            let rec = dataOffsets[shard] + r * bytesPerRow
                            let scale = Float(
                                Float16(
                                    bitPattern: UInt16(
                                        littleEndian: s.loadUnaligned(
                                            fromByteOffset: rec, as: UInt16.self))))
                            let bias = Float(
                                Float16(
                                    bitPattern: UInt16(
                                        littleEndian: s.loadUnaligned(
                                            fromByteOffset: rec + 2, as: UInt16.self))))
                            let rowStart = rec + 4
                            if bits == Self.nvfp4Bits {
                                let blocks = dim / 16
                                let codeBase = rec + 4 + blocks
                                for b in 0 ..< blocks {
                                    let bcode = s.loadUnaligned(
                                        fromByteOffset: rec + 4 + b, as: UInt8.self)
                                    let step = qwen4ExpE4M3ToFloat(bcode) * scale
                                    for c in b * 16 ..< (b + 1) * 16 {
                                        let byte = s.loadUnaligned(
                                            fromByteOffset: codeBase + c / 2, as: UInt8.self)
                                        let nib = c % 2 == 0 ? (byte & 0xF) : (byte >> 4)
                                        let mag = qwen4ExpE2M1[Int(nib & 0x7)]
                                        let v = (nib & 0x8) != 0 ? -mag : mag
                                        dstBuf[cursor + c] = Float16(v * step)
                                    }
                                }
                            } else if bits == 8 {
                                for c in 0 ..< dim {
                                    let q = s.loadUnaligned(fromByteOffset: rowStart + c, as: UInt8.self)
                                    dstBuf[cursor + c] = Float16(Float(q) * scale + bias)
                                }
                            } else {
                                for c in stride(from: 0, to: dim, by: 2) {
                                    let packed = s.loadUnaligned(
                                        fromByteOffset: rowStart + c / 2, as: UInt8.self)
                                    dstBuf[cursor + c] = Float16(Float(packed & 0xF) * scale + bias)
                                    dstBuf[cursor + c + 1] = Float16(Float(packed >> 4) * scale + bias)
                                }
                            }
                        }
                        cursor += dim
                    }
                }
            }
        }
        let result = MLXArray(out, [T, heads * dim])
        Self.stats.record(
            rows: T * heads,
            nanos: DispatchTime.now().uptimeNanoseconds - t0,
            pages: pages.count)
        return result
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
        // Diagnostic override, in the same spirit as MLX_QWEN4EXP_NGRAM_STATS.
        // The table encoding is self-describing, so pointing at a converted
        // directory is enough to A/B bf16 against int8 or int4 without
        // rewriting the checkpoint's own `ngram` directory. Unset in normal
        // operation, where the config's own directory is used.
        let dir =
            ProcessInfo.processInfo.environment["MLX_QWEN4EXP_NGRAM_DIR"].map {
                URL(fileURLWithPath: $0)
            } ?? root.appendingPathComponent(spec.directory)
        let t: Qwen4ExpNGramTable
        do {
            t = try Qwen4ExpNGramTable(directory: dir, spec: spec)
        } catch {
            fatalError("\(error)")
        }
        if let override = ProcessInfo.processInfo.environment["MLX_QWEN4EXP_NGRAM_DIR"] {
            FileHandle.standardError.write(
                Data("qwen4exp: n-gram table overridden to \(override)\n".utf8))
        }
        hasher = h
        table = t
        return (h, t)
    }

    /// Computes this forward's gids and starts their readahead, without
    /// gathering. Called before layer 0 so the faults resolve during layers 0
    /// and 1 rather than stalling the PLE at layer 2.
    func prefetchAhead(ids: MLXArray, prevContext: MLXArray) {
        guard Qwen4ExpNGramTable.prefetchEnabled, Qwen4ExpNGramTable.aheadEnabled else { return }
        let (h, t) = resolve()
        let B = ids.dim(0), S = ids.dim(1)
        let idRows = ids.asType(.int64).asArray(Int64.self)
        let ctxRows = prevContext.asType(.int64).asArray(Int64.self)
        let ctx = args.ngramSize - 1
        for b in 0 ..< B {
            let history =
                Array(ctxRows[(b * ctx) ..< ((b + 1) * ctx)]) + Array(idRows[(b * S) ..< ((b + 1) * S)])
            t.prefetchAhead(h.gids(history: history))
        }
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
        let (out, newState) = shortConvFunctional(x, state: state)
        if let cache { cache[2] = newState }
        return out
    }

    private func shortConvFunctional(_ x: MLXArray, state: MLXArray) -> (MLXArray, MLXArray) {
        let full = concatenated([state, x], axis: 1)
        return (silu(conv1d(full)), full[0..., (full.dim(1) - stateLen)..., 0...])
    }

    /// The n-gram embedding for this call, gathered on the host. Exposed so a
    /// compiled layer can take it as an input instead of tracing through the
    /// host readback.
    func gatheredEmbedding(ids: MLXArray, prevContext: MLXArray, dtype: DType) -> MLXArray {
        embedding(ids: ids, prevContext: prevContext).asType(dtype)
    }

    /// The PLE with its gathered embedding and conv state as explicit values.
    func forwardFunctional(hidden: MLXArray, embedding emb: MLXArray, convState: MLXArray)
        -> (out: MLXArray, convState: MLXArray)
    {
        let gated = gatedValue(hidden: hidden, embedding: emb)
        let (conv, newState) = shortConvFunctional(normConv(gated), state: convState)
        return (gated + conv, newState)
    }

    var convStateLength: Int { stateLen }

    private func gatedValue(hidden: MLXArray, embedding emb: MLXArray) -> MLXArray {
        let lead = Array(hidden.shape.dropLast())
        let key = normKey(keyProj(emb)).reshaped(lead + [hc, d])
        let value = valueProj(emb)
        let query = normQuery(hidden).reshaped(lead + [hc, d])
        var gate = (key * query).sum(axis: -1, keepDims: true) / Float(d).squareRoot()
        gate = sqrt(maximum(abs(gate), 1e-6)) * sign(gate)
        return (sigmoid(gate) * value.expandedDimensions(axis: -2)).reshaped(lead + [hc * d])
    }

    func callAsFunction(hidden: MLXArray, ids: MLXArray, prevContext: MLXArray, cache: ArraysCache?)
        -> MLXArray
    {
        let emb = embedding(ids: ids, prevContext: prevContext).asType(hidden.dtype)
        let gated = gatedValue(hidden: hidden, embedding: emb)
        return gated + shortConv(normConv(gated), cache: cache)
    }
}
