import Foundation

/// Offline re-encoding of the Qwen3.8-Flash-Next n-gram embedding table from
/// bf16 to a per-row affine int8 or int4 code, or to NVFP4.
///
/// This buys disk footprint AND read speed. On the real table a quantized
/// gather measures about 856 ms against bf16's 1014 ms for one 512-token
/// prefill, roughly 15 percent faster, and the table goes from 102.4 GB to
/// 49 GB (int8), 28 GB (nvfp4) or 25 GB (int4). int4 is the adopted encoding;
/// `docs/perf/qwen38-flash-2026-09.md` has the four-way comparison.
///
/// Two earlier claims in this comment were wrong and are recorded here so they
/// are not reintroduced. It said the re-encoding existed for RESIDENCY, on the
/// grounds that the table plus the tower exceed this machine's memory. The
/// table is memory-mapped and sparsely touched, so it never needed to be
/// resident and footprint is a disk argument. It also said the gather costs
/// 0.15 percent of a forward and that no speedup was available. That figure
/// came from a 6.4 MB fixture that was fully faulted in, which deleted the
/// page-fault term; on the real table the gather is about 14 percent of a
/// forward.
///
/// One scale and one bias per 160-value row. A row is one n-gram embedding, so
/// per-row scaling preserves each embedding's own dynamic range at a cost of 4
/// metadata bytes against 320 payload bytes. The table is 320,001,536 rows, so
/// this converts 102.4 GB to 52.5 GB (int8) or 26.9 GB (int4).
///
/// The row's scale, bias and payload are stored CONTIGUOUSLY, as one record of
/// `4 + dim*bits/8` bytes, rather than in three separate tensors. That layout
/// is what makes a quantized table cheaper to read than the bf16 one. The
/// production gather touches thousands of rows scattered across 320 million,
/// so no two rows share a 16 KiB page and each row costs its own page fault.
/// Splitting a row across a weight region, a scale region and a bias region
/// therefore costs THREE faults per row and measured 2.3x slower than bf16
/// despite moving a quarter of the bytes. One record is one fault.
/// E2M1 magnitudes by 3-bit magnitude code; a sign bit sits above these.
/// Exponent 0 is subnormal (0, 0.5); exponents 1 to 3 scale 1, 2 and 4 by
/// 1 + mantissa/2.
let e2m1Magnitude: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

/// Decodes one OCP FP8 E4M3 byte: bias 7, no infinities, 0x7F and 0xFF NaN.
func e4m3ToFloat(_ b: UInt8) -> Float {
    let sign: Float = (b & 0x80) != 0 ? -1 : 1
    let exp = Int((b >> 3) & 0x0F)
    let man = Int(b & 0x07)
    if exp == 0 { return sign * Float(man) * 0.001953125 }
    if exp == 15 && man == 7 { return .nan }
    return sign * (1 + Float(man) / 8) * exp2(Float(exp - 7))
}

/// The 128 finite non-negative E4M3 values, ascending, with their byte codes.
/// Encoding a positive scale is a binary search over this rather than bit
/// surgery, which keeps the mapping obviously correct at 25 million encodes
/// per shard.
let e4m3PositiveLadder: [(value: Float, code: UInt8)] = {
    var out: [(Float, UInt8)] = []
    for b in UInt8(0) ... UInt8(126) {
        let v = e4m3ToFloat(b)
        if v.isFinite { out.append((v, b)) }
    }
    return out.sorted { $0.0 < $1.0 }
}()

/// Nearest E4M3 code for a non-negative value, saturating at the ladder ends.
func floatToE4M3(_ x: Float) -> UInt8 {
    let ladder = e4m3PositiveLadder
    if !(x > 0) { return 0 }
    if x >= ladder[ladder.count - 1].value { return ladder[ladder.count - 1].code }
    var lo = 0, hi = ladder.count - 1
    while lo + 1 < hi {
        let mid = (lo + hi) / 2
        if ladder[mid].value <= x { lo = mid } else { hi = mid }
    }
    return (x - ladder[lo].value) <= (ladder[hi].value - x) ? ladder[lo].code : ladder[hi].code
}

public enum NGramTableQuantize {
    public enum Failure: Error, CustomStringConvertible {
        case badShard(String)
        case badBits(Int)
        public var description: String {
            switch self {
            case .badShard(let p): return "n-gram shard is not a single BF16 [rows, dim] 'weight': \(p)"
            case .badBits(let b):
                return "n-gram quantization supports bits 8, 4 or 40 (nvfp4), got \(b)"
            }
        }
    }

    /// `bits` is 8, 4, or `nvfp4Bits` for NVFP4. NVFP4 packs the same four
    /// bits per value as int4 but spends them as E2M1 floats under a per-16
    /// E4M3 block scale, so a row carries ten block scales instead of one
    /// affine scale. That is 92 bytes a row against int4's 84.
    public static let nvfp4Bits = 40

    public static func convert(sourceDir: URL, destDir: URL, bits: Int) throws {
        guard bits == 4 || bits == 8 || bits == nvfp4Bits else { throw Failure.badBits(bits) }
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)
        let shards = try FileManager.default.contentsOfDirectory(atPath: sourceDir.path)
            .filter { $0.hasPrefix("shard_") && $0.hasSuffix(".safetensors") }
            .sorted()
        for name in shards {
            try convertShard(
                source: sourceDir.appendingPathComponent(name),
                dest: destDir.appendingPathComponent(name),
                bits: bits)
        }
    }

    private static func convertShard(source: URL, dest: URL, bits: Int) throws {
        let data = try Data(contentsOf: source, options: [.alwaysMapped])
        guard data.count >= 8 else { throw Failure.badShard(source.path) }
        let headerLength = Int(data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian })
        let header = try JSONSerialization.jsonObject(
            with: data.subdata(in: 8 ..< (8 + headerLength))) as? [String: Any]
        guard let info = header?["weight"] as? [String: Any],
            (info["dtype"] as? String) == "BF16",
            let shape = info["shape"] as? [Int], shape.count == 2,
            let range = info["data_offsets"] as? [Int], range.count == 2
        else { throw Failure.badShard(source.path) }
        let rows = shape[0], dim = shape[1]
        let base = 8 + headerLength + range[0]
        let isNVFP4 = bits == nvfp4Bits
        let blocks = dim / 16
        let codes = bits == 8 ? dim : dim / 2
        // Contiguous per row. Affine: [scale f16][bias f16][codes].
        // NVFP4:  [rowScale f16][pad 2][blocks x E4M3][codes].
        let perRow = isNVFP4 ? 4 + blocks + codes : 4 + codes

        var payload = Data(count: rows * perRow)

        data.withUnsafeBytes { src in
            payload.withUnsafeMutableBytes { dst in
                let s = src.baseAddress!, d = dst.baseAddress!
                var values = [Float](repeating: 0, count: dim)
                for r in 0 ..< rows {
                    let rowStart = base + r * dim * 2
                    var lo = Float.greatestFiniteMagnitude, hi = -Float.greatestFiniteMagnitude
                    for c in 0 ..< dim {
                        let bits16 = s.loadUnaligned(fromByteOffset: rowStart + c * 2, as: UInt16.self)
                        let v = Float(bitPattern: UInt32(bits16) << 16)
                        values[c] = v
                        lo = min(lo, v); hi = max(hi, v)
                    }
                    let rowOut = d.advanced(by: r * perRow)
                    if isNVFP4 {
                        // Per-row global scale brings block scales into E4M3
                        // range. Strict NVFP4 makes this per tensor; per row is
                        // the analogue in a table whose rows are independent
                        // embeddings, and it is strictly the more accurate of
                        // the two. E2M1 tops out at 6 and E4M3 at 448.
                        let amax = max(abs(lo), abs(hi))
                        let rowScale = amax > 0 ? amax / (6 * 448) : 1
                        rowOut.storeBytes(
                            of: Float16(rowScale).bitPattern.littleEndian, toByteOffset: 0,
                            as: UInt16.self)
                        rowOut.storeBytes(of: UInt16(0), toByteOffset: 2, as: UInt16.self)
                        let rs = Float(Float16(rowScale))
                        for b in 0 ..< blocks {
                            var bmax: Float = 0
                            for c in b * 16 ..< (b + 1) * 16 { bmax = max(bmax, abs(values[c])) }
                            let blockScale = bmax > 0 ? bmax / 6 / rs : 0
                            let bcode = floatToE4M3(blockScale)
                            rowOut.storeBytes(of: bcode, toByteOffset: 4 + b, as: UInt8.self)
                            let step = e4m3ToFloat(bcode) * rs
                            for c in b * 16 ..< (b + 1) * 16 {
                                let t = step > 0 ? values[c] / step : 0
                                let sign: UInt8 = t < 0 ? 8 : 0
                                let m = abs(t)
                                var best = 0
                                var bestErr = Float.greatestFiniteMagnitude
                                for k in 0 ..< 8 {
                                    let e = abs(e2m1Magnitude[k] - m)
                                    if e < bestErr { bestErr = e; best = k }
                                }
                                let nib = sign | UInt8(best)
                                let off = 4 + blocks + c / 2
                                if c % 2 == 0 {
                                    rowOut.storeBytes(of: nib, toByteOffset: off, as: UInt8.self)
                                } else {
                                    let prev = rowOut.load(fromByteOffset: off, as: UInt8.self)
                                    rowOut.storeBytes(
                                        of: prev | (nib << 4), toByteOffset: off, as: UInt8.self)
                                }
                            }
                        }
                        continue
                    }
                    let levels = Float(bits == 8 ? 255 : 15)
                    let scale = (hi - lo) / levels
                    let safeScale = scale == 0 ? 1 : scale
                    rowOut.storeBytes(
                        of: Float16(scale).bitPattern.littleEndian, toByteOffset: 0, as: UInt16.self)
                    rowOut.storeBytes(
                        of: Float16(lo).bitPattern.littleEndian, toByteOffset: 2, as: UInt16.self)
                    if bits == 8 {
                        for c in 0 ..< dim {
                            let q = ((values[c] - lo) / safeScale).rounded()
                            rowOut.storeBytes(
                                of: UInt8(max(0, min(Float(255), q))), toByteOffset: 4 + c,
                                as: UInt8.self)
                        }
                    } else {
                        for c in stride(from: 0, to: dim, by: 2) {
                            let q0 = UInt8(max(0, min(Float(15), ((values[c] - lo) / safeScale).rounded())))
                            let q1 = UInt8(
                                max(0, min(Float(15), ((values[c + 1] - lo) / safeScale).rounded())))
                            rowOut.storeBytes(of: q0 | (q1 << 4), toByteOffset: 4 + c / 2, as: UInt8.self)
                        }
                    }
                }
            }
        }

        let outHeader = try JSONSerialization.data(withJSONObject: [
            "weight": [
                "dtype": "U8", "shape": [rows, perRow],
                "data_offsets": [0, payload.count],
            ]
        ])
        var file = Data()
        withUnsafeBytes(of: UInt64(outHeader.count).littleEndian) { file.append(contentsOf: $0) }
        file.append(outHeader)
        file.append(payload)
        try file.write(to: dest)
    }
}
