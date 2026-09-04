import Foundation

/// Offline re-encoding of the Qwen3.8-Flash-Next n-gram embedding table from
/// bf16 to a per-row affine int8 or int4 code.
///
/// This exists for footprint and residency, not speed. The bf16 table is
/// 102.4 GB and the transformer tower is about 87.2 GB, for a combined 189.6
/// GB that does not fit in the 128 GB this machine has. At int4 the table is
/// 26.9 GB, for a combined total near 114 GB, which fits. Measured gather cost
/// at real per-forward geometry is 1.630 ms in a release build against about
/// 1,060 ms of MoE work per forward -- about 0.15 percent -- so nothing here
/// is justified as a speedup; see `docs/perf/qwen38-flash-2026-09.md`.
///
/// One scale and one bias per 160-value row. A row is one n-gram embedding, so
/// per-row scaling preserves each embedding's own dynamic range at a cost of 4
/// metadata bytes against 320 payload bytes. The table is 320,001,536 rows, so
/// this converts 102.4 GB to 52.5 GB (int8) or 26.9 GB (int4).
public enum NGramTableQuantize {
    public enum Failure: Error, CustomStringConvertible {
        case badShard(String)
        case badBits(Int)
        public var description: String {
            switch self {
            case .badShard(let p): return "n-gram shard is not a single BF16 [rows, dim] 'weight': \(p)"
            case .badBits(let b): return "n-gram quantization supports bits 4 or 8, got \(b)"
            }
        }
    }

    public static func convert(sourceDir: URL, destDir: URL, bits: Int) throws {
        guard bits == 4 || bits == 8 else { throw Failure.badBits(bits) }
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
        let perRow = bits == 8 ? dim : dim / 2

        var payload = Data(count: rows * perRow)
        var scales = [UInt16](repeating: 0, count: rows)
        var biases = [UInt16](repeating: 0, count: rows)

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
                    let levels = Float(bits == 8 ? 255 : 15)
                    let scale = (hi - lo) / levels
                    let safeScale = scale == 0 ? 1 : scale
                    scales[r] = Float16(scale).bitPattern
                    biases[r] = Float16(lo).bitPattern
                    let rowOut = d.advanced(by: r * perRow)
                    if bits == 8 {
                        for c in 0 ..< dim {
                            let q = ((values[c] - lo) / safeScale).rounded()
                            rowOut.storeBytes(
                                of: UInt8(max(0, min(Float(255), q))), toByteOffset: c, as: UInt8.self)
                        }
                    } else {
                        for c in stride(from: 0, to: dim, by: 2) {
                            let q0 = UInt8(max(0, min(Float(15), ((values[c] - lo) / safeScale).rounded())))
                            let q1 = UInt8(
                                max(0, min(Float(15), ((values[c + 1] - lo) / safeScale).rounded())))
                            rowOut.storeBytes(of: q0 | (q1 << 4), toByteOffset: c / 2, as: UInt8.self)
                        }
                    }
                }
            }
        }

        let scaleBytes = scales.withUnsafeBufferPointer { Data(buffer: $0) }
        let biasBytes = biases.withUnsafeBufferPointer { Data(buffer: $0) }
        let outHeader = try JSONSerialization.data(withJSONObject: [
            "weight": [
                "dtype": "U8", "shape": [rows, perRow],
                "data_offsets": [0, payload.count],
            ],
            "scales": [
                "dtype": "F16", "shape": [rows],
                "data_offsets": [payload.count, payload.count + scaleBytes.count],
            ],
            "biases": [
                "dtype": "F16", "shape": [rows],
                "data_offsets": [
                    payload.count + scaleBytes.count,
                    payload.count + scaleBytes.count + biasBytes.count,
                ],
            ],
        ])
        var file = Data()
        withUnsafeBytes(of: UInt64(outHeader.count).littleEndian) { file.append(contentsOf: $0) }
        file.append(outHeader)
        file.append(payload)
        file.append(scaleBytes)
        file.append(biasBytes)
        try file.write(to: dest)
    }
}
