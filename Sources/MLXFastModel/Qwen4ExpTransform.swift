// Offline transform for Qwen3.8-Flash-Next (qwen4_exp). Lives in MLXFastModel
// because it quantizes the routed experts with MLX and Package.swift is frozen
// (MLXFastTransform links no MLX). Local fork only.
//
// Output tree:
//   config.json                      flattened text_config + quantization + ngram_table
//   model-NNNNN-of-MMMMM.safetensors runtime tensor names; dense bf16 byte copies,
//                                    experts as 4-bit weight/scales/biases
//   model.safetensors.index.json
//   ngram/shard_NNN.safetensors      one bf16 tensor `weight` per file, byte-copied
//   tokenizer files copied verbatim
import Foundation
import MLX
import MLXFastCore

public enum Qwen4ExpTransform {
    public struct Options {
        public var source: URL
        public var destination: URL
        public var expertGroupSize: Int
        public var expertBits: Int
        public var shardBytes: Int

        public init(
            source: URL, destination: URL, expertGroupSize: Int = 32, expertBits: Int = 4,
            shardBytes: Int = 4 << 30
        ) {
            self.source = source
            self.destination = destination
            self.expertGroupSize = expertGroupSize
            self.expertBits = expertBits
            self.shardBytes = shardBytes
        }
    }

    static let metadataFiles = [
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "generation_config.json",
    ]

    /// Source key -> runtime key, or nil to drop. Expert stacks are handled separately.
    static func runtimeKey(_ key: String) -> String? {
        if key.hasPrefix("model.visual.") { return nil }
        if key.hasPrefix("model.language_model.") {
            return "model." + key.dropFirst("model.language_model.".count)
        }
        if key.hasPrefix("mtp.") || key == "lm_head.weight" { return key }
        return nil
    }

    static func isNGramShard(_ key: String) -> Bool { key.contains(".ngram_embedding.shard_") }

    static func isExpertStack(_ key: String) -> Bool {
        key.hasSuffix(".mlp.experts.gate_up_proj") || key.hasSuffix(".mlp.experts.down_proj")
    }

    static func shardIndex(ofNGramKey key: String) -> Int {
        // "...ngram_embedding.shard_17.weight" -> 17
        let tail = key.components(separatedBy: ".shard_").last ?? ""
        return Int(tail.components(separatedBy: ".").first ?? "") ?? -1
    }

    public static func run(_ o: Options) throws {
        let fm = FileManager.default
        try fm.createDirectory(
            at: o.destination.appendingPathComponent("ngram"), withIntermediateDirectories: true)

        // 1. config.json: flatten text_config, add quantization; ngram_table is filled below.
        let root =
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: o.source.appendingPathComponent("config.json"))) as? [String: Any]
            ?? [:]
        guard var text = root["text_config"] as? [String: Any] else {
            throw MLXFastError.invalidInput("source config.json has no text_config")
        }
        text["model_type"] = "qwen4_exp_text"
        text["quantization"] = ["group_size": o.expertGroupSize, "bits": o.expertBits, "mode": "affine"]
        text.removeValue(forKey: "quantization_config")

        // 2. source index.
        let index =
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: o.source.appendingPathComponent("model.safetensors.index.json")))
            as? [String: Any] ?? [:]
        guard let weightMap = index["weight_map"] as? [String: String] else {
            throw MLXFastError.invalidInput("source index has no weight_map")
        }
        let bySourceFile = Dictionary(grouping: weightMap.keys.sorted()) { weightMap[$0]! }

        var ngramShards = 0
        var ngramRows = 0
        var ngramDim = 0
        var outputMap = [String: String]()
        var totalBytes = 0
        let writer = ShardWriter(directory: o.destination, shardBytes: o.shardBytes)

        for (file, keys) in bySourceFile.sorted(by: { $0.key < $1.key }) {
            let url = o.source.appendingPathComponent(file)
            let header = try Safetensors.readHeader(url)
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var expertsToQuantize = [String]()
            for key in keys {
                guard let info = header.tensors[key] else {
                    throw MLXFastError.invalidInput("\(key) missing from \(file) header")
                }
                if isNGramShard(key) {
                    let shard = shardIndex(ofNGramKey: key)
                    guard shard >= 0 else { throw MLXFastError.invalidInput("bad n-gram shard key \(key)") }
                    let bytes = try readBytes(handle, header: header, info: info)
                    try writeSafetensors(
                        url: o.destination.appendingPathComponent(
                            String(format: "ngram/shard_%03d.safetensors", shard)),
                        tensors: [("weight", info.dtype, info.shape, bytes)])
                    ngramShards += 1
                    ngramRows = info.shape[0]
                    ngramDim = info.shape[1]
                    continue
                }
                guard let rk = runtimeKey(key) else { continue }
                if isExpertStack(key) {
                    expertsToQuantize.append(key)
                    continue
                }
                let bytes = try readBytes(handle, header: header, info: info)
                var shape = info.shape
                // torch (C,1,K) -> mlx (C,K,1): same bytes, new shape
                if key.hasSuffix("conv1d.weight"), shape.count == 3, shape[1] == 1 {
                    shape = [shape[0], shape[2], 1]
                }
                try writer.append(name: rk, dtype: info.dtype, shape: shape, bytes: bytes)
                totalBytes += bytes.count
            }
            if !expertsToQuantize.isEmpty {
                let arrays = try MLX.loadArrays(url: url)
                for key in expertsToQuantize {
                    guard let rk = runtimeKey(key), let w = arrays[key] else { continue }
                    let suffix = key.hasSuffix("gate_up_proj") ? "experts.gate_up_proj" : "experts.down_proj"
                    let base = String(rk.dropLast(suffix.count))
                    var parts = [(String, MLXArray)]()
                    if key.hasSuffix("gate_up_proj") {
                        let mid = w.dim(-2) / 2
                        parts = [
                            (base + "switch_mlp.gate_proj", w[.ellipsis, 0 ..< mid, 0...]),
                            (base + "switch_mlp.up_proj", w[.ellipsis, mid..., 0...]),
                        ]
                    } else {
                        parts = [(base + "switch_mlp.down_proj", w)]
                    }
                    for (name, expert) in parts {
                        let q = MLX.quantized(expert, groupSize: o.expertGroupSize, bits: o.expertBits)
                        guard let biases = q.biases else {
                            throw MLXFastError.invalidInput("affine quantization returned no biases for \(name)")
                        }
                        eval(q.wq, q.scales, biases)
                        totalBytes += try writer.append(array: q.wq, name: name + ".weight")
                        totalBytes += try writer.append(array: q.scales, name: name + ".scales")
                        totalBytes += try writer.append(array: biases, name: name + ".biases")
                    }
                }
                MLX.Memory.clearCache()
            }
        }
        try writer.finish(into: &outputMap)
        guard ngramShards > 0 else {
            throw MLXFastError.invalidInput("no n-gram shards found in the source")
        }
        text["ngram_table"] = [
            "directory": "ngram", "shards": ngramShards, "rows_per_shard": ngramRows,
            "dim": ngramDim, "dtype": "bfloat16",
        ]
        try JSONSerialization.data(withJSONObject: text, options: [.prettyPrinted, .sortedKeys])
            .write(to: o.destination.appendingPathComponent("config.json"))
        let outIndex: [String: Any] = ["metadata": ["total_size": totalBytes], "weight_map": outputMap]
        try JSONSerialization.data(withJSONObject: outIndex, options: [.prettyPrinted, .sortedKeys])
            .write(to: o.destination.appendingPathComponent("model.safetensors.index.json"))
        for f in metadataFiles {
            let s = o.source.appendingPathComponent(f)
            let d = o.destination.appendingPathComponent(f)
            if fm.fileExists(atPath: s.path) {
                try? fm.removeItem(at: d)
                try fm.copyItem(at: s, to: d)
            }
        }
    }

    static func readBytes(_ handle: FileHandle, header: SafetensorsHeader, info: SafetensorInfo) throws -> Data {
        try handle.seek(toOffset: header.dataBaseOffset + UInt64(info.dataStart))
        let n = info.dataEnd - info.dataStart
        guard let d = try handle.read(upToCount: n), d.count == n else {
            throw MLXFastError.invalidInput("short read for \(info.name)")
        }
        return d
    }

    static func dtypeName(_ dtype: DType) throws -> String {
        switch dtype {
        case .uint32: return "U32"
        case .bfloat16: return "BF16"
        case .float16: return "F16"
        case .float32: return "F32"
        case .int64: return "I64"
        default: throw MLXFastError.invalidInput("unsupported dtype \(dtype)")
        }
    }

    /// Minimal safetensors writer: header JSON + payloads, one file.
    static func writeSafetensors(url: URL, tensors: [(name: String, dtype: String, shape: [Int], bytes: Data)])
        throws
    {
        var header = [String: Any]()
        var offset = 0
        for t in tensors {
            header[t.name] = ["dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, offset + t.bytes.count]]
            offset += t.bytes.count
        }
        header["__metadata__"] = ["format": "pt"]
        var headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while headerData.count % 8 != 0 { headerData.append(0x20) }
        var out = Data()
        var len = UInt64(headerData.count).littleEndian
        out.append(Data(bytes: &len, count: 8))
        out.append(headerData)
        try out.write(to: url)
        let h = try FileHandle(forWritingTo: url)
        defer { try? h.close() }
        try h.seekToEnd()
        for t in tensors { try h.write(contentsOf: t.bytes) }
    }

    /// Accumulates tensors and flushes a numbered shard when `shardBytes` is reached.
    final class ShardWriter {
        let directory: URL
        let shardBytes: Int
        private var pending = [(name: String, dtype: String, shape: [Int], bytes: Data)]()
        private var pendingBytes = 0
        private var flushed = [(file: URL, names: [String])]()

        init(directory: URL, shardBytes: Int) {
            self.directory = directory
            self.shardBytes = shardBytes
        }

        func append(name: String, dtype: String, shape: [Int], bytes: Data) throws {
            pending.append((name, dtype, shape, bytes))
            pendingBytes += bytes.count
            if pendingBytes >= shardBytes { try flush() }
        }

        /// Returns the byte count written.
        func append(array: MLXArray, name: String) throws -> Int {
            let bytes = array.asData(noCopy: false)
            try append(name: name, dtype: try Qwen4ExpTransform.dtypeName(array.dtype), shape: array.shape, bytes: bytes)
            return bytes.count
        }

        private func flush() throws {
            guard !pending.isEmpty else { return }
            let file = directory.appendingPathComponent(String(format: "model-%05d.safetensors", flushed.count + 1))
            try Qwen4ExpTransform.writeSafetensors(url: file, tensors: pending)
            flushed.append((file, pending.map { $0.name }))
            pending.removeAll()
            pendingBytes = 0
        }

        func finish(into map: inout [String: String]) throws {
            try flush()
            let n = flushed.count
            for (i, entry) in flushed.enumerated() {
                let final = directory.appendingPathComponent(
                    String(format: "model-%05d-of-%05d.safetensors", i + 1, n))
                try FileManager.default.moveItem(at: entry.file, to: final)
                for name in entry.names { map[name] = final.lastPathComponent }
            }
        }
    }
}
