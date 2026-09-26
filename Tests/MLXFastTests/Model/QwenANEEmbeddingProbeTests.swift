import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANEEmbeddingProbeTests {
    @Test func rejectsEmptyConfiguration() {
        #expect(throws: QwenANEEmbeddingError.self) { try QwenANEEmbedding(tileWeightBlobs: [:]) }
    }

    @Test func realTokenLookupAndEviction() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_EMBEDDING_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let weightsRoot = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_TARGET_WEIGHTS"]))
        let input = try MLX.loadArrays(url: root.appendingPathComponent("checkpoint-input.safetensors"))
        let tokens = try #require(input["input_tokens"]).asArray(Int32.self).map(Int.init)
        try #require(tokens.count == 32)
        let edgeTokens = [0, 2047, 2048, 248319, 248044, 248069]
        let edges = (0..<32).map { edgeTokens[$0 % edgeTokens.count] }
        let patterns = [tokens, Array(tokens.reversed()), edges, tokens]
        let tiles = Set(patterns.flatMap { $0 }.map { $0 / 2048 }).sorted()
        let data = try Data(contentsOf: weightsRoot.appendingPathComponent("model.safetensors.index.json"))
        let index = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let mapping = try #require(index["weight_map"] as? [String: String])
        let prefix = "language_model.model.embed_tokens"
        let shard = try #require(mapping[prefix + ".weight"])
        try #require(mapping[prefix + ".scales"] == shard && mapping[prefix + ".biases"] == shard)
        let arrays = try MLX.loadArrays(url: weightsRoot.appendingPathComponent(shard))
        let packed = try #require(arrays[prefix + ".weight"])
        let scales = try #require(arrays[prefix + ".scales"]), biases = try #require(arrays[prefix + ".biases"])
        try #require(packed.shape == [248320, 640] && scales.shape == [248320, 80] && biases.shape == scales.shape)
        var report: [String: Any] = ["phase": "prepare", "tiles": tiles, "loaded_tile_limit": 2,
            "scope": "Actual token embedding lookup on ANE with bounded program cache; selected tile weights prepared offline; not full vocabulary preload"]
        func save() throws {
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("embedding-lookup.json"), options: .atomic)
        }
        try save()
        var blobs: [Int: Data] = [:], tables: [Int: [Float16]] = [:]
        for tile in tiles {
            let start = tile * 2048, end = min(start + 2048, 248320)
            var weight = aneProbeAffineWeight(packed[start..<end], scales: scales[start..<end],
                biases: biases[start..<end], groupSize: 64, bits: 4)
            if end - start < 2048 { weight = concatenated([weight, MLXArray.zeros([2048 - (end - start), 5120], dtype: .float16)], axis: 0) }
            tables[tile] = weight.asArray(Float16.self)
            blobs[tile] = buildConvWeightBlob(f16Bytes(weight.T))
        }
        let lookup = try QwenANEEmbedding(tileWeightBlobs: blobs, maxLoadedTiles: 2)
        #expect(throws: QwenANEEmbeddingError.self) { try lookup(tokens: Array(repeating: -1, count: 32)) }
        #expect(throws: QwenANEEmbeddingError.self) { try lookup(tokens: Array(repeating: 248320, count: 32)) }
        let absent = try #require((0..<122).first { blobs[$0] == nil })
        #expect(throws: QwenANEEmbeddingError.self) { try lookup(tokens: Array(repeating: absent * 2048, count: 32)) }
        var firstOutput: [Float16] = [], trials: [[String: Any]] = []
        for (trial, ids) in patterns.enumerated() {
            report["phase"] = "lookup"; report["trial"] = trial; try save()
            let began = Date()
            let actual = try lookup(tokens: ids)
            let elapsed = Date().timeIntervalSince(began)
            try #require(actual.count == 32 * 5120 && actual.allSatisfy { $0.isFinite })
            var squared = 0.0, energy = 0.0, maximum = 0.0
            for lane in 0..<32 {
                let table = try #require(tables[ids[lane] / 2048])
                for channel in 0..<5120 {
                    let expected = Double(table[(ids[lane] % 2048) * 5120 + channel])
                    let delta = Double(actual[lane * 5120 + channel]) - expected
                    squared += delta * delta; energy += expected * expected; maximum = max(maximum, abs(delta))
                }
            }
            let relative = sqrt(squared / max(energy, Double.leastNormalMagnitude))
            trials.append(["trial": trial, "max_abs_error": maximum, "relative_rms_error": relative,
                "loaded_tiles": lookup.loadedTileCount, "lookup_including_load_seconds": elapsed])
            report["trials"] = trials; try save()
            try #require(lookup.loadedTileCount <= 2 && relative < 0.002)
            if trial == 0 { firstOutput = actual }
            if trial == 3 { try #require(actual == firstOutput) }
        }
        let captured = try #require(input["input"])
        try #require(captured.shape == [32, 10240])
        let original = captured.asArray(Float16.self)
        var nextInput: [Float16] = []
        for lane in 0..<32 {
            nextInput.append(contentsOf: firstOutput[(lane * 5120)..<((lane + 1) * 5120)])
            nextInput.append(contentsOf: original[(lane * 10240 + 5120)..<((lane + 1) * 10240)])
        }
        try MLX.save(arrays: ["embedding": MLXArray(firstOutput, [32, 5120]),
            "input": MLXArray(nextInput, [32, 10240]), "input_tokens": MLXArray(tokens.map(Int32.init))],
            url: root.appendingPathComponent("embedding-lookup.safetensors"))
        report["phase"] = "complete"; try save()
    }
}
