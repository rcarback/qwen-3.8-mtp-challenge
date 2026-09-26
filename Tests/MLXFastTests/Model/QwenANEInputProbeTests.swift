import CryptoKit
import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANEInputProbeTests {
    @Test func checkpointDualNormFC() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_INPUT_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let source = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_PROJECTION_HEAD"]))
        let arrays = try MLX.loadArrays(url: source)
        let archivePath = env["MLXFAST_ANE_INPUT_ARCHIVE"]
        func array(_ key: String) throws -> MLXArray { try #require(arrays[key]) }
        let weight = try aneProbeAffineWeight(array("fc.weight"), scales: array("fc.scales"),
            biases: array("fc.biases"), groupSize: 64, bits: 4)
        let embeddingNorm = try array("pre_fc_norm_embedding.weight").asType(.float16)
        let hiddenNorm = try array("pre_fc_norm_hidden.weight").asType(.float16)
        try #require(weight.shape == [5120, 10240])
        try #require(embeddingNorm.shape == [5120] && hiddenNorm.shape == [5120])
        let normWeights = concatenated([embeddingNorm, hiddenNorm])
        let blob = buildConvWeightBlob(f16Bytes(weight))
        let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        let text = buildQwenANEInputMILText(normWeights: normWeights.asArray(Float16.self), programTag: digest)
        try text.write(to: root.appendingPathComponent("checkpoint-input.mil"), atomically: true, encoding: .utf8)
        var result: [String: Any] = ["phase": "compile", "rows": 32,
            "scope": archivePath == nil ? "Checkpoint dual normalization and FC, synthetic embedding and hidden vectors; no embedding lookup" : "First 32 causal positions of a saved training seed; offline target embedding preparation, ANE dual normalization and FC",
            "archive": archivePath ?? "synthetic",
            "effective_weight_blob_sha256": digest]
        func save() throws {
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("checkpoint-input.json"), options: .atomic)
        }
        try save()
        do {
            let model = try ANEInMemoryModel(milText: text, weightBlob: blob)
            try model.compile(); result["phase"] = "load"; try save()
            try model.load(); defer { model.unload() }
            var inputs: [Float16] = []
            for row in 0..<32 {
                for half in 0..<2 {
                    for channel in 0..<5120 {
                        let integer = (channel * (half + 3) + row * 11) % 127 - 63
                        let scale: Float = half == 0 ? 64 : 16
                        inputs.append(Float16(Float(integer) / scale))
                    }
                }
            }
            var input = MLXArray(inputs, [32, 10240])
            var tokenIDs: [Int32] = []
            if let archivePath {
                let archive = URL(fileURLWithPath: archivePath)
                let framesData = try Data(contentsOf: archive.appendingPathComponent("frames.json"))
                let metadata = try #require(JSONSerialization.jsonObject(with: framesData) as? [String: Any])
                let frames = try #require(metadata["frames"] as? [[String: Any]])
                let first = try #require(frames.first)
                let tokens = try #require(first["inputTokens"] as? [Int])
                let tensors = try MLX.loadArrays(url: archive.appendingPathComponent("frames.safetensors"))
                let states = try #require(tensors["0"])
                try #require(states.shape == [1, tokens.count, 5120] && tokens.count >= 32)
                tokenIDs = tokens.prefix(32).map(Int32.init)
                try #require(tokenIDs.allSatisfy { $0 >= 0 && $0 < 248320 })
                let weightsRoot = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_TARGET_WEIGHTS"]))
                let indexData = try Data(contentsOf: weightsRoot.appendingPathComponent("model.safetensors.index.json"))
                let index = try #require(JSONSerialization.jsonObject(with: indexData) as? [String: Any])
                let mapping = try #require(index["weight_map"] as? [String: String])
                let prefix = "language_model.model.embed_tokens"
                let shard = try #require(mapping[prefix + ".weight"])
                try #require(mapping[prefix + ".scales"] == shard && mapping[prefix + ".biases"] == shard)
                let target = try MLX.loadArrays(url: weightsRoot.appendingPathComponent(shard))
                let ids = MLXArray(tokenIDs)
                let embedding = try aneProbeAffineWeight(take(#require(target[prefix + ".weight"]), ids, axis: 0),
                    scales: take(#require(target[prefix + ".scales"]), ids, axis: 0),
                    biases: take(#require(target[prefix + ".biases"]), ids, axis: 0), groupSize: 64, bits: 4)
                let stateRows = states[0, 0..<32, 0...].asType(.float16)
                input = concatenated([embedding, stateRows], axis: 1)
                try #require(input.shape == [32, 10240])
                result["frames_metadata_sha256"] = SHA256.hash(data: framesData).map { String(format: "%02x", $0) }.joined()
            }
            let pairs = input.asType(.float32).reshaped([32, 2, 5120])
            let unit = pairs * rsqrt(mean(pairs * pairs, axis: -1, keepDims: true) + 1e-6)
            let normalized = (unit * normWeights.asType(.float32).reshaped([1, 2, 5120]))
                .asType(.float16).reshaped([32, 10240])
            let reference = matmul(normalized, weight.T)
            // MLX eval materializes tensor operations, not executable source text.
            eval(reference)
            if let lookupPath = env["MLXFAST_ANE_INPUT_EMBEDDING"] {
                try #require(archivePath != nil)
                let lookupURL = URL(fileURLWithPath: lookupPath)
                let lookup = try MLX.loadArrays(url: lookupURL)
                let replacement = try #require(lookup["input"])
                let lookupTokens = try #require(lookup["input_tokens"])
                try #require(replacement.shape == [32, 10240])
                try #require(lookupTokens.asArray(Int32.self) == tokenIDs)
                try #require(replacement[0..., 5120...].asArray(Float16.self)
                    == input[0..., 5120...].asArray(Float16.self))
                try #require(replacement.asArray(Float16.self).allSatisfy { $0.isFinite })
                input = replacement
                result["scope"] = "Saved ANE token embeddings through decoder; original GPU embedding reference preserved"
                result["embedding_input_sha256"] = SHA256.hash(data: try Data(contentsOf: lookupURL))
                    .map { String(format: "%02x", $0) }.joined()
            }
            result["phase"] = "dispatch"; try save()
            let start = Date()
            let output = try ANEDirectDispatch.runConv(model: model, x: input,
                inputDim: 10240, outputDim: 5120, sequenceLength: 32)
            let actual = output.asArray(Float.self), expected = reference.asArray(Float.self)
            result["dispatch_and_read_seconds"] = Date().timeIntervalSince(start)
            try #require(actual.count == expected.count)
            let nonfinite = actual.filter { !$0.isFinite }.count
            result["nonfinite_values"] = nonfinite
            if nonfinite > 0 { try save(); try #require(nonfinite == 0) }
            var maximum: Double = 0, absolute: Double = 0, squared: Double = 0, referenceSquared: Double = 0
            for (value, reference) in zip(actual, expected) {
                let error = Double(value) - Double(reference)
                maximum = max(maximum, abs(error)); absolute += abs(error); squared += error * error
                referenceSquared += Double(reference) * Double(reference)
            }
            result["max_abs_error"] = maximum
            result["mean_abs_error"] = absolute / Double(actual.count)
            result["relative_rms_error"] = sqrt(squared / max(referenceSquared, Double.leastNormalMagnitude))
            result["phase"] = "complete"; try save()
            var saved = ["input": input, "ane_fc": MLXArray(actual, [32, 5120]), "reference_fc": reference]
            if archivePath != nil {
                saved["input_tokens"] = MLXArray(tokenIDs)
                saved["positions"] = MLXArray((0..<32).map(Int32.init))
            }
            try MLX.save(arrays: saved, url: root.appendingPathComponent("checkpoint-input.safetensors"))
            if archivePath != nil {
                try #require(expected.allSatisfy { $0.isFinite })
                try #require(sqrt(squared / max(referenceSquared, Double.leastNormalMagnitude)) <= 0.002)
            } else {
                try #require(maximum < 0.01)
            }
        } catch {
            result["error"] = String(describing: error); try save(); throw error
        }
    }
}
