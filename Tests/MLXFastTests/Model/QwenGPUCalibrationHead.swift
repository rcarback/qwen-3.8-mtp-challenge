import Foundation
import MLX
import Testing
@testable import MLXLLM

final class QwenGPUCalibrationHead {
    let fc: MLXArray, qkv: MLXArray, output: MLXArray
    let gate: MLXArray, up: MLXArray, down: MLXArray
    let preNorm: MLXArray, inputNorm: MLXArray, qkNorm: MLXArray
    let postNorm: MLXArray, finalNorm: MLXArray
    private let embedding: MLXArray, coarse: MLXArray, target: MLXArray

    init(headURL: URL, targetURL: URL, normOverrideURL: URL? = nil, progress: (String, Int) throws -> Void) throws {
        let head = try MLX.loadArrays(url: headURL)
        func norm(_ key: String, _ count: Int) throws -> MLXArray {
            let value = try #require(head[key]).asType(.float16)
            try #require(value.shape == [count] && all(isFinite(value)).item(Bool.self))
            return value
        }
        func expand(_ key: String, _ shape: [Int], bits: Int = 4) throws -> MLXArray {
            let value = try aneProbeAffineWeight(#require(head[key + ".weight"]),
                scales: #require(head[key + ".scales"]), biases: #require(head[key + ".biases"]), groupSize: 64, bits: bits)
            try #require(value.shape == shape)
            eval(value)
            return value
        }
        fc = try expand("fc", [5120, 10240])
        qkv = try concatenated([expand("layers.0.self_attn.q_proj", [12288, 5120]),
            expand("layers.0.self_attn.k_proj", [1024, 5120]), expand("layers.0.self_attn.v_proj", [1024, 5120])], axis: 0)
        output = try expand("layers.0.self_attn.o_proj", [5120, 6144])
        gate = try expand("layers.0.mlp.gate_proj", [17408, 5120])
        up = try expand("layers.0.mlp.up_proj", [17408, 5120])
        down = try expand("layers.0.mlp.down_proj", [5120, 17408])
        preNorm = try concatenated([norm("pre_fc_norm_embedding.weight", 5120), norm("pre_fc_norm_hidden.weight", 5120)], axis: 0).reshaped([2, 5120])
        inputNorm = try norm("layers.0.input_layernorm.weight", 5120)
        let qn = try norm("layers.0.self_attn.q_norm.weight", 256), kn = try norm("layers.0.self_attn.k_norm.weight", 256)
        qkNorm = stacked((0..<28).map { $0 < 24 ? qn : kn }, axis: 0)
        postNorm = try norm("layers.0.post_attention_layernorm.weight", 5120)
        if let normOverrideURL {
            let override = try MLX.loadArrays(url: normOverrideURL)
            try #require(Set(override.keys) == Set(["norm.weight"]))
            let raw = try #require(override["norm.weight"])
            try #require(raw.dtype == .bfloat16 && raw.shape == [5120])
            let value = raw.asType(.float16)
            try #require(all(isFinite(value)).item(Bool.self))
            finalNorm = value
        } else {
            finalNorm = try norm("norm.weight", 5120)
        }
        coarse = try expand("draft_lm_head", [98336, 5120], bits: 2)
        let index = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: targetURL.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any])
        let mapping = try #require(index["weight_map"] as? [String: String])
        func targetWeight(_ prefix: String, ranges: [Range<Int>]) throws -> MLXArray {
            let shard = try #require(mapping[prefix + ".weight"])
            try #require(mapping[prefix + ".scales"] == shard && mapping[prefix + ".biases"] == shard)
            let arrays = try MLX.loadArrays(url: targetURL.appendingPathComponent(shard))
            let packed = try #require(arrays[prefix + ".weight"]), scales = try #require(arrays[prefix + ".scales"]), biases = try #require(arrays[prefix + ".biases"])
            try #require(packed.shape == [248320, 640] && scales.shape == [248320, 80] && biases.shape == scales.shape)
            let value = concatenated(ranges.map { aneProbeAffineWeight(packed[$0], scales: scales[$0], biases: biases[$0], groupSize: 64, bits: 4) }, axis: 0)
            eval(value)
            return value
        }
        embedding = try targetWeight("language_model.model.embed_tokens", ranges: [0..<248320])
        try progress("embedding-ready", 122)
        target = try concatenated([targetWeight("language_model.lm_head", ranges: [0..<98304, 248044..<248070]),
            MLXArray.zeros([6, 5120], dtype: .float16)], axis: 0)
        eval(fc, qkv, output, gate, up, down, preNorm, inputNorm, qkNorm, postNorm, finalNorm, coarse, target)
        try progress("readout-ready", 49)
    }

    func pack(tokens: [Int], hidden: [Float16]) throws -> MLXArray {
        guard tokens.count == 32, tokens.allSatisfy({ (0..<248320).contains($0) }),
              hidden.count == 32 * 5120, hidden.allSatisfy({ $0.isFinite }) else {
            throw QwenGPUCompatibleError.invalidInput
        }
        let vectors = take(embedding, MLXArray(tokens.map(Int32.init)), axis: 0)
        return concatenated([vectors, MLXArray(hidden, [32, 5120])], axis: 1)
    }

    func makeDecoder() throws -> QwenGPUCalibrationDecoder { try QwenGPUCalibrationDecoder(head: self) }

    func predict(_ hidden: [Float16]) throws -> [Int32] {
        guard hidden.count == 32 * 5120, hidden.allSatisfy({ $0.isFinite }) else { throw QwenGPUCompatibleError.invalidInput }
        let x = MLXArray(hidden, [32, 5120])
        func projection(_ weight: MLXArray) -> MLXArray {
            concatenated(stride(from: 0, to: 98336, by: 2048).map {
                matmul(x, weight[$0..<min($0 + 2048, 98336)].T)
            }, axis: 1)
        }
        let ids = try QwenGPUCompatibleMath.shortlist(projection(coarse)[0..., 0..<98330])
        let scores = takeAlong(projection(target), ids, axis: 1)
        let tokenIDs = which(ids .< 98304, ids, ids + 149740).asType(.int32)
        return try QwenGPUCompatibleMath.winner(scores: scores, tokenIDs: tokenIDs).asArray(Int32.self)
    }
}

final class QwenGPUCalibrationDecoder {
    private let head: QwenGPUCalibrationHead
    private let cache: QwenGPUCompatibleCache
    private var valid = true
    private(set) var lastFinalNormInput: [Float16]?

    init(head: QwenGPUCalibrationHead) throws {
        self.head = head
        cache = try QwenGPUCompatibleCache()
    }

    func teacherCacheState() throws -> MLXArray {
        guard valid else { throw QwenGPUCompatibleError.invalidInput }
        return cache.state
    }

    func restoreTeacherPrefix(_ history: MLXArray, lengths: [Int]) throws {
        guard valid else { throw QwenGPUCompatibleError.invalidInput }
        try cache.restoreTeacherPrefix(history, lengths: lengths)
        lastFinalNormInput = nil
    }

    func callAsFunction(input: MLXArray, positions: [Int], slots: [Int], validLengths: [Int],
                        captureFinalNormInput: Bool = false) throws -> [Float16] {
        guard valid, input.shape == [32, 10240], positions.count == 32,
              positions.allSatisfy({ (0..<1024).contains($0) }) else { throw QwenGPUCompatibleError.invalidInput }
        do {
            let normalized = QwenGPUCompatibleMath.norm(input.reshaped([32, 2, 5120]), weight: head.preNorm).reshaped([32, 10240])
            let residual = matmul(normalized, head.fc.T)
            let qkv = matmul(QwenGPUCompatibleMath.norm(residual, weight: head.inputNorm), head.qkv.T)
            let queryGate = qkv[0..., 0..<12288].reshaped([32, 24, 512])
            let queries = queryGate[0..., 0..., 0..<256]
            let gates = queryGate[0..., 0..., 256..<512].reshaped([32, 6144])
            let keys = qkv[0..., 12288..<13312].reshaped([32, 4, 256])
            let values = qkv[0..., 13312..<14336].reshaped([32, 4, 256])
            let qk = try QwenGPUCompatibleMath.rotate(QwenGPUCompatibleMath.norm(
                concatenated([queries, keys], axis: 1), weight: head.qkNorm), positions: positions)
            let attention = try cache.step(newKV: concatenated([qk[0..., 24..<28, 0...], values], axis: 1),
                queries: qk[0..., 0..<24, 0...], slots: slots, validLengths: validLengths)
            let projected = matmul(attention / (exp(-gates) + 1), head.output.T)
            let afterAttention = residual + projected
            let mlpInput = QwenGPUCompatibleMath.norm(afterAttention, weight: head.postNorm)
            let gate = matmul(mlpInput, head.gate.T), up = matmul(mlpInput, head.up.T)
            let activated = (gate / (1 + exp(-gate))) * up
            let finalNormInput = afterAttention + matmul(activated, head.down.T)
            let output = QwenGPUCompatibleMath.norm(finalNormInput, weight: head.finalNorm)
            eval(output)
            guard all(isFinite(output)).item(Bool.self) else { throw QwenGPUCompatibleError.invalidInput }
            lastFinalNormInput = captureFinalNormInput ? finalNormInput.asType(.float16).asArray(Float16.self) : nil
            return output.asArray(Float16.self)
        } catch {
            valid = false
            lastFinalNormInput = nil
            throw error
        }
    }
}
