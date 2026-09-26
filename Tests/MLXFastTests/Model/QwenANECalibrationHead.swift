import CryptoKit
import Foundation
import MLX
import Testing
@testable import MLXLLM

final class QwenANECalibrationHead {
    let embedding: QwenANEEmbedding
    private let coarseBlobs: [Int: Data]
    let shortlist: QwenANEShortlistSelector
    private let targetBlobs: [Int: Data]
    private let input: ANEDirectDispatch.FP16Loop
    private let qkv: ANEDirectDispatch.FP16Loop
    private let qk: ANEDirectDispatch.FP16Loop
    private let output: ANEDirectDispatch.FP16Loop
    private let tail: ANEDirectDispatch.FP16Loop

    init(headURL: URL, targetURL: URL, progress: (String, Int) throws -> Void) throws {
        let head = try MLX.loadArrays(url: headURL)
        func norm(_ key: String, _ count: Int) throws -> [Float16] {
            let result = try #require(head[key]).asArray(Float16.self)
            try #require(result.count == count && result.allSatisfy { $0.isFinite })
            return result
        }
        func expand(_ prefix: String, bits: Int = 4) throws -> MLXArray {
            try aneProbeAffineWeight(#require(head[prefix + ".weight"]), scales: #require(head[prefix + ".scales"]),
                biases: #require(head[prefix + ".biases"]), groupSize: 64, bits: bits)
        }
        func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
        let fc = try expand("fc")
        try #require(fc.shape == [5120, 10240])
        let fcBlob = buildConvWeightBlob(f16Bytes(fc.contiguous()))
        let norms = try norm("pre_fc_norm_embedding.weight", 5120) + norm("pre_fc_norm_hidden.weight", 5120)
        input = try ANEDirectDispatch.FP16Loop(milText: buildQwenANEInputMILText(normWeights: norms, programTag: hash(fcBlob)),
            weightBlob: fcBlob, inputDim: 10240, outputDim: 5120, batch: 32)
        let qkvWeight = try concatenated(["q", "k", "v"].map { try expand("layers.0.self_attn.\($0)_proj") }, axis: 0)
        try #require(qkvWeight.shape == [14336, 5120])
        let qkvBlob = buildConvWeightBlob(f16Bytes(qkvWeight.contiguous()))
        qkv = try ANEDirectDispatch.FP16Loop(milText: buildQwenANEProjectionMILText(norm: norm("layers.0.input_layernorm.weight", 5120), programTag: hash(qkvBlob)),
            weightBlob: qkvBlob, inputDim: 5120, outputDim: 14336, batch: 32)
        let qNorm = try norm("layers.0.self_attn.q_norm.weight", 256), kNorm = try norm("layers.0.self_attn.k_norm.weight", 256)
        qk = try ANEDirectDispatch.FP16Loop(milText: buildQwenANEQKMILText(weights: (0..<28).flatMap { $0 < 24 ? qNorm : kNorm }),
            weightBlob: Data(), inputDim: 7169, outputDim: 7168, batch: 32)
        let oWeight = try expand("layers.0.self_attn.o_proj")
        try #require(oWeight.shape == [5120, 6144])
        let oBlob = buildConvWeightBlob(f16Bytes(oWeight.contiguous()))
        output = try ANEDirectDispatch.FP16Loop(milText: buildQwenANEGatedOutputMILText(programTag: hash(oBlob)),
            weightBlob: oBlob, inputDim: 12288, outputDim: 5120, batch: 32)
        let gate = try expand("layers.0.mlp.gate_proj"), up = try expand("layers.0.mlp.up_proj"), down = try expand("layers.0.mlp.down_proj")
        try #require(gate.shape == [17408, 5120] && up.shape == gate.shape && down.shape == [5120, 17408])
        let (tailBlob, offsets) = buildMultiWeightBlob(chunks: [f16Bytes(gate.contiguous()), f16Bytes(up.contiguous()), f16Bytes(down.contiguous())])
        tail = try ANEDirectDispatch.FP16Loop(milText: buildQwenANEDecoderTailMILText(offsets: offsets,
            postNorm: norm("layers.0.post_attention_layernorm.weight", 5120), finalNorm: norm("norm.weight", 5120), programTag: hash(tailBlob)),
            weightBlob: tailBlob, inputDim: 10240, outputDim: 5120, batch: 32, weightFileName: "weight.bin")
        let index = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: targetURL.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any])
        let mapping = try #require(index["weight_map"] as? [String: String])
        func targetArrays(_ prefix: String) throws -> (MLXArray, MLXArray, MLXArray) {
            let shard = try #require(mapping[prefix + ".weight"])
            try #require(mapping[prefix + ".scales"] == shard && mapping[prefix + ".biases"] == shard)
            let arrays = try MLX.loadArrays(url: targetURL.appendingPathComponent(shard))
            let packed = try #require(arrays[prefix + ".weight"]), scales = try #require(arrays[prefix + ".scales"]), biases = try #require(arrays[prefix + ".biases"])
            try #require(packed.shape == [248320, 640] && scales.shape == [248320, 80] && biases.shape == scales.shape)
            return (packed, scales, biases)
        }
        var embeddingBlobs: [Int: Data] = [:]
        let (ep, es, eb) = try targetArrays("language_model.model.embed_tokens")
        for tile in 0..<122 {
            let start = tile * 2048, end = min(start + 2048, 248320)
            var weight = aneProbeAffineWeight(ep[start..<end], scales: es[start..<end], biases: eb[start..<end], groupSize: 64, bits: 4)
            if end - start < 2048 { weight = concatenated([weight, MLXArray.zeros([2048 - (end - start), 5120], dtype: .float16)], axis: 0) }
            embeddingBlobs[tile] = buildConvWeightBlob(f16Bytes(weight.T.contiguous()))
            try progress("embedding-prepare", tile + 1)
        }
        embedding = try QwenANEEmbedding(tileWeightBlobs: embeddingBlobs, maxLoadedTiles: 8)
        var coarseBlobs: [Int: Data] = [:], targetBlobs: [Int: Data] = [:]
        let cp = try #require(head["draft_lm_head.weight"]), cs = try #require(head["draft_lm_head.scales"]), cb = try #require(head["draft_lm_head.biases"])
        try #require(cp.shape == [98336, 320] && cs.shape == [98336, 80] && cb.shape == cs.shape)
        let (tp, ts, tb) = try targetArrays("language_model.lm_head")
        for start in stride(from: 0, to: 98336, by: 2048) {
            let end = min(start + 2048, 98336)
            let coarse = aneProbeAffineWeight(cp[start..<end], scales: cs[start..<end], biases: cb[start..<end], groupSize: 64, bits: 2)
            coarseBlobs[start] = buildConvWeightBlob(f16Bytes(coarse.contiguous()))
            let begin = start < 98304 ? start : 248044, finish = start < 98304 ? start + 2048 : 248070
            var exact = aneProbeAffineWeight(tp[begin..<finish], scales: ts[begin..<finish], biases: tb[begin..<finish], groupSize: 64, bits: 4)
            if start == 98304 { exact = concatenated([exact, MLXArray.zeros([6, 5120], dtype: .float16)], axis: 0) }
            targetBlobs[start] = buildConvWeightBlob(f16Bytes(exact.contiguous()))
            try progress("readout-prepare", coarseBlobs.count)
        }
        try progress("readout-ready", 0)
        self.coarseBlobs = coarseBlobs
        self.targetBlobs = targetBlobs
        shortlist = try QwenANEShortlistSelector()
    }

    func makeDecoder() throws -> QwenANEDecoder {
        try QwenANEDecoder(inputProjection: input, qkvProjection: qkv, qkPreparation: qk, outputProjection: output, tail: tail)
    }

    func pack(tokens: [Int], hidden: [Float16]) throws -> [Float16] {
        try #require(tokens.count == 32 && hidden.count == 32 * 5120)
        let vectors = try embedding(tokens: tokens)
        var input: [Float16] = []
        for lane in 0..<32 {
            let range = (lane * 5120)..<((lane + 1) * 5120)
            input.append(contentsOf: vectors[range]); input.append(contentsOf: hidden[range])
        }
        return input
    }

    func predict(_ hidden: [Float16]) throws -> [Int32] {
        let selected = try autoreleasepool {
            let readout = try QwenANECoarseReadout(tileWeightBlobs: coarseBlobs)
            return try shortlist(readout(hidden))
        }
        return try autoreleasepool {
            let rerank = try QwenANEReranker(targetTileWeightBlobs: targetBlobs)
            return try rerank(hidden: hidden, compactIDs: selected.ids).tokens
        }
    }
}
