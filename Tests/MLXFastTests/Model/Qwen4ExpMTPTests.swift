import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXFastModel
@testable import MLXLLM

final class Qwen4ExpMTPTests: XCTestCase {
    func makeModelWithHead() throws -> (Qwen4ExpModel, URL) {
        let json = Qwen4ExpModelTests.tinyJSON.replacingOccurrences(
            of: "\"tie_word_embeddings\":false,", with: "\"tie_word_embeddings\":false,\"mtp_num_hidden_layers\":1,")
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen4exp-mtp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for s in 0 ..< 2 {
            try MLX.save(
                arrays: ["weight": MLXRandom.normal([16, 2]).asType(.bfloat16)],
                url: dir.appendingPathComponent(String(format: "shard_%03d.safetensors", s)))
        }
        Qwen4ExpRuntime.weightsDirectory = dir
        let cfg = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: Data(json.utf8))
        let model = Qwen4ExpModel(cfg)
        model.update(
            parameters: ModuleParameters.unflattened([
                "model.layers.1.ple.ple_embedding.layer_multipliers": MLXArray([Int64(3), 5, 7]),
                "model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes": MLXArray([Int64(7), 7, 8, 8]),
                "model.layers.1.ple.ple_embedding.ngram_heads_offsets": MLXArray([Int64(0), 7, 14, 22]),
            ]))
        return (model, dir)
    }

    func testHeadAttachesAndChains() throws {
        let (model, dir) = try makeModelWithHead()
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertTrue(model.hasMTPHead)
        XCTAssertNotNil(model.mtp)
        let keys = Set(model.parameters().flattened().map { $0.0 })
        for k in [
            "mtp.fc_embedding.weight", "mtp.fc_hidden.weight", "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight", "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.mlp.gate.weight", "mtp.hyper_connection_mixer.hc_norm.weight",
        ] {
            XCTAssertTrue(keys.contains(k), k)
        }
        XCTAssertFalse(keys.contains("mtp.layers.0.ple.key_proj.weight"))

        let ids = MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5)
        let cache = model.newCache(parameters: nil)
        _ = model(ids, cache: cache)
        let wide = model.lastWideResidual!
        XCTAssertEqual(wide.shape, [1, 5, 32])

        // draft step 1 on the trunk's wide residual with the next tokens' embeddings
        let next = MLXArray([Int32(2), 3, 4, 5, 6]).reshaped(1, 5)
        let mtpCache = model.makeMTPCache()
        XCTAssertEqual(mtpCache.count, 1)
        let (logits1, hidden1) = model.mtpForwardWithHidden(hidden: wide, nextTokenIds: next, cache: mtpCache)
        XCTAssertEqual(logits1.shape, [1, 5, 40])
        XCTAssertEqual(hidden1.shape, [1, 5, 32])
        XCTAssertTrue((abs(logits1.asType(.float32)) .< MLXArray(Float.infinity)).all().item())
        XCTAssertEqual(mtpCache[0].offset, 5)

        // draft step 2 chains on the head's own exported wide residual for one position
        let (logits2, hidden2) = model.mtpForwardWithHidden(
            hidden: hidden1[0..., 4 ..< 5, 0...], nextTokenIds: MLXArray([Int32(7)]).reshaped(1, 1), cache: mtpCache)
        XCTAssertEqual(logits2.shape, [1, 1, 40])
        XCTAssertEqual(hidden2.shape, [1, 1, 32])
        XCTAssertEqual(mtpCache[0].offset, 6)
    }

    func testSanitizeKeepsHeadTensorsWhenConfigured() throws {
        let (model, dir) = try makeModelWithHead()
        defer { try? FileManager.default.removeItem(at: dir) }
        let out = model.sanitize(weights: [
            "mtp.layers.0.mlp.experts.gate_up_proj": MLXArray.zeros([4, 16, 16]),
            "mtp.fc_hidden.weight": MLXArray.zeros([16, 16]),
        ])
        XCTAssertEqual(out["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 8, 16])
        XCTAssertNotNil(out["mtp.fc_hidden.weight"])
    }
}
