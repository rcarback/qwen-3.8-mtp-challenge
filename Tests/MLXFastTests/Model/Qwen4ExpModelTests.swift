import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen4ExpModelTests: XCTestCase {
    // linear head dims are 32: the vendored gated-delta kernel's floor.
    static let tinyJSON = """
        {"model_type":"qwen4_exp_text","hidden_size":16,"num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,
         "head_dim":8,"vocab_size":40,"rms_norm_eps":1e-6,"full_attention_interval":4,"num_experts":4,"num_experts_per_tok":2,
         "moe_intermediate_size":8,"shared_expert_intermediate_size":8,"linear_num_key_heads":2,"linear_num_value_heads":4,
         "linear_key_head_dim":32,"linear_value_head_dim":32,"linear_conv_kernel_dim":4,"output_gate_type":"sigmoid",
         "hc_count":2,"hc_lowrank":4,"indexer_n_heads":2,"indexer_kv_heads":1,"indexer_head_dim":4,"indexer_budget":64,
         "indexer_compress_ratio":2,"ngram_size":3,"heads_per_ngram":2,"split_ngram_parts":2,"ple_embed_dim":8,
         "ple_layer_ids":[2],"ple_conv_kernel_size":4,"eos_token_id":7,"partial_rotary_factor":0.5,"rope_theta":10000,
         "tie_word_embeddings":false,"ngram_table":{"directory":".","shards":2,"rows_per_shard":16,"dim":2,"dtype":"bfloat16"}}
        """

    func makeModel() throws -> (Qwen4ExpModel, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen4exp-model-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for s in 0 ..< 2 {
            try MLX.save(
                arrays: ["weight": MLXRandom.normal([16, 2]).asType(.bfloat16)],
                url: dir.appendingPathComponent(String(format: "shard_%03d.safetensors", s)))
        }
        Qwen4ExpRuntime.weightsDirectory = dir
        let cfg = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: Data(Self.tinyJSON.utf8))
        let model = Qwen4ExpModel(cfg)
        model.update(
            parameters: ModuleParameters.unflattened([
                "model.layers.1.ple.ple_embedding.layer_multipliers": MLXArray([Int64(3), 5, 7]),
                "model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes": MLXArray([Int64(7), 7, 8, 8]),
                "model.layers.1.ple.ple_embedding.ngram_heads_offsets": MLXArray([Int64(0), 7, 14, 22]),
            ]))
        return (model, dir)
    }

    func testForwardShapesAndCacheLayout() throws {
        let (model, dir) = try makeModel()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 4)
        XCTAssertTrue(cache[0] is MambaCache)
        XCTAssertTrue(cache[1] is ArraysCache)
        XCTAssertFalse(cache[1] is MambaCache)  // PLE layer: 4 slots
        XCTAssertTrue(cache[3] is Qwen4ExpAttnCache)
        let ids = MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5)
        let logits = model(ids, cache: cache)
        XCTAssertEqual(logits.shape, [1, 5, 40])
        XCTAssertTrue((abs(logits.asType(.float32)) .< MLXArray(Float.infinity)).all().item())
        XCTAssertEqual(model.lastWideResidual?.shape, [1, 5, 32])
        XCTAssertEqual(cache[0].offset, 5)
        XCTAssertEqual(cache[3].offset, 5)
        XCTAssertEqual((cache[1] as! ArraysCache)[3]!.asArray(Int32.self), [4, 5])  // n-gram context
    }

    func testStepwiseMatchesPrefill() throws {
        let (model, dir) = try makeModel()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ids = MLXArray([Int32(1), 2, 3, 7, 5, 6]).reshaped(1, 6)  // EOS (7) inside resets the n-gram context
        let full = model(ids, cache: model.newCache(parameters: nil))
        let cache = model.newCache(parameters: nil)
        var outs = [MLXArray]()
        for t in 0 ..< 6 {
            outs.append(model(ids[0..., t ..< (t + 1)], cache: cache))
        }
        XCTAssertTrue(allClose(concatenated(outs, axis: 1), full, rtol: 2e-2, atol: 2e-3).item())
    }

    func testSanitizeRenamesAndSplitsExperts() throws {
        let (model, dir) = try makeModel()
        defer { try? FileManager.default.removeItem(at: dir) }
        let src: [String: MLXArray] = [
            "model.language_model.layers.0.mlp.experts.gate_up_proj": MLXArray.zeros([4, 16, 16]),
            "model.language_model.layers.0.mlp.experts.down_proj": MLXArray.zeros([4, 16, 8]),
            "model.language_model.layers.0.linear_attn.conv1d.weight": MLXArray.zeros([24, 1, 4]),
            "model.language_model.embed_tokens.weight": MLXArray.zeros([40, 16]),
            "model.visual.blocks.0.attn.qkv.weight": MLXArray.zeros([2, 2]),
            "mtp.fc_hidden.weight": MLXArray.zeros([16, 16]),
            "lm_head.weight": MLXArray.zeros([40, 16]),
        ]
        let out = model.sanitize(weights: src)
        XCTAssertEqual(out["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 8, 16])
        XCTAssertEqual(out["model.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape, [4, 8, 16])
        XCTAssertEqual(out["model.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 16, 8])
        XCTAssertEqual(out["model.layers.0.linear_attn.conv1d.weight"]?.shape, [24, 4, 1])
        XCTAssertNotNil(out["model.embed_tokens.weight"])
        XCTAssertNotNil(out["lm_head.weight"])
        XCTAssertNil(out["model.visual.blocks.0.attn.qkv.weight"])
        XCTAssertNil(out["mtp.fc_hidden.weight"])  // dropped until the MTP task
        let passthrough = model.sanitize(weights: [
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": MLXArray.zeros([4, 8, 16])
        ])
        XCTAssertNotNil(passthrough["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
    }

    func testRegistryKnowsTheFamily() async throws {
        let cfgData = Data(Self.tinyJSON.utf8)
        let known = await LLMTypeRegistry.shared.contains("qwen4_exp")
        XCTAssertTrue(known)
        let m1 = try await LLMTypeRegistry.shared.createModel(configuration: cfgData, modelType: "qwen4_exp_text")
        XCTAssertTrue(m1 is Qwen4ExpModel)
        let m2 = try await LLMTypeRegistry.shared.createModel(configuration: cfgData, modelType: "qwen4_exp")
        XCTAssertTrue(m2 is Qwen4ExpModel)
    }
}
