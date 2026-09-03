import MLX
import MLXFastCore
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXFastModel
@testable import MLXLLM

final class Qwen4ExpTransformTests: XCTestCase {
    /// Writes a 2-layer tiny source checkpoint in the HF layout (bf16, nested text_config).
    /// hidden_size is 32 so the expert input dim quantizes at group size 32; linear head
    /// dims are 32 (the vendored gated-delta kernel's floor).
    static func writeTinySource(to dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let L = "model.language_model.layers"
        let H = 32  // hidden_size
        let HC = 2 * H  // hc_count 2
        func bf(_ shape: [Int]) -> MLXArray { MLXRandom.normal(shape).asType(.bfloat16) }
        func zeros(_ n: Int) -> MLXArray { MLXArray.zeros([n]).asType(.bfloat16) }
        var t: [String: MLXArray] = [
            "model.language_model.embed_tokens.weight": bf([40, H]),
            "lm_head.weight": bf([40, H]),
            "model.language_model.hyper_connection_mixer.hc_norm.weight": zeros(HC),
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": bf([4, HC]),
            "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": bf([HC, 4]),
            "model.visual.blocks.0.attn.qkv.weight": MLXArray.zeros([3, 3]).asType(.bfloat16),
            // The reference rebuilds these three from the config seed formula and
            // ignores what the checkpoint stores, so the tiny source has to store
            // exactly what it computes or the parity comparison is meaningless.
            // Values printed by tools/qwen38-flash/qwen4_exp.py NGramEmbedding for
            // this config: ngram_vocab_size_base 8 gives the first four primes
            // after 7, and the multipliers depend only on seed and vocab_size.
            "\(L).1.ple.ple_embedding.layer_multipliers":
                MLXArray([Int64(10_256_280_814_223_215), 52_896_835_257_613_783, 78_656_110_748_828_819]),
            "\(L).1.ple.ple_embedding.ngram_heads_vocab_sizes": MLXArray([Int64(11), 13, 17, 19]),
            "\(L).1.ple.ple_embedding.ngram_heads_offsets": MLXArray([Int64(0), 11, 24, 41]),
            "\(L).1.ple.key_proj.weight": bf([HC, 8]),
            "\(L).1.ple.value_proj.weight": bf([H, 8]),
            "\(L).1.ple.norm_key.weight": zeros(HC),
            "\(L).1.ple.norm_query.weight": zeros(HC),
            "\(L).1.ple.norm_conv.weight": zeros(HC),
            "\(L).1.ple.conv1d.weight": bf([HC, 1, 4]),
        ]
        for s in 0 ..< 2 {
            // 4 heads sized [11, 13, 17, 19] total 60 rows, padded to 60 and split
            // across split_ngram_parts = 2, so 30 rows per shard at head dim 2.
            t["\(L).1.ple.ple_embedding.ngram_embedding.shard_\(s).weight"] = bf([30, 2])
        }
        // keyDim = 2*32 = 64, valueDim = 4*32 = 128, convDim = 256
        for l in 0 ..< 2 {
            for hc in ["attn_hyper_connection", "mlp_hyper_connection"] {
                t["\(L).\(l).\(hc).hc_norm.weight"] = zeros(HC)
                t["\(L).\(l).\(hc).input_mix_weight_down.weight"] = bf([4, HC])
                t["\(L).\(l).\(hc).input_mix_weight_up.weight"] = bf([HC, 4])
                t["\(L).\(l).\(hc).block_inject_weight.weight"] = bf([2, HC])
            }
            t["\(L).\(l).mlp.gate.weight"] = bf([4, H])
            t["\(L).\(l).mlp.experts.gate_up_proj"] = bf([4, 64, H])
            t["\(L).\(l).mlp.experts.down_proj"] = bf([4, H, 32])
            t["\(L).\(l).mlp.shared_expert.gate_proj.weight"] = bf([8, H])
            t["\(L).\(l).mlp.shared_expert.up_proj.weight"] = bf([8, H])
            t["\(L).\(l).mlp.shared_expert.down_proj.weight"] = bf([H, 8])
            t["\(L).\(l).mlp.shared_expert_gate.weight"] = bf([1, H])
            t["\(L).\(l).linear_attn.A_log"] = zeros(4)
            t["\(L).\(l).linear_attn.dt_bias"] = MLXArray.ones([4]).asType(.bfloat16)
            t["\(L).\(l).linear_attn.conv1d.weight"] = bf([256, 1, 4])
            t["\(L).\(l).linear_attn.in_proj_qkv.weight"] = bf([256, H])
            t["\(L).\(l).linear_attn.in_proj_z.weight"] = bf([128, H])
            t["\(L).\(l).linear_attn.in_proj_b.weight"] = bf([4, H])
            t["\(L).\(l).linear_attn.in_proj_a.weight"] = bf([4, H])
            t["\(L).\(l).linear_attn.norm.weight"] = MLXArray.ones([32]).asType(.bfloat16)
            t["\(L).\(l).linear_attn.out_proj.weight"] = bf([H, 128])
        }
        let keys = t.keys.sorted()
        let a = Dictionary(uniqueKeysWithValues: keys.prefix(keys.count / 2).map { ($0, t[$0]!) })
        let b = Dictionary(uniqueKeysWithValues: keys.suffix(from: keys.count / 2).map { ($0, t[$0]!) })
        try MLX.save(arrays: a, url: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        try MLX.save(arrays: b, url: dir.appendingPathComponent("model-00002-of-00002.safetensors"))
        var weightMap = [String: String]()
        for k in a.keys { weightMap[k] = "model-00001-of-00002.safetensors" }
        for k in b.keys { weightMap[k] = "model-00002-of-00002.safetensors" }
        let index: [String: Any] = ["metadata": ["total_size": 0], "weight_map": weightMap]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        let config = """
            {"architectures":["Qwen4ExpForConditionalGeneration"],"model_type":"qwen4_exp","text_config":{
             "model_type":"qwen4_exp_text","hidden_size":32,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
             "head_dim":8,"vocab_size":40,"rms_norm_eps":1e-6,"full_attention_interval":4,
             "layer_types":["linear_attention","linear_attention"],"num_experts":4,"num_experts_per_tok":2,
             "moe_intermediate_size":32,"shared_expert_intermediate_size":8,"linear_num_key_heads":2,"linear_num_value_heads":4,
             "linear_key_head_dim":32,"linear_value_head_dim":32,"linear_conv_kernel_dim":4,"output_gate_type":"sigmoid",
             "hc_count":2,"hc_lowrank":4,"indexer_n_heads":2,"indexer_kv_heads":1,"indexer_head_dim":4,"indexer_budget":64,
             "indexer_compress_ratio":2,"ngram_size":3,"heads_per_ngram":2,"split_ngram_parts":2,"ple_embed_dim":8,
             "ngram_vocab_size_base":8,"make_ngram_vocab_size_divisible_by":2,
             "ple_layer_ids":[2],"ple_conv_kernel_size":4,"eos_token_id":7,"partial_rotary_factor":0.5,
             "rope_parameters":{"rope_theta":10000,"rope_type":"default"},"tie_word_embeddings":false,"mtp_num_hidden_layers":0},
             "vision_config":{"depth":1}}
            """
        try Data(config.utf8).write(to: dir.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
        try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer_config.json"))
    }

    func testTransformProducesLoadableTree() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen4exp-transform-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let src = root.appendingPathComponent("source")
        let dst = root.appendingPathComponent("weights")
        try Self.writeTinySource(to: src)
        try Qwen4ExpTransform.run(
            .init(source: src, destination: dst, expertGroupSize: 32, expertBits: 4, shardBytes: 32 << 10))

        // config
        let cfg =
            try JSONSerialization.jsonObject(with: Data(contentsOf: dst.appendingPathComponent("config.json")))
            as! [String: Any]
        XCTAssertEqual(cfg["model_type"] as? String, "qwen4_exp_text")
        XCTAssertEqual((cfg["quantization"] as? [String: Any])?["group_size"] as? Int, 32)
        XCTAssertEqual((cfg["ngram_table"] as? [String: Any])?["rows_per_shard"] as? Int, 30)
        XCTAssertNil(cfg["vision_config"])
        XCTAssertNil(cfg["text_config"])
        // n-gram shards
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: dst.appendingPathComponent("ngram/shard_001.safetensors").path))
        // index and shard contents
        let index =
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: dst.appendingPathComponent("model.safetensors.index.json")))
            as! [String: Any]
        let wm = index["weight_map"] as! [String: String]
        XCTAssertNotNil(wm["model.layers.0.mlp.switch_mlp.gate_proj.scales"])
        XCTAssertNotNil(wm["model.layers.0.linear_attn.in_proj_qkv.weight"])
        XCTAssertNotNil(wm["model.layers.1.ple.ple_embedding.layer_multipliers"])
        XCTAssertNil(wm["model.visual.blocks.0.attn.qkv.weight"])
        XCTAssertNil(wm.keys.first { $0.contains("ngram_embedding.shard_") })
        XCTAssertTrue(Set(wm.values).count >= 2, "32 KiB shards should split the tiny tree")

        // dense tensors are byte-identical; experts dequantize close to the source
        var source = try MLX.loadArrays(url: src.appendingPathComponent("model-00001-of-00002.safetensors"))
        source.merge(try MLX.loadArrays(url: src.appendingPathComponent("model-00002-of-00002.safetensors"))) {
            a, _ in a
        }
        var loaded = [String: MLXArray]()
        for f in Set(wm.values) {
            loaded.merge(try MLX.loadArrays(url: dst.appendingPathComponent(f))) { a, _ in a }
        }
        let dense = source["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]!
        XCTAssertTrue(MLX.arrayEqual(dense, loaded["model.layers.0.linear_attn.in_proj_qkv.weight"]!).item())
        XCTAssertEqual(loaded["model.layers.0.linear_attn.conv1d.weight"]!.shape, [256, 4, 1])
        XCTAssertEqual(loaded["model.layers.0.mlp.switch_mlp.gate_proj.weight"]!.dtype, .uint32)
        let deq = dequantized(
            loaded["model.layers.0.mlp.switch_mlp.down_proj.weight"]!,
            scales: loaded["model.layers.0.mlp.switch_mlp.down_proj.scales"]!,
            biases: loaded["model.layers.0.mlp.switch_mlp.down_proj.biases"]!, groupSize: 32, bits: 4)
        let want = source["model.language_model.layers.0.mlp.experts.down_proj"]!.asType(.float32)
        XCTAssertLessThan((abs(deq.asType(.float32) - want).mean() / abs(want).mean()).item(Float.self), 0.2)

        // the tree loads the way the factory loads it, and runs
        Qwen4ExpRuntime.weightsDirectory = dst
        let cfgData = try Data(contentsOf: dst.appendingPathComponent("config.json"))
        let model = Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: cfgData))
        var weights = model.sanitize(weights: loaded)
        quantize(model: model) { path, _ in weights["\(path).scales"] != nil ? (32, 4, .affine) : nil }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        weights.removeAll()
        let logits = model(MLXArray([Int32(1), 2, 3]).reshaped(1, 3), cache: model.newCache(parameters: nil))
        XCTAssertEqual(logits.shape, [1, 3, 40])
        XCTAssertTrue((abs(logits.asType(.float32)) .< MLXArray(Float.infinity)).all().item())
    }
}
