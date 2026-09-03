import XCTest

@testable import MLXLLM

final class Qwen4ExpConfigurationTests: XCTestCase {
    static let flat = """
        {"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,"num_attention_heads":24,
         "num_key_value_heads":2,"head_dim":256,"vocab_size":248320,"rms_norm_eps":1e-6,"full_attention_interval":4,
         "num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,
         "linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,
         "linear_conv_kernel_dim":4,"output_gate_type":"sigmoid","hc_count":4,"hc_lowrank":320,
         "indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4,
         "ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,
         "split_ngram_parts":128,"ple_embed_dim":2560,"ple_layer_ids":[2],"ple_conv_kernel_size":4,
         "eos_token_id":248044,"partial_rotary_factor":0.25,
         "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25,"rope_type":"default","mrope_section":[11,11,10]},
         "tie_word_embeddings":false,"mtp_num_hidden_layers":1,
         "ngram_table":{"directory":"ngram","shards":128,"rows_per_shard":2500012,"dim":160,"dtype":"bfloat16"}}
        """

    func testDecodesFlatTextConfig() throws {
        let c = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: Data(Self.flat.utf8))
        XCTAssertEqual(c.hiddenSize, 2560)
        XCTAssertEqual(c.hcDim, 10240)
        XCTAssertEqual(c.rotaryDims, 64)
        XCTAssertEqual(c.ropeTheta, 10_000_000)
        XCTAssertEqual(c.layerTypes.count, 48)
        XCTAssertTrue(c.isLinear(layer: 0))
        XCTAssertFalse(c.isLinear(layer: 3))
        XCTAssertFalse(c.isLinear(layer: 47))
        XCTAssertEqual(c.pleLayerIndices, [1])
        XCTAssertEqual(c.ngramHeads, 16)
        XCTAssertEqual(c.ngramHeadDim, 160)
        XCTAssertEqual(c.convDim, 10240)
        XCTAssertEqual(c.blockTopK, 512)
        XCTAssertEqual(c.ngramTable?.rowsPerShard, 2_500_012)
        XCTAssertEqual(c.mtpNumHiddenLayers, 1)
    }

    func testDecodesNestedOuterConfig() throws {
        let nested = "{\"model_type\":\"qwen4_exp\",\"text_config\":\(Self.flat)}"
        let c = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: Data(nested.utf8))
        XCTAssertEqual(c.modelType, "qwen4_exp")
        XCTAssertEqual(c.textConfig.numExperts, 512)
    }

    func testDecodesFlatOuterConfig() throws {
        let c = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: Data(Self.flat.utf8))
        XCTAssertEqual(c.modelType, "qwen4_exp_text")
        XCTAssertEqual(c.textConfig.hiddenLayers, 48)
    }

    func testDefaultsWhenLayerTypesAbsent() throws {
        let c = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: Data(Self.flat.utf8))
        XCTAssertEqual(c.layerTypes.filter { $0 == "full_attention" }.count, 12)
    }

    func testEOSListTakesFirstEntry() throws {
        let json = "{\"model_type\":\"qwen4_exp_text\",\"eos_token_id\":[248044,248045]}"
        let c = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(c.eosTokenId, 248044)
    }
}
