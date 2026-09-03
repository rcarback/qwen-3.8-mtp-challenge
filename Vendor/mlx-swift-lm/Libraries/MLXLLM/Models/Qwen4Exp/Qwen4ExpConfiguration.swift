// Qwen3.8-Flash-Next (HF model_type qwen4_exp). Local fork port; reference:
// mlx-lm PR 1788 (mlx_lm/models/qwen4_exp.py).
import Foundation
import MLXLMCommon

public struct Qwen4ExpNGramTableSpec: Codable, Sendable {
    public var directory: String
    public var shards: Int
    public var rowsPerShard: Int
    public var dim: Int
    public var dtype: String

    enum CodingKeys: String, CodingKey {
        case directory, shards, dim, dtype
        case rowsPerShard = "rows_per_shard"
    }

    public init(directory: String, shards: Int, rowsPerShard: Int, dim: Int, dtype: String) {
        self.directory = directory
        self.shards = shards
        self.rowsPerShard = rowsPerShard
        self.dim = dim
        self.dtype = dtype
    }
}

public struct Qwen4ExpTextConfiguration: Codable, Sendable {
    public var modelType: String = "qwen4_exp_text"
    public var hiddenSize: Int = 2560
    public var hiddenLayers: Int = 48
    public var attentionHeads: Int = 24
    public var kvHeads: Int = 2
    public var headDim: Int = 256
    public var vocabularySize: Int = 248320
    public var rmsNormEps: Float = 1e-6
    public var layerTypes: [String] = []
    public var fullAttentionInterval: Int = 4
    public var numExperts: Int = 512
    public var numExpertsPerTok: Int = 10
    public var moeIntermediateSize: Int = 640
    public var sharedExpertIntermediateSize: Int = 640
    public var linearNumKeyHeads: Int = 16
    public var linearNumValueHeads: Int = 48
    public var linearKeyHeadDim: Int = 128
    public var linearValueHeadDim: Int = 128
    public var linearConvKernelDim: Int = 4
    public var outputGateType: String = "sigmoid"
    public var hcCount: Int = 4
    public var hcLowrank: Int = 320
    public var indexerNHeads: Int = 4
    public var indexerKVHeads: Int = 1
    public var indexerHeadDim: Int = 128
    public var indexerBudget: Int = 2048
    public var indexerCompressRatio: Int = 4
    public var ngramSize: Int = 3
    public var headsPerNgram: Int = 8
    public var ngramVocabSizeBase: Int = 20_000_000
    public var makeNgramVocabSizeDivisibleBy: Int = 128
    public var splitNgramParts: Int = 128
    public var pleEmbedDim: Int = 2560
    public var pleLayerIds: [Int] = [2]
    public var pleConvKernelSize: Int = 4
    public var eosTokenId: Int = 248044
    public var partialRotaryFactor: Float = 0.25
    public var ropeTheta: Float = 10_000_000
    public var tieWordEmbeddings: Bool = false
    public var mtpNumHiddenLayers: Int = 0
    public var ngramTable: Qwen4ExpNGramTableSpec? = nil

    // Derived.
    public var hcDim: Int { hcCount * hiddenSize }
    public var rotaryDims: Int { Int(Float(headDim) * partialRotaryFactor) }
    public var keyDim: Int { linearNumKeyHeads * linearKeyHeadDim }
    public var valueDim: Int { linearNumValueHeads * linearValueHeadDim }
    public var convDim: Int { keyDim * 2 + valueDim }
    public var ngramHeads: Int { (ngramSize - 1) * headsPerNgram }
    public var ngramHeadDim: Int { pleEmbedDim / ngramHeads }
    public var blockTopK: Int { indexerBudget / indexerCompressRatio }
    /// `ple_layer_ids` are 1-based in the checkpoint (`[2]` means layer index 1).
    public var pleLayerIndices: [Int] { (0 ..< hiddenLayers).filter { pleLayerIds.contains($0 + 1) } }
    public func isLinear(layer: Int) -> Bool { layerTypes[layer] == "linear_attention" }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case layerTypes = "layer_types"
        case fullAttentionInterval = "full_attention_interval"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case outputGateType = "output_gate_type"
        case hcCount = "hc_count"
        case hcLowrank = "hc_lowrank"
        case indexerNHeads = "indexer_n_heads"
        case indexerKVHeads = "indexer_kv_heads"
        case indexerHeadDim = "indexer_head_dim"
        case indexerBudget = "indexer_budget"
        case indexerCompressRatio = "indexer_compress_ratio"
        case ngramSize = "ngram_size"
        case headsPerNgram = "heads_per_ngram"
        case ngramVocabSizeBase = "ngram_vocab_size_base"
        case makeNgramVocabSizeDivisibleBy = "make_ngram_vocab_size_divisible_by"
        case splitNgramParts = "split_ngram_parts"
        case pleEmbedDim = "ple_embed_dim"
        case pleLayerIds = "ple_layer_ids"
        case pleConvKernelSize = "ple_conv_kernel_size"
        case eosTokenId = "eos_token_id"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
        case ngramTable = "ngram_table"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func opt<T: Decodable>(_ k: CodingKeys, _ d: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? d
        }
        modelType = try opt(.modelType, modelType)
        hiddenSize = try opt(.hiddenSize, hiddenSize)
        hiddenLayers = try opt(.hiddenLayers, hiddenLayers)
        attentionHeads = try opt(.attentionHeads, attentionHeads)
        kvHeads = try opt(.kvHeads, kvHeads)
        headDim = try opt(.headDim, headDim)
        vocabularySize = try opt(.vocabularySize, vocabularySize)
        rmsNormEps = try opt(.rmsNormEps, rmsNormEps)
        layerTypes = try opt(.layerTypes, layerTypes)
        fullAttentionInterval = try opt(.fullAttentionInterval, fullAttentionInterval)
        numExperts = try opt(.numExperts, numExperts)
        numExpertsPerTok = try opt(.numExpertsPerTok, numExpertsPerTok)
        moeIntermediateSize = try opt(.moeIntermediateSize, moeIntermediateSize)
        sharedExpertIntermediateSize = try opt(.sharedExpertIntermediateSize, sharedExpertIntermediateSize)
        linearNumKeyHeads = try opt(.linearNumKeyHeads, linearNumKeyHeads)
        linearNumValueHeads = try opt(.linearNumValueHeads, linearNumValueHeads)
        linearKeyHeadDim = try opt(.linearKeyHeadDim, linearKeyHeadDim)
        linearValueHeadDim = try opt(.linearValueHeadDim, linearValueHeadDim)
        linearConvKernelDim = try opt(.linearConvKernelDim, linearConvKernelDim)
        outputGateType = try opt(.outputGateType, outputGateType)
        hcCount = try opt(.hcCount, hcCount)
        hcLowrank = try opt(.hcLowrank, hcLowrank)
        indexerNHeads = try opt(.indexerNHeads, indexerNHeads)
        indexerKVHeads = try opt(.indexerKVHeads, indexerKVHeads)
        indexerHeadDim = try opt(.indexerHeadDim, indexerHeadDim)
        indexerBudget = try opt(.indexerBudget, indexerBudget)
        indexerCompressRatio = try opt(.indexerCompressRatio, indexerCompressRatio)
        ngramSize = try opt(.ngramSize, ngramSize)
        headsPerNgram = try opt(.headsPerNgram, headsPerNgram)
        ngramVocabSizeBase = try opt(.ngramVocabSizeBase, ngramVocabSizeBase)
        makeNgramVocabSizeDivisibleBy = try opt(.makeNgramVocabSizeDivisibleBy, makeNgramVocabSizeDivisibleBy)
        splitNgramParts = try opt(.splitNgramParts, splitNgramParts)
        pleEmbedDim = try opt(.pleEmbedDim, pleEmbedDim)
        pleLayerIds = try opt(.pleLayerIds, pleLayerIds)
        pleConvKernelSize = try opt(.pleConvKernelSize, pleConvKernelSize)
        if let ids = try? c.decode([Int].self, forKey: .eosTokenId), let first = ids.first {
            eosTokenId = first
        } else {
            eosTokenId = try opt(.eosTokenId, eosTokenId)
        }
        partialRotaryFactor = try opt(.partialRotaryFactor, partialRotaryFactor)
        ropeTheta = try opt(.ropeTheta, ropeTheta)
        if let rp = try c.decodeIfPresent([String: Qwen4ExpJSONNumber].self, forKey: .ropeParameters) {
            if let t = rp["rope_theta"]?.value { ropeTheta = Float(t) }
            if let f = rp["partial_rotary_factor"]?.value { partialRotaryFactor = Float(f) }
        }
        tieWordEmbeddings = try opt(.tieWordEmbeddings, tieWordEmbeddings)
        mtpNumHiddenLayers = try opt(.mtpNumHiddenLayers, mtpNumHiddenLayers)
        ngramTable = try c.decodeIfPresent(Qwen4ExpNGramTableSpec.self, forKey: .ngramTable)
        if layerTypes.isEmpty {
            layerTypes = (0 ..< hiddenLayers).map {
                ($0 + 1) % fullAttentionInterval == 0 ? "full_attention" : "linear_attention"
            }
        }
        precondition(layerTypes.count == hiddenLayers, "layer_types must list every layer")
        precondition(pleEmbedDim % ngramHeads == 0, "ple_embed_dim must divide by the n-gram head count")
    }

    public func encode(to encoder: Encoder) throws {
        // The runtime never encodes; the transform writes config.json by hand.
        throw EncodingError.invalidValue(
            self, .init(codingPath: [], debugDescription: "Qwen4ExpTextConfiguration is not encodable"))
    }
}

/// `rope_parameters` mixes numbers and strings (`rope_type`, `mrope_section`); decode numbers only.
struct Qwen4ExpJSONNumber: Decodable, Sendable {
    let value: Double?
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        value = try? c.decode(Double.self)
    }
}

public struct Qwen4ExpConfiguration: Codable, Sendable {
    public var modelType: String
    public var textConfig: Qwen4ExpTextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        if let t = try c.decodeIfPresent(Qwen4ExpTextConfiguration.self, forKey: .textConfig) {
            textConfig = t
        } else {
            textConfig = try Qwen4ExpTextConfiguration(from: decoder)
        }
    }

    public func encode(to encoder: Encoder) throws {
        throw EncodingError.invalidValue(
            self, .init(codingPath: [], debugDescription: "Qwen4ExpConfiguration is not encodable"))
    }
}
