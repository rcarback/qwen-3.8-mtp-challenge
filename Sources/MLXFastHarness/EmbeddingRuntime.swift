import Foundation
import MLX
import MLXEmbedders
import MLXFastCore
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

extension RuntimeWorkerRequest {
    func validatedEmbeddingInput() throws -> EmbeddingWorkerInput {
        guard kind == "embedding_encode", let embeddingInput,
            promptTokens == nil, token == nil, seedTokens == nil, steps == nil,
            topK == nil, expectedToken == nil, maxBlockSize == nil,
            prefixTokens == nil, startOffset == nil, rowCount == nil,
            declaredBlockWidth == nil, seedTokenCount == nil, verifyBlockTokens == nil,
            conversationId == nil, turnBoundaries == nil, temperature == nil,
            topP == nil, samplingSeed == nil
        else { throw EmbeddingInputError("embedding request contains missing or unrelated fields") }
        try embeddingInput.validate()
        return embeddingInput
    }
}

/// A separate retrieval model. No chat cache or MTP session enters this adapter.
struct EmbeddingRuntime: Sendable {
    private let container: EmbedderModelContainer
    private let contextLimit: Int

    static func load(directory: URL) async throws -> EmbeddingRuntime {
        struct Configuration: Decodable {
            let modelType: String
            let hiddenSize: Int
            let numHiddenLayers: Int
            let vocabSize: Int
            let maxPositionEmbeddings: Int
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(
            Configuration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard config.modelType == "qwen3", config.hiddenSize == EmbeddingLimits.dimensions,
            config.numHiddenLayers == 36, config.vocabSize == 151665,
            config.maxPositionEmbeddings > 0
        else {
            throw MLXFastError.invalidInput(
                "embedding weights must use the Qwen3-Embedding-4B architecture: \(directory.path)"
            )
        }
        let container = try await EmbedderModelFactory.shared.loadContainer(
            from: directory, using: #huggingFaceTokenizerLoader())
        return EmbeddingRuntime(container: container, contextLimit: config.maxPositionEmbeddings)
    }

    static func prepare(
        _ input: EmbeddingWorkerInput, vocabularySize: Int, contextLimit: Int,
        encode: (String) -> [Int]
    ) throws -> [[Int]] {
        try input.validate()
        let batches: [[Int]]
        switch input {
        case .text(let text): batches = [encode(text)]
        case .texts(let texts): batches = texts.map(encode)
        case .tokens(let tokens): batches = [tokens]
        case .tokenBatches(let tokens): batches = tokens
        }
        try EmbeddingLimits.validateTokens(
            batches, vocabularySize: vocabularySize, contextLimit: contextLimit)
        return batches
    }

    static func normalize(_ values: [Float]) throws -> [Float] {
        guard values.allSatisfy(\.isFinite), let scale = values.map({ abs($0) }).max(), scale > 0
        else {
            throw MLXFastError.invalidInput("embedding has zero norm or non-finite values")
        }
        // Scale first so the float32 squared norm cannot overflow or underflow.
        let scaled = values.map { $0 / scale }
        let norm = sqrt(scaled.reduce(Float(0)) { $0 + $1 * $1 })
        return scaled.map { $0 / norm }
    }

    func embed(input: EmbeddingWorkerInput) async throws -> EmbeddingBatchResult {
        try await container.perform { context in
            let batches = try Self.prepare(
                input, vocabularySize: context.model.vocabularySize, contextLimit: contextLimit,
                encode: { context.tokenizer.encode(text: $0, addSpecialTokens: true) })
            var vectors: [[Float]] = []
            for tokens in batches {
                let ids = MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])
                let output = context.model(
                    ids, positionIds: nil, tokenTypeIds: nil,
                    attentionMask: MLXArray.ones([1, tokens.count], dtype: .int32))
                guard let hidden = output.hiddenStates,
                    hidden.shape == [1, tokens.count, EmbeddingLimits.dimensions]
                else {
                    throw MLXFastError.invalidInput(
                        "embedding model returned an invalid hidden-state shape")
                }
                // Qwen3's output already includes final RMSNorm. Pool without a second layer norm.
                let row = hidden[0, tokens.count - 1, 0...].asType(.float32)
                row.eval()
                vectors.append(try Self.normalize(row.asArray(Float.self)))
            }
            return EmbeddingBatchResult(
                vectors: vectors, promptTokens: batches.reduce(0) { $0 + $1.count })
        }
    }
}
