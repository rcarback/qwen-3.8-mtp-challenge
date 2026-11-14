import Foundation
import MLXFastCore

enum EmbeddingEncoding: String, Decodable {
    case float, base64
}

struct EmbeddingRequest: Decodable {
    let model: String
    let input: EmbeddingWorkerInput
    let encodingFormat: EmbeddingEncoding?
    let dimensions: Int?

    enum CodingKeys: String, CodingKey {
        case model, input, dimensions
        case encodingFormat = "encoding_format"
    }

    func validate(modelName: String) throws {
        guard model == modelName else {
            throw EmbeddingInputError("embedding model must be \(modelName)")
        }
        guard dimensions == nil else {
            throw EmbeddingInputError(
                "dimensions is not supported; this model returns \(EmbeddingLimits.dimensions) dimensions"
            )
        }
        try input.validate()
    }
}

enum EmbeddingValue: Encodable {
    case float([Float])
    case base64(String)

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .float(let values): try container.encode(values)
        case .base64(let value): try container.encode(value)
        }
    }
}

func encodeEmbedding(_ values: [Float], format: EmbeddingEncoding) throws -> EmbeddingValue {
    guard values.allSatisfy(\.isFinite) else {
        throw MLXFastError.invalidInput("embedding contains non-finite values")
    }
    switch format {
    case .float: return .float(values)
    case .base64:
        var bytes = Data(capacity: values.count * 4)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        return .base64(bytes.base64EncodedString())
    }
}

struct EmbeddingResponse: Encodable {
    struct Item: Encodable {
        let object = "embedding"
        let index: Int
        let embedding: EmbeddingValue
    }

    struct Usage: Encodable {
        let promptTokens: Int
        let totalTokens: Int
        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case totalTokens = "total_tokens"
        }
    }

    let object = "list"
    let model: String
    let data: [Item]
    let usage: Usage

    init(result: EmbeddingBatchResult, model: String, format: EmbeddingEncoding) throws {
        self.model = model
        data = try result.vectors.enumerated().map {
            Item(index: $0.offset, embedding: try encodeEmbedding($0.element, format: format))
        }
        usage = Usage(promptTokens: result.promptTokens, totalTokens: result.promptTokens)
    }
}
