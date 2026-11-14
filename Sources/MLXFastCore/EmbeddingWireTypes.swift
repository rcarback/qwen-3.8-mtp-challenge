import Foundation

public enum EmbeddingWorkerInput: Codable, Equatable, Sendable {
    case text(String)
    case texts([String])
    case tokens([Int])
    case tokenBatches([[Int]])

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if let text = try? value.decode(String.self) {
            self = .text(text)
        } else if let texts = try? value.decode([String].self) {
            self = .texts(texts)
        } else if let tokens = try? value.decode([Int].self) {
            self = .tokens(tokens)
        } else {
            self = .tokenBatches(try value.decode([[Int]].self))
        }
        try validate()
    }

    public func encode(to encoder: Encoder) throws {
        var value = encoder.singleValueContainer()
        switch self {
        case .text(let text): try value.encode(text)
        case .texts(let texts): try value.encode(texts)
        case .tokens(let tokens): try value.encode(tokens)
        case .tokenBatches(let batches): try value.encode(batches)
        }
    }

    public var count: Int {
        switch self {
        case .text, .tokens: 1
        case .texts(let texts): texts.count
        case .tokenBatches(let batches): batches.count
        }
    }

    public func validate() throws {
        guard (1 ... EmbeddingLimits.maxInputs).contains(count) else {
            throw EmbeddingInputError("input must contain 1...\(EmbeddingLimits.maxInputs) items")
        }
        switch self {
        case .text(let text):
            guard !text.isEmpty else { throw EmbeddingInputError("input must not be empty") }
        case .texts(let texts):
            guard texts.allSatisfy({ !$0.isEmpty }) else {
                throw EmbeddingInputError("input strings must not be empty")
            }
        case .tokens(let tokens): try EmbeddingLimits.validateTokens([tokens])
        case .tokenBatches(let batches): try EmbeddingLimits.validateTokens(batches)
        }
    }
}

public struct EmbeddingInputError: Error, CustomStringConvertible, Sendable {
    public let description: String
    public init(_ description: String) { self.description = description }
}

public enum EmbeddingLimits {
    public static let maxInputs = 32
    public static let maxTokensPerInput = 8192
    public static let maxTotalTokens = 32768
    public static let dimensions = 2560

    public static func validateTokens(
        _ batches: [[Int]], vocabularySize: Int = Int(Int32.max),
        contextLimit: Int = maxTokensPerInput
    ) throws {
        guard (1 ... maxInputs).contains(batches.count) else {
            throw EmbeddingInputError("input must contain 1...\(maxInputs) items")
        }
        var total = 0
        for tokens in batches {
            guard !tokens.isEmpty, tokens.count <= min(maxTokensPerInput, contextLimit) else {
                throw EmbeddingInputError(
                    "each input must contain 1...\(min(maxTokensPerInput, contextLimit)) tokens")
            }
            guard tokens.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
                throw EmbeddingInputError("input contains a token outside the embedding vocabulary")
            }
            total += tokens.count
        }
        guard total <= maxTotalTokens else {
            throw EmbeddingInputError("request exceeds \(maxTotalTokens) tokens")
        }
    }
}

public struct EmbeddingBatchResult: Codable, Sendable {
    public let vectors: [[Float]]
    public let promptTokens: Int

    public init(vectors: [[Float]], promptTokens: Int) {
        self.vectors = vectors
        self.promptTokens = promptTokens
    }

    public func validate(inputCount: Int) throws {
        guard vectors.count == inputCount,
            promptTokens >= inputCount, promptTokens <= EmbeddingLimits.maxTotalTokens,
            vectors.allSatisfy({
                $0.count == EmbeddingLimits.dimensions && $0.allSatisfy(\.isFinite)
            })
        else { throw MLXFastError.invalidInput("invalid embedding worker result") }
    }
}
