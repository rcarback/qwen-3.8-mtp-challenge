import Foundation

// The OpenAI chat-completions wire surface, narrowed to what a coding harness
// actually sends. Fields the decoder does not name are ignored, which is the
// correct posture: clients add parameters freely and a strict decoder would
// reject perfectly serviceable requests.
//
// LOCAL DEVELOPER TOOLING. Outside `editablePaths`.

struct ChatCompletionRequest: Decodable {
    let model: String?
    let messages: [ChatMessage]
    let tools: [OrderedJSON]?
    let stream: Bool?
    let maxTokens: Int?
    let stop: StopField?
    let n: Int?
    let temperature: Double?
    let topP: Double?
    let seed: UInt64?

    enum CodingKeys: String, CodingKey {
        case model, messages, tools, stream, stop, n, temperature, seed
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        messages = try container.decode([ChatMessage].self, forKey: .messages)
        tools = try container.decodeIfPresent([OrderedJSON].self, forKey: .tools)
        stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        stop = try container.decodeIfPresent(StopField.self, forKey: .stop)
        n = try container.decodeIfPresent(Int.self, forKey: .n)
        // Absent OR zero temperature means greedy, which is the session's
        // untouched argmax path. `top_k`, `presence_penalty`, and
        // `frequency_penalty` are deliberately not decoded: accepting a knob
        // this server cannot honour and silently ignoring it is worse than
        // never claiming it.
        temperature = try container.decodeIfPresent(
            Double.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        seed = try container.decodeIfPresent(UInt64.self, forKey: .seed)
        // `max_completion_tokens` is the newer spelling; honour either.
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
            ?? container.decodeIfPresent(Int.self, forKey: .maxCompletionTokens)
    }
}

struct ChatMessage: Decodable {
    let role: String
    let content: MessageContent?
    let toolCalls: [ToolCallPayload]?
    let toolCallId: String?

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
        case toolCallId = "tool_call_id"
    }

    private init(
        role: String, content: MessageContent?,
        toolCalls: [ToolCallPayload]?, toolCallId: String?
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
    }

    /// The assistant turn that requested one or more tool calls. Synthesised
    /// server-side when compaction retrieval answers an `expand` call, so the
    /// model sees a well-formed call/result pair rather than a bare result.
    static func assistantToolCall(_ calls: [ToolCallPayload]) -> ChatMessage {
        ChatMessage(
            role: "assistant", content: nil, toolCalls: calls, toolCallId: nil)
    }

    /// The matching tool result. `tool_call_id` must echo the call's id or the
    /// pair is invalid on the wire.
    static func toolResult(id: String, text: String) -> ChatMessage {
        ChatMessage(
            role: "tool", content: .text(text), toolCalls: nil, toolCallId: id)
    }

    /// Same message with different text. Used by tool-result compaction, which
    /// must preserve `tool_call_id` -- an orphaned tool message is invalid on
    /// the wire, and dropping the id would strand the assistant `tool_calls`
    /// entry that refers to it.
    func replacingContent(_ text: String) -> ChatMessage {
        ChatMessage(
            role: role, content: .text(text),
            toolCalls: toolCalls, toolCallId: toolCallId)
    }
}

/// Content arrives either as a bare string or as an array of typed parts.
enum MessageContent: Decodable {
    case text(String)
    case parts([String])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let single = try? container.decode(String.self) {
            self = .text(single)
            return
        }
        let items = try container.decode([Part].self)
        // Image and video parts are dropped: this is a text tower and the
        // checkpoint's `vision_config` is empty and never loaded.
        self = .parts(items.compactMap { $0.text })
    }

    private struct Part: Decodable {
        let type: String?
        let text: String?
    }

    var text: String {
        switch self {
        case .text(let value): return value
        case .parts(let values): return values.joined()
        }
    }
}

struct ToolCallPayload: Codable {
    let id: String
    let type: String
    let function: FunctionPayload
}

struct FunctionPayload: Codable {
    let name: String
    /// A JSON object encoded as a string, per the OpenAI schema.
    let arguments: String
}

enum StopField: Decodable {
    case one(String)
    case many([String])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let single = try? container.decode(String.self) {
            self = .one(single)
        } else {
            self = .many(try container.decode([String].self))
        }
    }

    var values: [String] {
        switch self {
        case .one(let value): return [value]
        case .many(let values): return values
        }
    }
}

// MARK: - Responses

struct ChatCompletionResponse: Encodable {
    let id: String
    let object = "chat.completion"
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: Usage

    struct Choice: Encodable {
        let index: Int
        let message: ResponseMessage
        let finishReason: String

        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
    }

    struct ResponseMessage: Encodable {
        let role = "assistant"
        let content: String?
        let toolCalls: [ToolCallPayload]?

        enum CodingKeys: String, CodingKey {
            case role, content
            case toolCalls = "tool_calls"
        }
    }

    struct Usage: Encodable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int
        /// Non-standard. OpenAI clients ignore unknown fields, and this is the
        /// only place the speculative-decode numbers can ride back to a caller
        /// that is not reading the server's stderr.
        let mtp: MTPUsageExtension

        enum CodingKeys: String, CodingKey {
            case mtp
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }
}

struct MTPUsageExtension: Encodable {
    let offeredDepth: Int
    let rounds: Int
    let acceptedDrafts: Int
    let rejectedDrafts: Int
    let acceptRate: Double?
    let effectiveDraftDepth: Double?
    let decodeTokensPerSecond: Double?
    let seedPrefillSeconds: Double

    enum CodingKeys: String, CodingKey {
        case rounds
        case offeredDepth = "offered_depth"
        case acceptedDrafts = "accepted_drafts"
        case rejectedDrafts = "rejected_drafts"
        case acceptRate = "accept_rate"
        case effectiveDraftDepth = "effective_draft_depth"
        case decodeTokensPerSecond = "decode_tokens_per_second"
        case seedPrefillSeconds = "seed_prefill_seconds"
    }
}

struct ChatCompletionChunk: Encodable {
    let id: String
    let object = "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [Choice]

    struct Choice: Encodable {
        let index: Int
        let delta: Delta
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index, delta
            case finishReason = "finish_reason"
        }
    }

    struct Delta: Encodable {
        let role: String?
        let content: String?
        let toolCalls: [StreamedToolCall]?

        enum CodingKeys: String, CodingKey {
            case role, content
            case toolCalls = "tool_calls"
        }
    }

    struct StreamedToolCall: Encodable {
        let index: Int
        let id: String
        let type: String
        let function: FunctionPayload
    }
}

struct ModelListResponse: Encodable {
    let object = "list"
    let data: [Entry]

    struct Entry: Encodable {
        let id: String
        let object = "model"
        let created: Int
        let ownedBy: String

        enum CodingKeys: String, CodingKey {
            case id, object, created
            case ownedBy = "owned_by"
        }
    }
}
