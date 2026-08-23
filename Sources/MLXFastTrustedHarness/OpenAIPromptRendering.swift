import Foundation
import MLXFastCore

/// A Swift port of `weights/chat_template.jinja`, narrowed to the branches an
/// OpenAI request can reach.
///
/// WHY A PORT AND NOT AN EVALUATOR. There is no Jinja engine in the frozen
/// dependency graph and none can be added. The template is 169 lines and half of
/// them are unreachable here: this server always renders with
/// `enable_thinking = false`, which makes `reasoning_instructions` the empty
/// string and collapses the template's four reasoning branches to one.
///
/// WHY THINKING IS DISABLED. With thinking open the model reasons past any
/// sensible budget -- the same failure the GPQA gate measured -- and the worker's
/// per-session decode ceiling is 1536 tokens. A coding harness needs the answer,
/// not the deliberation.
///
/// LOCAL DEVELOPER TOOLING. Outside `editablePaths`.
enum OpenAIPromptRendering {
    /// The literal instruction block the template emits after `</tools>`.
    /// Copied verbatim from `chat_template.jinja:68`; the model was tuned on
    /// these exact bytes, so it is a constant, not prose to improve.
    static let toolInstructions = """


        If you choose to call a function ONLY reply in the following format with NO suffix:

        <tool_call>
        <function=example_function_name>
        <parameter=example_parameter_1>
        value_1
        </parameter>
        <parameter=example_parameter_2>
        This is the value for the second parameter
        that can span
        multiple lines
        </parameter>
        </function>
        </tool_call>

        <IMPORTANT>
        Reminder:
        - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
        - Required parameters MUST be specified
        - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
        </IMPORTANT>
        """

    static func renderPrompt(
        messages: [ChatMessage], tools: [OrderedJSON]?
    ) throws -> String {
        guard !messages.isEmpty else {
            throw MLXFastError.invalidInput("no messages provided")
        }
        var out = ""
        let leadingSystem = messages.first.flatMap {
            $0.role == "system"
                ? ($0.content?.text ?? "").trimmingCharacters(
                    in: .whitespacesAndNewlines)
                : nil
        }

        if let tools, !tools.isEmpty {
            out += "<|im_start|>system\n"
            out += "# Tools\n\nYou have access to the following functions:\n\n<tools>"
            for tool in tools {
                out += "\n" + tool.serialized()
            }
            out += "\n</tools>"
            out += toolInstructions
            if let leadingSystem, !leadingSystem.isEmpty {
                out += "\n\n" + leadingSystem
            }
            out += "<|im_end|>\n"
        } else if let leadingSystem, !leadingSystem.isEmpty {
            out += "<|im_start|>system\n" + leadingSystem + "<|im_end|>\n"
        }

        for (index, message) in messages.enumerated() {
            let content = (message.content?.text ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            switch message.role {
            case "system":
                // Already emitted above. The template rejects a system message
                // anywhere but the front, and so does this.
                guard index == 0 else {
                    throw MLXFastError.invalidInput(
                        "a system message must be the first message")
                }
            case "user":
                out += "<|im_start|>user\n" + content + "<|im_end|>\n"
            case "assistant":
                // `reasoning_content` is never round-tripped by this server, so
                // the think block is always empty and always pre-closed.
                out += "<|im_start|>assistant\n<think>\n\n</think>\n\n" + content
                if let calls = message.toolCalls, !calls.isEmpty {
                    for (callIndex, call) in calls.enumerated() {
                        if callIndex == 0 {
                            out += content.isEmpty
                                ? "<tool_call>\n<function=\(call.function.name)>\n"
                                : "\n\n<tool_call>\n<function=\(call.function.name)>\n"
                        } else {
                            out += "\n<tool_call>\n<function=\(call.function.name)>\n"
                        }
                        out += renderArguments(call.function.arguments)
                        out += "</function>\n</tool_call>"
                    }
                }
                out += "<|im_end|>\n"
            case "tool":
                // `chat_template.jinja:148`: `loop.previtem` is falsy on the
                // very first message, so the leading `<|im_start|>user` is
                // emitted only when a previous message actually exists (and is
                // not itself a tool message) -- not merely when the role
                // differs from "tool".
                let previousExistsAndIsNotTool =
                    index > 0 && messages[index - 1].role != "tool"
                if previousExistsAndIsNotTool {
                    out += "<|im_start|>user"
                }
                out += "\n<tool_response>\n" + content + "\n</tool_response>"
                let nextRole = index + 1 < messages.count
                    ? messages[index + 1].role : nil
                if nextRole != "tool" {
                    out += "<|im_end|>\n"
                }
            default:
                throw MLXFastError.invalidInput(
                    "unexpected message role '\(message.role)'")
            }
        }

        out += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return out
    }

    /// Turn an OpenAI `arguments` JSON string back into the model's parameter
    /// blocks. String values go in raw; everything else goes in as JSON, exactly
    /// as `chat_template.jinja:138` does.
    private static func renderArguments(_ arguments: String) -> String {
        guard !arguments.isEmpty,
              let parsed = try? OrderedJSON.parse(arguments),
              case .object(let pairs) = parsed
        else { return "" }
        var out = ""
        for pair in pairs {
            out += "<parameter=\(pair.key)>\n"
            out += pair.value.stringValue ?? pair.value.serialized()
            out += "\n</parameter>\n"
        }
        return out
    }

    // MARK: - Parsing the model's tool calls

    /// Guarded exclusively by `callCounterLock`. Swift 6 strict concurrency
    /// requires an explicit opt-out for global mutable state; the lock is the
    /// actual synchronization, not this annotation.
    private nonisolated(unsafe) static var callCounter: UInt32 = 0
    private static let callCounterLock = NSLock()

    private static func nextCallID() -> String {
        callCounterLock.lock()
        defer { callCounterLock.unlock() }
        callCounter &+= 1
        return String(format: "call_%08x%08x", UInt32.random(in: 0...UInt32.max),
                      callCounter)
    }

    /// Extract every well-formed `<tool_call>` block from a reply.
    ///
    /// A truncated or malformed block yields nothing rather than throwing: the
    /// model produced it, so it is input, and input must not be able to fail a
    /// request that already spent a decode window.
    static func parseToolCalls(
        _ text: String, tools: [OrderedJSON]?
    ) -> [ToolCallPayload] {
        var calls: [ToolCallPayload] = []
        var cursor = text.startIndex
        while let open = text.range(of: "<tool_call>", range: cursor..<text.endIndex) {
            guard let close = text.range(
                of: "</tool_call>", range: open.upperBound..<text.endIndex)
            else { break }
            let body = String(text[open.upperBound..<close.lowerBound])
            if let call = parseOneCall(body, tools: tools) {
                calls.append(call)
            }
            cursor = close.upperBound
        }
        return calls
    }

    private static func parseOneCall(
        _ body: String, tools: [OrderedJSON]?
    ) -> ToolCallPayload? {
        guard let nameOpen = body.range(of: "<function="),
              let nameClose = body.range(
                  of: ">", range: nameOpen.upperBound..<body.endIndex)
        else { return nil }
        let name = String(body[nameOpen.upperBound..<nameClose.lowerBound])
            .trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty else { return nil }

        let types = parameterTypes(forTool: name, tools: tools)
        var pairs: [(String, OrderedJSON)] = []
        var cursor = nameClose.upperBound
        while let open = body.range(
            of: "<parameter=", range: cursor..<body.endIndex) {
            guard let openClose = body.range(
                    of: ">", range: open.upperBound..<body.endIndex),
                  let close = body.range(
                    of: "</parameter>",
                    range: openClose.upperBound..<body.endIndex)
            else { return nil }
            let key = String(body[open.upperBound..<openClose.lowerBound])
            // The template writes a newline after the open tag and before the
            // close tag; both belong to the framing, not to the value.
            var raw = String(body[openClose.upperBound..<close.lowerBound])
            if raw.hasPrefix("\n") { raw.removeFirst() }
            if raw.hasSuffix("\n") { raw.removeLast() }
            pairs.append((key, coerce(raw, declaredType: types[key])))
            cursor = close.upperBound
        }
        return ToolCallPayload(
            id: nextCallID(), type: "function",
            function: FunctionPayload(
                name: name,
                arguments: OrderedJSON.object(pairs).serialized()))
    }

    /// A parameter declared `string` is taken literally. Anything else is parsed
    /// as JSON, and falls back to a string when the model wrote something the
    /// schema did not describe -- a wrong-typed argument the tool can reject
    /// beats a dropped argument it cannot see.
    private static func coerce(
        _ raw: String, declaredType: String?
    ) -> OrderedJSON {
        if declaredType == "string" { return .string(raw) }
        if let parsed = try? OrderedJSON.parse(raw) { return parsed }
        return .string(raw)
    }

    private static func parameterTypes(
        forTool name: String, tools: [OrderedJSON]?
    ) -> [String: String] {
        guard let tools else { return [:] }
        for tool in tools {
            let function = tool["function"] ?? tool
            guard function["name"]?.stringValue == name,
                  case .object(let properties)? =
                      function["parameters"]?["properties"]
            else { continue }
            var types: [String: String] = [:]
            for property in properties {
                types[property.key] = property.value["type"]?.stringValue
            }
            return types
        }
        return [:]
    }

    // MARK: - Streaming gate

    /// Holds back any trailing text that could still become `<tool_call>`.
    ///
    /// Without this a client would see `<tool` arrive as assistant content one
    /// round before the server decides the reply is a tool call.
    struct ToolCallGate {
        static let marker = "<tool_call>"
        private var emitted = 0
        private(set) var stopped = false

        mutating func admit(_ full: String) -> (delta: String, sawToolCall: Bool) {
            if stopped { return ("", true) }
            if let marker = full.range(of: Self.marker) {
                let safe = full.distance(
                    from: full.startIndex, to: marker.lowerBound)
                let delta = slice(full, from: emitted, to: safe)
                emitted = max(emitted, safe)
                stopped = true
                return (delta, true)
            }
            // Longest suffix of `full` that is a proper prefix of the marker.
            var held = 0
            for length in stride(from: min(Self.marker.count - 1, full.count),
                                 through: 1, by: -1) {
                if full.hasSuffix(String(Self.marker.prefix(length))) {
                    held = length
                    break
                }
            }
            let safe = full.count - held
            guard safe > emitted else { return ("", false) }
            let delta = slice(full, from: emitted, to: safe)
            emitted = safe
            return (delta, false)
        }

        private func slice(_ text: String, from: Int, to: Int) -> String {
            guard to > from, to <= text.count else { return "" }
            let start = text.index(text.startIndex, offsetBy: from)
            let end = text.index(text.startIndex, offsetBy: to)
            return String(text[start..<end])
        }
    }
}
