import Foundation
import MLXFastCore

/// A Swift port of `weights/chat_template.jinja`, narrowed to the branches an
/// OpenAI request can reach.
///
/// WHY A PORT AND NOT AN EVALUATOR. There is no Jinja engine in the frozen
/// dependency graph and none can be added. The template is 169 lines and the
/// unreachable half is vision content, `add_vision_id`, and `preserve_thinking`.
///
/// REASONING IS A REQUEST KNOB, NOT A CONSTANT. Every render takes a
/// ``Reasoning`` value that decides both template positions the two knobs touch:
/// the instruction sentence in the system turn (`chat_template.jinja:59-60` and
/// `:80-85`) and whether the generation prompt leaves the think block open
/// (`:163-169`). It defaults to OFF rather than to the template's own
/// `enable_thinking = true` at `xhigh`; see ``Reasoning/off``.
///
/// LOCAL DEVELOPER TOOLING. Outside `editablePaths`.
enum OpenAIPromptRendering {
    /// The template's two reasoning controls, resolved together.
    ///
    /// They resolve together because the template resolves them together:
    /// `chat_template.jinja:46-56` reads `reasoning_effort` only INSIDE the
    /// `enable_thinking` branch, so an effort with thinking off is a value
    /// nothing ever reads.
    struct Reasoning: Equatable, Sendable {
        enum Effort: String, CaseIterable, Sendable {
            case xhigh, medium, low
        }

        let enabled: Bool
        let effort: Effort

        /// What this server renders when a request says nothing.
        ///
        /// NOT the template's default, which is thinking on at `xhigh`. The
        /// worker holds one KV session and decodes it to completion before the
        /// next request starts, so an open think block is charged to every
        /// other caller as queue time. Clients that want reasoning ask for it,
        /// and `resolve(enableThinking:effort:)` makes asking for an effort
        /// enough.
        static let off = Reasoning(enabled: false, effort: .xhigh)

        /// `chat_template.jinja:51-55`, verbatim. Only `xhigh` and `low` carry
        /// text: `medium` opens the think block and says nothing about effort,
        /// which is a distinct state from thinking being off.
        var instructions: String {
            guard enabled else { return "" }
            switch effort {
            case .xhigh:
                return "Reasoning effort is set to xhigh. Please think "
                    + "carefully through the task, validate key assumptions, "
                    + "consider plausible alternatives, and prioritize "
                    + "correctness, consistency, and clarity in the final "
                    + "answer."
            case .low:
                return "Reasoning effort is set to low. Keep your thinking "
                    + "brief and focused, moving directly to the conclusion "
                    + "without unnecessary elaboration."
            case .medium:
                return ""
            }
        }

        /// `chat_template.jinja:163-169`. With thinking off the block is
        /// pre-closed, so the model's first emitted token is already the
        /// answer. With thinking on it is left open and the model closes it
        /// itself, which is what ``ThinkSplitter`` exists to undo.
        var generationPrompt: String {
            enabled
                ? "<|im_start|>assistant\n<think>\n"
                : "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        }

        /// Resolve a request's two fields under the template's own rules.
        ///
        /// An effort with no `enable_thinking` turns thinking ON. Honouring
        /// `reasoning_effort: "low"` while leaving thinking off would discard
        /// the only thing the caller asked for, and silently accepting a knob
        /// this server does not honour is the posture `OpenAIWireTypes`
        /// deliberately refuses. An explicit `enable_thinking: false` still
        /// wins: it is the more specific statement.
        ///
        /// An unsupported effort is rejected rather than clamped, exactly as
        /// `chat_template.jinja:48-50` raises.
        static func resolve(
            enableThinking: Bool?, effort: String?
        ) throws -> Reasoning {
            var resolved = Effort.xhigh
            if let effort {
                guard let parsed = Effort(rawValue: effort) else {
                    throw MLXFastError.invalidInput(
                        "unexpected reasoning effort '\(effort)'. Supported "
                            + "values are xhigh (the template default), "
                            + "medium, and low")
                }
                resolved = parsed
            }
            return Reasoning(
                enabled: enableThinking ?? (effort != nil), effort: resolved)
        }
    }

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

    /// Character offsets at which a rendered TURN ends, in the order emitted.
    ///
    /// Every offset lands immediately after an `<|im_end|>\n`, which matters:
    /// the tokenizer treats those as special tokens and always breaks on them,
    /// so tokenizing a prefix that ends there yields the same tokens as the
    /// corresponding prefix of the whole render. A boundary in the middle of
    /// ordinary text would carry no such guarantee.
    static func renderPromptWithTurnBoundaries(
        messages: [ChatMessage], tools: [OrderedJSON]?, reasoning: Reasoning
    ) throws -> (prompt: String, turnEnds: [Int]) {
        var turnEnds: [Int] = []
        let prompt = try renderPrompt(
            messages: messages, tools: tools, reasoning: reasoning,
            turnEnds: &turnEnds)
        return (prompt, turnEnds)
    }

    static func renderPrompt(
        messages: [ChatMessage], tools: [OrderedJSON]?, reasoning: Reasoning
    ) throws -> String {
        var ignored: [Int] = []
        return try renderPrompt(
            messages: messages, tools: tools, reasoning: reasoning,
            turnEnds: &ignored)
    }

    private static func renderPrompt(
        messages: [ChatMessage], tools: [OrderedJSON]?, reasoning: Reasoning,
        turnEnds: inout [Int]
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

        // `chat_template.jinja:59-60` and `:80-85`: the reasoning sentence
        // leads the system turn wherever that turn comes from, and when there
        // is no other reason to emit one it becomes the whole turn.
        let instructions = reasoning.instructions
        let instructionBlock = instructions.isEmpty ? "" : instructions + "\n\n"

        if let tools, !tools.isEmpty {
            out += "<|im_start|>system\n"
            out += instructionBlock
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
            turnEnds.append(out.count)
        } else if let leadingSystem, !leadingSystem.isEmpty {
            out += "<|im_start|>system\n" + instructionBlock + leadingSystem
                + "<|im_end|>\n"
            turnEnds.append(out.count)
        } else if !instructions.isEmpty {
            out += "<|im_start|>system\n" + instructions + "<|im_end|>\n"
            turnEnds.append(out.count)
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
                turnEnds.append(out.count)
            case "assistant":
                // A PRIOR turn's think block is always empty and always
                // pre-closed, whatever this turn's reasoning setting is.
                // `chat_template.jinja:111-117` would replay
                // `message.reasoning_content`, but this server does not
                // round-trip it: the wire type never decodes the field, and
                // replaying a reply's whole chain of thought would charge
                // every later turn for reasoning the model has already spent.
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
                turnEnds.append(out.count)
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
                    turnEnds.append(out.count)
                }
            default:
                throw MLXFastError.invalidInput(
                    "unexpected message role '\(message.role)'")
            }
        }

        out += reasoning.generationPrompt
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

    // MARK: - Splitting the think block off a reply

    /// Separates the reasoning half of a reply from the answer half.
    ///
    /// With thinking enabled the generation prompt ends at an OPEN `<think>`
    /// (`chat_template.jinja:168`), so the model emits its reasoning first and
    /// writes the closing tag itself. Everything before `</think>` is the
    /// chain of thought and everything after it is the answer. With thinking
    /// disabled the prompt already carries a closed block, the model never
    /// writes another one, and this is a pass-through that costs one branch.
    ///
    /// Splitting here rather than at the caller has a second effect that
    /// matters: the tool-call gate and the stop strings then see the ANSWER
    /// only, so a model that writes `<tool_call>` or a caller's stop string
    /// while reasoning aloud no longer ends its own turn.
    ///
    /// The answer keeps whatever whitespace follows the tag, usually `\n\n`.
    /// Trimming it would make the two modes agree, because with thinking off
    /// those two newlines sit in the PROMPT instead and never reach the reply.
    /// It is left alone anyway: vLLM's reasoning parsers hand back the raw
    /// remainder, a caller comparing against a reference implementation should
    /// see the same bytes, and a server that silently edits model output is a
    /// worse default than one whose callers trim.
    struct ThinkSplitter {
        static let marker = "</think>"
        let enabled: Bool

        /// Character offset just past the close tag, once seen. Stable across
        /// rounds: the reply only ever grows at its end, so an offset found in
        /// one round means the same position in the next.
        private var answerStart: Int?
        /// Frozen the round the tag is found. The reasoning half cannot change
        /// after that, and recopying it every round is the quadratic cost
        /// ``ToolCallGate`` was rewritten to avoid.
        private var frozenReasoning: String?

        init(enabled: Bool) { self.enabled = enabled }

        /// True once the model has closed the block. A turn that ends with
        /// this still false spent its whole budget reasoning.
        var closed: Bool { answerStart != nil }

        mutating func split(
            _ full: String
        ) -> (reasoning: String, answer: String) {
            guard enabled else { return ("", full) }
            if answerStart == nil, let found = full.range(of: Self.marker) {
                answerStart = full.distance(
                    from: full.startIndex, to: found.upperBound)
                frozenReasoning = String(full[full.startIndex ..< found.lowerBound])
            }
            guard let start = answerStart, let reasoning = frozenReasoning else {
                // Still inside the block. Withhold a trailing run that could
                // still grow into `</think>`, so a half-written close tag never
                // reaches a caller as reasoning text it has to strip itself.
                return (Self.withoutPartialMarker(full), "")
            }
            let index = full.index(
                full.startIndex, offsetBy: Swift.min(start, full.count))
            return (reasoning, String(full[index...]))
        }

        /// Longest suffix of `text` that is a proper prefix of the marker,
        /// removed. Same rule as ``ToolCallGate``, and for the same reason.
        private static func withoutPartialMarker(_ text: String) -> String {
            for length in stride(
                from: Swift.min(marker.count - 1, text.count), through: 1,
                by: -1)
            {
                if text.hasSuffix(String(marker.prefix(length))) {
                    return String(text.dropLast(length))
                }
            }
            return text
        }
    }

    // MARK: - Streaming gate

    /// Holds back any trailing text that could still become `<tool_call>`.
    ///
    /// Without this a client would see `<tool` arrive as assistant content one
    /// round before the server decides the reply is a tool call.
    struct ToolCallGate {
        static let marker = "<tool_call>"
        private static let markerCount = marker.count
        private var emitted = 0
        private(set) var stopped = false

        /// WHY THIS SHAPE. `admit` receives the WHOLE reply so far, once per
        /// decode round, so anything that walks the string from its start is
        /// quadratic in the reply length. The previous version walked from
        /// `startIndex` up to five times per call: `range(of:)` over the whole
        /// string, `full.count`, a second `count` inside `slice`, and two
        /// `index(_:offsetBy:)` walks. This version walks forward once, for
        /// the `count` that `emitted` is expressed in, and everything else is
        /// bounded by the marker length or by the size of the new text.
        ///
        /// The marker search starts `markerCount` characters before the
        /// emitted cursor rather than at `startIndex`. That is not a heuristic:
        /// the hold-back below never advances `emitted` past the start of a
        /// possible marker, so a marker can never begin earlier than that, and
        /// the back-off covers a marker straddling the cursor.
        mutating func admit(_ full: String) -> (delta: String, sawToolCall: Bool) {
            if stopped { return ("", true) }
            let total = full.count
            let pending = Swift.max(0, total - emitted)
            // Walk BACKWARD from the end. `pending` is a handful of characters
            // per round; `emitted` is the whole reply.
            let cursor = full.index(full.endIndex, offsetBy: -pending)
            // Clamp against what is IN FRONT OF THE CURSOR, not against
            // `emitted`. The gate can be handed a string SHORTER than what it
            // already emitted: a stop-string hit truncates `full` before the
            // gate sees it (QwenRuntimeServe.swift, the stop branch), so
            // `emitted` may exceed `total`. Backing off by `emitted` would
            // then index before `startIndex` and trap.
            let searchBack = Swift.min(Self.markerCount, total - pending)
            let searchFrom = full.index(cursor, offsetBy: -searchBack)

            if let marker = full.range(
                of: Self.marker, range: searchFrom ..< full.endIndex)
            {
                stopped = true
                guard cursor < marker.lowerBound else {
                    emitted = Swift.max(emitted, total - pending)
                    return ("", true)
                }
                let delta = String(full[cursor ..< marker.lowerBound])
                emitted = full.distance(
                    from: full.startIndex, to: marker.lowerBound)
                return (delta, true)
            }

            // Longest suffix of `full` that is a proper prefix of the marker.
            var held = 0
            for length in stride(
                from: Swift.min(Self.markerCount - 1, total),
                through: 1, by: -1)
            {
                if full.hasSuffix(String(Self.marker.prefix(length))) {
                    held = length
                    break
                }
            }
            let safe = total - held
            guard safe > emitted else { return ("", false) }
            let end = full.index(full.endIndex, offsetBy: -held)
            let delta = String(full[cursor ..< end])
            emitted = safe
            return (delta, false)
        }
    }
}
