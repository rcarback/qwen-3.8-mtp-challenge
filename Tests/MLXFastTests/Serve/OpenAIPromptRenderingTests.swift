import Foundation
import Testing

@testable import MLXFastHarness

@Suite("OpenAI prompt rendering")
struct OpenAIPromptRenderingTests {
    private func message(
        _ role: String, _ text: String?,
        toolCalls: [ToolCallPayload]? = nil, toolCallId: String? = nil
    ) throws -> ChatMessage {
        var object = #"{"role":"\#(role)""#
        if let text { object += #","content":"\#(text)""# }
        if let toolCallId { object += #","tool_call_id":"\#(toolCallId)""# }
        if let toolCalls {
            let encoded = try JSONEncoder().encode(toolCalls)
            object += #","tool_calls":"# + String(decoding: encoded, as: UTF8.self)
        }
        object += "}"
        return try JSONDecoder().decode(
            ChatMessage.self, from: Data(object.utf8))
    }

    @Test("a bare user turn gets the thinking-disabled generation prompt")
    func rendersUserTurn() throws {
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [message("user", "hi")], tools: nil,
            reasoning: .off)
        #expect(rendered == """
            <|im_start|>user
            hi<|im_end|>
            <|im_start|>assistant
            <think>

            </think>


            """)
    }

    @Test("a system message leads the prompt")
    func rendersSystem() throws {
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [try message("system", "be terse"),
                       try message("user", "hi")],
            tools: nil, reasoning: .off)
        #expect(rendered.hasPrefix("<|im_start|>system\nbe terse<|im_end|>\n"))
    }

    @Test("an empty system message is omitted entirely")
    func skipsEmptySystem() throws {
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [try message("system", "   "), try message("user", "hi")],
            tools: nil, reasoning: .off)
        #expect(!rendered.contains("<|im_start|>system"))
    }

    @Test("tools render into a leading system block in declared order")
    func rendersTools() throws {
        let tool = try OrderedJSON.parse(
            #"{"type":"function","function":{"name":"read","description":"d"}}"#)
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [try message("user", "hi")], tools: [tool],
            reasoning: .off)
        #expect(rendered.hasPrefix("""
            <|im_start|>system
            # Tools

            You have access to the following functions:

            <tools>
            {"type":"function","function":{"name":"read","description":"d"}}
            </tools>
            """))
        #expect(rendered.contains("<tool_call>\n<function=example_function_name>"))
    }

    @Test("a system message follows the tool block rather than preceding it")
    func placesSystemAfterTools() throws {
        let tool = try OrderedJSON.parse(#"{"type":"function"}"#)
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [try message("system", "be terse"),
                       try message("user", "hi")],
            tools: [tool], reasoning: .off)
        let toolsIndex = rendered.range(of: "</tools>")!.lowerBound
        let systemIndex = rendered.range(of: "be terse")!.lowerBound
        #expect(toolsIndex < systemIndex)
    }

    @Test("an assistant tool call renders in the model's XML form")
    func rendersAssistantToolCall() throws {
        let call = ToolCallPayload(
            id: "call_1", type: "function",
            function: FunctionPayload(
                name: "read", arguments: #"{"path":"/tmp/x","lines":5}"#))
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [
                try message("user", "read it"),
                try message("assistant", nil, toolCalls: [call]),
                try message("tool", "ok", toolCallId: "call_1"),
            ],
            tools: nil, reasoning: .off)
        #expect(rendered.contains("""
            <tool_call>
            <function=read>
            <parameter=path>
            /tmp/x
            </parameter>
            <parameter=lines>
            5
            </parameter>
            </function>
            </tool_call>
            """))
        #expect(rendered.contains("<tool_response>\nok\n</tool_response>"))
    }

    @Test("consecutive tool results share one user turn")
    func groupsToolResponses() throws {
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [
                try message("user", "go"),
                try message("assistant", "working"),
                try message("tool", "a", toolCallId: "1"),
                try message("tool", "b", toolCallId: "2"),
            ],
            tools: nil, reasoning: .off)
        #expect(rendered.contains("""
            <|im_start|>user
            <tool_response>
            a
            </tool_response>
            <tool_response>
            b
            </tool_response><|im_end|>
            """))
    }

    @Test("no messages is rejected rather than rendered")
    func rejectsEmptyConversation() {
        #expect(throws: (any Error).self) {
            _ = try OpenAIPromptRendering.renderPrompt(
                messages: [], tools: nil, reasoning: .off)
        }
    }

    // MARK: - reasoning

    private func reasoning(
        _ effort: OpenAIPromptRendering.Reasoning.Effort
    ) -> OpenAIPromptRendering.Reasoning {
        OpenAIPromptRendering.Reasoning(enabled: true, effort: effort)
    }

    @Test("thinking on leaves the generation prompt's think block open")
    func opensThinkBlock() throws {
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [try message("user", "hi")], tools: nil,
            reasoning: reasoning(.medium))
        #expect(rendered.hasSuffix("<|im_start|>assistant\n<think>\n"))
        // `medium` opens the block and says nothing about effort, so a
        // conversation with no system message gains no system turn at all.
        #expect(!rendered.contains("<|im_start|>system"))
    }

    @Test("an effort with no system message becomes the whole system turn")
    func synthesisesSystemTurn() throws {
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [try message("user", "hi")], tools: nil,
            reasoning: reasoning(.low))
        #expect(rendered.hasPrefix("""
            <|im_start|>system
            Reasoning effort is set to low. Keep your thinking brief and \
            focused, moving directly to the conclusion without unnecessary \
            elaboration.<|im_end|>
            """))
    }

    @Test("the effort sentence leads an existing system message")
    func prependsToSystemMessage() throws {
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [try message("system", "be terse"),
                       try message("user", "hi")],
            tools: nil, reasoning: reasoning(.xhigh))
        let effortIndex = rendered.range(of: "Reasoning effort")!.lowerBound
        let systemIndex = rendered.range(of: "be terse")!.lowerBound
        #expect(effortIndex < systemIndex)
        #expect(rendered.contains("clarity in the final answer.\n\nbe terse"))
    }

    @Test("the effort sentence leads the tool block too")
    func prependsToToolBlock() throws {
        let tool = try OrderedJSON.parse(#"{"type":"function"}"#)
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [try message("user", "hi")], tools: [tool],
            reasoning: reasoning(.xhigh))
        #expect(rendered.hasPrefix(
            "<|im_start|>system\nReasoning effort is set to xhigh."))
        #expect(rendered.contains("final answer.\n\n# Tools"))
    }

    /// A prior assistant turn keeps its empty pre-closed block whatever this
    /// turn asks for. Replaying a whole chain of thought would charge every
    /// later turn for reasoning the model already spent.
    @Test("a prior assistant turn keeps its closed think block")
    func priorTurnsStayClosed() throws {
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [try message("user", "hi"),
                       try message("assistant", "hello"),
                       try message("user", "again")],
            tools: nil, reasoning: reasoning(.xhigh))
        #expect(rendered.contains(
            "<|im_start|>assistant\n<think>\n\n</think>\n\nhello<|im_end|>"))
        #expect(rendered.hasSuffix("<|im_start|>assistant\n<think>\n"))
    }

    @Test("thinking off renders exactly what it rendered before the knob")
    func offIsUnchanged() throws {
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [try message("system", "be terse"),
                       try message("user", "hi")],
            tools: nil, reasoning: .off)
        #expect(rendered == """
            <|im_start|>system
            be terse<|im_end|>
            <|im_start|>user
            hi<|im_end|>
            <|im_start|>assistant
            <think>

            </think>


            """)
    }
}

@Suite("Reasoning resolution")
struct ReasoningResolutionTests {
    @Test("a request that says nothing gets thinking off")
    func defaultsOff() throws {
        let resolved = try OpenAIPromptRendering.Reasoning.resolve(
            enableThinking: nil, effort: nil)
        #expect(resolved == .off)
        #expect(resolved.instructions.isEmpty)
    }

    /// The template reads the effort only inside the `enable_thinking` branch,
    /// so an effort that did not turn thinking on would be discarded silently.
    @Test("an effort alone turns thinking on")
    func effortImpliesThinking() throws {
        let resolved = try OpenAIPromptRendering.Reasoning.resolve(
            enableThinking: nil, effort: "low")
        #expect(resolved.enabled)
        #expect(resolved.effort == .low)
    }

    @Test("an explicit enable_thinking false beats an effort")
    func explicitDisableWins() throws {
        let resolved = try OpenAIPromptRendering.Reasoning.resolve(
            enableThinking: false, effort: "xhigh")
        #expect(!resolved.enabled)
        #expect(resolved.instructions.isEmpty)
    }

    @Test("enable_thinking alone takes the template's xhigh default")
    func thinkingAloneIsXhigh() throws {
        let resolved = try OpenAIPromptRendering.Reasoning.resolve(
            enableThinking: true, effort: nil)
        #expect(resolved.effort == .xhigh)
        #expect(resolved.instructions.hasPrefix(
            "Reasoning effort is set to xhigh."))
    }

    @Test("medium opens the block without an effort sentence")
    func mediumSaysNothing() throws {
        let resolved = try OpenAIPromptRendering.Reasoning.resolve(
            enableThinking: nil, effort: "medium")
        #expect(resolved.enabled)
        #expect(resolved.instructions.isEmpty)
    }

    @Test("an unsupported effort is rejected, not clamped")
    func rejectsUnknownEffort() {
        #expect(throws: (any Error).self) {
            _ = try OpenAIPromptRendering.Reasoning.resolve(
                enableThinking: nil, effort: "high")
        }
    }
}

@Suite("Think-block splitting")
struct ThinkSplitterTests {
    @Test("disabled, every reply is answer and nothing is reasoning")
    func passesThroughWhenDisabled() {
        var splitter = OpenAIPromptRendering.ThinkSplitter(enabled: false)
        let split = splitter.split("plain </think> text")
        #expect(split.reasoning.isEmpty)
        #expect(split.answer == "plain </think> text")
        #expect(!splitter.closed)
    }

    @Test("text before the close tag is reasoning, text after it is the answer")
    func splitsAtCloseTag() {
        var splitter = OpenAIPromptRendering.ThinkSplitter(enabled: true)
        #expect(splitter.split("weighing it").answer.isEmpty)
        #expect(splitter.split("weighing it").reasoning == "weighing it")
        let split = splitter.split("weighing it</think>\n\nthe answer")
        #expect(split.reasoning == "weighing it")
        #expect(split.answer == "\n\nthe answer")
        #expect(splitter.closed)
    }

    /// A trailing run that could still grow into `</think>` is withheld, so a
    /// half-written close tag never reaches a caller as reasoning it has to
    /// strip itself.
    @Test("a partial close tag is withheld until it resolves")
    func withholdsPartialTag() {
        var splitter = OpenAIPromptRendering.ThinkSplitter(enabled: true)
        #expect(splitter.split("done </thi").reasoning == "done ")
        #expect(splitter.split("done </think").reasoning == "done ")
        #expect(splitter.split("done </thing").reasoning == "done </thing")
    }

    /// The reasoning half is frozen when the tag is found. Later rounds append
    /// to the ANSWER only, and must not disturb what was already settled.
    @Test("the reasoning half stops growing once the block closes")
    func freezesReasoning() {
        var splitter = OpenAIPromptRendering.ThinkSplitter(enabled: true)
        _ = splitter.split("why</think>a")
        let later = splitter.split("why</think>ab")
        #expect(later.reasoning == "why")
        #expect(later.answer == "ab")
    }

    @Test("a turn that never closes the block yields no answer at all")
    func neverClosed() {
        var splitter = OpenAIPromptRendering.ThinkSplitter(enabled: true)
        let split = splitter.split("still thinking about it")
        #expect(split.answer.isEmpty)
        #expect(split.reasoning == "still thinking about it")
        #expect(!splitter.closed)
    }
}

@Suite("Tool-call parsing")
struct ToolCallParsingTests {
    private var readTool: OrderedJSON {
        get throws {
            try OrderedJSON.parse("""
                {"type":"function","function":{"name":"read","parameters":\
                {"type":"object","properties":{"path":{"type":"string"},\
                "lines":{"type":"integer"}}}}}
                """)
        }
    }

    @Test("a declared string parameter stays a JSON string")
    func parsesStringParameter() throws {
        let calls = OpenAIPromptRendering.parseToolCalls("""
            <tool_call>
            <function=read>
            <parameter=path>
            /tmp/x
            </parameter>
            </function>
            </tool_call>
            """, tools: [try readTool])
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "read")
        #expect(calls[0].function.arguments == #"{"path":"/tmp/x"}"#)
    }

    @Test("a declared non-string parameter is parsed as JSON")
    func parsesTypedParameter() throws {
        let calls = OpenAIPromptRendering.parseToolCalls("""
            <tool_call>
            <function=read>
            <parameter=lines>
            5
            </parameter>
            </function>
            </tool_call>
            """, tools: [try readTool])
        #expect(calls[0].function.arguments == #"{"lines":5}"#)
    }

    /// A non-string parameter whose value is not valid JSON must not be dropped
    /// and must not crash the request: it degrades to a string.
    @Test("an unparseable typed parameter degrades to a string")
    func degradesBadTypedParameter() throws {
        let calls = OpenAIPromptRendering.parseToolCalls("""
            <tool_call>
            <function=read>
            <parameter=lines>
            many
            </parameter>
            </function>
            </tool_call>
            """, tools: [try readTool])
        #expect(calls[0].function.arguments == #"{"lines":"many"}"#)
    }

    @Test("multi-line values keep their interior newlines")
    func keepsMultilineValues() throws {
        let calls = OpenAIPromptRendering.parseToolCalls("""
            <tool_call>
            <function=read>
            <parameter=path>
            line one
            line two
            </parameter>
            </function>
            </tool_call>
            """, tools: [try readTool])
        #expect(calls[0].function.arguments == #"{"path":"line one\nline two"}"#)
    }

    @Test("two calls in one reply both parse")
    func parsesTwoCalls() throws {
        let block = """
            <tool_call>
            <function=read>
            <parameter=path>
            a
            </parameter>
            </function>
            </tool_call>
            <tool_call>
            <function=read>
            <parameter=path>
            b
            </parameter>
            </function>
            </tool_call>
            """
        let calls = OpenAIPromptRendering.parseToolCalls(
            block, tools: [try readTool])
        #expect(calls.count == 2)
        #expect(calls[0].id != calls[1].id)
    }

    @Test("a truncated block yields no calls instead of throwing")
    func toleratesTruncation() throws {
        let calls = OpenAIPromptRendering.parseToolCalls(
            "<tool_call>\n<function=read>\n<parameter=path>\n/tmp",
            tools: [try readTool])
        #expect(calls.isEmpty)
    }
}

@Suite("Tool-call streaming gate")
struct ToolCallGateTests {
    @Test("plain text streams straight through")
    func streamsPlainText() {
        var gate = OpenAIPromptRendering.ToolCallGate()
        #expect(gate.admit("hello").delta == "hello")
        #expect(gate.admit("hello there").delta == " there")
    }

    /// A trailing run that could still grow into `<tool_call>` is withheld, so a
    /// half-written tag never reaches the client as content.
    @Test("a partial tag prefix is withheld until it resolves")
    func withholdsPartialTag() {
        var gate = OpenAIPromptRendering.ToolCallGate()
        #expect(gate.admit("done <to").delta == "done ")
        #expect(gate.admit("done <tool").delta == "")
        #expect(gate.admit("done <tomato").delta == "<tomato")
    }

    @Test("a complete tag stops the stream and reports itself")
    func stopsOnCompleteTag() {
        var gate = OpenAIPromptRendering.ToolCallGate()
        let result = gate.admit("ok <tool_call>\n<function=x>")
        #expect(result.delta == "ok ")
        #expect(result.sawToolCall)
        #expect(gate.admit("ok <tool_call>\n<function=x>\n").delta == "")
    }
}
