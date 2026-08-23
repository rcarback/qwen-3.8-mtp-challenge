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
            messages: [message("user", "hi")], tools: nil)
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
            tools: nil)
        #expect(rendered.hasPrefix("<|im_start|>system\nbe terse<|im_end|>\n"))
    }

    @Test("an empty system message is omitted entirely")
    func skipsEmptySystem() throws {
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [try message("system", "   "), try message("user", "hi")],
            tools: nil)
        #expect(!rendered.contains("<|im_start|>system"))
    }

    @Test("tools render into a leading system block in declared order")
    func rendersTools() throws {
        let tool = try OrderedJSON.parse(
            #"{"type":"function","function":{"name":"read","description":"d"}}"#)
        let rendered = try OpenAIPromptRendering.renderPrompt(
            messages: [try message("user", "hi")], tools: [tool])
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
            tools: [tool])
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
            tools: nil)
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
            tools: nil)
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
            _ = try OpenAIPromptRendering.renderPrompt(messages: [], tools: nil)
        }
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
