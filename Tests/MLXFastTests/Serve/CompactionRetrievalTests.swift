import Foundation
import Testing

@testable import MLXFastHarness

/// Compaction removes bytes from the PROMPT, not from the conversation. These
/// cover the path that makes that distinction true: the original is retained,
/// the stub names the handle that reaches it, and the model's call resolves.
///
/// Every failure here is SILENT in production -- a stub whose handle does not
/// resolve still renders, still prefills, and still produces an answer, just a
/// worse one built on text the model could not reach.
@Suite(.serialized)
struct CompactionRetrievalTests {

    private func freshStore() -> CompactionStore {
        CompactionStore.shared.reset()
        return CompactionStore.shared
    }

    @Test("a stubbed result is retained under the handle its stub advertises")
    func stubHandleResolves() {
        let store = freshStore()
        let original = String(repeating: "tool output line\n", count: 400)
        let messages = [
            ChatMessage.toolResult(id: "call_1", text: original),
            // Settled: an assistant turn follows, so this result is history.
            ChatMessage.assistantToolCall([]),
        ]
        let result = QwenRuntime.compactSettledToolResults(
            messages, minimumCharacters: 2048)
        #expect(result.stubbed == 1)

        // The handle in the stub text must be the one the store answers to.
        let stub = result.messages[0].content?.text ?? ""
        let handle = QwenRuntime.handle(for: original)
        #expect(stub.contains(handle))
        #expect(store.get(handle: handle) == original)
        #expect(QwenRuntime.resolveExpansion(handle: handle) == original)
    }

    @Test("an unsettled result is left alone")
    func freshResultSurvives() {
        _ = freshStore()
        let original = String(repeating: "x", count: 8192)
        // No assistant turn after it: this is the result the model must act on
        // THIS turn, so compacting it would be deleting context, not compacting.
        let messages = [ChatMessage.toolResult(id: "call_1", text: original)]
        let result = QwenRuntime.compactSettledToolResults(
            messages, minimumCharacters: 2048)
        #expect(result.stubbed == 0)
        #expect(result.messages[0].content?.text == original)
    }

    @Test("a result below the size floor is left alone")
    func smallResultSurvives() {
        _ = freshStore()
        let messages = [
            ChatMessage.toolResult(id: "call_1", text: "short"),
            ChatMessage.assistantToolCall([]),
        ]
        #expect(QwenRuntime.compactSettledToolResults(
            messages, minimumCharacters: 2048).stubbed == 0)
    }

    @Test("compaction preserves tool_call_id")
    func toolCallIdSurvives() {
        _ = freshStore()
        let messages = [
            ChatMessage.toolResult(
                id: "call_abc", text: String(repeating: "y", count: 4096)),
            ChatMessage.assistantToolCall([]),
        ]
        let out = QwenRuntime.compactSettledToolResults(
            messages, minimumCharacters: 2048).messages
        // An orphaned tool message is invalid on the wire and would strand the
        // assistant tool_calls entry that refers to it.
        #expect(out[0].toolCallId == "call_abc")
        #expect(out[0].role == "tool")
    }

    @Test("a missing handle answers the model instead of throwing")
    func missingHandleDegrades() {
        _ = freshStore()
        let answer = QwenRuntime.resolveExpansion(handle: "deadbeefdeadbeef")
        // Evicted, or produced before this server started. A failed retrieval
        // costs answer quality; it must never cost the request.
        #expect(answer.contains("deadbeefdeadbeef"))
        #expect(answer.lowercased().contains("re-run"))
    }

    @Test("the handle argument is parsed, and malformed input degrades")
    func handleArgumentParsing() {
        #expect(QwenRuntime.handleArgument(#"{"handle":"abc123"}"#) == "abc123")
        // The model writes this string. Malformed input must produce a
        // "no such handle" answer it can recover from, not a 500.
        #expect(QwenRuntime.handleArgument("not json") == "")
        #expect(QwenRuntime.handleArgument(#"{"other":"x"}"#) == "")
    }

    @Test("the expand tool schema is well formed and self-identifying")
    func expandSchemaRoundTrips() {
        guard let schema = QwenRuntime.expandToolSchema() else {
            Issue.record("expand schema failed to parse")
            return
        }
        // isExpandTool drives whether the tool gets injected twice, so the
        // schema and the recogniser have to agree.
        #expect(QwenRuntime.isExpandTool(schema))
    }

    @Test("identical content yields one stable handle")
    func handleIsContentDerived() {
        let a = String(repeating: "z", count: 3000)
        // Content-derived and therefore byte-identical across turns, which is
        // what keeps a compacted prefix stable rather than churning.
        #expect(QwenRuntime.handle(for: a) == QwenRuntime.handle(for: a))
        #expect(QwenRuntime.handle(for: a) != QwenRuntime.handle(for: a + "!"))
    }
}
