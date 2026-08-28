import Foundation
import Testing

@testable import MLXFastHarness

/// The serve round loop's text stage, driven with a fake decoder so it is
/// testable without a tokenizer or a model.
@Suite
struct QwenServeRoundTextTests {
    /// Feeds successive whole-prefix decodes, the way the real loop does.
    private struct FakeDecoder {
        var steps: [String]
        var index = 0
        mutating func next() -> String {
            defer { index += 1 }
            return index < steps.count ? steps[index] : (steps.last ?? "")
        }
    }

    @Test("the delta is the text not yet handed to the caller")
    func emitsOnlyNewText() {
        var stage = QwenRuntime.ServeRoundText(
            stopStrings: [], streaming: true)
        var decoder = FakeDecoder(steps: ["Hel", "Hello", "Hello there"])
        #expect(stage.advance { decoder.next() }.delta == "Hel")
        #expect(stage.advance { decoder.next() }.delta == "lo")
        #expect(stage.advance { decoder.next() }.delta == " there")
    }

    @Test("a partial tool-call marker is held back until it resolves")
    func holdsBackAPartialMarker() {
        var stage = QwenRuntime.ServeRoundText(
            stopStrings: [], streaming: true)
        var decoder = FakeDecoder(steps: ["ok <too", "ok <tool_call>{}"])
        #expect(stage.advance { decoder.next() }.delta == "ok ")
        let second = stage.advance { decoder.next() }
        #expect(second.sawToolCall)
        #expect(second.delta == "")
    }

    @Test("a stop string truncates the text and reports the hit")
    func truncatesAtAStopString() {
        var stage = QwenRuntime.ServeRoundText(
            stopStrings: ["END"], streaming: false)
        var decoder = FakeDecoder(steps: ["hello", "hello END tail"])
        _ = stage.advance { decoder.next() }
        let second = stage.advance { decoder.next() }
        #expect(second.hitStop)
        #expect(second.full == "hello ")
    }

    @Test("a multi-byte character completed by a later round is not doubled")
    func survivesASeamRepair() {
        var stage = QwenRuntime.ServeRoundText(
            stopStrings: [], streaming: true)
        var decoder = FakeDecoder(steps: ["caf\u{FFFD}", "café", "café au"])
        _ = stage.advance { decoder.next() }
        _ = stage.advance { decoder.next() }
        #expect(stage.advance { decoder.next() }.delta == " au")
    }
}
