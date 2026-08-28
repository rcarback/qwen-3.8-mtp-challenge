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

    /// The pre-optimization gate, kept verbatim as the oracle. If the fast
    /// gate ever disagrees with it on any input, the fast gate is wrong.
    private struct NaiveGate {
        static let marker = "<tool_call>"
        var emitted = 0
        var stopped = false

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
            var held = 0
            for length in stride(
                from: min(Self.marker.count - 1, full.count),
                through: 1, by: -1)
            {
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
            return String(text[start ..< end])
        }
    }

    /// Prefix sequences that exercise every branch: plain growth, a partial
    /// marker that resolves, a partial marker that does not, multi-byte
    /// characters, a seam repair, an empty reply, a marker at position zero,
    /// and a string that SHRINKS.
    ///
    /// The shrinking case is not hypothetical. A stop-string hit truncates
    /// `full` before the gate sees it, so the gate can be handed a string
    /// shorter than the count it already emitted; any implementation that
    /// walks backward by `emitted` traps there.
    private static let gateCorpora: [[String]] = [
        ["", "a", "ab", "abc"],
        ["ok <", "ok <t", "ok <to", "ok <tool_call>", "ok <tool_call>{}"],
        ["x<", "x<y", "x<yz"],
        ["caf\u{FFFD}", "café", "café au lait"],
        ["\u{1F600}", "\u{1F600}\u{1F601}", "\u{1F600}\u{1F601} hi"],
        ["<tool_call>", "<tool_call>{}"],
        [""],
        ["a long enough reply to matter", "ab"],
        ["hello world", "he", ""],
        ["ok x<tool_call>", "ok x<tool_call>{}"],
    ]

    @Test("the fast gate agrees with the naive gate on every corpus")
    func gateMatchesTheNaiveOracle() {
        for corpus in Self.gateCorpora {
            var fast = OpenAIPromptRendering.ToolCallGate()
            var naive = NaiveGate()
            for (step, full) in corpus.enumerated() {
                let fastResult = fast.admit(full)
                let naiveResult = naive.admit(full)
                #expect(
                    fastResult.delta == naiveResult.delta,
                    "delta differs at step \(step) of \(corpus)")
                #expect(
                    fastResult.sawToolCall == naiveResult.sawToolCall,
                    "sawToolCall differs at step \(step) of \(corpus)")
            }
        }
    }

    @Test("a non-streaming request with no stop strings decodes once")
    func skipsThePerRoundDecode() {
        var stage = QwenRuntime.ServeRoundText(
            stopStrings: [], streaming: false)
        #expect(!stage.runsPerRound)
        var calls = 0
        for _ in 0 ..< 5 {
            _ = stage.advance { calls += 1; return "text" }
        }
        #expect(calls == 0)
        let final = stage.finish { calls += 1; return "text <tool_call>{}" }
        #expect(calls == 1)
        #expect(final.sawToolCall)
        #expect(final.full == "text <tool_call>{}")
    }

    @Test("streaming or stop strings keep the per-round decode")
    func keepsThePerRoundDecodeWhenNeeded() {
        #expect(QwenRuntime.ServeRoundText(
            stopStrings: [], streaming: true).runsPerRound)
        #expect(QwenRuntime.ServeRoundText(
            stopStrings: ["END"], streaming: false).runsPerRound)
    }

    @Test("finish is idempotent for a stage that already ran per round")
    func finishDoesNotRedecodeAfterAPerRoundStage() {
        var stage = QwenRuntime.ServeRoundText(
            stopStrings: [], streaming: true)
        _ = stage.advance { "hello" }
        var calls = 0
        let final = stage.finish { calls += 1; return "hello" }
        #expect(calls == 0)
        #expect(final.full == "hello")
    }
}
