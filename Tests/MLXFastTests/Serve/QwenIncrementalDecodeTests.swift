import Foundation
import Testing

@testable import MLXFastHarness

/// The splice must reproduce the whole-prefix decode CHARACTER FOR CHARACTER
/// at every step. Anything less changes what the server returns.
@Suite
struct QwenIncrementalDecodeTests {
    /// A byte-level decoder stand-in: each token id maps to a byte, and the
    /// text is the UTF-8 interpretation of the concatenated bytes. This has
    /// the property that matters -- a character can straddle several tokens --
    /// without needing a model.
    private static func byteDecode(_ tokens: ArraySlice<Int>) -> String {
        String(decoding: tokens.map { UInt8($0 & 0xFF) }, as: UTF8.self)
    }

    private static let sequences: [[Int]] = [
        Array("hello world".utf8).map(Int.init),
        Array("café au lait".utf8).map(Int.init),
        Array("\u{1F600}\u{1F601}\u{1F602} done".utf8).map(Int.init),
        Array(repeating: 0x41, count: 300),
        [],
    ]

    @Test("the splice equals the whole-prefix decode at every prefix")
    func spliceMatchesWholePrefixDecode() {
        for sequence in Self.sequences {
            var incremental = QwenRuntime.IncrementalDetokenizer(window: 16)
            for length in 0 ... sequence.count {
                let prefix = Array(sequence.prefix(length))
                let expected = Self.byteDecode(prefix[...])
                let actual = incremental.text(for: prefix) {
                    Self.byteDecode($0)
                }
                #expect(
                    actual == expected,
                    "prefix of length \(length) diverged")
            }
        }
    }

    @Test("a window of one is still exact, only slower to converge")
    func exactAtTheSmallestWindow() {
        var incremental = QwenRuntime.IncrementalDetokenizer(window: 1)
        let sequence = Array("café".utf8).map(Int.init)
        for length in 0 ... sequence.count {
            let prefix = Array(sequence.prefix(length))
            #expect(
                incremental.text(for: prefix) { Self.byteDecode($0) }
                    == Self.byteDecode(prefix[...]))
        }
    }

    /// The stand-in decoder above proves the splice arithmetic. This proves it
    /// against the tokenizer the server actually uses, which is the only thing
    /// that can reject the four-byte argument for real. Opt-in: it needs the
    /// weights directory for the tokenizer files, but it loads no model.
    @Test("the splice equals the real tokenizer's whole-prefix decode")
    func spliceMatchesTheRealTokenizer() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            let weights = env["MLXFAST_QWEN_PREFILL_WEIGHTS"]
        else { return }
        let tokenizer = try QwenRuntime.loadLocalTokenizer(at: weights)
        let sample = """
            Here is a mixed sample: café, naïve, 日本語のテキスト, \
            emoji \u{1F600}\u{1F680}, and a fenced block:
            ```swift
            let x = 1  // comment
            ```
            """
        let tokens = tokenizer.encode(text: sample, addSpecialTokens: false)
        var incremental = QwenRuntime.IncrementalDetokenizer(window: 16)
        for length in 0 ... tokens.count {
            let prefix = Array(tokens.prefix(length))
            let expected = tokenizer.decode(
                tokens: prefix, skipSpecialTokens: true)
            let actual = incremental.text(for: prefix) {
                tokenizer.decode(tokens: Array($0), skipSpecialTokens: true)
            }
            #expect(actual == expected, "prefix of length \(length) diverged")
        }
    }
}
