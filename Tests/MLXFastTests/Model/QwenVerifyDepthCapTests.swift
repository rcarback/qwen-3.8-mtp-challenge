import Foundation
import Testing

@testable import MLXFastCore
@testable import MLXFastModel
@testable import MLXLLM
@testable import MLXLMCommon

/// Contract tests for the runtime-settable verify-depth caps.
///
/// The caps are read from the environment once at process start, so these
/// exercise the PURE parser and the derived ceiling. Nothing here loads a
/// model or touches a GPU stream.
@Suite
struct QwenVerifyDepthCapTests {

    /// The ceiling is the narrowest of three independently configured
    /// bounds. At the shipped settings the QMV width bound is what binds:
    /// width 9 means depth 8, which happens to equal the trusted ceiling.
    @Test("the proven-exact ceiling is the narrowest of its three bounds")
    func ceilingIsTheNarrowestBound() {
        let expected = Swift.max(
            0,
            Swift.min(
                Qwen36MTPLimits.maxDepth,
                Swift.min(
                    Qwen35CustomQMV.maxWidth - 1,
                    wideDecodeExactnessMaxQueryRows - 1)))
        #expect(Qwen36MTPBlockSession.provenExactDepthCeiling == expected)
        #expect(Qwen36MTPBlockSession.provenExactDepthCeiling >= 0)
    }

    /// A depth-`d` round projects `d + 1` rows, and rows past the replica's
    /// width bound leave the per-row-exact dispatch. This is the coupling
    /// that measurement found and that a comment alone would not enforce.
    @Test("the ceiling never admits a verify width the QMV replica drops")
    func ceilingRespectsTheQMVWidthBound() {
        #expect(
            Qwen36MTPBlockSession.provenExactDepthCeiling + 1
                <= Qwen35CustomQMV.maxWidth)
    }

    /// The sdpa exactness chunk must cover the widest round the cap allows,
    /// or the widest rounds would silently take the unchunked path.
    @Test("the ceiling never admits a verify width the sdpa chunk drops")
    func ceilingRespectsTheChunkBound() {
        #expect(
            Qwen36MTPBlockSession.provenExactDepthCeiling + 1
                <= wideDecodeExactnessMaxQueryRows)
    }

    @Test("every malformed override falls back to the supplied default")
    func parserRejectsMalformedInput() {
        for raw in [nil, "", " ", "five", "-1", "3.5", "7x", "0x4"] {
            #expect(
                Qwen36MTPBlockSession.parseDepthCap(raw, fallback: 5) == 5,
                "a malformed override must not move the cap")
        }
    }

    /// Out of range is refused rather than clamped: a caller who asked for a
    /// depth the process cannot run exactly has a configuration error, and
    /// silently serving a different number would hide it.
    @Test("an override above the proven ceiling falls back")
    func parserRefusesOverTheCeiling() {
        let ceiling = Qwen36MTPBlockSession.provenExactDepthCeiling
        #expect(
            Qwen36MTPBlockSession.parseDepthCap("\(ceiling + 1)", fallback: 5) == 5)
        #expect(
            Qwen36MTPBlockSession.parseDepthCap("99", fallback: 7) == 7)
    }

    @Test("a well-formed in-range override is taken verbatim")
    func parserAcceptsInRange() {
        for value in 0 ... Qwen36MTPBlockSession.provenExactDepthCeiling {
            #expect(
                Qwen36MTPBlockSession.parseDepthCap("\(value)", fallback: 5)
                    == value)
        }
        #expect(Qwen36MTPBlockSession.parseDepthCap(" 3 ", fallback: 5) == 3)
    }

    /// 0 is a legal cap and means "never draft", which is the serial control.
    /// It must be reachable, and it must not be confused with a parse failure.
    @Test("zero is a legal cap and is distinguishable from a parse failure")
    func zeroIsReachable() {
        #expect(Qwen36MTPBlockSession.parseDepthCap("0", fallback: 5) == 0)
        #expect(Qwen36MTPBlockSession.parseDepthCap("nope", fallback: 5) == 5)
    }
}
