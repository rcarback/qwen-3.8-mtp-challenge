import Foundation
import Testing

@testable import MLXLLM

/// The decode asyncEval ladder is inert between width 10 and width 511
/// (Qwen35.swift). `MLX_QWEN_MTP_LADDER` selects WHICH rungs fire, never
/// WHETHER any fire, so the band needs its own knob before it can be measured.
///
/// The parser is pure so it can be tested without MLX, a device, or weights.
/// The contract that matters most is the default: an unset variable must
/// reproduce the shipped condition exactly.
@Suite
struct WideVerifyLadderBandTests {
    @Test("unset and empty reproduce the shipped band")
    func defaultsToNine() {
        #expect(qwen35ParseLadderMaxWidth(nil) == 9)
        #expect(qwen35ParseLadderMaxWidth("") == 9)
    }

    @Test("a positive integer raises the band")
    func parsesPositiveIntegers() {
        #expect(qwen35ParseLadderMaxWidth("32") == 32)
        #expect(qwen35ParseLadderMaxWidth("1") == 1)
        #expect(qwen35ParseLadderMaxWidth("511") == 511)
    }

    @Test("junk and non-positive values fall back to the shipped band")
    func rejectsJunk() {
        #expect(qwen35ParseLadderMaxWidth("dense") == 9)
        #expect(qwen35ParseLadderMaxWidth("0") == 9)
        #expect(qwen35ParseLadderMaxWidth("-4") == 9)
        #expect(qwen35ParseLadderMaxWidth("32.5") == 9)
    }
}
