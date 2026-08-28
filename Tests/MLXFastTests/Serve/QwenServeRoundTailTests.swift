import Foundation
import Testing

@testable import MLXFastHarness

/// The serve stats bar has to separate the worker round from the parent's own
/// between-round work, because only the second half is this plan's to remove.
@Suite
struct QwenServeRoundTailTests {
    @Test("the host tail is every parent segment except the worker round")
    func hostTailExcludesTheWorkerRound() {
        var stats = QwenRuntime.QwenChatTurnStats()
        // `hostTailSeconds` is `seconds - seedPrefillSeconds -
        // workerRoundSeconds` (a residual, not a sum of the four segments
        // below), so the whole turn duration has to be set for the
        // expectation to mean anything: 10.0 - 8.0 == 2.0.
        stats.seconds = 10.0
        stats.workerRoundSeconds = 8.0
        stats.detokenizeSeconds = 1.0
        stats.stopScanSeconds = 0.25
        stats.gateSeconds = 0.5
        stats.streamEmitSeconds = 0.25
        #expect(stats.hostTailSeconds == 2.0)
    }

    @Test("the host tail share is measured against the decode window")
    func hostTailShareUsesTheDecodeWindow() {
        var stats = QwenRuntime.QwenChatTurnStats()
        stats.seconds = 12.0
        stats.seedPrefillSeconds = 2.0
        stats.workerRoundSeconds = 8.0
        stats.detokenizeSeconds = 2.0
        #expect(stats.hostTailShare == 0.2)
    }

    @Test("the share is nil before anything is measured")
    func shareIsAbsentWithoutAWindow() {
        #expect(QwenRuntime.QwenChatTurnStats().hostTailShare == nil)
    }

    @Test("the stats bar names the host tail")
    func statsBarNamesTheHostTail() {
        var stats = QwenRuntime.QwenChatTurnStats()
        stats.seconds = 10.0
        stats.emittedTokens = 100
        stats.rounds = 50
        stats.workerRoundSeconds = 9.0
        stats.detokenizeSeconds = 0.5
        stats.gateSeconds = 0.5
        let bar = QwenRuntime.renderStatsBar(stats, depth: 2)
        #expect(bar.contains("host tail"))
    }
}
