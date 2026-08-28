import Foundation
import Testing

// The reducer lives in this same test target (QwenRoundBudget.swift, added in
// Step 3), so no import of a product module is needed. It reads files; it
// links nothing.

/// The phase-0 budget is arithmetic over two instruments that already exist:
/// the `mtp-timed` report and the session's `MLX_QWEN_MTP_TRACE` lines. These
/// tests pin the arithmetic on synthetic inputs so the real run only has to
/// supply files.
@Suite
struct QwenRoundBudgetTests {
    static let sampleReport = """
        {
          "decode_seconds": 10.0,
          "seed_prefill_seconds": 2.0,
          "decode_token_count": 4,
          "block_request_seconds": [3.0, 1.0, 1.5, 2.0]
        }
        """

    static let sampleTrace = """
        mtp-trace: round=1 d=2 acc=1 draft_build_us=1000 verify_build_us=2000 \
        eval_wall_us=3000 readout_us=100 commit_us=200 upkeep_us=50 \
        round_us=900000 host_thread_cpu_ns=1234
        mtp-anchor: round=1 d=2 acc=1 pid=7 t_round0=0 t_draft_built=1 \
        t_snapshot_done=2 t_verify_built=3
        mtp-trace: round=2 d=2 acc=0 draft_build_us=1100 verify_build_us=2100 \
        eval_wall_us=3100 readout_us=110 commit_us=210 upkeep_us=60 \
        round_us=1100000 host_thread_cpu_ns=2345
        mtp-row: pos=5 ids=1,2 v=0x1p+0,0x1p-1
        """

    @Test("report timing is read from the published snake_case keys")
    func readsReportTiming() throws {
        let timing = try parseQwenTimedReportTiming(
            Data(Self.sampleReport.utf8))
        #expect(timing.decodeSeconds == 10.0)
        #expect(timing.seedPrefillSeconds == 2.0)
        #expect(timing.decodeTokenCount == 4)
        #expect(timing.blockRequestSeconds == [3.0, 1.0, 1.5, 2.0])
    }

    @Test("a report without a prefill key reads as zero, not as a failure")
    func toleratesAbsentPrefillKey() throws {
        let timing = try parseQwenTimedReportTiming(Data("""
            {"decode_seconds": 1.0, "decode_token_count": 1,
             "block_request_seconds": [1.0]}
            """.utf8))
        #expect(timing.seedPrefillSeconds == 0.0)
    }

    @Test("only mtp-trace lines are parsed, and every numeric field is kept")
    func parsesTraceRounds() {
        let rounds = parseQwenTraceRounds(Self.sampleTrace)
        #expect(rounds.count == 2)
        #expect(rounds[0].round == 1)
        #expect(rounds[0].fields["eval_wall_us"] == 3000)
        #expect(rounds[0].fields["round_us"] == 900_000)
        #expect(rounds[1].fields["readout_us"] == 110)
        #expect(rounds[0].fields["pid"] == nil)
    }

    @Test("the budget splits parent wall into worker wall plus protocol")
    func splitsProtocolCost() throws {
        let budget = makeQwenRoundBudget(
            report: try parseQwenTimedReportTiming(
                Data(Self.sampleReport.utf8)),
            trace: parseQwenTraceRounds(Self.sampleTrace))
        // Parent rounds after the first: 1.0, 1.5, 2.0 -> lower median 1.5.
        #expect(budget.parentRoundSecondsMedian == 1.5)
        // Worker rounds: 0.9, 1.1 -> lower median 0.9.
        #expect(budget.workerRoundSecondsMedian == 0.9)
        #expect(abs(budget.protocolSecondsMedian - 0.6) < 1e-9)
    }

    @Test("the parent tail is the window minus prefill minus every round")
    func computesParentTail() throws {
        let budget = makeQwenRoundBudget(
            report: try parseQwenTimedReportTiming(
                Data(Self.sampleReport.utf8)),
            trace: [])
        // 10.0 - 2.0 - (3.0 + 1.0 + 1.5 + 2.0) = 0.5
        #expect(abs(budget.parentTailSeconds - 0.5) < 1e-9)
        #expect(budget.workerRoundSecondsMedian == 0.0)
    }

    @Test("the rendered table names every suspect line")
    func rendersEverySuspect() throws {
        let text = renderQwenRoundBudget(makeQwenRoundBudget(
            report: try parseQwenTimedReportTiming(
                Data(Self.sampleReport.utf8)),
            trace: parseQwenTraceRounds(Self.sampleTrace)))
        for name in [
            "seed prefill", "parent round", "worker round", "protocol",
            "parent tail", "eval_wall_us", "verify_build_us", "readout_us",
            "commit_us", "upkeep_us",
        ] {
            #expect(text.contains(name), "budget omits \(name)")
        }
    }

    /// Reads a real pair of instrument files and prints the budget. Opt-in,
    /// like every other measurement test in this target: it does nothing
    /// unless both paths are exported.
    ///
    ///     mlxfast-swift mtp-timed --mtp-head <head> --golden <golden> \
    ///       --mtp-depth 0 --tokens 512 --output /tmp/timed.json
    ///
    /// To collect the matching trace the worker must be able to write a file,
    /// which the derived Seatbelt profile forbids
    /// (Sources/MLXFastCLI/main.swift:2784-2785), so that run additionally
    /// needs MLXFAST_NO_SANDBOX=1 MLX_QWEN_MTP_TRACE=1
    /// MLX_QWEN_MTP_TRACE_PATH=/tmp/trace.txt. That override exists here to
    /// READ TIMERS ONLY. No number produced under it may be reported as an
    /// improvement.
    @Test("print the budget for a real run")
    func printsRealBudget() throws {
        let env = ProcessInfo.processInfo.environment
        guard let reportPath = env["MLXFAST_ROUND_BUDGET_REPORT"] else {
            return
        }
        let tracePath = env["MLXFAST_ROUND_BUDGET_TRACE"]
        let timing = try parseQwenTimedReportTiming(
            try Data(contentsOf: URL(fileURLWithPath: reportPath)))
        let trace = tracePath.flatMap {
            try? String(contentsOfFile: $0, encoding: .utf8)
        }.map(parseQwenTraceRounds) ?? []
        print("\n" + renderQwenRoundBudget(
            makeQwenRoundBudget(report: timing, trace: trace)) + "\n")
    }
}
