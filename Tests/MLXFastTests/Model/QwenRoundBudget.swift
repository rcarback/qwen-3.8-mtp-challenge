import Foundation

// Phase-0 budget for the decode round path.
//
// TEST-TARGET ONLY. This is a reader for two instruments that already exist,
// not a shipped feature: `mtp-timed --output` writes the report
// (Sources/MLXFastCLI/main.swift:2139-2185) and the session writes the phase
// trace when MLX_QWEN_MTP_TRACE is set
// (Sources/MLXFastModel/Qwen36MTPBlockSession.swift:1714, :2837-2897).

/// The timing half of a `mtp-timed` report.
struct QwenTimedReportTiming {
    let decodeSeconds: Double
    /// Absent from reports whose run predates the measurement, and absent is
    /// zero rather than an error: the key is deliberately omitted rather than
    /// written as zero (main.swift:2149-2153).
    let seedPrefillSeconds: Double
    let decodeTokenCount: Int
    /// One entry per round, in order.
    let blockRequestSeconds: [Double]
}

/// One `mtp-trace:` line, reduced to its numeric fields.
struct QwenTraceRound {
    let round: Int
    let fields: [String: Double]
}

struct QwenRoundBudget {
    let seedPrefillSeconds: Double
    let decodeTokenCount: Int
    /// Lower median over rounds after the first. The first round is a measured
    /// one-time post-prefill warmup and the ranked stall guardrail excludes it
    /// for the same reason (main.swift:2156-2170).
    let parentRoundSecondsMedian: Double
    let workerRoundSecondsMedian: Double
    /// Parent-observed round wall minus worker-observed round wall. This IS
    /// the protocol cost: encode, pipe, decode, watchdog, both ways.
    let protocolSecondsMedian: Double
    /// Window minus prefill minus every round request. This is the parent's
    /// own between-round work, during which the worker is blocked on read and
    /// the GPU is idle.
    let parentTailSeconds: Double
    /// Lower median of every microsecond field the trace carries.
    let workerPhaseMedians: [String: Double]
}

enum QwenRoundBudgetError: Error, CustomStringConvertible {
    case malformedReport(String)

    var description: String {
        switch self {
        case .malformedReport(let detail):
            return "malformed mtp-timed report: \(detail)"
        }
    }
}

func parseQwenTimedReportTiming(
    _ data: Data
) throws -> QwenTimedReportTiming {
    guard let root = try JSONSerialization.jsonObject(with: data)
        as? [String: Any]
    else {
        throw QwenRoundBudgetError.malformedReport("top level is not an object")
    }
    guard let decodeSeconds = root["decode_seconds"] as? Double else {
        throw QwenRoundBudgetError.malformedReport(
            "no decode_seconds; this is a mtp-verify report, not a timed one")
    }
    return QwenTimedReportTiming(
        decodeSeconds: decodeSeconds,
        seedPrefillSeconds: (root["seed_prefill_seconds"] as? Double) ?? 0,
        decodeTokenCount: (root["decode_token_count"] as? Int) ?? 0,
        blockRequestSeconds:
            (root["block_request_seconds"] as? [Double]) ?? [])
}

func parseQwenTraceRounds(_ text: String) -> [QwenTraceRound] {
    var rounds: [QwenTraceRound] = []
    for line in text.split(separator: "\n") {
        guard line.hasPrefix("mtp-trace:") else { continue }
        var fields: [String: Double] = [:]
        var round = 0
        for token in line.split(separator: " ") {
            let parts = token.split(separator: "=", maxSplits: 1)
            guard parts.count == 2, let value = Double(parts[1]) else {
                continue
            }
            if parts[0] == "round" {
                round = Int(value)
            } else {
                fields[String(parts[0])] = value
            }
        }
        rounds.append(QwenTraceRound(round: round, fields: fields))
    }
    return rounds
}

/// Lower median: the same rule the report uses, so one definition of "p50"
/// exists across the budget and the payload
/// (Sources/MLXFastTrustedHarness/QwenRuntimeMTP.swift:407-408).
func qwenLowerMedian(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    return sorted[(sorted.count - 1) / 2]
}

func makeQwenRoundBudget(
    report: QwenTimedReportTiming,
    trace: [QwenTraceRound]
) -> QwenRoundBudget {
    let parentAfterFirst = Array(report.blockRequestSeconds.dropFirst())
    let parentMedian = qwenLowerMedian(parentAfterFirst)
    let workerRounds = trace
        .compactMap { $0.fields["round_us"].map { $0 / 1_000_000 } }
    let workerMedian = qwenLowerMedian(workerRounds)
    var phaseMedians: [String: Double] = [:]
    var names = Set<String>()
    for entry in trace { names.formUnion(entry.fields.keys) }
    for name in names where name.hasSuffix("_us") {
        phaseMedians[name] = qwenLowerMedian(
            trace.dropFirst().compactMap { $0.fields[name] })
    }
    return QwenRoundBudget(
        seedPrefillSeconds: report.seedPrefillSeconds,
        decodeTokenCount: report.decodeTokenCount,
        parentRoundSecondsMedian: parentMedian,
        workerRoundSecondsMedian: workerMedian,
        // Only meaningful when both instruments describe the same run. A zero
        // worker median means the trace was empty, and the caller sees that
        // in the rendered table rather than a fake protocol cost.
        protocolSecondsMedian: workerMedian > 0
            ? parentMedian - workerMedian : 0,
        parentTailSeconds: report.decodeSeconds - report.seedPrefillSeconds
            - report.blockRequestSeconds.reduce(0, +),
        workerPhaseMedians: phaseMedians)
}

func renderQwenRoundBudget(_ budget: QwenRoundBudget) -> String {
    func milliseconds(_ seconds: Double) -> String {
        String(format: "%9.3f ms", seconds * 1000)
    }
    var lines = [
        "decode round budget (medians over rounds after the first)",
        "  seed prefill (whole window) \(milliseconds(budget.seedPrefillSeconds))",
        "  parent round                \(milliseconds(budget.parentRoundSecondsMedian))",
        "  worker round                \(milliseconds(budget.workerRoundSecondsMedian))",
        "  protocol                    \(milliseconds(budget.protocolSecondsMedian))",
        "  parent tail (whole window)  \(milliseconds(budget.parentTailSeconds))",
        "  worker phases:",
    ]
    for name in budget.workerPhaseMedians.keys.sorted() {
        let value = budget.workerPhaseMedians[name] ?? 0
        let padded = name.padding(
            toLength: Swift.max(name.count, 20), withPad: " ",
            startingAt: 0)
        lines.append(
            "    \(padded) " + String(format: "%9.3f ms", value / 1000))
    }
    return lines.joined(separator: "\n")
}
