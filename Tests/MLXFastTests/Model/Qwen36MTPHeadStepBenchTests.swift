import Foundation
import Testing

@testable import MLXLLM
@testable import MLXFastModel

/// Splits one head draft step into host graph build and device execution.
///
/// This is an instrument, not a gate: it asserts only that both halves are
/// positive and finite, and prints the split. The number that matters is the
/// ratio, and a human reads it.
@Suite(.serialized)
struct Qwen36MTPHeadStepBenchTests {
    @Test("head draft step splits into host build and device execute")
    func headStepSplit() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }

        let historyRows = 2_048
        let split = qwen35BenchMTPHeadStep(
            iterations: 32, historyRows: historyRows)

        #expect(split.headBuild > 0)
        #expect(split.headEval > 0)
        #expect(split.projectionBuild > 0)
        #expect(split.projectionEval > 0)

        let cost = Qwen36MTPHeadCost.pinnedQwen38Head
        let projectionBytes = Qwen36MTPHeadCost.projectionBytes(
            rows: 98_336, hiddenSize: 5_120, bits: 4, groupSize: 64)
        let headBytes = cost.headModuleBytes
            + historyRows * cost.headKVBytesPerRow
        let stepSeconds =
            split.headBuild + split.headEval
            + split.projectionBuild + split.projectionEval

        func rate(_ bytes: Int, _ seconds: Double) -> Double {
            Double(bytes) / seconds / 1_000_000_000
        }

        print(String(
            format: """

                [head step @ %d history rows] median of 32
                  head module    build %6.2f ms   eval %6.2f ms   \
                %6.1f GB/s on eval
                  draft project  build %6.2f ms   eval %6.2f ms   \
                %6.1f GB/s on eval
                  step total     %6.2f ms   host share %4.1f%%   \
                %6.1f GB/s overall
                """,
            historyRows,
            1000 * split.headBuild, 1000 * split.headEval,
            rate(headBytes, split.headEval),
            1000 * split.projectionBuild, 1000 * split.projectionEval,
            rate(projectionBytes, split.projectionEval),
            1000 * stepSeconds,
            100 * (split.headBuild + split.projectionBuild) / stepSeconds,
            rate(headBytes + projectionBytes, stepSeconds)))
    }
}
