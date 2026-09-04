import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class Qwen4ExpNGramGatherTests: XCTestCase {
    /// The gather must report how much work it did. Without this, no claim
    /// about the n-gram table's cost is measurable -- the path has never had
    /// any instrumentation, and the table's 102.4 GB size has been used as a
    /// proxy for its cost, which the 3.58 MB-per-forward arithmetic refutes.
    func testGatherStatsCountRowsAndPages() throws {
        Qwen4ExpNGramTable.stats.reset()
        let table = try Qwen4ExpNGramTable.inMemoryFixture(rowsPerShard: 1000, dim: 160, shards: 2)

        // Two tokens, 16 heads each -- the real per-token head count.
        let gids: [[Int64]] = [
            (0 ..< 16).map { Int64($0 * 37) },
            (0 ..< 16).map { Int64(1000 + $0 * 41) },
        ]
        let out = table.gather(gids)
        out.eval()

        XCTAssertEqual(out.shape, [2, 16 * 160])
        let s = Qwen4ExpNGramTable.stats.snapshot()
        XCTAssertEqual(s.calls, 1)
        XCTAssertEqual(s.rows, 32, "16 heads x 2 tokens")
        XCTAssertGreaterThan(s.nanos, 0, "the gather must record elapsed time")
        if Qwen4ExpNGramTable.statsEnabled {
            XCTAssertGreaterThan(s.distinctPages, 0, "page accounting must be populated when enabled")
            XCTAssertLessThanOrEqual(s.distinctPages, 32, "at most one page per row")
        }
    }

    /// Real per-forward geometry: 700 tokens x 16 heads = 11,200 rows of 160
    /// bf16 values. Reports gather wall time so the table's cost can be stated
    /// as a fraction of a forward pass instead of guessed from its file size.
    ///
    /// Baseline for interpretation: one layer's routed MoE measures 22.0 ms at
    /// real geometry, and there are 48 layers, so a whole forward's MoE work is
    /// on the order of 1.06 s. The gather runs ONCE per forward (PLE is at
    /// layer 2 only). Anything under ~10 ms here is under 1 percent of the
    /// forward and is not worth optimizing further.
    func testGatherTimingAtRealGeometry() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a real GPU")
        let rowsPerShard = 2_500_012 / 1000  // scaled fixture, same row width
        let table = try Qwen4ExpNGramTable.inMemoryFixture(
            rowsPerShard: rowsPerShard, dim: 160, shards: 8)

        var rng = SystemRandomNumberGenerator()
        let total = Int64(rowsPerShard * 8)
        let gids: [[Int64]] = (0 ..< 700).map { _ in
            (0 ..< 16).map { _ in Int64.random(in: 0 ..< total, using: &rng) }
        }

        _ = table.gather(gids)  // warm: fault the pages in
        Qwen4ExpNGramTable.stats.reset()
        var best = Double.greatestFiniteMagnitude
        for _ in 0 ..< 5 {
            let t0 = Date()
            let out = table.gather(gids)
            out.eval()
            best = min(best, Date().timeIntervalSince(t0))
        }
        let s = Qwen4ExpNGramTable.stats.snapshot()
        print(
            "[ngram] 700 tok x 16 heads = \(s.rows / 5) rows; best \(String(format: "%.3f", best * 1000)) ms; "
                + "distinct 16KiB pages \(s.distinctPages / 5)")
        XCTAssertEqual(s.rows / 5, 11_200)
    }
}
