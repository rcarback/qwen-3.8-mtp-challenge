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
}
