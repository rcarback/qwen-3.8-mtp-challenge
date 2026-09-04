import Foundation
import MLX
import XCTest

@testable import MLXFastModel

final class MoEWorkQueueTests: XCTestCase {
    func testRowOffsetsOnSortedIndices() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a real GPU")

        // Experts 0 and 3 are empty; expert 1 has 2 rows, expert 2 has 3.
        let sorted = MLXArray([Int32(1), 1, 2, 2, 2])
        let offsets = MoEWorkQueue.rowOffsets(sortedIndices: sorted, numExperts: 4)
        offsets.eval()
        XCTAssertEqual(offsets.asArray(Int32.self), [0, 0, 2, 5, 5])
    }

    func testRowOffsetsAllRowsOneExpert() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a real GPU")

        let sorted = MLXArray([Int32(2), 2, 2, 2])
        let offsets = MoEWorkQueue.rowOffsets(sortedIndices: sorted, numExperts: 4)
        offsets.eval()
        XCTAssertEqual(offsets.asArray(Int32.self), [0, 0, 0, 4, 4])
    }

    func testBlockOffsetsRoundsUpPerExpert() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a real GPU")

        // Counts 0, 2, 3, 0 at blockRows 2 -> blocks 0, 1, 2, 0 -> prefix 0,0,1,3,3
        let rowOffsets = MLXArray([Int32(0), 0, 2, 5, 5])
        let blocks = MoEWorkQueue.blockOffsets(rowOffsets: rowOffsets, blockRows: 2)
        blocks.eval()
        XCTAssertEqual(blocks.asArray(Int32.self), [0, 0, 1, 3, 3])
    }

    /// The other three tests all run at numExperts = 4, far from the real
    /// 512-expert geometry: there `grid == threadGroup == (5,1,1)`, so Metal
    /// never pads the dispatch and the kernel's `e > n_experts` overflow
    /// guard is never exercised, and the binary search never runs over more
    /// than 5 rows. At the production geometry `grid = (513,1,1)` dispatches
    /// across three threadgroups of 256 (the last holding a single active
    /// thread), which is exactly the case that guard exists for, and the
    /// search runs over thousands of rows.
    ///
    /// This test builds a skewed, gappy 512-expert distribution — a long run
    /// of idle experts, a body of lightly (and irregularly) loaded experts,
    /// and one heavily overloaded expert — matching the shape real routing on
    /// this tower produces (147-239 of 512 experts idle, with a large
    /// imbalance between the lightest and heaviest loaded expert), and checks
    /// the kernel's full [513]-element output against a plain-Swift
    /// prefix-sum reference, not just spot values.
    func testRowOffsetsAtProductionScale() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a real GPU")

        let numExperts = 512

        // Build a skewed, gappy per-expert row-count distribution:
        //  - experts 0..<180: idle (a long run of empty experts up front).
        //  - experts 180..<480: a lightly-loaded body with irregular gaps
        //    (every 11th expert idle, the rest holding 1...9 rows).
        //  - expert 480: one heavily overloaded expert (5469 rows).
        //  - experts 481..<512: a sparse idle/lightly-loaded tail.
        var counts = [Int](repeating: 0, count: numExperts)
        for e in 180..<480 {
            let bucket = e % 11
            counts[e] = bucket == 0 ? 0 : 1 + (e % 9)
        }
        counts[480] = 5469
        for e in stride(from: 481, to: numExperts, by: 1) {
            counts[e] = (e % 4 == 0) ? 25 : 0
        }

        let totalRows = counts.reduce(0, +)
        let idleExperts = counts.filter { $0 == 0 }.count
        XCTAssertTrue(
            (147...239).contains(idleExperts),
            "test fixture should match the real idle-expert range, got \(idleExperts)")

        // Expand counts into a non-decreasing sorted expert-id array, exactly
        // the shape `gatherSort`'s second return value takes.
        var sortedIds = [Int32]()
        sortedIds.reserveCapacity(totalRows)
        for e in 0..<numExperts {
            sortedIds.append(contentsOf: repeatElement(Int32(e), count: counts[e]))
        }
        XCTAssertEqual(sortedIds.count, totalRows)

        // Plain-Swift exclusive prefix sum over counts, the reference the
        // kernel's binary search is checked against.
        var expectedRowOffsets = [Int32](repeating: 0, count: numExperts + 1)
        var running: Int32 = 0
        for e in 0..<numExperts {
            expectedRowOffsets[e] = running
            running += Int32(counts[e])
        }
        expectedRowOffsets[numExperts] = running

        let sorted = MLXArray(sortedIds)
        let offsets = MoEWorkQueue.rowOffsets(sortedIndices: sorted, numExperts: numExperts)
        offsets.eval()
        let actualRowOffsets = offsets.asArray(Int32.self)

        XCTAssertEqual(actualRowOffsets.count, numExperts + 1)
        XCTAssertEqual(actualRowOffsets, expectedRowOffsets)
        XCTAssertEqual(actualRowOffsets.last, Int32(totalRows))

        // blockOffsets on this same real-scale rowOffsets.
        let blockRows = 16
        let expectedTotalBlocks = counts.reduce(0) { $0 + ($1 + blockRows - 1) / blockRows }

        let blocks = MoEWorkQueue.blockOffsets(rowOffsets: offsets, blockRows: blockRows)
        blocks.eval()
        let actualBlocks = blocks.asArray(Int32.self)

        XCTAssertEqual(actualBlocks.count, numExperts + 1)
        XCTAssertEqual(actualBlocks.last, Int32(expectedTotalBlocks))
    }
}
