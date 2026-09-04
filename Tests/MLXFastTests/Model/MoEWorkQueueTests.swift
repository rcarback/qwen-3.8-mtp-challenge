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
}
