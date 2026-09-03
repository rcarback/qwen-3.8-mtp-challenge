import XCTest

@testable import MLXLLM

final class ANESplitMLPCacheTests: XCTestCase {
    func testBucketRoundsUpToTheNextPowerOfTwoTile() {
        XCTAssertEqual(ANESplitMLPCache.bucketedSequenceLength(1), 128)
        XCTAssertEqual(ANESplitMLPCache.bucketedSequenceLength(128), 128)
        XCTAssertEqual(ANESplitMLPCache.bucketedSequenceLength(129), 256)
        XCTAssertEqual(ANESplitMLPCache.bucketedSequenceLength(512), 512)
        XCTAssertEqual(ANESplitMLPCache.bucketedSequenceLength(732), 1024)
        XCTAssertEqual(ANESplitMLPCache.bucketedSequenceLength(1032), 2048)
    }

    func testDistinctLengthsInOneBucketShareAKey() {
        XCTAssertEqual(
            ANESplitMLPCache.bucketedSequenceLength(732),
            ANESplitMLPCache.bucketedSequenceLength(1024))
    }
}
