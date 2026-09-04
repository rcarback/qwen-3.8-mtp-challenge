import XCTest

@testable import MLXLLM

/// Drives `Qwen4ExpANEFused.reserveProgram`/`releaseProgram` directly against
/// the shared process-wide counters. Kept in its own `XCTestCase` so it does
/// not race the counter assertions in `Qwen4ExpANEFusedModeTests` (no
/// alphabetical-ordering dependency between the two classes). Needs no ANE.
final class Qwen4ExpANEProgramBudgetTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Qwen4ExpANEFused.resetForTesting()
    }

    func testReserveAcceptsUntilTheByteBudgetIsSpent() {
        let budget = Qwen4ExpANEFused.programBudgetBytes
        let chunk = budget / 2
        XCTAssertTrue(Qwen4ExpANEFused.reserveProgram(bytes: chunk, label: "t1"))
        XCTAssertTrue(Qwen4ExpANEFused.reserveProgram(bytes: chunk, label: "t2"))
        XCTAssertEqual(Qwen4ExpANEFused.reservedBytes(), chunk * 2)
        // A third reservation that would exceed the byte budget is refused.
        XCTAssertFalse(Qwen4ExpANEFused.reserveProgram(bytes: budget, label: "t3"))
        XCTAssertEqual(Qwen4ExpANEFused.reservedBytes(), chunk * 2)
        XCTAssertEqual(Qwen4ExpANEFused.reservedPrograms(), 2)
    }

    func testReserveAcceptsUntilTheCountLimitIsSpent() {
        let limit = Qwen4ExpANEFused.programCountLimit
        for i in 0 ..< limit {
            XCTAssertTrue(Qwen4ExpANEFused.reserveProgram(bytes: 1, label: "p\(i)"))
        }
        XCTAssertEqual(Qwen4ExpANEFused.reservedPrograms(), limit)
        // One more reservation, however small, is refused once the count
        // limit is spent.
        XCTAssertFalse(Qwen4ExpANEFused.reserveProgram(bytes: 1, label: "overflow"))
        XCTAssertEqual(Qwen4ExpANEFused.reservedPrograms(), limit)
    }

    /// Pins spec 5.7 outcome 3: without `releaseProgram`, a failed build
    /// silently retires budget forever.
    func testReleaseReturnsAFailedBuildsReservation() {
        let bytes = 1024
        XCTAssertTrue(Qwen4ExpANEFused.reserveProgram(bytes: bytes, label: "will-fail"))
        XCTAssertEqual(Qwen4ExpANEFused.reservedBytes(), bytes)
        XCTAssertEqual(Qwen4ExpANEFused.reservedPrograms(), 1)

        Qwen4ExpANEFused.releaseProgram(bytes: bytes)
        XCTAssertEqual(Qwen4ExpANEFused.reservedBytes(), 0)
        XCTAssertEqual(Qwen4ExpANEFused.reservedPrograms(), 0)

        // The same size reservation succeeds again now that the budget was
        // returned.
        XCTAssertTrue(Qwen4ExpANEFused.reserveProgram(bytes: bytes, label: "retry"))
        XCTAssertEqual(Qwen4ExpANEFused.reservedBytes(), bytes)
        XCTAssertEqual(Qwen4ExpANEFused.reservedPrograms(), 1)
    }
}
