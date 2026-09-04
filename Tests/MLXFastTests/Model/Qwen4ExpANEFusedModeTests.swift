import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Pins the GATE, not the numerics: with no ANE knob set, both fused lanes
/// must stay inert and no ANE program may even be attempted. Needs no ANE and
/// runs on every machine in the ordinary `swift test` pass.
final class Qwen4ExpANEFusedModeTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Qwen4ExpANEFused.resetForTesting()
    }

    func testLanesAreInertWhenModeUnset() throws {
        XCTAssertFalse(Qwen4ExpANEFused.splitEnabled)
        XCTAssertFalse(Qwen4ExpANEFused.sharedEnabled)
        XCTAssertEqual(Qwen4ExpANEFused.mode, .microbatch)
        // Documents that MLX_ANE_DIRECT is what actually holds the lanes shut;
        // fails loudly if this suite is ever run with the knob set.
        XCTAssertFalse(Qwen4ExpANELane.enabled)

        let (model, dir) = try Qwen4ExpModelTests().makeModel()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ids = MLXArray((0 ..< 100).map { Int32($0 % 37 + 1) }).reshaped(1, 100)
        let cache = model.newCache(parameters: nil)
        _ = model(ids, cache: cache)

        XCTAssertEqual(Qwen4ExpANEFused.reservedPrograms(), 0)
        XCTAssertEqual(Qwen4ExpANEFused.reservedBytes(), 0)
    }

    /// Documents precedence: when `MLX_QWEN4EXP_FORCE_MICROBATCH` is set, both
    /// fused lanes are off for the whole process regardless of
    /// `MLX_QWEN4EXP_ANE_MODE`. The statics are read once at process start, so
    /// the true branch is checked by the measurement plan, not by mutating the
    /// environment here.
    func testForcedMicroBatchDisarmsFusedLanes() {
        XCTAssertFalse(Qwen4ExpANEFused.forcedMicroBatchProbe)
    }

    /// Pins precedence rule 1: the legacy structural probe hook still takes
    /// the micro-batched path with GPU-only projections, unaffected by the
    /// new mode gate.
    func testForcedMicroBatchStillTakesLegacyArm() throws {
        let (model, dir) = try Qwen4ExpModelTests().makeModel()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ids = MLXArray((0 ..< 32).map { Int32($0 % 37 + 1) }).reshaped(1, 32)

        Qwen4ExpTextModel.forcedMicroBatch = 8
        defer { Qwen4ExpTextModel.forcedMicroBatch = nil }
        let pipedCache = model.newCache(parameters: nil)
        let piped = model(ids, cache: pipedCache)

        Qwen4ExpTextModel.forcedMicroBatch = nil
        let plainCache = model.newCache(parameters: nil)
        let plain = model(ids, cache: plainCache)

        XCTAssertTrue(allClose(piped, plain, rtol: 2e-2, atol: 2e-3).item())
    }
}
