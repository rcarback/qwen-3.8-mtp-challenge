import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen4ExpANESplitProjectionTests: XCTestCase {
    /// Synthetic asymmetric projection, non-bucket-aligned S.
    /// out = 384, in = 192, bucket 128, S = 100, F = 128.
    func testSplitMatchesPureGPUProjection() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1")
        try XCTSkipUnless(ANERuntime.available())

        let w = (MLXRandom.normal([384, 192]) / Float(192).squareRoot()).asType(.bfloat16)
        let x = MLXRandom.normal([100, 192]).asType(.bfloat16)
        eval(w, x)

        XCTAssertEqual(Qwen4ExpANEFused.bucket(100), 128)

        let split = try Qwen4ExpANESplitProjection(weight: w, logicalOut: 384, fraction: 0.3125, sequenceLength: 128)
        XCTAssertEqual(split.f, 128)  // round(0.3125 x 384 / 64) x 64 = 2 x 64

        let got = try split(x)
        XCTAssertEqual(got.shape, [100, 384])

        let want = matmul(x, w.transposed(1, 0))
        let aneBand = got[0..., 0 ..< 128].asType(.float32)
        let wantAneBand = want[0..., 0 ..< 128].asType(.float32)
        let diff = MLX.abs(aneBand - wantAneBand)
        eval(diff)
        let maxAbs = diff.max().item(Float.self)
        let meanAbs = diff.mean().item(Float.self)
        XCTAssertLessThan(maxAbs, 0.05, "maxAbs=\(maxAbs)")
        XCTAssertLessThan(meanAbs, 0.005, "meanAbs=\(meanAbs)")

        // GPU rows: compare against a SHAPE-MATCHED reference, not the column
        // band of the full N=384 reference (MLX tile/accumulation order is not
        // guaranteed shape-invariant).
        let wantGPUBand = matmul(x, w[128 ..< 384, 0...].transposed(1, 0))
        XCTAssertTrue(allClose(got[0..., 128...], wantGPUBand).item())
    }

    /// Pins the concat order: ANE rows first, then GPU rows.
    func testSplitRowOrderIsPrefixThenSuffix() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1")
        try XCTSkipUnless(ANERuntime.available())

        var w = (MLXRandom.normal([384, 192]) / Float(192).squareRoot()).asType(.bfloat16)
        w[0 ..< 128, 0...] = MLXArray.zeros([128, 192], dtype: .bfloat16)
        eval(w)
        let x = MLXRandom.normal([100, 192]).asType(.bfloat16)
        eval(x)

        let split = try Qwen4ExpANESplitProjection(weight: w, logicalOut: 384, fraction: 0.3125, sequenceLength: 128)
        let got = try split(x)

        let aneBand = got[0..., 0 ..< 128].asType(.float32)
        XCTAssertEqual(MLX.abs(aneBand).max().item(Float.self), 0, accuracy: 1e-6)

        let wantSuffix = matmul(x, w[128 ..< 384, 0...].transposed(1, 0))
        XCTAssertTrue(allClose(got[0..., 128...], wantSuffix).item())
    }

    /// Pins the empty-GPU-leg guard: a fraction that would leave zero GPU rows
    /// must throw rather than silently return a short row.
    func testDegenerateFractionThrows() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1")
        try XCTSkipUnless(ANERuntime.available())

        let w = (MLXRandom.normal([384, 192]) / Float(192).squareRoot()).asType(.bfloat16)
        eval(w)
        XCTAssertThrowsError(
            try Qwen4ExpANESplitProjection(weight: w, logicalOut: 384, fraction: 1.0, sequenceLength: 128)
        ) { error in
            guard case Qwen4ExpANESplitError.degenerateSplit = error else {
                XCTFail("expected degenerateSplit, got \(error)")
                return
            }
        }
    }

    /// Pins the 0.3 arithmetic: `F` is a share of the LOGICAL output row
    /// count, not the physical weight's own row count.
    func testLogicalOutSetsTheSplitPoint() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1")
        try XCTSkipUnless(ANERuntime.available())

        let w = (MLXRandom.normal([384, 192]) / Float(192).squareRoot()).asType(.bfloat16)
        eval(w)
        let split = try Qwen4ExpANESplitProjection(weight: w, logicalOut: 512, fraction: 0.3125, sequenceLength: 128)
        XCTAssertEqual(split.f, 192)  // round(0.3125 x 512 / 64) x 64 = 3 x 64
    }
}
