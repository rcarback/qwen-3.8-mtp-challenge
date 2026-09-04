import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXFastModel

final class FusedRoutedMoETests: XCTestCase {
    /// Reference: what the MLX path computes for one expert's rows.
    private func reference(
        xSorted: MLXArray, gateUp: MLXArray, down: MLXArray,
        rowOffsets: [Int32], numExperts: Int
    ) -> MLXArray {
        var out: [MLXArray] = []
        let hidden = down.dim(2)
        for e in 0..<numExperts {
            let lo = Int(rowOffsets[e]), hi = Int(rowOffsets[e + 1])
            if lo == hi { continue }
            let rows = xSorted[lo..<hi]  // [n, inDim]
            let gu = MLX.matmul(rows, gateUp[e].transposed())  // [n, 2*hidden]
            let g = gu[.ellipsis, 0..<hidden]
            let u = gu[.ellipsis, hidden..<(2 * hidden)]
            let inter = MLXNN.silu(g) * u
            out.append(MLX.matmul(inter, down[e].transposed()))  // [n, inDim]
        }
        return MLX.concatenated(out, axis: 0)
    }

    func testDenseFusedMatchesReference() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a real GPU")

        // Small but ASYMMETRIC, and with a skewed row distribution that includes
        // an empty expert and one expert wider than blockRows. Square or uniform
        // shapes hide transposed and block-boundary errors. hidden is 32, not
        // the brief's original 24, so it satisfies the Task 3 index-arithmetic
        // precondition (input and hidden dims must be multiples of 32) that
        // this task enforces early; see FusedRoutedMoETests file header.
        let experts = 5, inDim = 64, hidden = 32
        let counts: [Int32] = [3, 0, 20, 1, 7]  // 31 rows, one > blockRows(16)
        var offsets: [Int32] = [0]
        for c in counts { offsets.append(offsets.last! + c) }
        let rows = Int(offsets.last!)

        MLXRandom.seed(7)
        let x = MLXRandom.normal([rows, inDim]).asType(.float16)
        let gateUp = MLXRandom.normal([experts, 2 * hidden, inDim]).asType(.float16) * 0.05
        let down = MLXRandom.normal([experts, inDim, hidden]).asType(.float16) * 0.05

        let rowOffsets = MLXArray(offsets)
        let blockOffsets = MoEWorkQueue.blockOffsets(
            rowOffsets: rowOffsets, blockRows: FusedRoutedMoE.blockRows)

        let got = FusedRoutedMoE.denseForward(
            xSorted: x, gateUp: gateUp, down: down,
            rowOffsets: rowOffsets, blockOffsets: blockOffsets)
        let want = reference(
            xSorted: x, gateUp: gateUp, down: down,
            rowOffsets: offsets, numExperts: experts)

        got.eval()
        want.eval()
        XCTAssertEqual(got.shape, [rows, inDim])
        let maxAbs = MLX.abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(maxAbs, 2e-2, "fused dense output diverged from the reference")
    }
}
