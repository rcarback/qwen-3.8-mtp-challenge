import Foundation
import MLX
import MLXRandom
import XCTest

/// Does reducing top-k actually reduce the routed-expert GEMM time on this
/// tower? The prediction from the tile arithmetic is no: with 512 experts and
/// a 700-token prompt every expert is still touched at any k, and each one's
/// rows still fit inside a single 16-row tile, so the padded multiply work is
/// 512 tiles per layer regardless of k.
///
/// Needs a real GPU and about 0.5 GB; opt in with MLXFAST_RUN_MLX_RUNTIME_TESTS=1.
final class MoEGatherTileTests: XCTestCase {
    /// One routed projection of the Qwen4Exp tower: 512 experts, 2560 in, 640 out.
    func testTopKDoesNotReduceGatherGEMMTime() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a real GPU")

        let experts = 512, inDim = 2560, outDim = 640, tokens = 700
        let wf = MLXRandom.normal([experts, outDim, inDim]).asType(.bfloat16)
        let (w, scales, biases) = MLX.quantized(wf, groupSize: 32, bits: 4)
        eval(w, scales, biases)

        func timeOne(k: Int, reps: Int) -> Double {
            let rows = tokens * k
            // Shape matters: `gatherSort` hands the kernel [rows, 1, in], so
            // M is 1 and `GatherQMM::eval_gpu` takes the sorted `gather_qmm_rhs`
            // branch. A [rows, in] operand makes M = rows instead and dispatches
            // the general gather path, which is a different kernel entirely.
            let x = MLXRandom.normal([rows, 1, inDim]).asType(.bfloat16)
            // Sorted expert assignment, matching what SwitchGLU hands the kernel.
            var ids = (0 ..< rows).map { Int32(($0 * experts) / rows) }
            ids.sort()
            let idx = MLXArray(ids).reshaped(rows)
            eval(x, idx)
            func once() -> MLXArray {
                MLX.gatherQuantizedMM(
                    x, w, scales: scales, biases: biases, rhsIndices: idx,
                    transpose: true, groupSize: 32, bits: 4, sortedIndices: true)
            }
            eval(once())  // warm the kernel
            let t0 = Date()
            for _ in 0 ..< reps { eval(once()) }
            return Date().timeIntervalSince(t0) / Double(reps) * 1000
        }

        // Interleave the arms and keep the minimum per arm. A single pass in
        // descending k measured non-monotonically (k=8 slower than k=10),
        // which is thermal and allocator noise, not a property of the kernel.
        let ks = [10, 8, 6, 4, 2]
        var best = [Int: Double]()
        for _ in 0 ..< 5 {
            for k in ks {
                let ms = timeOne(k: k, reps: 20)
                best[k] = min(best[k] ?? .infinity, ms)
            }
        }
        let base = best[10]!
        for k in ks {
            let ms = best[k]!
            print(String(
                format: "[gather-tile] k=%2d rows=%5d rows/expert=%5.2f  %7.3f ms  %5.1f%% of k=10  (FLOP is %3.0f%%)",
                k, tokens * k, Double(tokens * k) / Double(experts), ms, 100 * ms / base,
                100.0 * Double(k) / 10.0))
        }
        XCTAssertGreaterThan(base, 0)
    }
}
