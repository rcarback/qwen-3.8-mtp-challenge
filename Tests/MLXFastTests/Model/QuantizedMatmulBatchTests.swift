import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXFastCore

/// Does the affine 4-bit matmul batch?
///
/// Prefill sits at a flat ~13 ms per token from 106 tokens to 20k, and the
/// gated-delta recurrence is only 1.2% of that. The remaining suspect is the
/// quantized matmul: a vector (qmv) path processes one row at a time, so cost
/// per row is CONSTANT in batch size, which is precisely the observed curve. A
/// batched GEMM must show cost per row falling steeply as M grows.
///
/// Shapes are the model's own: hidden 5120, MLP 17408, affine 4-bit group-64.
@Suite(.serialized)
struct QuantizedMatmulBatchTests {
    @Test("affine 4-bit matmul cost per row versus batch")
    func batchScaling() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let K = 5120, N = 17408
        let w = MLXRandom.randInt(0 ..< Int32.max, [N, K * 4 / 32])
            .asType(.uint32)
        let scales = MLXRandom.normal([N, K / 64]).asType(.bfloat16)
        let biases = MLXRandom.normal([N, K / 64]).asType(.bfloat16)
        eval(w, scales, biases)

        print("\nAffine 4-bit matmul  [M, \(K)] x [\(K), \(N)]")
        print("      M   seconds   ms/row    GFLOP/s")
        for M in [1, 8, 64, 256, 1024, 4096] {
            let x = MLXRandom.normal([1, M, K]).asType(.bfloat16)
            eval(x)
            func once() -> MLXArray {
                quantizedMM(
                    x, w, scales: scales, biases: biases, transpose: true,
                    groupSize: 64, bits: 4, mode: .affine)
            }
            for _ in 0 ..< 3 { eval(once()) }
            var best = Double.greatestFiniteMagnitude
            for _ in 0 ..< 5 {
                let start = Date()
                eval(once())
                best = Swift.min(best, Date().timeIntervalSince(start))
            }
            let flops = 2.0 * Double(M) * Double(K) * Double(N)
            print(String(
                format: "  %5d  %8.4f  %7.4f  %9.1f",
                M, best, 1000 * best / Double(M), flops / best / 1e9))
        }
        print("")
    }
}
