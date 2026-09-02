import Foundation
import MLX
import MLXRandom
import Testing

/// Separates "shape diversity costs something" from "more than one GEMM in a
/// process costs something".
///
/// `GemmChainCostTests` showed seven DIFFERENT projections chaining at 7.36
/// TFLOPS where one big GEMM alone reads 13.5. Two readings:
///
///   * the seven shapes are individually less efficient (D2 already showed
///     rate falls steeply below N=8192, and one projection is 96 rows), or
///   * running several GEMMs in one process degrades them regardless of shape
///
/// This chains N GEMMs of the SAME square shape, so shape efficiency is held
/// constant and only the count varies. `MLXFAST_CHAIN_COUNT=1` is the
/// control.
///
///   rate flat in count      -> chaining is free; the diverse result is a
///                              shape-mix effect and there is nothing to fix
///   rate falls with count   -> a real per-process or per-dependency cost
///
/// Square 5120x5120 keeps output shape equal to input shape so the chain is a
/// genuine dependency, not a reshape. One count per process.
@Suite(.serialized)
struct SameShapeChainTests {
    private static func timeIt(_ body: () -> [MLXArray]) -> Double {
        eval(body())
        var best = Double.infinity
        for _ in 0 ..< 3 {
            let start = Date()
            eval(body())
            best = Swift.min(best, Date().timeIntervalSince(start))
        }
        return best
    }

    @Test("N same-shape GEMMs chained, alone in the process")
    func sameShapeChainPoint() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
              let cRaw = env["MLXFAST_CHAIN_COUNT"], let count = Int(cRaw),
              count >= 1
        else { return }
        let M = 1024
        let D = 5120

        let x = MLXRandom.normal([1, M, D]).asType(.bfloat16)
        // Distinct weights per step so nothing can be cached across the chain,
        // but identical shape so efficiency is held constant.
        var wqs: [(MLXArray, MLXArray, MLXArray)] = []
        for _ in 0 ..< count {
            let w = MLXRandom.normal([D, D]).asType(.bfloat16)
            let (q, s, b) = quantized(w, groupSize: 64, bits: 4)
            wqs.append((q, s, b ?? s))
        }
        eval(x)
        for (q, s, b) in wqs { eval(q, s, b) }

        let dt = Self.timeIt {
            var h = x
            for (q, s, b) in wqs {
                h = quantizedMM(
                    h, q, scales: s, biases: b, transpose: true,
                    groupSize: 64, bits: 4)
            }
            return [h]
        }
        let flops = 2.0 * Double(M) * Double(D) * Double(D) * Double(count)
        print(String(
            format: "SAMECHAIN\t%d\t%.4f\t%.3f",
            count, 1000 * dt, flops / dt / 1e12))
    }
}
