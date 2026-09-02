import Foundation
import MLX
@testable import MLXLLM
import MLXRandom
import Testing

/// Does MLX actually OVERLAP a CPU stream with a GPU stream, or does it
/// serialise them?
///
/// `qwen35ColumnSplitQuantizedMM` was built and proven bit-exact per half, but
/// never speed-tested. Correctness says the halves land on the devices they
/// claim; it says nothing about whether the two halves run at the same time.
/// If MLX serialises, wall time rises monotonically from fraction 0 and the
/// whole heterogeneous-pipeline idea dies here for the price of one sweep.
///
/// The curve is self-diagnosing:
///   * monotonically worse from 0  -> serialised, idea is dead
///   * dip then rise               -> real overlap, the minimum is the balance
///
/// Predicted optimum is `cpu_rate / (cpu_rate + gpu_rate)`. With the CPU
/// quantized path on NEON (~0.107 TFLOPS) against Metal at 13.45 that is
/// ~0.8%, indistinguishable from noise -- which is itself the argument for
/// routing the CPU half through SME (2.01 TFLOPS, optimum ~13%).
///
/// One fraction per process, because position inside a process moves GPU
/// timings by more than this effect. Driven by
/// `tools/column-split-sweep.sh`.
@Suite(.serialized)
struct ColumnSplitSpeedTests {
    private static func timeIt(_ body: () throws -> [MLXArray]) rethrows -> Double {
        eval(try body())
        var best = Double.infinity
        for _ in 0 ..< 3 {
            let start = Date()
            eval(try body())
            best = Swift.min(best, Date().timeIntervalSince(start))
        }
        return best
    }

    @Test("one CPU column fraction, alone in its process")
    func columnSplitSpeedPoint() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
              let fRaw = env["MLXFAST_SPLIT_FRACTION"],
              let fraction = Double(fRaw)
        else { return }
        let M = Int(env["MLXFAST_GEMM_M"] ?? "1024") ?? 1024
        let N = Int(env["MLXFAST_GEMM_N"] ?? "17408") ?? 17408
        let K = Int(env["MLXFAST_GEMM_K"] ?? "5120") ?? 5120

        let x = MLXRandom.normal([1, M, K]).asType(.bfloat16)
        let w = MLXRandom.normal([N, K]).asType(.bfloat16)
        let (wq, scales, biasesOpt) = quantized(w, groupSize: 64, bits: 4)
        guard let biases = biasesOpt else {
            Issue.record("affine quantization returned no biases")
            return
        }
        eval(x, wq, scales, biases)

        // Fraction 0 goes through the helper's own cpuN == 0 arm, which is a
        // plain GPU quantizedMM. That keeps the control on the same code path
        // as the split rather than in a separate call site.
        let dt = try Self.timeIt {
            let split = try qwen35ColumnSplitQuantizedMM(
                x, wq, scales: scales, biases: biases,
                groupSize: 64, bits: 4, mode: .affine,
                cpuColumnFraction: fraction)
            return [split.output]
        }
        let probe = try qwen35ColumnSplitQuantizedMM(
            x, wq, scales: scales, biases: biases,
            groupSize: 64, bits: 4, mode: .affine,
            cpuColumnFraction: fraction)
        let flops = 2.0 * Double(M) * Double(N) * Double(K)
        print(String(
            format: "SPLITPOINT\t%.4f\t%d\t%d\t%d\t%d\t%d\t%.4f\t%.3f",
            fraction, M, N, K, probe.cpuColumns, probe.gpuColumns,
            1000 * dt, flops / dt / 1e12))
    }
}
