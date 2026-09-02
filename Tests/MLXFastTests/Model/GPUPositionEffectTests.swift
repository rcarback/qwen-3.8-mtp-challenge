import Foundation
import MLX
import MLXRandom
import Testing

/// Isolates the "position in process" effect that dominates every GPU timing
/// on this machine.
///
/// The observation (`PrefillMatmulCostTests.singleGemmPoint`): a square bf16
/// GEMM reads about 14.7 TFLOPS when its block runs first in a fresh process
/// and about 6.4 when any other measurement block ran before it. The effect
/// survives `Memory.clearCache()`, does not survive a process boundary, and a
/// fresh process reads full speed immediately after a heavy run in another
/// process -- which rules out GPU clock and thermal state.
///
/// That leaves a per-process resource. This suite runs ONE prelude and ONE
/// measurement per process so the prelude is the only variable, and it can
/// measure either a compute-bound or a bandwidth-bound kernel. The cross
/// product answers the question the single-kernel observation cannot:
///
///   * If a bandwidth prelude degrades a compute measurement, and a compute
///     prelude degrades a bandwidth measurement, the degraded resource is the
///     memory system, not the shader issue path.
///   * If only like degrades like, two independent effects exist.
///   * If `alloc` alone degrades without any dispatch, the trigger is
///     allocation and page mapping, not execution.
///
/// Driven one point per process by `tools/position-effect-sweep.sh`.
@Suite(.serialized)
struct GPUPositionEffectTests {
    /// Best of three, warmed once. `eval` forces the lazy graph, so the timer
    /// brackets real GPU work rather than graph construction.
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

    /// Elements in the bandwidth array: 2^26 bf16 = 128 MiB, so one pass
    /// reads 128 MiB and writes 128 MiB. Far past any cache on this part, so
    /// the kernel is bound by DRAM rather than by issue rate.
    private static let bwElements = 1 << 26

    /// Square GEMM side for the compute-bound probe. 4096^3 is the shape the
    /// 14.7 TFLOPS reference was taken at.
    private static let gemmN = 4096

    /// Runs the requested prelude. Each arm is designed to touch exactly one
    /// suspected resource so the measurement that follows attributes cleanly.
    private static func runPrelude(_ name: String) throws {
        switch name {
        case "none":
            // Control. Nothing precedes the measurement.
            break

        case "cpu":
            // Burns wall clock on the CPU with no MLX op at all. If this
            // degrades the measurement, the trigger is not GPU state.
            var acc = 0.0
            let deadline = Date().addingTimeInterval(0.20)
            var i = 0
            while Date() < deadline {
                acc += Double(i).squareRoot()
                i += 1
            }
            #expect(acc.isFinite)

        case "alloc":
            // Allocates and materialises a large buffer, then drops it. One
            // trivial dispatch, but a large allocation and page mapping. This
            // separates "allocated memory" from "ran a heavy kernel".
            var a: MLXArray? = MLXRandom.normal([bwElements]).asType(.bfloat16)
            eval(a!)
            a = nil

        case "tinygpu":
            // A single one-element dispatch. Establishes whether ANY GPU
            // submission is enough to poison what follows.
            let a = MLXArray([1.0] as [Float])
            eval([a + 1])

        case "biggpu":
            // The known poison: a full square bf16 GEMM.
            let a = MLXRandom.normal([gemmN, gemmN]).asType(.bfloat16)
            let b = MLXRandom.normal([gemmN, gemmN]).asType(.bfloat16)
            eval(a, b)
            eval([matmul(a, b)])

        case "bandwidth":
            // Heavy DRAM traffic, negligible arithmetic.
            let a = MLXRandom.normal([bwElements]).asType(.bfloat16)
            eval(a)
            eval([a * MLXArray(2.0).asType(.bfloat16)])

        default:
            Issue.record("unknown MLXFAST_POS_PRELUDE '\(name)'")
            throw CancellationError()
        }
    }

    @Test("one prelude and one measurement, alone in the process")
    func positionEffectPoint() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let prelude = env["MLXFAST_POS_PRELUDE"] ?? "none"
        let measure = env["MLXFAST_POS_MEASURE"] ?? "gemm"
        // Refuse an unknown measurement rather than silently reporting one
        // kernel under another kernel's label.
        guard measure == "gemm" || measure == "bandwidth" else {
            Issue.record("MLXFAST_POS_MEASURE must be gemm or bandwidth")
            return
        }
        // Optional: drop the MLX allocator cache between prelude and
        // measurement. The original observation says this does not help; the
        // flag makes that reproducible rather than remembered.
        let clear = env["MLXFAST_POS_CLEARCACHE"] == "1"

        try Self.runPrelude(prelude)
        if clear {
            Memory.clearCache()
            eval([MLXArray([1.0] as [Float])])
        }

        let dt: Double
        let metric: Double
        let unit: String
        if measure == "gemm" {
            let n = Self.gemmN
            let a = MLXRandom.normal([n, n]).asType(.bfloat16)
            let b = MLXRandom.normal([n, n]).asType(.bfloat16)
            eval(a, b)
            dt = Self.timeIt { [matmul(a, b)] }
            metric = 2.0 * Double(n) * Double(n) * Double(n) / dt / 1e12
            unit = "TFLOPS"
        } else {
            let a = MLXRandom.normal([Self.bwElements]).asType(.bfloat16)
            let two = MLXArray(2.0).asType(.bfloat16)
            eval(a, two)
            dt = Self.timeIt { [a * two] }
            // One read plus one write of a bf16 array.
            metric = 2.0 * Double(Self.bwElements) * 2.0 / dt / 1e9
            unit = "GB/s"
        }

        print(String(
            format: "POSPOINT\t%@\t%@\t%@\t%.4f\t%.3f\t%@",
            prelude, measure, clear ? "clear" : "noclear",
            1000 * dt, metric, unit))
    }
}
