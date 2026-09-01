import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXFastModel

/// Task E (ANE IOSurface / procedure-bank plan): the gated A/B timing that
/// decides whether the ANE∥GPU overlap built in Task D1 actually beats the
/// all-GPU serve path at real Qwen prefill dimensions.
///
/// This is the go/no-go gate for the whole ANE-offload effort. If the
/// concurrent split does NOT beat the pure all-GPU reference here, then D2
/// (Metal zero-copy) and wiring `ANEFusedSplitMLP` into the Qwen35 MLP
/// prefill path are both moot.
///
/// Discipline (mirrors `ColumnSplitSpeedTests`):
///   * ONE fraction per process, via `MLXFAST_ANE_FRACTION` -- position
///     inside a process moves GPU timings by more than the effect measured.
///   * Warmup, then best-of-N (the minimum is the least-contended sample).
///   * The honest baseline is the pure all-GPU `referenceForward` (three
///     4-bit `quantizedMM`s), NOT `ANEFusedSplitMLP` at fraction 0 -- the
///     candidate must beat the real serve path, not a hobbled version of
///     itself.
///   * Both legs are timed in the SAME process so the ratio cancels machine
///     state; an interleaved A/B/A/B pass checks for order effects.
///
/// Reported number: `speedup = baseline_time / candidate_time`. >1 means the
/// overlap paid off; the curve over fractions locates the balance point.
///
/// Driven by `tools/ane-fraction-sweep.sh`. Gated OFF by default: requires
/// `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` AND `MLXFAST_ANE_FRACTION=<f>`.
@Suite(.serialized)
struct ANEFusedSplitSpeedTests {
    private static let hidden = 5_120
    private static let inter = 17_408

    private static func quantizedWeight(out: Int, inn: Int, seed: UInt64) -> (MLXArray, MLXArray, MLXArray) {
        MLXRandom.seed(seed)
        let scale = Float(1.0 / Double(inn).squareRoot())
        let w = (MLXRandom.normal([out, inn]) * scale).asType(.bfloat16)
        let (wq, scales, biases0) = quantized(w, groupSize: 64, bits: 4)
        let biases = biases0 ?? scales
        eval(wq, scales, biases)
        return (wq, scales, biases)
    }

    private static func referenceForward(
        _ x: MLXArray,
        gateWq: MLXArray, gateScales: MLXArray, gateBiases: MLXArray,
        upWq: MLXArray, upScales: MLXArray, upBiases: MLXArray,
        downWq: MLXArray, downScales: MLXArray, downBiases: MLXArray
    ) -> MLXArray {
        let gate = quantizedMM(x, gateWq, scales: gateScales, biases: gateBiases,
                                transpose: true, groupSize: 64, bits: 4)
        let up = quantizedMM(x, upWq, scales: upScales, biases: upBiases,
                              transpose: true, groupSize: 64, bits: 4)
        let h = (silu(gate) * up).asType(.bfloat16)
        let y = quantizedMM(h, downWq, scales: downScales, biases: downBiases,
                             transpose: true, groupSize: 64, bits: 4)
        return y
    }

    /// Best-of-N wall time forcing full materialization each iteration.
    private static func best(of n: Int, _ body: () throws -> MLXArray) rethrows -> Double {
        var best = Double.infinity
        for _ in 0 ..< n {
            let start = Date()
            let y = try body()
            eval(y)
            best = Swift.min(best, Date().timeIntervalSince(start))
        }
        return best
    }

    @Test("ANEFusedSplitMLP concurrent vs all-GPU reference: one fraction, alone in its process")
    func aneFractionSpeedPoint() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
              let fRaw = env["MLXFAST_ANE_FRACTION"],
              let fraction = Double(fRaw)
        else { return }
        try #require(ANERuntime.available())

        let hidden = Self.hidden
        let inter = Self.inter
        let s = Int(env["MLXFAST_SEQ_LEN"] ?? "512") ?? 512
        let iters = Int(env["MLXFAST_TIMING_ITERS"] ?? "9") ?? 9

        let (gateWq, gateScales, gateBiases) = Self.quantizedWeight(out: inter, inn: hidden, seed: 1)
        let (upWq, upScales, upBiases) = Self.quantizedWeight(out: inter, inn: hidden, seed: 2)
        let (downWq, downScales, downBiases) = Self.quantizedWeight(out: hidden, inn: inter, seed: 3)
        let x = MLXRandom.normal([s, hidden]).asType(.bfloat16)
        eval(x)

        let split = try ANEFusedSplitMLP(
            gateW: gateWq, gateScales: gateScales, gateBiases: gateBiases,
            upW: upWq, upScales: upScales, upBiases: upBiases,
            downW: downWq, downScales: downScales, downBiases: downBiases,
            hidden: hidden, inter: inter, sequenceLength: s, aneFraction: fraction)

        let baseline: () throws -> MLXArray = {
            Self.referenceForward(
                x,
                gateWq: gateWq, gateScales: gateScales, gateBiases: gateBiases,
                upWq: upWq, upScales: upScales, upBiases: upBiases,
                downWq: downWq, downScales: downScales, downBiases: downBiases)
        }
        let candidate: () throws -> MLXArray = { try split(x) }

        // Isolation mode: measure exactly ONE leg alone in this process, so
        // neither leg can contaminate the other's scheduling. `both`
        // (default) keeps the grouped + interleaved comparison.
        let leg = env["MLXFAST_TIMING_LEG"] ?? "both"
        if leg == "baseline" || leg == "candidate" {
            let body = (leg == "baseline") ? baseline : candidate
            eval(try body())               // warmup only this leg
            let t = try Self.best(of: iters, body)
            print("ANE-SPEED-ISO fraction=\(fraction) S=\(s) iters=\(iters) leg=\(leg) time=\(t * 1e3)ms")
            #expect(t.isFinite)
            return
        }

        // Warmup both (compile/first-dispatch costs out of the measurement).
        eval(try baseline())
        eval(try candidate())

        // Grouped best-of-N.
        let baseGrouped = try Self.best(of: iters, baseline)
        let candGrouped = try Self.best(of: iters, candidate)

        // Interleaved A/B/A/B to catch order effects.
        var baseInter = Double.infinity
        var candInter = Double.infinity
        for _ in 0 ..< iters {
            let s0 = Date(); eval(try baseline()); baseInter = Swift.min(baseInter, Date().timeIntervalSince(s0))
            let s1 = Date(); eval(try candidate()); candInter = Swift.min(candInter, Date().timeIntervalSince(s1))
        }

        let speedupGrouped = baseGrouped / candGrouped
        let speedupInter = baseInter / candInter
        print("ANE-SPEED fraction=\(fraction) S=\(s) iters=\(iters)")
        print("ANE-SPEED grouped  baseline=\(baseGrouped * 1e3)ms candidate=\(candGrouped * 1e3)ms speedup=\(speedupGrouped)")
        print("ANE-SPEED interleaved baseline=\(baseInter * 1e3)ms candidate=\(candInter * 1e3)ms speedup=\(speedupInter)")

        // This test never fails on a timing threshold -- it reports. The
        // #expect keeps it a real (non-skipped) test when gated on.
        #expect(candGrouped.isFinite && baseGrouped.isFinite)
    }
}
