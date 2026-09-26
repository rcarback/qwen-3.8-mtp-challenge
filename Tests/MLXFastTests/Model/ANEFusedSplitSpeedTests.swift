import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXFastModel
@testable import MLXLLM

/// Opt-in paired timing of the reusable ANE/GPU bridge against the production
/// GPU layer at real Qwen prefill dimensions. This isolates one MLP layer;
/// full-model measurements must establish any request speedup.
///
/// Discipline (mirrors `ColumnSplitSpeedTests`):
///   * ONE fraction per process, via `MLXFAST_ANE_FRACTION` -- position
///     inside a process moves GPU timings by more than the effect measured.
///   * Warmup, then best-of-N (the minimum is the least-contended sample).
///   * The baseline calls `Qwen35FusedMLP` with offload disabled, preserving
///     production dispatch at the measured row count and environment.
///   * Both legs run in the same process. Individual AB/BA pairs expose
///     drift and order effects that the best-of-N summaries can hide.
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

    private static func productionLayer(
        gateWq: MLXArray, gateScales: MLXArray, gateBiases: MLXArray,
        upWq: MLXArray, upScales: MLXArray, upBiases: MLXArray,
        downWq: MLXArray, downScales: MLXArray, downBiases: MLXArray
    ) -> Qwen35FusedMLP {
        let layer = Qwen35FusedMLP(dimensions: hidden, hiddenDimensions: inter)
        layer.update(modules: ModuleChildren.unflattened([
            ("gate_proj", QuantizedLinear(weight: gateWq, scales: gateScales,
                biases: gateBiases, groupSize: 64, bits: 4)),
            ("up_proj", QuantizedLinear(weight: upWq, scales: upScales,
                biases: upBiases, groupSize: 64, bits: 4)),
            ("down_proj", QuantizedLinear(weight: downWq, scales: downScales,
                biases: downBiases, groupSize: 64, bits: 4)),
        ]))
        return layer
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

    @Test("ANEFusedSplitMLP concurrent vs production GPU layer",
          .enabled(if: ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
              && ProcessInfo.processInfo.environment["MLXFAST_ANE_FRACTION"] != nil))
    func aneFractionSpeedPoint() throws {
        let env = ProcessInfo.processInfo.environment
        let fRaw = try #require(env["MLXFAST_ANE_FRACTION"])
        let fraction = try #require(Double(fRaw), "MLXFAST_ANE_FRACTION must be numeric")
        try #require(!ANESplitConfig.enabled, "Set MLX_ANE_DIRECT=0 so the production control cannot offload")
        try #require(!ANESplitConfig.verify, "Timing must run without ANE verification")
        try #require(fraction > 0 && fraction < 1)
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

        let production = Self.productionLayer(
                gateWq: gateWq, gateScales: gateScales, gateBiases: gateBiases,
                upWq: upWq, upScales: upScales, upBiases: upBiases,
                downWq: downWq, downScales: downScales, downBiases: downBiases)
        let baseline: () throws -> MLXArray = { production(x) }
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

        // Preserve each paired sample; minima alone hide control drift.
        for pair in 0 ..< max(3, iters) {
            let first = pair.isMultiple(of: 2) ? baseline : candidate
            let second = pair.isMultiple(of: 2) ? candidate : baseline
            let a = try Self.best(of: 1, first)
            let b = try Self.best(of: 1, second)
            let control = pair.isMultiple(of: 2) ? a : b
            let treatment = pair.isMultiple(of: 2) ? b : a
            print("ANE-PAIR index=\(pair) order=\(pair.isMultiple(of: 2) ? "AB" : "BA") S=\(s) baseline_ms=\(control * 1e3) candidate_ms=\(treatment * 1e3) speedup=\(control / treatment) fraction=\(fraction) weight_form=\(ANESplitConfig.weightForm.rawValue) gpu_suffix=\(ANESplitConfig.fp16GpuSuffix ? "fp16" : "current")")
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
