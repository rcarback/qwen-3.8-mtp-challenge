import Foundation
import MLX
import MLXFastModel
import MLXNN
import MLXRandom
import Testing

/// Task 7 (ANE+GPU concurrent offload plan): head-to-head A-vs-B comparison
/// HARNESS for `CoarseOffloadMLP` ("Approach A", Task 5) and
/// `ChannelSplitMLP` ("Approach B", Task 6) against the all-GPU reference
/// MLP. This file delivers two things:
///
///   1. A correctness gate (always runs, guarded only by
///      `MLXFAST_RUN_MLX_RUNTIME_TESTS=1`) that proves all three variants
///      agree with the all-GPU reference at the ranked prefill shape
///      (S=512), on the SAME weights, before any timing is trusted.
///   2. A timing measurement GATED behind `OFFLOAD_COMPARE_TIMING=1`. The
///      user requires all real timing to run on a confirmed-idle machine
///      with their explicit go-ahead, so this path must NO-OP (return
///      immediately, no measurement) whenever that var is unset -- which is
///      every normal `swift test` invocation, including CI. The controller
///      runs the timing separately on a user-pinged idle box; this task does
///      not run it.
///
/// See `.superpowers/sdd/2026-08-30-ane-gpu-concurrent-offload/task-7-brief.md`.
@Suite(.serialized)
struct OffloadComparisonTests {
    // MARK: - Shared weight/reference helpers
    //
    // Copied (not refactored into a shared file, per the task-7 brief) from
    // `CoarseOffloadMLPTests` / `ChannelSplitMLPTests`: same `1/sqrt(inn)`
    // weight scaling so activations land at the well-conditioned,
    // ~unit-variance scale a real post-RMSNorm transformer MLP sees, and the
    // same all-GPU `quantizedMM`-only reference forward.

    private static func quantizedWeight(out: Int, inn: Int, seed: UInt64) -> (MLXArray, MLXArray, MLXArray) {
        MLXRandom.seed(seed)
        let scale = Float(1.0 / Double(inn).squareRoot())
        let w = (MLXRandom.normal([out, inn]) * scale).asType(.bfloat16)
        let (wq, scales, biases0) = quantized(w, groupSize: 64, bits: 4)
        let biases = biases0 ?? scales
        eval(wq, scales, biases)
        return (wq, scales, biases)
    }

    private struct Weights {
        let gateWq: MLXArray, gateScales: MLXArray, gateBiases: MLXArray
        let upWq: MLXArray, upScales: MLXArray, upBiases: MLXArray
        let downWq: MLXArray, downScales: MLXArray, downBiases: MLXArray
    }

    private static func makeWeights(hidden: Int, inter: Int, seed: UInt64) -> Weights {
        let (gateWq, gateScales, gateBiases) = quantizedWeight(out: inter, inn: hidden, seed: seed)
        let (upWq, upScales, upBiases) = quantizedWeight(out: inter, inn: hidden, seed: seed + 1)
        let (downWq, downScales, downBiases) = quantizedWeight(out: hidden, inn: inter, seed: seed + 2)
        return Weights(
            gateWq: gateWq, gateScales: gateScales, gateBiases: gateBiases,
            upWq: upWq, upScales: upScales, upBiases: upBiases,
            downWq: downWq, downScales: downScales, downBiases: downBiases)
    }

    /// All-GPU reference MLP: `down(silu(gate(x)) * up(x))`, every
    /// projection via the real 4-bit affine group-64 `quantizedMM` path.
    private static func referenceForward(_ x: MLXArray, _ w: Weights) -> MLXArray {
        let gate = quantizedMM(x, w.gateWq, scales: w.gateScales, biases: w.gateBiases,
                                transpose: true, groupSize: 64, bits: 4)
        let up = quantizedMM(x, w.upWq, scales: w.upScales, biases: w.upBiases,
                              transpose: true, groupSize: 64, bits: 4)
        let h = (silu(gate) * up).asType(.bfloat16)
        let y = quantizedMM(h, w.downWq, scales: w.downScales, biases: w.downBiases,
                             transpose: true, groupSize: 64, bits: 4)
        eval(y)
        return y
    }

    // MARK: - Part 1: correctness gate (always runs)

    /// The tighter of the two variants' measured tolerances
    /// (`ChannelSplitMLPTests`'s: max 0.09375, mean 0.01) so that a pass here
    /// proves BOTH variants are within the bound the comparison assumes --
    /// using the looser `CoarseOffloadMLPTests` bound (0.1) would let a
    /// borderline `ChannelSplitMLP` regression slip through unnoticed.
    private static let tolerance: Float = 0.09375
    private static let meanTolerance: Float = 0.01

    private func assertVariant(
        name: String, got: MLXArray, want: MLXArray
    ) {
        #expect(got.shape == want.shape, "\(name): shape mismatch: got \(String(describing: got.shape)), want \(String(describing: want.shape))")
        let absDiff = abs(got.asType(.float32) - want.asType(.float32))
        let maxAbsDiff = absDiff.max().item(Float.self)
        let meanAbsDiff = absDiff.mean().item(Float.self)
        print("OffloadComparisonTests \(name): maxAbsDiff=\(maxAbsDiff) meanAbsDiff=\(meanAbsDiff)")
        #expect(maxAbsDiff < Self.tolerance, "\(name) diverged from all-GPU reference: maxAbsErr=\(maxAbsDiff)")
        #expect(meanAbsDiff < Self.meanTolerance, "\(name) diverged from all-GPU reference: meanAbsErr=\(meanAbsDiff)")
    }

    /// All three variants (`CoarseOffloadMLP`, `ChannelSplitMLP` at each swept
    /// `aneFraction`) must match the all-GPU reference, on the SAME weights,
    /// at the ranked prefill shape S=512. This is the gate the timing
    /// comparison below leans on: it proves the A-vs-B-vs-reference
    /// comparison is apples-to-apples before any wall-clock number is
    /// trusted.
    @Test("all offload variants match the all-GPU reference at S=512")
    func allVariantsMatchReference() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        let hidden = 5120, inter = 17408, S = 512
        let w = Self.makeWeights(hidden: hidden, inter: inter, seed: 100)

        MLXRandom.seed(103)
        let x = MLXRandom.normal([S, hidden]).asType(.bfloat16)
        eval(x)

        let want = Self.referenceForward(x, w)

        let coarse = try CoarseOffloadMLP(
            gateW: w.gateWq, gateScales: w.gateScales, gateBiases: w.gateBiases,
            upW: w.upWq, upScales: w.upScales, upBiases: w.upBiases,
            downW: w.downWq, downScales: w.downScales, downBiases: w.downBiases,
            sequenceLength: S)
        let gotCoarse = try coarse(x)
        eval(gotCoarse)
        assertVariant(name: "CoarseOffloadMLP", got: gotCoarse, want: want)

        for fraction in Self.aneFractions {
            let split = try ChannelSplitMLP(
                gateW: w.gateWq, gateScales: w.gateScales, gateBiases: w.gateBiases,
                upW: w.upWq, upScales: w.upScales, upBiases: w.upBiases,
                downW: w.downWq, downScales: w.downScales, downBiases: w.downBiases,
                sequenceLength: S, aneFraction: fraction)
            let gotSplit = try split(x)
            eval(gotSplit)
            assertVariant(name: "ChannelSplitMLP(aneFraction=\(fraction))", got: gotSplit, want: want)
        }
    }

    // MARK: - Part 2: timing (GATED OFF by default)

    /// The `ChannelSplitMLP` fraction sweep, chosen so the controller can
    /// pick the best fraction rather than committing to one blind.
    private static let aneFractions: [Double] = [0.25, 0.375, 0.5]

    /// The prefill shapes swept: the ranked window (512) and 2x that, so the
    /// controller can see whether the ratio holds as S grows.
    private static let timingShapes: [Int] = [512, 1024]

    private static func timeCycle(_ body: () throws -> MLXArray) rethrows -> Double {
        let start = Date()
        eval(try body())
        return Date().timeIntervalSince(start)
    }

    /// One variant's `callAsFunction` as a closure timing `timeCycle` can
    /// call repeatedly without rebuilding weights/models each cycle.
    private struct Variant {
        let name: String
        let run: () throws -> MLXArray
    }

    /// Runs `warmup` (discarded) + `cycles` PAIRED cycles alternating
    /// reference then variant, so host noise (thermal drift, scheduler
    /// jitter) affects both sides of each pair roughly equally and divides
    /// out of the ratio -- the same adjacent-phase discipline
    /// `ANEChannelSplitPoCTests` uses for its GPU-alone-vs-split comparison.
    /// Returns (referenceSeconds, variantSeconds) per cycle.
    private static func pairedCycles(
        cycles: Int, reference: () throws -> MLXArray, variant: () throws -> MLXArray
    ) rethrows -> (ref: [Double], variant: [Double]) {
        // Untimed warmup cycle for both sides, discarded -- absorbs the
        // cold-start bump the other timing suites in this plan flag as an
        // outlier source (see `ANEChannelSplitPoCTests`).
        _ = try timeCycle(reference)
        _ = try timeCycle(variant)

        var refTimes: [Double] = []
        var variantTimes: [Double] = []
        for _ in 0 ..< cycles {
            refTimes.append(try timeCycle(reference))
            variantTimes.append(try timeCycle(variant))
        }
        return (refTimes, variantTimes)
    }

    /// A-vs-B-vs-reference wall-clock comparison across the fraction sweep
    /// and both timed shapes. NO-OPS (returns immediately, times nothing)
    /// unless `OFFLOAD_COMPARE_TIMING=1` is set -- the user requires this
    /// measurement to run only on a confirmed-idle machine with their
    /// explicit go-ahead, so it must never fire inside a normal `swift test`
    /// run (including CI). The controller sets the var and runs this alone,
    /// with `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` also set (this test does not
    /// touch the correctness `#expect`s above; it is purely informational).
    @Test("offload variants timing comparison vs all-GPU reference")
    func timingComparison() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["OFFLOAD_COMPARE_TIMING"] == "1" else { return }
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        let hidden = 5120, inter = 17408
        let cycles = 6

        var table = "# Task 7: A-vs-B offload comparison (ratio vs all-GPU reference)\n\n"
        table += "| variant | S | ratio (ref_s / variant_s) | ref median s/cycle | variant median s/cycle |\n"
        table += "|---|---|---|---|---|\n"
        var rawLines: [String] = []

        for S in Self.timingShapes {
            let w = Self.makeWeights(hidden: hidden, inter: inter, seed: UInt64(1000 + S))
            MLXRandom.seed(UInt64(2000 + S))
            let x = MLXRandom.normal([S, hidden]).asType(.bfloat16)
            eval(x)

            let referenceRun: () throws -> MLXArray = { Self.referenceForward(x, w) }

            let coarse = try CoarseOffloadMLP(
                gateW: w.gateWq, gateScales: w.gateScales, gateBiases: w.gateBiases,
                upW: w.upWq, upScales: w.upScales, upBiases: w.upBiases,
                downW: w.downWq, downScales: w.downScales, downBiases: w.downBiases,
                sequenceLength: S)

            var variants: [Variant] = [
                Variant(name: "CoarseOffloadMLP", run: { try coarse(x) }),
            ]
            for fraction in Self.aneFractions {
                let split = try ChannelSplitMLP(
                    gateW: w.gateWq, gateScales: w.gateScales, gateBiases: w.gateBiases,
                    upW: w.upWq, upScales: w.upScales, upBiases: w.upBiases,
                    downW: w.downWq, downScales: w.downScales, downBiases: w.downBiases,
                    sequenceLength: S, aneFraction: fraction)
                variants.append(Variant(name: "ChannelSplitMLP(aneFraction=\(fraction))", run: { try split(x) }))
            }

            for v in variants {
                let (refTimes, variantTimes) = try Self.pairedCycles(
                    cycles: cycles, reference: referenceRun, variant: v.run)
                let refMedian = refTimes.sorted()[refTimes.count / 2]
                let variantMedian = variantTimes.sorted()[variantTimes.count / 2]
                let ratio = refMedian / variantMedian
                table += "| \(v.name) | \(S) | \(String(format: "%.4f", ratio)) | " +
                    "\(String(format: "%.4f", refMedian)) | \(String(format: "%.4f", variantMedian)) |\n"
                for (i, (r, s)) in zip(refTimes, variantTimes).enumerated() {
                    rawLines.append(
                        "RAW\t\(v.name)\tS=\(S)\tcycle=\(i)\tref_s=\(String(format: "%.6f", r))\t" +
                            "variant_s=\(String(format: "%.6f", s))")
                }
                print("OffloadComparisonTests timing \(v.name) S=\(S): ratio=\(ratio) refMedian=\(refMedian) variantMedian=\(variantMedian)")
            }
        }

        table += "\n## Raw per-cycle seconds\n\n```\n" + rawLines.joined(separator: "\n") + "\n```\n"

        let reportPath = env["OFFLOAD_COMPARE_REPORT"] ??
            "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/" +
            "3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/offload-compare-report.md"
        try? table.write(toFile: reportPath, atomically: true, encoding: .utf8)

        // Deliberately no #expect on the ratios themselves -- this path is
        // informational, driven by the controller on an idle box. The
        // numerical-equivalence gate lives entirely in
        // `allVariantsMatchReference` above, so even this timing path stays
        // anchored to a correctness proof.
    }
}
