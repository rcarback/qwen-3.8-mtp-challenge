// LOCAL M4 FORK ONLY. Private Apple frameworks; breaks on macOS updates.
// Never wired into the ranked forward; gated by MLXFAST_ANE_DIRECT=1 at call sites.
//
// Task C (ANE IOSurface / procedure-bank plan): combines Task B's fused
// ANE MLP (`ANEFusedMLP`, gate/up/down for the first `F` intermediate
// channels as ONE ANE program, fp16) with a GPU 4-bit complementary
// partial (the remaining `inter-F` channels, native `quantizedMM`) into a
// drop-in MLP, validated at real Qwen dimensions
// (hidden=5120, inter=17408). Mirrors `ANEOffload/ChannelSplitMLP.swift`'s
// slicing conventions (row slice for the output-channel axis of gate/up,
// column slice for `down`'s input-channel axis), but splits ACROSS the
// gate/up/down chain rather than per-projection: gate/up/down all use the
// SAME `F` intermediate channels, so `down`'s ANE prefix and GPU suffix
// are dequantized/sliced along its INPUT (inter) axis, not its output
// axis. See
// `.superpowers/sdd/2026-08-31-ane-iosurface-procedure-bank/task-C-brief.md`.
import Foundation
import MLX
import MLXNN

public final class ANEFusedSplitMLP {
    private static let groupSize = 64
    private static let bits = 4

    private let hidden: Int
    private let inter: Int
    private let f: Int
    private let aneMLP: ANEFusedMLP?

    private let gateSuffixWq: MLXArray?
    private let gateSuffixScales: MLXArray?
    private let gateSuffixBiases: MLXArray?
    private let upSuffixWq: MLXArray?
    private let upSuffixScales: MLXArray?
    private let upSuffixBiases: MLXArray?
    private let downSuffixWq: MLXArray?
    private let downSuffixScales: MLXArray?
    private let downSuffixBiases: MLXArray?

    /// `gate`/`up`: 4-bit affine group-64 triples, logical shape
    /// `[inter, hidden]`. `down`: 4-bit affine group-64 triple, logical
    /// shape `[hidden, inter]`. `aneFraction` (clamped to `[0,1]`) selects
    /// `F = round(aneFraction*inter/64)*64` intermediate channels to run
    /// through a single fused ANE program (fp16); the remaining
    /// `inter-F` channels run through native 4-bit `quantizedMM` on the
    /// GPU. `F==0` is pure GPU (the all-GPU reference, bit-identical);
    /// `F==inter` is pure ANE.
    public init(
        gateW: MLXArray, gateScales: MLXArray, gateBiases: MLXArray,
        upW: MLXArray, upScales: MLXArray, upBiases: MLXArray,
        downW: MLXArray, downScales: MLXArray, downBiases: MLXArray,
        hidden: Int, inter: Int, sequenceLength: Int, aneFraction: Double
    ) throws {
        let clampedFraction = min(max(aneFraction, 0.0), 1.0)
        let rawF = Int((clampedFraction * Double(inter) / 64.0).rounded()) * 64
        let f = min(max(rawF, 0), inter)
        self.hidden = hidden
        self.inter = inter
        self.f = f

        if f > 0 {
            let gatePrefix = ANEWeightPrep.dequantizeFP16(
                wq: gateW, scales: gateScales, biases: gateBiases, channelStart: 0, channelEnd: f)
            let upPrefix = ANEWeightPrep.dequantizeFP16(
                wq: upW, scales: upScales, biases: upBiases, channelStart: 0, channelEnd: f)
            let downPrefix = ANEWeightPrep.dequantizeFP16Columns(
                wq: downW, scales: downScales, biases: downBiases, columnStart: 0, columnEnd: f)
            eval(gatePrefix, upPrefix, downPrefix)
            aneMLP = try ANEFusedMLP(
                hidden: hidden, innerFraction: f, sequenceLength: sequenceLength,
                gate: gatePrefix, up: upPrefix, down: downPrefix)
        } else {
            aneMLP = nil
        }

        if f < inter {
            let gsWq = gateW[f ..< inter, 0...]
            let gsScales = gateScales[f ..< inter, 0...]
            let gsBiases = gateBiases[f ..< inter, 0...]
            let usWq = upW[f ..< inter, 0...]
            let usScales = upScales[f ..< inter, 0...]
            let usBiases = upBiases[f ..< inter, 0...]

            // `down`'s packed axis is the INPUT (inter) axis, not the
            // output axis: down's logical shape is `[hidden, inter]`, so a
            // GPU suffix over `[F..<inter]` intermediate channels is a
            // COLUMN slice of the packed `wq`/`scales`/`biases`, not a row
            // slice. Ratios are read back from the shipped arrays' own
            // shapes (rather than hardcoding the bits=4 pack factor)
            // so a mismatched envelope trips the precondition instead of
            // silently slicing the wrong span.
            precondition(downW.shape[1] != 0 && inter % downW.shape[1] == 0,
                         "ANEFusedSplitMLP: down wq column count \(downW.shape[1]) does not evenly divide inter=\(inter)")
            let packRatio = inter / downW.shape[1]
            precondition(f % packRatio == 0,
                         "ANEFusedSplitMLP: F=\(f) is not a multiple of down's pack ratio \(packRatio)")
            let wqColStart = f / packRatio
            let wqColEnd = inter / packRatio

            precondition(downScales.shape[1] != 0 && inter % downScales.shape[1] == 0,
                         "ANEFusedSplitMLP: down scales column count \(downScales.shape[1]) does not evenly divide inter=\(inter)")
            let groupRatio = inter / downScales.shape[1]
            precondition(f % groupRatio == 0,
                         "ANEFusedSplitMLP: F=\(f) is not a multiple of down's group ratio \(groupRatio)")
            let groupColStart = f / groupRatio
            let groupColEnd = inter / groupRatio

            let dsWq = downW[0..., wqColStart ..< wqColEnd]
            let dsScales = downScales[0..., groupColStart ..< groupColEnd]
            let dsBiases = downBiases[0..., groupColStart ..< groupColEnd]

            eval(gsWq, gsScales, gsBiases, usWq, usScales, usBiases, dsWq, dsScales, dsBiases)
            gateSuffixWq = gsWq
            gateSuffixScales = gsScales
            gateSuffixBiases = gsBiases
            upSuffixWq = usWq
            upSuffixScales = usScales
            upSuffixBiases = usBiases
            downSuffixWq = dsWq
            downSuffixScales = dsScales
            downSuffixBiases = dsBiases
        } else {
            gateSuffixWq = nil
            gateSuffixScales = nil
            gateSuffixBiases = nil
            upSuffixWq = nil
            upSuffixScales = nil
            upSuffixBiases = nil
            downSuffixWq = nil
            downSuffixScales = nil
            downSuffixBiases = nil
        }
    }

    /// `x: [S, hidden] -> [S, hidden]`. When `0<F<inter` the ANE and GPU
    /// partials run concurrently (Task D1): `makeInput` prepares the ANE
    /// request on this thread (MLX `eval` inside), `ConcurrentEngines.run`
    /// dispatches the MLX-free `predict` to a background queue while the
    /// GPU partial's `quantizedMM` chain runs (and evals) here, then
    /// `readOutput` converts the ANE result back to MLX on this thread
    /// before the two partials add. MLX `eval` therefore only ever happens
    /// on the calling thread.
    public func callAsFunction(_ x: MLXArray) throws -> MLXArray {
        if f == 0 {
            let y = try gpuPartial(x)
            eval(y)
            return y
        }
        guard let aneMLP else {
            preconditionFailure("ANEFusedSplitMLP: F=\(f)>0 but no ANEFusedMLP was built")
        }
        if f == inter {
            let y = try aneMLP(x.asType(.float16)).asType(.bfloat16)
            eval(y)
            return y
        }

        // CALLER thread: prepares the ANE input (MLX eval inside makeInput).
        let prepared = try aneMLP.makeInput(x.asType(.float16))
        let (_, gpu) = try ConcurrentEngines.run(
            ane: { try aneMLP.predict(prepared) }, // background: ObjC evaluate ONLY, no MLX
            gpu: {
                let g = try self.gpuPartial(x)
                eval(g)
                return g
            })
        // CALLER thread: reads the ANE output back into MLX.
        let anePartial = aneMLP.readOutput(prepared)
        let y = anePartial.asType(gpu.dtype) + gpu
        eval(y)
        return y
    }

    /// TEST-ONLY (visible via `@testable import`): identical math to
    /// `callAsFunction`'s mixed `0<F<inter` path, but runs the ANE and GPU
    /// partials sequentially with no `ConcurrentEngines`. Used to prove the
    /// concurrent refactor introduces no numeric change or corruption.
    func sequentialCallAsFunctionForTesting(_ x: MLXArray) throws -> MLXArray {
        precondition(f > 0 && f < inter, "sequentialCallAsFunctionForTesting only exercises the mixed ANE+GPU path")
        guard let aneMLP else {
            preconditionFailure("ANEFusedSplitMLP: F=\(f)>0 but no ANEFusedMLP was built")
        }
        let anePartial = try aneMLP(x.asType(.float16))
        let gpu = try gpuPartial(x)
        let y = anePartial.asType(gpu.dtype) + gpu
        eval(y)
        return y
    }

    /// Runs the `[F..<inter]` suffix chain through native 4-bit
    /// `quantizedMM`: identical to the all-GPU reference restricted to
    /// those channels.
    private func gpuPartial(_ x: MLXArray) throws -> MLXArray {
        guard let gateSuffixWq, let gateSuffixScales, let gateSuffixBiases,
              let upSuffixWq, let upSuffixScales, let upSuffixBiases,
              let downSuffixWq, let downSuffixScales, let downSuffixBiases
        else {
            preconditionFailure("ANEFusedSplitMLP: F=\(f)<inter=\(inter) but no GPU suffix triples were built")
        }
        let g = quantizedMM(x, gateSuffixWq, scales: gateSuffixScales, biases: gateSuffixBiases,
                             transpose: true, groupSize: Self.groupSize, bits: Self.bits)
        let u = quantizedMM(x, upSuffixWq, scales: upSuffixScales, biases: upSuffixBiases,
                             transpose: true, groupSize: Self.groupSize, bits: Self.bits)
        let act = (silu(g) * u).asType(.bfloat16)
        let y = quantizedMM(act, downSuffixWq, scales: downSuffixScales, biases: downSuffixBiases,
                             transpose: true, groupSize: Self.groupSize, bits: Self.bits)
        return y.asType(.bfloat16)
    }
}
