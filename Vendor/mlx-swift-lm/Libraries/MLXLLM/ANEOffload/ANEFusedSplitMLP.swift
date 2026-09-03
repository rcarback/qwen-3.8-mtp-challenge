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
    /// The row count the ANE program was compiled at. `callAsFunction`
    /// accepts any `S <= sequenceLength` and pads up to it for the ANE leg
    /// only; see `padForANE`.
    private let sequenceLength: Int
    private let aneMLP: ANEFusedMLP?
    // GPU-fp16 ablation (MLX_ANE_FP16_GPU=1): the dequantized fp16 prefix
    // weights, kept so the prefix partial can run as a pure-GPU fp16 matmul
    // instead of on the ANE. Isolates fp16-vs-4bit from ANE-specific error.
    private let ablate: Bool
    private let gatePrefixFP16: MLXArray?
    private let upPrefixFP16: MLXArray?
    private let downPrefixFP16: MLXArray?

    private let gateSuffixWq: MLXArray?
    private let gateSuffixScales: MLXArray?
    private let gateSuffixBiases: MLXArray?
    private let upSuffixWq: MLXArray?
    private let upSuffixScales: MLXArray?
    private let upSuffixBiases: MLXArray?
    private let downSuffixWq: MLXArray?
    private let downSuffixScales: MLXArray?
    private let downSuffixBiases: MLXArray?

    /// `F = round(aneFraction*inter/64)*64`, clamped to `[0, inter]`: the
    /// number of intermediate channels the ANE side owns.
    public static func prefixChannels(inter: Int, aneFraction: Double) -> Int {
        let clampedFraction = min(max(aneFraction, 0.0), 1.0)
        let rawF = Int((clampedFraction * Double(inter) / 64.0).rounded()) * 64
        return min(max(rawF, 0), inter)
    }

    /// `gate`/`up`: 4-bit affine group-64 triples, logical shape
    /// `[inter, hidden]`. `down`: 4-bit affine group-64 triple, logical
    /// shape `[hidden, inter]`. `aneFraction` (clamped to `[0,1]`) selects
    /// `F = round(aneFraction*inter/64)*64` intermediate channels to run
    /// through a single fused ANE program (fp16); the remaining
    /// `inter-F` channels run through native 4-bit `quantizedMM` on the
    /// GPU. `F==0` is pure GPU (the all-GPU reference, bit-identical);
    /// `F==inter` is pure ANE. `prefixFP16`, when given, supplies the ANE
    /// side's fp16 `[F,hidden]`/`[F,hidden]`/`[hidden,F]` weights directly
    /// (the bf16 base slices, see `ANEBF16WeightSource`) instead of
    /// dequantizing the 4-bit prefix; the GPU suffix is unaffected.
    public init(
        gateW: MLXArray, gateScales: MLXArray, gateBiases: MLXArray,
        upW: MLXArray, upScales: MLXArray, upBiases: MLXArray,
        downW: MLXArray, downScales: MLXArray, downBiases: MLXArray,
        hidden: Int, inter: Int, sequenceLength: Int, aneFraction: Double,
        prefixFP16: ANEPrefixWeights? = nil
    ) throws {
        let f = Self.prefixChannels(inter: inter, aneFraction: aneFraction)
        self.hidden = hidden
        self.inter = inter
        self.f = f
        self.sequenceLength = sequenceLength

        self.ablate = ANESplitConfig.fp16GpuAblate
        if f > 0 {
            let gatePrefix: MLXArray
            let upPrefix: MLXArray
            let downPrefix: MLXArray
            if let prefixFP16 {
                precondition(prefixFP16.gate.shape == [f, hidden] && prefixFP16.up.shape == [f, hidden]
                             && prefixFP16.down.shape == [hidden, f],
                             "ANEFusedSplitMLP: prefixFP16 shapes \(prefixFP16.gate.shape)/\(prefixFP16.up.shape)/\(prefixFP16.down.shape) do not match F=\(f) hidden=\(hidden)")
                gatePrefix = prefixFP16.gate.asType(.float16)
                upPrefix = prefixFP16.up.asType(.float16)
                downPrefix = prefixFP16.down.asType(.float16)
            } else {
                gatePrefix = ANEWeightPrep.dequantizeFP16(
                    wq: gateW, scales: gateScales, biases: gateBiases, channelStart: 0, channelEnd: f)
                upPrefix = ANEWeightPrep.dequantizeFP16(
                    wq: upW, scales: upScales, biases: upBiases, channelStart: 0, channelEnd: f)
                downPrefix = ANEWeightPrep.dequantizeFP16Columns(
                    wq: downW, scales: downScales, biases: downBiases, columnStart: 0, columnEnd: f)
            }
            eval(gatePrefix, upPrefix, downPrefix)
            if ablate {
                // GPU-fp16 ablation: keep the fp16 prefix for a GPU matmul,
                // skip the ANE program build entirely (no ANE, no sandbox).
                gatePrefixFP16 = gatePrefix
                upPrefixFP16 = upPrefix
                downPrefixFP16 = downPrefix
                aneMLP = nil
            } else {
                gatePrefixFP16 = nil
                upPrefixFP16 = nil
                downPrefixFP16 = nil
                aneMLP = try ANEFusedMLP(
                    hidden: hidden, innerFraction: f, sequenceLength: sequenceLength,
                    gate: gatePrefix, up: upPrefix, down: downPrefix)
            }
        } else {
            aneMLP = nil
            gatePrefixFP16 = nil
            upPrefixFP16 = nil
            downPrefixFP16 = nil
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

    /// Pads `[S, hidden] -> [sequenceLength, hidden]` with zero rows.
    ///
    /// The ANE program is compiled at one fixed row count, so a shorter
    /// input has to be padded up to it. Only the ANE leg needs that. The
    /// GPU suffix is a shape-agnostic `quantizedMM` chain, so it runs on
    /// the real rows alone, and padding it computes rows that are thrown
    /// away.
    ///
    /// That matters because the GPU owns the larger share. At the deployed
    /// `MLX_ANE_FRACTION` of 0.3125 the ANE holds `F` of `inter` channels
    /// and the GPU holds the other 68.75%, so 68.75% of any padding waste
    /// is GPU-side and removable here. A 732-token prompt compiled at 1024
    /// discards 28.5% of all MLP work when both engines pad, and 12.5%
    /// when only the ANE does.
    ///
    /// Zero rows are exact for the same reason the padding is legal at
    /// all: this is a per-token MLP (gate/up/down plus SwiGLU) with no
    /// cross-token mixing, so an all-zero input row can only produce an
    /// output row that the caller discards. It would NOT be legal for an
    /// attention program, where padded rows enter the softmax.
    private func padForANE(_ x: MLXArray) -> MLXArray {
        let s = x.dim(0)
        precondition(
            s <= sequenceLength,
            "ANEFusedSplitMLP: input has \(s) rows but the ANE program was compiled at \(sequenceLength)")
        if s == sequenceLength { return x }
        return concatenated([x, MLXArray.zeros([sequenceLength - s, hidden], dtype: x.dtype)], axis: 0)
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
        if ablate {
            // GPU-fp16 ablation: the SAME fp16 prefix function the ANE would
            // compute, but on the GPU, plus the identical 4-bit suffix. If
            // this still flips token-0 vs the 4-bit golden, the divergence is
            // fp16-vs-4bit representation, not ANE-specific.
            let anePartial = gpuFp16Prefix(x)
            if f == inter {
                eval(anePartial)
                return anePartial
            }
            let gpu = try gpuPartial(x)
            let y = anePartial.asType(gpu.dtype) + gpu
            eval(y)
            return y
        }
        guard let aneMLP else {
            preconditionFailure("ANEFusedSplitMLP: F=\(f)>0 but no ANEFusedMLP was built")
        }
        if f == inter {
            let padded = try aneMLP(padForANE(x.asType(.float16))).asType(.bfloat16)
            let y = padded[0 ..< x.dim(0), 0...]
            eval(y)
            return y
        }

        // CALLER thread: prepares the ANE input (MLX eval inside makeInput).
        // Only this leg is padded to the program's compiled row count; the
        // GPU partial below runs on the caller's real rows.
        let tokens = x.dim(0)
        let prepared = try aneMLP.makeInput(padForANE(x.asType(.float16)))
        let (_, gpu) = try ConcurrentEngines.run(
            ane: { try aneMLP.predict(prepared) }, // background: ObjC evaluate ONLY, no MLX
            gpu: {
                let g = try self.gpuPartial(x)
                eval(g)
                return g
            })
        // CALLER thread: reads the ANE output back into MLX, dropping the
        // padded rows before the add so both operands are `[tokens, hidden]`.
        let anePartial = aneMLP.readOutput(prepared)
        let anePrefix = tokens == sequenceLength ? anePartial : anePartial[0 ..< tokens, 0...]
        let y = anePrefix.asType(gpu.dtype) + gpu
        eval(y)
        return y
    }

    /// TEST-ONLY (public so `@testable import MLXFastModel`'s re-export of
    /// `MLXLLM` can reach it): identical math to `callAsFunction`'s mixed
    /// `0<F<inter` path, but runs the ANE and GPU partials sequentially with
    /// no `ConcurrentEngines`. Used to prove the concurrent refactor
    /// introduces no numeric change or corruption.
    public func sequentialCallAsFunctionForTesting(_ x: MLXArray) throws -> MLXArray {
        precondition(f > 0 && f < inter, "sequentialCallAsFunctionForTesting only exercises the mixed ANE+GPU path")
        guard let aneMLP else {
            preconditionFailure("ANEFusedSplitMLP: F=\(f)>0 but no ANEFusedMLP was built")
        }
        let tokens = x.dim(0)
        let anePadded = try aneMLP(padForANE(x.asType(.float16)))
        let anePartial = tokens == sequenceLength ? anePadded : anePadded[0 ..< tokens, 0...]
        let gpu = try gpuPartial(x)
        let y = anePartial.asType(gpu.dtype) + gpu
        eval(y)
        return y
    }

    /// GPU-fp16 ablation: the `[0..<F]` prefix SwiGLU-down computed as a pure
    /// GPU fp16 dequant-matmul (same fp16 weights the ANE would use). `[S,hidden]`.
    private func gpuFp16Prefix(_ x: MLXArray) -> MLXArray {
        guard let gatePrefixFP16, let upPrefixFP16, let downPrefixFP16 else {
            preconditionFailure("ANEFusedSplitMLP: ablation prefix weights missing")
        }
        let x16 = x.asType(.float16)
        let g = matmul(x16, gatePrefixFP16.transposed(1, 0))          // [S,F]
        let u = matmul(x16, upPrefixFP16.transposed(1, 0))            // [S,F]
        let act = (silu(g) * u)                                        // [S,F] fp16
        let y = matmul(act, downPrefixFP16.transposed(1, 0))          // [S,hidden]
        return y.asType(.bfloat16)
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
