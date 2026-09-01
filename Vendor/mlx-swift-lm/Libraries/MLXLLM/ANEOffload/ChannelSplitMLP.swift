// Task 6 (ANE+GPU concurrent offload plan): "Approach B / channel-split" --
// split EACH of gate/up/down by OUTPUT channels: the ANE computes a
// fraction of the channels (fp16) concurrent with the GPU computing the
// rest (4-bit), then concat. See
// `.superpowers/sdd/2026-08-30-ane-gpu-concurrent-offload/task-6-brief.md`.
// Task 5's `CoarseOffloadMLP` is "Approach A / coarse" (one whole
// projection per engine); Task 7 times the two against each other.
import Foundation
import MLX
import MLXNN

/// Per-projection channel split of `down(silu(gate(x)) * up(x))`. For each
/// of `gate`, `up`, `down` the first `F` output channels run on the ANE
/// (fp16, via a warm `ANEGemm` built at init) concurrently with the
/// remaining `out-F` channels on the GPU (4-bit affine group-64
/// `quantizedMM`, exactly as shipped), then the two parts are concatenated.
/// `F` is `round(aneFraction * out / 64) * 64`. 64-alignment is a schedule
/// choice, not a packing constraint -- groups pack along the `in` axis, so
/// any integer `F` is a valid row slice of the shipped 4-bit operand.
public final class ChannelSplitMLP {
    private static let groupSize = 64
    private static let bits = 4

    /// One projection's channel split. `aneGemm` and the GPU suffix triple
    /// are mutually exclusive only at the F==0 / F==out edges -- everywhere
    /// in between both are present and `callAsFunction` runs them
    /// concurrently.
    private struct ProjSplit {
        let out: Int
        let f: Int
        let aneGemm: ANEGemm?
        let suffixWq: MLXArray?
        let suffixScales: MLXArray?
        let suffixBiases: MLXArray?
    }

    private let gate: ProjSplit
    private let up: ProjSplit
    private let down: ProjSplit

    /// 4-bit affine group-64 weights, as shipped: gateW/upW `[inter, hidden]`,
    /// downW `[hidden, inter]`. `aneFraction` is clamped to `[0,1]`; for each
    /// projection the first `F = round(aneFraction*out/64)*64` output
    /// channels are dequantized to fp16 and compiled into a warm `ANEGemm`
    /// for `sequenceLength`, and the remaining `[F..<out]` rows of the
    /// quantized triple are kept as the GPU suffix operand -- sliced
    /// directly from the shipped 4-bit `wq`/`scales`/`biases` along the
    /// output-channel (row) axis, never dequantized-then-requantized.
    public init(
        gateW: MLXArray, gateScales: MLXArray, gateBiases: MLXArray,
        upW: MLXArray, upScales: MLXArray, upBiases: MLXArray,
        downW: MLXArray, downScales: MLXArray, downBiases: MLXArray,
        sequenceLength: Int, aneFraction: Double
    ) throws {
        let clampedFraction = min(max(aneFraction, 0.0), 1.0)
        gate = try Self.makeSplit(
            wq: gateW, scales: gateScales, biases: gateBiases,
            fraction: clampedFraction, sequenceLength: sequenceLength)
        up = try Self.makeSplit(
            wq: upW, scales: upScales, biases: upBiases,
            fraction: clampedFraction, sequenceLength: sequenceLength)
        down = try Self.makeSplit(
            wq: downW, scales: downScales, biases: downBiases,
            fraction: clampedFraction, sequenceLength: sequenceLength)
    }

    /// Computes `F` for one projection's `out` and builds its ANE prefix
    /// (`ANEGemm`, if `F>0`) and GPU suffix triple (a row slice of the
    /// shipped quantized weight, if `F<out`).
    private static func makeSplit(
        wq: MLXArray, scales: MLXArray, biases: MLXArray,
        fraction: Double, sequenceLength: Int
    ) throws -> ProjSplit {
        let out = wq.shape[0]
        let rawF = Int((fraction * Double(out) / 64.0).rounded()) * 64
        let f = min(max(rawF, 0), out)

        var aneGemm: ANEGemm?
        if f > 0 {
            let prefixFp16 = ANEWeightPrep.dequantizeFP16(
                wq: wq, scales: scales, biases: biases,
                channelStart: 0, channelEnd: f)
            aneGemm = try ANEGemm(weight: prefixFp16, sequenceLength: sequenceLength)
        }

        var suffixWq: MLXArray?
        var suffixScales: MLXArray?
        var suffixBiases: MLXArray?
        if f < out {
            let sliceWq = wq[f ..< out, 0...]
            let sliceScales = scales[f ..< out, 0...]
            let sliceBiases = biases[f ..< out, 0...]
            eval(sliceWq, sliceScales, sliceBiases)
            suffixWq = sliceWq
            suffixScales = sliceScales
            suffixBiases = sliceBiases
        }

        return ProjSplit(
            out: out, f: f, aneGemm: aneGemm,
            suffixWq: suffixWq, suffixScales: suffixScales, suffixBiases: suffixBiases)
    }

    /// `x: [S, hidden] -> [S, hidden]`. Each of gate/up (on `x`) and down
    /// (on `silu(gate)*up`) is dispatched through `projSplit`, which runs
    /// the ANE prefix and GPU suffix concurrently (or takes the pure-GPU /
    /// pure-ANE fast path at the F==0 / F==out edges).
    public func callAsFunction(_ x: MLXArray) throws -> MLXArray {
        let gateOut = try projSplit(x, gate)
        let upOut = try projSplit(x, up)
        let h = (silu(gateOut) * upOut).asType(.bfloat16)
        let y = try projSplit(h, down)
        return y
        // PERF: measured in Task 7 (A-vs-B fraction sweep, idle-box, user-gated)
    }

    /// Runs one projection's channel split on input `a`. `makeInput` and
    /// `readOutput` run on this (the calling) thread -- both do MLX `eval`
    /// internally -- and only `aneGemm.predict` (Core ML, no MLX) runs on
    /// `ConcurrentEngines.run`'s background queue; `gpu` runs ON the calling
    /// thread (see `ConcurrentEngines.run`), so there is zero MLX `eval` off
    /// the caller. Result is normalized to bf16 on every path so gate/up/down
    /// outputs combine consistently regardless of which edge case each
    /// projection took.
    private func projSplit(_ a: MLXArray, _ split: ProjSplit) throws -> MLXArray {
        if split.f == 0 {
            guard let suffixWq = split.suffixWq, let suffixScales = split.suffixScales,
                  let suffixBiases = split.suffixBiases
            else {
                preconditionFailure("ChannelSplitMLP: F==0 but no GPU suffix triple was built")
            }
            let g = quantizedMM(a, suffixWq, scales: suffixScales, biases: suffixBiases,
                                 transpose: true, groupSize: Self.groupSize, bits: Self.bits)
            eval(g)
            return g.asType(.bfloat16)
        }

        guard let aneGemm = split.aneGemm else {
            preconditionFailure("ChannelSplitMLP: F>0 but no ANEGemm was built for this projection")
        }

        if split.f == split.out {
            let input = try aneGemm.makeInput(a)
            let output = try aneGemm.predict(input)
            return aneGemm.readOutput(output).asType(.bfloat16)
        }

        guard let suffixWq = split.suffixWq, let suffixScales = split.suffixScales,
              let suffixBiases = split.suffixBiases
        else {
            preconditionFailure("ChannelSplitMLP: 0<F<out but no GPU suffix triple was built")
        }

        // CALLER thread: prepares the ANE input (MLX eval inside makeInput).
        let aneInput = try aneGemm.makeInput(a)
        let (aneOutput, gpuArr) = try ConcurrentEngines.run(
            ane: { try aneGemm.predict(aneInput) }, // background: Core ML ONLY
            gpu: {
                let g = quantizedMM(a, suffixWq, scales: suffixScales, biases: suffixBiases,
                                     transpose: true, groupSize: Self.groupSize, bits: Self.bits)
                eval(g)
                return g
            })
        // CALLER thread: reads the ANE output back into MLX.
        let anePart = aneGemm.readOutput(aneOutput)
        let combined = concatenated([anePart.asType(gpuArr.dtype), gpuArr], axis: 1)
        eval(combined)
        return combined.asType(.bfloat16)
    }
}
