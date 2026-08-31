// Task 5 (ANE+GPU concurrent offload plan): "Approach A / coarse" -- compute
// the WHOLE `up` projection on the ANE concurrent with the WHOLE `gate`
// projection on the GPU, then `down` on the GPU. See
// `.superpowers/sdd/2026-08-30-ane-gpu-concurrent-offload/task-5-brief.md`.
import CoreML
import Foundation
import MLX
import MLXNN

/// Whole-projection ANE/GPU split of `down(silu(gate(x)) * up(x))`. `gate`
/// and `down` stay 4-bit affine group-64, run on the GPU via `quantizedMM`
/// exactly as shipped; only `up` is dequantized to fp16 once (at init) for
/// the ANE. This is the coarsest possible split -- one whole projection per
/// engine -- as opposed to Task 6's channel-split approach.
public final class CoarseOffloadMLP {
    private static let groupSize = 64
    private static let bits = 4

    private let aneUp: ANEGemm
    private let gateWq: MLXArray
    private let gateScales: MLXArray
    private let gateBiases: MLXArray
    private let downWq: MLXArray
    private let downScales: MLXArray
    private let downBiases: MLXArray

    /// 4-bit affine group-64 weights, as shipped: gateW/upW `[inter, hidden]`,
    /// downW `[hidden, inter]`. `up` is dequantized to fp16 once here and
    /// compiled into a warm `ANEGemm` for the given `sequenceLength`; `gate`
    /// and `down` stay 4-bit and are dispatched through `quantizedMM` on
    /// every `callAsFunction`.
    public init(
        gateW: MLXArray, gateScales: MLXArray, gateBiases: MLXArray,
        upW: MLXArray, upScales: MLXArray, upBiases: MLXArray,
        downW: MLXArray, downScales: MLXArray, downBiases: MLXArray,
        sequenceLength: Int
    ) throws {
        self.gateWq = gateW
        self.gateScales = gateScales
        self.gateBiases = gateBiases
        self.downWq = downW
        self.downScales = downScales
        self.downBiases = downBiases

        let inter = upW.shape[0]
        let upFp16 = ANEWeightPrep.dequantizeFP16(
            wq: upW, scales: upScales, biases: upBiases,
            channelStart: 0, channelEnd: inter)
        aneUp = try ANEGemm(weight: upFp16, sequenceLength: sequenceLength)
    }

    /// x: `[S, hidden]` -> `[S, hidden]`. `up` runs on the ANE (Core ML,
    /// fp16) concurrently with `gate` on the GPU (4-bit `quantizedMM`, evaled
    /// inside the `gpu` closure); `down` then runs on the GPU.
    ///
    /// `makeInput`/`readOutput` run on this (the calling) thread -- both do
    /// MLX `eval` internally -- and only `aneUp.predict` (Core ML, no MLX)
    /// runs on `ConcurrentEngines.run`'s background queue, so the `gpu`
    /// closure's `eval` is the only MLX `eval` ever invoked off this thread.
    public func callAsFunction(_ x: MLXArray) throws -> MLXArray {
        let upInput = try aneUp.makeInput(x)
        let (upOut, gateArr) = try ConcurrentEngines.run(
            ane: { try self.aneUp.predict(upInput) },
            gpu: {
                let g = quantizedMM(x, self.gateWq, scales: self.gateScales, biases: self.gateBiases,
                                     transpose: true, groupSize: Self.groupSize, bits: Self.bits)
                eval(g)
                return g
            })
        let up = aneUp.readOutput(upOut)
        // gateArr is bf16 (from quantizedMM on a bf16 x), up is fp16 (ANE
        // output) -- cast both to bf16 before combining so `down`'s
        // quantizedMM sees the same dtype the all-GPU reference produces.
        let h = (silu(gateArr) * up).asType(.bfloat16)
        let y = quantizedMM(h, downWq, scales: downScales, biases: downBiases,
                             transpose: true, groupSize: Self.groupSize, bits: Self.bits)
        eval(y)
        return y
        // PERF: measured in Task 7 (A-vs-B, idle-box, user-gated)
    }
}
