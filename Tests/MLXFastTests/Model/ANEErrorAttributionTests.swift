import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXLLM

/// Attributes the ANE offload's correctness error. The serve-path ablation
/// showed the GPU-fp16 fraction fails at token 31 while the ANE fraction fails
/// at token 0 -- so the ANE carries a large error beyond fp16-vs-4bit. This
/// measures, at real Qwen size on the SAME prefix channels:
///   A = ANE fused SwiGLU-down (gate/up conv, ANE silu, mul, down conv)
///   B = GPU fp16 with exact silu (same fp16 weights)
///   C = native 4-bit quantizedMM (the golden's arithmetic)
/// maxAbs(A,B) = ANE-specific error (silu LUT / radix-4 / padding).
/// maxAbs(B,C) = fp16-vs-4bit floor.
/// If A,B >> B,C, the dominant, fixable error is ANE-specific.
/// Gated on MLXFAST_RUN_MLX_RUNTIME_TESTS.
@Suite(.serialized)
struct ANEErrorAttributionTests {
    @Test("attribute ANE error: ANE-vs-GPUfp16 vs fp16-vs-4bit at real size")
    func attribute() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        try #require(ANERuntime.available())
        let hidden = 5_120, inter = 17_408, S = 512
        let f = 2_176  // fraction 0.125, known to compile

        // 4-bit gate/up [inter,hidden], down [hidden,inter], scaled 1/sqrt(fan_in).
        func q(out: Int, inn: Int, seed: UInt64) -> (MLXArray, MLXArray, MLXArray) {
            MLXRandom.seed(seed)
            let w = (MLXRandom.normal([out, inn]) * Float(1.0 / Double(inn).squareRoot())).asType(.bfloat16)
            let (wq, s, b0) = quantized(w, groupSize: 64, bits: 4)
            let b = b0 ?? s
            eval(wq, s, b)
            return (wq, s, b)
        }
        let (gWq, gS, gB) = q(out: inter, inn: hidden, seed: 1)
        let (uWq, uS, uB) = q(out: inter, inn: hidden, seed: 2)
        let (dWq, dS, dB) = q(out: hidden, inn: inter, seed: 3)
        let x = MLXRandom.normal([S, hidden]).asType(.bfloat16)
        eval(x)

        // fp16 prefix weights (same ones the ANE program is built from).
        let gPre = ANEWeightPrep.dequantizeFP16(wq: gWq, scales: gS, biases: gB, channelStart: 0, channelEnd: f)
        let uPre = ANEWeightPrep.dequantizeFP16(wq: uWq, scales: uS, biases: uB, channelStart: 0, channelEnd: f)
        let dPre = ANEWeightPrep.dequantizeFP16Columns(wq: dWq, scales: dS, biases: dB, columnStart: 0, columnEnd: f)
        eval(gPre, uPre, dPre)

        // A: ANE fused partial.
        let ane = try ANEFusedMLP(hidden: hidden, innerFraction: f, sequenceLength: S, gate: gPre, up: uPre, down: dPre)
        let A = try ane(x.asType(.float16)).asType(.float32)
        eval(A)

        // B: GPU fp16, exact silu, same fp16 weights.
        let x16 = x.asType(.float16)
        let gb = matmul(x16, gPre.transposed(1, 0))
        let ub = matmul(x16, uPre.transposed(1, 0))
        let actb = silu(gb) * ub
        let B = matmul(actb, dPre.transposed(1, 0)).asType(.float32)
        eval(B)

        // C: native 4-bit prefix (the golden's arithmetic) for the same F.
        let gWqP = gWq[0 ..< f, 0...], gSP = gS[0 ..< f, 0...], gBP = gB[0 ..< f, 0...]
        let uWqP = uWq[0 ..< f, 0...], uSP = uS[0 ..< f, 0...], uBP = uB[0 ..< f, 0...]
        // down column slice [:, 0:f] on the packed axis.
        let packRatio = inter / dWq.shape[1]
        let grpRatio = inter / dS.shape[1]
        let dWqP = dWq[0..., 0 ..< (f / packRatio)]
        let dSP = dS[0..., 0 ..< (f / grpRatio)]
        let dBP = dB[0..., 0 ..< (f / grpRatio)]
        let gc = quantizedMM(x, gWqP, scales: gSP, biases: gBP, transpose: true, groupSize: 64, bits: 4)
        let uc = quantizedMM(x, uWqP, scales: uSP, biases: uBP, transpose: true, groupSize: 64, bits: 4)
        let actc = (silu(gc) * uc).asType(.bfloat16)
        let C = quantizedMM(actc, dWqP, scales: dSP, biases: dBP, transpose: true, groupSize: 64, bits: 4).asType(.float32)
        eval(C)

        func maxAbs(_ p: MLXArray, _ q: MLXArray) -> Float {
            let d = MLX.abs(p - q); eval(d); return d.max().item(Float.self)
        }
        let aneSpecific = maxAbs(A, B)
        let fp16Floor = maxAbs(B, C)
        let aneVs4bit = maxAbs(A, C)
        print("ANE-ATTRIB: ANE-vs-GPUfp16(ANE-specific)=\(aneSpecific)  GPUfp16-vs-4bit(fp16 floor)=\(fp16Floor)  ANE-vs-4bit(total)=\(aneVs4bit)")

        // Localize the ANE-specific error: a single ANE gate conv vs the exact
        // GPU fp16 matmul of the SAME fp16 weights. If this is ~fp16 noise the
        // conv (radix-4 tiles) is fine and the error is in silu/mul/down; if
        // it is already ~aneSpecific, the conv itself carries it.
        let milConv = buildConvMILText(inputDim: hidden, outputDim: f, sequenceLength: S)
        let convBlob = buildConvWeightBlob(f16Bytes(gPre))
        let convModel = try ANEInMemoryModel(milText: milConv, weightBlob: convBlob)
        try convModel.compile(); try convModel.load(); defer { convModel.unload() }
        let aneGate = try ANEDirectDispatch.runConv(model: convModel, x: x.asType(.float16), inputDim: hidden, outputDim: f, sequenceLength: S).asType(.float32)
        let gpuGate = matmul(x16, gPre.transposed(1, 0)).asType(.float32)
        eval(aneGate, gpuGate)
        let convError = maxAbs(aneGate, gpuGate)
        print("ANE-ATTRIB: single-conv ANE-vs-GPUfp16=\(convError)")
        #expect(aneSpecific.isFinite && fp16Floor.isFinite)
    }
}
