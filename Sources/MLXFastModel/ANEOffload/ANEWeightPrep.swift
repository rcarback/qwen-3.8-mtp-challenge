import Foundation
import MLX

/// Prepares MLX 4-bit affine-quantized projection weights for the ANE
/// (fp16) side of a concurrent ANE/GPU offload split.
///
/// The ANE primitive needs a plain fp16 `[out, in]` weight (or a
/// output-channel slice `[F, in]`), dequantized from the model's shipped
/// MLX 4-bit affine group-64 weights. This must match the GPU
/// `quantizedMM` path bit-for-bit, so it calls MLX's own `dequantized(...)`
/// op rather than hand-rolling the affine dequant math.
public enum ANEWeightPrep {
    /// Dequantize MLX 4-bit (wq,scales,biases,groupSize:64,bits:4) to fp16 [out,in],
    /// optionally only output-channel rows [channelStart..<channelEnd].
    public static func dequantizeFP16(
        wq: MLXArray, scales: MLXArray, biases: MLXArray,
        channelStart: Int, channelEnd: Int
    ) -> MLXArray {
        let full = dequantized(wq, scales: scales, biases: biases, groupSize: 64, bits: 4)
        return full[channelStart ..< channelEnd, 0...].asType(.float16)
    }
}
