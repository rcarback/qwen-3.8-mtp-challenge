import MLX

func aneProbeAffineWeight(
    _ packed: MLXArray, scales: MLXArray, biases: MLXArray,
    groupSize: Int, bits: Int
) -> MLXArray {
    // Affine dequantization computes in the coefficient dtype. Its output-dtype
    // option only casts afterward, so promote coefficients before expansion.
    MLX.dequantized(packed, scales: scales.asType(.float32), biases: biases.asType(.float32),
        groupSize: groupSize, bits: bits, mode: .affine).asType(.float16)
}
