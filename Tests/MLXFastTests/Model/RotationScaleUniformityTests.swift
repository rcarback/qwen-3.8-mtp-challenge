import Foundation
import MLX
import MLXLLM
import MLXRandom
import Testing

/// Does the Hadamard rotation equalize the per-group quantization scales?
///
/// If it does, a fused quantized SDPA kernel can carry ONE scale per head
/// vector instead of one per group of 64, which both shrinks the stored
/// metadata and collapses the per-group fixup to a single multiply. That
/// optimization is available only because the rotation flattens outliers, so
/// this measures the spread of per-group ranges with and without it.
@Suite(.serialized)
struct RotationScaleUniformityTests {
    @Test("rotation equalizes per-group scales across a head vector")
    func scaleSpread() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(0x5157_454E)
        guard let rotation = Qwen35KVRotation(headDimension: 256) else {
            Issue.record("rotation construction failed"); return
        }

        // Realistic key vectors: Gaussian bulk with persistent outlier channels,
        // which is the shape that makes group scales disagree.
        var k = MLXRandom.normal([1, 4, 2048, 256]).asType(.float32)
        k[.ellipsis, 17] = MLXArray(Float(30))
        k[.ellipsis, 200] = MLXArray(Float(-25))
        k[.ellipsis, 91] = MLXArray(Float(12))

        func groupScaleSpread(_ x: MLXArray) -> (Float, Float) {
            // Four groups of 64 along the head dimension.
            let g = x.reshaped([-1, 4, 64])
            let ranges = MLX.max(g, axis: -1) - MLX.min(g, axis: -1)
            let lo = MLX.min(ranges, axis: -1)
            let hi = MLX.max(ranges, axis: -1)
            // Ratio of widest to narrowest group within each vector. 1.0 means
            // one scale would serve all four groups exactly.
            let ratio = hi / MLX.maximum(lo, MLXArray(Float(1e-6)))
            return (
                MLX.mean(ratio).item(Float.self),
                MLX.max(ratio).item(Float.self)
            )
        }

        let plain = groupScaleSpread(k)
        let rotated = groupScaleSpread(rotation.forward(k))
        print("SCALESPREAD plain mean=\(plain.0) max=\(plain.1)")
        print("SCALESPREAD rot   mean=\(rotated.0) max=\(rotated.1)")
        print("SCALESPREAD improvement mean=\(plain.0 / rotated.0)x")
    }
}
