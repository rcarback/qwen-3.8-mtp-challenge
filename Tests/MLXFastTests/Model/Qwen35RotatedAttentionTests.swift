import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import Testing

/// Attention is exercised here through `attentionWithCacheUpdate` directly
/// rather than through a loaded model, so these run without the 14 GiB
/// checkpoint. They assert the property the injection relies on: rotating
/// queries, keys, and values and inverse-rotating the output reproduces plain
/// attention, and does so more accurately than unrotated quantization.
@Suite(.serialized)
struct Qwen35RotatedAttentionTests {
    private func outlierKeys(_ shape: [Int]) -> MLXArray {
        let k = MLXRandom.normal(shape).asType(.float32)
        // Channels 17 and 200 are the outliers an affine group scale pays for.
        k[.ellipsis, 17] = MLXArray(Float(30))
        k[.ellipsis, 200] = MLXArray(Float(-25))
        return k
    }

    private func meanAbsError(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.mean(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
    }

    private func attend(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, cache: any KVCache
    ) -> MLXArray {
        attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache,
            scale: 1.0 / 16.0, mask: .causal)
    }

    @Test("rotation is exact on an unquantized cache")
    func exactOnBF16() {
        MLXRandom.seed(0x5157_454E)
        guard let rotation = Qwen35KVRotation(headDimension: 256) else {
            Issue.record("rotation construction failed"); return
        }
        let q = MLXRandom.normal([1, 16, 8, 256]).asType(.bfloat16)
        let k = outlierKeys([1, 4, 8, 256]).asType(.bfloat16)
        let v = MLXRandom.normal([1, 4, 8, 256]).asType(.bfloat16)

        let plain = attend(q, k, v, cache: KVCacheSimple())
        let rotated = rotation.inverse(
            attend(rotation.forward(q), rotation.forward(k),
                   rotation.forward(v), cache: KVCacheSimple()))
        // bfloat16 has about 3 decimal digits, and the Hadamard transform
        // sums 256 terms, so this compares means rather than maxima.
        #expect(meanAbsError(plain, rotated) < 2e-2)
    }

    @Test("rotation reduces quantization error at 4 bits")
    func improvesQuantizedFidelity() {
        MLXRandom.seed(0x5157_454E)
        guard let rotation = Qwen35KVRotation(headDimension: 256) else {
            Issue.record("rotation construction failed"); return
        }
        let q = MLXRandom.normal([1, 16, 8, 256]).asType(.bfloat16)
        let k = outlierKeys([1, 4, 8, 256]).asType(.bfloat16)
        let v = MLXRandom.normal([1, 4, 8, 256]).asType(.bfloat16)

        let reference = attend(q, k, v, cache: KVCacheSimple())

        let plainQuant = attend(
            q, k, v, cache: QuantizedKVCache(groupSize: 64, bits: 4))
        let rotatedQuant = rotation.inverse(
            attend(rotation.forward(q), rotation.forward(k),
                   rotation.forward(v),
                   cache: QuantizedKVCache(groupSize: 64, bits: 4)))

        let plainError = meanAbsError(reference, plainQuant)
        let rotatedError = meanAbsError(reference, rotatedQuant)
        print("KVQUANT plainError=\(plainError) rotatedError=\(rotatedError)")
        // The whole premise of the plan. If this fails, stop and measure
        // before writing any more code.
        #expect(rotatedError < plainError)
    }
}
