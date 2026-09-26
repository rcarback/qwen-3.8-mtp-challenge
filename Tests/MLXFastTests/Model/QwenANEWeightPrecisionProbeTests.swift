import Foundation
import MLX
import Testing

@Suite(.serialized)
struct QwenANEWeightPrecisionProbeTests {
    @Test func checkpointExpansionPrecision() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_WEIGHT_PRECISION_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let source = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_PROJECTION_HEAD"]))
        let arrays = try MLX.loadArrays(url: source)
        let capture = try MLX.loadArrays(url: root.appendingPathComponent("checkpoint-qkv.safetensors"))
        let input = try #require(capture["gpu_projection_input"]).asType(.float16)
        let prefix = "layers.0.self_attn.q_proj"
        let packed = try #require(arrays[prefix + ".weight"])[0..<2048]
        let scales = try #require(arrays[prefix + ".scales"])[0..<2048]
        let biases = try #require(arrays[prefix + ".biases"])[0..<2048]
        let inferred = MLX.dequantized(packed, scales: scales, biases: biases, groupSize: 64, bits: 4, mode: .affine)
        let oldWeight = inferred.asType(.float16)
        let outputCastWeight = MLX.dequantized(packed, scales: scales, biases: biases,
            groupSize: 64, bits: 4, mode: .affine, dtype: .float32).asType(.float16)
        let wideWeight = aneProbeAffineWeight(packed, scales: scales, biases: biases, groupSize: 64, bits: 4)
        let native = quantizedMM(input, packed, scales: scales, biases: biases,
            transpose: true, groupSize: 64, bits: 4, mode: .affine)
        let oldOutput = matmul(input, oldWeight.T), wideOutput = matmul(input, wideWeight.T)
        // MLX eval materializes tensor operations, not executable source text.
        eval(oldOutput, wideOutput, native)
        let expected = native.asArray(Float.self)
        func errors(_ output: MLXArray) throws -> [String: Double] {
            let actual = output.asArray(Float.self)
            try #require(actual.count == expected.count && actual.allSatisfy { $0.isFinite } && expected.allSatisfy { $0.isFinite })
            var maximum = 0.0, absolute = 0.0, squared = 0.0, energy = 0.0
            for (a, e) in zip(actual, expected) {
                let delta = Double(a) - Double(e)
                maximum = max(maximum, abs(delta)); absolute += abs(delta); squared += delta * delta
                energy += Double(e) * Double(e)
            }
            return ["max_abs_error": maximum, "mean_abs_error": absolute / Double(actual.count),
                    "relative_rms_error": sqrt(squared / max(energy, Double.leastNormalMagnitude))]
        }
        let changed = sum((oldWeight .!= wideWeight).asType(.int32))
        let castOnlyChanged = sum((oldWeight .!= outputCastWeight).asType(.int32))
        eval(changed, castOnlyChanged)
        let result: [String: Any] = ["phase": "complete", "rows": 32, "projection_features": 2048,
            "inferred_dequantized_dtype": String(describing: inferred.dtype),
            "changed_weights": changed.item(Int.self), "weight_count": oldWeight.size,
            "output_dtype_only_changed_weights": castOnlyChanged.item(Int.self),
            "inferred_then_fp16": try errors(oldOutput), "fp32_coefficients_then_fp16": try errors(wideOutput),
            "scope": "GPU-only coefficient-expansion control against direct affine quantized multiplication"]
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("weight-precision.json"), options: .atomic)
    }
}
