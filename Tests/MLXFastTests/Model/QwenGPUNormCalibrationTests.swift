import CryptoKit
import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenGPUNormCalibrationTests {
    @Test func fixedStagePrecisionComparison() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_GPU_NORM_CALIBRATION"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            let root = URL(fileURLWithPath: try #require(env["MLXFAST_GPU_NORM_ROOT"]))
            let output = URL(fileURLWithPath: try #require(env["MLXFAST_GPU_NORM_OUTPUT"]))
            func json(_ url: URL) throws -> [String: Any] {
                try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            }
            func hash(_ data: Data) -> String {
                SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            }
            let projectionMetadata = try json(root.appendingPathComponent("checkpoint-normalized-qkv.json"))
            let headURL = URL(fileURLWithPath: try #require(projectionMetadata["source"] as? String))
            let head = try MLX.loadArrays(url: headURL)
            func expand(_ prefix: String) throws -> MLXArray {
                try aneProbeAffineWeight(#require(head[prefix + ".weight"]), scales: #require(head[prefix + ".scales"]),
                    biases: #require(head[prefix + ".biases"]), groupSize: 64, bits: 4)
            }
            var measurements: [[String: Any]] = []
            for stage in ["fc", "qkv"] {
                let stem = stage == "fc" ? "checkpoint-input" : "checkpoint-normalized-qkv"
                let captureURL = root.appendingPathComponent(stem + ".safetensors")
                let capture = try MLX.loadArrays(url: captureURL)
                let metadata = try json(root.appendingPathComponent(stem + ".json"))
                let weight = try stage == "fc" ? expand("fc") : concatenated(["q", "k", "v"].map {
                    try expand("layers.0.self_attn.\($0)_proj")
                }, axis: 0)
                let blobHash = hash(buildConvWeightBlob(f16Bytes(weight.contiguous())))
                try #require(blobHash == metadata["effective_weight_blob_sha256"] as? String)
                let input = try #require(capture["input"]).asType(.float16)
                let norm: MLXArray
                let shaped: MLXArray
                if stage == "fc" {
                    norm = try concatenated([#require(head["pre_fc_norm_embedding.weight"]),
                        #require(head["pre_fc_norm_hidden.weight"])], axis: 0).asType(.float16).reshaped([2, 5120])
                    shaped = input.reshaped([32, 2, 5120])
                } else {
                    norm = try #require(head["layers.0.input_layernorm.weight"]).asType(.float16)
                    shaped = input
                }
                let expected = try #require(capture[stage == "fc" ? "ane_fc" : "ane_qkv"]).asType(.float32)
                for mode in ["fp16-graph", "wide-all", "wide-statistics", "wide-unit"] {
                    let wide = shaped.asType(.float32)
                    let inverse = rsqrt(mean(wide * wide, axis: -1, keepDims: true) + 1e-6)
                    let normalized: MLXArray
                    switch mode {
                    case "wide-all": normalized = (wide * inverse * norm.asType(.float32)).asType(.float16)
                    case "wide-statistics": normalized = (shaped * inverse.asType(.float16)) * norm
                    case "wide-unit": normalized = (wide * inverse).asType(.float16) * norm
                    default:
                        let variance = mean(shaped * shaped, axis: -1, keepDims: true) + MLXArray(Float16(1e-6))
                        normalized = (shaped * rsqrt(variance)) * norm
                    }
                    let actual = matmul(normalized.reshaped(input.shape), weight.T).asType(.float32)
                    eval(actual)
                    try #require(all(isFinite(actual)).item(Bool.self))
                    let error = actual - expected
                    let relative = sqrt(Double(mean(error * error).item(Float.self)) / Double(mean(expected * expected).item(Float.self)))
                    measurements.append(["stage": stage, "mode": mode, "relative_rms_error": relative,
                        "max_abs_error": Double(max(abs(error)).item(Float.self)),
                        "capture_sha256": hash(try Data(contentsOf: captureURL)), "effective_weight_blob_sha256": blobHash])
                }
            }
            let report: [String: Any] = ["phase": "complete", "head_sha256": hash(try Data(contentsOf: headURL)),
                "scope": "Fixed actual ANE FC and QKV stage inputs from one coding seed; GPU normalization variants, no model tuning",
                "measurements": measurements]
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: output, options: .atomic)
        }
    }
}
