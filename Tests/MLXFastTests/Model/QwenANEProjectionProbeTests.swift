import CryptoKit
import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANEProjectionProbeTests {
    @Test func checkpointQKV() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_PROJECTION_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
        let source = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_PROJECTION_HEAD"]))
        let arrays = try MLX.loadArrays(url: source)
        func array(_ key: String) throws -> MLXArray { try #require(arrays[key]) }
        let normalizeInput = env["MLXFAST_ANE_PROJECTION_INPUT_NORM"] == "1"
        let fcInput = env["MLXFAST_ANE_PROJECTION_FC_INPUT"] == "1"
        try #require(!fcInput || normalizeInput)
        let stem = normalizeInput ? "checkpoint-normalized-qkv" : "checkpoint-qkv"
        let norm = try array("layers.0.input_layernorm.weight").asType(.float16)
        try #require(norm.shape == [5120])
        var projections: [MLXArray] = []
        for name in ["q", "k", "v"] {
            let prefix = "layers.0.self_attn.\(name)_proj"
            let weight = try array(prefix + ".weight")
            let scales = try array(prefix + ".scales")
            let biases = try array(prefix + ".biases")
            try #require(weight.dim(1) == 640 && scales.dim(1) == 80 && biases.shape == scales.shape)
            projections.append(aneProbeAffineWeight(weight, scales: scales, biases: biases,
                groupSize: 64, bits: 4))
        }
        let weight = concatenated(projections, axis: 0)
        try #require(weight.shape == [14336, 5120])
        eval(weight)
        let dense = weight.asArray(Float16.self)
        let blob = buildConvWeightBlob(dense.withUnsafeBytes { Data($0) })
        let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        let text = buildQwenANEProjectionMILText(norm: normalizeInput ? norm.asArray(Float16.self) : nil,
                                                programTag: digest)
        try text.write(to: root.appendingPathComponent("ane-batch/\(stem).mil"), atomically: true, encoding: .utf8)
        let path = root.appendingPathComponent("ane-batch/\(stem).json")
        var result: [String: Any] = ["phase": "compile", "source": source.path, "effective_weight_blob_sha256": digest,
            "rows": 32, "q_rows_including_gate": 12288, "k_rows": 1024, "v_rows": 1024]
        result["input_normalization"] = normalizeInput
        result["mil_sha256"] = SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
        func save() throws {
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]).write(to: path, options: .atomic)
        }
        try save()
        do {
            let model = try ANEInMemoryModel(milText: text, weightBlob: blob)
            try model.compile(); result["phase"] = "load"; try save()
            try model.load(); defer { model.unload() }
            let inputs: [Float16] = (0..<(32 * 5120)).map { index in
                let integer = (index * 7 + index / 5120) % 101 - 50
                return Float16(Float(integer) / 32.0)
            }
            let input: MLXArray
            let referenceInput: MLXArray
            var positions: MLXArray?
            if fcInput {
                let fcPath = root.appendingPathComponent("ane-batch/checkpoint-input.safetensors")
                let fc = try MLX.loadArrays(url: fcPath)
                positions = fc["positions"]
                if let positions { try #require(positions.shape == [32] && positions.dtype == .int32) }
                input = try #require(fc["ane_fc"]).asType(.float16)
                referenceInput = try #require(fc["reference_fc"]).asType(.float16)
                result["fc_input_sha256"] = SHA256.hash(data: try Data(contentsOf: fcPath))
                    .map { String(format: "%02x", $0) }.joined()
            } else {
                input = MLXArray(inputs, [32, 5120])
                referenceInput = input
            }
            try #require(input.shape == [32, 5120] && referenceInput.shape == [32, 5120])
            let wideInput = referenceInput.asType(.float32)
            let normalized = (wideInput * rsqrt(mean(wideInput * wideInput, axis: -1, keepDims: true) + 1e-6)
                              * norm.asType(.float32)).asType(.float16)
            let projectionInput = normalizeInput ? normalized : referenceInput
            let reference = matmul(projectionInput, weight.T)
            eval(reference)
            result["phase"] = "dispatch"; try save()
            let started = Date()
            let predicted = try ANEDirectDispatch.runConv(model: model, x: input, inputDim: 5120, outputDim: 14336, sequenceLength: 32)
            let actual = predicted.asArray(Float.self)
            let expected = reference.asArray(Float.self)
            let elapsed = Date().timeIntervalSince(started)
            try #require(actual.count == expected.count)
            let nonfinite = actual.filter { !$0.isFinite }.count
            result["nonfinite_values"] = nonfinite
            if nonfinite != 0 { try save(); try #require(nonfinite == 0) }
            var maximum: Float = 0
            var absolute: Double = 0, squared: Double = 0, referenceSquared: Double = 0
            for (value, reference) in zip(actual, expected) {
                let error: Double = Double(value) - Double(reference)
                maximum = max(maximum, Float(abs(error)))
                absolute += abs(error); squared += error * error
                referenceSquared += Double(reference) * Double(reference)
            }
            let relative = sqrt(squared / max(referenceSquared, Double.leastNormalMagnitude))
            result["max_abs_error"] = maximum
            result["mean_abs_error"] = absolute / Double(actual.count)
            result["relative_rms_error"] = relative
            result["dispatch_and_read_seconds"] = elapsed
            result["phase"] = "complete"; try save()
            var saved = ["input": input, "gpu_projection_input": projectionInput,
                                 "gpu_qkv": reference, "ane_qkv": MLXArray(actual, [32, 14336]),
                                 "q_norm": try array("layers.0.self_attn.q_norm.weight"),
                                 "k_norm": try array("layers.0.self_attn.k_norm.weight")]
            if let positions { saved["positions"] = positions }
            if normalizeInput {
                saved["residual"] = input
                saved["reference_residual"] = referenceInput
            }
            try MLX.save(arrays: saved, url: root.appendingPathComponent("ane-batch/\(stem).safetensors"))
            try #require(relative < 0.01)
        } catch {
            result["error"] = String(describing: error); try save(); throw error
        }
    }
}
