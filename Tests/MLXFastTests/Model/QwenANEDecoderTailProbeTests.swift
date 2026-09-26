import CryptoKit
import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANEDecoderTailProbeTests {
    @Test func checkpointDecoderTail() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_DECODER_TAIL_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let source = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_PROJECTION_HEAD"]))
        let arrays = try MLX.loadArrays(url: source)
        let projections = try MLX.loadArrays(url: root.appendingPathComponent("checkpoint-qkv.safetensors"))
        let attention = try MLX.loadArrays(url: root.appendingPathComponent("checkpoint-gated-output.safetensors"))
        let residual = try #require(projections["residual"])
        let referenceResidual = try #require(projections["reference_residual"])
        let candidateAttention = try #require(attention["ane_output"])
        let referenceAttention = try #require(attention["reference_output"])
        try #require([residual, referenceResidual, candidateAttention, referenceAttention].allSatisfy { $0.shape == [32, 5120] })
        func array(_ key: String) throws -> MLXArray { try #require(arrays[key]) }
        func weight(_ name: String) throws -> MLXArray {
            let prefix = "layers.0.mlp.\(name)_proj"
            return try aneProbeAffineWeight(array(prefix + ".weight"), scales: array(prefix + ".scales"),
                biases: array(prefix + ".biases"), groupSize: 64, bits: 4)
        }
        let gate = try weight("gate"), up = try weight("up"), down = try weight("down")
        try #require(gate.shape == [17408, 5120] && up.shape == gate.shape && down.shape == [5120, 17408])
        let postNorm = try array("layers.0.post_attention_layernorm.weight").asType(.float16)
        let finalNorm = try array("norm.weight").asType(.float16)
        try #require(postNorm.shape == [5120] && finalNorm.shape == [5120])
        let (blob, offsets) = buildMultiWeightBlob(chunks: [f16Bytes(gate), f16Bytes(up), f16Bytes(down)])
        let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        let text = buildQwenANEDecoderTailMILText(offsets: offsets,
            postNorm: postNorm.asArray(Float16.self), finalNorm: finalNorm.asArray(Float16.self), programTag: digest)
        try text.write(to: root.appendingPathComponent("checkpoint-decoder-tail.mil"), atomically: true, encoding: .utf8)
        var result: [String: Any] = ["phase": "compile", "rows": 32,
            "scope": "Checkpoint decoder chain with synthetic residual inputs and history",
            "effective_weight_blob_sha256": digest]
        func save() throws {
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("checkpoint-decoder-tail.json"), options: .atomic)
        }
        try save()
        do {
            let model = try ANEInMemoryModel(milText: text, weightBlob: blob, weightFileName: "weight.bin")
            try model.compile(); result["phase"] = "load"; try save()
            try model.load(); defer { model.unload() }
            func normalized(_ input: MLXArray, _ weights: MLXArray) -> MLXArray {
                let wide = input.asType(.float32)
                return (wide * rsqrt(mean(wide * wide, axis: -1, keepDims: true) + 1e-6)
                        * weights.asType(.float32)).asType(.float16)
            }
            let afterAttention = (referenceResidual.asType(.float16) + referenceAttention.asType(.float16)).asType(.float16)
            let mlpInput = normalized(afterAttention, postNorm)
            let g = matmul(mlpInput, gate.T), u = matmul(mlpInput, up.T)
            let activation = (g / (1 + exp(-g))).asType(.float16)
            let mlpOutput = matmul((activation * u).asType(.float16), down.T)
            let reference = normalized((afterAttention + mlpOutput).asType(.float16), finalNorm)
            // MLX eval materializes tensor operations; it does not evaluate source text.
            eval(reference)
            let input = concatenated([residual.asType(.float16), candidateAttention.asType(.float16)], axis: 1)
            eval(input)
            result["phase"] = "dispatch"; try save()
            let start = Date()
            let output = try ANEDirectDispatch.runConv(model: model, x: input,
                inputDim: 10240, outputDim: 5120, sequenceLength: 32)
            let actual = output.asArray(Float.self), expected = reference.asArray(Float.self)
            result["dispatch_and_read_seconds"] = Date().timeIntervalSince(start)
            try #require(actual.count == expected.count)
            let nonfinite = actual.filter { !$0.isFinite }.count
            result["nonfinite_values"] = nonfinite
            if nonfinite > 0 { try save(); try #require(nonfinite == 0) }
            var maximum: Double = 0, absolute: Double = 0, squared: Double = 0, referenceSquared: Double = 0
            for (value, reference) in zip(actual, expected) {
                let error = Double(value) - Double(reference)
                maximum = max(maximum, abs(error)); absolute += abs(error); squared += error * error
                referenceSquared += Double(reference) * Double(reference)
            }
            result["max_abs_error"] = maximum
            result["mean_abs_error"] = absolute / Double(actual.count)
            result["relative_rms_error"] = sqrt(squared / max(referenceSquared, Double.leastNormalMagnitude))
            result["phase"] = "complete"; try save()
            try MLX.save(arrays: ["ane_hidden": MLXArray(actual, [32, 5120]), "reference_hidden": reference],
                url: root.appendingPathComponent("checkpoint-decoder-tail.safetensors"))
            try #require(maximum < 0.01)
        } catch {
            result["error"] = String(describing: error); try save(); throw error
        }
    }
}
