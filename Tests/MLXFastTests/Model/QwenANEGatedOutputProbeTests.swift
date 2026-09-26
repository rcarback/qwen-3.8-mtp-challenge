import CryptoKit
import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANEGatedOutputProbeTests {
    @Test func checkpointGatedOutput() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_GATED_OUTPUT_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
        let source = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_PROJECTION_HEAD"]))
        let arrays = try MLX.loadArrays(url: source)
        let attention = try MLX.loadArrays(url: root.appendingPathComponent("ane-batch/checkpoint-attention.safetensors"))
        let projections = try MLX.loadArrays(url: root.appendingPathComponent("ane-batch/checkpoint-qkv.safetensors"))
        func array(_ key: String) throws -> MLXArray { try #require(arrays[key]) }
        let prefix = "layers.0.self_attn.o_proj"
        let weight = try aneProbeAffineWeight(array(prefix + ".weight"), scales: array(prefix + ".scales"),
            biases: array(prefix + ".biases"), groupSize: 64, bits: 4)
        try #require(weight.shape == [5120, 6144])
        eval(weight)
        let blob = buildConvWeightBlob(weight.asArray(Float16.self).withUnsafeBytes { Data($0) })
        let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        let text = buildQwenANEGatedOutputMILText(programTag: digest)
        try text.write(to: root.appendingPathComponent("ane-batch/checkpoint-gated-output.mil"), atomically: true, encoding: .utf8)
        let path = root.appendingPathComponent("ane-batch/checkpoint-gated-output.json")
        var result: [String: Any] = ["phase": "compile", "effective_weight_blob_sha256": digest,
                                   "rows": 32, "scope": "Checkpoint weights, synthetic upstream inputs and history"]
        func save() throws {
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]).write(to: path, options: .atomic)
        }
        try save()
        do {
            let model = try ANEInMemoryModel(milText: text, weightBlob: blob)
            try model.compile(); result["phase"] = "load"; try save()
            try model.load(); defer { model.unload() }
            func gates(_ name: String) throws -> [Float16] {
                let tensor = try #require(projections[name])
                try #require(tensor.shape == [32, 14336])
                let flat = tensor.asArray(Float16.self)
                var values: [Float16] = []
                for row in 0..<32 {
                    for head in 0..<24 {
                        let start = row * 14336 + head * 512 + 256
                        values += flat[start..<(start + 256)]
                    }
                }
                return values
            }
            let candidateAttention = try #require(attention["ane_attention"])
            let referenceAttention = try #require(attention["reference_attention"])
            try #require(candidateAttention.shape == [32, 6144] && referenceAttention.shape == [32, 6144])
            let a = candidateAttention.asArray(Float16.self), g = try gates("ane_qkv")
            var input: [Float16] = []
            for row in 0..<32 {
                input += a[(row * 6144)..<((row + 1) * 6144)]
                input += g[(row * 6144)..<((row + 1) * 6144)]
            }
            let referenceGate = try MLXArray(gates("gpu_qkv"), [32, 6144]).asType(.float32)
            let referenceGated = (referenceAttention.asType(.float32) / (1 + exp(-referenceGate))).asType(.float16)
            let reference = matmul(referenceGated, weight.T)
            eval(reference)
            result["phase"] = "dispatch"; try save()
            let started = Date()
            let predicted = try ANEDirectDispatch.runConv(model: model, x: MLXArray(input, [32, 12288]), inputDim: 12288, outputDim: 5120, sequenceLength: 32)
            let actual = predicted.asArray(Float.self), expected = reference.asArray(Float.self)
            try #require(actual.count == expected.count)
            result["dispatch_and_read_seconds"] = Date().timeIntervalSince(started)
            let nonfinite = actual.filter { !$0.isFinite }.count
            result["nonfinite_values"] = nonfinite
            if nonfinite > 0 { try save(); try #require(nonfinite == 0) }
            var maximum: Float = 0
            var absolute: Double = 0, squared: Double = 0, referenceSquared: Double = 0
            for (value, reference) in zip(actual, expected) {
                let error: Double = Double(value) - Double(reference)
                maximum = max(maximum, Float(abs(error)))
                absolute += abs(error); squared += error * error
                referenceSquared += Double(reference) * Double(reference)
            }
            result["max_abs_error"] = maximum
            result["mean_abs_error"] = absolute / Double(actual.count)
            result["relative_rms_error"] = sqrt(squared / max(referenceSquared, Double.leastNormalMagnitude))
            result["phase"] = "complete"; try save()
            try MLX.save(arrays: ["ane_output": MLXArray(actual, [32, 5120]), "reference_output": reference],
                         url: root.appendingPathComponent("ane-batch/checkpoint-gated-output.safetensors"))
            try #require(maximum < 0.01)
        } catch {
            result["error"] = String(describing: error); try save(); throw error
        }
    }
}
