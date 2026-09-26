import CryptoKit
import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANEReadoutProbeTests {
    @Test func checkpointCoarseVocabulary() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_READOUT_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let source = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_PROJECTION_HEAD"]))
        let arrays = try MLX.loadArrays(url: source)
        let hiddenPath = root.appendingPathComponent("checkpoint-decoder-tail.safetensors")
        let hidden = try MLX.loadArrays(url: hiddenPath)
        let candidate = try #require(hidden["ane_hidden"]).asType(.float16)
        let reference = try #require(hidden["reference_hidden"]).asType(.float16)
        try #require(candidate.shape == [32, 5120] && reference.shape == candidate.shape)
        let packed = try #require(arrays["draft_lm_head.weight"])
        let scales = try #require(arrays["draft_lm_head.scales"])
        let biases = try #require(arrays["draft_lm_head.biases"])
        let vocabulary = 98336, realVocabulary = 98330, tileWidth = 2048
        try #require(packed.shape == [vocabulary, 320] && scales.shape == [vocabulary, 80] && biases.shape == scales.shape)
        func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
        var result: [String: Any] = ["phase": "prepare", "rows": 32, "vocabulary": vocabulary,
            "real_vocabulary": realVocabulary, "hidden_sha256": hash(try Data(contentsOf: hiddenPath)),
            "scope": "Full coarse projection with synthetic upstream history; no exact rerank or runtime selection"]
        var records: [[String: Any]] = []
        func save() throws {
            result["tiles"] = records
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("checkpoint-readout.json"), options: .atomic)
        }
        try save()
        var candidateTiles: [MLXArray] = [], referenceTiles: [MLXArray] = [], localReferenceTiles: [MLXArray] = []
        var squared: Double = 0, referenceSquared: Double = 0, maximum: Double = 0, localMaximum: Double = 0
        do {
            for start in stride(from: 0, to: vocabulary, by: tileWidth) {
                let end = min(start + tileWidth, vocabulary), count = end - start
                result["phase"] = "compile"; result["tile_start"] = start; try save()
                let weight = aneProbeAffineWeight(packed[start..<end], scales: scales[start..<end],
                    biases: biases[start..<end], groupSize: 64, bits: 2)
                let blob = buildConvWeightBlob(f16Bytes(weight))
                let digest = hash(blob)
                let text = buildConvMILText(inputDim: 5120, outputDim: count, sequenceLength: 32, programTag: digest)
                let model = try ANEInMemoryModel(milText: text, weightBlob: blob)
                try model.compile(); result["phase"] = "load"; try save()
                try model.load(); defer { model.unload() }
                let expectedTensor = matmul(reference, weight.T)
                let localTensor = matmul(candidate, weight.T)
                // MLX eval materializes tensor operations, not executable source text.
                eval(expectedTensor, localTensor)
                let expected = expectedTensor.asArray(Float.self), local = localTensor.asArray(Float.self)
                result["phase"] = "dispatch"; try save()
                let began = Date()
                let output = try ANEDirectDispatch.runConv(model: model, x: candidate,
                    inputDim: 5120, outputDim: count, sequenceLength: 32)
                let actual = output.asArray(Float.self)
                let elapsed = Date().timeIntervalSince(began)
                try #require(actual.count == expected.count && local.count == actual.count)
                let nonfinite = actual.filter { !$0.isFinite }.count
                result["nonfinite_values"] = nonfinite
                if nonfinite > 0 { try save(); try #require(nonfinite == 0) }
                var tileMaximum: Double = 0, tileLocalMaximum: Double = 0
                for index in actual.indices {
                    let error = Double(actual[index]) - Double(expected[index])
                    tileMaximum = max(tileMaximum, abs(error))
                    tileLocalMaximum = max(tileLocalMaximum, abs(Double(actual[index]) - Double(local[index])))
                    squared += error * error
                    referenceSquared += Double(expected[index]) * Double(expected[index])
                }
                maximum = max(maximum, tileMaximum); localMaximum = max(localMaximum, tileLocalMaximum)
                candidateTiles.append(MLXArray(actual, [32, count]))
                referenceTiles.append(expectedTensor); localReferenceTiles.append(localTensor)
                records.append(["start": start, "count": count, "weight_blob_sha256": digest,
                    "max_abs_error": tileMaximum, "local_max_abs_error": tileLocalMaximum,
                    "cold_dispatch_and_read_seconds": elapsed])
                try save()
            }
            let actual = concatenated(candidateTiles, axis: 1)
            let expected = concatenated(referenceTiles, axis: 1)
            let local = concatenated(localReferenceTiles, axis: 1)
            eval(actual, expected, local)
            try MLX.save(arrays: ["ane_logits": actual, "reference_logits": expected, "local_reference_logits": local],
                url: root.appendingPathComponent("checkpoint-readout.safetensors"))
            func maxima(_ tensor: MLXArray) -> [Int] {
                let flat = tensor.asArray(Float.self)
                return (0..<32).map { row in
                    let offset = row * vocabulary
                    var best = 0
                    for index in 1..<realVocabulary where flat[offset + index] > flat[offset + best] { best = index }
                    return best
                }
            }
            let actualIDs = maxima(actual), referenceIDs = maxima(expected), localIDs = maxima(local)
            result["diagnostic_coarse_argmax_matches"] = zip(actualIDs, referenceIDs).filter { $0 == $1 }.count
            result["diagnostic_local_argmax_matches"] = zip(actualIDs, localIDs).filter { $0 == $1 }.count
            result["max_abs_error"] = maximum
            result["local_max_abs_error"] = localMaximum
            result["relative_rms_error"] = sqrt(squared / max(referenceSquared, Double.leastNormalMagnitude))
            result["phase"] = "complete"; try save()
            try #require(maximum < 0.01)
        } catch {
            result["error"] = String(describing: error); try save(); throw error
        }
    }
}
