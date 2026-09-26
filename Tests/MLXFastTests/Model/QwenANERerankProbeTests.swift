import CryptoKit
import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANERerankProbeTests {
    @Test func checkpointShortlistRerank() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_RERANK_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let weightsRoot = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_TARGET_WEIGHTS"]))
        let indexData = try Data(contentsOf: weightsRoot.appendingPathComponent("model.safetensors.index.json"))
        let index = try #require(JSONSerialization.jsonObject(with: indexData) as? [String: Any])
        let mapping = try #require(index["weight_map"] as? [String: String])
        let prefix = "language_model.lm_head"
        let shard = try #require(mapping[prefix + ".weight"])
        try #require(mapping[prefix + ".scales"] == shard && mapping[prefix + ".biases"] == shard)
        let arrays = try MLX.loadArrays(url: weightsRoot.appendingPathComponent(shard))
        let hidden = try MLX.loadArrays(url: root.appendingPathComponent("checkpoint-decoder-tail.safetensors"))
        let logits = try MLX.loadArrays(url: root.appendingPathComponent("checkpoint-readout.safetensors"))
        let selected = try MLX.loadArrays(url: root.appendingPathComponent("selection-readout.safetensors"))
        let selection = try #require(selected["compact_ids"])
        try #require(selection.shape == [32, 32])
        let candidateIDs = selection.asArray(Int32.self).map(Int.init)
        let referenceLogits = try #require(logits["reference_logits"])
        try #require(referenceLogits.shape == [32, 98336])
        let flat = referenceLogits.asArray(Float.self)
        try #require(flat.allSatisfy { $0.isFinite })
        var referenceIDs: [Int] = []
        for row in 0..<32 {
            let offset = row * 98336
            let ranked = (0..<98330).sorted {
                let a = flat[offset + $0], b = flat[offset + $1]
                return a == b ? $0 < $1 : a > b
            }
            referenceIDs.append(contentsOf: ranked.prefix(32))
        }
        try #require(candidateIDs.allSatisfy { (0..<98330).contains($0) })
        let union = Array(Set(candidateIDs + referenceIDs)).sorted()
        let lookup = Dictionary(uniqueKeysWithValues: union.enumerated().map { ($0.element, $0.offset) })
        let fullIDs = union.map { Int32($0 < 98304 ? $0 : 248044 + $0 - 98304) }
        let packed = try #require(arrays[prefix + ".weight"])
        let scales = try #require(arrays[prefix + ".scales"])
        let biases = try #require(arrays[prefix + ".biases"])
        try #require(packed.shape == [248320, 640] && scales.shape == [248320, 80] && biases.shape == scales.shape)
        let indices = MLXArray(fullIDs)
        let weight = aneProbeAffineWeight(take(packed, indices, axis: 0),
            scales: take(scales, indices, axis: 0), biases: take(biases, indices, axis: 0), groupSize: 64, bits: 4)
        let candidate = try #require(hidden["ane_hidden"]).asType(.float16)
        let reference = try #require(hidden["reference_hidden"]).asType(.float16)
        try #require(candidate.shape == [32, 5120] && reference.shape == candidate.shape)
        let blob = buildConvWeightBlob(f16Bytes(weight))
        let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        let model = try ANEInMemoryModel(milText: buildConvMILText(inputDim: 5120,
            outputDim: union.count, sequenceLength: 32, programTag: digest), weightBlob: blob)
        try model.compile(); try model.load(); defer { model.unload() }
        let actual = try ANEDirectDispatch.runConv(model: model, x: candidate,
            inputDim: 5120, outputDim: union.count, sequenceLength: 32)
        let expected = matmul(reference, weight.T), local = matmul(candidate, weight.T)
        // MLX eval materializes tensor operations, not executable source text.
        eval(actual, expected, local)
        let a = actual.asArray(Float.self), e = expected.asArray(Float.self), l = local.asArray(Float.self)
        try #require(a.count == 32 * union.count && e.count == a.count && l.count == a.count)
        try #require(a.allSatisfy { $0.isFinite } && e.allSatisfy { $0.isFinite } && l.allSatisfy { $0.isFinite })
        func winners(_ scores: [Float], _ shortlist: [Int]) throws -> [Int] {
            var result: [Int] = []
            for row in 0..<32 {
                let ids = Array(shortlist[(row * 32)..<((row + 1) * 32)])
                var best = ids[0]
                for id in ids.dropFirst() {
                    let value = scores[row * union.count + (try #require(lookup[id]))]
                    let previous = scores[row * union.count + (try #require(lookup[best]))]
                    if value > previous || (value == previous && id < best) { best = id }
                }
                result.append(best < 98304 ? best : 248044 + best - 98304)
            }
            return result
        }
        let actualWinners = try winners(a, candidateIDs)
        let referenceWinners = try winners(e, referenceIDs)
        let localWinners = try winners(l, candidateIDs)
        let shortlistControl = try winners(e, candidateIDs)
        let report: [String: Any] = ["phase": "complete", "rows": 32,
            "union_candidates": union.count, "weight_blob_sha256": digest,
            "ane_tokens": actualWinners, "reference_tokens": referenceWinners,
            "local_gpu_tokens": localWinners, "reference_on_ane_shortlist_tokens": shortlistControl,
            "end_to_end_matches": zip(actualWinners, referenceWinners).filter { $0 == $1 }.count,
            "same_input_matches": zip(actualWinners, localWinners).filter { $0 == $1 }.count,
            "shortlist_only_matches": zip(shortlistControl, referenceWinners).filter { $0 == $1 }.count,
            "scope": "Synthetic saved chain; actual target affine4 weights expanded to FP16; offline union staging and CPU winner oracle; not native BF16 rerank or runtime integration"]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("checkpoint-rerank.json"), options: .atomic)
        try MLX.save(arrays: ["ane_scores": actual, "reference_scores": expected, "local_scores": local,
            "union_compact_ids": MLXArray(union.map(Int32.init))], url: root.appendingPathComponent("checkpoint-rerank.safetensors"))
    }
}
