import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANECoarseReadoutProbeTests {
    @Test func rejectsMissingTiles() {
        #expect(throws: QwenANECoarseReadoutError.self) {
            try QwenANECoarseReadout(tileWeightBlobs: [:])
        }
    }

    @Test func retainedProgramsMatchSavedReadout() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_COARSE_READOUT_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let head = try MLX.loadArrays(url: URL(fileURLWithPath: try #require(env["MLXFAST_ANE_PROJECTION_HEAD"])))
        let hidden = try MLX.loadArrays(url: root.appendingPathComponent("retained-cache-embedding.safetensors"))
        let saved = try MLX.loadArrays(url: root.appendingPathComponent("checkpoint-readout.safetensors"))
        let input = try #require(hidden["ane_hidden"]).asArray(Float16.self)
        let expected = try #require(saved["ane_logits"]).asArray(Float16.self)
        try #require(input.count == 32 * 5120 && expected.count == 32 * 98336)
        let packed = try #require(head["draft_lm_head.weight"])
        let scales = try #require(head["draft_lm_head.scales"]), biases = try #require(head["draft_lm_head.biases"])
        try #require(packed.shape == [98336, 320] && scales.shape == [98336, 80] && biases.shape == scales.shape)
        var report: [String: Any] = ["phase": "prepare", "scope": "49 resident coarse ANE programs; saved live-decoder hidden states; no live final rerank"]
        func save() throws {
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("retained-coarse-readout.json"), options: .atomic)
        }
        try save()
        do {
            var blobs: [Int: Data] = [:]
            for start in stride(from: 0, to: 98336, by: 2048) {
                let end = min(start + 2048, 98336)
                let weight = aneProbeAffineWeight(packed[start..<end], scales: scales[start..<end],
                    biases: biases[start..<end], groupSize: 64, bits: 2)
                blobs[start] = buildConvWeightBlob(f16Bytes(weight))
            }
            report["phase"] = "load"; try save()
            let readout = try QwenANECoarseReadout(tileWeightBlobs: blobs)
            #expect(throws: QwenANECoarseReadoutError.self) { try readout([]) }
            var invalid = input; invalid[0] = .nan
            #expect(throws: QwenANECoarseReadoutError.self) { try readout(invalid) }
            var trials: [[String: Any]] = []
            for reversed in [false, true] {
                let lanes = reversed ? Array((0..<32).reversed()) : Array(0..<32)
                let rows = lanes.flatMap { Array(input[($0 * 5120)..<(($0 + 1) * 5120)]) }
                report["phase"] = "dispatch"; report["reversed"] = reversed; try save()
                let began = Date()
                let actual = try readout(rows)
                let seconds = Date().timeIntervalSince(began)
                var mismatches = 0
                for (lane, original) in lanes.enumerated() {
                    mismatches += zip(actual[(lane * 98336)..<((lane + 1) * 98336)],
                        expected[(original * 98336)..<((original + 1) * 98336)]).filter { $0 != $1 }.count
                }
                trials.append(["reversed": reversed, "mismatched_values": mismatches,
                    "dispatch_and_read_seconds": seconds])
                report["trials"] = trials; try save()
                try #require(mismatches == 0)
            }
            report["phase"] = "complete"; try save()
        } catch {
            report["error"] = String(describing: error); try save(); throw error
        }
    }
}
