import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANEFinalSelectorProbeTests {
    @Test func boundaryTiesAndRealRerank() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_FINAL_SELECTOR_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let selector = try QwenANEFinalSelector()
        var ids = (0..<32).flatMap { lane in
            let row: [Int32] = [248319, 248044, 2048, 2047, 0] + (5..<32).map { Int32($0 + lane * 32) }
            return Array(row.reversed())
        }
        var scores = (0..<1024).map { Float16(($0 * 13) % 7) }
        #expect(throws: QwenANEFinalSelectorError.self) { try selector(scores: [], tokenIDs: ids) }
        var invalid = ids; invalid[0] = 248320
        #expect(throws: QwenANEFinalSelectorError.self) { try selector(scores: scores, tokenIDs: invalid) }
        var invalidScores = scores; invalidScores[0] = .nan
        #expect(throws: QwenANEFinalSelectorError.self) { try selector(scores: invalidScores, tokenIDs: ids) }
        for allEqual in [false, true] {
            if allEqual { scores = Array(repeating: -65504, count: 1024) }
            let tokens = try selector(scores: scores, tokenIDs: ids)
            for lane in 0..<32 {
                let begin = lane * 32
                let best = try #require((begin..<(begin + 32)).min {
                    scores[$0] == scores[$1] ? ids[$0] < ids[$1] : scores[$0] > scores[$1]
                })
                try #require(tokens[lane] == ids[best])
            }
        }
        let selection = try MLX.loadArrays(url: root.appendingPathComponent("selection-readout.safetensors"))
        let rerank = try MLX.loadArrays(url: root.appendingPathComponent("checkpoint-rerank.safetensors"))
        let compact = try #require(selection["compact_ids"]).asArray(Int32.self)
        let union = try #require(rerank["union_compact_ids"]).asArray(Int32.self)
        let values = try #require(rerank["ane_scores"]).asArray(Float16.self)
        let lookup = Dictionary(uniqueKeysWithValues: union.enumerated().map { ($0.element, $0.offset) })
        try #require(compact.count == 1024 && values.count == 32 * union.count)
        ids = compact.map { $0 < 98304 ? $0 : 248044 + $0 - 98304 }
        scores = try compact.enumerated().map { index, id in
            values[(index / 32) * union.count + (try #require(lookup[id]))]
        }
        let tokens = try selector(scores: scores, tokenIDs: ids)
        let data = try Data(contentsOf: root.appendingPathComponent("checkpoint-rerank.json"))
        let control = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let expected = try #require(control["ane_tokens"] as? [Int]).map(Int32.init)
        try #require(tokens == expected)
        try JSONSerialization.data(withJSONObject: ["phase": "complete", "synthetic_rows": 64,
            "real_rows": 32, "real_tokens": tokens.map(Int.init), "matches_cpu_oracle": 32,
            "scope": "ANE final selection from supplied FP16 rerank scores; target projection still separate"], options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("final-selector.json"), options: .atomic)
    }
}
