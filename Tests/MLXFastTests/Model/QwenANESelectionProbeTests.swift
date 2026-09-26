import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANESelectionProbeTests {
    @Test func shortlistSelection() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_SELECTION_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let valuesOnly = env["MLXFAST_ANE_SELECTION_VALUES_ONLY"] == "1"
        let reductions = env["MLXFAST_ANE_SELECTION_REDUCTIONS"] == "1"
        let readout = env["MLXFAST_ANE_SELECTION_READOUT"] == "1"
        try #require(!readout || reductions)
        try #require(!reductions || !valuesOnly)
        let outputFeatures = valuesOnly ? 32 : 64
        var text = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
          func main<ios18>(tensor<fp16, [1, 2048, 1, 32]> x) {
            int32 axis = const()[name=string("axis"), val=int32(1)];
            int32 k = const()[name=string("k"), val=int32(32)];
            bool ascending = const()[name=string("ascending"), val=bool(false)];
            bool sorted = const()[name=string("sorted"), val=bool(true)];
            bool indices = const()[name=string("indices"), val=bool(\(valuesOnly ? "false" : "true"))];
            bool interleave = const()[name=string("interleave"), val=bool(false)];
            string outputType = const()[name=string("outputType"), val=string("fp16")];
            tensor<fp16, [1, 32, 1, 32]> values, tensor<int32, [1, 32, 1, 32]> ids = topk(x=x, k=k, axis=axis, ascending=ascending, sort=sorted, return_indices=indices)[name=string("top")];
            tensor<fp16, [1, 32, 1, 32]> floatIDs = cast(x=ids, dtype=outputType)[name=string("floatIDs")];
            tensor<fp16, [1, 64, 1, 32]> y = concat(values=(values, floatIDs), axis=axis, interleave=interleave)[name=string("y")];
          } -> (y);
        }
        """
        if valuesOnly {
            text = text.replacingOccurrences(of: "tensor<fp16, [1, 32, 1, 32]> values, tensor<int32, [1, 32, 1, 32]> ids =", with: "tensor<fp16, [1, 32, 1, 32]> values =")
            text = text.components(separatedBy: "\n").filter {
                !$0.contains("floatIDs =") && !$0.contains("y = concat")
            }.joined(separator: "\n").replacingOccurrences(of: "} -> (y);", with: "} -> (values);")
        }
        if reductions { text = buildQwenANESelectionMILText() }
        let stem = readout ? "selection-readout" : (reductions ? "selection-reductions" : (valuesOnly ? "selection-values-only" : "selection-probe"))
        try text.write(to: root.appendingPathComponent("\(stem).mil"), atomically: true, encoding: .utf8)
        var result: [String: Any] = ["phase": "compile", "rows": 32, "candidates_per_row": 2048,
            "selected_per_row": 32, "scope": "Synthetic shortlist tiles; indices fit exactly in FP16"]
        func save() throws {
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("\(stem).json"), options: .atomic)
        }
        try save()
        do {
            if readout {
                let selector = try ANEDirectDispatch.FP16Loop(milText: text, weightBlob: Data(),
                    inputDim: 2048, outputDim: 64, batch: 32)
                let arrays = try MLX.loadArrays(url: root.appendingPathComponent("checkpoint-readout.safetensors"))
                let logits = try #require(arrays["ane_logits"])
                try #require(logits.shape == [32, 98336])
                let flat = logits.asArray(Float16.self)
                try #require(flat.allSatisfy { $0.isFinite && $0 > -65504 })
                result["scope"] = "Full coarse ANE shortlist; host packing and ID decoding, CPU sorting only for the test oracle"
                var slots = Array(repeating: [(score: Float16, id: Int)](), count: 32)
                var timings: [Double] = []
                func select(_ input: [Float16]) throws -> [Float] {
                    let began = Date()
                    let values = try selector(input).map(Float.init)
                    timings.append(Date().timeIntervalSince(began))
                    try #require(values.count == 32 * 64 && values.allSatisfy { $0.isFinite })
                    return values
                }
                result["phase"] = "tile-selection"; try save()
                for begin in stride(from: 0, to: 98330, by: 2048) {
                    var input = [Float16](repeating: -65504, count: 32 * 2048)
                    let count = min(2048, 98330 - begin)
                    for row in 0..<32 {
                        for index in 0..<count { input[row * 2048 + index] = flat[row * 98336 + begin + index] }
                    }
                    let selected = try select(input)
                    for row in 0..<32 {
                        for rank in 0..<32 {
                            let rawID = selected[row * 64 + 32 + rank]
                            try #require(rawID >= 0 && rawID < 2048 && rawID.rounded() == rawID)
                            slots[row].append((Float16(selected[row * 64 + rank]), begin + Int(rawID)))
                        }
                    }
                    result["completed_tiles"] = timings.count; try save()
                }
                var mergeInput = [Float16](repeating: -65504, count: 32 * 2048)
                for row in 0..<32 {
                    try #require(slots[row].count <= 2048)
                    for index in slots[row].indices { mergeInput[row * 2048 + index] = slots[row][index].score }
                }
                result["phase"] = "merge-selection"; try save()
                let merged = try select(mergeInput)
                var ids: [Int32] = [], scores: [Float] = []
                var mismatches = 0
                for row in 0..<32 {
                    let offset = row * 98336
                    let expected = Array((0..<98330).sorted {
                        flat[offset + $0] == flat[offset + $1] ? $0 < $1 : flat[offset + $0] > flat[offset + $1]
                    }.prefix(32))
                    for rank in 0..<32 {
                        let rawSlot = merged[row * 64 + 32 + rank]
                        try #require(rawSlot >= 0 && rawSlot < Float(slots[row].count) && rawSlot.rounded() == rawSlot)
                        let chosen = slots[row][Int(rawSlot)]
                        try #require(chosen.id < 98330)
                        if chosen.id != expected[rank] || merged[row * 64 + rank] != Float(flat[offset + expected[rank]]) {
                            mismatches += 1
                        }
                        ids.append(Int32(chosen.id)); scores.append(merged[row * 64 + rank])
                    }
                }
                try MLX.save(arrays: ["compact_ids": MLXArray(ids, [32, 32]), "scores": MLXArray(scores, [32, 32])],
                    url: root.appendingPathComponent("selection-readout.safetensors"))
                result["dispatch_and_read_seconds"] = timings
                result["mismatched_values_or_indices"] = mismatches
                result["phase"] = "complete"; try save()
                try #require(mismatches == 0)
                return
            }
            let model = try ANEInMemoryModel(milText: text, weightBlob: Data())
            try model.compile(); result["phase"] = "load"; try save()
            try model.load(); defer { model.unload() }
            let patterns = reductions ? ["unique", "ties", "all-equal"] : ["unique"]
            var trials: [[String: Any]] = []
            for pattern in patterns {
                var input: [Float16] = [], expected: [Float] = []
                for row in 0..<32 {
                    var scores: [Int] = []
                    for candidate in 0..<2048 {
                        let unique = (candidate * 37 + row * 13) % 2048 - 1024
                        scores.append(pattern == "unique" ? unique : (pattern == "ties" ? unique / 8 : 1))
                    }
                    input += scores.map { Float16($0) }
                    let ids = (0..<2048).sorted {
                        scores[$0] == scores[$1] ? $0 < $1 : scores[$0] > scores[$1]
                    }.prefix(32)
                    expected += ids.map { Float(scores[$0]) }
                    if !valuesOnly { expected += ids.map { Float($0) } }
                }
                result["phase"] = "dispatch"; try save()
                let start = Date()
                let output = try ANEDirectDispatch.runConv(model: model, x: MLXArray(input, [32, 2048]),
                    inputDim: 2048, outputDim: outputFeatures, sequenceLength: 32)
                let actual = output.asArray(Float.self)
                let elapsed = Date().timeIntervalSince(start)
                try #require(actual.count == expected.count)
                let mismatches = zip(actual, expected).filter { $0 != $1 }.count
                trials.append(["pattern": pattern, "dispatch_and_read_seconds": elapsed,
                               "mismatched_values_or_indices": mismatches])
                result["trials"] = trials
                try save()
                try #require(mismatches == 0)
            }
            result["phase"] = "complete"; try save()
        } catch {
            result["error"] = String(describing: error); try save(); throw error
        }
    }
}
