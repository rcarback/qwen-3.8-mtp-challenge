import CryptoKit
import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANEReservedReplayTests {
    private struct Case {
        let id: String
        let domain: String
        let fold: Int
        let tokens: [[Int]]
        let hidden: [[Float16]]
        let native: [[String: Any]]
    }

    @Test func fullReservedReplay() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_RESERVED_REPLAY"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_CALIBRATION_ROOT"]))
        let experimentURL = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_CONFIRMATION_MANIFEST"]))
        let candidateID = try #require(env["MLXFAST_ANE_CONFIRMATION_CANDIDATE"])
        try #require(["original", "joint-030"].contains(candidateID))
        let headID = "0"
        let output = experimentURL.deletingLastPathComponent().appendingPathComponent("run-\(candidateID)")
        try #require(!FileManager.default.fileExists(atPath: output.path))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        func hash(_ url: URL) throws -> String {
            SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
        }
        func json(_ url: URL) throws -> [String: Any] {
            try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        }
        let manifestURL = root.appendingPathComponent("manifest.json")
        let oracleURL = root.appendingPathComponent("native-oracles.json")
        let manifest = try json(manifestURL), oracle = try json(oracleURL)
        let manifestHash = try hash(manifestURL)
        try #require(oracle["calibration_manifest_sha256"] as? String == manifestHash)
        let controls = try #require(oracle["heads"] as? [[String: Any]])
        let control = try #require(controls.first { String($0["index"] as? Int ?? Int.min) == headID })
        let baseURL = URL(fileURLWithPath: try #require(control["head_path"] as? String)).appendingPathComponent("model.safetensors")
        let baseHash = try hash(baseURL)
        try #require(control["head_sha256"] as? String == baseHash)
        let experiment = try json(experimentURL)
        try #require(experiment["selected"] as? String == "original")
        let candidates = try #require(experiment["candidates"] as? [[String: Any]])
        let candidate = try #require(candidates.first { $0["id"] as? String == candidateID })
        let headURL = URL(fileURLWithPath: try #require(candidate["head_path"] as? String))
        let headHash = try hash(headURL)
        try #require(candidate["head_sha256"] as? String == headHash)
        try #require(manifest["split"] as? String == "test")
        let expected = try #require(manifest["expected_opportunities"] as? Int)
        try #require(expected > 0)
        let entries = try #require(manifest["records"] as? [[String: Any]])
        let nativeCases = try #require(oracle["records"] as? [[String: Any]])
        try #require(entries.count == 40 && nativeCases.count == 40)
        var report: [String: Any] = ["phase": "initialize", "head": headID, "expected_opportunities": expected, "candidate_id": candidateID, "engine": "ane-direct", "experiment_sha256": try hash(experimentURL),
            "manifest_sha256": try hash(manifestURL), "oracles_sha256": try hash(oracleURL),
            "head_sha256": try hash(headURL), "scope": "40 test cases, full saved windows, frozen selection unchanged; independent per-lane caches"]
        let predictionURL = output.appendingPathComponent("predictions.jsonl")
        try Data().write(to: predictionURL)
        let stream = try FileHandle(forWritingTo: predictionURL)
        defer { try? stream.close() }
        var written = 0
        var predictionHash = SHA256()
        func append(_ row: [String: Any]) throws {
            var data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
            data.append(10)
            try stream.write(contentsOf: data)
            predictionHash.update(data: data)
            written += 1
        }
        func save() throws {
            report["written_opportunities"] = written
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: output.appendingPathComponent("result.json"), options: .atomic)
        }
        try save()
        do {
            let model = try QwenANECalibrationHead(headURL: headURL,
                targetURL: URL(fileURLWithPath: try #require(env["MLXFAST_ANE_TARGET_WEIGHTS"]))) { phase, count in
                report["phase"] = phase; report["prepared_tiles"] = count; try save()
            }
            for begin in stride(from: 0, to: entries.count, by: 32) {
                report["phase"] = "load-cases"; report["batch_begin"] = begin; try save()
                var cases: [Case] = []
                for entry in entries[begin..<min(begin + 32, entries.count)] {
                    let id = try #require(entry["id"] as? String)
                    let capture = URL(fileURLWithPath: try #require(entry["capture"] as? String))
                    let metadataURL = capture.appendingPathComponent("frames.json"), tensorsURL = capture.appendingPathComponent("frames.safetensors")
                    let metadataHash = try hash(metadataURL), tensorsHash = try hash(tensorsURL)
                    try #require(entry["frames_sha256"] as? String == metadataHash)
                    try #require(entry["tensors_sha256"] as? String == tensorsHash)
                    let frames = try #require(json(metadataURL)["frames"] as? [[String: Any]])
                    let tensors = try MLX.loadArrays(url: tensorsURL)
                    let nativeCase = try #require(nativeCases.first { $0["id"] as? String == id })
                    let nativeHeads = try #require(nativeCase["heads"] as? [String: Any])
                    let native = try #require(nativeHeads[headID] as? [[String: Any]])
                    try #require(entry["split"] as? String == "test")
                    let opportunities = try #require(entry["opportunities"] as? Int)
                    try #require(opportunities > 0 && frames.count == opportunities + 1 && native.count == opportunities + 1)
                    var tokens: [[Int]] = [], hidden: [[Float16]] = []
                    for index in 0..<opportunities {
                        let ids = try #require(frames[index]["inputTokens"] as? [Int])
                        try #require(!ids.isEmpty && (index == 0 || ids.count == 1))
                        let tensor = try #require(tensors[String(index)])
                        try #require(tensor.shape == [1, ids.count, 5120])
                        tokens.append(ids); hidden.append(tensor.asArray(Float16.self))
                    }
                    try #require(tokens[0].count + opportunities < 1024)
                    cases.append(Case(id: id, domain: try #require(entry["domain"] as? String),
                        fold: try #require(entry["fold"] as? Int), tokens: tokens, hidden: hidden, native: native))
                }
                let actualLanes = cases.count
                let padding = try #require(cases.first)
                while cases.count < 32 { cases.append(padding) }
                let decoder = try model.makeDecoder()
                let lengths = cases.map { $0.tokens[0].count }
                var last = [Float16](repeating: 0, count: 32 * 5120)
                for step in 0..<(try #require(lengths.max())) {
                    report["phase"] = "prime"; report["prime_step"] = step; try save()
                    let positions = lengths.map { min(step, $0 - 1) }
                    var tokens: [Int] = [], states: [Float16] = []
                    for lane in 0..<32 {
                        let row = positions[lane]
                        tokens.append(cases[lane].tokens[0][row])
                        states.append(contentsOf: cases[lane].hidden[0][(row * 5120)..<((row + 1) * 5120)])
                    }
                    let hidden = try decoder(input: model.pack(tokens: tokens, hidden: states), positions: positions,
                        slots: lengths.map { step < $0 ? step : -1 }, validLengths: positions.map { $0 + 1 })
                    for lane in 0..<32 where step == lengths[lane] - 1 {
                        let range = (lane * 5120)..<((lane + 1) * 5120)
                        last.replaceSubrange(range, with: hidden[range])
                    }
                }
                for opportunity in 0..<(try #require(cases.map { $0.tokens.count }.max())) {
                    report["phase"] = "predict"; report["opportunity"] = opportunity; try save()
                    if opportunity > 0 {
                        let frames = cases.map { min(opportunity, $0.tokens.count - 1) }
                        let positions = (0..<32).map { lengths[$0] + frames[$0] - 1 }
                        let tokens = try (0..<32).map { try #require(cases[$0].tokens[frames[$0]].last) }
                        let states = (0..<32).flatMap { cases[$0].hidden[frames[$0]].suffix(5120) }
                        let slots = (0..<32).map { opportunity < cases[$0].tokens.count ? positions[$0] : -1 }
                        last = try decoder(input: model.pack(tokens: tokens, hidden: states), positions: positions,
                            slots: slots, validLengths: positions.map { $0 + 1 })
                    }
                    let first = try model.predict(last)
                    let positions = (0..<32).map { lengths[$0] + min(opportunity, cases[$0].tokens.count - 1) }
                    let slots = (0..<32).map { opportunity < cases[$0].tokens.count ? positions[$0] : -1 }
                    let secondHidden = try decoder(input: model.pack(tokens: first.map(Int.init), hidden: last), positions: positions,
                        slots: slots, validLengths: positions.map { $0 + 1 })
                    let second = try model.predict(secondHidden)
                    for lane in 0..<actualLanes where opportunity < cases[lane].tokens.count {
                        let item = cases[lane], native = item.native[opportunity]
                        try append(["id": item.id, "domain": item.domain, "fold": item.fold,
                            "prediction_index": opportunity + 1, "ane_first": Int(first[lane]), "ane_second": Int(second[lane]),
                            "native_first": try #require(native["draft_token"] as? Int),
                            "native_second": try #require(native["second_draft_token"] as? Int),
                            "target_first": try #require(native["target_token"] as? Int),
                            "target_second": try #require(item.native[opportunity + 1]["target_token"] as? Int)])
                    }
                    try save()
                }
            }
            try #require(written == expected)
            try stream.synchronize()
            report["predictions_sha256"] = predictionHash.finalize().map { String(format: "%02x", $0) }.joined()
            report["phase"] = "complete"; try save()
        } catch {
            report["error"] = String(describing: error); try save(); throw error
        }
    }
}
