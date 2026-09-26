import CryptoKit
import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenGPUTrainingReplayTests {
    private struct Case {
        let id: String
        let domain: String
        let fold: Int
        let tokens: [[Int]]
        let hidden: [[Float16]]
        let native: [[String: Any]]
    }

    @Test func fullTrainingReplay() throws {
        try Device.withDefaultDevice(.gpu) { try runTraining() }
    }

    private func runTraining() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_GPU_TRAINING_REPLAY"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_GPU_TRAINING_ROOT"]))
        let headID = env["MLXFAST_GPU_TRAINING_HEAD"] ?? "-1"
        func hash(_ url: URL) throws -> String {
            SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
        }
        func json(_ url: URL) throws -> [String: Any] {
            try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        }
        let manifestURL = root.appendingPathComponent("manifest.json")
        let oracleURL = root.appendingPathComponent("native-oracles.json")
        let manifest = try json(manifestURL), oracle = try json(oracleURL)
        let split = env["MLXFAST_GPU_REPLAY_SPLIT"] ?? "train"
        let expectedCases = try Self.caseCount(split: split)
        if split != "train" {
            try #require(manifest["split"] as? String == split)
            try #require(manifest["frozen_candidates_sha256"] as? String != nil)
        }
        let manifestHash = try hash(manifestURL)
        try #require(oracle["calibration_manifest_sha256"] as? String == manifestHash)
        let controls = try #require(oracle["heads"] as? [[String: Any]])
        let control = try #require(controls.first { String($0["index"] as? Int ?? Int.min) == headID })
        let headURL = URL(fileURLWithPath: try #require(control["head_path"] as? String)).appendingPathComponent("model.safetensors")
        let headHash = try hash(headURL)
        try #require(control["head_sha256"] as? String == headHash)
        let entries = try #require(manifest["records"] as? [[String: Any]])
        let nativeCases = try #require(oracle["records"] as? [[String: Any]])
        try #require(entries.count == expectedCases && nativeCases.count == expectedCases)
        let expected = try #require(manifest["expected_opportunities"] as? Int)
        try #require(expected > 0)
        let ids = try entries.map { try #require($0["id"] as? String) }
        try #require(Set(ids).count == expectedCases)
        try #require(Set(ids) == Set(try nativeCases.map { try #require($0["id"] as? String) }))
        var report: [String: Any] = ["phase": "initialize", "engine": "gpu-compatible", "normalization": "fp32-rms-fp16-output", "head": headID, "expected_opportunities": expected,
            "manifest_sha256": try hash(manifestURL), "oracles_sha256": try hash(oracleURL),
            "head_sha256": try hash(headURL), "split": split,
            "scope": "\(expectedCases) \(split) cases, full windows, paired against frozen native controls"]
        var output = root.appendingPathComponent("gpu-run-head-\(headID)")
        var normOverrideURL: URL?
        if let candidateID = env["MLXFAST_GPU_CANDIDATE_ID"] {
            try #require(!candidateID.isEmpty && candidateID.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-") }))
            let candidateManifestURL = URL(fileURLWithPath: try #require(env["MLXFAST_GPU_CANDIDATE_MANIFEST"]))
            let candidates = try json(candidateManifestURL)
            try #require(headID == "0" && candidates["native_control"] as? String == headID)
            try #require(candidates["training_manifest_sha256"] as? String == manifestHash)
            let base = try #require(candidates["base_head"] as? [String: Any])
            try #require(base["head_sha256"] as? String == headHash)
            let records = try #require(candidates["candidates"] as? [[String: Any]])
            let matches = records.filter { $0["id"] as? String == candidateID }
            try #require(matches.count == 1)
            let candidate = try #require(matches.first)
            let normURL = URL(fileURLWithPath: try #require(candidate["norm_path"] as? String))
            let normHash = try hash(normURL)
            try #require(candidate["sha256"] as? String == normHash)
            normOverrideURL = normURL
            output = candidateManifestURL.deletingLastPathComponent().appendingPathComponent("run-\(candidateID)")
            report["candidate_id"] = candidateID
            report["native_control"] = headID
            report["norm_sha256"] = normHash
            report["candidate_manifest_sha256"] = try hash(candidateManifestURL)
        } else {
            try #require(env["MLXFAST_GPU_CANDIDATE_MANIFEST"] == nil)
        }
        let exportTeacherState = env["MLXFAST_GPU_TEACHER_EXPORT"] != nil
        var teacherBatches: [[String: Any]] = []
        if let exportPath = env["MLXFAST_GPU_TEACHER_EXPORT"] {
            try #require(!exportPath.isEmpty && headID == "0" && normOverrideURL == nil)
            output = URL(fileURLWithPath: exportPath)
            report["teacher_state_export"] = true
            report["teacher_state_format"] = 1
            let target = URL(fileURLWithPath: try #require(env["MLXFAST_GPU_TARGET_WEIGHTS"]))
            report["target_weights_path"] = target.standardizedFileURL.path
            report["target_index_sha256"] = try hash(target.appendingPathComponent("model.safetensors.index.json"))
        }
        var replayDirectory: URL?
        var replayBatches: [[String: Any]] = []
        if let replayPath = env["MLXFAST_GPU_TEACHER_REPLAY"] {
            try #require(split == "train", "Cached replay is currently audited only for training")
            try #require(!replayPath.isEmpty && !exportTeacherState && headID == "0" && normOverrideURL != nil)
            let directory = URL(fileURLWithPath: replayPath)
            let exportURL = directory.appendingPathComponent("result.json")
            let auditURL = directory.appendingPathComponent("export-audit.json")
            let exported = try json(exportURL), audit = try json(auditURL)
            let exportHash = try hash(exportURL)
            let oracleHash = try hash(oracleURL)
            try #require(audit["phase"] as? String == "complete" && audit["error"] == nil)
            try #require(exported["phase"] as? String == "complete" && exported["error"] == nil)
            try #require(exported["teacher_state_export"] as? Bool == true && exported["teacher_state_format"] as? Int == 1)
            try #require(exported["head"] as? String == headID && exported["candidate_id"] == nil)
            try #require(exported["head_sha256"] as? String == headHash)
            try #require(exported["manifest_sha256"] as? String == manifestHash)
            try #require(exported["oracles_sha256"] as? String == oracleHash)
            try #require(exported["engine"] as? String == "gpu-compatible" && exported["normalization"] as? String == "fp32-rms-fp16-output")
            try #require(audit["result_sha256"] as? String == exportHash)
            let predictionsHash = try hash(directory.appendingPathComponent("predictions.jsonl"))
            try #require(audit["predictions_sha256"] as? String == predictionsHash && exported["predictions_sha256"] as? String == predictionsHash)
            try #require(audit["cases"] as? Int == 160 && audit["opportunities"] as? Int == expected && audit["changed_prediction_rows"] as? Int == 0)
            try #require(exported["expected_opportunities"] as? Int == expected && exported["written_opportunities"] as? Int == expected)
            let reference = root.appendingPathComponent("gpu-run-head-0")
            let referenceResultHash = try hash(reference.appendingPathComponent("result.json"))
            let referencePredictionsHash = try hash(reference.appendingPathComponent("predictions.jsonl"))
            try #require(audit["reference_result_sha256"] as? String == referenceResultHash)
            try #require(audit["reference_predictions_sha256"] as? String == referencePredictionsHash)
            let target = URL(fileURLWithPath: try #require(env["MLXFAST_GPU_TARGET_WEIGHTS"]))
            let targetIndexHash = try hash(target.appendingPathComponent("model.safetensors.index.json"))
            try #require(exported["target_weights_path"] as? String == target.standardizedFileURL.path)
            try #require(exported["target_index_sha256"] as? String == targetIndexHash)
            replayBatches = try #require(exported["teacher_batches"] as? [[String: Any]])
            let auditedBatches = try #require(audit["teacher_batches"] as? [[String: Any]])
            try #require(replayBatches.count == 5)
            let exportedLayout = try JSONSerialization.data(withJSONObject: replayBatches, options: [.sortedKeys])
            let auditedLayout = try JSONSerialization.data(withJSONObject: auditedBatches, options: [.sortedKeys])
            try #require(exportedLayout == auditedLayout)
            for (index, batch) in replayBatches.enumerated() {
                try #require(batch["batch_begin"] as? Int == index * 32)
                try #require(batch["file"] as? String == "teacher-batch-\(index * 32).safetensors")
            }
            replayDirectory = directory
            report["teacher_replay_mode"] = "cached"
            report["teacher_replay_directory"] = directory.standardizedFileURL.path
            report["teacher_replay_export_result_sha256"] = exportHash
            report["teacher_replay_export_audit_sha256"] = try hash(auditURL)
        }
        try #require(!FileManager.default.fileExists(atPath: output.path), "Preserve the existing run before replaying this head")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let predictionsURL = output.appendingPathComponent("predictions.jsonl")
        try Data().write(to: predictionsURL, options: .withoutOverwriting)
        let stream = try FileHandle(forWritingTo: predictionsURL)
        defer { try? stream.close() }
        var predictionHash = SHA256()
        var written = 0
        func append(_ row: [String: Any]) throws {
            var data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
            data.append(0x0a)
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
            let model = try QwenGPUCalibrationHead(headURL: headURL,
                targetURL: URL(fileURLWithPath: try #require(env["MLXFAST_GPU_TARGET_WEIGHTS"])), normOverrideURL: normOverrideURL) { phase, count in
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
                    try #require(entry["split"] as? String == split)
                    let opportunities = try #require(entry["opportunities"] as? Int)
                    try #require(opportunities > 0)
                    try #require(frames.count == opportunities + 1 && native.count == opportunities + 1)
                    var tokens: [[Int]] = [], hidden: [[Float16]] = []
                    for index in 0..<opportunities {
                        let ids = try #require(frames[index]["inputTokens"] as? [Int])
                        let tensor = try #require(tensors[String(index)])
                        try #require(!ids.isEmpty && (index == 0 || ids.count == 1))
                        try #require(tensor.shape == [1, ids.count, 5120])
                        tokens.append(ids); hidden.append(tensor.asArray(Float16.self))
                    }
                    try #require(tokens[0].count == entry["seed_tokens"] as? Int)
                    try #require(tokens[0].count + opportunities < 1024)
                    cases.append(Case(id: id, domain: try #require(entry["domain"] as? String),
                        fold: try #require(entry["fold"] as? Int), tokens: tokens, hidden: hidden, native: native))
                }
                let actualLanes = cases.count
                let padding = try #require(cases.first)
                while cases.count < 32 { cases.append(padding) }
                let decoder = try model.makeDecoder()
                let lengths = cases.map { $0.tokens[0].count }
                let replayBatch = try replayDirectory.map { directory in
                    let activeCases = Array(cases.prefix(actualLanes))
                    return try QwenGPUTeacherBatch.read(receipt: replayBatches[begin / 32], ids: activeCases.map(\.id),
                        seeds: Array(lengths.prefix(actualLanes)), opportunities: activeCases.map { $0.tokens.count }, directory: directory)
                }
                var last = [Float16](repeating: 0, count: 32 * 5120)
                var lastNormInput = [Float16](repeating: 0, count: exportTeacherState ? 32 * 5120 : 0)
                var firstInputs = [[Float16]](repeating: [], count: actualLanes)
                let seedSteps = try #require(lengths.max())
                let primeSteps = replayDirectory == nil ? seedSteps : 0
                for step in 0..<primeSteps {
                    report["phase"] = "prime"; report["prime_step"] = step; try save()
                    let positions = lengths.map { min(step, $0 - 1) }
                    var tokens: [Int] = [], states: [Float16] = []
                    for lane in 0..<32 {
                        let row = positions[lane]
                        tokens.append(cases[lane].tokens[0][row])
                        states.append(contentsOf: cases[lane].hidden[0][(row * 5120)..<((row + 1) * 5120)])
                    }
                    let hidden = try decoder(input: model.pack(tokens: tokens, hidden: states), positions: positions,
                        slots: lengths.map { step < $0 ? step : -1 }, validLengths: positions.map { $0 + 1 },
                        captureFinalNormInput: exportTeacherState)
                    let captured = exportTeacherState ? try #require(decoder.lastFinalNormInput) : []
                    for lane in 0..<32 where step == lengths[lane] - 1 {
                        let range = (lane * 5120)..<((lane + 1) * 5120)
                        last.replaceSubrange(range, with: hidden[range])
                        if exportTeacherState { lastNormInput.replaceSubrange(range, with: captured[range]) }
                    }
                }
                for opportunity in 0..<(try #require(cases.map { $0.tokens.count }.max())) {
                    report["phase"] = "predict"; report["opportunity"] = opportunity; try save()
                    if let replayBatch {
                        let inputs = try replayBatch.firstInputs(at: opportunity)
                        last = QwenGPUCompatibleMath.norm(MLXArray(inputs, [32, 5120]), weight: model.finalNorm).asArray(Float16.self)
                        let teacherLengths = (0..<32).map { lengths[$0] + min(opportunity, cases[$0].tokens.count - 1) }
                        try decoder.restoreTeacherPrefix(replayBatch.history, lengths: teacherLengths)
                    } else if opportunity > 0 {
                        let frames = cases.map { min(opportunity, $0.tokens.count - 1) }
                        let positions = (0..<32).map { lengths[$0] + frames[$0] - 1 }
                        let tokens = try (0..<32).map { try #require(cases[$0].tokens[frames[$0]].last) }
                        let states = (0..<32).flatMap { cases[$0].hidden[frames[$0]].suffix(5120) }
                        let slots = (0..<32).map { opportunity < cases[$0].tokens.count ? positions[$0] : -1 }
                        last = try decoder(input: model.pack(tokens: tokens, hidden: states), positions: positions,
                            slots: slots, validLengths: positions.map { $0 + 1 }, captureFinalNormInput: exportTeacherState)
                        if exportTeacherState { lastNormInput = try #require(decoder.lastFinalNormInput) }
                    }
                    if exportTeacherState {
                        for lane in 0..<actualLanes where opportunity < cases[lane].tokens.count {
                            firstInputs[lane].append(contentsOf: lastNormInput[(lane * 5120)..<((lane + 1) * 5120)])
                        }
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
                            "prediction_index": opportunity + 1, "gpu_first": Int(first[lane]), "gpu_second": Int(second[lane]),
                            "native_first": try #require(native["draft_token"] as? Int),
                            "native_second": try #require(native["second_draft_token"] as? Int),
                            "target_first": try #require(native["target_token"] as? Int),
                            "target_second": try #require(item.native[opportunity + 1]["target_token"] as? Int)])
                    }
                    try save()
                }
                if exportTeacherState {
                    let activeCases = Array(cases.prefix(actualLanes))
                    var batch = try QwenGPUTeacherBatch.write(ids: activeCases.map(\.id),
                        seeds: activeCases.map { $0.tokens[0].count }, opportunities: activeCases.map { $0.tokens.count },
                        firstInputs: firstInputs, history: decoder.teacherCacheState(),
                        url: output.appendingPathComponent("teacher-batch-\(begin).safetensors"))
                    batch["batch_begin"] = begin
                    teacherBatches.append(batch)
                    report["teacher_batches"] = teacherBatches
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

    static func caseCount(split: String) throws -> Int {
        switch split {
        case "train": return 160
        case "validation": return 20
        case "test": return 40
        default: throw NSError(domain: "GPUReplaySplit", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Unknown replay split: \(split)"])
        }
    }

    @Test func reservedSplitContract() throws {
        #expect(try Self.caseCount(split: "train") == 160)
        #expect(try Self.caseCount(split: "validation") == 20)
        #expect(try Self.caseCount(split: "test") == 40)
        #expect(throws: (any Error).self) { try Self.caseCount(split: "validation-typo") }
    }
}
