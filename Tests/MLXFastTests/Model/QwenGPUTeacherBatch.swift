import CryptoKit
import Foundation
import MLX

enum QwenGPUTeacherBatchError: Error {
    case invalidBatch
    case outputAlreadyExists
}

enum QwenGPUTeacherBatch {
    struct Loaded {
        let history: MLXArray
        private let features: [[Float16]]
        private let opportunities: [Int]

        fileprivate init(history: MLXArray, features: [[Float16]], opportunities: [Int]) {
            self.history = history
            self.features = features
            self.opportunities = opportunities
        }

        func firstInputs(at index: Int) throws -> [Float16] {
            guard index >= 0, opportunities.contains(where: { index < $0 }) else {
                throw QwenGPUTeacherBatchError.invalidBatch
            }
            return (0..<32).flatMap { lane in
                let start = min(index, opportunities[lane] - 1) * 5120
                return features[lane][start..<(start + 5120)]
            }
        }
    }

    static func read(receipt: [String: Any], ids: [String], seeds: [Int], opportunities: [Int],
                     directory: URL, capacity: Int = 1024) throws -> Loaded {
        let count = ids.count
        guard (1...32).contains(count), Set(ids).count == count, ids.allSatisfy({ !$0.isEmpty }),
              seeds.count == count, opportunities.count == count, (1...1024).contains(capacity),
              let file = receipt["file"] as? String, file.hasSuffix(".safetensors"),
              !file.contains("/"), !file.contains("\\"),
              let expectedHash = receipt["sha256"] as? String,
              let records = receipt["records"] as? [[String: Any]], records.count == count else {
            throw QwenGPUTeacherBatchError.invalidBatch
        }
        let url = directory.appendingPathComponent(file)
        let actualHash = SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
        guard actualHash == expectedHash else { throw QwenGPUTeacherBatchError.invalidBatch }
        let arrays = try MLX.loadArrays(url: url)
        let keys = (0..<count).flatMap { ["features-\($0)", "teacher-\($0)"] }
        guard Set(arrays.keys) == Set(keys) else { throw QwenGPUTeacherBatchError.invalidBatch }
        var features: [[Float16]] = [], histories: [MLXArray] = []
        var counts = opportunities
        for lane in 0..<count {
            let seed = seeds[lane], rows = opportunities[lane]
            guard (1...capacity).contains(seed), (1...capacity).contains(rows), seed + rows - 1 <= capacity else {
                throw QwenGPUTeacherBatchError.invalidBatch
            }
            let teacherRows = seed + rows - 1
            let record = records[lane]
            guard Set(record.keys) == Set(["id", "seed_tokens", "opportunities", "teacher_rows", "features_key", "history_key"]),
                  record["id"] as? String == ids[lane], record["seed_tokens"] as? Int == seed,
                  record["opportunities"] as? Int == rows, record["teacher_rows"] as? Int == teacherRows,
                  record["features_key"] as? String == "features-\(lane)", record["history_key"] as? String == "teacher-\(lane)",
                  let input = arrays["features-\(lane)"], let history = arrays["teacher-\(lane)"],
                  input.dtype == .float16, input.shape == [rows, 5120], history.dtype == .float16,
                  history.shape == [8, teacherRows, 256], all(isFinite(input)).item(Bool.self),
                  all(isFinite(history)).item(Bool.self) else {
                throw QwenGPUTeacherBatchError.invalidBatch
            }
            features.append(input.asArray(Float16.self))
            histories.append(teacherRows == capacity ? history : concatenated([
                history, MLXArray.zeros([8, capacity - teacherRows, 256], dtype: .float16),
            ], axis: 1))
        }
        while features.count < 32 {
            features.append(features[0]); histories.append(histories[0]); counts.append(counts[0])
        }
        return Loaded(history: stacked(histories, axis: 0), features: features, opportunities: counts)
    }

    static func write(ids: [String], seeds: [Int], opportunities: [Int],
                      firstInputs: [[Float16]], history: MLXArray, url: URL) throws -> [String: Any] {
        guard !FileManager.default.fileExists(atPath: url.path) else {
            throw QwenGPUTeacherBatchError.outputAlreadyExists
        }
        let count = ids.count
        guard (1...32).contains(count), Set(ids).count == count, ids.allSatisfy({ !$0.isEmpty }),
              seeds.count == count, opportunities.count == count, firstInputs.count == count,
              history.dtype == .float16, history.ndim == 4, history.dim(0) == 32,
              history.dim(1) == 8, history.dim(3) == 256, (1...1024).contains(history.dim(2)),
              all(isFinite(history)).item(Bool.self) else {
            throw QwenGPUTeacherBatchError.invalidBatch
        }
        var arrays: [String: MLXArray] = [:]
        var records: [[String: Any]] = []
        for lane in 0..<count {
            let seed = seeds[lane], rows = opportunities[lane], capacity = history.dim(2)
            guard (1...capacity).contains(seed), (1...capacity).contains(rows),
                  seed + rows - 1 <= capacity, firstInputs[lane].count == rows * 5120,
                  firstInputs[lane].allSatisfy({ $0.isFinite }) else {
                throw QwenGPUTeacherBatchError.invalidBatch
            }
            let teacherRows = seed + rows - 1
            let featuresKey = "features-\(lane)", historyKey = "teacher-\(lane)"
            arrays[featuresKey] = MLXArray(firstInputs[lane], [rows, 5120])
            arrays[historyKey] = history[lane, 0..., 0..<teacherRows, 0...]
            records.append(["id": ids[lane], "seed_tokens": seed, "opportunities": rows,
                "teacher_rows": teacherRows, "features_key": featuresKey, "history_key": historyKey])
        }
        try MLX.save(arrays: arrays, url: url)
        let digest = SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
        return ["file": url.lastPathComponent, "sha256": digest, "records": records]
    }
}
