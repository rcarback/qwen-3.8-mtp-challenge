import CryptoKit
import Foundation
import MLX
import Testing

@Suite(.serialized)
struct QwenGPUTeacherLoadTests {
    @Test func loadsExactFeaturesAndPadsOnlyOutsideTeacherHistory() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_GPU_COMPATIBLE_MATH_TESTS"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".local/status-work/tmp/teacher-load-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let first = [Float16](repeating: -0.0, count: 5120) + [Float16](repeating: 2, count: 5120)
            let second = [Float16](repeating: 3, count: 5120)
            let history = MLXArray.ones([32, 8, 4, 256], dtype: .float16)
            let receipt = try QwenGPUTeacherBatch.write(ids: ["long", "short"], seeds: [2, 1],
                opportunities: [2, 1], firstInputs: [first, second], history: history,
                url: directory.appendingPathComponent("batch.safetensors"))
            let loaded = try QwenGPUTeacherBatch.read(receipt: receipt, ids: ["long", "short"], seeds: [2, 1],
                opportunities: [2, 1], directory: directory, capacity: 3)
            #expect(loaded.history.shape == [32, 8, 3, 256])
            #expect(loaded.history[0].asArray(Float16.self).allSatisfy { $0 == 1 })
            #expect(loaded.history[1, 0..., 0, 0...].asArray(Float16.self).allSatisfy { $0 == 1 })
            #expect(loaded.history[1, 0..., 1..<3, 0...].asArray(Float16.self).allSatisfy { $0 == 0 })
            let row0 = try loaded.firstInputs(at: 0)
            #expect(row0.count == 32 * 5120)
            #expect(row0.prefix(5120).map(\.bitPattern) == first.prefix(5120).map(\.bitPattern))
            let row1 = try loaded.firstInputs(at: 1)
            #expect(row1.prefix(5120).allSatisfy { $0 == 2 })
            #expect(row1[5120..<10240].allSatisfy { $0 == 3 })
            #expect(row1.suffix(5120).allSatisfy { $0 == 2 })
            for index in [-1, 2] {
                #expect(throws: QwenGPUTeacherBatchError.self) { try loaded.firstInputs(at: index) }
            }
            let cache = try QwenGPUCompatibleCache(capacity: 3)
            try cache.restoreTeacherPrefix(loaded.history, lengths: (0..<32).map { $0 == 1 ? 1 : 2 })
            #expect(cache.state[0, 0..., 2, 0...].asArray(Float16.self).allSatisfy { $0 == 0 })
        }
    }

    @Test func rejectsStaleFilesWrongCasesAndInvalidTensorPayloads() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_GPU_COMPATIBLE_MATH_TESTS"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".local/status-work/tmp/teacher-load-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let url = directory.appendingPathComponent("batch.safetensors")
            let receipt = try QwenGPUTeacherBatch.write(ids: ["case"], seeds: [1], opportunities: [1],
                firstInputs: [[Float16](repeating: 1, count: 5120)],
                history: MLXArray.ones([32, 8, 2, 256], dtype: .float16), url: url)
            var wrongHash = receipt
            wrongHash["sha256"] = String(repeating: "0", count: 64)
            var unsafe = receipt
            unsafe["file"] = "../batch.safetensors"
            for bad in [wrongHash, unsafe] {
                #expect(throws: QwenGPUTeacherBatchError.self) {
                    try QwenGPUTeacherBatch.read(receipt: bad, ids: ["case"], seeds: [1], opportunities: [1], directory: directory)
                }
            }
            #expect(throws: QwenGPUTeacherBatchError.self) {
                try QwenGPUTeacherBatch.read(receipt: receipt, ids: ["different"], seeds: [1], opportunities: [1], directory: directory)
            }
            let valid = try MLX.loadArrays(url: url)
            let invalidFeatures = [MLXArray.ones([1, 5120], dtype: .float32),
                                   MLXArray.ones([1, 5119], dtype: .float16),
                                   MLXArray([Float16](repeating: .nan, count: 5120), [1, 5120])]
            for (index, badFeature) in invalidFeatures.enumerated() {
                var badArrays = valid
                badArrays["features-0"] = badFeature
                let badURL = directory.appendingPathComponent("invalid-\(index).safetensors")
                try MLX.save(arrays: badArrays, url: badURL)
                var updated = receipt
                updated["file"] = badURL.lastPathComponent
                updated["sha256"] = SHA256.hash(data: try Data(contentsOf: badURL)).map { String(format: "%02x", $0) }.joined()
                #expect(throws: QwenGPUTeacherBatchError.self) {
                    try QwenGPUTeacherBatch.read(receipt: updated, ids: ["case"], seeds: [1], opportunities: [1], directory: directory)
                }
            }
        }
    }
}
