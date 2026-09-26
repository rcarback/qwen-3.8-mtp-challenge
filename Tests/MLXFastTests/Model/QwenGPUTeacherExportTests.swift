import CryptoKit
import Foundation
import MLX
import Testing

@Suite(.serialized)
struct QwenGPUTeacherExportTests {
    @Test func preservesFeatureBitsAndExcludesSpeculativeHistory() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_GPU_COMPATIBLE_MATH_TESTS"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".local/status-work/tmp/teacher-export-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let url = directory.appendingPathComponent("batch.safetensors")
            let values: [Float16] = (0..<10240).map { (index: Int) -> Float16 in
                if index % 7 == 0 { return Float16(bitPattern: 0x8000) }
                return Float16(index % 31) / Float16(32)
            }
            let history = MLXArray.ones([32, 8, 4, 256], dtype: .float16)
            history[0, 0..., 3, 0...] = MLXArray(Float16(99))
            history[1, 0..., 2..<4, 0...] = MLXArray(Float16(88))
            let receipt = try QwenGPUTeacherBatch.write(ids: ["long", "short"], seeds: [2, 1],
                opportunities: [2, 2], firstInputs: [values, values], history: history, url: url)
            let records = try #require(receipt["records"] as? [[String: Any]])
            #expect(records.count == 2)
            #expect(records[0]["teacher_rows"] as? Int == 3)
            #expect(records[1]["teacher_rows"] as? Int == 2)
            #expect(records[0]["id"] as? String == "long")
            let arrays = try MLX.loadArrays(url: url)
            #expect(Set(arrays.keys) == Set(["features-0", "features-1", "teacher-0", "teacher-1"]))
            for lane in 0..<2 {
                let features = try #require(arrays["features-\(lane)"])
                #expect(features.dtype == .float16 && features.shape == [2, 5120])
                #expect(features.asArray(Float16.self).map(\.bitPattern) == values.map(\.bitPattern))
                let kv = try #require(arrays["teacher-\(lane)"])
                #expect(kv.dtype == .float16 && kv.shape == [8, 3 - lane, 256])
                #expect(kv.asArray(Float16.self).allSatisfy { $0 == 1 })
            }
            let data = try Data(contentsOf: url)
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            #expect(receipt["sha256"] as? String == digest)
            #expect(throws: QwenGPUTeacherBatchError.self) {
                try QwenGPUTeacherBatch.write(ids: ["other"], seeds: [1], opportunities: [2],
                    firstInputs: [values], history: history, url: url)
            }
            #expect(try Data(contentsOf: url) == data)
        }
    }

    @Test func invalidBatchDoesNotCreateAnArtifact() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_GPU_COMPATIBLE_MATH_TESTS"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".local/status-work/tmp/teacher-export-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let url = directory.appendingPathComponent("invalid.safetensors")
            let history = MLXArray.zeros([32, 8, 4, 256], dtype: .float16)
            let values = [Float16](repeating: 1, count: 5120)
            for seeds in [[0], [5], []] {
                #expect(throws: QwenGPUTeacherBatchError.self) {
                    try QwenGPUTeacherBatch.write(ids: ["case"], seeds: seeds, opportunities: [1],
                        firstInputs: [values], history: history, url: url)
                }
            }
            for features in [[], [Float16](repeating: .nan, count: 5120)] {
                #expect(throws: QwenGPUTeacherBatchError.self) {
                    try QwenGPUTeacherBatch.write(ids: ["case"], seeds: [1], opportunities: [1],
                        firstInputs: [features], history: history, url: url)
                }
            }
            #expect(throws: QwenGPUTeacherBatchError.self) {
                try QwenGPUTeacherBatch.write(ids: ["same", "same"], seeds: [1, 1], opportunities: [1, 1],
                    firstInputs: [values, values], history: history, url: url)
            }
            #expect(throws: QwenGPUTeacherBatchError.self) {
                try QwenGPUTeacherBatch.write(ids: ["case"], seeds: [1], opportunities: [1],
                    firstInputs: [values], history: history.asType(.float32), url: url)
            }
            #expect(!FileManager.default.fileExists(atPath: url.path))
        }
    }
}
