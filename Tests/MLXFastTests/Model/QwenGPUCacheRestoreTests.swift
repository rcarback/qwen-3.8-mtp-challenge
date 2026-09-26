import Foundation
import MLX
import Testing

@Suite(.serialized)
struct QwenGPUCacheRestoreTests {
    @Test func restoredPrefixesMatchSequentialCacheAndIsolateCandidates() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_GPU_COMPATIBLE_MATH_TESTS"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            let teacher = try QwenGPUCompatibleCache(capacity: 4)
            let replay = try QwenGPUCompatibleCache(capacity: 4)
            let queries = MLXArray.zeros([32, 24, 256], dtype: .float16)
            func kv(_ value: Float16) -> MLXArray {
                concatenated([
                    MLXArray.ones([32, 4, 256], dtype: .float16) * MLXArray(value / 16),
                    MLXArray.ones([32, 4, 256], dtype: .float16) * MLXArray(value),
                ], axis: 1)
            }
            for position in 0..<4 {
                _ = try teacher.step(newKV: kv(Float16(position + 1)), queries: queries,
                    slots: Array(repeating: position, count: 32),
                    validLengths: Array(repeating: position + 1, count: 32))
            }
            let history = teacher.state
            let original = history.asArray(Float16.self)
            let prefixLengths = (0..<32).map { $0 % 4 }
            try replay.restoreTeacherPrefix(history, lengths: prefixLengths)
            for lane in 0..<32 {
                #expect(replay.state[lane, 0..., prefixLengths[lane]..<4, 0...]
                    .asArray(Float16.self).allSatisfy { $0 == 0 })
            }
            let sequential = try QwenGPUCompatibleCache(capacity: 4)
            for position in 0..<3 {
                _ = try sequential.step(newKV: kv(Float16(position + 1)), queries: queries,
                    slots: prefixLengths.map { position < $0 ? position : -1 },
                    validLengths: prefixLengths.map { max(1, min(position + 1, $0)) })
            }
            #expect(replay.state.asArray(Float16.self) == sequential.state.asArray(Float16.self))
            let expected = try sequential.step(newKV: kv(8), queries: queries,
                slots: prefixLengths, validLengths: prefixLengths.map { $0 + 1 })
            let actual = try replay.step(newKV: kv(8), queries: queries,
                slots: prefixLengths, validLengths: prefixLengths.map { $0 + 1 })
            #expect(actual.asArray(Float16.self) == expected.asArray(Float16.self))

            // A different speculative candidate must not change the saved teacher history.
            _ = try replay.step(newKV: kv(30), queries: queries,
                slots: prefixLengths, validLengths: prefixLengths.map { $0 + 1 })
            try replay.restoreTeacherPrefix(history, lengths: prefixLengths)
            let repeated = try replay.step(newKV: kv(8), queries: queries,
                slots: prefixLengths, validLengths: prefixLengths.map { $0 + 1 })
            #expect(repeated.asArray(Float16.self) == expected.asArray(Float16.self))
            #expect(history.asArray(Float16.self) == original)

            try replay.restoreTeacherPrefix(history, lengths: Array(repeating: 4, count: 32))
            #expect(replay.state.asArray(Float16.self) == original)
        }
    }

    @Test func invalidRestoreLeavesExistingCacheUnchanged() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_GPU_COMPATIBLE_MATH_TESTS"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            let cache = try QwenGPUCompatibleCache(capacity: 4)
            let history = MLXArray.ones([32, 8, 4, 256], dtype: .float16)
            let valid = Array(repeating: 2, count: 32)
            try cache.restoreTeacherPrefix(history, lengths: valid)
            let before = cache.state.asArray(Float16.self)
            for lengths in [Array(repeating: -1, count: 32), Array(repeating: 5, count: 32), [2]] {
                #expect(throws: QwenGPUCompatibleError.self) {
                    try cache.restoreTeacherPrefix(history, lengths: lengths)
                }
            }
            for invalid in [history.asType(.float32), MLXArray.ones([32, 8, 3, 256], dtype: .float16),
                            history * MLXArray(Float16.infinity)] {
                #expect(throws: QwenGPUCompatibleError.self) {
                    try cache.restoreTeacherPrefix(invalid, lengths: valid)
                }
            }
            #expect(cache.state.asArray(Float16.self) == before)
        }
    }
}
