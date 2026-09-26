import Foundation
import MLX
import Testing

@Suite(.serialized)
struct QwenGPUCompatibleMathTests {
    @Test func cacheOverwriteStoppedLanesAndSelectionTies() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_GPU_COMPATIBLE_MATH_TESTS"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            let large = MLXArray.ones([32, 5120], dtype: .float16)
            large[0, 0] = MLXArray(Float16(512))
            let normalized = QwenGPUCompatibleMath.norm(large, weight: MLXArray.ones([5120], dtype: .float16))
            #expect(all(isFinite(normalized)).item(Bool.self))
            #expect(normalized[0, 0].item(Float.self) > 60)
            #expect(normalized[1, 0].item(Float.self) == 1)
            let cache = try QwenGPUCompatibleCache(capacity: 4)
            let keys = MLXArray.zeros([32, 4, 256], dtype: .float16)
            let queries = MLXArray.zeros([32, 24, 256], dtype: .float16)
            func kv(_ value: Float16) -> MLXArray {
                concatenated([keys, MLXArray.ones([32, 4, 256], dtype: .float16) * MLXArray(value)], axis: 1)
            }
            let first = try cache.step(newKV: kv(2), queries: queries, slots: Array(repeating: 0, count: 32), validLengths: Array(repeating: 1, count: 32))
            #expect(first.asArray(Float16.self).allSatisfy { $0 == 2 })
            let second = try cache.step(newKV: kv(4), queries: queries,
                slots: (0..<32).map { $0 == 0 ? -1 : 1 }, validLengths: (0..<32).map { $0 == 0 ? 1 : 2 })
            #expect(second[0].asArray(Float16.self).allSatisfy { $0 == 2 })
            #expect(second[1].asArray(Float16.self).allSatisfy { $0 == 3 })
            let replaced = try cache.step(newKV: kv(6), queries: queries, slots: Array(repeating: 0, count: 32), validLengths: Array(repeating: 1, count: 32))
            #expect(replaced.asArray(Float16.self).allSatisfy { $0 == 6 })
            let scores = MLXArray.zeros([32, 40], dtype: .float16)
            let ids = try QwenGPUCompatibleMath.shortlist(scores)
            #expect(ids[0].asArray(Int32.self) == Array(0..<32).map(Int32.init))
            let tiedTokens = MLXArray(Array((248288..<248320).reversed()).map(Int32.init)).reshaped([1, 32])
                + MLXArray.zeros([32, 32], dtype: .int32)
            let winners = try QwenGPUCompatibleMath.winner(scores: scores[0..., 0..<32], tokenIDs: tiedTokens)
            #expect(winners.asArray(Int32.self).allSatisfy { $0 == 248288 })
            let vectors = MLXArray.ones([32, 28, 256], dtype: .float16)
            let rotated = try QwenGPUCompatibleMath.rotate(vectors, positions: Array(repeating: 0, count: 32))
            #expect(rotated.asArray(Float16.self) == vectors.asArray(Float16.self))
            let positions = (0..<32).map { [1, 2, 3, 511, 512, 1023][$0 % 6] }
            let varied = try QwenGPUCompatibleMath.rotate(vectors, positions: positions).asArray(Float16.self)
            for lane in 0..<32 {
                var c = Float16(1), s = Float16(0)
                for bit in (0..<10).reversed() where positions[lane] & (1 << bit) != 0 {
                    let bc = Float16(Foundation.cos(Double(1 << bit)))
                    let bs = Float16(Foundation.sin(Double(1 << bit)))
                    let nc = c * bc - s * bs, ns = s * bc + c * bs
                    c = nc; s = ns
                }
                #expect(abs(Float(varied[lane * 28 * 256]) - Float(c - s)) < 0.002)
                #expect(varied[lane * 28 * 256 + 64] == 1)
            }
            #expect(throws: QwenGPUCompatibleError.self) {
                try QwenGPUCompatibleMath.rotate(vectors, positions: Array(repeating: 1024, count: 32))
            }
        }
    }
}
