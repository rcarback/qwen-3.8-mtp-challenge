import Foundation
import MLX
import MLXLLM
import MLXRandom
import Testing

@Suite(.serialized)
struct Qwen35KVRotationTests {
    private func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
    }

    @Test("a non power-of-two head dimension is refused")
    func refusesBadDimension() {
        #expect(Qwen35KVRotation(headDimension: 100) == nil)
        #expect(Qwen35KVRotation(headDimension: 0) == nil)
        #expect(Qwen35KVRotation(headDimension: 256) != nil)
    }

    @Test("the sign vector is deterministic across calls and balanced")
    func signsAreDeterministic() {
        let first = Qwen35KVRotation.signValues(count: 256, seed: 12345)
        let second = Qwen35KVRotation.signValues(count: 256, seed: 12345)
        #expect(first == second)
        #expect(first.allSatisfy { $0 == 1.0 || $0 == -1.0 })
        // A balanced generator puts the count near 128. Allow generous slack:
        // this asserts the generator is not degenerate, not that it is fair.
        let positives = first.filter { $0 > 0 }.count
        #expect(positives > 96 && positives < 160)
        #expect(Qwen35KVRotation.signValues(count: 256, seed: 999) != first)
    }

    @Test("inverse undoes forward")
    func roundTrip() {
        guard let rotation = Qwen35KVRotation(headDimension: 256) else {
            Issue.record("rotation construction failed")
            return
        }
        let x = MLXRandom.normal([2, 4, 7, 256]).asType(.float32)
        let restored = rotation.inverse(rotation.forward(x))
        #expect(maxAbsDifference(x, restored) < 1e-4)
    }

    @Test("the rotation preserves inner products")
    func preservesInnerProducts() {
        guard let rotation = Qwen35KVRotation(headDimension: 256) else {
            Issue.record("rotation construction failed")
            return
        }
        let q = MLXRandom.normal([1, 4, 5, 256]).asType(.float32)
        let k = MLXRandom.normal([1, 4, 9, 256]).asType(.float32)
        let plain = MLX.matmul(q, k.transposed(0, 1, 3, 2))
        let rotated = MLX.matmul(
            rotation.forward(q), rotation.forward(k).transposed(0, 1, 3, 2))
        // Scores are order 16 for 256-dimensional unit-variance inputs, so a
        // 1e-3 absolute tolerance is roughly 1e-4 relative in float32.
        #expect(maxAbsDifference(plain, rotated) < 1e-3)
    }

    @Test("rotation flattens the per-group dynamic range of an outlier vector")
    func flattensOutliers() {
        guard let rotation = Qwen35KVRotation(headDimension: 256) else {
            Issue.record("rotation construction failed")
            return
        }
        // One channel 40x the rest: the shape that wastes an affine group's
        // scale in a real key vector.
        let raw = MLXRandom.normal([1, 1, 1, 256]).asType(.float32)
        raw[0, 0, 0, 17] = MLXArray(Float(40))
        let groupRange = { (v: MLXArray) -> Float in
            let groups = v.reshaped(4, 64)
            let spans = MLX.max(groups, axis: 1) - MLX.min(groups, axis: 1)
            return MLX.max(spans).item(Float.self)
        }
        let before = groupRange(raw)
        let after = groupRange(rotation.forward(raw))
        // The outlier is spread over 256 coordinates, so the widest group
        // span must shrink by a large factor, not a marginal one.
        #expect(after < before / 4)
    }
}
