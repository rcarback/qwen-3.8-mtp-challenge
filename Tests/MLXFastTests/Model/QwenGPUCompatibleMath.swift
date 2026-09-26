import Foundation
import MLX

enum QwenGPUCompatibleError: Error {
    case invalidInput
}

/// GPU operations corresponding to the FP16 tensors in the ANE MIL programs.
/// Kernel accumulation and compiler fusion still require measured calibration.
enum QwenGPUCompatibleMath {
    private static let angles: [(cosine: [Float16], sine: [Float16])] = (0..<10).map { bit in
        let values = (0..<32).map { Double(1 << bit) * pow(10_000_000.0, -2 * Double($0) / 64) }
        return (values.map { Float16(Foundation.cos($0)) }, values.map { Float16(Foundation.sin($0)) })
    }

    static func norm(_ input: MLXArray, weight: MLXArray) -> MLXArray {
        let x = input.asType(.float16).asType(.float32)
        let variance = mean(x * x, axis: -1, keepDims: true) + 1e-6
        return (x * rsqrt(variance) * weight.asType(.float16).asType(.float32)).asType(.float16)
    }

    static func rotate(_ input: MLXArray, positions: [Int]) throws -> MLXArray {
        guard input.shape == [32, 28, 256], positions.count == 32,
              positions.allSatisfy({ (0..<1024).contains($0) }) else {
            throw QwenGPUCompatibleError.invalidInput
        }
        var cosine = MLXArray.ones([32, 1, 32], dtype: .float16)
        var sine = MLXArray.zeros([32, 1, 32], dtype: .float16)
        let position = MLXArray(positions.map(Int32.init)).reshaped([32, 1, 1])
        for bit in (0..<10).reversed() {
            let active = ((position >> bit) & 1).asType(.float16)
            let c = active * MLXArray(angles[bit].cosine).reshaped([1, 1, 32]) + (1 - active)
            let s = active * MLXArray(angles[bit].sine).reshaped([1, 1, 32])
            let nextCosine = cosine * c - sine * s
            let nextSine = sine * c + cosine * s
            cosine = nextCosine; sine = nextSine
        }
        let x = input.asType(.float16)
        let left = x[0..., 0..., 0..<32], right = x[0..., 0..., 32..<64]
        return concatenated([left * cosine - right * sine, left * sine + right * cosine,
                             x[0..., 0..., 64..<256]], axis: -1)
    }

    static func shortlist(_ logits: MLXArray) throws -> MLXArray {
        guard logits.dtype == .float16, logits.ndim == 2, logits.dim(0) == 32, logits.dim(1) >= 32,
              all(isFinite(logits)).item(Bool.self), min(logits).item(Float.self) > -65504 else {
            throw QwenGPUCompatibleError.invalidInput
        }
        let columns = logits.dim(1)
        let ids = MLXArray(Array(0..<columns).map(Int32.init)).reshaped([1, columns])
        var scores = logits.asType(.float16)
        var selected: [MLXArray] = []
        for _ in 0..<32 {
            let best = max(scores, axis: 1, keepDims: true)
            let id = min(which(scores .== best, ids, Int32(columns)), axis: 1, keepDims: true)
            selected.append(id)
            scores = which(ids .== id, MLXArray(Float16(-65504)), scores)
        }
        return concatenated(selected, axis: 1)
    }

    static func winner(scores: MLXArray, tokenIDs: MLXArray) throws -> MLXArray {
        guard scores.dtype == .float16, scores.shape == [32, 32], tokenIDs.shape == scores.shape,
              tokenIDs.dtype == .int32, all(isFinite(scores)).item(Bool.self),
              min(tokenIDs).item(Int32.self) >= 0, max(tokenIDs).item(Int32.self) < 248320 else {
            throw QwenGPUCompatibleError.invalidInput
        }
        let rounded = scores.asType(.float16)
        return min(which(rounded .== max(rounded, axis: 1, keepDims: true), tokenIDs, Int32(248320)), axis: 1)
    }
}

final class QwenGPUCompatibleCache {
    let capacity: Int
    private(set) var state: MLXArray

    init(capacity: Int = 1024) throws {
        guard (1...1024).contains(capacity) else { throw QwenGPUCompatibleError.invalidInput }
        self.capacity = capacity
        state = MLXArray.zeros([32, 8, capacity, 256], dtype: .float16)
    }

    func restoreTeacherPrefix(_ history: MLXArray, lengths: [Int]) throws {
        guard history.dtype == .float16, history.shape == [32, 8, capacity, 256],
              lengths.count == 32, lengths.allSatisfy({ (0...capacity).contains($0) }),
              all(isFinite(history)).item(Bool.self) else {
            throw QwenGPUCompatibleError.invalidInput
        }
        let positions = MLXArray(Array(0..<capacity).map(Int32.init)).reshaped([1, 1, capacity, 1])
        let ends = MLXArray(lengths.map(Int32.init)).reshaped([32, 1, 1, 1])
        let restored = which(positions .< ends, history, MLXArray(Float16(0)))
        // MLX.eval materializes tensor operations; it does not evaluate source text.
        eval(restored)
        state = restored
    }

    func step(newKV: MLXArray, queries: MLXArray, slots: [Int], validLengths: [Int]) throws -> MLXArray {
        guard newKV.shape == [32, 8, 256], queries.shape == [32, 24, 256],
              slots.count == 32, validLengths.count == 32,
              zip(slots, validLengths).allSatisfy({ slot, length in
                  (1...capacity).contains(length) && (slot == -1 || (slot >= 0 && slot < length))
              }), all(isFinite(newKV)).item(Bool.self), all(isFinite(queries)).item(Bool.self) else {
            throw QwenGPUCompatibleError.invalidInput
        }
        let positions = MLXArray(Array(0..<capacity).map(Int32.init))
        let slotArray = MLXArray(slots.map(Int32.init)).reshaped([32, 1, 1, 1])
        let updated = which(positions.reshaped([1, 1, capacity, 1]) .== slotArray,
                            newKV.asType(.float16).reshaped([32, 8, 1, 256]), state)
        let keys = updated[0..., 0..<4, 0..., 0...]
        let values = updated[0..., 4..<8, 0..., 0...]
        let q = queries.asType(.float16).reshaped([32, 4, 6, 256])
        let dot = matmul(q, keys.transposed(0, 1, 3, 2))
        let visible = positions.reshaped([1, 1, 1, capacity]) .< MLXArray(validLengths.map(Int32.init)).reshaped([32, 1, 1, 1])
        let mask = which(visible, MLXArray(Float16(0)), MLXArray(Float16(-10000)))
        let probabilities = softmax(dot * MLXArray(Float16(0.0625)) + mask, axis: -1)
        let output = matmul(values.transposed(0, 1, 3, 2), probabilities.transposed(0, 1, 3, 2))
            .transposed(0, 1, 3, 2).reshaped([32, 6144])
        eval(updated, output)
        guard all(isFinite(output)).item(Bool.self) else { throw QwenGPUCompatibleError.invalidInput }
        state = updated
        return output
    }
}
