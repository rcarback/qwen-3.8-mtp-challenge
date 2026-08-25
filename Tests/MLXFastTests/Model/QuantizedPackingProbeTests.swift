import Foundation
import MLX
import MLXRandom
import Testing

/// Establishes the exact affine packing layout a fused quantized SDPA kernel
/// must decode. Verifying this against MLX's own dequantize first means a
/// later kernel mismatch is a kernel bug, not a guess about the format.
@Suite(.serialized)
struct QuantizedPackingProbeTests {
    @Test("affine 4-bit packing layout is little-endian nibbles within uint32")
    func packingLayout() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(7)
        let groupSize = 64
        let bits = 4
        let w = MLXRandom.normal([2, 256]).asType(.float32)
        let q = MLX.quantized(w, groupSize: groupSize, bits: bits)

        print("PACK w=\(w.shape) wq=\(q.wq.shape) dtype=\(q.wq.dtype) "
            + "scales=\(q.scales.shape) biases=\(String(describing: q.biases?.shape))")

        let reference = MLX.dequantized(
            q.wq, scales: q.scales, biases: q.biases,
            groupSize: groupSize, bits: bits)

        // Hypothesis: wq is uint32, 32/bits values per word, value j of a word
        // occupying bits [j*bits, (j+1)*bits), and dequant is code*scale+bias
        // with scale/bias indexed by (element index / groupSize).
        let perWord = 32 / bits
        let words = q.wq.asArray(UInt32.self)
        let scales = q.scales.asArray(Float.self)
        let biases = (q.biases ?? MLXArray.zeros(like: q.scales)).asArray(Float.self)
        let cols = w.dim(1)
        let wordsPerRow = cols / perWord
        let groupsPerRow = cols / groupSize

        var manual = [Float](repeating: 0, count: w.size)
        for row in 0 ..< w.dim(0) {
            for col in 0 ..< cols {
                let word = words[row * wordsPerRow + col / perWord]
                let shift = (col % perWord) * bits
                let code = Float((word >> UInt32(shift)) & UInt32((1 << bits) - 1))
                let g = row * groupsPerRow + col / groupSize
                manual[row * cols + col] = code * scales[g] + biases[g]
            }
        }
        let manualArray = MLXArray(manual, [w.dim(0), cols])
        let maxDiff = MLX.max(MLX.abs(manualArray - reference)).item(Float.self)
        print("PACK manual-vs-mlx maxdiff=\(maxDiff)")
        #expect(maxDiff < 1e-5)
    }
}
