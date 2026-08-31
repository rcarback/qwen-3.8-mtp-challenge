import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXFastModel

/// Confirms ANEWeightPrep.dequantizeFP16 is bit-identical to MLX's own
/// `dequantized(...)` op for the same output-channel slice. The ANE offload
/// path needs a plain fp16 [F, in] weight; this must match the GPU
/// quantizedMM path exactly, so the helper must call MLX's dequant op rather
/// than hand-rolling the affine dequant math.
@Suite(.serialized)
struct ANEWeightPrepTests {
    @Test("dequantizeFP16 matches MLX's own dequantized() for a channel slice")
    func matchesReferenceDequant() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        let out = 256
        let inn = 128
        let channelStart = 64
        let channelEnd = 192

        let w = MLXRandom.normal([out, inn]).asType(.float16)
        let (wq, scales, biasesOpt) = quantized(w, groupSize: 64, bits: 4)
        let biases = biasesOpt ?? scales

        let expectedFull = dequantized(wq, scales: scales, biases: biases, groupSize: 64, bits: 4)
        let expected = expectedFull[channelStart ..< channelEnd, 0...].asType(.float16)

        let actual = ANEWeightPrep.dequantizeFP16(
            wq: wq, scales: scales, biases: biases,
            channelStart: channelStart, channelEnd: channelEnd)

        #expect(actual.shape == [channelEnd - channelStart, inn])
        #expect(actual.dtype == .float16)

        let maxAbsDiff = MLX.max(MLX.abs(actual.asType(.float32) - expected.asType(.float32))).item(Float.self)
        #expect(maxAbsDiff == 0)
    }
}
