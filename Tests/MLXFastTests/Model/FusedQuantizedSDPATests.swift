import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

/// The fused kernel must compute the same thing as the decomposed path it
/// replaces. These compare against `quantizedScaledDotProductAttention`
/// directly rather than against bfloat16 attention, because the quantization
/// error is shared and only the reduction order differs.
@Suite(.serialized)
struct FusedQuantizedSDPATests {
    private static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
    }

    private static func meanMagnitude(_ a: MLXArray) -> Float {
        MLX.mean(MLX.abs(a.asType(.float32))).item(Float.self)
    }

    @Test("the support predicate accepts 4 and 8 bits and refuses the rest")
    func supportPredicate() {
        // 3-bit is refused because 8 elements at 3 bits is 24 bits, so a lane
        // would straddle a uint32 boundary.
        for bits in [4, 8] {
            #expect(
                FusedQuantizedSDPA.isSupported(
                    headDim: 256, valueHeadDim: 256, queryRows: 3, bits: bits,
                    groupSize: 64, mode: .affine, hasSinks: false, hasBiases: true))
        }
        for bits in [2, 3, 5, 6] {
            #expect(
                !FusedQuantizedSDPA.isSupported(
                    headDim: 256, valueHeadDim: 256, queryRows: 3, bits: bits,
                    groupSize: 64, mode: .affine, hasSinks: false, hasBiases: true))
        }
        // A head dimension that is not a multiple of 32 cannot be split across
        // 32 lanes.
        #expect(
            !FusedQuantizedSDPA.isSupported(
                headDim: 100, valueHeadDim: 100, queryRows: 1, bits: 4,
                groupSize: 64, mode: .affine, hasSinks: false, hasBiases: true))
        // Prefill widths are out of scope: this is a decode kernel.
        #expect(
            !FusedQuantizedSDPA.isSupported(
                headDim: 256, valueHeadDim: 256, queryRows: 64, bits: 4,
                groupSize: 64, mode: .affine, hasSinks: false, hasBiases: true))
        // Attention sinks are not implemented.
        #expect(
            !FusedQuantizedSDPA.isSupported(
                headDim: 256, valueHeadDim: 256, queryRows: 1, bits: 4,
                groupSize: 64, mode: .affine, hasSinks: true, hasBiases: true))
        // Affine biases are required.
        #expect(
            !FusedQuantizedSDPA.isSupported(
                headDim: 256, valueHeadDim: 256, queryRows: 1, bits: 4,
                groupSize: 64, mode: .affine, hasSinks: false, hasBiases: false))
    }

    @Test("fused output matches the decomposed path at 4 and 8 bits")
    func parity() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(0x5157_454E)

        let (b, qHeads, kvHeads, dim, group) = (1, 24, 4, 256, 64)
        let scale = 1.0 / Float(dim).squareRoot()

        for bits in [4, 8] {
            for queryRows in [1, 2, 3] {
                for keyCount in [37, 512, 4096] {
                    let q = MLXRandom.normal([b, qHeads, queryRows, dim]).asType(.bfloat16)
                    let k = MLXRandom.normal([b, kvHeads, keyCount, dim]).asType(.bfloat16)
                    let v = MLXRandom.normal([b, kvHeads, keyCount, dim]).asType(.bfloat16)
                    let qk = MLX.quantized(k, groupSize: group, bits: bits)
                    let qv = MLX.quantized(v, groupSize: group, bits: bits)

                    let reference = quantizedScaledDotProductAttention(
                        queries: q,
                        quantizedKeys: (qk.wq, qk.scales, qk.biases),
                        quantizedValues: (qv.wq, qv.scales, qv.biases),
                        scale: scale, mask: .causal,
                        groupSize: group, bits: bits, mode: .affine)

                    let fused = FusedQuantizedSDPA.attention(
                        queries: q,
                        quantizedKeys: (qk.wq, qk.scales, qk.biases),
                        quantizedValues: (qv.wq, qv.scales, qv.biases),
                        scale: scale, causal: true, groupSize: group, bits: bits)

                    #expect(fused.shape == reference.shape)
                    let diff = Self.maxAbsDifference(fused, reference)
                    let magnitude = Self.meanMagnitude(reference)
                    // Both paths carry identical quantization error, so only the
                    // reduction order differs. bfloat16 inputs with float32
                    // accumulation put this well below one percent of magnitude.
                    #expect(diff < max(magnitude * 0.05, 1e-3),
                        "bits=\(bits) rows=\(queryRows) keys=\(keyCount) diff=\(diff) magnitude=\(magnitude)")
                }
            }
        }
    }

    @Test("fused output is correct on non-contiguous cache slices")
    func nonContiguousInputs() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(11)

        // A real QuantizedKVCache hands back `[..., ..<offset, ...]` slices of
        // an over-allocated buffer. This drives the kernel through the cache
        // itself so the stride indexing is exercised the way production does.
        let (b, qHeads, kvHeads, dim, group, bits) = (1, 24, 4, 256, 64, 4)
        let scale = 1.0 / Float(dim).squareRoot()
        let cache = QuantizedKVCache(groupSize: group, bits: bits)

        // Two writes, so `offset` is not a multiple of the allocation step and
        // the returned slice is a genuine sub-range.
        for rows in [300, 37] {
            let k = MLXRandom.normal([b, kvHeads, rows, dim]).asType(.bfloat16)
            let v = MLXRandom.normal([b, kvHeads, rows, dim]).asType(.bfloat16)
            _ = cache.updateQuantized(keys: k, values: v)
        }
        let k = MLXRandom.normal([b, kvHeads, 1, dim]).asType(.bfloat16)
        let v = MLXRandom.normal([b, kvHeads, 1, dim]).asType(.bfloat16)
        let (qk, qv) = cache.updateQuantized(keys: k, values: v)

        let q = MLXRandom.normal([b, qHeads, 1, dim]).asType(.bfloat16)
        let reference = quantizedScaledDotProductAttention(
            queries: q, quantizedKeys: qk, quantizedValues: qv,
            scale: scale, mask: .causal, groupSize: group, bits: bits, mode: .affine)
        let fused = FusedQuantizedSDPA.attention(
            queries: q, quantizedKeys: qk, quantizedValues: qv,
            scale: scale, causal: true, groupSize: group, bits: bits)

        let diff = Self.maxAbsDifference(fused, reference)
        let magnitude = Self.meanMagnitude(reference)
        #expect(diff < max(magnitude * 0.05, 1e-3),
            "diff=\(diff) magnitude=\(magnitude)")
    }
}
