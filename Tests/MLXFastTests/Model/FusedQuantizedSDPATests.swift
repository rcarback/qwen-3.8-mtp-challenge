import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

/// Neither the fused kernel nor the decomposed path it replaces is precise
/// enough to serve as the other's oracle, so both are measured against
/// attention computed in float32 over the same dequantized keys and values.
/// `parity` runs in float32 to expose the kernel's own reduction error, which
/// the bfloat16 output dtype would otherwise hide; `bfloat16Parity` then
/// checks the production dtype against the path being replaced.
@Suite(.serialized)
struct FusedQuantizedSDPATests {
    private static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
    }

    /// Tolerances scale with the largest element, not the mean: the measured
    /// quantity is a maximum, and a max compared against a fraction of a mean
    /// is not a bound on anything.
    private static func maxMagnitude(_ a: MLXArray) -> Float {
        MLX.max(MLX.abs(a.asType(.float32))).item(Float.self)
    }

    /// Attention computed in float32 from the dequantized keys and values.
    /// Neither path under test is precise enough to be the other's oracle:
    /// the decomposed reference rounds its scores, its softmax and its output
    /// to bfloat16, while the fused kernel accumulates in float32. This is
    /// what both of them are approximating.
    private static func float32Golden(
        queries: MLXArray,
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        scale: Float, groupSize: Int, bits: Int
    ) -> MLXArray {
        let q = queries.asType(.float32)
        let kvHeads = keys.0.dim(1)
        let repeats = q.dim(1) / kvHeads
        func expand(_ t: (MLXArray, MLXArray, MLXArray?)) -> MLXArray {
            let d = MLX.dequantized(
                t.0, scales: t.1, biases: t.2,
                groupSize: groupSize, bits: bits, mode: .affine
            ).asType(.float32)
            return MLX.repeated(d, count: repeats, axis: 1)
        }
        let k = expand(keys)
        let v = expand(values)

        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        // `.causal` aligns the final query row with the final key, so query
        // row i may attend keys 0 through (N - L + i).
        let (rows, count) = (q.dim(2), k.dim(2))
        let qPos = MLXArray(0 ..< rows).reshaped([rows, 1]) + (count - rows)
        let kPos = MLXArray(0 ..< count).reshaped([1, count])
        scores = MLX.where(kPos .<= qPos, scores, MLXArray(-Float.infinity))
        return MLX.matmul(MLX.softmax(scores, axis: -1), v)
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

    @Test("fused output matches float32 attention at 4 and 8 bits")
    func parity() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(0x5157_454E)

        let (b, qHeads, kvHeads, dim, group) = (1, 24, 4, 256, 64)
        let scale = 1.0 / Float(dim).squareRoot()

        for bits in [4, 8] {
            for queryRows in [1, 2, 3] {
                for keyCount in [37, 512, 4096] {
                    let q = MLXRandom.normal([b, qHeads, queryRows, dim]).asType(.float32)
                    let k = MLXRandom.normal([b, kvHeads, keyCount, dim]).asType(.float32)
                    let v = MLXRandom.normal([b, kvHeads, keyCount, dim]).asType(.float32)
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

                    let golden = Self.float32Golden(
                        queries: q,
                        keys: (qk.wq, qk.scales, qk.biases),
                        values: (qv.wq, qv.scales, qv.biases),
                        scale: scale, groupSize: group, bits: bits)

                    #expect(fused.shape == reference.shape)
                    let fusedError = Self.maxAbsDifference(fused, golden)
                    let magnitude = Self.maxMagnitude(golden)
                    // Float32 in, float32 out: nothing but the kernel's own
                    // accumulation order separates it from the golden.
                    #expect(fusedError < magnitude * 1e-5,
                        "bits=\(bits) rows=\(queryRows) keys=\(keyCount) fused=\(fusedError) magnitude=\(magnitude)")
                }
            }
        }
    }

    @Test("fused output is no worse than the decomposed path in bfloat16")
    func bfloat16Parity() throws {
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
                    let golden = Self.float32Golden(
                        queries: q,
                        keys: (qk.wq, qk.scales, qk.biases),
                        values: (qv.wq, qv.scales, qv.biases),
                        scale: scale, groupSize: group, bits: bits)

                    #expect(fused.shape == reference.shape)
                    let fusedError = Self.maxAbsDifference(fused, golden)
                    let referenceError = Self.maxAbsDifference(reference, golden)
                    let magnitude = Self.maxMagnitude(golden)
                    // Both paths round their output to bfloat16, and that
                    // shared floor dominates: the bar is that the fused kernel
                    // does not lose ground, not that it wins.
                    #expect(fusedError <= referenceError * 1.25,
                        "bits=\(bits) rows=\(queryRows) keys=\(keyCount) fused=\(fusedError) reference=\(referenceError)")
                    #expect(fusedError < magnitude * 0.02,
                        "bits=\(bits) rows=\(queryRows) keys=\(keyCount) fused=\(fusedError) magnitude=\(magnitude)")
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

        let golden = Self.float32Golden(
            queries: q, keys: qk, values: qv,
            scale: scale, groupSize: group, bits: bits)
        let fusedError = Self.maxAbsDifference(fused, golden)
        let referenceError = Self.maxAbsDifference(reference, golden)
        let magnitude = Self.maxMagnitude(golden)
        #expect(fusedError <= referenceError * 1.25,
            "fused=\(fusedError) reference=\(referenceError)")
        #expect(fusedError < magnitude * 0.02,
            "fused=\(fusedError) magnitude=\(magnitude)")
    }

    @Test("attentionWithCacheUpdate routes a quantized cache through the fused kernel")
    func dispatchThroughAttentionUtils() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(23)

        let (b, qHeads, kvHeads, dim, group, bits) = (1, 24, 4, 256, 64, 4)
        let scale = 1.0 / Float(dim).squareRoot()

        // The dispatch guard is a pure function of these shape/type values, so
        // asserting `isSupported` here proves the second round's call (the
        // only round where useFused can matter -- the first round's 128 query
        // rows are prefill-width and always fall through) satisfies the guard
        // and is actually routed to the fused kernel when useFused is true,
        // not merely that both paths happen to agree.
        #expect(
            FusedQuantizedSDPA.isSupported(
                headDim: dim, valueHeadDim: dim, queryRows: 3, bits: bits,
                groupSize: group, mode: .affine, hasSinks: false, hasBiases: true))

        func attend(useFused: Bool) -> MLXArray {
            let cache = QuantizedKVCache(groupSize: group, bits: bits)
            var last = MLXArray.zeros([1])
            MLXRandom.seed(23)
            for rows in [128, 3] {
                let q = MLXRandom.normal([b, qHeads, rows, dim]).asType(.bfloat16)
                let k = MLXRandom.normal([b, kvHeads, rows, dim]).asType(.bfloat16)
                let v = MLXRandom.normal([b, kvHeads, rows, dim]).asType(.bfloat16)
                last = attentionWithCacheUpdate(
                    queries: q, keys: k, values: v, cache: cache,
                    scale: scale, mask: .causal,
                    fusedQuantizedEnabled: useFused)
            }
            return last
        }

        let fused = attend(useFused: true)
        let decomposed = attend(useFused: false)
        #expect(fused.shape == decomposed.shape)
        let diff = Self.maxAbsDifference(fused, decomposed)
        let magnitude = Self.maxMagnitude(decomposed)
        // Both paths emit bfloat16 and approximate the same computation, so the
        // shared output rounding dominates. This checks that the fused kernel is
        // routed and produces equivalent output, not that it is bit-identical;
        // the parity tests are the numerical gate.
        #expect(diff < magnitude * 0.02,
            "diff=\(diff) magnitude=\(magnitude)")
    }

    @Test("fused output matches float32 attention on non-contiguous slices")
    func nonContiguousInputsFloat32() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(11)

        // The bfloat16 output dtype puts a floor under the previous test that
        // is far above any plausible stride error. Driving the same cache in
        // float32 removes that floor, so the strided indexing is held to the
        // same 1e-5 bound the contiguous case is held to.
        let (b, qHeads, kvHeads, dim, group, bits) = (1, 24, 4, 256, 64, 4)
        let scale = 1.0 / Float(dim).squareRoot()
        let cache = QuantizedKVCache(groupSize: group, bits: bits)

        for rows in [300, 37] {
            let k = MLXRandom.normal([b, kvHeads, rows, dim]).asType(.float32)
            let v = MLXRandom.normal([b, kvHeads, rows, dim]).asType(.float32)
            _ = cache.updateQuantized(keys: k, values: v)
        }
        let k = MLXRandom.normal([b, kvHeads, 1, dim]).asType(.float32)
        let v = MLXRandom.normal([b, kvHeads, 1, dim]).asType(.float32)
        let (qk, qv) = cache.updateQuantized(keys: k, values: v)

        let q = MLXRandom.normal([b, qHeads, 1, dim]).asType(.float32)
        let fused = FusedQuantizedSDPA.attention(
            queries: q, quantizedKeys: qk, quantizedValues: qv,
            scale: scale, causal: true, groupSize: group, bits: bits)
        let golden = Self.float32Golden(
            queries: q, keys: qk, values: qv,
            scale: scale, groupSize: group, bits: bits)

        let fusedError = Self.maxAbsDifference(fused, golden)
        let magnitude = Self.maxMagnitude(golden)
        #expect(fusedError < magnitude * 1e-5,
            "fused=\(fusedError) magnitude=\(magnitude)")
    }

    @Test("fused output matches float32 attention at every supported width")
    func everySupportedQueryWidth() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(0x5157_454E)

        // `isSupported` admits widths 1 through 8, and the draft policy may
        // legally ask for any of them, so every admitted width must actually
        // dispatch. The register footprint grows with the width, so a wide
        // specialization could fail to reach the 1024 threads the kernel
        // requests, and the predicate has already promised support by then.
        let (b, qHeads, kvHeads, dim, group) = (1, 24, 4, 256, 64)
        let scale = 1.0 / Float(dim).squareRoot()

        for bits in [4, 8] {
            for queryRows in 1 ... 8 {
                #expect(
                    FusedQuantizedSDPA.isSupported(
                        headDim: dim, valueHeadDim: dim, queryRows: queryRows,
                        bits: bits, groupSize: group, mode: .affine,
                        hasSinks: false, hasBiases: true))

                let q = MLXRandom.normal([b, qHeads, queryRows, dim]).asType(.float32)
                let k = MLXRandom.normal([b, kvHeads, 512, dim]).asType(.float32)
                let v = MLXRandom.normal([b, kvHeads, 512, dim]).asType(.float32)
                let qk = MLX.quantized(k, groupSize: group, bits: bits)
                let qv = MLX.quantized(v, groupSize: group, bits: bits)

                let fused = FusedQuantizedSDPA.attention(
                    queries: q,
                    quantizedKeys: (qk.wq, qk.scales, qk.biases),
                    quantizedValues: (qv.wq, qv.scales, qv.biases),
                    scale: scale, causal: true, groupSize: group, bits: bits)
                let golden = Self.float32Golden(
                    queries: q,
                    keys: (qk.wq, qk.scales, qk.biases),
                    values: (qv.wq, qv.scales, qv.biases),
                    scale: scale, groupSize: group, bits: bits)

                let error = Self.maxAbsDifference(fused, golden)
                let magnitude = Self.maxMagnitude(golden)
                #expect(error < magnitude * 1e-5,
                    "bits=\(bits) rows=\(queryRows) error=\(error) magnitude=\(magnitude)")
            }
        }
    }
}
