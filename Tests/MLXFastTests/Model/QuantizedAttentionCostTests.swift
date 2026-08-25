import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

/// Why quantized KV decode is slower than bfloat16 decode despite moving fewer
/// KV bytes. The bfloat16 path runs the fused flash-style SDPA kernel, which
/// never materializes the score matrix. The quantized path has no fused kernel
/// in vendored MLX, so it runs quantized matmul, an explicit mask select, a
/// softmax pass, and a second quantized matmul, materializing
/// `[B, heads, L, kL]` scores in between.
///
/// This measures the per-layer ratio at the real decode shape so the cost of a
/// fused quantized kernel can be weighed against its benefit.
@Suite(.serialized)
struct QuantizedAttentionCostTests {
    private static func timeSeconds(
        iterations: Int, _ body: () -> MLXArray
    ) -> Double {
        // Warm the kernels first: first-call JIT and allocation are not the
        // steady-state cost this measures.
        for _ in 0 ..< 3 { MLX.eval(body()) }
        let start = Date()
        for _ in 0 ..< iterations { MLX.eval(body()) }
        return Date().timeIntervalSince(start) / Double(iterations)
    }

    @Test("fused bf16 SDPA versus decomposed quantized SDPA at decode shape")
    func fusedVersusDecomposed() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }

        // Qwen 3.8 full-attention geometry, MTP depth 2 (3 query rows).
        let (b, qHeads, kvHeads, dim) = (1, 24, 4, 256)
        let queryRows = 3
        let scale = 1.0 / Float(dim).squareRoot()

        for contextLength in [4096, 16384, 32768, 65536, 131_072] {
            let q = MLXRandom.normal([b, qHeads, queryRows, dim]).asType(.bfloat16)
            let k = MLXRandom.normal([b, kvHeads, contextLength, dim]).asType(.bfloat16)
            let v = MLXRandom.normal([b, kvHeads, contextLength, dim]).asType(.bfloat16)

            // GQA broadcast is what the fused kernel does internally.
            let fused = Self.timeSeconds(iterations: 10) {
                MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: .causal)
            }

            let qk = MLX.quantized(k, groupSize: 64, bits: 3)
            let qv = MLX.quantized(v, groupSize: 64, bits: 3)
            let decomposed = Self.timeSeconds(iterations: 10) {
                quantizedScaledDotProductAttention(
                    queries: q,
                    quantizedKeys: (qk.wq, qk.scales, qk.biases),
                    quantizedValues: (qv.wq, qv.scales, qv.biases),
                    scale: scale, mask: .causal,
                    groupSize: 64, bits: 3, mode: .affine)
            }

            let scoreBytes = b * qHeads * queryRows * contextLength * 2
            print(
                "QATTN ctx=\(contextLength) "
                    + "fused=\(String(format: "%.4f", fused * 1000))ms "
                    + "quantized=\(String(format: "%.4f", decomposed * 1000))ms "
                    + "ratio=\(String(format: "%.2f", decomposed / fused))x "
                    + "scores=\(scoreBytes / 1024)KiB/layer")
        }
    }
}
