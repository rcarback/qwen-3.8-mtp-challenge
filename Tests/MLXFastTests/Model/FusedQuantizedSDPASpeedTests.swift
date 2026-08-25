import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

/// Speed of the fused quantized kernel against the decomposed path it replaces
/// and against the fused bfloat16 kernel, at both supported bit widths.
///
/// The bfloat16 figure is the target, not a competitor: matching it means the
/// KV memory reduction becomes free. The decomposed figure is what ships today.
@Suite(.serialized)
struct FusedQuantizedSDPASpeedTests {
    private static func seconds(_ body: () -> MLXArray) -> Double {
        for _ in 0 ..< 3 { MLX.eval(body()) }
        let start = Date()
        for _ in 0 ..< 20 { MLX.eval(body()) }
        return Date().timeIntervalSince(start) / 20.0
    }

    @Test("fused quantized attention speed at 4 and 8 bits")
    func speed() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(0x5157_454E)

        let (b, qHeads, kvHeads, dim, group) = (1, 24, 4, 256, 64)
        let scale = 1.0 / Float(dim).squareRoot()
        // Query width 3 is the shipped MTP depth-2 decode shape.
        let queryRows = 3

        for keyCount in [4096, 16384, 32768, 65536] {
            let q = MLXRandom.normal([b, qHeads, queryRows, dim]).asType(.bfloat16)
            let k = MLXRandom.normal([b, kvHeads, keyCount, dim]).asType(.bfloat16)
            let v = MLXRandom.normal([b, kvHeads, keyCount, dim]).asType(.bfloat16)

            let bf16 = Self.seconds {
                MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: .causal)
            }

            for bits in [4, 8] {
                let qk = MLX.quantized(k, groupSize: group, bits: bits)
                let qv = MLX.quantized(v, groupSize: group, bits: bits)
                let decomposed = Self.seconds {
                    quantizedScaledDotProductAttention(
                        queries: q,
                        quantizedKeys: (qk.wq, qk.scales, qk.biases),
                        quantizedValues: (qv.wq, qv.scales, qv.biases),
                        scale: scale, mask: .causal,
                        groupSize: group, bits: bits, mode: .affine)
                }
                let fused = Self.seconds {
                    FusedQuantizedSDPA.attention(
                        queries: q,
                        quantizedKeys: (qk.wq, qk.scales, qk.biases),
                        quantizedValues: (qv.wq, qv.scales, qv.biases),
                        scale: scale, causal: true, groupSize: group, bits: bits)
                }
                // KiB per token across the 16 full-attention layers:
                // 16 layers * 2 (K and V) * 4 kv heads * 256 dims.
                let bitsPerElement = Double(bits) + 32.0 / Double(group)
                let kibPerToken = 16.0 * 2.0 * 4.0 * 256.0 * bitsPerElement / 8.0 / 1024.0
                print(
                    "FUSEDSPEED bits=\(bits) ctx=\(keyCount) "
                        + "bf16=\(String(format: "%.4f", bf16 * 1000))ms "
                        + "decomposed=\(String(format: "%.4f", decomposed * 1000))ms "
                        + "fused=\(String(format: "%.4f", fused * 1000))ms "
                        + "fused-vs-bf16=\(String(format: "%.2f", fused / bf16))x "
                        + "fused-vs-decomposed=\(String(format: "%.2f", decomposed / fused))x "
                        + "kv=\(String(format: "%.1f", kibPerToken))KiB/token")
                // The fused path must at minimum beat what ships today.
                #expect(fused < decomposed)
            }
        }
    }
}
