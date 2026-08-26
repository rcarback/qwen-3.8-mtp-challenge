import Foundation
import MLX
import MLXFastCore
import Testing

/// Prices the pieces of a Qwen 3.8 prefill step against each other.
///
/// The gated-delta recurrence was the standing suspect for the flat prefill
/// curve and `GatedDeltaScanCostTests` cleared it: 0.535 ms/token for all 48
/// layers against a whole-model 13.4. This prices the remaining candidates --
/// 4-bit quantized GEMM at the model's real projection shapes, the same GEMM
/// dequantized to bf16, and full attention -- and reports the achieved TFLOPS
/// so the gap to the hardware is visible rather than inferred.
@Suite(.serialized)
struct PrefillMatmulCostTests {
    private static func timeIt(_ body: () -> [MLXArray]) -> Double {
        eval(body())
        var best = Double.infinity
        for _ in 0 ..< 3 {
            let start = Date()
            eval(body())
            best = Swift.min(best, Date().timeIntervalSince(start))
        }
        return best
    }

    @Test("quantized versus dense GEMM at Qwen 3.8 projection shapes")
    func gemmCost() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        // (name, out, in) for one layer's projections, at hidden 5120.
        let shapes: [(String, Int, Int)] = [
            ("mlp.gate/up", 17408, 5120),
            ("mlp.down", 5120, 17408),
            ("gdn.in_qkv", 10240, 5120),
            ("attn.q", 6144, 5120),
        ]
        print("\nGEMM cost, 4-bit affine g64 vs bf16 (best of 3)")
        print("  shape            T      q4 ms   q4 TFLOPS   bf16 ms  bf16 TFLOPS  ratio")
        for (name, outF, inF) in shapes {
            for T in [256, 1024, 4096] {
                let x = MLXRandom.normal([1, T, inF]).asType(.bfloat16)
                let w = MLXRandom.normal([outF, inF]).asType(.bfloat16)
                let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4)
                eval(x, w, wq, scales, biases)
                let flops = 2.0 * Double(T) * Double(outF) * Double(inF)
                let q4 = Self.timeIt {
                    [quantizedMatmul(
                        x, wq, scales: scales, biases: biases,
                        transpose: true, groupSize: 64, bits: 4)]
                }
                let bf = Self.timeIt { [matmul(x, w.T)] }
                print(String(
                    format: "  %-14s %5d  %9.3f  %10.2f  %9.3f  %11.2f  %5.2fx",
                    (name as NSString).utf8String!, T,
                    1000 * q4, flops / q4 / 1e12,
                    1000 * bf, flops / bf / 1e12, q4 / bf))
            }
        }
        print("")
    }

    @Test("full attention cost at prefill depths")
    func attentionCost() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        // 24 query heads, 4 KV heads, head_dim 256, 16 full-attention layers.
        print("\nFull attention (24q/4kv heads, D=256), one layer")
        print("      T   seconds   ms/token   x16 layers ms/token")
        for T in [1024, 4096, 8192] {
            let q = MLXRandom.normal([1, 24, T, 256]).asType(.bfloat16)
            let k = MLXRandom.normal([1, 4, T, 256]).asType(.bfloat16)
            let v = MLXRandom.normal([1, 4, T, 256]).asType(.bfloat16)
            eval(q, k, v)
            let dt = Self.timeIt {
                [MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v,
                    scale: 1 / 16.0, mask: .causal)]
            }
            print(String(
                format: "  %5d  %8.3f  %9.4f  %18.3f",
                T, dt, 1000 * dt / Double(T), 16 * 1000 * dt / Double(T)))
        }
        print("")
    }

    @Test("gated-delta depthwise conv1d cost")
    func convCost() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        // The GDN causal conv is depthwise over the full 10240-wide qkv stream
        // with kernel 4. MLX issue #2180 measured its depthwise Metal path at
        // ~9x PyTorch MPS at groups=256; this runs at groups=10240, in 48
        // layers, so it is worth pricing directly rather than assuming.
        let width = 10240
        print("\nGDN depthwise conv1d (groups 10240, k=4), one layer")
        print("      T   seconds   ms/token   x48 layers ms/token")
        for T in [256, 1024, 4096] {
            let x = MLXRandom.normal([1, T + 3, width]).asType(.bfloat16)
            let w = MLXRandom.normal([width, 4, 1]).asType(.bfloat16)
            eval(x, w)
            let dt = Self.timeIt { [conv1d(x, w, groups: width)] }
            print(String(
                format: "  %5d  %8.3f  %9.4f  %18.3f",
                T, dt, 1000 * dt / Double(T), 48 * 1000 * dt / Double(T)))
        }
        print("")
    }

    @Test("square GEMM reference peak")
    func peakCost() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        // Reference point for the model-shape numbers above. Run in the same
        // conditions and compared as a RATIO, so GPU contention and memory
        // pressure -- which move both legs together -- cannot fake the answer.
        // A square GEMM far above the model shapes means the gap is dispatch
        // or shape; a square GEMM at the same level means the machine is.
        print("\nSquare bf16 GEMM reference")
        print("      N     ms    TFLOPS")
        for N in [1024, 2048, 4096] {
            let a = MLXRandom.normal([N, N]).asType(.bfloat16)
            let b = MLXRandom.normal([N, N]).asType(.bfloat16)
            eval(a, b)
            let dt = Self.timeIt { [matmul(a, b)] }
            let flops = 2.0 * Double(N) * Double(N) * Double(N)
            print(String(format: "  %5d  %6.2f  %8.2f",
                         N, 1000 * dt, flops / dt / 1e12))
        }
        print("")
    }
}
