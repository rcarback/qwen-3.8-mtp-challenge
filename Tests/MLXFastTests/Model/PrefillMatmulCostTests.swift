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

    /// Drops the MLX allocator cache and waits for the GPU to go idle.
    ///
    /// The suite is serialized and every test shares one allocator, so a block
    /// that allocates heavily leaves the next block measuring a degraded
    /// machine. This was not hypothetical: running `gemmCost` and `peakCost`
    /// in one process read the square bf16 reference at 6.35 TFLOPS, against
    /// 14.66, 14.65 and 14.74 for the same block measured in isolation. The
    /// contamination depressed the CONTROL and left the model shapes looking
    /// healthy, which is the direction that would have closed the
    /// investigation early. Call this between measurement blocks.
    private static func quiesce() {
        Memory.clearCache()
        // A synchronous evaluation flushes anything still queued, so the next
        // timed block does not absorb the tail of the previous one.
        eval([MLXArray([1.0])])
    }

    /// Measures ONE GEMM shape and exits, so nothing precedes it in the
    /// process.
    ///
    /// Sequential in-process blocks are not comparable on this machine. The
    /// square bf16 reference reads 14.69, 14.60 and 14.66 TFLOPS when its
    /// block runs first in a fresh process, and 6.35 to 6.62 when any other
    /// measurement block ran before it. The effect survives
    /// `Memory.clearCache()` and does NOT survive a process boundary: a fresh
    /// process reads 14.60 immediately after a heavy GEMM run in a separate
    /// process, which rules out GPU clock or thermal state. Whatever the
    /// mechanism, position in the process is the dominant term, and it is
    /// larger than every effect this suite tries to measure.
    ///
    /// `tools/gemm-point-sweep.sh` drives this one point per process.
    /// Shape comes from the environment: MLXFAST_GEMM_M, _N, _K and _MODE
    /// (`q4` or `bf16`).
    @Test("one GEMM point, alone in its process")
    func singleGemmPoint() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
              let mRaw = env["MLXFAST_GEMM_M"], let M = Int(mRaw),
              let nRaw = env["MLXFAST_GEMM_N"], let N = Int(nRaw),
              let kRaw = env["MLXFAST_GEMM_K"], let K = Int(kRaw)
        else { return }
        let mode = env["MLXFAST_GEMM_MODE"] ?? "q4"
        let x = MLXRandom.normal([1, M, K]).asType(.bfloat16)
        let w = MLXRandom.normal([N, K]).asType(.bfloat16)
        let flops = 2.0 * Double(M) * Double(N) * Double(K)
        let dt: Double
        if mode == "bf16" {
            eval(x, w)
            dt = Self.timeIt { [matmul(x, w.T)] }
        } else {
            let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4)
            eval(x, wq, scales, biases ?? scales)
            dt = Self.timeIt {
                [quantizedMM(
                    x, wq, scales: scales, biases: biases,
                    transpose: true, groupSize: 64, bits: 4)]
            }
        }
        // One machine-readable line, so the driver does not parse a table.
        print(String(
            format: "GEMMPOINT\t%@\t%d\t%d\t%d\t%.4f\t%.3f",
            mode, M, N, K, 1000 * dt, flops / dt / 1e12))
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
        Self.quiesce()
        print("\nGEMM cost, 4-bit affine g64 vs bf16 (best of 3)")
        print("  shape            T      q4 ms   q4 TFLOPS   bf16 ms  bf16 TFLOPS  ratio")
        for (name, outF, inF) in shapes {
            for T in [256, 1024, 4096] {
                Self.quiesce()
                let x = MLXRandom.normal([1, T, inF]).asType(.bfloat16)
                let w = MLXRandom.normal([outF, inF]).asType(.bfloat16)
                let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4)
                eval(x, w, wq, scales, biases ?? scales)
                let flops = 2.0 * Double(T) * Double(outF) * Double(inF)
                let q4 = Self.timeIt {
                    [quantizedMM(
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

    @Test("fused attention is reached at head dim 256")
    func fusedAtHeadDim256() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        // Head dim 128 IS on the fused list and 256 is not, so the two run
        // different kernels today. Comparing them at matched FLOPs turns that
        // dispatch difference into a number: the unfused path is several times
        // slower per FLOP, and closing the gap is what this task delivers.
        //
        // The two legs are measured sequentially, so contention that drifts
        // between them (a concurrent process on the same GPU) can move the
        // ratio even though each leg is individually a best-of-3. Interleave
        // the legs instead and take the median of several complete ratio
        // samples, so drift has to persist across the whole interleaved
        // sequence to move the reported statistic.
        func makeQKV(headDim: Int, heads: Int, T: Int) -> (MLXArray, MLXArray, MLXArray) {
            let q = MLXRandom.normal([1, heads, T, headDim]).asType(.bfloat16)
            let k = MLXRandom.normal([1, 4, T, headDim]).asType(.bfloat16)
            let v = MLXRandom.normal([1, 4, T, headDim]).asType(.bfloat16)
            eval(q, k, v)
            return (q, k, v)
        }
        func attentionSeconds(
            headDim: Int, q: MLXArray, k: MLXArray, v: MLXArray
        ) -> Double {
            let start = Date()
            eval([MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v,
                scale: 1 / Float(headDim).squareRoot(), mask: .causal)])
            return Date().timeIntervalSince(start)
        }
        // 48 heads at D=128 and 24 heads at D=256 do the same total work.
        let (q128, k128, v128) = makeQKV(headDim: 128, heads: 48, T: 4096)
        let (q256, k256, v256) = makeQKV(headDim: 256, heads: 24, T: 4096)
        // Warm up both kernels once (JIT compile, allocator warmup) before
        // any timed sample.
        _ = attentionSeconds(headDim: 128, q: q128, k: k128, v: v128)
        _ = attentionSeconds(headDim: 256, q: q256, k: k256, v: v256)

        var ratios: [Double] = []
        for _ in 0 ..< 5 {
            let fused = attentionSeconds(headDim: 128, q: q128, k: k128, v: v128)
            let target = attentionSeconds(headDim: 256, q: q256, k: k256, v: v256)
            ratios.append(target / fused)
        }
        ratios.sort()
        let median = ratios[ratios.count / 2]
        print(String(
            format: "\n  ratios: %@\n  median ratio %.2fx\n",
            ratios.map { String(format: "%.2fx", $0) }.joined(separator: ", "),
            median))
        #expect(median < 1.5)
    }

    @Test("fused and unfused attention agree at head dim 256")
    func fusedMatchesUnfused() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        // Query length 8 or below takes the vector path and 9 or above takes
        // the full path (`query_sequence_length > 8` at line 631), so this
        // pair straddles the dispatch boundary the change moves.
        let T = 64
        let q = MLXRandom.normal([1, 24, T, 256]).asType(.float32)
        let k = MLXRandom.normal([1, 4, T, 256]).asType(.float32)
        let v = MLXRandom.normal([1, 4, T, 256]).asType(.float32)
        eval(q, k, v)
        let fused = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / 16.0, mask: .causal)
        // Reference: the same maths written in ops, which never dispatches the
        // fused kernel whatever the head dim.
        let kb = repeated(k, count: 6, axis: 1)
        let vb = repeated(v, count: 6, axis: 1)
        var scores = matmul(q, kb.transposed(0, 1, 3, 2)) * (1 / 16.0)
        let causal = MLXArray(0 ..< T).reshaped([T, 1])
            .< MLXArray(0 ..< T).reshaped([1, T])
        scores = MLX.where(causal, MLXArray(-Float.infinity), scores)
        let reference = matmul(softmax(scores, axis: -1), vb)
        eval(fused, reference)
        let error = abs(fused - reference).max().item(Float.self)
        print(String(format: "\n  max abs error %.3e\n", error))
        #expect(error < 1e-3)
    }

    @Test("fused bf16 attention agrees with fp32 reference at head dim 256 (production tile)")
    func fusedMatchesUnfusedProductionTile() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        // fusedMatchesUnfused above runs fp32, which at bd=256 takes the
        // narrow fallback tile (BQ=16, BK=8, WM=2) -- fp32 does not fit the
        // full tile. bf16, the dtype the model actually runs, takes the full
        // production tile (BQ=32, BK=16, WM=4) instead. That tile has no
        // numerical coverage without this test.
        let T = 64
        // Build the "true" values in fp32, then round down to bf16 for the
        // fused call -- so the fused kernel exercises real bf16 rounding
        // rather than values that happen to be exactly representable.
        let qf = MLXRandom.normal([1, 24, T, 256]).asType(.float32)
        let kf = MLXRandom.normal([1, 4, T, 256]).asType(.float32)
        let vf = MLXRandom.normal([1, 4, T, 256]).asType(.float32)
        eval(qf, kf, vf)
        let q = qf.asType(.bfloat16)
        let k = kf.asType(.bfloat16)
        let v = vf.asType(.bfloat16)
        eval(q, k, v)
        let fused = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / 16.0, mask: .causal)
        // Reference computed in fp32 from the fp32 inputs (before the bf16
        // rounding above), via ops -- never dispatches the fused kernel
        // whatever the head dim, so it carries none of the fused kernel's
        // rounding.
        let kb = repeated(kf, count: 6, axis: 1)
        let vb = repeated(vf, count: 6, axis: 1)
        var scores = matmul(qf, kb.transposed(0, 1, 3, 2)) * (1 / 16.0)
        let causal = MLXArray(0 ..< T).reshaped([T, 1])
            .< MLXArray(0 ..< T).reshaped([1, T])
        scores = MLX.where(causal, MLXArray(-Float.infinity), scores)
        let reference = matmul(softmax(scores, axis: -1), vb)
        let fusedF32 = fused.asType(.float32)
        eval(fusedF32, reference)
        let error = abs(fusedF32 - reference).max().item(Float.self)
        print(String(format: "\n  max abs error (bf16 vs fp32 reference) %.3e\n", error))
        // Bound, derived rather than fit to the measurement: bf16 has an
        // 8-bit significand, so a single rounding carries relative error
        // near 2^-8 ~ 3.9e-3. Q, K, and V are each rounded once on the way
        // in; QK^T then sums D=256 products and A@V sums T=64 products, and
        // treating those per-term roundings as uncorrelated gives error
        // growth on the order of sqrt(reduction length) -- sqrt(256) = 16
        // and sqrt(64) = 8. Output magnitude is O(1) (V ~ N(0,1), softmax
        // weights sum to 1), so the expected order of magnitude is
        // 3.9e-3 times a double-digit factor, i.e. a few 1e-2, not 1e-3
        // (fp32's bound above) and not 1e-1.
        #expect(error < 5e-2)
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
        Self.quiesce()
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

        // The bf16 reference above cannot attribute the projection gap on its
        // own. A projection is BOTH quantized and tall-thin, and those are
        // separate costs: dequantization work per output element, and a shape
        // that may under-fill the tile. Comparing a 4-bit tall-thin GEMM
        // against a bf16 square one measures their sum and names neither.
        //
        // This square 4-bit reference holds the quantization fixed and varies
        // only the shape, so the difference between it and the projection
        // figures is attributable to shape alone. The difference between it
        // and the bf16 square above is attributable to quantization alone.
        Self.quiesce()
        print("Square 4-bit affine g64 GEMM reference")
        print("      N     ms    TFLOPS")
        for N in [1024, 2048, 4096] {
            let a = MLXRandom.normal([N, N]).asType(.bfloat16)
            let b = MLXRandom.normal([N, N]).asType(.bfloat16)
            let (bq, scales, biases) = quantized(b, groupSize: 64, bits: 4)
            eval(a, bq, scales, biases ?? scales)
            let dt = Self.timeIt {
                [quantizedMM(
                    a, bq, scales: scales, biases: biases,
                    transpose: true, groupSize: 64, bits: 4)]
            }
            let flops = 2.0 * Double(N) * Double(N) * Double(N)
            print(String(format: "  %5d  %6.2f  %8.2f",
                         N, 1000 * dt, flops / dt / 1e12))
        }
        print("")

        // The production projections run at M = 256 with N = 17408, which is
        // 68 tiles of N for every 8 tiles of M at the kernel's fixed 32x32
        // tiling. This sweep holds the total work constant and varies only the
        // aspect ratio, so a tall-thin penalty shows up as a fall along the
        // row and nothing else can explain it.
        Self.quiesce()
        print("Constant-work aspect sweep, 4-bit affine g64, K = 5120")
        print("      M        N     ms    TFLOPS")
        let work = 256 * 17408
        for M in [256, 512, 1024, 2048, 4096] {
            let N = work / M
            let x = MLXRandom.normal([1, M, 5120]).asType(.bfloat16)
            let w = MLXRandom.normal([N, 5120]).asType(.bfloat16)
            let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4)
            eval(x, wq, scales, biases ?? scales)
            let dt = Self.timeIt {
                [quantizedMM(
                    x, wq, scales: scales, biases: biases,
                    transpose: true, groupSize: 64, bits: 4)]
            }
            let flops = 2.0 * Double(M) * Double(N) * 5120.0
            print(String(format: "  %5d  %7d  %6.2f  %8.2f",
                         M, N, 1000 * dt, flops / dt / 1e12))
        }
        print("")
    }
}
