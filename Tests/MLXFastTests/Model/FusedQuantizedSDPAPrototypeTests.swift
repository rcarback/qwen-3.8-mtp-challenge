import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

/// PROTOTYPE: a fused quantized decode SDPA kernel.
///
/// The shipped quantized attention path has no fused kernel. It runs a
/// quantized matmul, materializes a `[B, heads, L, N]` score matrix, applies a
/// mask, runs softmax, then runs a second quantized matmul. Measured against
/// the fused bfloat16 kernel it is 9x to 12x slower, and the gap grows with
/// context because the materialized scores grow with context.
///
/// This kernel is the quantized analogue of MLX's `sdpa_vector`: it streams
/// quantized key and value rows, dequantizes them in registers, and carries an
/// online softmax, so no score matrix is ever written to memory.
///
/// Scope is deliberately the decode shape only: small L, affine group-64,
/// 4-bit, no attention sinks, causal or no mask. It exists to answer one
/// question -- can a fused quantized kernel come within about 2x of fused
/// bfloat16 -- and not to be a production kernel.
@Suite(.serialized)
struct FusedQuantizedSDPAPrototypeTests {
    /// Each lane owns `D / 32` contiguous elements. At 4 bits that is exactly
    /// one uint32 word, and because `D / 32` divides the group size of 64,
    /// every element a lane holds shares one scale and bias. So the inner loop
    /// is one word load plus one scale lookup, with no group-boundary case.
    static func source(dim d: Int, keys n: Int, queryRows l: Int, gqa: Int, group: Int)
        -> String
    {
        let bits = 4
        let perWord = 32 / bits
        let qkPerThread = d / 32
        let wordsPerKeyRow = d / perWord
        let scalesPerKeyRow = d / group
        return """
            constexpr int BN = 32;
            constexpr int BD = 32;
            constexpr int QKPT = \(qkPerThread);
            constexpr int D = \(d);
            constexpr int N = \(n);
            constexpr int L = \(l);
            constexpr int GQA = \(gqa);
            constexpr int KWORDS = \(wordsPerKeyRow);
            constexpr int KSCALES = \(scalesPerKeyRow);
            constexpr uint MASK = \((1 << bits) - 1);
            constexpr int BITS = \(bits);

            uint lane = thread_position_in_threadgroup.x;
            uint sg   = thread_position_in_threadgroup.y;
            uint qbh  = threadgroup_position_in_grid.x;
            int kvh = int(qbh) / GQA;

            threadgroup float outputs[BN * BD];
            threadgroup float max_scores[BN];
            threadgroup float sum_exp_scores[BN];

            // All L query rows live in this threadgroup so each quantized KV
            // word is loaded and unpacked ONCE and reused across every row.
            float qv[L][QKPT];
            float o[L][QKPT];
            float max_score[L];
            float sum_exp_score[L];
            for (int r = 0; r < L; r++) {
                auto qp = q + (qbh * L + uint(r)) * D + lane * QKPT;
                for (int j = 0; j < QKPT; j++) { qv[r][j] = qp[j]; }
                for (int j = 0; j < QKPT; j++) { o[r][j] = 0.0f; }
                max_score[r] = -INFINITY;
                sum_exp_score[r] = 0.0f;
            }

            int gidx = int(lane * QKPT) / \(group);

            for (int i = int(sg); i < N; i += BN) {
                long krow = (long(kvh) * N + i);
                uint kw = kq[krow * KWORDS + lane];
                float ks = kscales[krow * KSCALES + gidx];
                float kb = kbiases[krow * KSCALES + gidx];

                // Unpack the key once into registers.
                float kd[QKPT];
                for (int j = 0; j < QKPT; j++) {
                    kd[j] = float((kw >> uint(j * BITS)) & MASK) * ks + kb;
                }

                bool any = false;
                float exps[L];
                float facs[L];
                for (int r = 0; r < L; r++) {
                    exps[r] = 0.0f;
                    facs[r] = 1.0f;
                    if (i <= (N - L + r)) {
                        any = true;
                        float score = 0.0f;
                        for (int j = 0; j < QKPT; j++) { score += qv[r][j] * kd[j]; }
                        score = simd_sum(score);
                        float new_max = max(max_score[r], score);
                        facs[r] = fast::exp(max_score[r] - new_max);
                        exps[r] = fast::exp(score - new_max);
                        max_score[r] = new_max;
                        sum_exp_score[r] = sum_exp_score[r] * facs[r] + exps[r];
                    }
                }

                if (any) {
                    uint vw = vq[krow * KWORDS + lane];
                    float vs = vscales[krow * KSCALES + gidx];
                    float vb = vbiases[krow * KSCALES + gidx];
                    float vd[QKPT];
                    for (int j = 0; j < QKPT; j++) {
                        vd[j] = float((vw >> uint(j * BITS)) & MASK) * vs + vb;
                    }
                    for (int r = 0; r < L; r++) {
                        for (int j = 0; j < QKPT; j++) {
                            o[r][j] = o[r][j] * facs[r] + exps[r] * vd[j];
                        }
                    }
                }
            }

            // Cross-simdgroup combine, once per query row.
            for (int r = 0; r < L; r++) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0) {
                    max_scores[sg] = max_score[r];
                    sum_exp_scores[sg] = sum_exp_score[r];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float ms = max_scores[lane];
                float new_max = simd_max(ms);
                float factor = fast::exp(ms - new_max);
                float total = simd_sum(sum_exp_scores[lane] * factor);

                for (int i = 0; i < QKPT; i++) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    outputs[lane * BD + sg] = o[r][i];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    float acc = simd_sum(outputs[sg * BD + lane] * factor);
                    o[r][i] = total == 0.0f ? acc : (acc / total);
                }
                if (lane == 0) {
                    auto op = out + (qbh * L + uint(r)) * D + sg * QKPT;
                    for (int i = 0; i < QKPT; i++) { op[i] = o[r][i]; }
                }
            }
            """
    }

    static func makeKernel(d: Int, n: Int, l: Int, gqa: Int, group: Int)
        -> MLXFast.MLXFastKernel
    {
        MLXFast.metalKernel(
            name: "fused_quantized_sdpa_d\(d)_n\(n)_l\(l)",
            inputNames: ["q", "kq", "kscales", "kbiases", "vq", "vscales", "vbiases"],
            outputNames: ["out"],
            source: source(dim: d, keys: n, queryRows: l, gqa: gqa, group: group),
            header: "#include <metal_simdgroup>\n#include <metal_math>\n")
    }

    @Test("fused quantized SDPA matches the decomposed path and is faster")
    func prototype() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(0x5157_454E)

        let (b, qHeads, kvHeads, d, group) = (1, 24, 4, 256, 64)
        let gqa = qHeads / kvHeads
        let scale = 1.0 / Float(d).squareRoot()

        for (l, n) in [(1, 16384), (3, 4096), (3, 16384), (3, 32768), (3, 65536)] {
            let q = MLXRandom.normal([b, qHeads, l, d]).asType(.float32)
            let k = MLXRandom.normal([b, kvHeads, n, d]).asType(.float32)
            let v = MLXRandom.normal([b, kvHeads, n, d]).asType(.float32)

            let qk = MLX.quantized(k, groupSize: group, bits: 4)
            let qv = MLX.quantized(v, groupSize: group, bits: 4)
            let zeros = MLXArray.zeros(like: qk.scales)

            let reference = quantizedScaledDotProductAttention(
                queries: q,
                quantizedKeys: (qk.wq, qk.scales, qk.biases ?? zeros),
                quantizedValues: (qv.wq, qv.scales, qv.biases ?? zeros),
                scale: scale, mask: .causal,
                groupSize: group, bits: 4, mode: .affine)

            let kernel = Self.makeKernel(d: d, n: n, l: l, gqa: gqa, group: group)
            let qScaled = (q * scale).reshaped([b * qHeads, l, d])
            func run() -> MLXArray {
                kernel(
                    [qScaled,
                     qk.wq.reshaped([b * kvHeads, n, d / 8]), qk.scales.reshaped([b * kvHeads, n, d / group]),
                     (qk.biases ?? zeros).reshaped([b * kvHeads, n, d / group]),
                     qv.wq.reshaped([b * kvHeads, n, d / 8]), qv.scales.reshaped([b * kvHeads, n, d / group]),
                     (qv.biases ?? zeros).reshaped([b * kvHeads, n, d / group])],
                    grid: (b * qHeads * 32, 32, 1),
                    threadGroup: (32, 32, 1),
                    outputShapes: [[b * qHeads, l, d]],
                    outputDTypes: [.float32])[0]
            }

            let mine = run().reshaped([b, qHeads, l, d])
            let diff = MLX.max(MLX.abs(mine - reference)).item(Float.self)
            let refMag = MLX.mean(MLX.abs(reference)).item(Float.self)

            func timeIt(_ body: () -> MLXArray) -> Double {
                for _ in 0 ..< 3 { MLX.eval(body()) }
                let start = Date()
                for _ in 0 ..< 20 { MLX.eval(body()) }
                return Date().timeIntervalSince(start) / 20.0
            }
            let kb = k.asType(.bfloat16)
            let vb = v.asType(.bfloat16)
            let qb = q.asType(.bfloat16)
            let tFused = timeIt {
                MLXFast.scaledDotProductAttention(
                    queries: qb, keys: kb, values: vb, scale: scale, mask: .causal)
            }
            let tDecomposed = timeIt {
                quantizedScaledDotProductAttention(
                    queries: q,
                    quantizedKeys: (qk.wq, qk.scales, qk.biases ?? zeros),
                    quantizedValues: (qv.wq, qv.scales, qv.biases ?? zeros),
                    scale: scale, mask: .causal,
                    groupSize: group, bits: 4, mode: .affine)
            }
            let tMine = timeIt { run() }

            print("FUSEDQ L=\(l) n=\(n) maxdiff=\(diff) refmag=\(refMag) "
                + "bf16=\(String(format: "%.4f", tFused * 1000))ms "
                + "decomposed=\(String(format: "%.4f", tDecomposed * 1000))ms "
                + "mine=\(String(format: "%.4f", tMine * 1000))ms "
                + "| vs-bf16=\(String(format: "%.2f", tMine / tFused))x "
                + "speedup-vs-decomposed=\(String(format: "%.2f", tDecomposed / tMine))x")
            #expect(diff < 2e-2)
        }
    }
}
