import Foundation
import MLX
import MLXFast

/// One persistent Metal kernel for the routed-expert GLU. Replaces the
/// gate/up/silu/down launch quartet; the router, sort, unsort and weighted sum
/// stay on the MLX side so the final reduction order is unchanged.
///
/// This variant dequantizes 4-bit affine group-32 expert weights inline
/// (Task 3). Task 2's dense fp16 variant (`denseForward`, the
/// `moe_fused_dense` kernel) debugged the persistent work-queue scheduler in
/// isolation from dequantization and has been deleted outright — replaced,
/// not kept alongside this one, per the project's replace-do-not-deprecate
/// rule.
///
/// **The packing contract** (verified empirically against `MLX.quantized` /
/// `MLX.dequantized` before relying on it): MLX affine 4-bit storage packs
/// eight weights per `uint32`, **low nibble first**, with one `(scale, bias)`
/// fp16 pair per group of 32 weights along the flat row-major index (i.e. the
/// **input** axis, since input is the fastest-varying dimension of a
/// `[..., out, in]`-shaped weight). For output row `o` of expert `e` at input
/// position `k`, with `base = (e * OUT + o) * IN`:
/// `idx = base + k`, packed word `w[idx >> 3]`, nibble `(word >> ((idx & 7) *
/// 4)) & 0xF`, group `s[idx >> 5]` / `b[idx >> 5]`,
/// `value = scale * nibble + bias`. This holds only when the input dimension
/// is a multiple of 32; see the `precondition` below.
///
/// Scheduling note (Ruling C3): the brief's original design claims work items
/// with a `device atomic_uint *` counter passed as an input. MLX's generated
/// kernel signature qualifies every input as `const device`, which makes that
/// cast ill-formed, and `MLXFast.metalKernel`'s `atomicOutputs` flag makes
/// *every* declared output atomic (see
/// `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/common/metal_kernel.cpp`,
/// `write_signature`) — there is no way to make only the counter atomic while
/// leaving the fp16 `y` output as a plain buffer, and `atomic<half>` is not a
/// valid Metal type in any case. So this kernel uses the brief's named
/// fallback instead: a **static partition** with no atomics at all.
/// Threadgroup `g` of `G` total threadgroups processes work items
/// `g, g + G, g + 2G, ...`, with the loop bound read from
/// `block_offsets[numExperts]`. This removes the dynamic load balancing that
/// is the main reason a persistent work-queue kernel exists — a later
/// measurement task should treat this as a statically scheduled kernel until
/// dynamic claiming is revisited.
public enum FusedRoutedMoE {
    /// Rows per work item. Sized so the fp16 intermediate
    /// (`blockRows * hiddenDim * 2` bytes) fits threadgroup memory with room to
    /// spare, and so that the 14.69-row per-expert mean lands inside one block:
    /// an expert whose rows exceed this re-reads its weights once per block.
    public static let blockRows = 16

    /// Threads per threadgroup. 256 gives each thread 40 of the 10,240 output
    /// values a full block produces at inDim 2560.
    public static let threadgroupSize = 256

    private static let quantizedKernel = MLXFast.metalKernel(
        name: "moe_fused_q4g32",
        inputNames: [
            "x", "gate_up_w", "gate_up_s", "gate_up_b",
            "down_w", "down_s", "down_b",
            "row_offsets", "block_offsets",
        ],
        outputNames: ["y"],
        source: """
            threadgroup half inter[BLOCK_ROWS * HIDDEN_DIM];

            const uint tid = thread_position_in_threadgroup.x;
            const uint gid = threadgroup_position_in_grid.x;
            const uint num_groups = threadgroups_per_grid.x;
            const uint total_items = (uint)block_offsets[N_EXPERTS];

            // Static partition: threadgroup `gid` of `num_groups` owns work
            // items gid, gid + num_groups, gid + 2*num_groups, ... No atomic
            // claim counter -- see the Ruling C3 note in the Swift file header.
            for (uint item = gid; item < total_items; item += num_groups) {
                // Map the flat item index to (expert, row block) by scanning the
                // per-expert block prefix sum. N_EXPERTS is 512 at most, and this
                // runs once per item, not once per row.
                int e = 0;
                for (int c = 1; c <= N_EXPERTS; ++c) {
                    if ((uint)block_offsets[c] > item) { e = c - 1; break; }
                }
                const uint local_block = item - (uint)block_offsets[e];
                const int  row_lo = row_offsets[e] + (int)(local_block * BLOCK_ROWS);
                const int  row_hi = min(row_offsets[e + 1], row_lo + BLOCK_ROWS);
                const int  n_rows = row_hi - row_lo;

                // Stage 1: gate and up over the full K, dequantizing the 4-bit
                // affine group-32 weights inline, then SiLU product into
                // threadgroup memory. Each thread owns a strided set of
                // (row, channel) pairs.
                for (uint slot = tid; slot < (uint)(n_rows * HIDDEN_DIM);
                     slot += THREADGROUP_SIZE) {
                    const int r = (int)(slot / HIDDEN_DIM);
                    const int c = (int)(slot % HIDDEN_DIM);
                    float acc_g = 0.0f;
                    float acc_u = 0.0f;
                    const device half *xr = x + (uint)(row_lo + r) * IN_DIM;
                    const uint base_g = ((uint)e * 2u * HIDDEN_DIM + (uint)c) * IN_DIM;
                    const uint base_u = ((uint)e * 2u * HIDDEN_DIM + (uint)(HIDDEN_DIM + c)) * IN_DIM;
                    for (int k = 0; k < IN_DIM; ++k) {
                        const float xv = (float)xr[k];
                        acc_g = fma(xv, dq(gate_up_w, gate_up_s, gate_up_b, base_g, k), acc_g);
                        acc_u = fma(xv, dq(gate_up_w, gate_up_s, gate_up_b, base_u, k), acc_u);
                    }
                    const float s = acc_g / (1.0f + exp(-acc_g));   // SiLU
                    inter[r * HIDDEN_DIM + c] = (half)(s * acc_u);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Stage 2: down projection straight out of threadgroup memory,
                // dequantizing the down-projection weights inline.
                for (uint slot = tid; slot < (uint)(n_rows * IN_DIM);
                     slot += THREADGROUP_SIZE) {
                    const int r = (int)(slot / IN_DIM);
                    const int o = (int)(slot % IN_DIM);
                    float acc = 0.0f;
                    const uint base_d = ((uint)e * IN_DIM + (uint)o) * HIDDEN_DIM;
                    for (int h = 0; h < HIDDEN_DIM; ++h) {
                        acc = fma((float)inter[r * HIDDEN_DIM + h],
                                  dq(down_w, down_s, down_b, base_d, h), acc);
                    }
                    y[(uint)(row_lo + r) * IN_DIM + (uint)o] = (half)acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """,
        header: """
            #include <metal_stdlib>
            using namespace metal;

            // Dequantize one 4-bit affine group-32 weight. `w` is the packed
            // uint32 weight buffer (8 nibbles per word, low nibble first),
            // `s`/`b` are the per-group fp16 scale/bias buffers, `base` is the
            // flat row-major (expert, out) offset, and `k` is the position
            // along the input axis within that row. See the packing contract
            // in the Swift file's module doc comment.
            inline float dq(const device uint32_t *w, const device half *s,
                            const device half *b, uint base, int k) {
                const uint idx = base + (uint)k;
                const uint word = w[idx >> 3];
                const uint nib = (word >> ((idx & 7u) * 4u)) & 0xFu;
                const uint grp = idx >> 5;
                return fma((float)nib, (float)s[grp], (float)b[grp]);
            }
            """,
        ensureRowContiguous: true)

    /// `xSorted` is `[rows, inDim]`. `gateUpWeight` is the packed 4-bit affine
    /// group-32 quantization of a `[E, 2 * hiddenDim, inDim]` fp16 weight
    /// (`gateUpScales` / `gateUpBiases` its per-group fp16 scale/bias
    /// buffers); `downWeight` likewise quantizes `[E, inDim, hiddenDim]`.
    /// Returns `[rows, inDim]`.
    public static func forward(
        xSorted: MLXArray,
        gateUpWeight: MLXArray, gateUpScales: MLXArray, gateUpBiases: MLXArray,
        downWeight: MLXArray, downScales: MLXArray, downBiases: MLXArray,
        rowOffsets: MLXArray, blockOffsets: MLXArray,
        hiddenDim: Int, inDim: Int, numExperts: Int,
        // Task 5, 2026-09-04, real per-layer geometry (512 experts, inDim
        // 2560, hidden 640, 7000 rows, one 574-row hot expert, 190 idle):
        // control (naive per-expert quantizedMM loop) 109.3ms; tg32 939.2ms;
        // tg64 680.5ms (winner); tg128 754.5ms; tg256 920.9ms; tg512 922.3ms.
        // See .superpowers/sdd/2026-09-04-fused-moe-kernel/task-5-report.md
        // for the full arm table.
        threadgroups: Int = 64
    ) -> MLXArray {
        let rows = xSorted.dim(0)

        // The dequantization helper indexes packed weights and scales by
        // `idx >> 3` / `idx >> 5` from a flat element index, which is only
        // correct when both the contracted (input) dimension and the hidden
        // dimension are multiples of 32 (Ruling C4). A future geometry that
        // violates this fails loudly here instead of silently reading the
        // wrong scale inside the kernel.
        precondition(
            inDim % 32 == 0 && hiddenDim % 32 == 0,
            "FusedRoutedMoE requires inDim and hiddenDim to be multiples of 32 "
                + "(got inDim=\(inDim), hiddenDim=\(hiddenDim)); the dequantization "
                + "index arithmetic (idx >> 3, idx >> 5) is only correct under that "
                + "constraint.")

        let out = quantizedKernel(
            [
                xSorted, gateUpWeight, gateUpScales, gateUpBiases,
                downWeight, downScales, downBiases,
                rowOffsets, blockOffsets,
            ],
            template: [
                ("BLOCK_ROWS", blockRows),
                ("HIDDEN_DIM", hiddenDim),
                ("IN_DIM", inDim),
                ("N_EXPERTS", numExperts),
                ("THREADGROUP_SIZE", threadgroupSize),
            ],
            grid: (threadgroups * threadgroupSize, 1, 1),
            threadGroup: (threadgroupSize, 1, 1),
            outputShapes: [[rows, inDim]],
            outputDTypes: [.float16])
        return out[0]
    }
}
