import Foundation
import MLX
import MLXFast

/// One persistent Metal kernel for the routed-expert GLU. Replaces the
/// gate/up/silu/down launch quartet; the router, sort, unsort and weighted sum
/// stay on the MLX side so the final reduction order is unchanged.
///
/// This variant runs over **dense fp16** expert weights on purpose (see the
/// module-level plan, Task 2): the persistent work-queue scheduler is
/// debugged here in isolation from the 4-bit dequantization Task 3 adds.
/// Task 3 replaces the dense weight load with the quantized one and deletes
/// this dense variant outright.
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

    private static let denseKernel = MLXFast.metalKernel(
        name: "moe_fused_dense",
        inputNames: ["x", "gate_up", "down", "row_offsets", "block_offsets"],
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

                // Stage 1: gate and up over the full K, then SiLU product into
                // threadgroup memory. Each thread owns a strided set of
                // (row, channel) pairs.
                for (uint slot = tid; slot < (uint)(n_rows * HIDDEN_DIM);
                     slot += THREADGROUP_SIZE) {
                    const int r = (int)(slot / HIDDEN_DIM);
                    const int c = (int)(slot % HIDDEN_DIM);
                    float acc_g = 0.0f;
                    float acc_u = 0.0f;
                    const device half *xr = x + (uint)(row_lo + r) * IN_DIM;
                    const device half *wg = gate_up
                        + ((uint)e * 2u * HIDDEN_DIM + (uint)c) * IN_DIM;
                    const device half *wu = gate_up
                        + ((uint)e * 2u * HIDDEN_DIM + (uint)(HIDDEN_DIM + c)) * IN_DIM;
                    for (int k = 0; k < IN_DIM; ++k) {
                        const float xv = (float)xr[k];
                        acc_g = fma(xv, (float)wg[k], acc_g);
                        acc_u = fma(xv, (float)wu[k], acc_u);
                    }
                    const float s = acc_g / (1.0f + exp(-acc_g));   // SiLU
                    inter[r * HIDDEN_DIM + c] = (half)(s * acc_u);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Stage 2: down projection straight out of threadgroup memory.
                for (uint slot = tid; slot < (uint)(n_rows * IN_DIM);
                     slot += THREADGROUP_SIZE) {
                    const int r = (int)(slot / IN_DIM);
                    const int o = (int)(slot % IN_DIM);
                    float acc = 0.0f;
                    const device half *wd = down
                        + ((uint)e * IN_DIM + (uint)o) * HIDDEN_DIM;
                    for (int h = 0; h < HIDDEN_DIM; ++h) {
                        acc = fma((float)inter[r * HIDDEN_DIM + h], (float)wd[h], acc);
                    }
                    y[(uint)(row_lo + r) * IN_DIM + (uint)o] = (half)acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """,
        header: """
            #include <metal_stdlib>
            using namespace metal;
            """,
        ensureRowContiguous: true)

    /// `xSorted` is `[rows, inDim]`. `gateUp` is `[E, 2 * hiddenDim, inDim]`,
    /// `down` is `[E, inDim, hiddenDim]`, both `.float16`.
    /// Returns `[rows, inDim]`.
    public static func denseForward(
        xSorted: MLXArray, gateUp: MLXArray, down: MLXArray,
        rowOffsets: MLXArray, blockOffsets: MLXArray,
        threadgroups: Int = 256
    ) -> MLXArray {
        let rows = xSorted.dim(0)
        let inDim = xSorted.dim(1)
        let experts = down.dim(0)
        let hidden = down.dim(2)

        // Task 3's dequantization indexes packed weights and scales by
        // `idx >> 3` / `idx >> 5` from a flat element index, which is only
        // correct when both the contracted (input) dimension and the hidden
        // dimension are multiples of 32 (Ruling C4). Enforced here, one task
        // early, so a future geometry fails loudly instead of silently
        // reading the wrong scale once Task 3 lands.
        precondition(
            inDim % 32 == 0 && hidden % 32 == 0,
            "FusedRoutedMoE requires inDim and hiddenDim to be multiples of 32 "
                + "(got inDim=\(inDim), hiddenDim=\(hidden)); Task 3's dequantization "
                + "index arithmetic (idx >> 3, idx >> 5) is only correct under that "
                + "constraint.")

        let out = denseKernel(
            [xSorted, gateUp, down, rowOffsets, blockOffsets],
            template: [
                ("BLOCK_ROWS", blockRows),
                ("HIDDEN_DIM", hidden),
                ("IN_DIM", inDim),
                ("N_EXPERTS", experts),
                ("THREADGROUP_SIZE", threadgroupSize),
            ],
            grid: (threadgroups * threadgroupSize, 1, 1),
            threadGroup: (threadgroupSize, 1, 1),
            outputShapes: [[rows, inDim]],
            outputDTypes: [.float16])
        return out[0]
    }
}
