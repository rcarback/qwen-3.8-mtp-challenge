import Foundation
import MLX
import MLXFast

/// One Metal kernel for the router's selection tail: top-k over the expert
/// logits and the precise float32 softmax over the selected ones, replacing
/// `argPartition` + slice + `takeAlong` + `softmax`.
///
/// **Scope: the tail only, not the gate GEMM.** The bead this implements
/// describes fusing all four router operations. The GEMM is left alone
/// deliberately: it is already one tuned `matmul` launch, and this repository's
/// one prior hand-written replacement for a tuned kernel (`FusedRoutedMoE`) ran
/// 48x slower than the library path it replaced. What is left is three launches
/// over a `[rows, 512]` tensor that exist only to move data between library
/// primitives, and those are what this collapses.
///
/// **Why the tie-break trap in the bead does not bind.** The stated risk is
/// that `argPartition` has an unspecified order among equal values, so a fresh
/// kernel selects differently. Order is not observable here: the caller feeds
/// the indices to a gather and then reduces the gathered rows with
/// `(y * w).sum(axis: -2)`, which is order-independent up to float
/// associativity. Only the SET has to match, and it can differ only when two
/// logits are exactly equal in float32 astride the k-boundary. This kernel
/// breaks such a tie toward the lower expert index, deterministically; MLX does
/// not promise any particular choice. `FusedRouterSelectTests` measures how
/// often that boundary tie actually occurs on real routing.
public enum FusedRouterSelect {
    /// One threadgroup per row. 256 threads cover 512 experts at two each.
    public static let threadgroupSize = 256

    private static let kernel = MLXFast.metalKernel(
        name: "router_topk_softmax",
        inputNames: ["logits"],
        outputNames: ["indices", "weights"],
        source: """
            threadgroup float sval[THREADGROUP_SIZE];
            threadgroup int sidx[THREADGROUP_SIZE];
            threadgroup float chosen[TOP_K];
            threadgroup int chosen_idx[TOP_K];

            const uint tid = thread_position_in_threadgroup.x;
            const uint row = threadgroup_position_in_grid.x;
            const device float *L = logits + row * (uint)N_EXPERTS;

            // Each thread owns PER_THREAD of the N_EXPERTS logits for the whole
            // selection, so the logits are read from device memory once and the
            // k rounds below run entirely out of registers and threadgroup
            // memory. This is the launch the fusion is buying.
            float v[PER_THREAD];
            int ix[PER_THREAD];
            for (int j = 0; j < PER_THREAD; ++j) {
                uint e = tid + (uint)j * (uint)THREADGROUP_SIZE;
                v[j] = (e < (uint)N_EXPERTS) ? L[e] : -INFINITY;
                ix[j] = (int)e;
            }

            for (int k = 0; k < TOP_K; ++k) {
                float bv = -INFINITY;
                int bi = N_EXPERTS;
                for (int j = 0; j < PER_THREAD; ++j) {
                    // Lower index wins an exact tie, at every level of the
                    // reduction, so the selected SET does not depend on how the
                    // threads happen to be scheduled.
                    if (v[j] > bv || (v[j] == bv && ix[j] < bi)) { bv = v[j]; bi = ix[j]; }
                }
                sval[tid] = bv;
                sidx[tid] = bi;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = (uint)THREADGROUP_SIZE / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        float ov = sval[tid + s];
                        int oi = sidx[tid + s];
                        if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                            sval[tid] = ov;
                            sidx[tid] = oi;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (tid == 0) { chosen[k] = sval[0]; chosen_idx[k] = sidx[0]; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                int win = chosen_idx[k];
                for (int j = 0; j < PER_THREAD; ++j) {
                    if (ix[j] == win) { v[j] = -INFINITY; }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // TOP_K is 10. A parallel softmax over ten values would cost more in
            // barriers than it saves, so one thread finishes the row.
            if (tid == 0) {
                float m = -INFINITY;
                for (int k = 0; k < TOP_K; ++k) { m = max(m, chosen[k]); }
                float sum = 0.0f;
                for (int k = 0; k < TOP_K; ++k) {
                    float e = exp(chosen[k] - m);
                    chosen[k] = e;
                    sum += e;
                }
                for (int k = 0; k < TOP_K; ++k) {
                    indices[row * (uint)TOP_K + (uint)k] = chosen_idx[k];
                    weights[row * (uint)TOP_K + (uint)k] = chosen[k] / sum;
                }
            }
            """,
        ensureRowContiguous: true)

    /// `logits` is `[rows, numExperts]` float32. Returns `[rows, topK]` int32
    /// expert indices and `[rows, topK]` float32 softmax weights over exactly
    /// those logits -- the same quantity as
    /// `softmax(takeAlong(logits, argPartition(...)), precise: true)`.
    public static func forward(logits: MLXArray, topK: Int, numExperts: Int)
        -> (indices: MLXArray, weights: MLXArray)
    {
        precondition(
            logits.ndim == 2 && logits.dim(1) == numExperts,
            "FusedRouterSelect: logits must be [rows, \(numExperts)], got \(logits.shape).")
        precondition(
            logits.dtype == .float32,
            "FusedRouterSelect: logits must be float32, got \(logits.dtype). The selection "
                + "and the softmax both run in float32 to match the eager router.")
        precondition(
            topK <= numExperts,
            "FusedRouterSelect: topK \(topK) exceeds numExperts \(numExperts).")
        // `chosen` and `chosen_idx` are threadgroup arrays of TOP_K elements and
        // `sval`/`sidx` of THREADGROUP_SIZE; all four together stay far inside
        // Metal's 32 KiB threadgroup limit at these sizes.
        let rows = logits.dim(0)
        let perThread = (numExperts + threadgroupSize - 1) / threadgroupSize
        let out = kernel(
            [logits],
            template: [
                ("N_EXPERTS", numExperts),
                ("TOP_K", topK),
                ("THREADGROUP_SIZE", threadgroupSize),
                ("PER_THREAD", perThread),
            ],
            grid: (rows * threadgroupSize, 1, 1),
            threadGroup: (threadgroupSize, 1, 1),
            outputShapes: [[rows, topK], [rows, topK]],
            outputDTypes: [.int32, .float32])
        return (out[0], out[1])
    }
}
