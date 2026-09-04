import Foundation
import MLX
import MLXFast

/// Index structures the fused routed-MoE kernel uses to schedule work items
/// without a host round trip.
public enum MoEWorkQueue {
    /// One thread per expert, binary-searching the lower bound of its id in the
    /// non-decreasing `sortedIndices`. 512 threads per layer; the alternative
    /// (a one-hot sum) would materialise a `[numExperts, rows]` intermediate.
    private static let offsetsKernel = MLXFast.metalKernel(
        name: "moe_row_offsets",
        inputNames: ["idx"],
        outputNames: ["out"],
        source: """
            uint e = thread_position_in_grid.x;
            if (e > (uint)n_experts) return;
            if (e == (uint)n_experts) { out[e] = n_rows; return; }
            // lower_bound: first position whose value is >= e
            int lo = 0;
            int hi = n_rows;
            while (lo < hi) {
                int mid = lo + (hi - lo) / 2;
                if (idx[mid] < (int)e) { lo = mid + 1; } else { hi = mid; }
            }
            out[e] = lo;
            """,
        header: "",
        ensureRowContiguous: true)

    /// `rowOffsets[e]` is the first sorted row belonging to expert `e`;
    /// `rowOffsets[numExperts]` is `rows`. Shape `[numExperts + 1]`, `.int32`.
    public static func rowOffsets(sortedIndices: MLXArray, numExperts: Int) -> MLXArray {
        let rows = sortedIndices.dim(0)
        let out = offsetsKernel(
            [sortedIndices.asType(.int32)],
            template: [("n_experts", numExperts), ("n_rows", rows)],
            grid: (numExperts + 1, 1, 1),
            threadGroup: (min(numExperts + 1, 256), 1, 1),
            outputShapes: [[numExperts + 1]],
            outputDTypes: [.int32])
        return out[0]
    }

    /// Exclusive prefix sum over `ceil(rowCount(e) / blockRows)`.
    /// `blockOffsets[numExperts]` is the total work-item count.
    /// Shape `[numExperts + 1]`, `.int32`.
    public static func blockOffsets(rowOffsets: MLXArray, blockRows: Int) -> MLXArray {
        let n = rowOffsets.dim(0) - 1
        let counts = rowOffsets[1...] - rowOffsets[..<n]
        let blocks = (counts + Int32(blockRows - 1)).floorDivide(Int32(blockRows))
        let prefix = MLX.cumsum(blocks, axis: 0)
        return MLX.concatenated([MLXArray([Int32(0)]), prefix]).asType(.int32)
    }
}
