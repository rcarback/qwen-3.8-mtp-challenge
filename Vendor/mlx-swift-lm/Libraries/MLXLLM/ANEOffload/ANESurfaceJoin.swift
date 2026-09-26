import MLX

/// Converts channel-major ANE output into an independent, lazy MLX result.
enum ANESurfaceJoin {
    private static let add = MLXFast.metalKernel(
        name: "ane_surface_cast_add", inputNames: ["ane", "gpu"], outputNames: ["out"],
        source: """
        uint tx = thread_position_in_threadgroup.x;
        uint ty = thread_position_in_threadgroup.y;
        uint channel_base = threadgroup_position_in_grid.x * 32;
        uint row_base = threadgroup_position_in_grid.y * 32;
        threadgroup half tile[32][33];
        // Adjacent lanes load adjacent rows from one channel.
        for (uint offset = 0; offset < 32; offset += 8) {
            uint channel = channel_base + ty + offset;
            uint row = row_base + tx;
            tile[ty + offset][tx] = channel < HIDDEN && row < ROWS
                ? ane[channel * STRIDE + row] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // After the transpose, adjacent lanes store adjacent channels.
        for (uint offset = 0; offset < 32; offset += 8) {
            uint row = row_base + ty + offset;
            uint channel = channel_base + tx;
            if (row < ROWS && channel < HIDDEN) {
                uint i = row * HIDDEN + channel;
                // Preserve the existing cast-before-add rounding boundary.
                T converted = T(tile[tx][ty + offset]);
                out[i] = T(float(converted) + float(gpu[i]));
            }
        }
        """)
    private static let gather = MLXFast.metalKernel(
        name: "ane_surface_cast", inputNames: ["ane"], outputNames: ["out"],
        source: """
        uint tx = thread_position_in_threadgroup.x;
        uint ty = thread_position_in_threadgroup.y;
        uint channel_base = threadgroup_position_in_grid.x * 32;
        uint row_base = threadgroup_position_in_grid.y * 32;
        threadgroup half tile[32][33];
        for (uint offset = 0; offset < 32; offset += 8) {
            uint channel = channel_base + ty + offset;
            uint row = row_base + tx;
            tile[ty + offset][tx] = channel < HIDDEN && row < ROWS
                ? ane[channel * STRIDE + row] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 0; offset < 32; offset += 8) {
            uint row = row_base + ty + offset;
            uint channel = channel_base + tx;
            if (row < ROWS && channel < HIDDEN) {
                out[row * HIDDEN + channel] = T(tile[tx][ty + offset]);
            }
        }
        """)

    static func callAsFunction(_ ane: MLXArray, gpu: MLXArray?, rows: Int,
                              hidden: Int, stride: Int, dtype: DType) -> MLXArray {
        precondition(ane.shape == [hidden, stride] && rows <= stride)
        precondition(gpu == nil || gpu!.shape == [rows, hidden])
        if let gpu { precondition(gpu.dtype == dtype) }
        let kernel = gpu == nil ? gather : add
        let inputs = gpu.map { [ane, $0] } ?? [ane]
        return kernel(inputs,
            template: [("T", dtype), ("ROWS", rows), ("HIDDEN", hidden), ("STRIDE", stride)],
            grid: (((hidden + 31) / 32) * 32, ((rows + 31) / 32) * 8, 1),
            threadGroup: (32, 8, 1),
            outputShapes: [[rows, hidden]], outputDTypes: [dtype])[0]
    }
}
