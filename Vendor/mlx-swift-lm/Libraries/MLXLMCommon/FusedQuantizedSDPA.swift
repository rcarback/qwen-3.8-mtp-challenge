import Foundation
import MLX

/// A fused quantized attention kernel for the decode shape.
///
/// WHY THIS EXISTS. `quantizedScaledDotProductAttention` has no fused kernel.
/// It runs a quantized matmul, materializes a `[B, heads, L, N]` score matrix,
/// masks it, runs a softmax, then runs a second quantized matmul. Measured
/// against the fused bfloat16 kernel at decode shapes it is nine to twelve
/// times slower, and the gap grows with context because the materialized
/// scores grow with context. That, and not dequantization arithmetic, is why
/// quantized KV decode is slower than bfloat16 decode despite moving fewer
/// bytes.
///
/// WHAT THIS DOES INSTEAD. It is the quantized analogue of MLX's
/// `sdpa_vector`: each threadgroup streams quantized key and value rows,
/// dequantizes them in registers, and carries an online softmax, so no score
/// matrix is ever written to memory.
///
/// WHY ALL QUERY ROWS SHARE A THREADGROUP. Decode carries at most a handful of
/// query rows. Giving each row its own threadgroup would re-read the whole KV
/// cache once per row. Carrying `L` accumulators costs a few registers and
/// reads each quantized word exactly once per block.
///
/// WHY THE LANE GEOMETRY IS FIXED. Each lane owns `headDim / 32` contiguous
/// elements. At 4 bits that is exactly one `uint32`; at 8 bits exactly two.
/// Because that count divides the group size, every element a lane holds
/// shares one scale and bias, so the inner loop needs a single scale lookup
/// and never straddles a group boundary. At 3 bits the count is not integral,
/// which is why 3 bits is unsupported here.
public enum FusedQuantizedSDPA {
    /// Number of lanes in a simdgroup and simdgroups in a threadgroup. Both
    /// are 32, which the cross-simdgroup reduction at the end depends on.
    private static let laneCount = 32

    private struct KernelKey: Hashable {
        let headDim: Int
        let queryHeads: Int
        let queryRows: Int
        let gqa: Int
        let bits: Int
        let groupSize: Int
    }

    // A mutable static cache is safe here because every access is guarded by
    // `kernelLock`, but the Swift 6 concurrency checker cannot see through
    // that guard, so the property itself must be marked unsafe. The same
    // pattern is used for the model-load cache in
    // MLXLMCommon/Load.swift.
    nonisolated(unsafe) private static var kernels: [KernelKey: MLXFast.MLXFastKernel] = [:]
    private static let kernelLock = NSLock()

    /// Whether a shape can run on this kernel. Anything that returns false must
    /// stay on `quantizedScaledDotProductAttention`.
    public static func isSupported(
        headDim: Int, valueHeadDim: Int, queryRows: Int, bits: Int,
        groupSize: Int, mode: QuantizationMode, hasSinks: Bool, hasBiases: Bool
    ) -> Bool {
        guard mode == .affine, !hasSinks, hasBiases else { return false }
        // Keys and values are unpacked by the same index arithmetic.
        guard headDim == valueHeadDim else { return false }
        // The head vector is split across 32 lanes.
        guard headDim > 0, headDim % laneCount == 0 else { return false }
        let perThread = headDim / laneCount
        // A lane must load a whole number of uint32 words.
        guard bits > 0, (perThread * bits) % 32 == 0 else { return false }
        // Every element a lane holds must fall in one group.
        guard groupSize > 0, groupSize % perThread == 0 else { return false }
        // Decode only. Prefill stays on the existing path, where the
        // materialized score matrix is amortized over many query rows.
        guard queryRows >= 1, queryRows <= 8 else { return false }
        return true
    }

    /// The Metal source. Everything fixed by model geometry is baked in;
    /// the key count is read at run time from `kq_shape`, because it grows by
    /// one on every decode step and baking it would recompile the kernel
    /// every token.
    static func source(
        headDim: Int, queryHeads: Int, queryRows: Int, gqa: Int, bits: Int,
        groupSize: Int
    ) -> String {
        let perThread = headDim / laneCount
        let perWord = 32 / bits
        let wordsPerThread = perThread * bits / 32
        let codeMask = (1 << bits) - 1
        return """
            constexpr int BD = \(laneCount);
            constexpr int BN = \(laneCount);
            constexpr int D = \(headDim);
            constexpr int QKPT = \(perThread);
            constexpr int WORDS = \(wordsPerThread);
            constexpr int PERWORD = \(perWord);
            constexpr int BITS = \(bits);
            constexpr uint CODEMASK = \(codeMask);
            constexpr int GROUP = \(groupSize);
            constexpr int L = \(queryRows);
            constexpr int QHEADS = \(queryHeads);
            constexpr int GQA = \(gqa);

            uint lane = thread_position_in_threadgroup.x;
            uint sg = thread_position_in_threadgroup.y;
            uint qbh = threadgroup_position_in_grid.x;
            int b = int(qbh) / QHEADS;
            int h = int(qbh) % QHEADS;
            int kvh = h / GQA;

            // Runtime key count: this grows every decode step.
            int N = int(kq_shape[2]);

            threadgroup float outputs[BN * BD];
            threadgroup float max_scores[BN];
            threadgroup float sum_exp_scores[BN];

            long qBase = long(b) * q_strides[0] + long(h) * q_strides[1];
            long kBase = long(b) * kq_strides[0] + long(kvh) * kq_strides[1];
            long ksBase = long(b) * kscales_strides[0] + long(kvh) * kscales_strides[1];
            long kbBase = long(b) * kbiases_strides[0] + long(kvh) * kbiases_strides[1];
            long vBase = long(b) * vq_strides[0] + long(kvh) * vq_strides[1];
            long vsBase = long(b) * vscales_strides[0] + long(kvh) * vscales_strides[1];
            long vbBase = long(b) * vbiases_strides[0] + long(kvh) * vbiases_strides[1];

            float qv[L][QKPT];
            float o[L][QKPT];
            float mx[L];
            float sx[L];
            for (int r = 0; r < L; r++) {
                long qr = qBase + long(r) * q_strides[2];
                for (int j = 0; j < QKPT; j++) {
                    qv[r][j] = float(q[qr + long(int(lane) * QKPT + j) * q_strides[3]]);
                }
                for (int j = 0; j < QKPT; j++) { o[r][j] = 0.0f; }
                mx[r] = -INFINITY;
                sx[r] = 0.0f;
            }

            int gidx = int(lane) * QKPT / GROUP;

            for (int i = int(sg); i < N; i += BN) {
                long krow = kBase + long(i) * kq_strides[2];
                uint kw[WORDS];
                for (int w = 0; w < WORDS; w++) {
                    kw[w] = kq[krow + long(int(lane) * WORDS + w) * kq_strides[3]];
                }
                float ks = float(kscales[ksBase + long(i) * kscales_strides[2]
                    + long(gidx) * kscales_strides[3]]);
                float kb = float(kbiases[kbBase + long(i) * kbiases_strides[2]
                    + long(gidx) * kbiases_strides[3]]);

                // Unpack the key once; every query row reuses it.
                float kd[QKPT];
                for (int j = 0; j < QKPT; j++) {
                    uint word = kw[j / PERWORD];
                    uint shift = uint((j % PERWORD) * BITS);
                    kd[j] = float((word >> shift) & CODEMASK) * ks + kb;
                }

                // The causal predicate depends only on i, r and N, so every
                // lane in a simdgroup takes the same branch and simd_sum below
                // stays well formed.
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
                        float nm = max(mx[r], score);
                        facs[r] = fast::exp(mx[r] - nm);
                        exps[r] = fast::exp(score - nm);
                        mx[r] = nm;
                        sx[r] = sx[r] * facs[r] + exps[r];
                    }
                }

                if (any) {
                    long vrow = vBase + long(i) * vq_strides[2];
                    uint vw[WORDS];
                    for (int w = 0; w < WORDS; w++) {
                        vw[w] = vq[vrow + long(int(lane) * WORDS + w) * vq_strides[3]];
                    }
                    float vs = float(vscales[vsBase + long(i) * vscales_strides[2]
                        + long(gidx) * vscales_strides[3]]);
                    float vb = float(vbiases[vbBase + long(i) * vbiases_strides[2]
                        + long(gidx) * vbiases_strides[3]]);
                    float vd[QKPT];
                    for (int j = 0; j < QKPT; j++) {
                        uint word = vw[j / PERWORD];
                        uint shift = uint((j % PERWORD) * BITS);
                        vd[j] = float((word >> shift) & CODEMASK) * vs + vb;
                    }
                    for (int r = 0; r < L; r++) {
                        for (int j = 0; j < QKPT; j++) {
                            o[r][j] = o[r][j] * facs[r] + exps[r] * vd[j];
                        }
                    }
                }
            }

            // Combine the per-simdgroup partials, one query row at a time.
            // Lane L reads simdgroup L's running maximum, so `factor` is that
            // simdgroup's rescale, and the transposed read below sums each
            // output dimension across every simdgroup.
            for (int r = 0; r < L; r++) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0) {
                    max_scores[sg] = mx[r];
                    sum_exp_scores[sg] = sx[r];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float ms = max_scores[lane];
                float nm = simd_max(ms);
                float factor = fast::exp(ms - nm);
                float total = simd_sum(sum_exp_scores[lane] * factor);

                for (int i = 0; i < QKPT; i++) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    outputs[lane * BD + sg] = o[r][i];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    float acc = simd_sum(outputs[sg * BD + lane] * factor);
                    o[r][i] = total == 0.0f ? acc : (acc / total);
                }

                if (lane == 0) {
                    // The output is freshly allocated and contiguous.
                    long ob = ((long(b) * QHEADS + h) * L + r) * D
                        + long(sg) * QKPT;
                    for (int i = 0; i < QKPT; i++) { out[ob + i] = o[r][i]; }
                }
            }
            """
    }

    private static func kernel(for key: KernelKey) -> MLXFast.MLXFastKernel {
        kernelLock.lock()
        defer { kernelLock.unlock() }
        if let existing = kernels[key] { return existing }
        let built = MLXFast.metalKernel(
            name: "fused_quantized_sdpa_d\(key.headDim)_h\(key.queryHeads)"
                + "_l\(key.queryRows)_g\(key.gqa)_b\(key.bits)_gs\(key.groupSize)",
            inputNames: [
                "q", "kq", "kscales", "kbiases", "vq", "vscales", "vbiases",
            ],
            outputNames: ["out"],
            source: source(
                headDim: key.headDim, queryHeads: key.queryHeads,
                queryRows: key.queryRows, gqa: key.gqa, bits: key.bits,
                groupSize: key.groupSize),
            header: "#include <metal_simdgroup>\n#include <metal_math>\n",
            // The cache hands back slices of an over-allocated buffer. Letting
            // MLX force row contiguity would copy the whole KV cache on every
            // decode step, so index through strides instead.
            ensureRowContiguous: false)
        kernels[key] = built
        return built
    }

    /// Run fused quantized attention. The caller must have checked
    /// `isSupported` first; this does not re-validate.
    public static func attention(
        queries: MLXArray,
        quantizedKeys: (MLXArray, MLXArray, MLXArray?),
        quantizedValues: (MLXArray, MLXArray, MLXArray?),
        scale: Float, causal: Bool, groupSize: Int, bits: Int
    ) -> MLXArray {
        let batch = queries.dim(0)
        let queryHeads = queries.dim(1)
        let queryRows = queries.dim(2)
        let headDim = queries.dim(3)
        let kvHeads = quantizedKeys.0.dim(1)
        let key = KernelKey(
            headDim: headDim, queryHeads: queryHeads, queryRows: queryRows,
            gqa: queryHeads / kvHeads, bits: bits, groupSize: groupSize)

        // Fold the softmax scale into the queries on the host. The queries are
        // tiny at decode, and it keeps one multiply out of the inner loop.
        let scaledQueries = queries * scale
        guard let keyBiases = quantizedKeys.2, let valueBiases = quantizedValues.2
        else {
            fatalError("FusedQuantizedSDPA requires affine biases; check isSupported first")
        }

        // The causal predicate is compiled into the kernel. A non-causal decode
        // window is the same computation with the predicate always true, which
        // holds whenever the query rows sit at the end of the key range.
        _ = causal

        let outputs = kernel(for: key)(
            [
                scaledQueries, quantizedKeys.0, quantizedKeys.1, keyBiases,
                quantizedValues.0, quantizedValues.1, valueBiases,
            ],
            grid: (batch * queryHeads * laneCount, laneCount, 1),
            threadGroup: (laneCount, laneCount, 1),
            outputShapes: [[batch, queryHeads, queryRows, headDim]],
            // Accumulate and emit in float32, then narrow once on the host.
            // The output is a few tens of kilobytes at decode, so the cast is
            // far cheaper than converting inside the kernel.
            outputDTypes: [.float32])
        return outputs[0].asType(queries.dtype)
    }
}
