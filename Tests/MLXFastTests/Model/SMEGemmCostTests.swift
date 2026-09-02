import Accelerate
import Foundation
import Testing

/// What can the CPU actually do on our GEMM shapes when it uses the matrix
/// unit instead of a scalar SIMD loop?
///
/// MLX's CPU quantized path (`_qmm_t_simd`) measured 0.033 GFLOPS in the
/// column-split sweep -- roughly 3,000x below single-core NEON and 60,000x
/// below SME. That measured the wrong thing: it is a hand-written float SIMD
/// loop, not the matrix unit. This measures the matrix unit.
///
/// Route: Accelerate's `cblas_sgemm`, which dispatches to the AMX/SME block on
/// Apple silicon. `sysctl` on this host reports FEAT_SME, FEAT_SME2 and
/// SVL=512 bits, and published microbenchmarks put the P-cluster at ~2008
/// GFLOPS fp32 (SME FMOPA) against ~107 GFLOPS for single-core NEON FMLA.
/// The unit is per-CLUSTER, not per-core, so this is a shared ceiling and
/// more threads do not raise it.
///
/// Modes:
///   `sgemm`         - dense fp32 GEMM only. The ceiling.
///   `dequant`       - 4-bit affine group-64 -> fp32 expansion only. The tax.
///   `dequant_sgemm` - both, which is what a real CPU column slice must pay.
///
/// One shape per process, matching the GPU harness discipline.
@Suite(.serialized)
struct SMEGemmCostTests {
    /// Expands affine 4-bit group-64 weights to fp32, the layout `cblas_sgemm`
    /// needs. Two nibbles per byte, one scale and bias per 64 values.
    private static func dequantize(
        packed: UnsafePointer<UInt8>, scales: UnsafePointer<Float>,
        biases: UnsafePointer<Float>, out: UnsafeMutablePointer<Float>,
        rows: Int, cols: Int
    ) {
        let groupsPerRow = cols / 64
        for r in 0 ..< rows {
            let rowByteBase = r * (cols / 2)
            let rowGroupBase = r * groupsPerRow
            for g in 0 ..< groupsPerRow {
                let s = scales[rowGroupBase + g]
                let b = biases[rowGroupBase + g]
                let byteBase = rowByteBase + g * 32
                let outBase = r * cols + g * 64
                for i in 0 ..< 32 {
                    let byte = packed[byteBase + i]
                    out[outBase + 2 * i] = Float(byte & 0x0F) * s + b
                    out[outBase + 2 * i + 1] = Float(byte >> 4) * s + b
                }
            }
        }
    }

    private static func timeIt(_ body: () -> Void) -> Double {
        body()
        var best = Double.infinity
        for _ in 0 ..< 3 {
            let start = Date()
            body()
            best = Swift.min(best, Date().timeIntervalSince(start))
        }
        return best
    }

    @Test("CPU matrix-unit GEMM, alone in its process")
    func smeGemmPoint() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let M = Int(env["MLXFAST_SME_M"] ?? "1024") ?? 1024
        let N = Int(env["MLXFAST_SME_N"] ?? "2176") ?? 2176
        let K = Int(env["MLXFAST_SME_K"] ?? "5120") ?? 5120
        let mode = env["MLXFAST_SME_MODE"] ?? "sgemm"
        guard mode == "sgemm" || mode == "dequant" || mode == "dequant_sgemm"
        else {
            Issue.record("MLXFAST_SME_MODE must be sgemm|dequant|dequant_sgemm")
            return
        }
        guard K % 64 == 0 else {
            Issue.record("K must be a multiple of the group size")
            return
        }

        let a = UnsafeMutablePointer<Float>.allocate(capacity: M * K)
        let b = UnsafeMutablePointer<Float>.allocate(capacity: N * K)
        let c = UnsafeMutablePointer<Float>.allocate(capacity: M * N)
        let packed = UnsafeMutablePointer<UInt8>.allocate(capacity: N * K / 2)
        let groups = N * K / 64
        let scales = UnsafeMutablePointer<Float>.allocate(capacity: groups)
        let biases = UnsafeMutablePointer<Float>.allocate(capacity: groups)
        defer {
            a.deallocate(); b.deallocate(); c.deallocate()
            packed.deallocate(); scales.deallocate(); biases.deallocate()
        }
        for i in 0 ..< M * K { a[i] = Float.random(in: -1 ... 1) }
        for i in 0 ..< N * K / 2 { packed[i] = UInt8.random(in: 0 ... 255) }
        for i in 0 ..< groups {
            scales[i] = 0.01
            biases[i] = -0.08
        }
        Self.dequantize(
            packed: packed, scales: scales, biases: biases, out: b,
            rows: N, cols: K)

        let gemm = {
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                Int32(M), Int32(N), Int32(K), 1.0,
                a, Int32(K), b, Int32(K), 0.0, c, Int32(N))
        }
        let deq = {
            Self.dequantize(
                packed: packed, scales: scales, biases: biases, out: b,
                rows: N, cols: K)
        }

        let dt: Double
        switch mode {
        case "sgemm": dt = Self.timeIt(gemm)
        case "dequant": dt = Self.timeIt(deq)
        default: dt = Self.timeIt { deq(); gemm() }
        }

        let flops = 2.0 * Double(M) * Double(N) * Double(K)
        print(String(
            format: "SMEPOINT\t%@\t%d\t%d\t%d\t%.4f\t%.3f",
            mode, M, N, K, 1000 * dt, flops / dt / 1e9))
    }
}
