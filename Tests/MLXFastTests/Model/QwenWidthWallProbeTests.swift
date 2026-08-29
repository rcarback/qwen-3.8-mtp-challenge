import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXFastCore
@testable import MLXLLM
@testable import MLXLMCommon

/// Op-level bit-exactness probes for the MTP verify-width wall.
///
/// The wall is a claim about ARITHMETIC, not about argmax: a wide verify
/// round must produce, for every row, the bit-identical value a serial
/// (one-row) round at the same absolute position would produce. Two ops can
/// break that, and they break it independently:
///
///   1. the sdpa, whose fused vector path serves `qL * gqa <= 32` and with
///      `gqa = 24 / 4 = 6` therefore stops at exactly `qL = 5`; and
///   2. the quantized projections, where MLX's own dispatch switches from
///      qmv to qmm at `M >= get_qmv_batch_limit(K, N)` and the candidate's
///      `Qwen35CustomQMV` replica only covers `2 ... MLX_QWEN_QMV_MAX_WIDTH`.
///
/// These tests isolate each op so a failure names the op instead of naming a
/// width. They are opt-in twice over: the runtime-tests gate plus
/// `MLXFAST_RUN_WIDTH_WALL_PROBE=1`, and they print a per-width verdict so a
/// SKIP cannot be mistaken for a PASS.
@Suite(.serialized)
struct QwenWidthWallProbeTests {

    private static var enabled: Bool {
        let env = ProcessInfo.processInfo.environment
        return env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
            && env["MLXFAST_RUN_WIDTH_WALL_PROBE"] == "1"
    }

    /// Bit patterns of a bfloat16/float array, as float32 words. Comparing
    /// these and not `==` keeps `-0.0` distinguishable from `0.0` and makes a
    /// NaN compare unequal to itself, which is what "bit-exact" means.
    private static func bits(_ a: MLXArray) -> [UInt32] {
        a.asType(.float32).asArray(Float.self).map { $0.bitPattern }
    }

    // MARK: - 1. the sdpa

    /// Feed `rows` query rows through `attentionWithCacheUpdate` one row at a
    /// time, which is exactly what a serial (depth-0) trajectory does, and
    /// return the concatenated bit patterns.
    private static func serialAttention(
        base: MLXArray, baseV: MLXArray,
        q: MLXArray, k: MLXArray, v: MLXArray, scale: Float
    ) -> [UInt32] {
        let cache = KVCacheSimple()
        _ = cache.update(keys: base, values: baseV)
        var out: [UInt32] = []
        for i in 0 ..< q.dim(2) {
            let y = attentionWithCacheUpdate(
                queries: q[0..., 0..., i ..< (i + 1), 0...],
                keys: k[0..., 0..., i ..< (i + 1), 0...],
                values: v[0..., 0..., i ..< (i + 1), 0...],
                cache: cache, scale: scale, mask: .causal)
            eval(y)
            out += bits(y.transposed(0, 2, 1, 3))
        }
        return out
    }

    private static func wideAttention(
        base: MLXArray, baseV: MLXArray,
        q: MLXArray, k: MLXArray, v: MLXArray, scale: Float
    ) -> [UInt32] {
        let cache = KVCacheSimple()
        _ = cache.update(keys: base, values: baseV)
        let y = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale,
            mask: .causal)
        eval(y)
        return bits(y.transposed(0, 2, 1, 3))
    }

    /// Feed `rows` query rows in consecutive blocks of `step` rows, which is
    /// what a session running at verify width `step` does.
    private static func steppedAttention(
        base: MLXArray, baseV: MLXArray,
        q: MLXArray, k: MLXArray, v: MLXArray, scale: Float, step: Int
    ) -> [UInt32] {
        let cache = KVCacheSimple()
        _ = cache.update(keys: base, values: baseV)
        var out: [UInt32] = []
        var start = 0
        while start < q.dim(2) {
            let end = Swift.min(start + step, q.dim(2))
            let y = attentionWithCacheUpdate(
                queries: q[0..., 0..., start ..< end, 0...],
                keys: k[0..., 0..., start ..< end, 0...],
                values: v[0..., 0..., start ..< end, 0...],
                cache: cache, scale: scale, mask: .causal)
            eval(y)
            out += bits(y.transposed(0, 2, 1, 3))
            start = end
        }
        return out
    }

    /// Largest absolute difference in float32 bit patterns (an ULP count for
    /// same-signed finite values), and the largest decimal gap.
    private static func compare(
        _ want: [UInt32], _ got: [UInt32], width: Int, perRow: Int
    ) -> (badRows: [Int], maxULP: UInt32, maxGap: Double) {
        var badRows: [Int] = []
        var maxULP: UInt32 = 0
        var maxGap = 0.0
        for r in 0 ..< width {
            let lo = r * perRow, hi = lo + perRow
            var bad = false
            for i in lo ..< hi where want[i] != got[i] {
                bad = true
                let a = want[i], b = got[i]
                maxULP = Swift.max(maxULP, a > b ? a - b : b - a)
                maxGap = Swift.max(
                    maxGap,
                    Double(abs(Float(bitPattern: a) - Float(bitPattern: b))))
            }
            if bad { badRows.append(r) }
        }
        return (badRows, maxULP, maxGap)
    }

    @Test("sdpa: how far a wide causal decode is from serial, and from width 5")
    func sdpaWidthExactness() throws {
        guard Self.enabled else { return }
        let qHeads = 24, kvHeads = 4, headDim = 256
        let perRow = qHeads * headDim
        let scale = 1.0 / Float(headDim).squareRoot()
        print("\n[WALL] wideDecodeExactnessMaxQueryRows = \(wideDecodeExactnessMaxQueryRows)")
        print("[WALL] Qwen35CustomQMV.maxWidth = \(Qwen35CustomQMV.maxWidth)")
        print("[WALL] sdpa  ctx  w  |  vs serial (w=1)      |  vs stepped w=5")
        for context in [64, 600] {
            for width in 1 ... 16 {
                MLXRandom.seed(20260828)
                let base = MLXRandom.normal([1, kvHeads, context, headDim])
                    .asType(.bfloat16)
                let baseV = MLXRandom.normal([1, kvHeads, context, headDim])
                    .asType(.bfloat16)
                let q = MLXRandom.normal([1, qHeads, width, headDim]).asType(.bfloat16)
                let k = MLXRandom.normal([1, kvHeads, width, headDim]).asType(.bfloat16)
                let v = MLXRandom.normal([1, kvHeads, width, headDim]).asType(.bfloat16)
                eval(base, baseV, q, k, v)
                let serial = Self.steppedAttention(
                    base: base, baseV: baseV, q: q, k: k, v: v, scale: scale, step: 1)
                let five = Self.steppedAttention(
                    base: base, baseV: baseV, q: q, k: k, v: v, scale: scale, step: 5)
                let wide = Self.wideAttention(
                    base: base, baseV: baseV, q: q, k: k, v: v, scale: scale)
                let s = Self.compare(serial, wide, width: width, perRow: perRow)
                let f = Self.compare(five, wide, width: width, perRow: perRow)
                print(String(
                    format: "[WALL] sdpa %4d %2d  |  %@ rows=%2d ulp=%3d gap=%.3e  |  %@ rows=%2d",
                    context, width,
                    s.badRows.isEmpty ? "EXACT" : "DRIFT", s.badRows.count,
                    Int(s.maxULP), s.maxGap,
                    f.badRows.isEmpty ? "EXACT" : "DRIFT", f.badRows.count))
            }
        }
        print("")
    }

    @Test("diag: what does .causal align to")
    func causalAlignment() throws {
        guard Self.enabled else { return }
        let qHeads = 24, kvHeads = 4, headDim = 256, context = 64, width = 5
        let scale = 1.0 / Float(headDim).squareRoot()
        MLXRandom.seed(1)
        let k = MLXRandom.normal([1, kvHeads, context + width, headDim]).asType(.bfloat16)
        let v = MLXRandom.normal([1, kvHeads, context + width, headDim]).asType(.bfloat16)
        let q = MLXRandom.normal([1, qHeads, width, headDim]).asType(.bfloat16)
        eval(k, v, q)
        let wide = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal)
        eval(wide)
        for r in 0 ..< width {
            let qr = q[0..., 0..., r ..< (r + 1), 0...]
            // bottom-right hypothesis: row r sees keys[0 ..< context + 1 + r]
            let br = MLXFast.scaledDotProductAttention(
                queries: qr, keys: k[0..., 0..., 0 ..< (context + 1 + r), 0...],
                values: v[0..., 0..., 0 ..< (context + 1 + r), 0...],
                scale: scale, mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
            // top-left hypothesis: row r sees keys[0 ..< r + 1]
            let tl = MLXFast.scaledDotProductAttention(
                queries: qr, keys: k[0..., 0..., 0 ..< (r + 1), 0...],
                values: v[0..., 0..., 0 ..< (r + 1), 0...],
                scale: scale, mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
            eval(br, tl)
            let want = Self.bits(wide[0..., 0..., r ..< (r + 1), 0...])
            let brBits = Self.bits(br), tlBits = Self.bits(tl)
            func gap(_ a: [UInt32], _ b: [UInt32]) -> Double {
                var m = 0.0
                for i in 0 ..< a.count {
                    m = Swift.max(m, Double(abs(
                        Float(bitPattern: a[i]) - Float(bitPattern: b[i]))))
                }
                return m
            }
            print(String(
                format: "[WALL] causal row=%d  bottomRight exact=%@ gap=%.3e  topLeft exact=%@ gap=%.3e",
                r, want == brBits ? "Y" : "n", gap(want, brBits),
                want == tlBits ? "Y" : "n", gap(want, tlBits)))
        }
        print("")
    }

    // MARK: - 2. the quantized projections

    @Test("qmm: a wide projection is bit-exact per row against M = 1")
    func projectionWidthExactness() throws {
        guard Self.enabled else { return }
        // The model's own transposed affine 4-bit group-64 shapes.
        let shapes: [(String, Int, Int)] = [
            ("fa.qkv      ", 5120, 8192),
            ("fa.o_proj   ", 6144, 5120),
            ("mlp.gate_up ", 5120, 34816),
            ("mlp.down    ", 17408, 5120),
            ("lm_head     ", 5120, 248320),
        ]
        for (name, K, N) in shapes {
            MLXRandom.seed(7)
            let w = MLXRandom.randInt(0 ..< Int32.max, [N, K * 4 / 32]).asType(.uint32)
            let scales = MLXRandom.normal([N, K / 64]).asType(.bfloat16)
            let biases = MLXRandom.normal([N, K / 64]).asType(.bfloat16)
            let x = MLXRandom.normal([1, 16, K]).asType(.bfloat16)
            eval(w, scales, biases, x)
            // Serial reference: one row at a time, which is the M = 1 dispatch.
            var want: [[UInt32]] = []
            for r in 0 ..< 16 {
                let y = qwen35RoutedQuantizedMM(
                    x[0..., r ..< (r + 1), 0...], w, scales: scales,
                    biases: biases, groupSize: 64, bits: 4, mode: .affine)
                eval(y)
                want.append(Self.bits(y))
            }
            for M in 2 ... 16 {
                let y = qwen35RoutedQuantizedMM(
                    x[0..., 0 ..< M, 0...], w, scales: scales, biases: biases,
                    groupSize: 64, bits: 4, mode: .affine)
                eval(y)
                let got = Self.bits(y)
                var badRows: [Int] = []
                for r in 0 ..< M {
                    if Array(got[r * N ..< (r + 1) * N]) != want[r] { badRows.append(r) }
                }
                print(String(
                    format: "[WALL] proj %@ M=%2d  %@  badRows=%@",
                    name, M, badRows.isEmpty ? "EXACT" : "DRIFT",
                    badRows.isEmpty ? "-" : "\(badRows.count) of \(M)"))
            }
        }
        print("")
    }
}
