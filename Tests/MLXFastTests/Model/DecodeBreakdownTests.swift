import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXFastCore

/// Where does a 59.41 ms decode step go?
///
/// `DecodeStepCostTests` measured the whole step; the router accounts for about
/// 8.6 ms of it and nothing accounted for the rest. This times each major
/// component at DECODE geometry, one row, with the real shapes from
/// config.json: hidden 2560, 24 query heads over 2 KV heads at head_dim 256,
/// 512 experts of which 10 route per token, moe_intermediate 640, vocab
/// 248320, 48 layers on a 4-layer repeat so 12 full-attention and 36 linear.
///
/// EVERY component is timed CHAINED, one eval() around the whole block, because
/// timing sub-steps separately forces a GPU submission each and inflates the
/// result. That mistake put the router at 38.1 ms per token when the in-situ
/// figure is 8.6 ms.
///
/// Run-to-run spread is about 25 percent on the per-instance figures even with
/// warmup, so the ORDERING is the durable result and the absolute values are
/// not. Two warmed runs gave routed MoE 0.459 and 0.360 ms, rms_norm 0.233 and
/// 0.203 ms, and identical ordering. `lm_head`, one large op, is stable at
/// 0.94 to 0.95 ms.
///
/// These are component floors, not a decomposition: the sum need not equal the
/// whole, because in situ these blocks share submissions and overlap. Read the
/// ordering and the magnitudes, not the total.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test -c release \
///       --force-resolved-versions --filter DecodeBreakdown
final class DecodeBreakdownTests: XCTestCase {
    private func time(_ body: () -> Void) -> Double {
        body()
        var best = Double.greatestFiniteMagnitude
        for _ in 0 ..< 5 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< 20 { body() }
            best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 / 20)
        }
        return best
    }

    /// A 4-bit affine group-32 weight of the given shape, as production stores
    /// every projection in this model.
    private func q4(_ out: Int, _ inDim: Int) -> (MLXArray, MLXArray, MLXArray) {
        let w = MLXRandom.normal([out, inDim]).asType(.float16)
        eval(w)
        let (wq, s, b) = quantized(w, groupSize: 32, bits: 4)
        eval(wq, s, b!)
        return (wq, s, b!)
    }

    func testDecodeComponentCosts() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "needs a GPU")

        let hidden = 2560, heads = 24, kvHeads = 2, headDim = 256
        let vocab = 248320, moeInter = 640, topK = 10
        let fullLayers = 12, linearLayers = 36, layers = 48
        let decodeStepMs = 59.41  // measured, DecodeStepCostTests

        let x = MLXRandom.normal([1, hidden]).asType(.float16)
        eval(x)

        // Warm the kernel families this test uses BEFORE timing any of them.
        // Without this the first component measured absorbs process warmup: in
        // the router test the same quantity read 0.320 ms as the first arm and
        // 0.180 ms once warmed, which is a 1.8x error in whichever component
        // happens to be timed first.
        do {
            let ww = MLXRandom.normal([512, hidden]).asType(.float16)
            eval(ww)
            let (wq0, s0, b0) = quantized(ww, groupSize: 32, bits: 4)
            eval(wq0, s0, b0!)
            let nw = MLXRandom.normal([hidden]).asType(.float16)
            eval(nw)
            for _ in 0 ..< 20 {
                eval(
                    quantizedMatmul(
                        x, wq0, scales: s0, biases: b0, transpose: true, groupSize: 32, bits: 4))
                eval(x * rsqrt((x * x).mean(axis: -1, keepDims: true) + 1e-6) * nw)
            }
        }

        func qmm(_ t: (MLXArray, MLXArray, MLXArray), _ v: MLXArray) -> MLXArray {
            quantizedMatmul(v, t.0, scales: t.1, biases: t.2, transpose: true,
                groupSize: 32, bits: 4)
        }

        var report = [(String, Double, Int)]()  // name, ms per instance, instances

        // --- lm_head: one row against a 248320 vocab, the largest single GEMM
        // in a decode step and run exactly once per token.
        let lmHead = q4(vocab, hidden)
        report.append(("lm_head", time { eval(qmm(lmHead, x)) }, 1))

        // --- full attention projections: q [24*256], k/v [2*256], o back to
        // hidden. SDPA at one query row against a short cache is negligible
        // beside the projections, so this is the projection cost.
        let wq = q4(heads * headDim, hidden)
        let wk = q4(kvHeads * headDim, hidden)
        let wv = q4(kvHeads * headDim, hidden)
        let wo = q4(hidden, heads * headDim)
        report.append((
            "full_attn projections",
            time {
                let q = qmm(wq, x), k = qmm(wk, x), v = qmm(wv, x)
                let ctx = q + 0 * k.sum() + 0 * v.sum()  // keep k,v live
                eval(qmm(wo, ctx))
            }, fullLayers))

        // --- gated delta (linear attention): 16 key heads and 48 value heads
        // at head_dim 128, plus a conv of kernel 4. Approximated by its
        // projections, which dominate at one row.
        let lq = q4(16 * 128, hidden)
        let lk = q4(16 * 128, hidden)
        let lv = q4(48 * 128, hidden)
        let lo = q4(hidden, 48 * 128)
        report.append((
            "linear_attn projections",
            time {
                let a = qmm(lq, x), b = qmm(lk, x), c = qmm(lv, x)
                let ctx = c + 0 * a.sum() + 0 * b.sum()
                eval(qmm(lo, ctx))
            }, linearLayers))

        // --- routed MoE at one row: 10 experts, each gate+up+down. Production
        // uses a gather-GEMM; at one row this is the work it gathers.
        let eGate = q4(moeInter, hidden)
        let eUp = q4(moeInter, hidden)
        let eDown = q4(hidden, moeInter)
        report.append((
            "routed MoE (10 experts)",
            time {
                var acc = MLXArray.zeros([1, hidden]).asType(.float16)
                for _ in 0 ..< topK {
                    let g = qmm(eGate, x), u = qmm(eUp, x)
                    acc = acc + qmm(eDown, g * u)
                }
                eval(acc)
            }, layers))

        // --- shared expert: same shape, one instance per layer.
        report.append((
            "shared expert",
            time {
                let g = qmm(eGate, x), u = qmm(eUp, x)
                eval(qmm(eDown, g * u))
            }, layers))

        // --- rms norm, two per layer.
        let normW = MLXRandom.normal([hidden]).asType(.float16)
        eval(normW)
        report.append((
            "rms_norm x2",
            time {
                let a = x * rsqrt((x * x).mean(axis: -1, keepDims: true) + 1e-6) * normW
                eval(a * rsqrt((a * a).mean(axis: -1, keepDims: true) + 1e-6) * normW)
            }, layers))

        print("[decode-breakdown] one row, chained, against a \(decodeStepMs)ms step")
        var accounted = 0.0
        for (name, ms, n) in report.sorted(by: { $0.1 * Double($0.2) > $1.1 * Double($1.2) }) {
            let total = ms * Double(n)
            accounted += total
            print(
                "  \(name.padding(toLength: 24, withPad: " ", startingAt: 0)) "
                    + "\(String(format: "%7.3f", ms))ms x\(String(format: "%3d", n)) = "
                    + "\(String(format: "%6.1f", total))ms  "
                    + "\(String(format: "%5.1f", total / decodeStepMs * 100))%")
        }
        print(
            "  router (measured separately)                    =    8.6ms   14.5%")
        accounted += 8.6
        print(
            "  ---- accounted \(String(format: "%.1f", accounted))ms of \(decodeStepMs)ms "
                + "= \(String(format: "%.0f", accounted / decodeStepMs * 100))%")
    }
}
