import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXFastCore

/// Does MLX `compile` recover the launch overhead that dominates decode?
///
/// `DecodeBreakdownTests` found a 59.41 ms decode step is launch-bound: two RMS
/// norms per layer cost 17 percent while `lm_head`, the largest GEMM in the
/// step, costs 1.6 percent. `CompiledDecode` in MLXLMCommon exists for exactly
/// this and Qwen4Exp never calls it, but its `eligible(_:)` gate requires
/// `Compilable*` cache types and Qwen4Exp returns `Qwen4ExpAttnCache`,
/// `ArraysCache` and `MambaCache`.
///
/// The fixed-shape blocks need no cache and no facility. This measures what
/// plain `compile` buys on each of them at one row, which is the unblocked half
/// of that work and decides whether wiring the rest is worth it.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test -c release \
///       --force-resolved-versions --filter DecodeCompileGain
final class DecodeCompileGainTests: XCTestCase {
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

    private func q4(_ out: Int, _ inDim: Int) -> (MLXArray, MLXArray, MLXArray) {
        let w = MLXRandom.normal([out, inDim]).asType(.float16)
        eval(w)
        let (wq, s, b) = quantized(w, groupSize: 32, bits: 4)
        eval(wq, s, b!)
        return (wq, s, b!)
    }

    func testCompileGainOnFixedShapeBlocks() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "needs a GPU")

        let hidden = 2560, heads = 24, kvHeads = 2, headDim = 256, moeInter = 640
        let decodeStepMs = 59.41
        let x = MLXRandom.normal([1, hidden]).asType(.float16)
        let normW = MLXRandom.normal([hidden]).asType(.float16)
        eval(x, normW)

        func qmm(_ t: (MLXArray, MLXArray, MLXArray), _ v: MLXArray) -> MLXArray {
            quantizedMatmul(
                v, t.0, scales: t.1, biases: t.2, transpose: true, groupSize: 32, bits: 4)
        }

        var rows = [(String, Double, Double, Int)]()  // name, eager, compiled, instances

        // --- two RMS norms, the surprise of the breakdown at 17 percent ---
        let normEager: (MLXArray) -> MLXArray = { v in
            let a = v * rsqrt((v * v).mean(axis: -1, keepDims: true) + 1e-6) * normW
            return a * rsqrt((a * a).mean(axis: -1, keepDims: true) + 1e-6) * normW
        }
        let normCompiled = compile(normEager)
        rows.append((
            "rms_norm x2", time { eval(normEager(x)) }, time { eval(normCompiled(x)) }, 48))

        // --- full-attention projections ---
        let wq = q4(heads * headDim, hidden), wk = q4(kvHeads * headDim, hidden)
        let wv = q4(kvHeads * headDim, hidden), wo = q4(hidden, heads * headDim)
        let attnEager: (MLXArray) -> MLXArray = { v in
            let q = qmm(wq, v), k = qmm(wk, v), val = qmm(wv, v)
            return qmm(wo, q + 0 * k.sum() + 0 * val.sum())
        }
        let attnCompiled = compile(attnEager)
        rows.append((
            "full_attn projections", time { eval(attnEager(x)) },
            time { eval(attnCompiled(x)) }, 12))

        // --- linear-attention projections ---
        let lq = q4(16 * 128, hidden), lk = q4(16 * 128, hidden)
        let lv = q4(48 * 128, hidden), lo = q4(hidden, 48 * 128)
        let linEager: (MLXArray) -> MLXArray = { v in
            let a = qmm(lq, v), b = qmm(lk, v), c = qmm(lv, v)
            return qmm(lo, c + 0 * a.sum() + 0 * b.sum())
        }
        let linCompiled = compile(linEager)
        rows.append((
            "linear_attn projections", time { eval(linEager(x)) },
            time { eval(linCompiled(x)) }, 36))

        // --- shared expert: gate, up, silu-style product, down ---
        let eGate = q4(moeInter, hidden), eUp = q4(moeInter, hidden)
        let eDown = q4(hidden, moeInter)
        let sharedEager: (MLXArray) -> MLXArray = { v in
            let g = qmm(eGate, v), u = qmm(eUp, v)
            return qmm(eDown, g * u)
        }
        let sharedCompiled = compile(sharedEager)
        rows.append((
            "shared expert", time { eval(sharedEager(x)) },
            time { eval(sharedCompiled(x)) }, 48))

        // --- the router: fixed shape at one row until the selection is used ---
        let gate = MLXRandom.normal([hidden, 512]).asType(.bfloat16)
        eval(gate)
        let routerEager: (MLXArray) -> MLXArray = { v in
            let l = matmul(v.asType(.float32), gate.asType(.float32))
            return MLX.softmax(MLX.top(l, k: 10, axis: -1), axis: -1)
        }
        let routerCompiled = compile(routerEager)
        rows.append((
            "router", time { eval(routerEager(x)) }, time { eval(routerCompiled(x)) }, 48))

        print("[compile-gain] one row, against a \(decodeStepMs)ms decode step")
        var savedTotal = 0.0
        for (name, eager, comp, n) in rows {
            let saved = (eager - comp) * Double(n)
            savedTotal += saved
            let pct = eager > 0 ? (eager - comp) / eager * 100 : 0
            print(
                "  \(name.padding(toLength: 24, withPad: " ", startingAt: 0)) "
                    + "eager \(String(format: "%6.3f", eager))ms  "
                    + "compiled \(String(format: "%6.3f", comp))ms  "
                    + "(\(String(format: "%+5.1f", -pct))%)  "
                    + "x\(String(format: "%2d", n)) saves \(String(format: "%6.2f", saved))ms")
        }
        print(
            "  ---- total saved \(String(format: "%.1f", savedTotal))ms of "
                + "\(decodeStepMs)ms = \(String(format: "%.1f", savedTotal / decodeStepMs * 100))%")
    }
}
