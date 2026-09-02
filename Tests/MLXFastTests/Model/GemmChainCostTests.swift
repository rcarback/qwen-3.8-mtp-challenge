import Foundation
import MLX
import MLXRandom
import Testing

/// Does a GEMM run as fast inside a dependent chain as it does alone?
///
/// Section 13 priced a gated-delta layer's GEMMs at the 13.5 TFLOPS measured
/// for ONE GEMM alone in a process, concluded the layer spends only half its
/// time on matrix work, and inferred the other half is low-intensity
/// operators. The scan and conv1d measurements then came in small -- together
/// about 12% of that bucket -- so the inference is in trouble and there are
/// two live explanations:
///
///   (a) the remaining non-GEMM operators (norms, gating, reshapes, dispatch)
///       really do cost ~2.2 s per chunk, or
///   (b) the GEMMs themselves run slower inside a real forward than they do
///       alone, and the "non-GEMM half" is partly an artifact of pricing them
///       at an unreachable rate.
///
/// This test runs a gated-delta layer's GEMM sequence with NOTHING else --
/// no norms, no activations, no recurrence -- so its aggregate rate is
/// directly comparable to the isolated single-GEMM number.
///
///   chain rate ~= 13.5 TFLOPS  -> (b) is dead, the operators are the cost
///   chain rate materially lower -> (b) is real, and section 13's split moves
///
/// One shape per process, per the position discipline in
/// `PrefillMatmulCostTests`.
@Suite(.serialized)
struct GemmChainCostTests {
    private static func timeIt(_ body: () -> [MLXArray]) -> Double {
        eval(body())
        var best = Double.infinity
        for _ in 0 ..< 3 {
            let start = Date()
            eval(body())
            best = Swift.min(best, Date().timeIntervalSince(start))
        }
        return best
    }

    /// One quantized projection: weights [out, inn], affine 4-bit group 64.
    private struct Proj {
        let wq: MLXArray
        let scales: MLXArray
        let biases: MLXArray
        let out: Int
        let inn: Int

        init(out: Int, inn: Int) {
            let w = MLXRandom.normal([out, inn]).asType(.bfloat16)
            let (q, s, b) = quantized(w, groupSize: 64, bits: 4)
            self.wq = q
            self.scales = s
            self.biases = b ?? s
            self.out = out
            self.inn = inn
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            quantizedMM(
                x, wq, scales: scales, biases: biases, transpose: true,
                groupSize: 64, bits: 4)
        }

        var flops: Double { 2.0 * Double(out) * Double(inn) }
    }

    @Test("gated-delta layer GEMM chain, alone in its process")
    func gemmChainPoint() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let S = Int(env["MLXFAST_CHAIN_S"] ?? "1024") ?? 1024
        let hidden = 5120
        let inter = 17408

        // The gated-delta layer's projection set, from weights/config.json.
        // in_proj_b and in_proj_a are folded into one 96-row projection; they
        // are 0.1% of the FLOPs and splitting them changes nothing.
        let qkv = Proj(out: 10240, inn: hidden)
        let z = Proj(out: 6144, inn: hidden)
        let ba = Proj(out: 96, inn: hidden)
        let outProj = Proj(out: hidden, inn: 6144)
        let gate = Proj(out: inter, inn: hidden)
        let up = Proj(out: inter, inn: hidden)
        let down = Proj(out: hidden, inn: inter)

        let x = MLXRandom.normal([1, S, hidden]).asType(.bfloat16)
        let mid = MLXRandom.normal([1, S, 6144]).asType(.bfloat16)
        let act = MLXRandom.normal([1, S, inter]).asType(.bfloat16)
        eval(x, mid, act)

        // Isolated: each projection timed on its own, summed. This is the
        // number section 13's arithmetic implicitly assumed.
        var isolatedSeconds = 0.0
        for (p, input) in [
            (qkv, x), (z, x), (ba, x), (outProj, mid), (gate, x), (up, x),
            (down, act),
        ] {
            isolatedSeconds += Self.timeIt { [p(input)] }
        }

        // Chained: the same projections, dependent, in one eval. Shapes are
        // bridged by slicing rather than by real ops so that ONLY GEMM work
        // is timed.
        let chained = Self.timeIt {
            let qkvOut = qkv(x)
            let zOut = z(x)
            let baOut = ba(x)
            let o = outProj(zOut[0..., 0..., 0 ..< 6144])
            let g = gate(o)
            let u = up(o)
            let d = down(g + u)
            return [d, qkvOut, baOut]
        }

        let totalFlops = Double(S)
            * (qkv.flops + z.flops + ba.flops + outProj.flops + gate.flops
                + up.flops + down.flops)
        print(String(
            format: "CHAINPOINT\t%d\t%.4f\t%.3f\t%.4f\t%.3f",
            S,
            1000 * isolatedSeconds, totalFlops / isolatedSeconds / 1e12,
            1000 * chained, totalFlops / chained / 1e12))
    }
}
