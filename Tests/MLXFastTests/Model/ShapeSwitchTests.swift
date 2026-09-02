import Foundation
import MLX
import MLXRandom
import Testing

/// Confirms the mechanism behind the chained-GEMM dilution (§31): is the ~1.5x
/// residual (beyond per-shape ceiling and dependency) caused by SWITCHING
/// between different GEMM shapes -- Metal pipeline-state changes + allocator
/// churn -- rather than by the shapes themselves?
///
/// Same shapes, same FLOP, same chain depth; only the ORDER differs:
///   ALTERNATE: s1,s2,s1,s2,...  (switches shape every op)
///   GROUPED:   s1 x N, then s2 x N  (one switch)
/// ratio = alt_time / grp_time. > 1 => grouped is faster => shape-switching is
/// the cost, and grouping/pipeline-caching is the fix. ~1 => order is free and
/// the residual is something else. Paired per cycle so host noise divides out.
@Suite(.serialized)
struct ShapeSwitchTests {
    private struct Proj {
        let wq: MLXArray, scales: MLXArray, biases: MLXArray, out: Int, inn: Int
        init(out: Int, inn: Int) {
            let w = MLXRandom.normal([out, inn]).asType(.bfloat16)
            let (q, s, b) = quantized(w, groupSize: 64, bits: 4)
            wq = q; scales = s; biases = b ?? s; self.out = out; self.inn = inn
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            quantizedMM(x, wq, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4)
        }
    }

    @Test("shape-switch cost: alternating vs grouped GEMM shapes")
    func shapeSwitchCost() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        var report = "# Shape-switch cost: alternating vs grouped GEMM shapes\n\n"
        report += "Same shapes/FLOP/chain-depth, only order differs. ratio = alt/grp. >1 => grouping " +
            "(fewer pipeline-state switches + less allocator churn) is faster => shape-switching is the " +
            "§31 dilution residual, and grouping same-shape projections is the fix.\n\n"
        let reportPath = env["SHAPE_SWITCH_REPORT"] ??
            "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/" +
            "3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/shape-switch-report.md"
        func flush() { try? report.write(toFile: reportPath, atomically: true, encoding: .utf8) }
        defer { flush() }

        let S = 512, hidden = 5120
        let s1 = Proj(out: 10240, inn: hidden) // qkv shape
        let s2 = Proj(out: 17408, inn: hidden) // gate/up shape
        let x = MLXRandom.normal([1, S, hidden]).asType(.bfloat16)
        eval(x)
        let N = 8

        // Chain a sequence: each op's [1,S,out] output sliced back to [1,S,hidden]
        // to feed the next, so all sequences have identical dependency depth and
        // ONLY the shape-switch pattern differs.
        func runSeq(_ ops: [Proj]) -> Double {
            func body() -> MLXArray {
                var cur = x
                for op in ops { cur = op(cur)[0..., 0..., 0 ..< hidden] }
                return cur
            }
            eval(body())
            var best = Double.infinity
            for _ in 0 ..< 3 { let t = Date(); eval(body()); best = Swift.min(best, Date().timeIntervalSince(t)) }
            return best
        }

        let alternating = (0 ..< N).flatMap { _ in [s1, s2] }
        let grouped = Array(repeating: s1, count: N) + Array(repeating: s2, count: N)

        report += "| cycle | alt ms | grouped ms | alt/grp ratio |\n|---|---|---|---|\n"
        var ratios: [Double] = []
        for c in 0 ..< 6 {
            let alt = runSeq(alternating) * 1000
            let grp = runSeq(grouped) * 1000
            let r = grp > 0 ? alt / grp : 0
            ratios.append(r)
            report += String(format: "| %d | %.3f | %.3f | %.3f |\n", c, alt, grp, r)
            flush()
        }
        let mean = ratios.reduce(0, +) / Double(ratios.count)
        let sorted = ratios.sorted()
        report += String(format: "\n- [MEASURED] mean alt/grp ratio = %.3f (min %.3f, max %.3f) over %d cycles\n",
                         mean, sorted.first ?? 0, sorted.last ?? 0, ratios.count)
        report += "\nRead: ratio > ~1.1 => shape-switching is a real cost; grouping same-shape projections " +
            "and/or caching Metal pipeline states recovers it (the §31 lever). ratio ~1.0 => order is free.\n"
    }
}
