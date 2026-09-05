import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXFastCore

/// What does the MoE router actually cost, and is fusing it worth a kernel?
///
/// The router is a gate GEMM (hidden 2560 -> 512 experts), a top-k of 10, and
/// a softmax over the selected logits, once per layer for 48 layers. A fused
/// kernel would collapse those launches into one. Before writing that kernel,
/// this measures the three stages so the ceiling on fusing them is a number
/// rather than an estimate.
///
/// The baseline every speed claim in this repository is stated against is the
/// production routed MoE at 22.0 ms per layer at real geometry. Anything the
/// router costs is measured against that, never against a naive per-expert
/// loop, which flatters every result.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test -c release \
///       --force-resolved-versions --filter routerCost
final class RouterCostTests: XCTestCase {
    func testRouterStageCosts() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "needs a GPU")

        let hidden = 2560, experts = 512, topK = 10
        let moePerLayerMs = 22.0  // measured production baseline

        for tokens in [128, 700] {
            let x = MLXRandom.normal([tokens, hidden]).asType(.float16)
            // The router gate ships in bf16 per layer; MLX upcasts for the
            // softmax, which is one of the costs a fused kernel would remove.
            let gate = MLXRandom.normal([hidden, experts]).asType(.bfloat16)
            eval(x, gate)

            func time(_ label: String, _ body: () -> Void) -> Double {
                body()
                var best = Double.greatestFiniteMagnitude
                for _ in 0 ..< 5 {
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    for _ in 0 ..< 50 { body() }
                    best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 / 50)
                }
                return best
            }

            let gemmMs = time("gemm") {
                eval(matmul(x.asType(.float32), gate.asType(.float32)))
            }
            let logits = matmul(x.asType(.float32), gate.asType(.float32))
            eval(logits)
            let topkMs = time("topk") { eval(MLX.top(logits, k: topK, axis: -1)) }
            let sel = MLX.top(logits, k: topK, axis: -1)
            eval(sel)
            let softmaxMs = time("softmax") { eval(MLX.softmax(sel, axis: -1)) }

            let total = gemmMs + topkMs + softmaxMs
            let share = total / moePerLayerMs * 100
            print(
                "[router] tokens=\(tokens): gemm=\(String(format: "%.3f", gemmMs))ms "
                    + "topk=\(String(format: "%.3f", topkMs))ms "
                    + "softmax=\(String(format: "%.3f", softmaxMs))ms "
                    + "total=\(String(format: "%.3f", total))ms "
                    + "= \(String(format: "%.2f", share))% of the 22.0ms per-layer MoE")
        }
    }
}
