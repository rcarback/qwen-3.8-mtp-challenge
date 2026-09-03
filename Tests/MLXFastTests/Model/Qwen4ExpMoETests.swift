import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen4ExpMoETests: XCTestCase {
    func tinyArgs() -> Qwen4ExpTextConfiguration {
        var a = Qwen4ExpTextConfiguration()
        a.hiddenSize = 8
        a.numExperts = 4
        a.numExpertsPerTok = 2
        a.moeIntermediateSize = 6
        a.sharedExpertIntermediateSize = 6
        return a
    }

    /// Naive per-token reference: run every selected expert densely and combine
    /// with the router's top-k softmax, then add the gated shared expert.
    func reference(_ m: Qwen4ExpSparseMoeBlock, _ x: MLXArray) -> MLXArray {
        let (idx, w) = m.route(x)
        let p = Dictionary(uniqueKeysWithValues: m.switchMLP.parameters().flattened())
        let gateW = p["gate_proj.weight"]!.asType(.float32)
        let upW = p["up_proj.weight"]!.asType(.float32)
        let downW = p["down_proj.weight"]!.asType(.float32)
        var out = MLXArray.zeros(like: x).asType(.float32)
        for b in 0 ..< x.dim(0) {
            for s in 0 ..< x.dim(1) {
                let xi = x[b, s].asType(.float32)
                for j in 0 ..< m.topK {
                    let e = Int(idx[b, s, j].item(Int32.self))
                    let h = silu(matmul(gateW[e], xi)) * matmul(upW[e], xi)
                    out[b, s] = out[b, s] + w[b, s, j] * matmul(downW[e], h)
                }
            }
        }
        let shared = sigmoid(m.sharedExpertGate(x)) * m.sharedExpert(x)
        return out.asType(x.dtype) + shared
    }

    func testMatchesNaiveReference() {
        let m = Qwen4ExpSparseMoeBlock(tinyArgs())
        let x = MLXRandom.normal([2, 3, 8])
        XCTAssertTrue(allClose(m(x), reference(m, x), rtol: 1e-3, atol: 1e-4).item())
    }

    func testQuantizedExpertsStayClose() {
        var a = tinyArgs()
        a.hiddenSize = 64
        a.moeIntermediateSize = 64
        a.sharedExpertIntermediateSize = 64
        let m = Qwen4ExpSparseMoeBlock(a)
        let x = MLXRandom.normal([1, 4, 64])
        let dense = m(x)
        quantize(model: m) { path, _ in path.contains("switch_mlp") ? (32, 4, .affine) : nil }
        let expertKeys = Set(m.switchMLP.parameters().flattened().map { $0.0 })
        XCTAssertTrue(expertKeys.contains("gate_proj.scales"))
        let sharedKeys = Set(m.sharedExpert.parameters().flattened().map { $0.0 })
        XCTAssertFalse(sharedKeys.contains("gate_proj.scales"))
        let q = m(x)
        // 4-bit group-32 on random normal weights lands near 10% relative error
        XCTAssertLessThan((abs(q - dense).mean() / abs(dense).mean()).item(Float.self), 0.2)
    }
}
