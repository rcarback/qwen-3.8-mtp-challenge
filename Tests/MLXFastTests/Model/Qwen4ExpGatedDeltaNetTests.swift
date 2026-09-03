import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen4ExpGatedDeltaNetTests: XCTestCase {
    func tinyArgs() -> Qwen4ExpTextConfiguration {
        var a = Qwen4ExpTextConfiguration()
        a.hiddenSize = 16
        a.linearNumKeyHeads = 2
        a.linearNumValueHeads = 4
        // the vendored gated-delta kernel needs head dims that are multiples of 32
        a.linearKeyHeadDim = 32
        a.linearValueHeadDim = 32
        a.linearConvKernelDim = 4
        return a
    }

    func testShapesAndParameterKeys() {
        let m = Qwen4ExpGatedDeltaNet(tinyArgs())
        let keys = Set(m.parameters().flattened().map { $0.0 })
        for k in [
            "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight",
            "conv1d.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight",
        ] {
            XCTAssertTrue(keys.contains(k), k)
        }
        // convDim = 2*keyDim + valueDim = 2*64 + 128
        XCTAssertEqual(m.inProjQKV.weight.shape, [256, 16])
        XCTAssertEqual(m(MLXRandom.normal([1, 5, 16]), mask: nil, cache: nil).shape, [1, 5, 16])
    }

    func testStepwiseMatchesPrefill() {
        let m = Qwen4ExpGatedDeltaNet(tinyArgs())
        let x = MLXRandom.normal([1, 6, 16])
        let full = m(x, mask: nil, cache: nil)
        let cache = MambaCache()
        var outs = [MLXArray]()
        for t in 0 ..< 6 {
            outs.append(m(x[0..., t ..< (t + 1), 0...], mask: nil, cache: cache))
        }
        XCTAssertEqual(cache.offset, 6)
        XCTAssertTrue(allClose(concatenated(outs, axis: 1), full, rtol: 1e-2, atol: 1e-3).item())
    }
}
