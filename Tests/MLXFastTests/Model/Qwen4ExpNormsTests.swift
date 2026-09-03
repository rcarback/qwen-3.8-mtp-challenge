import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen4ExpNormsTests: XCTestCase {
    func testZeroCentredNormIsPlainRMSNormAtZeroWeight() {
        let n = Qwen4ExpRMSNorm(dimensions: 8, eps: 1e-6)
        let x = MLXRandom.normal([2, 3, 8])
        let want = MLXFast.rmsNorm(x, weight: MLXArray.ones([8]), eps: 1e-6)
        XCTAssertTrue(allClose(n(x), want, atol: 1e-5).item())
    }

    func testGroupedNormNormalisesEachStream() {
        let n = Qwen4ExpRMSNorm(dimensions: 8, groupSize: 4, eps: 1e-6)
        n.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXArray(converting: [0, 0, 0, 0, 1, 1, 1, 1])
            ]))
        var x = MLXArray.zeros([1, 1, 8])
        x[0, 0, 0] = MLXArray(3.0)
        x[0, 0, 4] = MLXArray(300.0)
        let y = n(x)
        // rms of [3,0,0,0] is 1.5; 3/1.5 * (1+0) = 2
        XCTAssertEqual(y[0, 0, 0].item(Float.self), 2.0, accuracy: 1e-3)
        // rms of [300,0,0,0] is 150; 300/150 * (1+1) = 4
        XCTAssertEqual(y[0, 0, 4].item(Float.self), 4.0, accuracy: 1e-3)
    }

    func testGatedNormSigmoid() {
        let n = Qwen4ExpRMSNormGated(dimensions: 4, eps: 1e-6, gate: "sigmoid")
        let x = MLXRandom.normal([1, 2, 4])
        let g = MLXRandom.normal([1, 2, 4])
        let want =
            sigmoid(g.asType(.float32))
            * MLXFast.rmsNorm(x, weight: MLXArray.ones([4]), eps: 1e-6).asType(.float32)
        XCTAssertTrue(allClose(n(x, gate: g), want.asType(x.dtype), atol: 1e-5).item())
    }

    func testGatedResidualShapesAndCombine() {
        var args = Qwen4ExpTextConfiguration()
        args.hiddenSize = 8
        args.hcCount = 4
        args.hcLowrank = 3
        let gr = Qwen4ExpGatedResidual(args, combine: true)
        let hyper = MLXRandom.normal([2, 5, 32])
        let (mixed, inject) = gr.mix(hyper)
        XCTAssertEqual(mixed.shape, [2, 5, 8])
        XCTAssertEqual(inject!.shape, [2, 5, 4])
        let branch = MLXRandom.normal([2, 5, 8])
        let out = gr.combine(hyper, branch: branch, inject: inject!)
        XCTAssertEqual(out.shape, [2, 5, 32])
        // stream k of the output is hyper stream k + inject[k] * branch
        let k = 2
        let want =
            hyper[0..., 0..., (k * 8) ..< ((k + 1) * 8)] + inject![0..., 0..., k ..< (k + 1)] * branch
        XCTAssertTrue(allClose(out[0..., 0..., (k * 8) ..< ((k + 1) * 8)], want, atol: 1e-5).item())
        let mixer = Qwen4ExpGatedResidual(args, combine: false)
        XCTAssertNil(mixer.mix(hyper).inject)
    }
}
