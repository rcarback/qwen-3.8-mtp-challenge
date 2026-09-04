import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen4ExpANESharedExpertTests: XCTestCase {
    /// Synthetic asymmetric shared expert, non-bucket-aligned S.
    /// hidden = 192, inter = 320, bucket 128, S = 100.
    func testSharedExpertMatchesPureGPUMLP() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1")
        try XCTSkipUnless(ANERuntime.available())

        let g = (MLXRandom.normal([320, 192]) / Float(192).squareRoot()).asType(.bfloat16)
        let u = (MLXRandom.normal([320, 192]) / Float(192).squareRoot()).asType(.bfloat16)
        let d = (MLXRandom.normal([192, 320]) / Float(320).squareRoot()).asType(.bfloat16)
        let x = MLXRandom.normal([100, 192]).asType(.bfloat16)
        eval(g, u, d, x)

        let prog = try Qwen4ExpANESharedExpert(gate: g, up: u, down: d, sequenceLength: 128)
        let got = try prog(x)
        XCTAssertEqual(got.shape, [100, 192])

        let x32 = x.asType(.float32)
        let g32 = g.asType(.float32)
        let u32 = u.asType(.float32)
        let d32 = d.asType(.float32)
        let want = matmul(
            silu(matmul(x32, g32.transposed(1, 0))) * matmul(x32, u32.transposed(1, 0)),
            d32.transposed(1, 0))
        eval(want)

        let diff = MLX.abs(got.asType(.float32) - want)
        eval(diff)
        let maxAbs = diff.max().item(Float.self)
        let meanAbs = diff.mean().item(Float.self)
        XCTAssertLessThan(maxAbs, 0.1, "maxAbs=\(maxAbs)")
        XCTAssertLessThan(meanAbs, 0.01, "meanAbs=\(meanAbs)")
    }

    /// Covers the `routed + sigmoid(gate) * shared` combine, which the
    /// weight-level test above does not reach. Realistic widths (192/320),
    /// not the 16-wide `Qwen4ExpModelTests.makeModel()` config: nothing in
    /// this tree compiles an ANE conv below 192 channels.
    func testCombineMatchesPlainBlock() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1")
        try XCTSkipUnless(ANERuntime.available())

        var a = Qwen4ExpTextConfiguration()
        a.hiddenSize = 192
        a.numExperts = 4
        a.numExpertsPerTok = 2
        a.moeIntermediateSize = 192
        a.sharedExpertIntermediateSize = 320
        let block = Qwen4ExpSparseMoeBlock(a)
        let x = MLXRandom.normal([1, 100, 192]).asType(.bfloat16)  // buckets to 128
        eval(x)

        let prog = try Qwen4ExpANESharedExpert(
            gate: block.sharedExpert.gateProj.weight,
            up: block.sharedExpert.upProj.weight,
            down: block.sharedExpert.downProj.weight,
            sequenceLength: 128)

        let offloaded = try block.offloadedForward(x, program: prog)
        let plain = block(x)
        eval(offloaded, plain)

        let diff = MLX.abs(offloaded.asType(.float32) - plain.asType(.float32))
        eval(diff)
        let maxAbs = diff.max().item(Float.self)
        XCTAssertLessThan(maxAbs, 0.05, "maxAbs=\(maxAbs)")
    }
}
