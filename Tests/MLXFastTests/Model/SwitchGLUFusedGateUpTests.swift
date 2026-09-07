// Fused gate/up gather-GEMM for the stacked MoE experts (SonicMoE,
// arXiv 2512.14080): one gathered matmul over a `[E, 2 * I, H]` stack instead
// of two over `[E, I, H]` each.
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SwitchGLUFusedGateUpTests: XCTestCase {
    let inputDims = 64
    let hiddenDims = 64
    let numExperts = 8

    /// Two `SwitchGLU` instances holding the SAME weights: one unfused, one
    /// with the fused gate/up stack forced on.
    func makePair(quantized: Bool, numExperts: Int? = nil) -> (SwitchGLU, SwitchGLU) {
        let e = numExperts ?? self.numExperts
        let a = SwitchGLU(inputDims: inputDims, hiddenDims: hiddenDims, numExperts: e)
        let b = SwitchGLU(inputDims: inputDims, hiddenDims: hiddenDims, numExperts: e)
        if quantized {
            quantize(model: a) { _, _ in (32, 4, .affine) }
            quantize(model: b) { _, _ in (32, 4, .affine) }
        }
        b.update(parameters: ModuleParameters.unflattened(a.parameters().flattened()))
        a.forceFuseGateUp = false
        b.forceFuseGateUp = true
        XCTAssertNil(a.fusedGateUp(), "control must retain separate gate/up stacks")
        return (a, b)
    }

    /// Deliberately lopsided routing: expert 0 takes most rows, some experts
    /// take none. Shape `[1, tokens, k]`.
    func skewedIndices(tokens: Int, k: Int, experts: Int) -> MLXArray {
        var flat = [Int32]()
        flat.reserveCapacity(tokens * k)
        for t in 0 ..< tokens {
            for j in 0 ..< k {
                // t*j spread modulo a shrinking range so low experts dominate.
                flat.append(Int32((t &* (j &+ 1)) % max(1, experts - (j % 3)) % experts))
            }
        }
        return MLXArray(flat).reshaped(1, tokens, k)
    }

    func assertExactlyEqual(_ lhs: MLXArray, _ rhs: MLXArray, _ what: String) {
        XCTAssertEqual(lhs.shape, rhs.shape, what)
        eval(lhs, rhs)
        XCTAssertTrue((lhs .== rhs).all().item(Bool.self), "\(what): not bit-identical")
    }

    // MARK: - Numeric equality

    func testFusedEqualsUnfusedQuantizedShortNoSort() {
        // indices.size < 64 -> the un-sorted gather path.
        let (a, b) = makePair(quantized: true)
        let idx = skewedIndices(tokens: 5, k: 4, experts: numExperts)
        XCTAssertLessThan(idx.size, 64)
        let x = MLXRandom.normal([1, 5, inputDims])
        assertExactlyEqual(b(x, idx), a(x, idx), "quantized, unsorted")
        // The fused build releases the unfused stacks (1x1x1 placeholders).
        XCTAssertEqual(b.gateProj?.denseExpertWeight(0).shape, [1, 1])
        XCTAssertEqual(b.upProj?.denseExpertWeight(0).shape, [1, 1])
        XCTAssertEqual(a.gateProj?.denseExpertWeight(0).shape, [hiddenDims, inputDims])
    }

    func testFusedEqualsUnfusedQuantizedSortedPath() {
        // indices.size >= 64 -> gatherSort / scatterUnsort path.
        let (a, b) = makePair(quantized: true)
        let idx = skewedIndices(tokens: 33, k: 4, experts: numExperts)
        XCTAssertGreaterThanOrEqual(idx.size, 64)
        let x = MLXRandom.normal([1, 33, inputDims])
        assertExactlyEqual(b(x, idx), a(x, idx), "quantized, sorted")
    }

    func testFusedEqualsUnfusedDense() {
        let (a, b) = makePair(quantized: false)
        let idx = skewedIndices(tokens: 33, k: 4, experts: numExperts)
        let x = MLXRandom.normal([1, 33, inputDims])
        assertExactlyEqual(b(x, idx), a(x, idx), "dense")
    }

    func testFusedStackIsQuantizedAndDoubleWidth() {
        let (_, b) = makePair(quantized: true)
        let idx = skewedIndices(tokens: 5, k: 4, experts: numExperts)
        _ = b(MLXRandom.normal([1, 5, inputDims]), idx)
        guard let fused = b.fusedGateUp() else { return XCTFail("no fused stack") }
        XCTAssertTrue(fused is QuantizedSwitchLinear, "fused stack must stay quantized")
        // Double-width output rows, group axis intact: dequantizing one expert
        // yields [2 * hiddenDims, inputDims].
        XCTAssertEqual(fused.denseExpertWeight(0).shape, [2 * hiddenDims, inputDims])
    }

    /// The tower's real per-expert geometry (hidden 2560, moe_intermediate 640,
    /// 4-bit affine group-32), with the expert count cut to keep the test small.
    /// Guards against the fused stack's wider output (1280 rows) selecting a
    /// different gather-GEMM tiling than the 640-row unfused calls.
    func testFusedEqualsUnfusedAtTowerGeometry() {
        let hidden = 2560
        let inter = 640
        let experts = 4
        let a = SwitchGLU(inputDims: hidden, hiddenDims: inter, numExperts: experts)
        let b = SwitchGLU(inputDims: hidden, hiddenDims: inter, numExperts: experts)
        quantize(model: a) { _, _ in (32, 4, .affine) }
        quantize(model: b) { _, _ in (32, 4, .affine) }
        b.update(parameters: ModuleParameters.unflattened(a.parameters().flattened()))
        a.forceFuseGateUp = false
        b.forceFuseGateUp = true
        XCTAssertNil(a.fusedGateUp(), "control must retain separate gate/up stacks")
        let idx = skewedIndices(tokens: 40, k: 4, experts: experts)
        let x = MLXRandom.normal([1, 40, hidden])
        assertExactlyEqual(b(x, idx), a(x, idx), "tower geometry")
    }

    // MARK: - Shape at the tower's k

    func testShapeAtTopK10() {
        let experts = 512
        let a = SwitchGLU(inputDims: inputDims, hiddenDims: hiddenDims, numExperts: experts)
        a.forceFuseGateUp = true
        let idx = skewedIndices(tokens: 7, k: 10, experts: experts)
        let out = a(MLXRandom.normal([1, 7, inputDims]), idx)
        XCTAssertEqual(out.shape, [1, 7, 10, inputDims])
        eval(out)
    }

    // MARK: - Offload lane still reaches the per-expert weights

    func testDenseGateUpDownSurvivesFusion() {
        let (a, b) = makePair(quantized: true)
        let idx = skewedIndices(tokens: 5, k: 4, experts: numExperts)
        _ = b(MLXRandom.normal([1, 5, inputDims]), idx)
        XCTAssertEqual(b.gateProj?.denseExpertWeight(0).shape, [1, 1])
        for expert in [0, 3, numExperts - 1] {
            guard let (gu, uu, du) = a.denseGateUpDown(expert: expert),
                let (gf, uf, df) = b.denseGateUpDown(expert: expert)
            else { return XCTFail("denseGateUpDown returned nil") }
            XCTAssertEqual(gu.shape, [hiddenDims, inputDims])
            assertExactlyEqual(gf, gu, "gate expert \(expert)")
            assertExactlyEqual(uf, uu, "up expert \(expert)")
            assertExactlyEqual(df, du, "down expert \(expert)")
        }
    }
}
