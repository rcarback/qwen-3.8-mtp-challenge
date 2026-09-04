import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM

/// The grouped routed-expert lane's gather/scatter plan. These tests are pure
/// MLX: the planner never touches the ANE, so they run everywhere and are not
/// gated behind `MLXFAST_RUN_MLX_RUNTIME_TESTS`. The ANE program itself
/// (`ANEGroupedExpertMLP`) is exercised by the gated test at the bottom.
final class Qwen4ExpANEExpertLaneTests: XCTestCase {
    // Deliberately asymmetric and small: hidden != inter, capacity below the
    // busiest expert's load, one hot expert with no tokens at all.
    private let hidden = 8
    private let inter = 5
    private let numExperts = 6
    private let topK = 2
    private let capacity = 3
    private let hotIds = [0, 1, 2, 3]  // experts 4 and 5 stay cold

    /// Hand-built routing, 12 tokens x top-2. Assignment counts per expert:
    /// 0 -> 5 (OVERFLOWS capacity 3), 1 -> 4 (overflows), 2 -> 2 (fits),
    /// 3 -> 0 (hot but idle), 4 -> 7, 5 -> 6 (both cold).
    private let routing: [[Int32]] = [
        [0, 4], [0, 5], [0, 4], [0, 5], [0, 4],
        [1, 4], [1, 5], [1, 4], [1, 5],
        [2, 4], [2, 5], [5, 4],
    ]

    private func slotTable() -> MLXArray {
        var table = [Int32](repeating: -1, count: numExperts)
        for (slot, expert) in hotIds.enumerated() { table[expert] = Int32(slot) }
        return MLXArray(table)
    }

    private func indices() -> MLXArray {
        MLXArray(routing.flatMap { $0 }).reshaped(routing.count, topK)
    }

    /// `[E, inter, hidden]` gate/up and `[E, hidden, inter]` down.
    private func makeExperts() -> (MLXArray, MLXArray, MLXArray) {
        MLXRandom.seed(7)
        let g = MLXRandom.normal([numExperts, inter, hidden]) / Float(hidden).squareRoot()
        let u = MLXRandom.normal([numExperts, inter, hidden]) / Float(hidden).squareRoot()
        let d = MLXRandom.normal([numExperts, hidden, inter]) / Float(inter).squareRoot()
        eval(g, u, d)
        return (g, u, d)
    }

    /// `x` is `[rows, hidden]`; returns expert `e`'s SwiGLU over every row.
    private func expertFFN(_ x: MLXArray, _ e: Int, _ w: (MLXArray, MLXArray, MLXArray)) -> MLXArray {
        let gate = matmul(x, w.0[e].transposed(1, 0))
        let up = matmul(x, w.1[e].transposed(1, 0))
        return matmul(silu(gate) * up, w.2[e].transposed(1, 0))
    }

    func testPlanRespectsCapacityAndIdleExperts() {
        let plan = Qwen4ExpANEExpertPlanner.plan(
            indices: indices(), slotOf: slotTable(), hotCount: hotIds.count, capacity: capacity)
        let handled = plan.handled.asType(.int32)
        eval(handled)
        let flatHandled = handled.reshaped(-1).asArray(Int32.self)
        let flatIndices = routing.flatMap { $0 }

        var served = [Int: Int]()
        for (n, expert) in flatIndices.enumerated() where flatHandled[n] == 1 {
            served[Int(expert), default: 0] += 1
        }
        XCTAssertEqual(served[0], capacity, "expert 0 has 5 assignments and must fill exactly C")
        XCTAssertEqual(served[1], capacity, "expert 1 has 4 assignments and must fill exactly C")
        XCTAssertEqual(served[2], 2, "expert 2 fits under C")
        XCTAssertNil(served[3], "hot expert 3 receives no tokens")
        XCTAssertNil(served[4], "cold expert must never be served")
        XCTAssertNil(served[5], "cold expert must never be served")

        // Every packed row either holds a real token or points at the sink.
        let source = plan.source
        eval(source)
        XCTAssertEqual(source.shape, [hotIds.count * capacity])
        let filled = source.asArray(Int32.self).filter { $0 < Int32(routing.count) }.count
        XCTAssertEqual(filled, capacity + capacity + 2)
    }

    /// The round trip that matters: gather -> per-slot expert FFN -> combine,
    /// plus the GPU leg over the assignments the plan did NOT serve, must
    /// reproduce the plain MoE routed sum exactly.
    func testGroupedRoundTripMatchesPureGPURoutedSum() {
        let w = makeExperts()
        let idx = indices()
        let tokens = routing.count
        MLXRandom.seed(11)
        let x = MLXRandom.normal([tokens, hidden])
        let weights = MLX.softmax(MLXRandom.normal([tokens, topK]), axis: -1, precise: true)
        eval(x, weights)

        let plan = Qwen4ExpANEExpertPlanner.plan(
            indices: idx, slotOf: slotTable(), hotCount: hotIds.count, capacity: capacity)
        let packed = Qwen4ExpANEExpertPlanner.gather(x, plan: plan)
        XCTAssertEqual(packed.shape, [hotIds.count * capacity, hidden])

        // Stand-in for the ANE program: exact fp32 SwiGLU per capacity slice.
        var slices: [MLXArray] = []
        for (slot, expert) in hotIds.enumerated() {
            let rows = packed[(slot * capacity) ..< ((slot + 1) * capacity), 0...]
            slices.append(expertFFN(rows, expert, w))
        }
        let packedOut = concatenated(slices, axis: 0)

        let aneSum = Qwen4ExpANEExpertPlanner.combine(
            packedOut, plan: plan, weights: weights, tokens: tokens)

        // GPU leg: the same routed sum with the served assignments zeroed.
        let masked = weights * (1 - plan.handled.asType(weights.dtype))
        var gpuSum = MLXArray.zeros([tokens, hidden])
        var reference = MLXArray.zeros([tokens, hidden])
        for slot in 0 ..< topK {
            for t in 0 ..< tokens {
                let expert = Int(routing[t][slot])
                let row = expertFFN(x[t ..< (t + 1), 0...], expert, w)
                let full = row * weights[t ..< (t + 1), slot ..< (slot + 1)]
                let part = row * masked[t ..< (t + 1), slot ..< (slot + 1)]
                reference[t ..< (t + 1), 0...] = reference[t ..< (t + 1), 0...] + full
                gpuSum[t ..< (t + 1), 0...] = gpuSum[t ..< (t + 1), 0...] + part
            }
        }

        let got = gpuSum + aneSum
        eval(got, reference)
        XCTAssertTrue(
            allClose(got, reference, rtol: 1e-4, atol: 1e-5).item(),
            "grouped round trip diverged: max=\(MLX.abs(got - reference).max().item(Float.self))")
    }

    /// A zero-capacity-usable configuration must still be exact: with no hot
    /// experts the plan serves nothing and the GPU keeps every assignment.
    func testEmptyHotSetServesNothing() {
        let plan = Qwen4ExpANEExpertPlanner.plan(
            indices: indices(), slotOf: MLXArray([Int32](repeating: -1, count: numExperts)),
            hotCount: hotIds.count, capacity: capacity)
        let handled = plan.handled.asType(.int32).sum()
        eval(handled)
        XCTAssertEqual(handled.item(Int32.self), 0)
    }

    /// The lane must be completely inert with the mode unset. `mode` is read
    /// once from `MLX_QWEN4EXP_ANE_MODE`, which the test process never sets,
    /// so `experts` can never be selected here.
    func testLaneInertWhenModeUnset() {
        XCTAssertNil(ProcessInfo.processInfo.environment["MLX_QWEN4EXP_ANE_MODE"])
        XCTAssertNotEqual(Qwen4ExpANEFused.mode, .experts)
        XCTAssertFalse(Qwen4ExpANEFused.expertsEnabled)

        var a = Qwen4ExpTextConfiguration()
        a.hiddenSize = hidden
        a.numExperts = numExperts
        a.numExpertsPerTok = topK
        a.moeIntermediateSize = inter
        a.sharedExpertIntermediateSize = inter
        let block = Qwen4ExpSparseMoeBlock(a)
        let x = MLXRandom.normal([1, 16, hidden]).asType(.bfloat16)
        eval(x)

        // Same call twice: with the lane off nothing is built, cached or
        // frozen, so the forward is bit-identical.
        let first = block(x)
        let second = block(x)
        eval(first, second)
        XCTAssertTrue((first .== second).all().item(), "the lane-off forward must be deterministic")
        XCTAssertEqual(first.shape, [1, 16, hidden])
    }

    /// The grouped ANE program itself. Needs the private ANE frameworks, so it
    /// SKIPS rather than passes when the runtime tests are not enabled.
    func testGroupedProgramMatchesPerExpertGPU() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1")
        try XCTSkipUnless(ANERuntime.available())

        // Nothing in this tree compiles an ANE conv below 192 channels, so the
        // program test uses realistic widths rather than the 8/5 above.
        let h = 192
        let i = 320
        let group = 2
        let cap = 32
        MLXRandom.seed(3)
        var gate: [MLXArray] = []
        var up: [MLXArray] = []
        var down: [MLXArray] = []
        for _ in 0 ..< group {
            gate.append((MLXRandom.normal([i, h]) / Float(h).squareRoot()).asType(.bfloat16))
            up.append((MLXRandom.normal([i, h]) / Float(h).squareRoot()).asType(.bfloat16))
            down.append((MLXRandom.normal([h, i]) / Float(i).squareRoot()).asType(.bfloat16))
        }
        let packed = MLXRandom.normal([cap, group * h]).asType(.float16)
        eval(packed)

        let program = try ANEGroupedExpertMLP(gate: gate, up: up, down: down, capacity: cap)
        let got = try program(packed)
        XCTAssertEqual(got.shape, [cap, group * h])

        for g in 0 ..< group {
            let block = packed[0..., (g * h) ..< ((g + 1) * h)].asType(.float32)
            let want = matmul(
                silu(matmul(block, gate[g].asType(.float32).transposed(1, 0)))
                    * matmul(block, up[g].asType(.float32).transposed(1, 0)),
                down[g].asType(.float32).transposed(1, 0))
            let diff = MLX.abs(got[0..., (g * h) ..< ((g + 1) * h)].asType(.float32) - want)
            eval(diff)
            XCTAssertLessThan(diff.max().item(Float.self), 0.2, "expert \(g) block diverged")
        }
    }
}
