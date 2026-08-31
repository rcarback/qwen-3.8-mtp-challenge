import Foundation
import MLX
import MLXFastModel
import MLXRandom
import Testing

/// Task 2 (ANE+GPU concurrent offload plan): promotes the Task 1 PoC's inline
/// ANE matmul into a reusable, warm `ANEGemm` primitive -- see
/// `.superpowers/sdd/2026-08-30-ane-gpu-concurrent-offload/task-2-brief.md`.
@Suite(.serialized)
struct ANEGemmTests {
    @Test("ANEGemm matches MLX dense matmul within fp16 tolerance")
    func aneGemmCorrect() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        MLXRandom.seed(0)
        let S = 512, K = 5120, out = 2048
        let w = MLXRandom.normal([out, K]).asType(.float16)
        let x = MLXRandom.normal([S, K]).asType(.float16)
        eval(w, x)
        let g = try ANEGemm(weight: w, sequenceLength: S)
        let got = try g(x)
        let want = matmul(x, w.transposed(1, 0))
        eval(want)
        // Tolerance widened from the brief's literal 0.2 to 0.3: at this
        // seed/shape, exactly 1 of the 1,048,576 output elements lands at a
        // fp16 rounding-boundary where the ANE conv path and the MLX dense
        // matmul path round to opposite sides (diff 0.25). A diagnostic
        // comparing both against an fp32 reference showed both paths are
        // independently ~equally accurate (max err ~0.125 each vs fp32,
        // mean err ~0.01) -- this is fp16 rounding noise, not divergent
        // computation, matching the "near-tie" fp16 behavior documented
        // elsewhere in this repo's correctness contracts.
        #expect((abs(got - want).max()).item(Float.self) < 0.3)
        // Mean-error lock alongside the max check: a uniform bias (e.g. a
        // wrong weight/activation layout or a scale error) could stay under
        // the 0.3 max bound yet be wrong on every element, which the max-only
        // check would miss. Measured ~2.7e-4 on this seed/shape -- ~40x
        // margin below this bound -- so this catches a systematic error the
        // max check alone would not.
        let meanErr = (abs(got - want)).mean().item(Float.self)
        #expect(meanErr < 0.01)
    }

    @Test("ANEGemm matches MLX dense matmul at S=1, not a multiple of the ANE's 32-wide padding")
    func aneGemmArbitrarySequenceLength1() throws {
        try assertArbitrarySequenceLength(S: 1, seed: 100)
    }

    @Test("ANEGemm matches MLX dense matmul at S=500, not a multiple of the ANE's 32-wide padding")
    func aneGemmArbitrarySequenceLength500() throws {
        try assertArbitrarySequenceLength(S: 500, seed: 101)
    }

    /// Shared body for the arbitrary-S regression tests: prior to the
    /// stride-correct `multiArray_1C1S_toMLX` fix, any `S` not already a
    /// multiple of 32 hit an `MLXArray.init` precondition crash (the ANE
    /// output is 64-byte/32-element padded on its trailing sequence axis).
    private func assertArbitrarySequenceLength(S: Int, seed: UInt64) throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        MLXRandom.seed(seed)
        let K = 5120, out = 2048
        let w = MLXRandom.normal([out, K]).asType(.float16)
        let x = MLXRandom.normal([S, K]).asType(.float16)
        eval(w, x)
        let g = try ANEGemm(weight: w, sequenceLength: S)
        let got = try g(x)
        let want = matmul(x, w.transposed(1, 0))
        eval(want)
        // Same fp16 tolerance as `aneGemmCorrect` above.
        #expect((abs(got - want).max()).item(Float.self) < 0.3)
        let meanErr = (abs(got - want)).mean().item(Float.self)
        #expect(meanErr < 0.01)
    }

    @Test("split path (makeInput -> predict -> readOutput) matches the callAsFunction convenience bit-identically")
    func splitPathMatchesConvenience() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        MLXRandom.seed(2)
        let S = 512, K = 5120, out = 2048
        let w = MLXRandom.normal([out, K]).asType(.float16)
        let x = MLXRandom.normal([S, K]).asType(.float16)
        eval(w, x)
        let g = try ANEGemm(weight: w, sequenceLength: S)

        let viaConvenience = try g(x)
        let input = try g.makeInput(x)
        let output = try g.predict(input)
        let viaSplit = g.readOutput(output)
        eval(viaConvenience, viaSplit)

        let maxDiff = (abs(viaConvenience - viaSplit).max()).item(Float.self)
        #expect(maxDiff == 0, "split path diverged from callAsFunction by \(maxDiff)")
    }

    @Test("ANEGemm per-call latency is stable across repeated calls (no recompile per call)")
    func aneGemmWarmupStable() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        MLXRandom.seed(1)
        let S = 512, K = 5120, out = 2048
        let w = MLXRandom.normal([out, K]).asType(.float16)
        let x = MLXRandom.normal([S, K]).asType(.float16)
        eval(w, x)
        let g = try ANEGemm(weight: w, sequenceLength: S)

        // Discard the first call: even after `init` compiles the model, the
        // very first `prediction` can pay a one-time ANE program load/warm
        // cost distinct from steady-state per-call latency.
        _ = try g(x)

        var latencies: [Double] = []
        for _ in 0 ..< 8 {
            let t0 = Date()
            _ = try g(x)
            latencies.append(Date().timeIntervalSince(t0))
        }
        let mean = latencies.reduce(0, +) / Double(latencies.count)
        let maxLatency = latencies.max() ?? 0
        // A per-call recompile would show as an outlier many times the mean;
        // steady-state warm calls should stay within a tight band.
        #expect(maxLatency < mean * 5 + 0.05,
                "call latency not stable: mean=\(mean) max=\(maxLatency) all=\(latencies)")
    }
}
