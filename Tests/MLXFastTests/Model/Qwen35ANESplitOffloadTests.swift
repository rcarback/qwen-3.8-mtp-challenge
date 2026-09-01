import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXFastModel

/// Task D2 wiring: `Qwen35MLP.forward` with the ANE∥GPU offload enabled
/// (`MLXFAST_ANE_DIRECT=1`) must equal the all-GPU SwiGLU within the ANE
/// fraction's fp16-vs-4bit tolerance, at the real prefill shape, including
/// the `[1, S, hidden]` reshape round-trip. Gated on MLXFAST_ANE_DIRECT=1
/// AND MLXFAST_RUN_MLX_RUNTIME_TESTS=1 because `ANESplitConfig.enabled` is
/// read once from the environment at process start.
@Suite(.serialized)
struct Qwen35ANESplitOffloadTests {
    private static func qLinear(out: Int, inn: Int, seed: UInt64) throws -> Qwen35LinearWeight {
        MLXRandom.seed(seed)
        let w = (MLXRandom.normal([out, inn]) * Float(1.0 / Double(inn).squareRoot())).asType(.bfloat16)
        let (wq, s, b0) = quantized(w, groupSize: 64, bits: 4)
        eval(wq, s, b0 ?? s)
        return try Qwen35LinearWeight(
            weight: wq, scales: s, biases: b0 ?? s,
            logicalShape: [out, inn], groupSize: 64, bits: 4)
    }

    @Test("Qwen35MLP.forward ANE path equals the all-GPU SwiGLU at [1,512,hidden]")
    func forwardMatchesGPU() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
              env["MLXFAST_ANE_DIRECT"] == "1" else { return }
        try #require(ANERuntime.available())
        try #require(ANESplitConfig.enabled, "ANESplitConfig.enabled must be true under MLXFAST_ANE_DIRECT=1")

        let hidden = 5_120, inter = 17_408, S = 512
        let weights = Qwen35MLPWeights(
            gateProjection: try Self.qLinear(out: inter, inn: hidden, seed: 1),
            upProjection: try Self.qLinear(out: inter, inn: hidden, seed: 2),
            downProjection: try Self.qLinear(out: hidden, inn: inter, seed: 3))

        let x = MLXRandom.normal([1, S, hidden]).asType(.bfloat16)
        eval(x)

        // GPU reference: the exact all-GPU SwiGLU forward uses when disabled.
        let refGate = silu(Qwen35Ops.linear(x, weights.gateProjection))
        let refUp = Qwen35Ops.linear(x, weights.upProjection)
        let ref = Qwen35Ops.linear(refGate * refUp, weights.downProjection)
        eval(ref)

        let y = Qwen35MLP.forward(x, weights: weights)
        eval(y)
        #expect(y.shape == [1, S, hidden], "shape must be preserved, got \(y.shape)")

        // Prove the ANE path actually ran (cache built a program for S=512).
        #expect(weights.aneCache.program(
            forSequenceLength: S,
            gate: weights.gateProjection, up: weights.upProjection, down: weights.downProjection) != nil,
            "the ANE split program must have been built for S=\(S)")

        let diff = MLX.abs(y.asType(.float32) - ref.asType(.float32))
        eval(diff)
        let maxAbs = diff.max().item(Float.self)
        let meanAbs = diff.mean().item(Float.self)
        print("QWEN35-ANE-MLP maxAbs=\(maxAbs) meanAbs=\(meanAbs)")
        #expect(maxAbs < 0.09375, "maxAbs=\(maxAbs)")
        #expect(meanAbs < 0.02, "meanAbs=\(meanAbs)")
    }

    /// The all-GPU SwiGLU the disabled/fallback path must reproduce exactly.
    private static func gpuSwiGLU(_ x: MLXArray, _ w: Qwen35MLPWeights) -> MLXArray {
        let gate = silu(Qwen35Ops.linear(x, w.gateProjection))
        let up = Qwen35Ops.linear(x, w.upProjection)
        return Qwen35Ops.linear(gate * up, w.downProjection)
    }

    /// Flag-independent contract lock: whenever the ANE path is NOT taken --
    /// which includes the default flag-off serve path -- `forward` must equal
    /// the three-line GPU SwiGLU bit-for-bit. Uses S=100 (not 32-aligned), so
    /// the cache returns nil regardless of MLXFAST_ANE_DIRECT, exercising the
    /// exact fallback branch that runs in production when the flag is unset.
    @Test("Qwen35MLP.forward equals the GPU SwiGLU exactly when the ANE path is skipped")
    func fallbackEqualsGPUExact() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let hidden = 5_120, inter = 17_408, S = 100 // not a multiple of 32 -> no ANE
        let weights = Qwen35MLPWeights(
            gateProjection: try Self.qLinear(out: inter, inn: hidden, seed: 1),
            upProjection: try Self.qLinear(out: inter, inn: hidden, seed: 2),
            downProjection: try Self.qLinear(out: hidden, inn: inter, seed: 3))
        let x = MLXRandom.normal([1, S, hidden]).asType(.bfloat16)
        eval(x)
        let y = Qwen35MLP.forward(x, weights: weights)
        let ref = Self.gpuSwiGLU(x, weights)
        eval(y, ref)
        #expect(y.shape == [1, S, hidden])
        let maxAbs = MLX.abs(y.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
        #expect(maxAbs == 0.0, "fallback path must be bit-identical to GPU SwiGLU; maxAbs=\(maxAbs)")
    }

    @Test("Qwen35MLP.forward is the untouched GPU path for decode width S=1")
    func decodeStaysGPU() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
              env["MLXFAST_ANE_DIRECT"] == "1" else { return }
        let hidden = 5_120, inter = 17_408
        let weights = Qwen35MLPWeights(
            gateProjection: try Self.qLinear(out: inter, inn: hidden, seed: 1),
            upProjection: try Self.qLinear(out: inter, inn: hidden, seed: 2),
            downProjection: try Self.qLinear(out: hidden, inn: inter, seed: 3))
        // S=1 is below minSequenceLength and not 32-aligned -> no ANE program.
        #expect(weights.aneCache.program(
            forSequenceLength: 1,
            gate: weights.gateProjection, up: weights.upProjection, down: weights.downProjection) == nil)
        let x = MLXRandom.normal([1, 1, hidden]).asType(.bfloat16)
        let y = Qwen35MLP.forward(x, weights: weights)
        let ref = Self.gpuSwiGLU(x, weights)
        eval(y, ref)
        #expect(y.shape == [1, 1, hidden])
        let maxAbs = MLX.abs(y.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
        #expect(maxAbs == 0.0, "decode must stay on the GPU path; maxAbs=\(maxAbs)")
    }
}
