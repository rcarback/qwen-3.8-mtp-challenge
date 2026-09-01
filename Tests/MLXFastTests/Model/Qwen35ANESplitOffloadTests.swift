import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXLLM

/// Task MOVE wiring: the vendored `Qwen35FusedMLP.callAsFunction` prefill
/// path with the ANE∥GPU offload enabled (`MLX_ANE_DIRECT=1`) must equal
/// the all-GPU SwiGLU within the ANE fraction's fp16-vs-4bit tolerance, at
/// the real prefill shape. Gated on MLX_ANE_DIRECT=1 AND
/// MLXFAST_RUN_MLX_RUNTIME_TESTS=1 because `ANESplitConfig.enabled` is read
/// once from the environment at process start.
@Suite(.serialized)
struct Qwen35ANESplitOffloadTests {
    private static func quantizedLinear(out: Int, inn: Int, seed: UInt64) -> QuantizedLinear {
        MLXRandom.seed(seed)
        let w = (MLXRandom.normal([out, inn]) * Float(1.0 / Double(inn).squareRoot())).asType(.bfloat16)
        let linear = Linear(weight: w)
        return QuantizedLinear(linear, groupSize: 64, bits: 4, mode: .affine)
    }

    private static func mlp(hidden: Int, inter: Int) -> Qwen35FusedMLP {
        let m = Qwen35FusedMLP(dimensions: hidden, hiddenDimensions: inter)
        let gate = Self.quantizedLinear(out: inter, inn: hidden, seed: 1)
        let up = Self.quantizedLinear(out: inter, inn: hidden, seed: 2)
        let down = Self.quantizedLinear(out: hidden, inn: inter, seed: 3)
        m.update(modules: ModuleChildren.unflattened([
            ("gate_proj", gate), ("up_proj", up), ("down_proj", down),
        ]))
        return m
    }

    private static func gpuSwiGLU(_ x: MLXArray, _ m: Qwen35FusedMLP) -> MLXArray {
        m.downProj(silu(m.gateProj(x)) * m.upProj(x))
    }

    @Test("Qwen35FusedMLP.callAsFunction ANE path equals the all-GPU SwiGLU at [1,512,hidden]")
    func forwardMatchesGPU() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
              env["MLX_ANE_DIRECT"] == "1" else { return }
        try #require(ANERuntime.available())
        try #require(ANESplitConfig.enabled, "ANESplitConfig.enabled must be true under MLX_ANE_DIRECT=1")

        let hidden = 5_120, inter = 17_408, S = 512
        let m = Self.mlp(hidden: hidden, inter: inter)

        let x = MLXRandom.normal([1, S, hidden]).asType(.bfloat16)
        eval(x)

        let ref = Self.gpuSwiGLU(x, m)
        eval(ref)

        let y = m(x)
        eval(y)
        #expect(y.shape == [1, S, hidden], "shape must be preserved, got \(y.shape)")

        let diff = MLX.abs(y.asType(.float32) - ref.asType(.float32))
        eval(diff)
        let maxAbs = diff.max().item(Float.self)
        let meanAbs = diff.mean().item(Float.self)
        print("QWEN35-ANE-MLP maxAbs=\(maxAbs) meanAbs=\(meanAbs)")
        #expect(maxAbs < 0.09375, "maxAbs=\(maxAbs)")
        #expect(meanAbs < 0.02, "meanAbs=\(meanAbs)")
    }

    /// Flag-independent contract lock: whenever the ANE path is NOT taken --
    /// which includes the default flag-off serve path -- `callAsFunction`
    /// must equal the GPU SwiGLU bit-for-bit. Uses S=100 (not 32-aligned),
    /// so the cache returns nil regardless of MLX_ANE_DIRECT, exercising
    /// the exact fallback branch that runs in production when the flag is
    /// unset.
    @Test("Qwen35FusedMLP.callAsFunction equals the GPU SwiGLU exactly when the ANE path is skipped")
    func fallbackEqualsGPUExact() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let hidden = 5_120, inter = 17_408, S = 100 // not a multiple of 32 -> no ANE
        let m = Self.mlp(hidden: hidden, inter: inter)
        let x = MLXRandom.normal([1, S, hidden]).asType(.bfloat16)
        eval(x)
        let y = m(x)
        let ref = Self.gpuSwiGLU(x, m)
        eval(y, ref)
        #expect(y.shape == [1, S, hidden])
        let maxAbs = MLX.abs(y.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
        #expect(maxAbs == 0.0, "fallback path must be bit-identical to GPU SwiGLU; maxAbs=\(maxAbs)")
    }

    @Test("Qwen35FusedMLP.callAsFunction is the untouched GPU path for decode width S=1")
    func decodeStaysGPU() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
              env["MLX_ANE_DIRECT"] == "1" else { return }
        let hidden = 5_120, inter = 17_408
        let m = Self.mlp(hidden: hidden, inter: inter)
        let x = MLXRandom.normal([1, 1, hidden]).asType(.bfloat16)
        let y = m(x)
        let ref = Self.gpuSwiGLU(x, m)
        eval(y, ref)
        #expect(y.shape == [1, 1, hidden])
        let maxAbs = MLX.abs(y.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
        #expect(maxAbs == 0.0, "decode must stay on the GPU path; maxAbs=\(maxAbs)")
    }

    /// Exercises the `ANESplitMLPCache` guard logic directly: below
    /// `minSequenceLength` (128) the cache must refuse to build a program
    /// regardless of the flag, and 4-bit/group-64 triples above the
    /// threshold succeed only when the flag is enabled.
    @Test("ANESplitMLPCache refuses short sequences and honors the enabled flag")
    func cacheGuardLogic() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let hidden = 5_120, inter = 17_408
        let m = Self.mlp(hidden: hidden, inter: inter)
        guard let g = m.gateProj as? QuantizedLinear,
              let u = m.upProj as? QuantizedLinear,
              let d = m.downProj as? QuantizedLinear,
              let gb = g.biases, let ub = u.biases, let db = d.biases
        else {
            Issue.record("expected QuantizedLinear projections with biases")
            return
        }
        let cache = ANESplitMLPCache()
        let shortResult = cache.program(
            forSequenceLength: 1,
            gateW: g.weight, gateScales: g.scales, gateBiases: gb,
            gateBits: g.bits, gateGroupSize: g.groupSize,
            upW: u.weight, upScales: u.scales, upBiases: ub,
            upBits: u.bits, upGroupSize: u.groupSize,
            downW: d.weight, downScales: d.scales, downBiases: db,
            downBits: d.bits, downGroupSize: d.groupSize,
            hidden: hidden, inter: inter)
        #expect(shortResult == nil, "S=1 is below minSequenceLength; the cache must refuse")

        let longResult = cache.program(
            forSequenceLength: 512,
            gateW: g.weight, gateScales: g.scales, gateBiases: gb,
            gateBits: g.bits, gateGroupSize: g.groupSize,
            upW: u.weight, upScales: u.scales, upBiases: ub,
            upBits: u.bits, upGroupSize: u.groupSize,
            downW: d.weight, downScales: d.scales, downBiases: db,
            downBits: d.bits, downGroupSize: d.groupSize,
            hidden: hidden, inter: inter)
        if env["MLX_ANE_DIRECT"] == "1" {
            #expect(longResult != nil, "S=512 with 4-bit/group-64 weights must build under the flag")
        } else {
            #expect(longResult == nil, "the cache must refuse when MLX_ANE_DIRECT is unset")
        }
    }
}
