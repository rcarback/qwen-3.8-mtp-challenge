import CoreML
import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXFastModel
@testable import MLXLLM

/// Measures `r`, the ANE-to-GPU throughput ratio on expert-shaped GEMMs.
///
/// Every ANE/GPU split estimate in `docs/perf` is a guess until this number
/// exists. For a partition that runs both engines concurrently on the same
/// work, the optimal ANE share is `f* = r / (1 + r)` and the best achievable
/// speedup over GPU alone is `1 + r`. So r decides whether a split is worth
/// building at all: r = 1 would mean a 2x ceiling, r = 0.1 a 1.1x ceiling that
/// no amount of engineering recovers.
///
/// Shapes are the real routed-expert projections of Qwen3.8-Flash-Next: hidden
/// 2560, moe_intermediate 640, so gate and up are 2560 -> 640 and down is
/// 640 -> 2560. Both engines run fp16 dense GEMMs here. That is deliberate and
/// it is the honest framing of the comparison: production runs 4-bit affine
/// group-32 weights through `gatherQuantizedMM` on the GPU, which the ANE
/// cannot do at all, so this measures the ceiling a dense fp16 ANE lane could
/// reach against a dense fp16 GPU lane, not against production.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test -c release \
///       --force-resolved-versions --filter expertGemmRatio
@Suite(.serialized)
struct ANEExpertGemmRatioTests {
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    private final class MLBox: @unchecked Sendable {
        let model: MLModel
        let input: MLDictionaryFeatureProvider
        init(_ m: MLModel, _ i: MLDictionaryFeatureProvider) { model = m; input = i }
    }

    /// One expert-shaped projection on each engine, timed separately.
    private func measure(K: Int, F: Int, S: Int, label: String) async throws {
        let w = MLXRandom.normal([F, K]).asType(.float16)
        let x = MLXRandom.normal([S, K]).asType(.float16)
        eval(w, x)
        // 2*K*F*S flops: one multiply and one add per weight per row.
        let flops = 2.0 * Double(K) * Double(F) * Double(S)

        // --- GPU arm ---
        let wt = w.transposed(1, 0)
        eval(matmul(x, wt))  // warm
        var gpuBest = Double.greatestFiniteMagnitude
        for _ in 0 ..< 5 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< 20 { eval(matmul(x, wt)) }
            let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 / 20
            gpuBest = min(gpuBest, dt)
        }

        // --- ANE arm ---
        let spec = buildConvMatmul(K: K, F: F, S: S, weight: f16Bytes(w))
        let asset = try MLModelAsset(specification: spec)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        let model = try await MLModel.load(asset: asset, configuration: cfg)
        let xa = try mlxToMultiArray_1C1S(x)
        let input = try MLDictionaryFeatureProvider(
            dictionary: ["a": MLFeatureValue(multiArray: xa)])
        let box = MLBox(model, input)
        // autoreleasepool selects the SYNCHRONOUS prediction overload; a bare
        // call resolves to the async one and would need an await, which would
        // time a suspension point rather than the ANE.
        try autoreleasepool { _ = try box.model.prediction(from: box.input) }  // warm

        var aneBest = Double.greatestFiniteMagnitude
        for _ in 0 ..< 5 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< 20 {
                try autoreleasepool { _ = try box.model.prediction(from: box.input) }
            }
            let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 / 20
            aneBest = min(aneBest, dt)
        }

        let gpuTF = flops / gpuBest / 1e12
        let aneTF = flops / aneBest / 1e12
        let r = aneTF / gpuTF
        print(
            "[expert-r] \(label) K=\(K) F=\(F) S=\(S): "
                + "gpu \(String(format: "%.3f", gpuBest * 1000))ms = \(String(format: "%.2f", gpuTF)) TF/s; "
                + "ane \(String(format: "%.3f", aneBest * 1000))ms = \(String(format: "%.2f", aneTF)) TF/s; "
                + "r=\(String(format: "%.3f", r)) -> f*=\(String(format: "%.3f", r / (1 + r))) "
                + "ceiling=\(String(format: "%.2f", 1 + r))x")
    }

    @Test("expert-shaped GEMM: ANE vs GPU throughput ratio", .enabled(if: enabled))
    func expertGemmRatio() async throws {
        try #require(ANERuntime.available())
        // Warm both engines before timing anything. The router sweep taught
        // this the expensive way: its first-measured shape absorbed process
        // warmup and read 1.8x high, which inverted a conclusion.
        do {
            let w = MLXRandom.normal([640, 2560]).asType(.float16)
            let v = MLXRandom.normal([64, 2560]).asType(.float16)
            eval(w, v)
            for _ in 0 ..< 20 { eval(matmul(v, w.transposed(1, 0))) }
            let spec = buildConvMatmul(K: 2560, F: 640, S: 64, weight: f16Bytes(w))
            let asset = try MLModelAsset(specification: spec)
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndNeuralEngine
            let m = try await MLModel.load(asset: asset, configuration: cfg)
            let va = try mlxToMultiArray_1C1S(v)
            let inp = try MLDictionaryFeatureProvider(
                dictionary: ["a": MLFeatureValue(multiArray: va)])
            let box = MLBox(m, inp)
            for _ in 0 ..< 10 {
                try autoreleasepool { _ = try box.model.prediction(from: box.input) }
            }
        }

        // gate/up: 2560 -> 640.  down: 640 -> 2560.
        // S=1 IS DECODE and yvk never measured it. Decode runs one row per
        // token by construction, which is the launch-bound regime where the
        // GPU is slowest and the ANE looked competitive. S=16 and above are
        // prefill widths where batching is available.
        for S in [1, 2, 4, 16, 128, 1024] {
            try await measure(K: 2560, F: 640, S: S, label: "gate_up")
            try await measure(K: 640, F: 2560, S: S, label: "down   ")
        }
    }
}
