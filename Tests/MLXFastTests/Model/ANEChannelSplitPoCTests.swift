import CoreML
import Foundation
import MLX
import MLXRandom
import Testing

/// Task 1 PoC (ANE+GPU concurrent prefill offload plan): does splitting one
/// real projection's output channels between the ANE (Core ML, fp16) and the
/// GPU (MLX, 4-bit quantized), run concurrently, beat the GPU doing all of it
/// on this M4? This is the de-risk gate for the whole plan -- see
/// `.superpowers/sdd/2026-08-30-ane-gpu-concurrent-offload/task-1-brief.md`.
@Suite(.serialized)
struct ANEChannelSplitPoCTests {
    // The gate/up projection shape at the ranked prefill window.
    static let S = 512, K = 5120, N = 17408

    @Test("channel-split gate output equals full GPU output (fp16 tolerance)")
    func splitMatchesFull() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        MLXRandom.seed(0)
        let F = 6976 // ANE channel count (~40% of 17408, multiple of 64)
        let x = MLXRandom.normal([Self.S, Self.K]).asType(.float16)
        let w = MLXRandom.normal([Self.N, Self.K]).asType(.float16) // [out, in]
        eval(x, w)

        // Full reference: dense fp16 matmul x @ w.T -> [S, N]
        let full = matmul(x, w.transposed(1, 0))
        eval(full)

        // GPU suffix: channels [F..<N]
        let gpuPart = matmul(x, w[F ..< Self.N, 0...].transposed(1, 0)) // [S, N-F]
        // ANE prefix: channels [0..<F], via Core ML (built in Step 3)
        let anePart = try await Self.aneMatmul(x: x, wPrefix: w[0 ..< F, 0...], F: F) // [S, F]
        let combined = concatenated([anePart, gpuPart], axis: 1) // [S, N]
        eval(gpuPart, combined)

        let err = (abs(combined - full).max()).item(Float.self)
        #expect(err < 0.2, "channel-split diverged from full: maxAbsErr=\(err)")
    }

    /// Runs `x[S,K] @ wPrefix[F,K].T` on the ANE via a one-op Core ML MIL
    /// program (a 1x1 `conv` -- see `buildConvMatmul` in `ANEMILBuilder.swift`).
    /// `MLModelAsset(specification:)` is Apple's own in-memory compiler path:
    /// no SIP-off, no entitlement forging, no trustcache edits -- Apple mints
    /// the signed token for the model asset it compiles.
    static func aneMatmul(x: MLXArray, wPrefix: MLXArray, F: Int) async throws -> MLXArray {
        let wBytes = f16Bytes(wPrefix)
        let spec = buildConvMatmul(K: Self.K, F: F, S: Self.S, weight: wBytes)
        let asset = try MLModelAsset(specification: spec)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        let model = try await MLModel.load(asset: asset, configuration: cfg)
        let xa = try mlxToMultiArray_1C1S(x)
        let input = try MLDictionaryFeatureProvider(dictionary: ["a": MLFeatureValue(multiArray: xa)])
        let out = try await model.prediction(from: input)
        guard let ya = out.featureValue(for: "y")?.multiArrayValue else {
            throw NSError(domain: "poc", code: 2, userInfo: [NSLocalizedDescriptionKey: "no ANE output 'y'"])
        }
        return multiArray_1C1S_toMLX(ya)
    }

    /// Boxes the Core ML handles so a `MLModel.prediction(from:)` call (the
    /// synchronous overload) can be dispatched to a background queue -- the
    /// same pattern as `ANEMetalPipelineTests.hybridCombinedThroughput` /
    /// `ANEMetalPartitionTests.MLBox`. MLX `eval` only ever runs on the
    /// calling (main) thread; the ANE side only ever touches Core ML types.
    private final class MLBox: @unchecked Sendable {
        let model: MLModel
        let input: MLFeatureProvider
        init(_ m: MLModel, _ i: MLFeatureProvider) { model = m; input = i }
    }

    /// Cross-thread result handoff via `DispatchGroup` join (happens-before/
    /// -after via the group), so no lock is needed for this narrow use.
    private final class ResultBox: @unchecked Sendable { var value = 0 }

    @Test("channel-split gate beats GPU-alone on M4 (paired)")
    func splitBeatsGPU() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        var report = "# Task 1 PoC: channel-split gate projection ANE+GPU vs GPU-alone (M4)\n\n"
        report += "Paired per cycle: GPU-alone (all N channels, 4-bit quantizedMM), then GPU suffix " +
            "(N-F channels, 4-bit) + ANE prefix (F channels, fp16 conv) concurrent. " +
            "ratio = (gpu_suffix_conc + ane_prefix_conc useful TFLOPS) / gpu_alone. ratio>1.05 => split beats " +
            "GPU-alone with margin. Adjacent phases => host noise divides out.\n\n"
        let reportPath = env["SPLIT_POC_REPORT"] ??
            "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/" +
            "3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/split-poc-report.md"
        func flush() { try? report.write(toFile: reportPath, atomically: true, encoding: .utf8) }
        defer { flush() }

        MLXRandom.seed(1)
        let F = 6976
        let x = MLXRandom.normal([Self.S, Self.K]).asType(.float16)
        let w = MLXRandom.normal([Self.N, Self.K]).asType(.float16)
        eval(x, w)

        // Build the ANE model ONCE, outside any timing loop -- the Espresso/
        // ANE compile can take 1-3 minutes and pegs a core; that cost is not
        // part of the measured ratio (matches the disk-cache framing in
        // ANEMetalPartitionTests: compilation is a build-time artifact step).
        let wPrefix = w[0 ..< F, 0...]
        let wBytes = f16Bytes(wPrefix)
        let spec = buildConvMatmul(K: Self.K, F: F, S: Self.S, weight: wBytes)
        let asset = try MLModelAsset(specification: spec)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        let model = try await MLModel.load(asset: asset, configuration: cfg)
        let xa = try mlxToMultiArray_1C1S(x)
        let input = try MLDictionaryFeatureProvider(dictionary: ["a": MLFeatureValue(multiArray: xa)])
        _ = try await model.prediction(from: input) // warm
        let box = MLBox(model, input)

        // GPU side: real 4-bit affine group-64 quantized path (the actual
        // Qwen forward path), not bf16 (which would run at a different rate
        // and skew the ratio).
        let (wqAll, scAll, biAll0) = quantized(w, groupSize: 64, bits: 4)
        let biAll = biAll0 ?? scAll
        eval(wqAll, scAll, biAll)
        func gpuAllOp() -> MLXArray {
            quantizedMM(x, wqAll, scales: scAll, biases: biAll, transpose: true, groupSize: 64, bits: 4)
        }
        eval(gpuAllOp())

        let wSuffix = w[F ..< Self.N, 0...]
        let (wqSuf, scSuf, biSuf0) = quantized(wSuffix, groupSize: 64, bits: 4)
        let biSuf = biSuf0 ?? scSuf
        eval(wqSuf, scSuf, biSuf)
        func gpuSuffixOp() -> MLXArray {
            quantizedMM(x, wqSuf, scales: scSuf, biases: biSuf, transpose: true, groupSize: 64, bits: 4)
        }
        eval(gpuSuffixOp())

        let gpuAllFlop = 2.0 * Double(Self.S) * Double(Self.K) * Double(Self.N)
        let gpuSuffixFlop = 2.0 * Double(Self.S) * Double(Self.K) * Double(Self.N - F)
        let aneFlop = 2.0 * Double(Self.S) * Double(Self.K) * Double(F)
        let window = 1.0

        func gpuIters(_ op: () -> MLXArray, until dl: Date) -> Int {
            var k = 0
            while Date() < dl { eval(op()); k += 1 }
            return k
        }

        report += "| cycle | gpu_alone TF | gpu_suffix_conc TF | ane_prefix_conc TF | combined TF | ratio |\n" +
            "|---|---|---|---|---|---|\n"
        // `DispatchGroup.wait()` is unavailable from an `async` context (Swift
        // steers toward TaskGroup there), but this join is genuinely
        // synchronous: the ANE role runs on a background dispatch queue while
        // the GPU role runs MLX `eval` on the calling thread, exactly the
        // pattern `ANEMetalPartitionTests.timeSplit` uses from a plain
        // (non-async) static helper. Route the loop through one here so the
        // `wait()` call sits outside the enclosing `async` function.
        func runOneCycle() -> (solo: Double, gpuConc: Double, aneConc: Double) {
            // Phase A: GPU alone, all N channels.
            let solo = Double(gpuIters(gpuAllOp, until: Date().addingTimeInterval(window)))
                * gpuAllFlop / window / 1e12

            // Phase B: GPU suffix + ANE prefix concurrent over one shared window.
            let dl = Date().addingTimeInterval(window)
            let aneRes = ResultBox()
            let g = DispatchGroup()
            g.enter()
            let boxRef = box
            DispatchQueue.global(qos: .userInitiated).async {
                var n = 0
                while Date() < dl { autoreleasepool { _ = try? boxRef.model.prediction(from: boxRef.input) }; n += 1 }
                aneRes.value = n
                g.leave()
            }
            let gpuN = gpuIters(gpuSuffixOp, until: dl)
            g.wait()
            let gpuConc = Double(gpuN) * gpuSuffixFlop / window / 1e12
            let aneConc = Double(aneRes.value) * aneFlop / window / 1e12
            return (solo, gpuConc, aneConc)
        }
        var ratios: [Double] = []
        for c in 0 ..< 6 {
            let (solo, gpuConc, aneConc) = runOneCycle()
            let combined = gpuConc + aneConc
            let ratio = solo > 0 ? combined / solo : 0
            ratios.append(ratio)
            report += String(format: "| %d | %.2f | %.2f | %.2f | %.2f | %.3f |\n",
                              c, solo, gpuConc, aneConc, combined, ratio)
            flush()
        }
        let mean = ratios.reduce(0, +) / Double(ratios.count)
        let sorted = ratios.sorted()
        report += String(format: "\n- [MEASURED] mean split/GPU-alone ratio = %.3f (min %.3f, max %.3f) over %d cycles\n",
                          mean, sorted.first ?? 0, sorted.last ?? 0, ratios.count)
        report += "\nPass criterion: mean ratio > 1.05. Read: ratio > 1 => the channel split delivers more " +
            "total useful throughput than the GPU computing every channel alone; ratio <= 1 => GPU-alone " +
            "wins and the offload is not worth it on this hardware.\n"

        #expect(mean > 1.05, "channel-split PoC did not beat GPU-alone with margin: mean ratio=\(mean)")
    }
}
