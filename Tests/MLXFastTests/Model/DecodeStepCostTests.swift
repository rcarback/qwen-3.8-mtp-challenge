import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel

/// How long is one decode step, so per-token overheads have a denominator?
///
/// The router measures 0.729 ms per layer at a single row, which is 35.0 ms per
/// token across 48 layers. Whether that matters depends entirely on what a
/// decode step costs in total, and nothing in this repository had measured it.
/// Every per-token optimisation estimate needs this number.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_QWEN4EXP_WEIGHTS=<weights> \
///       swift test -c release --force-resolved-versions --filter decodeStepCost
@Suite(.serialized)
struct DecodeStepCostTests {
    private final class Box<T>: @unchecked Sendable { var value: T? }

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    @Test("one decode step, real model", .enabled(if: enabled))
    func decodeStepCost() throws {
        let env = ProcessInfo.processInfo.environment
        guard let weights = env["MLXFAST_QWEN4EXP_WEIGHTS"] else { return }
        let weightsURL = URL(fileURLWithPath: weights)
        Qwen4ExpRuntime.weightsDirectory = weightsURL

        let box = Box<ModelContext>()
        let failure = Box<String>()
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            do {
                box.value = try await LLMModelFactory.shared.load(
                    from: weightsURL, using: #huggingFaceTokenizerLoader())
            } catch {
                failure.value = String(describing: error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        guard let context = box.value else {
            throw MLXFastError.invalidInput("failed to load model: \(failure.value ?? "?")")
        }

        // Seed with a short prefill so the cache is populated, then time decode
        // steps one token at a time, which is the regime per-token overheads
        // are paid in.
        let seed = (0 ..< 64).map { Int32(1000 + $0) }
        let cache = context.model.newCache(parameters: nil)
        let prefill = context.model(
            LMInput.Text(tokens: MLXArray(seed).reshaped([1, seed.count])), cache: cache,
            state: nil)
        eval(prefill.logits)

        var next = Int32(42)
        // Warm: the first decode step after a prefill pays one-off costs.
        for _ in 0 ..< 3 {
            let o = context.model(
                LMInput.Text(tokens: MLXArray([next]).reshaped([1, 1])), cache: cache, state: nil)
            eval(o.logits)
        }

        var best = Double.greatestFiniteMagnitude
        var total = 0.0
        let steps = 32
        for _ in 0 ..< 3 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< steps {
                let o = context.model(
                    LMInput.Text(tokens: MLXArray([next]).reshaped([1, 1])), cache: cache,
                    state: nil)
                eval(o.logits)
            }
            let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 / Double(steps)
            best = min(best, dt)
            total += dt
            next = (next + 1) % 1000
        }
        let routerPerToken = 35.0  // 0.729 ms x 48 layers, measured separately
        print(
            "[decode] best=\(String(format: "%.2f", best))ms/token "
                + "mean=\(String(format: "%.2f", total / 3))ms/token; "
                + "router 35.0ms/token = \(String(format: "%.1f", routerPerToken / best * 100))% "
                + "of the best step")
    }
}
