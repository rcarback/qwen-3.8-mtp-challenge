import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel
@testable import MLXLLM

/// Steady-state decode cost of the REAL model, one arm per process.
///
/// `DecodeStepCostTests` sweeps rows in one process and reports 2200 ms at its
/// first arm and 19.8 ms/token at its last. That spread is not a property of
/// row count: an 87 GB model faults in from disk during the first arm, so the
/// sweep charges the whole cold start to whichever shape runs first. Any A/B on
/// decode has to warm the residency out of the measurement before timing it.
///
/// The compiled glue is selected by environment, so an arm is a whole process:
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_QWEN4EXP_WEIGHTS=<weights> \
///       MLX_QWEN4EXP_COMPILE_GLUE=0 MLX_QWEN4EXP_COMPILE_SHARED=0 \
///       swift test -c release --force-resolved-versions --filter decodeCompileReal
@Suite(.serialized)
struct DecodeCompileRealTests {
    private final class Box<T>: @unchecked Sendable { var value: T? }

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    @Test("steady-state decode, one arm", .enabled(if: enabled), .timeLimit(.minutes(60)))
    func decodeCompileReal() throws {
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

        let cache = context.model.newCache(parameters: nil)
        let seed = (0 ..< 64).map { Int32(1000 + $0) }
        let prefill = context.model(
            LMInput.Text(tokens: MLXArray(seed).reshaped([1, seed.count])), cache: cache,
            state: nil)
        eval(prefill.logits)

        // One decode step, timed individually so the warm-up curve is visible
        // rather than assumed to have flattened.
        var tokenId = Int32(42)
        func step() -> Double {
            let tok = MLXArray([tokenId]).reshaped([1, 1])
            tokenId = (tokenId + 17) % 900 + 100
            let t0 = DispatchTime.now().uptimeNanoseconds
            let o = context.model(LMInput.Text(tokens: tok), cache: cache, state: nil)
            eval(o.logits)
            return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        }

        let warm = 220, timed = 200
        var curve = [Double]()
        for _ in 0 ..< warm { curve.append(step()) }
        // The PLE n-gram gather is host code -- it materialises the token ids to
        // the CPU, reads rows out of a 102 GB mmap, and builds a fresh MLXArray.
        // The table cannot be resident alongside an 87 GB model on a 128 GB box,
        // so at decode this is 16 random cold faults per step with almost no
        // readahead cover. Charging it against the step is the only way to know
        // whether decode is stalled on the GPU or on the SSD.
        Qwen4ExpNGramTable.stats.reset()
        var samples = [Double]()
        for _ in 0 ..< timed { samples.append(step()) }
        let ng = Qwen4ExpNGramTable.stats.snapshot()

        let sorted = samples.sorted()
        let median = sorted[sorted.count / 2]
        let mean = samples.reduce(0, +) / Double(samples.count)
        func band(_ s: ArraySlice<Double>) -> Double { s.reduce(0, +) / Double(s.count) }

        print("[decode-arm] glue=\(env["MLX_QWEN4EXP_COMPILE_GLUE"] ?? "1") "
            + "shared=\(env["MLX_QWEN4EXP_COMPILE_SHARED"] ?? "1")")
        print("  warm-up curve (means of 44):"
            + curve.chunked(44).map { String(format: " %7.2f", band($0[...])) }.joined())
        print(String(
            format: "  steady state over %d steps: median %.3f ms  mean %.3f ms  "
                + "p10 %.3f  p90 %.3f",
            timed, median, mean, sorted[timed / 10], sorted[timed * 9 / 10]))
        let ngMs = Double(ng.nanos) / 1e6 / Double(timed)
        print(String(
            format: "  n-gram gather: %d calls, %d rows, %.3f ms per step = %.1f%% of the step",
            ng.calls, ng.rows, ngMs, ngMs / median * 100))
    }
}

extension Array {
    fileprivate func chunked(_ n: Int) -> [[Element]] {
        stride(from: 0, to: count, by: n).map { Array(self[$0 ..< Swift.min($0 + n, count)]) }
    }
}
