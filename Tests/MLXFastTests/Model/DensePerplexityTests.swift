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

/// Teacher-forced next-token loss on real prose, one arm per process.
///
/// A greedy-decode divergence index says two arms differ; it cannot say which
/// is better, and any quantization of the dense weights will diverge from bf16
/// early -- upstream's own 4-bit tree would. The decision between q4 and q8
/// dense is a quality decision, so it needs a quality number: mean negative
/// log-likelihood of the actual next token over 512 teacher-forced positions,
/// position-independent and comparable across arms.
///
///     DARKBLOOM_DENSE_QUANT_BITS=<unset|4|8> MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
///       MLXFAST_QWEN4EXP_WEIGHTS=<weights> MLXFAST_PPL_PROMPTS=<dir of .txt> \
///       swift test -c release --force-resolved-versions --filter densePerplexity
@Suite(.serialized)
struct DensePerplexityTests {
    private final class Box<T>: @unchecked Sendable { var value: T? }

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    @Test("teacher-forced NLL, one arm", .enabled(if: enabled), .timeLimit(.minutes(30)))
    func densePerplexity() throws {
        let env = ProcessInfo.processInfo.environment
        guard let weights = env["MLXFAST_QWEN4EXP_WEIGHTS"], let dir = env["MLXFAST_PPL_PROMPTS"]
        else { return }
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

        let files = try FileManager.default.contentsOfDirectory(atPath: dir)
            .filter { $0.hasSuffix(".txt") }.sorted()
        let arm = env["DARKBLOOM_DENSE_QUANT_BITS"].map { "q\($0)" } ?? "bf16"
        print("[ppl] dense=\(arm)")
        var totalNLL = 0.0, totalN = 0
        for f in files {
            let text = try String(contentsOfFile: dir + "/" + f, encoding: .utf8)
            let all = context.tokenizer.encode(text: text, addSpecialTokens: false)
            guard all.count >= 513 else { print("  \(f): only \(all.count) tokens, skipped"); continue }
            let ids = Array(all[0 ..< 513])
            let tokens = MLXArray(ids.map { Int32($0) }).reshaped([1, 513])
            let cache = context.model.newCache(parameters: nil)
            // Input positions 0..<512 predict targets 1...512.
            let out = context.model(
                LMInput.Text(tokens: tokens[0..., 0 ..< 512]), cache: cache, state: nil)
            let logits = out.logits[0].asType(.float32)  // [512, V]
            let logp = logits - logits.logSumExp(axis: -1, keepDims: true)
            let targets = MLXArray(ids[1 ... 512].map { Int32($0) }).reshaped([512, 1])
            let nll = -MLX.takeAlong(logp, targets, axis: -1).squeezed(axis: -1)
            eval(nll)
            let v = nll.asArray(Float.self).map { Double($0) }
            let mean = v.reduce(0, +) / Double(v.count)
            totalNLL += v.reduce(0, +); totalN += v.count
            let name = f.replacingOccurrences(of: ".txt", with: "")
            print(String(format: "  %-10@ nll %.4f   ppl %.3f", name as NSString, mean, exp(mean)))
        }
        let mean = totalNLL / Double(max(totalN, 1))
        print(String(format: "  %-10@ nll %.4f   ppl %.3f   over %d positions", "ALL" as NSString, mean, exp(mean), totalN))
    }
}
