import Foundation
import MLX
import MLXLLM
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel

/// Runs ONE real-prompt prefill through the all-GPU model with the
/// `MLX_ANE_CAPTURE_DIR` hook on, so the captured post-norm MLP inputs are the
/// activations the production forward actually produces.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
///     MLXFAST_QWEN_PREFILL_WEIGHTS=weights \
///     MLX_ANE_CAPTURE_DIR=<dir> \
///     MLXFAST_ANE_CAPTURE_PROMPT=README.md \
///     swift test --force-resolved-versions --filter realActivationCapture
@Suite(.serialized)
struct ANERealActivationCaptureTests {
    private final class Box<T>: @unchecked Sendable { var value: T? }

    @Test("capture real MLP inputs from one 512-token prefill")
    func realActivationCapture() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            let weights = env["MLXFAST_QWEN_PREFILL_WEIGHTS"],
            let captureDir = env["MLX_ANE_CAPTURE_DIR"]
        else { return }
        let promptPath = env["MLXFAST_ANE_CAPTURE_PROMPT"] ?? "README.md"
        let sequenceLength = Int(env["MLXFAST_ANE_CAPTURE_TOKENS"] ?? "512") ?? 512

        let targetURL = URL(fileURLWithPath: weights)
        let headPath = env["MLXFAST_QWEN_PREFILL_HEAD"]
            ?? NSHomeDirectory() + "/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head"
        let headURL = URL(fileURLWithPath: headPath)
        // Same load pattern as QwenPhaseBreakdownTests: the head attachment
        // wrapper is what makes the transformed overlay's lm_head key load.
        let context = try Qwen36MTPHeadAttachment.withHeadAttached(
            backboneDirectory: targetURL, headDirectory: headURL
        ) { _ in
            let box = Box<ModelContext>()
            let failure = Box<String>()
            let semaphore = DispatchSemaphore(value: 0)
            Task {
                do {
                    box.value = try await LLMModelFactory.shared.load(
                        from: targetURL, using: #huggingFaceTokenizerLoader())
                } catch {
                    failure.value = String(describing: error)
                }
                semaphore.signal()
            }
            semaphore.wait()
            guard let loaded = box.value else {
                throw MLXFastError.invalidInput("failed to load backbone: \(failure.value ?? "?")")
            }
            return loaded
        }

        let text = try String(contentsOfFile: promptPath, encoding: .utf8)
        let allTokens = context.tokenizer.encode(text: text, addSpecialTokens: false)
        try #require(allTokens.count >= sequenceLength,
                     "prompt has \(allTokens.count) tokens, need \(sequenceLength)")
        let tokens = MLXArray(allTokens[0 ..< sequenceLength].map { Int32($0) })
            .reshaped([1, sequenceLength])

        let cache = context.model.newCache(parameters: nil)
        let output = context.model(LMInput.Text(tokens: tokens), cache: cache, state: nil)
        eval(output.logits)

        // Sanity: the captured run must itself be healthy. Greedy argmax of
        // the last few positions should not be one repeated token.
        let last = output.logits[0, (sequenceLength - 8)..., 0...].argMax(axis: -1)
        eval(last)
        print("ANE-CAPTURE: last-8 argmax \(last.asArray(Int32.self))")

        for layer in ANEActivationCapture.layers.sorted() {
            let path = URL(fileURLWithPath: captureDir)
                .appendingPathComponent("mlp-layer\(layer).safetensors").path
            #expect(FileManager.default.fileExists(atPath: path), "missing capture for layer \(layer)")
        }

        // Optional per-layer diff against an earlier capture of the same
        // prompt (MLXFAST_ANE_CAPTURE_COMPARE_DIR): where do two forwards
        // (for example all-GPU vs hybrid) part ways?
        if let compareDir = env["MLXFAST_ANE_CAPTURE_COMPARE_DIR"] {
            print("ANE-CAPTURE-DIFF: layer | max|x| here | max|x| there | maxAbs diff | rows(>1% of max) | first row over 1%")
            for layer in ANEActivationCapture.layers.sorted() {
                let name = "mlp-layer\(layer).safetensors"
                let here = try loadArrays(url: URL(fileURLWithPath: captureDir).appendingPathComponent(name))["x"]!
                    .asType(.float32)
                let there = try loadArrays(url: URL(fileURLWithPath: compareDir).appendingPathComponent(name))["x"]!
                    .asType(.float32)
                let diff = MLX.abs(here - there)
                let scale = MLX.abs(there).max()
                let rowMax = diff.max(axis: 1)  // [S]
                let bad = rowMax .> (scale * 0.01)
                eval(diff, scale, rowMax, bad)
                let badRows = bad.asArray(Bool.self)
                let firstBad = badRows.firstIndex(of: true).map(String.init) ?? "-"
                print(String(
                    format: "ANE-CAPTURE-DIFF: %5d | %8.2f | %8.2f | %8.4f | %4d | %@",
                    layer, MLX.abs(here).max().item(Float.self), scale.item(Float.self),
                    diff.max().item(Float.self), badRows.filter { $0 }.count, firstBad))
            }
        }
    }
}
