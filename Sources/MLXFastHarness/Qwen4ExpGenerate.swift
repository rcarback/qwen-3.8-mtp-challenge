// Local fork only: serial greedy generation for Qwen3.8-Flash-Next (qwen4_exp)
// with timing, used for correctness smoke and GPU-only vs ANE-lane A/B runs.
// Lives in the worker support target because the tokenizer loader macro and
// the model library are linked here, not in MLXFastModel.
import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

public enum Qwen4ExpGenerate {
    public struct Options {
        public var weights: URL
        public var prompt: String
        public var maxTokens: Int
        public var chat: Bool
        public var jsonOut: URL?

        public init(weights: URL, prompt: String, maxTokens: Int = 128, chat: Bool = true, jsonOut: URL? = nil) {
            self.weights = weights
            self.prompt = prompt
            self.maxTokens = maxTokens
            self.chat = chat
            self.jsonOut = jsonOut
        }
    }

    public static func run(_ o: Options) throws {
        Qwen4ExpRuntime.weightsDirectory = o.weights
        let context = try waitForAsync {
            try await LLMModelFactory.shared.load(from: o.weights, using: #huggingFaceTokenizerLoader())
        }
        guard let model = context.model as? Qwen4ExpModel else {
            throw MLXFastError.invalidInput("loaded \(type(of: context.model)), expected Qwen4ExpModel")
        }
        let tokenizer = context.tokenizer
        let ids: [Int]
        if o.chat {
            ids = try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": o.prompt]], tools: nil, additionalContext: nil)
        } else {
            ids = tokenizer.encode(text: o.prompt, addSpecialTokens: true)
        }
        var eos = Set([model.configuration.textConfig.eosTokenId])
        if let t = tokenizer.eosToken, let id = tokenizer.convertTokenToId(t) { eos.insert(id) }
        let cache = model.newCache(parameters: nil)

        let t0 = Date()
        var logits = model(MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count), cache: cache)
        var next = argMax(logits[0, -1], axis: -1)
        eval(next)
        let prefill = Date().timeIntervalSince(t0)

        var out = [Int]()
        let t1 = Date()
        for _ in 0 ..< o.maxTokens {
            let tok = Int(next.item(Int32.self))
            if eos.contains(tok) { break }
            out.append(tok)
            logits = model(next.reshaped(1, 1), cache: cache)
            next = argMax(logits[0, -1], axis: -1)
            eval(next)
        }
        let decode = Date().timeIntervalSince(t1)
        print(tokenizer.decode(tokenIds: out, skipSpecialTokens: false))
        let stats: [String: Any] = [
            "prompt_tokens": ids.count,
            "prefill_seconds": prefill,
            "decode_tokens": out.count,
            "decode_tokens_per_second": out.isEmpty ? 0 : Double(out.count) / decode,
            "peak_memory_gb": Double(MLX.Memory.peakMemory) / 1e9,
        ]
        let line = try JSONSerialization.data(withJSONObject: stats, options: [.sortedKeys])
        print(String(decoding: line, as: UTF8.self))
        if let j = o.jsonOut { try line.write(to: j) }
    }

    private final class ResultBox<T>: @unchecked Sendable {
        var result: Result<T, Error>?
    }

    private static func waitForAsync<T>(_ operation: @escaping () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        let box = ResultBox<T>()
        nonisolated(unsafe) let unsafeOperation = operation
        nonisolated(unsafe) let unsafeBox = box
        Task {
            do {
                unsafeBox.result = .success(try await unsafeOperation())
            } catch {
                unsafeBox.result = .failure(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        guard let result = box.result else {
            throw MLXFastError.invalidInput("the model load completed without a result")
        }
        return try result.get()
    }
}
