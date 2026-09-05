import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel

/// Does re-encoding the n-gram table change what the model says?
///
/// The round-trip and real-shard tests in `Qwen4ExpNGramQuantizeTests` bound
/// reconstruction error in embedding space. They answer how far an embedding
/// moves, not whether the network cares. This runs one real 512-token prefill
/// through the whole 48-layer tower and records the decision at every
/// position, so two runs over different table encodings can be compared.
///
/// Run it twice, once per encoding, and diff the two files. The table is
/// self-describing, so `MLX_QWEN4EXP_NGRAM_DIR` is the only thing that changes:
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
///     MLXFAST_QWEN4EXP_WEIGHTS=<weights> \
///     MLXFAST_NGRAM_AB_PROMPT=<prompt.txt> \
///     MLXFAST_NGRAM_AB_OUT=<out.json> \
///     [MLX_QWEN4EXP_NGRAM_DIR=<weights>/ngram-int4] \
///     swift test -c release --force-resolved-versions --filter ngramEncodingDivergence
///
/// One model per process: the tower is about 87 GB, so two arms in one process
/// would not fit. Decoding is greedy and MLX is deterministic on identical
/// input, so two processes give a comparable answer.
@Suite(.serialized)
struct Qwen4ExpNGramEncodingDivergenceTests {
    private final class Box<T>: @unchecked Sendable { var value: T? }

    @Test("n-gram encoding logit divergence over one real prefill")
    func ngramEncodingDivergence() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            let weights = env["MLXFAST_QWEN4EXP_WEIGHTS"],
            let promptPath = env["MLXFAST_NGRAM_AB_PROMPT"],
            let outPath = env["MLXFAST_NGRAM_AB_OUT"]
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

        let text = try String(contentsOfFile: promptPath, encoding: .utf8)
        let all = context.tokenizer.encode(text: text, addSpecialTokens: false)
        try #require(all.count >= 512, "prompt has \(all.count) tokens, need 512")
        let ids = Array(all[0 ..< 512])
        let tokens = MLXArray(ids.map { Int32($0) }).reshaped([1, 512])

        let cache = context.model.newCache(parameters: nil)
        let out = context.model(LMInput.Text(tokens: tokens), cache: cache, state: nil)
        let logits = out.logits.asType(.float32)
        eval(logits)

        // Per position: the argmax, its value, and the runner-up. That is
        // enough to compare two arms on decision agreement AND to say whether
        // any disagreement was a near-tie or a real change of mind. Dumping
        // raw logits would be 512 x 248320 floats, about 508 MB per arm.
        let vocab = logits.dim(2)
        let flat = logits.reshaped([512, vocab])
        let top = MLX.top(flat, k: 2, axis: -1)
        let topIdx = MLX.argSort(flat, axis: -1)[0..., (vocab - 2)...]
        eval(top, topIdx)
        let topValues = top.asArray(Float.self)
        let topIndices = topIdx.asArray(Int32.self)

        var rows: [[String: Any]] = []
        rows.reserveCapacity(512)
        for p in 0 ..< 512 {
            // argSort is ascending, so the last column is the argmax.
            let best = Int(topIndices[p * 2 + 1]), second = Int(topIndices[p * 2])
            let bestV = max(topValues[p * 2], topValues[p * 2 + 1])
            let secondV = min(topValues[p * 2], topValues[p * 2 + 1])
            rows.append([
                "pos": p, "argmax": best, "runner_up": second,
                "top1": bestV, "top2": secondV, "gap": bestV - secondV,
            ])
        }
        let payload: [String: Any] = [
            "ngram_dir": env["MLX_QWEN4EXP_NGRAM_DIR"] ?? "<default: config ngram_table.directory>",
            "prompt": promptPath,
            "prompt_tokens": ids,
            "positions": rows,
        ]
        try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: outPath))
        print("[ngram-ab] wrote \(outPath) for table \(payload["ngram_dir"]!)")
    }
}
