import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel

/// Measurement instrument, not a gate: generates a greedy continuation
/// under the CURRENT environment's numeric policy and reports agreement
/// against a baseline run. It never fails on divergence -- the human (or
/// runbook) judges the report. Timing printed here is DIRECTIONAL ONLY:
/// no thermal gate, no pairing; use ./benchmark-qwen-mtp.sh for timing.
@Suite(.serialized)
struct QwenNumericAgreementHarness {
    @Test("generate under the current numeric policy and report")
    func generateAndReport() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            let weights = env["MLXFAST_QWEN_PREFILL_WEIGHTS"],
            let head = env["MLXFAST_QWEN_PREFILL_HEAD"],
            let promptPath = env["MLXFAST_AGREEMENT_PROMPT"],
            let outPath = env["MLXFAST_AGREEMENT_OUT"]
        else { return }
        let decodeTokens = Int(env["MLXFAST_AGREEMENT_TOKENS"] ?? "") ?? 512
        let depth = Int(env["MLXFAST_AGREEMENT_DEPTH"] ?? "") ?? 2

        let targetURL = URL(fileURLWithPath: weights)
        let context = try Qwen36MTPHeadAttachment.withHeadAttached(
            backboneDirectory: targetURL,
            headDirectory: URL(fileURLWithPath: head)
        ) { _ in
            let box = UnsafeSendableBox<ModelContext>()
            let semaphore = DispatchSemaphore(value: 0)
            Task {
                box.value = try? await LLMModelFactory.shared.load(
                    from: targetURL, using: #huggingFaceTokenizerLoader())
                semaphore.signal()
            }
            semaphore.wait()
            guard let loaded = box.value else {
                throw MLXFastError.invalidInput("failed to load backbone")
            }
            return loaded
        }
        guard let model = context.model as? any Qwen36MTPTarget else {
            Issue.record("backbone is not an MTP target"); return
        }

        let prompt = try String(
            contentsOfFile: promptPath, encoding: .utf8)
        let seed = context.tokenizer.encode(text: prompt)

        let session = try Qwen36MTPBlockSession(model: model, stopTokens: [])
        let start = Date()
        var emitted = [try session.begin(seedTokens: seed)]
        var rounds = 0
        var drafted = 0
        var accepted = 0
        while emitted.count < decodeTokens {
            let round = try session.generateRound(depth: depth)
            emitted.append(contentsOf: round.tokens)
            rounds += 1
            drafted += round.draftTokens.count
            accepted += round.acceptedDraftCount
        }
        let elapsed = Date().timeIntervalSince(start)
        emitted = Array(emitted.prefix(decodeTokens))

        try emitted.map(String.init).joined(separator: "\n")
            .write(toFile: outPath, atomically: true, encoding: .utf8)

        let acceptRate = drafted > 0
            ? Double(accepted) / Double(drafted) * 100 : 0
        print("harness: seed \(seed.count) tokens, emitted \(emitted.count), "
            + "rounds \(rounds), depth \(depth), "
            + "accepted \(accepted)/\(drafted) drafts "
            + "(\(String(format: "%.1f", acceptRate))%), "
            + "\(String(format: "%.1f", elapsed))s wall (directional)")

        if let baselinePath = env["MLXFAST_AGREEMENT_BASELINE"] {
            let reference = try String(
                contentsOfFile: baselinePath, encoding: .utf8)
                .split(separator: "\n").compactMap { Int($0) }
            print(QwenAgreementReport.compare(
                reference: reference, candidate: emitted).summary)
        }
    }
}
