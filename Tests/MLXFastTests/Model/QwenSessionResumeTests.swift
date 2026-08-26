import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel

/// A restored resume point must reproduce the token stream EXACTLY.
///
/// This is the correctness gate for the prefix cache: if a restore diverges by
/// even one token, resuming a conversation from cache would silently produce a
/// different answer than continuing it would have. Head-side state is allowed
/// to differ (it only proposes), so the check is on emitted tokens, not on
/// accept counts.
@Suite(.serialized)
struct QwenSessionResumeTests {
    @Test("restore reproduces the continuation token-for-token")
    func restoreIsExact() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            let weights = env["MLXFAST_QWEN_PREFILL_WEIGHTS"],
            let head = env["MLXFAST_QWEN_PREFILL_HEAD"]
        else { return }

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

        let seed = (0 ..< 64).map { ($0 * 7919) % 90_000 + 10 }
        let session = try Qwen36MTPBlockSession(model: model, stopTokens: [])
        _ = try session.begin(seedTokens: seed)

        // Warm past the first round, then capture.
        for _ in 0 ..< 2 { _ = try session.generateRound(depth: 2) }
        let snapshot = session.snapshotState()
        print("""

          snapshot: kv \(snapshot.kvBytes / (1024 * 1024)) MiB, \
          recurrent \(snapshot.recurrentBytes / (1024 * 1024)) MiB, \
          committed \(snapshot.committedTokenCount)
        """)

        var reference: [Int] = []
        for _ in 0 ..< 4 {
            reference.append(contentsOf: try session.generateRound(depth: 2).tokens)
        }

        // Rewind and replay the same span.
        try session.restoreState(snapshot)
        var replayed: [Int] = []
        for _ in 0 ..< 4 {
            replayed.append(contentsOf: try session.generateRound(depth: 2).tokens)
        }

        print("  reference \(reference)")
        print("  replayed  \(replayed)\n")
        #expect(replayed == reference, "a restored session diverged")

        // And a second restore of the SAME snapshot must also work: a rewind
        // may revisit one round more than once.
        try session.restoreState(snapshot)
        var again: [Int] = []
        for _ in 0 ..< 4 {
            again.append(contentsOf: try session.generateRound(depth: 2).tokens)
        }
        #expect(again == reference, "the snapshot was consumed by its first use")
    }
}
