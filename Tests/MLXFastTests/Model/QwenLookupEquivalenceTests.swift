import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel

/// Lookup drafting must not change what the model says.
///
/// It CANNOT change it by the accept walk: every emitted token is the target's
/// own argmax over a row the target computed. It CAN change it numerically,
/// because a width-16 or width-32 verify leaves the wide-decode exactness chunk
/// in `AttentionUtils.swift:229` (which covers `6 <= qL <= 9`) and takes the
/// unfused gated-delta in-projections (`Qwen35.swift:1143`, gated on `S <= 9`).
/// The repo's own width-wall note records that such rows drift in top-2 VALUES
/// while ids hold.
///
/// So the assertion is not blind equality. Any divergence must sit at a
/// position whose top-2 margin on the lookup-off leg is below 1.0 -- a near-tie,
/// the same tolerance the track's own correctness contract describes. A
/// divergence at a well-separated position is a real defect.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
///     MLXFAST_QWEN_PREFILL_WEIGHTS=weights \
///     MLXFAST_QWEN_PREFILL_HEAD=<head dir> \
///     swift test --force-resolved-versions --filter lookupEquivalence
@Suite(.serialized)
struct QwenLookupEquivalenceTests {
    private static let nearTieMargin = 1.0

    @Test("lookupEquivalence")
    func lookupEquivalence() throws {
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
            Issue.record("backbone is not an MTP target")
            return
        }

        // A deliberately repeat-heavy seed: one block of text repeated three
        // times, which is the structure the feature targets and which a real
        // agent turn produces when it quotes a file back.
        let block = context.tokenizer.encode(
            text: """
                func renderStatsBar(_ stats: TurnStats, depth: Int) -> String {
                    var fields: [String] = []
                    fields.append("decode")
                    fields.append("accept")
                    fields.append("depth")
                    return fields.joined(separator: " | ")
                }
                """,
            addSpecialTokens: false)
        let seed = block + block + block

        struct Leg {
            var tokens: [Int] = []
            var margins: [Double] = []
            var rounds = 0
            var seconds = 0.0
            var lookupRounds = 0
            var lookupAccepted = 0
            var lookupProposed = 0
        }

        func run(withLookup: Bool) throws -> Leg {
            let session = try Qwen36MTPBlockSession(
                model: model, stopTokens: [])
            if withLookup {
                session.lookupIndex = NGramPromptLookupIndex(
                    configuration: .shipped)
            }
            try session.warmAllDepths(maxDepth: Qwen36MTPLimits.maxDepth)
            _ = try session.begin(seedTokens: seed)
            session.resetLookupHistory(seed)
            var leg = Leg()
            let started = Date()
            while leg.tokens.count < 192 {
                let result = try session.generateRound(depth: 8)
                leg.rounds += 1
                // The worker's own ledger guard, applied here so a broken
                // wide round fails in the test rather than in the server.
                #expect(
                    result.acceptedDraftCount + result.rejectedDraftCount + 1
                        == result.declaredRows)
                #expect(result.perRowTop2Tokens.count == result.declaredRows)
                #expect(result.perRowTop2Logits.count == result.declaredRows)
                for (index, token) in result.tokens.enumerated() {
                    leg.tokens.append(token)
                    let row = result.perRowTop2Logits[index]
                    leg.margins.append(
                        row.count >= 2 ? row[0] - row[1] : .infinity)
                }
                if result.tokens.isEmpty { break }
            }
            leg.seconds = Date().timeIntervalSince(started)
            leg.lookupRounds = session.lookupRoundCount
            leg.lookupAccepted = session.lookupAcceptedTotal
            leg.lookupProposed = session.lookupProposedTotal
            return leg
        }

        let off = try run(withLookup: false)
        let on = try run(withLookup: true)

        print(String(
            format: """

                  lookup off: %d tok in %d rounds, %.2f s, %.1f tok/s
                  lookup on : %d tok in %d rounds, %.2f s, %.1f tok/s
                  lookup rounds %d, accepted %d of %d proposed
                """,
            off.tokens.count, off.rounds, off.seconds,
            Double(off.tokens.count) / off.seconds,
            on.tokens.count, on.rounds, on.seconds,
            Double(on.tokens.count) / on.seconds,
            on.lookupRounds, on.lookupAccepted, on.lookupProposed))

        #expect(on.lookupRounds > 0, "no round drafted from the lookup source")

        let shared = min(off.tokens.count, on.tokens.count)
        var firstDivergence: Int?
        for index in 0 ..< shared where off.tokens[index] != on.tokens[index] {
            firstDivergence = index
            break
        }
        if let index = firstDivergence {
            let margin = off.margins[index]
            print(String(
                format: "  diverged at %d: off=%d on=%d, off margin %.4f",
                index, off.tokens[index], on.tokens[index], margin))
            #expect(
                margin < Self.nearTieMargin,
                Comment(rawValue: "lookup drafting changed the emitted "
                    + "token at position \(index) where the target's top-2 "
                    + "margin was \(margin); a well-separated flip is a "
                    + "defect, not a near-tie"))
        } else {
            print("  streams agree over \(shared) tokens")
        }
    }
}
