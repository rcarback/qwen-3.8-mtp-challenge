import Foundation
import Testing

@testable import MLXFastModel

/// MEASUREMENT SCAFFOLDING, not a behaviour assertion.
///
/// Replays an agent-CLI workload against the REAL `QwenSessionCacheStore` --
/// real eviction order, real byte accounting, real lookup order -- and reports
/// how many prompt tokens each arm has to prefill. It exists to answer whether
/// 8-bit KV and the raised cache budget actually relieve the concurrency
/// penalty, separately from the live-server measurement, which is blocked by
/// the restore-path ledger bug.
///
/// Run with:
///   swift test --force-resolved-versions \
///     --filter QwenSessionCacheStoreConcurrencyProbe
@Suite(.serialized)
struct QwenSessionCacheStoreConcurrencyProbe {
    private let MiB = 1024 * 1024
    private let GiB = 1024 * 1024 * 1024
    private struct Payload { let depth: Int }

    /// Flat recurrent cost of one checkpoint, as the worker charges it.
    private let roundBytes = 144 * 1024 * 1024

    /// Conversation shape taken from the live run: four conversations, five
    /// rounds, distinct leading header, growing by a tool-result block a turn.
    /// `sharedPrefix` is how many leading tokens every conversation holds in
    /// common before its own content starts. 32 models a real jcode session,
    /// which opens with a per-session date/cwd block and so diverges almost
    /// immediately. A large value models the "one long shared system prompt"
    /// shape, where cross-conversation prefix reuse is actually possible.
    private func conversations(
        count: Int, rounds: Int, firstTurn: Int, growth: Int,
        sharedPrefix: Int
    ) -> [[[Int]]] {
        (0 ..< count).map { c in
            (0 ..< rounds).map { r in
                let n = firstTurn + r * growth
                return (0 ..< n).map { i in
                    i < sharedPrefix ? 700_000 + i : 900_000 * (c + 1) + i
                }
            }
        }
    }

    private func replay(
        budgetBytes: Int, kvBytesPerToken: Int, interleaved: Bool,
        useDeepestPrefixMatch: Bool, sharedPrefix: Int
    ) -> (prefilled: Int, resumes: [String: Int]) {
        let store = QwenSessionCacheStore<Payload>(budgetBytes: budgetBytes)
        let convs = conversations(
            count: 4, rounds: 5, firstTurn: 6571, growth: 5133,
            sharedPrefix: sharedPrefix)
        var order: [(Int, Int)] = []
        if interleaved {
            for r in 0 ..< 5 { for c in 0 ..< 4 { order.append((c, r)) } }
        } else {
            for c in 0 ..< 4 { for r in 0 ..< 5 { order.append((c, r)) } }
        }

        var prefilled = 0
        var resumes = ["bestMatch": 0, "chunkMatch": 0,
                       "prefixScan": 0, "miss": 0]
        var live: [Int] = []
        for (c, r) in order {
            let tokens = convs[c][r]
            let cid = sharedPrefix >= 128 ? "shared" : "conv-\(c)"
            // Turn boundaries: one per round, as a chat client produces.
            let boundaries = (0 ... r).map { convs[c][$0].count }
                .filter { $0 <= tokens.count }
            let keys = QwenPrefillChunking.chainKeys(
                for: tokens, boundaries: boundaries)

            var best = 0
            // Path 0: the serve-level live extend, which needs no store.
            if !live.isEmpty, live.count < tokens.count,
               Array(tokens.prefix(live.count)) == live {
                best = live.count
            } else if let hit = store.bestMatch(
                conversation: cid, incoming: tokens) {
                best = hit.round.tokenCount; resumes["bestMatch"]! += 1
            } else if let hit = store.chunkMatch(
                keys: keys, incoming: tokens) {
                best = hit.round.tokenCount; resumes["chunkMatch"]! += 1
            } else if useDeepestPrefixMatch,
                      let hit = store.deepestPrefixMatch(incoming: tokens) {
                best = hit.round.tokenCount; resumes["prefixScan"]! += 1
            } else {
                resumes["miss"]! += 1
            }
            prefilled += tokens.count - best

            // Record exactly what the worker records: a chunk checkpoint per
            // boundary (KV charged as the delta along the chain) and the
            // conversation round (KV charged at its high-water mark).
            var previousKV = 0
            for key in keys where !store.hasChunk(key: key.key) {
                let kv = key.tokenCount * kvBytesPerToken
                store.recordChunk(
                    key: key.key,
                    tokens: Array(tokens.prefix(key.tokenCount)),
                    state: Payload(depth: key.tokenCount),
                    roundBytes: roundBytes,
                    kvBytes: Swift.max(0, kv - previousKV))
                previousKV = kv
            }
            store.record(
                conversation: cid, tokens: tokens,
                state: Payload(depth: tokens.count),
                roundBytes: roundBytes,
                kvBytes: tokens.count * kvBytesPerToken)
            live = tokens
        }
        return (prefilled, resumes)
    }

    @Test("probe: prefill cost by order, KV width and budget")
    func probe() {
        func pad(_ s: String, _ n: Int) -> String {
            s.count >= n ? s : s + String(repeating: " ", count: n - s.count)
        }
        for (shapeLabel, shared) in [("distinct openings (real jcode)", 32),
                                     ("8k shared system prompt", 8192)] {
            print("\n=== QwenSessionCacheStore probe: \(shapeLabel) ===")
            print(pad("order", 12) + pad("KV/tok", 8) + pad("budget", 8)
                + pad("prefill", 12) + pad("no-scan", 12) + "resume paths")
            for (order, interleaved) in [("sequential", false),
                                         ("interleaved", true)] {
                for (kvLabel, kv) in [("64 KiB", 64 * 1024),
                                      ("32 KiB", 32 * 1024)] {
                    for (budgetLabel, budget) in [("8 GiB", 8 * GiB),
                                                  ("32 GiB", 32 * GiB),
                                                  ("64 GiB", 64 * GiB)] {
                        let hit = replay(
                            budgetBytes: budget, kvBytesPerToken: kv,
                            interleaved: interleaved,
                            useDeepestPrefixMatch: true,
                            sharedPrefix: shared)
                        let without = replay(
                            budgetBytes: budget, kvBytesPerToken: kv,
                            interleaved: interleaved,
                            useDeepestPrefixMatch: false,
                            sharedPrefix: shared)
                        print(pad(order, 12) + pad(kvLabel, 8)
                            + pad(budgetLabel, 8)
                            + pad("\(hit.prefilled)", 12)
                            + pad("\(without.prefilled)", 12)
                            + "\(hit.resumes)")
                    }
                }
            }
            print("=== end probe ===")
        }
    }
}
