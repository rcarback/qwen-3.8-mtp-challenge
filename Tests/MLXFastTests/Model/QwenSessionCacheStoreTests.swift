import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXFastModel

@Suite(.serialized)
struct QwenSessionCacheStoreTests {
    private let MiB = 1024 * 1024
    private let GiB = 1024 * 1024 * 1024

    /// A stand-in payload: the store never inspects it, only its declared
    /// byte cost, so the logic is testable without a 14 GiB model.
    private struct Payload { let tag: Int }
    private func state(_ megabytes: Int) -> Payload { Payload(tag: megabytes) }

    @Test("default budget is clamped to a quarter of physical memory")
    func budgetClamp() {
        #expect(QwenSessionCacheBudget.clampedDefault(
            physicalMemory: 128 * GiB) == 32 * GiB)
        #expect(QwenSessionCacheBudget.clampedDefault(
            physicalMemory: 512 * GiB) == 64 * GiB)   // capped by the default
        #expect(QwenSessionCacheBudget.clampedDefault(
            physicalMemory: 64 * GiB) == 16 * GiB)
    }

    @Test("deepest usable round wins; an equal-length prompt falls back")
    func matching() {
        let store = QwenSessionCacheStore<Payload>(budgetBytes: 1 * GiB)
        store.record(conversation: "c", tokens: [1, 2, 3],
                     state: state(1), roundBytes: MiB, kvBytes: MiB)
        store.record(conversation: "c", tokens: [1, 2, 3, 4, 5],
                     state: state(1), roundBytes: MiB, kvBytes: MiB)

        let hit = store.bestMatch(conversation: "c", incoming: [1, 2, 3, 4, 5, 6])
        #expect(hit?.round.tokenCount == 5, "should resume from the DEEPEST round")
        #expect(hit?.tail == [6])

        // Diverges at index 2: only the 3-token round is a valid prefix... and
        // it is not, because token 3 differs.
        #expect(store.bestMatch(conversation: "c", incoming: [1, 2, 9, 9]) == nil)
        // An equal-length prompt cannot resume from the round that ENDS there
        // (no row left to decode from), but unlike the single-entry
        // `ServePrefixDecision` it still resumes from an earlier round rather
        // than restarting -- 2 tokens of prefill instead of the whole prompt.
        let equalLength = store.bestMatch(
            conversation: "c", incoming: [1, 2, 3, 4, 5])
        #expect(equalLength?.round.tokenCount == 3)
        #expect(equalLength?.tail == [4, 5])
        #expect(store.bestMatch(conversation: "other", incoming: [1, 2, 3]) == nil)
    }

    @Test("a rewind resumes from an earlier round instead of restarting")
    func rewind() {
        let store = QwenSessionCacheStore<Payload>(budgetBytes: 1 * GiB)
        store.record(conversation: "c", tokens: [1, 2, 3],
                     state: state(1), roundBytes: MiB, kvBytes: MiB)
        store.record(conversation: "c", tokens: [1, 2, 3, 4, 5],
                     state: state(1), roundBytes: MiB, kvBytes: MiB)
        // The user edits turn 2: the prompt now diverges after token 3.
        let hit = store.bestMatch(conversation: "c", incoming: [1, 2, 3, 7, 8])
        #expect(hit?.round.tokenCount == 3, "must fall back to the last common round")
        #expect(hit?.tail == [7, 8])
    }

    @Test("eviction drops whole conversations, least recently used first")
    func eviction() {
        // 3 x 100 MiB rounds fit; the fourth must evict.
        let store = QwenSessionCacheStore<Payload>(budgetBytes: 350 * MiB)
        for name in ["a", "b", "c"] {
            store.record(conversation: name, tokens: [1],
                         state: state(100), roundBytes: 100 * MiB, kvBytes: 0)
        }
        #expect(store.conversationCount == 3)
        _ = store.bestMatch(conversation: "a", incoming: [1, 2])   // touch "a"
        store.record(conversation: "d", tokens: [1],
                     state: state(100), roundBytes: 100 * MiB, kvBytes: 0)
        #expect(store.conversationCount == 3)
        #expect(store.bestMatch(conversation: "b", incoming: [1, 2]) == nil,
                "b was least recently used and should be gone")
        #expect(store.bestMatch(conversation: "a", incoming: [1, 2]) != nil,
                "a was touched and should survive")
        #expect(store.currentBytes <= 350 * MiB)
    }

    @Test("the protected conversation sheds old rounds but keeps its newest")
    func shedsRoundsNotItself() {
        let store = QwenSessionCacheStore<Payload>(budgetBytes: 250 * MiB)
        for n in 1 ... 5 {
            store.record(conversation: "solo", tokens: Array(1 ... n),
                         state: state(100), roundBytes: 100 * MiB, kvBytes: 0)
        }
        #expect(store.conversationCount == 1)
        #expect(store.currentBytes <= 250 * MiB)
        #expect(store.bestMatch(conversation: "solo",
                                incoming: Array(1 ... 6)) != nil,
                "the newest round must survive eviction")
    }

    @Test("a quantized cache reports fewer bytes than a bfloat16 one")
    func quantizedCacheIsSmaller() {
        // 4 kv heads, 256 head dim, 512 positions: the shape one full
        // attention layer holds.
        let shape = [1, 4, 512, 256]
        let plain = KVCacheSimple()
        _ = plain.update(
            keys: MLXArray.zeros(shape, dtype: .bfloat16),
            values: MLXArray.zeros(shape, dtype: .bfloat16))

        let quantized = QuantizedKVCache(groupSize: 64, bits: 4)
        _ = quantized.updateQuantized(
            keys: MLXArray.zeros(shape, dtype: .bfloat16),
            values: MLXArray.zeros(shape, dtype: .bfloat16))

        let plainBytes = plain.state.reduce(0) { $0 + $1.nbytes }
        let quantizedBytes = quantized.state.reduce(0) { $0 + $1.nbytes }
        // 16 bits down to 4.5 bits per element is a factor of 3.56. Assert a
        // conservative factor of 3 so buffer slack does not make this flaky.
        #expect(quantizedBytes * 3 < plainBytes)
    }

    // MARK: - Content-addressed prefill checkpoints

    @Test("chain keys cover complete chunks only, and identify the prefix")
    func chainKeysCoverCompleteChunks() {
        let tokens = Array(0 ..< 10)
        let keys = QwenPrefillChunking.chainKeys(for: tokens, chunkSize: 4)
        // 10 tokens at chunk 4 is two complete chunks; the 2-token tail gets
        // no key, because the next prompt will not end where this one did.
        #expect(keys.map(\.tokenCount) == [4, 8])

        // A stream agreeing on the first 4 tokens shares key 0 and NOT key 1:
        // the key is chained over the whole prefix, not over its own chunk.
        var divergent = tokens
        divergent[5] = 999
        let other = QwenPrefillChunking.chainKeys(for: divergent, chunkSize: 4)
        #expect(other[0].key == keys[0].key)
        #expect(other[1].key != keys[1].key)
    }

    @Test("chunk size zero disables checkpointing")
    func chunkingDisabled() {
        #expect(QwenPrefillChunking
            .chainKeys(for: Array(0 ..< 100), chunkSize: 0).isEmpty)
    }

    @Test("a shared prefix resumes across different conversations")
    func sharedPrefixMatchesAcrossConversations() {
        let store = QwenSessionCacheStore<Payload>(budgetBytes: 8 * GiB)
        // One client prefills a long prompt and checkpoints at 4 and 8.
        let first = Array(0 ..< 12)
        for entry in QwenPrefillChunking.chainKeys(for: first, chunkSize: 4) {
            store.recordChunk(
                key: entry.key, tokens: Array(first.prefix(entry.tokenCount)),
                state: state(entry.tokenCount),
                roundBytes: 144 * MiB, kvBytes: MiB)
        }

        // A DIFFERENT client sends the same first 9 tokens then diverges. It
        // never supplies a conversation id matching the first client's.
        var second = Array(0 ..< 12)
        second[9] = 777
        let keys = QwenPrefillChunking.chainKeys(for: second, chunkSize: 4)
        let hit = store.chunkMatch(keys: keys, incoming: second)
        // Deepest usable checkpoint is 8: the divergence at index 9 is past it.
        #expect(hit?.round.tokenCount == 8)
        #expect(hit?.tail == Array(second.dropFirst(8)))
    }

    @Test("an early divergence falls back to the shallower checkpoint")
    func earlyDivergenceFallsBack() {
        let store = QwenSessionCacheStore<Payload>(budgetBytes: 8 * GiB)
        let first = Array(0 ..< 12)
        for entry in QwenPrefillChunking.chainKeys(for: first, chunkSize: 4) {
            store.recordChunk(
                key: entry.key, tokens: Array(first.prefix(entry.tokenCount)),
                state: state(entry.tokenCount),
                roundBytes: 144 * MiB, kvBytes: MiB)
        }
        var second = Array(0 ..< 12)
        second[5] = 777          // inside the SECOND chunk
        let keys = QwenPrefillChunking.chainKeys(for: second, chunkSize: 4)
        #expect(store.chunkMatch(keys: keys, incoming: second)?
            .round.tokenCount == 4)

        var third = Array(0 ..< 12)
        third[1] = 777           // inside the FIRST chunk: nothing is reusable
        let thirdKeys = QwenPrefillChunking.chainKeys(for: third, chunkSize: 4)
        #expect(store.chunkMatch(keys: thirdKeys, incoming: third) == nil)
    }

    @Test("re-recording a checkpoint does not charge for it twice")
    func recordChunkIsIdempotent() {
        let store = QwenSessionCacheStore<Payload>(budgetBytes: 8 * GiB)
        let tokens = Array(0 ..< 8)
        let entry = QwenPrefillChunking
            .chainKeys(for: tokens, chunkSize: 4)[0]
        for _ in 0 ..< 5 {
            store.recordChunk(
                key: entry.key, tokens: Array(tokens.prefix(4)),
                state: state(4), roundBytes: 144 * MiB, kvBytes: MiB)
        }
        // Five identical requests, one retained checkpoint. Without the guard
        // each would append another 144 MiB round to the same entry.
        #expect(store.currentBytes == 144 * MiB + MiB)
    }

    @Test("a checkpoint ending exactly at the prompt end is not usable")
    func exactLengthCheckpointRejected() {
        let store = QwenSessionCacheStore<Payload>(budgetBytes: 8 * GiB)
        let tokens = Array(0 ..< 8)
        let keys = QwenPrefillChunking.chainKeys(for: tokens, chunkSize: 4)
        for entry in keys {
            store.recordChunk(
                key: entry.key, tokens: Array(tokens.prefix(entry.tokenCount)),
                state: state(entry.tokenCount),
                roundBytes: 144 * MiB, kvBytes: MiB)
        }
        // The 8-token checkpoint ends where this prompt ends, leaving no row
        // to read the next token from, so the 4-token one wins.
        #expect(store.chunkMatch(keys: keys, incoming: tokens)?
            .round.tokenCount == 4)
    }

    @Test("checkpoints are spent before any conversation is evicted")
    func checkpointsEvictBeforeConversations() {
        let store = QwenSessionCacheStore<Payload>(budgetBytes: 600 * MiB)
        // A live client's resume point, recorded FIRST so it is also the least
        // recently used -- the property that would doom it under a plain LRU.
        store.record(
            conversation: "live-client", tokens: [1, 2, 3], state: state(1),
            roundBytes: 144 * MiB, kvBytes: MiB)

        // Three checkpoints, enough to push past the budget.
        for index in 0 ..< 3 {
            store.recordChunk(
                key: "k\(index)", tokens: Array(0 ... index),
                state: state(index), roundBytes: 144 * MiB, kvBytes: MiB)
        }

        // The conversation survives: checkpoints are recomputable, its resume
        // point is not.
        #expect(store.bestMatch(
            conversation: "live-client", incoming: [1, 2, 3, 4]) != nil)
        #expect(store.currentBytes <= 600 * MiB)
    }

    @Test("supplied turn boundaries place the checkpoints")
    func turnBoundariesPlaceCheckpoints() {
        let tokens = Array(0 ..< 100)
        // Turn ends at 30, 55 and 90. With a chunk of 20 all three clear the
        // stride, so all three earn a checkpoint -- boundaries a fixed stride
        // would have put at 20, 40, 60, 80, none of which is a turn end.
        let keys = QwenPrefillChunking.chainKeys(
            for: tokens, boundaries: [30, 55, 90], chunkSize: 20)
        #expect(keys.map(\.tokenCount) == [30, 55, 90])
    }

    @Test("boundaries closer together than a chunk are thinned")
    func closeBoundariesAreThinned() {
        let tokens = Array(0 ..< 100)
        // A burst of short turns. Each would otherwise cost 144 MiB to save at
        // most a chunk of prefill.
        let keys = QwenPrefillChunking.chainKeys(
            for: tokens, boundaries: [10, 12, 14, 40, 42, 80],
            chunkSize: 20)
        #expect(keys.map(\.tokenCount) == [40, 80])
    }

    @Test("the final boundary survives thinning however close it sits")
    func finalBoundaryAlwaysKept() {
        let tokens = Array(0 ..< 100)
        // 82 is only 2 tokens past 80, so uniform thinning would drop it -- and
        // it is the boundary a prompt diverging at 83 would actually resume
        // from. Losing it costs a whole chunk of re-prefill to recover from a
        // difference of two tokens.
        let keys = QwenPrefillChunking.chainKeys(
            for: tokens, boundaries: [40, 80, 82], chunkSize: 20)
        #expect(keys.map(\.tokenCount) == [40, 80, 82])
    }

    @Test("boundary keys still identify the whole prefix")
    func boundaryKeysChainOverThePrefix() {
        let first = Array(0 ..< 100)
        var second = first
        second[45] = 999          // between boundary 30 and boundary 55
        let bounds = [30, 55, 90]
        let a = QwenPrefillChunking.chainKeys(
            for: first, boundaries: bounds, chunkSize: 20)
        let b = QwenPrefillChunking.chainKeys(
            for: second, boundaries: bounds, chunkSize: 20)
        #expect(a[0].key == b[0].key)     // agree up to 30
        #expect(a[1].key != b[1].key)     // diverge before 55
        #expect(a[2].key != b[2].key)     // and stay diverged
    }

    @Test("out-of-range boundaries are dropped, not trusted")
    func outOfRangeBoundariesDropped() {
        let tokens = Array(0 ..< 50)
        let keys = QwenPrefillChunking.chainKeys(
            for: tokens, boundaries: [-5, 0, 25, 900], chunkSize: 10)
        #expect(keys.map(\.tokenCount) == [25])
    }
}
