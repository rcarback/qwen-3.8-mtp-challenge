import Foundation
import MLX
import MLXLMCommon

/// Byte-budgeted LRU over retained decode state, so an interrupt, a failed
/// request, a rewind, or a switch between concurrent conversations does not
/// cost a full re-prefill.
///
/// WHY THIS EXISTS. `ServePrefixDecision` keeps exactly ONE history and
/// restarts on anything that is not a strict extension of it. A restart at 20k
/// context costs ~200-300 s of prefill against ~0.4 s for an extension, so a
/// single interrupt, edited message, or failed request converts a 2 s turn into
/// a multi-minute one -- and `invalidate()` poisons the following turn too.
///
/// WHAT A ROUND COSTS, which is what makes this affordable. Two very different
/// quantities:
///
///   * The 16 full-attention layers' KV is APPEND-ONLY, at 64 KiB per token
///     (16 layers x 2 x 4 kv-heads x 256 head-dim x 2 B), 32 KiB under
///     `DARKBLOOM_KV_QUANT_BITS=8`. Rounds of one conversation differ by a
///     handful of tokens, and rewinding to an earlier round is just a smaller
///     offset into the same buffer. `KVSnapshotAliasingTests` establishes that
///     a retained `cache.state` slice is copy-on-write and is NOT corrupted by
///     later in-place writes, so retaining a round's slice costs no copy and no
///     extra bytes -- it only keeps the conversation's buffer alive.
///   * The 48 gated-delta layers' recurrent state is DENSE and changes every
///     token, so it cannot be shared or paged. It is a flat 144 MiB per round
///     (48 layers x [1, 48, 128, 128] fp32), independent of context length.
///
/// So a conversation costs `tokens x 64 KiB` once, plus 144 MiB per retained
/// round. Measured against the naive full-copy-per-round alternative that is
/// 4.6x cheaper at 20k with 8 rounds and 7.5x at 262k -- and it inverts the
/// tradeoff: retaining more ROUNDS is nearly free, while more CONTEXT is
/// expensive. Keep deep round history; be stingy about concurrent long
/// conversations.
///
/// EVICTION IS AT CONVERSATION GRANULARITY, deliberately. Dropping one round
/// frees only its 144 MiB: the shared KV buffer stays alive as long as any
/// round of that conversation (or the live session) still references it. Only
/// evicting the whole conversation returns the context bytes.
/// Generic over the retained payload so the eviction and prefix-matching
/// logic can be tested without loading a 14 GiB model. The worker instantiates
/// it as `QwenSessionCacheStore<Qwen36MTPBlockSession.SessionSnapshot>`.
public final class QwenSessionCacheStore<Payload>: @unchecked Sendable {
    /// One retained round: everything needed to resume decoding at exactly
    /// this many committed tokens.
    public struct Round {
        /// Committed token count this round ends at. The KV slices are already
        /// cut to it; this is carried for the prefix match and for accounting.
        public let tokenCount: Int
        /// The retained resume point. Full-attention KV rides along as
        /// copy-on-write slices (free); the gated-delta recurrent arrays are
        /// the real cost.
        public let state: Payload
        /// Charged bytes for this round ALONE -- the dense recurrent state.
        /// The shared KV is charged once to the conversation.
        public let roundBytes: Int
    }

    public struct Conversation {
        public var tokens: [Int]
        public var rounds: [Round]
        /// Bytes charged for the shared, append-only KV at the conversation's
        /// high-water mark. Charged once, not per round.
        public var kvBytes: Int
        public var lastUsed: UInt64
    }

    /// Default budget. 64 GiB holds roughly 27 concurrent conversations at 20k
    /// context with 8 rounds each, or 3 at the 262144 ceiling (7 under
    /// `DARKBLOOM_KV_QUANT_BITS=8`). It is CLAMPED to a quarter of physical RAM
    /// so the default does not break a smaller machine: on a 128 GiB box the
    /// full 64 GiB stands (78.1 GiB with the 14.1 GiB model, ~50 GiB spare),
    /// while a 64 GiB box gets 16 GiB.
    private let budgetBytes: Int
    private var conversations: [String: Conversation] = [:]
    private var clock: UInt64 = 0
    private let lock = NSLock()
    private var diskRoot: URL?
    private var diskFingerprint: QwenPrefillDiskCache.Fingerprint?

    public init(
        budgetBytes: Int = QwenSessionCacheBudget.clampedDefault()
    ) {
        self.budgetBytes = Swift.max(0, budgetBytes)
    }

    public var currentBytes: Int {
        lock.lock(); defer { lock.unlock() }
        return unsafeCurrentBytes
    }

    private var unsafeCurrentBytes: Int {
        conversations.values.reduce(0) { total, conversation in
            total + conversation.kvBytes
                + conversation.rounds.reduce(0) { $0 + $1.roundBytes }
        }
    }

    public var conversationCount: Int {
        lock.lock(); defer { lock.unlock() }
        return conversations.count
    }

    /// Longest retained round whose token prefix matches `incoming`, or nil.
    ///
    /// Returns the DEEPEST such round: resuming from further along means
    /// prefilling fewer tail tokens, and every round of a conversation shares
    /// the same KV buffer so a deeper one costs no more to hold.
    public func bestMatch(
        conversation id: String, incoming: [Int]
    ) -> (round: Round, tail: [Int])? {
        lock.lock(); defer { lock.unlock() }
        guard var conversation = conversations[id] else { return nil }
        var best: (Round, [Int])?
        for round in conversation.rounds {
            // A round that ends exactly at `incoming.count` leaves no row to
            // read the next token from, the same reason `ServePrefixDecision`
            // treats an equal-length prompt as a restart.
            guard round.tokenCount < incoming.count,
                  round.tokenCount <= conversation.tokens.count,
                  Array(conversation.tokens.prefix(round.tokenCount))
                      == Array(incoming.prefix(round.tokenCount))
            else { continue }
            if best == nil || round.tokenCount > best!.0.tokenCount {
                best = (round, Array(incoming.dropFirst(round.tokenCount)))
            }
        }
        if best != nil {
            clock += 1
            conversation.lastUsed = clock
            conversations[id] = conversation
        }
        return best
    }

    /// Retain a round. `kvBytes` is the conversation's high-water KV cost;
    /// passing a smaller value than already recorded keeps the larger, because
    /// the buffer does not shrink when a round is rewound.
    public func record(
        conversation id: String, tokens: [Int], state: Payload,
        roundBytes: Int, kvBytes: Int
    ) {
        lock.lock(); defer { lock.unlock() }
        clock += 1
        var conversation = conversations[id]
            ?? Conversation(tokens: [], rounds: [], kvBytes: 0, lastUsed: clock)
        conversation.tokens = tokens
        conversation.kvBytes = Swift.max(conversation.kvBytes, kvBytes)
        conversation.rounds.append(
            Round(tokenCount: tokens.count, state: state, roundBytes: roundBytes))
        conversation.lastUsed = clock
        conversations[id] = conversation
        evictToBudget(protecting: id)
    }

    /// Drop a conversation outright (session closed, or its state is unknown
    /// after a failure that this store cannot reason about).
    public func drop(conversation id: String) {
        lock.lock(); defer { lock.unlock() }
        conversations.removeValue(forKey: id)
    }

    /// True when a prefill checkpoint is already retained under `key`.
    ///
    /// Checked before recording, because every request sharing a system prompt
    /// derives the SAME chunk keys. Without this, each one would append another
    /// round to the same entry and leak 144 MiB of recurrent state per turn.
    public func hasChunk(key: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return conversations[QwenPrefillChunking.namespace + key] != nil
    }

    /// Retain a content-addressed prefill checkpoint.
    ///
    /// Stored as a single-round conversation under a namespaced key so it
    /// reuses the existing byte budget and LRU eviction unchanged. Eviction at
    /// conversation granularity is the RIGHT granularity here: a chunk
    /// checkpoint is exactly one round, so LRU over conversations is LRU over
    /// checkpoints.
    public func recordChunk(
        key: String, tokens: [Int], state: Payload,
        roundBytes: Int, kvBytes: Int
    ) {
        // Idempotent. Every request sharing a prefix derives the same keys, so
        // re-recording would append a second round holding another 144 MiB of
        // recurrent state for a checkpoint already retained.
        guard !hasChunk(key: key) else { return }
        record(
            conversation: QwenPrefillChunking.namespace + key, tokens: tokens,
            state: state, roundBytes: roundBytes, kvBytes: kvBytes)
    }

    /// Deepest retained checkpoint whose tokens are a prefix of `incoming`.
    ///
    /// `keys` arrives shallowest-first and is walked in reverse, so the first
    /// hit is the deepest and leaves the least tail to prefill. The token
    /// comparison is not redundant with the key: the key is a hash, and a
    /// collision would otherwise resume from an unrelated cache.
    public func chunkMatch(
        keys: [(key: String, tokenCount: Int)], incoming: [Int]
    ) -> (round: Round, tail: [Int])? {
        lock.lock(); defer { lock.unlock() }
        for entry in keys.reversed() {
            guard var conversation = conversations[
                QwenPrefillChunking.namespace + entry.key] else { continue }
            guard let round = conversation.rounds.first,
                  // A checkpoint ending exactly at `incoming.count` leaves no
                  // row to read the next token from, the same reason
                  // `bestMatch` treats an equal-length prompt as a restart.
                  round.tokenCount < incoming.count,
                  conversation.tokens.count >= round.tokenCount,
                  Array(conversation.tokens.prefix(round.tokenCount))
                      == Array(incoming.prefix(round.tokenCount))
            else { continue }
            clock += 1
            conversation.lastUsed = clock
            conversations[QwenPrefillChunking.namespace + entry.key] = conversation
            return (round, Array(incoming.dropFirst(round.tokenCount)))
        }
        return nil
    }

    /// Enable disk persistence for prefill checkpoints.
    ///
    /// Conversations are deliberately NOT persisted. A conversation holds a
    /// live client's resume point and is meaningless after a restart, whereas
    /// a checkpoint is content-addressed: it belongs to whichever stream
    /// derives the same key next, including a stream that does not exist yet.
    public func attachDisk(
        root: URL, fingerprint: QwenPrefillDiskCache.Fingerprint
    ) {
        lock.lock(); defer { lock.unlock() }
        diskRoot = root
        diskFingerprint = fingerprint
    }

    /// Record a checkpoint to disk. In-memory recording stays the caller's job.
    public func recordChunkPersisting(
        key: String, entry: QwenPrefillDiskCache.CacheEntry
    ) throws {
        lock.lock()
        let root = diskRoot
        let fingerprint = diskFingerprint
        lock.unlock()
        guard let root, let fingerprint else { return }
        try QwenPrefillDiskCache.write(
            entry, key: key, fingerprint: fingerprint, root: root)
    }

    /// Deepest on-disk checkpoint whose tokens are a prefix of `incoming`.
    ///
    /// Walked in reverse for the same reason `chunkMatch` is: the first hit is
    /// the deepest and leaves the least tail to prefill. The token comparison
    /// is not redundant with the key -- the key is a hash, and a collision
    /// would otherwise resume from an unrelated cache.
    public func diskChunkMatch(
        keys: [(key: String, tokenCount: Int)], incoming: [Int]
    ) -> (key: String, entry: QwenPrefillDiskCache.CacheEntry)? {
        lock.lock()
        let root = diskRoot
        let fingerprint = diskFingerprint
        lock.unlock()
        guard let root, let fingerprint else { return nil }
        for entry in keys.reversed() {
            // `read` both throws and returns an optional, so `try?` yields a
            // double optional. Flatten it explicitly rather than relying on
            // shorthand shadowing inside a single guard.
            let found = (try? QwenPrefillDiskCache.read(
                key: entry.key, fingerprint: fingerprint, root: root)) ?? nil
            guard let candidate = found else { continue }
            guard candidate.tokens.count < incoming.count,
                  candidate.tokens == Array(
                      incoming.prefix(candidate.tokens.count))
            else { continue }
            return (entry.key, candidate)
        }
        return nil
    }

    /// Remove a checkpoint from disk. Best-effort: called when a restored
    /// checkpoint fails to rebuild into live caches, so the malformed file is
    /// not hit again on the next request or after a restart.
    public func dropDiskEntry(key: String) {
        lock.lock()
        let root = diskRoot
        lock.unlock()
        guard let root else { return }
        _ = try? FileManager.default.removeItem(
            at: QwenPrefillDiskCache.url(key: key, root: root))
    }

    /// Evict until the budget is met: whole conversations, least recently used
    /// first, because only a whole conversation returns its shared KV bytes.
    /// Within the protected conversation, oldest rounds go first -- that frees
    /// 144 MiB each and keeps the newest resume point.
    private func evictToBudget(protecting id: String) {
        func lruVictim(checkpoints: Bool) -> String? {
            conversations
                .filter {
                    $0.key != id
                        && $0.key.hasPrefix(QwenPrefillChunking.namespace)
                            == checkpoints
                }
                .min { $0.value.lastUsed < $1.value.lastUsed }?
                .key
        }
        while unsafeCurrentBytes > budgetBytes {
            // Prefill checkpoints are the shed-able pool. A checkpoint is pure
            // cache -- losing it costs re-prefill and nothing else -- whereas a
            // conversation holds a live client's only resume point, and
            // dropping that strands the client on a full restart. So spend
            // every checkpoint before touching a single conversation.
            if let victim = lruVictim(checkpoints: true) {
                conversations.removeValue(forKey: victim)
                continue
            }
            if let victim = lruVictim(checkpoints: false) {
                conversations.removeValue(forKey: victim)
                continue
            }
            // Only the protected conversation is left. Shed its oldest rounds,
            // never the last one: dropping that would defeat the purpose.
            guard var only = conversations[id], only.rounds.count > 1 else { return }
            only.rounds.removeFirst()
            conversations[id] = only
        }
    }
}

/// Budget policy, kept out of the generic type because Swift does not allow
/// static stored properties there.
public enum QwenSessionCacheBudget {
    /// 64 GiB holds roughly 27 concurrent conversations at 20k context with 8
    /// rounds each, or 3 at the 262144 ceiling (7 under
    /// `DARKBLOOM_KV_QUANT_BITS=8`).
    public static let defaultBytes = 64 * 1024 * 1024 * 1024

    /// Clamped to a quarter of physical RAM so the default does not break a
    /// smaller machine: a 128 GiB box keeps the full 64 GiB (78.1 GiB with the
    /// 14.1 GiB model, ~50 GiB spare); a 64 GiB box gets 16 GiB.
    public static func clampedDefault(
        physicalMemory: Int = Int(ProcessInfo.processInfo.physicalMemory)
    ) -> Int {
        Swift.min(defaultBytes, physicalMemory / 4)
    }
}


/// Content-addressed prefill checkpointing.
///
/// WHY CONTENT-ADDRESSED. A conversation id identifies a CLIENT; a chunk key
/// identifies TOKENS. Two connections that share a system prompt derive the
/// same keys and so share the same checkpoints, which is the case that matters
/// when several agent sessions run against one server -- they differ in their
/// last few hundred tokens and agree on the first eighteen thousand.
///
/// WHY CHAINED. Key `i` is a hash of every token up to boundary `i`, not of
/// chunk `i` alone. Two streams therefore share key `i` only if they agree on
/// the whole prefix, so a match at key `i` is proof the prefix is identical --
/// up to hash collision, which `chunkMatch` rules out by comparing tokens.
public enum QwenPrefillChunking {
    /// Keeps content-addressed checkpoints from colliding with the
    /// conversation ids the serve layer supplies. Lives here rather than on
    /// the store because a generic type cannot hold a static stored property.
    public static let namespace = "\u{1}chunk:"

    /// Tokens per checkpoint. `DARKBLOOM_PREFILL_CHUNK` overrides it; 0
    /// disables checkpointing entirely.
    ///
    /// The default trades memory against granularity. Each checkpoint costs a
    /// FLAT 144 MiB of gated-delta recurrent state regardless of chunk size --
    /// the append-only attention KV rides along copy-on-write and is free --
    /// so halving the chunk doubles the memory and buys only a shorter
    /// re-prefill tail on a miss.
    public static let chunkSize: Int = {
        guard let raw = ProcessInfo.processInfo
            .environment["DARKBLOOM_PREFILL_CHUNK"],
              let value = Int(raw), value >= 0
        else { return 4096 }
        return value
    }()

    /// One key per boundary, shallowest first, using boundaries the caller
    /// supplies instead of a fixed stride.
    ///
    /// A chat client grows its prompt by whole turns, so a turn boundary is
    /// exactly where the NEXT request stops agreeing with this one. A
    /// fixed-stride boundary lands mid-turn and strands every token after it.
    /// Offsets must be strictly increasing and inside the token range; anything
    /// else is dropped rather than trusted.
    public static func chainKeys(
        for tokens: [Int], boundaries: [Int],
        chunkSize: Int = QwenPrefillChunking.chunkSize
    ) -> [(key: String, tokenCount: Int)] {
        guard chunkSize > 0 else { return [] }
        // Thin the EARLY boundaries and keep the last one whatever its
        // spacing.
        //
        // `chunkMatch` always resumes from the deepest match, so the final
        // boundary is the one that actually gets used; the earlier ones are
        // only insurance against an earlier divergence. Thinning uniformly
        // therefore spends its savings in exactly the wrong place: it can drop
        // the boundary sitting a few tokens before the divergence and force a
        // whole chunk of re-prefill -- ~63 s at the measured 65 tok/s -- to
        // recover from a difference of tens of tokens.
        let stride = Swift.max(1, chunkSize)
        let usable = boundaries.filter { $0 > 0 && $0 <= tokens.count }
        var kept: [Int] = []
        for boundary in usable where boundary > (kept.last ?? 0) + stride - 1 {
            kept.append(boundary)
        }
        if let last = usable.last, kept.last != last {
            kept.append(last)
        }
        var keys: [(key: String, tokenCount: Int)] = []
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        var index = 0
        for boundary in kept where boundary > index {
            for position in index ..< boundary {
                hash = Self.mix(hash, tokens[position])
            }
            index = boundary
            keys.append((String(hash, radix: 36), index))
        }
        return keys
    }

    /// FNV-1a over one token's eight bytes.
    private static func mix(_ hash: UInt64, _ token: Int) -> UInt64 {
        var hash = hash
        var value = UInt64(bitPattern: Int64(token))
        for _ in 0 ..< 8 {
            hash ^= value & 0xff
            hash = hash &* 0x0000_0100_0000_01b3
            value >>= 8
        }
        return hash
    }

    /// One key per COMPLETE chunk, shallowest first.
    ///
    /// A partial trailing chunk gets no key: its boundary is wherever this
    /// particular prompt happened to end, which the next request is unlikely
    /// to land on, so a checkpoint there would cost 144 MiB to serve nobody.
    public static func chainKeys(
        for tokens: [Int], chunkSize: Int = QwenPrefillChunking.chunkSize
    ) -> [(key: String, tokenCount: Int)] {
        guard chunkSize > 0 else { return [] }
        var keys: [(key: String, tokenCount: Int)] = []
        // FNV-1a, chosen over `Hasher` because `Hasher` is randomly seeded per
        // process: its keys would not survive a worker restart, and the whole
        // point of a content address is that it does.
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        var index = 0
        while index + chunkSize <= tokens.count {
            for position in index ..< (index + chunkSize) {
                hash = Self.mix(hash, tokens[position])
            }
            index += chunkSize
            keys.append((String(hash, radix: 36), index))
        }
        return keys
    }
}
