import Foundation

/// Policy for prompt-lookup drafting: how much context a suffix match must
/// agree on, and how many tokens each agreement length is allowed to propose.
///
/// LOCAL SERVE FORK ONLY. The ranked track excludes input-derived drafting, and
/// nothing in this file is packaged by a submission: `MLXFastCore` sits outside
/// `benchmark.json` `editablePaths`.
public struct NGramPromptLookupConfiguration: Equatable, Sendable {
    /// The width of the gram the site table is keyed on, and the shortest
    /// agreement any proposal may rest on.
    public let minimumOrder: Int
    /// How far back the agreement walk measures. Beyond this the extra
    /// evidence does not change the rung, so the walk stops.
    public let maximumOrder: Int
    /// How many recent sites are retained per gram. More sites means a better
    /// chance of finding the longest agreement rather than only the most
    /// recent one; the cost is four integers per distinct gram.
    public let candidateSiteLimit: Int
    /// Draft counts a proposal may take, ascending. Each rung plus one is a
    /// verify width, and every shipped rung lands on a MEASURED cost point
    /// (widths 4, 9, 16 and 32) so the round's price is known rather than
    /// interpolated.
    public let ladder: [Int]
    /// `ladderThresholds[i]` is the shortest matched suffix that unlocks
    /// `ladder[i]`, ascending. Longer agreement buys a wider round.
    public let ladderThresholds: [Int]

    public init(
        minimumOrder: Int,
        maximumOrder: Int,
        candidateSiteLimit: Int,
        ladder: [Int],
        ladderThresholds: [Int]
    ) {
        precondition(minimumOrder >= 1, "the gram order must be positive")
        precondition(
            maximumOrder >= minimumOrder,
            "the maximum order must not be below the minimum order")
        precondition(
            candidateSiteLimit >= 1, "at least one site must be retained")
        precondition(
            ladder.count == ladderThresholds.count && !ladder.isEmpty,
            "the ladder and its thresholds must be non-empty and equal length")
        precondition(
            Self.isStrictlyAscending(ladder)
                && Self.isStrictlyAscending(ladderThresholds),
            "the ladder and its thresholds must be strictly ascending")
        precondition(
            ladder[0] >= 1
                && ladder[ladder.count - 1]
                    <= NGramPromptLookupIndex.maximumSupportedDrafts,
            "the ladder must stay within 1 ... "
                + "\(NGramPromptLookupIndex.maximumSupportedDrafts) drafts")
        self.minimumOrder = minimumOrder
        self.maximumOrder = maximumOrder
        self.candidateSiteLimit = candidateSiteLimit
        self.ladder = ladder
        self.ladderThresholds = ladderThresholds
    }

    /// The shipped policy. The thresholds are the starting values the phase-0
    /// corpus measurement either confirms or replaces; the ladder is fixed by
    /// the measured verify-cost table and should not move without a fresh
    /// width sweep.
    public static let shipped = NGramPromptLookupConfiguration(
        minimumOrder: 3,
        maximumOrder: 12,
        candidateSiteLimit: 4,
        ladder: [3, 8, 15, 31],
        ladderThresholds: [3, 5, 8, 12])

    /// Read the policy from the process environment, or nil for "off".
    ///
    /// `DARKBLOOM_` and not `MLXFAST_`: the runtime worker starts from an
    /// empty environment and admits only an allowlist, in which `DARKBLOOM_`
    /// is a permitted prefix and `MLXFAST_` is deliberately excluded. An
    /// `MLXFAST_` name would be dropped at spawn and the knob would look
    /// applied while doing nothing.
    ///
    ///     DARKBLOOM_QWEN_LOOKUP_DRAFT        "1" enables; anything else, or
    ///                                        absent, leaves the feature off
    ///     DARKBLOOM_QWEN_LOOKUP_LADDER       comma-separated draft counts,
    ///                                        ascending, default "3,8,15,31"
    ///     DARKBLOOM_QWEN_LOOKUP_THRESHOLDS   comma-separated matched-suffix
    ///                                        thresholds, ascending, default
    ///                                        "3,5,8,12"
    ///
    /// The overrides exist so a threshold sweep can hold ONE worker binary
    /// across every sample. Rebuilding the worker to move a constant
    /// invalidates every persisted prefill checkpoint and costs a cold
    /// multi-minute prefill per sample.
    ///
    /// A malformed override FAILS CLOSED with a named reason on standard
    /// error, rather than silently reverting to the shipped policy: a sweep
    /// that quietly measured the default twice is worse than one that stops.
    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> NGramPromptLookupConfiguration? {
        guard environment["DARKBLOOM_QWEN_LOOKUP_DRAFT"] == "1" else {
            return nil
        }
        let ladder = parseList(
            environment["DARKBLOOM_QWEN_LOOKUP_LADDER"],
            default: shipped.ladder,
            name: "DARKBLOOM_QWEN_LOOKUP_LADDER")
        let thresholds = parseList(
            environment["DARKBLOOM_QWEN_LOOKUP_THRESHOLDS"],
            default: shipped.ladderThresholds,
            name: "DARKBLOOM_QWEN_LOOKUP_THRESHOLDS")
        guard let ladder, let thresholds else { return nil }
        guard ladder.count == thresholds.count else {
            refuse(
                "DARKBLOOM_QWEN_LOOKUP_LADDER has \(ladder.count) entries and "
                    + "DARKBLOOM_QWEN_LOOKUP_THRESHOLDS has "
                    + "\(thresholds.count); they must agree")
            return nil
        }
        guard isStrictlyAscending(ladder), isStrictlyAscending(thresholds),
              ladder[0] >= 1, thresholds[0] >= 1,
              ladder[ladder.count - 1]
                  <= NGramPromptLookupIndex.maximumSupportedDrafts
        else {
            refuse(
                "the lookup ladder must be strictly ascending, positive, and "
                    + "at most "
                    + "\(NGramPromptLookupIndex.maximumSupportedDrafts) drafts "
                    + "wide; got \(ladder) with thresholds \(thresholds)")
            return nil
        }
        return NGramPromptLookupConfiguration(
            minimumOrder: shipped.minimumOrder,
            maximumOrder: shipped.maximumOrder,
            candidateSiteLimit: shipped.candidateSiteLimit,
            ladder: ladder,
            ladderThresholds: thresholds)
    }

    /// nil means "the variable was present and unusable"; the default is
    /// returned only when the variable is absent entirely.
    private static func parseList(
        _ raw: String?, default fallback: [Int], name: String
    ) -> [Int]? {
        guard let raw else { return fallback }
        let parts = raw.split(separator: ",", omittingEmptySubsequences: false)
        let values = parts.compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard !values.isEmpty, values.count == parts.count else {
            refuse("\(name)='\(raw)' is not a comma-separated integer list")
            return nil
        }
        return values
    }

    private static func refuse(_ reason: String) {
        FileHandle.standardError.write(Data(
            ("mlxfast: prompt-lookup drafting refused: " + reason
                + "; the feature stays off\n").utf8))
    }

    static func isStrictlyAscending(_ values: [Int]) -> Bool {
        for index in 1 ..< Swift.max(values.count, 1)
        where values[index] <= values[index - 1] {
            return false
        }
        return true
    }
}

/// Incrementally maintained prompt-lookup index over one decode session's
/// token history.
///
/// The index answers one question: given the tokens emitted so far, where did
/// this exact recent context occur before, and what followed it there. It
/// PROPOSES only. The target verify decides every emitted token, so a wrong
/// proposal costs a rejected row and can never change output.
///
/// Cost is O(1) amortised per appended token: one gram hash, one dictionary
/// write. Memory is the history itself (eight bytes per token) plus roughly
/// forty bytes per distinct gram.
public final class NGramPromptLookupIndex {
    /// The widest round this index will ever propose, and the value
    /// `Qwen36MTPLimits.maxLookupDepth` must agree with.
    public static let maximumSupportedDrafts = 31

    public let configuration: NGramPromptLookupConfiguration

    private var history: [Int] = []
    /// Gram hash to the continuation start positions of its most recent
    /// occurrences, oldest first.
    private var sites: [UInt64: [Int]] = [:]

    public init(configuration: NGramPromptLookupConfiguration = .shipped) {
        self.configuration = configuration
    }

    public var tokenCount: Int { history.count }

    /// Rebuild from a complete history. Used at the start of every turn: the
    /// worker always holds the whole prompt even when the session resumed its
    /// caches from a checkpoint and therefore forwarded only a suffix.
    public func reset(to tokens: [Int]) {
        history.removeAll(keepingCapacity: true)
        sites.removeAll(keepingCapacity: true)
        history.reserveCapacity(tokens.count)
        for token in tokens { append(token) }
    }

    public func append(_ tokens: [Int]) {
        for token in tokens { append(token) }
    }

    public func append(_ token: Int) {
        history.append(token)
        let order = configuration.minimumOrder
        guard history.count >= order else { return }
        let gramStart = history.count - order
        let key = Self.gramKey(history, at: gramStart, order: order)
        var bucket = sites[key] ?? []
        // The continuation of this gram starts where the gram ends. The newest
        // entry has no continuation yet; `longestMatch` filters it out.
        bucket.append(gramStart + order)
        if bucket.count > configuration.candidateSiteLimit {
            bucket.removeFirst(bucket.count - configuration.candidateSiteLimit)
        }
        sites[key] = bucket
    }

    public struct Match: Equatable {
        /// Index into the history where the earlier continuation begins.
        public let continuationStart: Int
        /// How many tokens of context the earlier site agrees with, capped at
        /// `configuration.maximumOrder`.
        public let matchedSuffixLength: Int
    }

    /// The retained site whose backward agreement with the current suffix is
    /// longest, breaking ties toward the most recent site.
    public func longestMatch() -> Match? {
        let order = configuration.minimumOrder
        guard history.count >= order else { return nil }
        let key = Self.gramKey(
            history, at: history.count - order, order: order)
        guard let bucket = sites[key] else { return nil }
        var best: Match?
        for site in bucket where site < history.count {
            let length = backwardAgreement(endingBefore: site)
            // A shorter agreement than the gram order means the bucket was
            // reached through a hash collision. Discard it rather than draft
            // from context that does not actually match.
            guard length >= order else { continue }
            if let current = best, length < current.matchedSuffixLength {
                continue
            }
            best = Match(continuationStart: site, matchedSuffixLength: length)
        }
        return best
    }

    public struct Proposal: Equatable {
        /// Continuation tokens copied verbatim from the earlier site, in
        /// order. Exactly one ladder rung long.
        public let tokens: [Int]
        public let matchedSuffixLength: Int

        /// Spelled out rather than left to the memberwise initializer, which a
        /// public struct declares as internal and which the tests in the
        /// consuming module therefore cannot call.
        public init(tokens: [Int], matchedSuffixLength: Int) {
            self.tokens = tokens
            self.matchedSuffixLength = matchedSuffixLength
        }
    }

    /// The drafts this round should propose, or nil when nothing qualifies.
    ///
    /// The count is always a ladder rung, never the raw amount of context that
    /// happens to be available. A width off the ladder would be a verify shape
    /// the warm never compiled, and the first round to hit it would pay a
    /// Metal pipeline compile inside the request.
    public func propose() -> Proposal? {
        guard let match = longestMatch() else { return nil }
        guard let unlocked = rung(
            forMatchedSuffixLength: match.matchedSuffixLength)
        else { return nil }
        // Bounded by where the CURRENT occurrence of the matched suffix
        // begins, not by the end of history. Tokens at or past that point
        // only reproduce the matched context itself rather than offering a
        // genuinely earlier continuation to draft from.
        let currentMatchStart = history.count - match.matchedSuffixLength
        let available = currentMatchStart - match.continuationStart
        let allowed = Swift.min(unlocked, available)
        guard let count = configuration.ladder.last(where: { $0 <= allowed })
        else { return nil }
        let end = match.continuationStart + count
        return Proposal(
            tokens: Array(history[match.continuationStart ..< end]),
            matchedSuffixLength: match.matchedSuffixLength)
    }

    /// The widest rung this agreement length unlocks, or nil below the lowest
    /// threshold.
    func rung(forMatchedSuffixLength length: Int) -> Int? {
        var unlocked: Int?
        for (index, threshold) in configuration.ladderThresholds.enumerated()
        where length >= threshold {
            unlocked = configuration.ladder[index]
        }
        return unlocked
    }

    /// How many tokens immediately before `site` equal the tokens immediately
    /// before the end of the history.
    ///
    /// `site < history.count` is guaranteed by the caller, so the read index
    /// into the current suffix is always strictly greater than the read index
    /// into the earlier one and cannot run off the front first.
    private func backwardAgreement(endingBefore site: Int) -> Int {
        var length = 0
        var earlier = site - 1
        var current = history.count - 1
        while earlier >= 0,
              length < configuration.maximumOrder,
              history[earlier] == history[current]
        {
            length += 1
            earlier -= 1
            current -= 1
        }
        return length
    }

    /// FNV-1a over the gram's token bytes. A collision costs accept rate and
    /// nothing else: `longestMatch` re-compares the actual tokens before it
    /// trusts a site.
    private static func gramKey(
        _ tokens: [Int], at start: Int, order: Int
    ) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for index in start ..< (start + order) {
            var value = UInt64(bitPattern: Int64(tokens[index]))
            for _ in 0 ..< 8 {
                hash = (hash ^ (value & 0xff)) &* 0x100_0000_01b3
                value >>= 8
            }
        }
        return hash
    }
}
