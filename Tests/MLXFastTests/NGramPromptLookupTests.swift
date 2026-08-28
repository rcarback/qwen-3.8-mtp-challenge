@testable import MLXFastCore
import Testing

@Suite
struct NGramPromptLookupIndexTests {
    private func index(
        _ tokens: [Int],
        configuration: NGramPromptLookupConfiguration = .shipped
    ) -> NGramPromptLookupIndex {
        let index = NGramPromptLookupIndex(configuration: configuration)
        index.reset(to: tokens)
        return index
    }

    @Test("a history shorter than the minimum order has no match")
    func shortHistoryHasNoMatch() {
        #expect(index([]).longestMatch() == nil)
        #expect(index([7, 8]).longestMatch() == nil)
    }

    @Test("a history with no repeat has no match")
    func uniqueHistoryHasNoMatch() {
        #expect(index([1, 2, 3, 4, 5, 6]).longestMatch() == nil)
    }

    @Test("a repeated trigram points at the continuation after the earlier site")
    func repeatedTrigramFindsTheEarlierContinuation() throws {
        //          0  1  2  3  4  5  6  7
        let match = try #require(
            index([1, 2, 3, 9, 9, 1, 2, 3]).longestMatch())
        #expect(match.continuationStart == 3)
        #expect(match.matchedSuffixLength == 3)
    }

    @Test("a longer repeat reports the longer matched suffix")
    func longerRepeatReportsTheLongerSuffix() throws {
        //          0  1  2  3  4  5  6  7  8  9 10 11
        let match = try #require(
            index([4, 5, 1, 2, 3, 7, 7, 4, 5, 1, 2, 3]).longestMatch())
        #expect(match.continuationStart == 5)
        #expect(match.matchedSuffixLength == 5)
    }

    @Test("the longer agreement wins over a more recent shorter one")
    func longerAgreementBeatsMoreRecentSite() throws {
        // Site at 5 agrees on five tokens; the more recent site at 11 agrees
        // on only three.
        //          0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
        let tokens = [4, 5, 1, 2, 3, 6, 8, 8, 8, 1, 2, 3, 7, 8, 4, 5, 1, 2, 3]
        let match = try #require(index(tokens).longestMatch())
        #expect(match.continuationStart == 5)
        #expect(match.matchedSuffixLength == 5)
    }

    @Test("the most recent site wins an agreement-length tie")
    func mostRecentSiteWinsATie() throws {
        //          0  1  2  3  4  5  6  7  8  9
        let match = try #require(
            index([1, 2, 3, 7, 1, 2, 3, 8, 1, 2, 3]).longestMatch())
        #expect(match.continuationStart == 7)
        #expect(match.matchedSuffixLength == 3)
    }

    @Test("the matched suffix length is capped at the maximum order")
    func matchedSuffixIsCappedAtTheMaximumOrder() throws {
        let repeated = Array(1 ... 40)
        let match = try #require(index(repeated + repeated).longestMatch())
        #expect(match.matchedSuffixLength
            == NGramPromptLookupConfiguration.shipped.maximumOrder)
    }

    @Test("append and reset agree")
    func appendAndResetAgree() throws {
        let tokens = [1, 2, 3, 9, 9, 1, 2, 3]
        let built = NGramPromptLookupIndex(configuration: .shipped)
        for token in tokens { built.append(token) }
        let reference = index(tokens)
        #expect(built.tokenCount == reference.tokenCount)
        #expect(built.longestMatch() == reference.longestMatch())
    }

    @Test("reset discards the previous history")
    func resetDiscardsPreviousHistory() {
        let built = index([1, 2, 3, 9, 9, 1, 2, 3])
        built.reset(to: [5, 6, 7])
        #expect(built.tokenCount == 3)
        #expect(built.longestMatch() == nil)
    }
}

@Suite
struct NGramPromptLookupProposalTests {
    private func index(_ tokens: [Int]) -> NGramPromptLookupIndex {
        let index = NGramPromptLookupIndex(configuration: .shipped)
        index.reset(to: tokens)
        return index
    }

    @Test("no match means no proposal")
    func noMatchMeansNoProposal() {
        #expect(index([1, 2, 3, 4, 5, 6]).propose() == nil)
    }

    @Test("a three-token agreement takes the lowest rung")
    func threeTokenAgreementTakesTheLowestRung() throws {
        // Agreement 3 unlocks rung 3; eight continuation tokens are available,
        // so the proposal snaps down to the rung rather than to what is left.
        let tokens = [1, 2, 3] + Array(50 ..< 70) + [1, 2, 3]
        let proposal = try #require(index(tokens).propose())
        #expect(proposal.matchedSuffixLength == 3)
        #expect(proposal.tokens == [50, 51, 52])
    }

    @Test("a five-token agreement takes the second rung")
    func fiveTokenAgreementTakesTheSecondRung() throws {
        let head = [90, 91, 1, 2, 3]
        let tokens = head + Array(50 ..< 70) + head
        let proposal = try #require(index(tokens).propose())
        #expect(proposal.matchedSuffixLength == 5)
        #expect(proposal.tokens == Array(50 ..< 58))
    }

    @Test("the proposal snaps down to a ladder rung when context runs out")
    func proposalSnapsDownWhenContextRunsOut() throws {
        // Agreement 5 unlocks rung 8, but only four continuation tokens exist,
        // so the proposal takes rung 3 and never an unwarmed width.
        let head = [90, 91, 1, 2, 3]
        let tokens = head + [50, 51, 52, 53] + head
        let proposal = try #require(index(tokens).propose())
        #expect(proposal.tokens == [50, 51, 52])
    }

    @Test("fewer continuation tokens than the lowest rung means no proposal")
    func tooFewContinuationTokensMeansNoProposal() {
        let head = [90, 91, 1, 2, 3]
        #expect(index(head + [50, 51] + head).propose() == nil)
    }

    @Test("a twelve-token agreement takes the top rung")
    func twelveTokenAgreementTakesTheTopRung() throws {
        let head = Array(200 ..< 215)
        let tokens = head + Array(50 ..< 100) + head
        let proposal = try #require(index(tokens).propose())
        #expect(proposal.matchedSuffixLength == 12)
        #expect(proposal.tokens.count == 31)
        #expect(proposal.tokens == Array(50 ..< 81))
    }
}

@Suite
struct NGramPromptLookupConfigurationTests {
    @Test("an absent flag disables the feature")
    func absentFlagDisablesTheFeature() {
        #expect(NGramPromptLookupConfiguration.fromEnvironment([:]) == nil)
        #expect(NGramPromptLookupConfiguration.fromEnvironment(
            ["DARKBLOOM_QWEN_LOOKUP_DRAFT": "0"]) == nil)
    }

    @Test("the flag alone selects the shipped policy")
    func flagAloneSelectsTheShippedPolicy() {
        #expect(
            NGramPromptLookupConfiguration.fromEnvironment(
                ["DARKBLOOM_QWEN_LOOKUP_DRAFT": "1"])
                == NGramPromptLookupConfiguration.shipped)
    }

    @Test("a well-formed override replaces the ladder and its thresholds")
    func wellFormedOverrideReplacesTheLadder() throws {
        let parsed = try #require(NGramPromptLookupConfiguration.fromEnvironment([
            "DARKBLOOM_QWEN_LOOKUP_DRAFT": "1",
            "DARKBLOOM_QWEN_LOOKUP_LADDER": "4,8",
            "DARKBLOOM_QWEN_LOOKUP_THRESHOLDS": "3,6",
        ]))
        #expect(parsed.ladder == [4, 8])
        #expect(parsed.ladderThresholds == [3, 6])
    }

    @Test("a malformed override fails closed rather than falling back")
    func malformedOverrideFailsClosed() {
        for bad in [
            "4,4", "8,4", "0,4", "4,99", "4,x", "",
        ] {
            #expect(
                NGramPromptLookupConfiguration.fromEnvironment([
                    "DARKBLOOM_QWEN_LOOKUP_DRAFT": "1",
                    "DARKBLOOM_QWEN_LOOKUP_LADDER": bad,
                ]) == nil,
                "ladder '\(bad)' was accepted")
        }
        #expect(
            NGramPromptLookupConfiguration.fromEnvironment([
                "DARKBLOOM_QWEN_LOOKUP_DRAFT": "1",
                "DARKBLOOM_QWEN_LOOKUP_LADDER": "4,8",
                "DARKBLOOM_QWEN_LOOKUP_THRESHOLDS": "3",
            ]) == nil,
            "a length mismatch was accepted")
    }
}
