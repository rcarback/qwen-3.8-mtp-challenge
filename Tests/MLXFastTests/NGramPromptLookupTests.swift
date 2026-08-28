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
