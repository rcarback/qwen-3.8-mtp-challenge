import Foundation
import Testing
import Tokenizers

@testable import MLXFastCore
@testable import MLXFastHarness

/// PHASE 0, the go/no-go for prompt-lookup drafting.
///
/// Two tables over real session material:
///
///   1. The existing single-token analyzer's longest-match hit rate, at orders
///      1 through 8, which says how often the next token is recoverable at all.
///   2. The quantity the economics actually turn on: for each matched suffix
///      length, the mean and median number of CONSECUTIVE correct tokens the
///      shipped proposal would have produced. A rung only pays if the run
///      reaches its break-even.
///
/// Break-even, from the verify-cost sweep in
/// `Tests/MLXFastTests/Model/QwenPhaseBreakdownTests.swift` (serial 53.3 ms
/// per token at ~2k depth):
///
///     rung  3 -> width  4 ->  79.4 ms -> pays from 1 accepted draft
///     rung  8 -> width  9 -> 226.2 ms -> pays from 4 accepted drafts
///     rung 15 -> width 16 -> 334.7 ms -> pays from 6 accepted drafts
///     rung 31 -> width 32 -> 335.1 ms -> pays from 6 accepted drafts
///
/// GATE: proceed when the mean correct run length is at least 5 for positions
/// whose matched suffix length is at least 5. Between 4 and 5 is a marginal
/// pass and requires raising the rung-8 threshold first. Below 4, stop.
///
///     MLXFAST_QWEN_PREFILL_WEIGHTS=weights \
///     MLXFAST_LOOKUP_CORPUS=<file or directory of UTF-8 transcripts> \
///     swift test --force-resolved-versions --filter lookupCorpus
@Suite(.serialized)
struct QwenLookupCorpusTests {
    @Test("lookupCorpusSelfSimilarity")
    func lookupCorpusSelfSimilarity() throws {
        let env = ProcessInfo.processInfo.environment
        guard let weights = env["MLXFAST_QWEN_PREFILL_WEIGHTS"],
              let corpus = env["MLXFAST_LOOKUP_CORPUS"]
        else { return }

        let tokenizer = try QwenRuntime.loadLocalTokenizer(at: weights)
        let documents = try Self.documents(at: corpus)
        guard !documents.isEmpty else {
            Issue.record("no readable documents under \(corpus)")
            return
        }

        var runsByMatchLength: [Int: [Int]] = [:]
        var eligiblePositions = 0
        var totalPositions = 0

        for text in documents {
            let tokens = tokenizer.encode(
                text: text, addSpecialTokens: false)
            guard tokens.count > 64 else { continue }

            // The analyzer's own view: half the document is context, half is
            // the continuation it must predict.
            let split = tokens.count / 2
            let report = try NGramSelfSimilarity.analyze(
                contextTokens: Array(tokens[0 ..< split]),
                continuationTokens: Array(tokens[split...]),
                orders: [1, 2, 3, 4, 5, 6, 8])
            print(String(
                format: "  [%6d tok] longest-match hit %.3f, "
                    + "opportunity %.3f, optimistic %.3f",
                tokens.count,
                report.longestMatchMostRecentHitRate,
                report.longestMatchOpportunityRate,
                report.optimisticAnyOrderHitRate))

            // The shipped proposal's own view, replayed position by position.
            let index = NGramPromptLookupIndex(configuration: .shipped)
            for position in 0 ..< tokens.count {
                if position >= 16 {
                    totalPositions += 1
                    if let match = index.longestMatch(),
                       match.continuationStart < position
                    {
                        eligiblePositions += 1
                        let run = Self.correctRunLength(
                            tokens: tokens,
                            from: position,
                            copyingFrom: match.continuationStart,
                            limit: NGramPromptLookupIndex
                                .maximumSupportedDrafts)
                        runsByMatchLength[
                            match.matchedSuffixLength, default: []
                        ].append(run)
                    }
                }
                index.append(tokens[position])
            }
        }

        print("\n  correct continuation run length by matched suffix length")
        print("     k   positions     mean   median      p75      max")
        var pooledAtFiveOrMore: [Int] = []
        for k in runsByMatchLength.keys.sorted() {
            let runs = runsByMatchLength[k]!.sorted()
            if k >= 5 { pooledAtFiveOrMore.append(contentsOf: runs) }
            print(String(
                format: "  %4d  %10d  %7.2f  %7d  %7d  %7d",
                k, runs.count,
                Double(runs.reduce(0, +)) / Double(runs.count),
                runs[runs.count / 2],
                runs[(runs.count * 3) / 4],
                runs[runs.count - 1]))
        }
        print(String(
            format: "\n  eligible positions %d of %d (%.1f%%)",
            eligiblePositions, totalPositions,
            100 * Double(eligiblePositions) / Double(max(totalPositions, 1))))

        guard !pooledAtFiveOrMore.isEmpty else {
            Issue.record("no position reached a matched suffix length of 5")
            return
        }
        let gateMean = Double(pooledAtFiveOrMore.reduce(0, +))
            / Double(pooledAtFiveOrMore.count)
        print(String(
            format: "  GATE: mean run at k >= 5 is %.2f over %d positions "
                + "(pass at 5.00, marginal 4.00, stop below)\n",
            gateMean, pooledAtFiveOrMore.count))
        #expect(
            gateMean >= 4.0,
            "prompt-lookup drafting does not pay on this corpus: mean run "
                + "\(gateMean) at k >= 5 is below the rung-8 break-even of 4")
    }

    /// How many consecutive tokens starting at `from` equal the tokens
    /// starting at `copyingFrom`.
    static func correctRunLength(
        tokens: [Int], from: Int, copyingFrom: Int, limit: Int
    ) -> Int {
        var run = 0
        while run < limit,
              from + run < tokens.count,
              copyingFrom + run < from,
              tokens[from + run] == tokens[copyingFrom + run]
        {
            run += 1
        }
        return run
    }

    /// A file, or every regular file directly inside a directory, read as
    /// UTF-8. Unreadable entries are skipped rather than failing the run.
    static func documents(at path: String) throws -> [String] {
        let manager = FileManager.default
        var isDirectory: ObjCBool = false
        guard manager.fileExists(atPath: path, isDirectory: &isDirectory)
        else { return [] }
        if !isDirectory.boolValue {
            return (try? String(contentsOfFile: path, encoding: .utf8))
                .map { [$0] } ?? []
        }
        let names = try manager.contentsOfDirectory(atPath: path).sorted()
        return names.compactMap {
            try? String(
                contentsOfFile: path + "/" + $0, encoding: .utf8)
        }
    }
}
