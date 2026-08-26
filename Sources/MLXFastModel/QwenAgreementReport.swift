/// Agreement between two emitted token streams produced under different
/// numeric policies (KV bits, snapshot dtype, substituted MTP head).
///
/// "Benchmark-lossless" is a claim about a workload, not a technique, so
/// the unit of evidence is this comparison over a real session's tokens.
/// Two cases matter and read differently:
///
///   * Policies that change TARGET numerics (KV bits, state dtype) are
///     EXPECTED to diverge eventually; the questions are how deep and how
///     often.
///   * Policies that change only the HEAD must NOT diverge at all -- the
///     head only proposes, the target decides -- so any mismatch is a bug,
///     not a quality trade.
public struct QwenAgreementReport: Equatable {
    /// Positions compared: the shorter of the two streams.
    public let comparedLength: Int
    /// First mismatching position, or nil when the overlap agrees fully.
    public let firstDivergence: Int?
    /// Total mismatching positions inside the overlap.
    public let mismatchCount: Int
    /// `candidate.count - reference.count`.
    public let lengthDelta: Int

    public static func compare(
        reference: [Int], candidate: [Int]
    ) -> QwenAgreementReport {
        let limit = Swift.min(reference.count, candidate.count)
        var first: Int?
        var mismatches = 0
        for index in 0 ..< limit where reference[index] != candidate[index] {
            if first == nil { first = index }
            mismatches += 1
        }
        return QwenAgreementReport(
            comparedLength: limit,
            firstDivergence: first,
            mismatchCount: mismatches,
            lengthDelta: candidate.count - reference.count)
    }

    public var summary: String {
        let rate = comparedLength > 0
            ? Double(mismatchCount) / Double(comparedLength) * 100 : 0
        let divergence = firstDivergence.map { "first divergence at \($0)" }
            ?? "no divergence"
        return "agreement: compared \(comparedLength) tokens, \(divergence), "
            + "\(mismatchCount) mismatches "
            + "(\(String(format: "%.4f", rate))%), "
            + "length delta \(lengthDelta)"
    }
}
