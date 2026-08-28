import Testing

@testable import MLXFastModel

/// The cap is arithmetic over row indices, and getting the pairing wrong is
/// silent: the head would fuse each hidden state with the wrong next token and
/// simply stop predicting. So the pairing is pinned here, without a model.
@Suite
struct Qwen36MTPHeadPrimingCapTests {
    @Test("an uncapped prime keeps every pair")
    func uncappedPrimeKeepsEveryPair() {
        let range = Qwen36MTPBlockSession.headPrimingRange(
            primeCount: 10, cap: nil)
        #expect(range.hiddenRows == 0 ..< 10)
        #expect(range.tokenIndices == 1 ..< 11)
    }

    @Test("a cap keeps the last pairs and keeps them aligned")
    func cappedPrimeKeepsTheLastPairs() {
        let range = Qwen36MTPBlockSession.headPrimingRange(
            primeCount: 10, cap: 4)
        #expect(range.hiddenRows == 6 ..< 10)
        #expect(range.tokenIndices == 7 ..< 11)
        #expect(range.hiddenRows.count == range.tokenIndices.count)
    }

    @Test("a cap larger than the prime is a no-op")
    func capLargerThanPrime() {
        let range = Qwen36MTPBlockSession.headPrimingRange(
            primeCount: 3, cap: 100)
        #expect(range.hiddenRows == 0 ..< 3)
        #expect(range.tokenIndices == 1 ..< 4)
    }
}
