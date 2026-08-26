import Testing

@testable import MLXFastModel

struct QwenAgreementReportTests {
    @Test("identical streams report no divergence")
    func identical() {
        let report = QwenAgreementReport.compare(
            reference: [1, 2, 3], candidate: [1, 2, 3])
        #expect(report.firstDivergence == nil)
        #expect(report.mismatchCount == 0)
        #expect(report.comparedLength == 3)
        #expect(report.lengthDelta == 0)
        #expect(report.summary.contains("no divergence"))
    }

    @Test("first divergence and mismatch count are reported over the overlap")
    func diverging() {
        let report = QwenAgreementReport.compare(
            reference: [1, 2, 3, 4, 5], candidate: [1, 2, 9, 4, 8, 7])
        #expect(report.firstDivergence == 2)
        #expect(report.mismatchCount == 2)      // positions 2 and 4
        #expect(report.comparedLength == 5)      // min of the two lengths
        #expect(report.lengthDelta == 1)         // candidate is longer
    }

    @Test("empty inputs are handled, not crashed on")
    func empty() {
        let report = QwenAgreementReport.compare(reference: [], candidate: [1])
        #expect(report.comparedLength == 0)
        #expect(report.firstDivergence == nil)
        #expect(report.lengthDelta == 1)
    }
}
