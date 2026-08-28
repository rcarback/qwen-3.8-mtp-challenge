import MLXFastCore
import Testing

@testable import MLXFastModel

@Suite
struct QwenLookupDraftDecisionTests {
    private typealias Session = Qwen36MTPBlockSession

    @Test("the lookup bound agrees with what the index can produce")
    func theLookupBoundAgreesWithTheIndex() {
        #expect(Qwen36MTPLimits.maxLookupDepth
            == NGramPromptLookupIndex.maximumSupportedDrafts)
        #expect(Qwen36MTPLimits.maxLookupDepth > Qwen36MTPLimits.maxDepth)
    }

    @Test("the serial control never consults the lookup source")
    func theSerialControlNeverConsultsTheLookupSource() {
        var consulted = false
        let source = Session.resolveDraftSource(
            offeredDepth: Qwen36MTPLimits.serialControlDepth,
            headDraftCount: 4,
            lookupProposal: {
                consulted = true
                return NGramPromptLookupIndex.Proposal(
                    tokens: [1, 2, 3], matchedSuffixLength: 5)
            }())
        #expect(source == .none)
        #expect(!consulted, "the depth-0 control evaluated the lookup index")
    }

    @Test("no proposal falls through to the head schedule")
    func noProposalFallsThroughToTheHead() {
        #expect(
            Session.resolveDraftSource(
                offeredDepth: 8, headDraftCount: 5, lookupProposal: nil)
                == .head(5))
    }

    @Test("a head schedule of zero is an adaptive skip")
    func aHeadScheduleOfZeroIsAnAdaptiveSkip() {
        #expect(
            Session.resolveDraftSource(
                offeredDepth: 8, headDraftCount: 0, lookupProposal: nil)
                == .none)
    }

    @Test("a proposal wins over the head schedule")
    func aProposalWinsOverTheHeadSchedule() {
        #expect(
            Session.resolveDraftSource(
                offeredDepth: 2, headDraftCount: 2,
                lookupProposal: NGramPromptLookupIndex.Proposal(
                    tokens: Array(0 ..< 15), matchedSuffixLength: 9))
                == .lookup(Array(0 ..< 15)))
    }

    @Test("a proposal is not bounded by the parent offer")
    func aProposalIsNotBoundedByTheParentOffer() {
        #expect(
            Session.resolveDraftSource(
                offeredDepth: 1, headDraftCount: 1,
                lookupProposal: NGramPromptLookupIndex.Proposal(
                    tokens: Array(0 ..< 31), matchedSuffixLength: 12))
                == .lookup(Array(0 ..< 31)))
    }

    @Test("an over-wide proposal is refused rather than truncated")
    func anOverWideProposalIsRefused() {
        #expect(
            Session.resolveDraftSource(
                offeredDepth: 8, headDraftCount: 3,
                lookupProposal: NGramPromptLookupIndex.Proposal(
                    tokens: Array(0 ..< 32), matchedSuffixLength: 12))
                == .head(3))
    }

    @Test("an empty proposal falls through to the head")
    func anEmptyProposalFallsThroughToTheHead() {
        #expect(
            Session.resolveDraftSource(
                offeredDepth: 8, headDraftCount: 3,
                lookupProposal: NGramPromptLookupIndex.Proposal(
                    tokens: [], matchedSuffixLength: 12))
                == .head(3))
    }

    @Test("only widths the head loop never compiles need a lookup warm")
    func onlyWidthsTheHeadLoopNeverCompilesNeedAWarm() {
        #expect(Session.lookupWarmWidths(ladder: [3, 8, 15, 31]) == [16, 32])
        #expect(Session.lookupWarmWidths(ladder: [3, 8]) == [])
        #expect(Session.lookupWarmWidths(ladder: [10, 20]) == [11, 21])
    }
}

@Suite
struct QwenLookupWarmSurfaceTests {
    @Test("the shape warm covers every ladder width above the head's own")
    func theShapeWarmCoversEveryLadderWidth() throws {
        let session = try String(
            contentsOfFile: "Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
            encoding: .utf8)
        #expect(
            session.contains("Self.lookupWarmWidths("),
            Comment(rawValue: "warmAllDepthShapes does not compile the "
                + "lookup ladder widths; the first lookup round would pay a "
                + "pipeline compile inside the request"))
        #expect(
            session.contains("ladder: lookupIndex.configuration.ladder"),
            Comment(rawValue: "the lookup warm does not read the installed "
                + "index's own ladder, so an overridden ladder would leave "
                + "its widths cold"))
        // The warm must compile the wide verify SHAPE, not only the forward:
        // the top-2 reduction kernels are specialised per row count and the
        // prefix replay is what a partial acceptance runs.
        #expect(session.contains("Self.linearTopTwoRows(wideLogits)"))
        #expect(session.contains("committedRows: width - 1"))
    }
}
