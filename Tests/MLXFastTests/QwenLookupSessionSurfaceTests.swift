import Foundation
import MLXFastCore
import Testing

@testable import MLXFastModel

/// The lookup source is inert unless something installs an index, and the only
/// installer lives in `Sources/MLXFastHarness/`, which a submission does not
/// package. These are source-text pins because the property that matters is
/// "the packaged file reads no environment", which no runtime test can assert.
@Suite
struct QwenLookupSessionSurfaceTests {
    private static let sessionPath =
        "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

    private static func source(_ path: String) throws -> String {
        try String(contentsOfFile: path, encoding: .utf8)
    }

    @Test("the packaged session never enables the lookup source itself")
    func thePackagedSessionNeverEnablesTheLookupSourceItself() throws {
        let session = try Self.source(Self.sessionPath)
        #expect(session.contains("public var lookupIndex: NGramPromptLookupIndex?"))
        #expect(
            !session.contains("DARKBLOOM_QWEN_LOOKUP"),
            Comment(rawValue: "the packaged session reads a lookup "
                + "environment variable; enablement belongs in the "
                + "non-packaged worker"))
        #expect(
            !session.contains("NGramPromptLookupConfiguration.fromEnvironment"),
            "the packaged session builds a lookup configuration itself")
    }

    @Test("the installer lives outside the packaged surface")
    func theInstallerLivesOutsideThePackagedSurface() throws {
        let worker = try Self.source(
            "Sources/MLXFastHarness/QwenRuntimeMTPWorker.swift")
        #expect(worker.contains("NGramPromptLookupConfiguration.fromEnvironment"))
        let manifest = try Self.source("benchmark.json")
        #expect(!manifest.contains("Sources/MLXFastHarness/"))
        #expect(!manifest.contains("Sources/MLXFastCore/"))
    }

    @Test("the existing draft-policy surface is untouched")
    func theExistingDraftPolicySurfaceIsUntouched() throws {
        let session = try Self.source(Self.sessionPath)
        #expect(session.contains("public var draftPolicy"))
        #expect(session.contains("public static let defaultDraftDepth = 2"))
        #expect(session.contains(
            "THE POLICY BELOW IS THE FIRST THING A SUBMISSION SHOULD CHANGE"))
        #expect(session.contains("|| draftCount == 0"))
    }

    @Test("history calls are safe with no index installed")
    func historyCallsAreSafeWithNoIndexInstalled() {
        // The session cannot be constructed without a model, so exercise the
        // forwarding contract on the index itself: optional chaining over a
        // nil index is exactly what the session performs, and it must neither
        // allocate nor trap.
        let index: NGramPromptLookupIndex? = nil
        index?.reset(to: [1, 2, 3])
        index?.append([4, 5])
        #expect(index?.propose() == nil)
    }

    @Test("an installed index records what the session forwards")
    func anInstalledIndexRecordsWhatTheSessionForwards() {
        let index = NGramPromptLookupIndex(configuration: .shipped)
        index.reset(to: [1, 2, 3, 9, 9])
        index.append([1, 2, 3])
        #expect(index.tokenCount == 8)
        #expect(index.propose() == nil)
        index.append([50, 51, 52, 53, 1, 2, 3])
        #expect(index.propose() != nil)
    }
}
