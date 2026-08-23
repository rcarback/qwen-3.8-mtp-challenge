import Foundation
import Testing

@testable import MLXFastHarness

@Suite("Prefix reuse decision")
struct PrefixReuseTests {
    @Test("a strict extension reuses the session and forwards only the tail")
    func reusesOnExtension() {
        let decision = ServePrefixDecision.make(
            previous: [1, 2, 3], incoming: [1, 2, 3, 4, 5])
        #expect(decision == .extend([4, 5]))
    }

    @Test("an identical prompt has nothing to forward and must restart")
    func restartsOnIdenticalPrompt() {
        // Extending by zero tokens would leave no row to read a next token
        // from, so this is a reset rather than a no-op extension.
        let decision = ServePrefixDecision.make(
            previous: [1, 2, 3], incoming: [1, 2, 3])
        #expect(decision == .restart)
    }

    @Test("a diverging prefix restarts")
    func restartsOnDivergence() {
        #expect(
            ServePrefixDecision.make(previous: [1, 2, 3], incoming: [1, 9, 3, 4])
                == .restart)
    }

    @Test("a shorter prompt restarts because recurrent state cannot rewind")
    func restartsOnShorterPrompt() {
        #expect(
            ServePrefixDecision.make(previous: [1, 2, 3, 4], incoming: [1, 2])
                == .restart)
    }

    @Test("the first request of a process restarts")
    func restartsWithNoHistory() {
        #expect(ServePrefixDecision.make(previous: [], incoming: [1, 2]) == .restart)
    }
}
