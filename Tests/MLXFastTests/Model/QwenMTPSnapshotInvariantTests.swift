import Foundation
import Testing

@testable import MLXFastRuntimeWorkerSupport

/// The resume point is a PAIR -- a token array and the session state reached by
/// consuming it -- and it is only usable when both halves describe the same
/// position.
///
/// They can disagree in exactly one way. A decode round at depth 2 commits one
/// to three tokens at once, and the serve loop stops walking that round at EOS
/// or its `max_tokens` budget, dropping the rest
/// (`QwenRuntimeServe.swift:655-668`). The parent then files `seed + emitted`
/// while the session has already folded the whole round into its state, so the
/// state runs ahead by however many tokens were discarded. Restoring that pair
/// sets the session counters from the snapshot and the ledger seed from the
/// shorter array; they disagree permanently and the next decode round throws.
///
/// Measured on a live server before this guard existed: every resume that fired
/// returned HTTP 500, three of three, the offset ahead by exactly the discarded
/// count. It vanished at depth 0, where a round commits one token and nothing
/// can be dropped -- which is the tell that identified the mechanism.
@Suite(.serialized)
struct QwenMTPSnapshotInvariantTests {

    @Test("a snapshot matching its filed token array is recordable")
    func matchingPairIsAccepted() {
        // The ordinary case: the parent kept every token the round committed.
        #expect(qwenMTPSnapshotDescribes(
            tokenCount: 15270, seedTokenCount: 15267, committedTokenCount: 3))
        // A fresh seed with nothing decoded yet.
        #expect(qwenMTPSnapshotDescribes(
            tokenCount: 8481, seedTokenCount: 8481, committedTokenCount: 0))
    }

    /// The three failures observed on the live server, by their real numbers.
    /// Each is a round whose tail the parent discarded, leaving the session
    /// ahead of the array it filed.
    @Test("a snapshot ahead of its filed token array is refused")
    func snapshotAheadIsRefused() {
        // seed 15267 + 3 committed = 15270 described, but the parent filed
        // only 15268: it cut the round after one of the three tokens.
        #expect(!qwenMTPSnapshotDescribes(
            tokenCount: 15268, seedTokenCount: 15267, committedTokenCount: 3))
        // +2 surplus, the depth-2 full-accept case.
        #expect(!qwenMTPSnapshotDescribes(
            tokenCount: 19756, seedTokenCount: 19755, committedTokenCount: 3))
        // +1 surplus, an earlier cut.
        #expect(!qwenMTPSnapshotDescribes(
            tokenCount: 25228, seedTokenCount: 25228, committedTokenCount: 1))
    }

    /// The guard must be an equality, not a lower bound. An array LONGER than
    /// the state is just as unusable: restoring it would leave the ledger
    /// expecting rows the recurrent state never folded in.
    @Test("a filed array longer than the snapshot is also refused")
    func arrayAheadIsRefused() {
        #expect(!qwenMTPSnapshotDescribes(
            tokenCount: 15275, seedTokenCount: 15267, committedTokenCount: 3))
    }

    /// Depth 0 is the control that identified the mechanism: a round commits
    /// exactly one token, so the parent can never cut one short, so the pair
    /// always agrees and the bug cannot occur.
    @Test("depth-0 rounds always describe their array")
    func depthZeroAlwaysAgrees() {
        var seed = 4096
        for committed in 0 ..< 64 {
            #expect(qwenMTPSnapshotDescribes(
                tokenCount: seed + committed,
                seedTokenCount: seed, committedTokenCount: committed))
        }
        seed = 0
        #expect(qwenMTPSnapshotDescribes(
            tokenCount: 0, seedTokenCount: 0, committedTokenCount: 0))
    }
}
