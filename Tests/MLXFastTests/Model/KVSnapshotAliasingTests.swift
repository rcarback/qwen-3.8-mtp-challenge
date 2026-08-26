import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

@testable import MLXFastCore

/// Is a retained `cache.state` a SNAPSHOT or a VIEW?
///
/// `KVCacheSimple.update` writes in place into a preallocated buffer
/// (`self.keys?[.ellipsis, previous ..< offset, 0...] = keys`) and `state`
/// returns a slice of that same buffer. If the slice aliases, retaining it as a
/// prefix-cache snapshot would be silently corrupted by the next decode step --
/// the failure mode is wrong logits, not a crash, so it must be settled before
/// anything is built on it.
@Suite(.serialized)
struct KVSnapshotAliasingTests {
    @Test("retained cache state does not alias later writes")
    func stateIsNotAliased() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let B = 1, H = 4, D = 8
        let cache = KVCacheSimple()

        func step(_ fill: Float, _ n: Int) {
            let k = MLXArray.full([B, H, n, D], values: MLXArray(fill))
            let v = MLXArray.full([B, H, n, D], values: MLXArray(fill))
            _ = cache.update(keys: k, values: v)
        }

        step(1.0, 4)
        let snapshot = cache.state            // retained as-is, no copy
        // `+ 0` forces a materialised result rather than a view of the buffer.
        let copied = cache.state.map { $0 + MLXArray(Float(0)) }
        eval(snapshot + copied)
        let snapBefore = snapshot[0].sum().item(Float.self)
        let copyBefore = copied[0].sum().item(Float.self)

        // Fill the rest of the SAME preallocated step-256 buffer.
        step(9.0, 4)
        eval(snapshot + copied + cache.state)

        let snapAfter = snapshot[0].sum().item(Float.self)
        let copyAfter = copied[0].sum().item(Float.self)
        let snapRows = snapshot[0].dim(2)

        print("""

          retained state rows      \(snapRows)   (expected 4)
          retained sum  before/after \(snapBefore) / \(snapAfter)
          copied   sum  before/after \(copyBefore) / \(copyAfter)
          verdict: retained slice \(snapBefore == snapAfter ? "is SAFE" : "ALIASES -- must copy")

        """)
        #expect(copyBefore == copyAfter, "an explicit copy must never alias")
        #expect(snapBefore == snapAfter, "retained state aliased a later write")
    }
}
