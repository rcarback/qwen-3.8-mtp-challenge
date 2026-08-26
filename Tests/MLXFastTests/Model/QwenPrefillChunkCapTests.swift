import Foundation
import Testing

@testable import MLXFastModel

/// Pure tests for the prefill chunk cap. No model load, so these run in the
/// ordinary `swift test` pass rather than behind
/// `MLXFAST_RUN_MLX_RUNTIME_TESTS`.
@Suite
struct QwenPrefillChunkCapTests {
    typealias Session = Qwen36MTPBlockSession

    @Test("an absent or unparseable override keeps the shipped default")
    func defaultsWhenUnset() {
        #expect(Session.resolveChunkCap(nil) == 1024)
        #expect(Session.resolveChunkCap("") == 1024)
        #expect(Session.resolveChunkCap("wide") == 1024)
        #expect(Session.resolveChunkCap("2048.5") == 1024)
    }

    @Test("a valid override is taken verbatim")
    func takesValidOverride() {
        #expect(Session.resolveChunkCap("512") == 512)
        #expect(Session.resolveChunkCap("2048") == 2048)
        #expect(Session.resolveChunkCap("8192") == 8192)
    }

    @Test("the resolver clamps rather than accepting an unsafe cap")
    func clampsOutOfRange() {
        // Below 512 the ranked 512-token seed would stop being a single
        // dispatch, which is the property the doc comment protects.
        #expect(Session.resolveChunkCap("256") == 512)
        #expect(Session.resolveChunkCap("0") == 512)
        #expect(Session.resolveChunkCap("-4096") == 512)
        #expect(Session.resolveChunkCap("999999") == 16384)
    }

    @Test("chunk size derives from the budget and clamps to the range")
    func chunkSizeDerivation() {
        let budget = 64 << 20
        // At zero cached positions the derived size is the whole budget, so
        // the cap is what binds.
        #expect(Session.prefillChunkSize(cached: 0, cap: 1024, budget: budget) == 1024)
        #expect(Session.prefillChunkSize(cached: 0, cap: 4096, budget: budget) == 4096)
        // 64 Mi / 16384 = 4096, so the cap binds at 1024 and the budget binds
        // at 4096. This is the pair of caps taking genuinely different
        // schedules that the collapsed 4096/8192 reading appeared to deny.
        #expect(Session.prefillChunkSize(cached: 16384, cap: 1024, budget: budget) == 1024)
        #expect(Session.prefillChunkSize(cached: 16384, cap: 4096, budget: budget) == 4096)
        // 64 Mi / 65536 = 1024.
        #expect(Session.prefillChunkSize(cached: 65536, cap: 4096, budget: budget) == 1024)
        // Deep enough that the derived size falls under the lower bound.
        #expect(Session.prefillChunkSize(cached: 1 << 20, cap: 4096, budget: budget) == 256)
    }

    @Test("the shipped range still reads 256 through 1024 with no override")
    func shippedRange() {
        // The test process sets no override, so this pins the default that the
        // server and the ranked path both take.
        #expect(Session.prefillChunkRange.lowerBound == 256)
        #expect(Session.prefillChunkRange.upperBound == 1024)
    }
}
