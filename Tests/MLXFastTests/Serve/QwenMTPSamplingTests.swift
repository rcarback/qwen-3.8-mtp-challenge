import Foundation
import Testing

#if canImport(MLX)
import MLX
@testable import MLXFastModel

@Suite(
    "MTP sampling",
    .enabled(if: ProcessInfo.processInfo
        .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"))
struct QwenMTPSamplingTests {
    /// A one-hot logit row must select its own peak no matter the seed: the
    /// distribution has no other mass to draw.
    @Test("a degenerate distribution always draws its single token")
    func drawsFromDegenerateRow() {
        var row = [Float](repeating: -60, count: 8)
        row[3] = 60
        let logits = MLXArray(row).reshaped([1, 1, 8])
        for seed in UInt64(0) ..< 8 {
            let token = Qwen36MTPBlockSession.sampledTokenForTesting(
                logits, temperature: 1, topP: 1, seed: seed)
            #expect(token == 3)
        }
    }

    /// Nucleus filtering with a tiny mass must collapse to the single most
    /// probable token, which is the property that makes top_p usable as a
    /// determinism knob.
    @Test("topP just above zero collapses to the argmax")
    func collapsesUnderTinyTopP() {
        let logits = MLXArray([Float(0.1), 5.0, 0.2, 0.3]).reshaped([1, 1, 4])
        for seed in UInt64(0) ..< 8 {
            let token = Qwen36MTPBlockSession.sampledTokenForTesting(
                logits, temperature: 1, topP: 0.01, seed: seed)
            #expect(token == 1)
        }
    }

    /// The whole point of sampling: a flat distribution must not always yield
    /// the same token. Eight draws over four equally likely tokens collide
    /// entirely with probability 4 * (1/4)^8, about one in sixteen thousand.
    @Test("a flat distribution produces more than one token across seeds")
    func variesAcrossSeeds() {
        let logits = MLXArray([Float(0), 0, 0, 0]).reshaped([1, 1, 4])
        let drawn = Set((UInt64(0) ..< 8).map { seed in
            Qwen36MTPBlockSession.sampledTokenForTesting(
                logits, temperature: 1, topP: 1, seed: seed)
        })
        #expect(drawn.count > 1)
    }

    /// Same seed, same logits, same token. Without this a request carrying an
    /// explicit seed would not be reproducible.
    @Test("the same seed reproduces the same token")
    func isReproducible() {
        let logits = MLXArray([Float(0), 1, 2, 1]).reshaped([1, 1, 4])
        let first = Qwen36MTPBlockSession.sampledTokenForTesting(
            logits, temperature: 1, topP: 1, seed: 99)
        let second = Qwen36MTPBlockSession.sampledTokenForTesting(
            logits, temperature: 1, topP: 1, seed: 99)
        #expect(first == second)
    }
}
#endif
