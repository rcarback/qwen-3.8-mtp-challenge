import Foundation
import MLX
import MLXRandom
import Testing
@testable import MLXFastModel

/// How many fixed-shape programs one process can hold loaded before the ANE
/// daemon answers `Program load failure (0x50004)`. The dense lane lost 10 of
/// 64 programs once about 118 were resident (~167 MB each); the MoE fused
/// lanes lost 62 of 96 bucket-1024 programs once about 96 small ones (10 to
/// 26 MB) were resident. Two sizes separate a count limit from a byte limit.
/// Needs the real ANE; opt in with MLXFAST_RUN_MLX_RUNTIME_TESTS=1.
@Suite(.serialized)
struct ANEProgramCountLimitTests {
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    private func probe(outputDim: Int, inputDim: Int, sequenceLength: Int, maxPrograms: Int) throws
        -> (built: Int, error: String?)
    {
        var kept: [ANEInMemoryModel] = []
        defer { kept.forEach { $0.unload() } }
        for i in 0 ..< maxPrograms {
            let w = MLXRandom.normal([outputDim, inputDim]).asType(.float16)
            eval(w)
            let blob = buildConvWeightBlob(f16Bytes(w))
            let text = buildConvMILText(
                inputDim: inputDim, outputDim: outputDim, sequenceLength: sequenceLength,
                programTag: "count-probe-\(outputDim)x\(inputDim)-S\(sequenceLength)-\(i)")
            let m = try ANEInMemoryModel(milText: text, weightBlob: blob)
            do {
                try m.compile()
                try m.load()
            } catch {
                return (i, "\(error)")
            }
            kept.append(m)
        }
        return (maxPrograms, nil)
    }

    @Test("small programs: 64x64 at S=128", .enabled(if: enabled))
    func smallPrograms() throws {
        try #require(ANERuntime.available())
        let r = try probe(outputDim: 64, inputDim: 64, sequenceLength: 128, maxPrograms: 400)
        print("[count-probe] 64x64 S=128 (8 KB weights): built \(r.built); first failure: \(r.error ?? "none")")
    }

    @Test("medium programs: 5120x2560 at S=1024, 26 MB each", .enabled(if: enabled))
    func mediumPrograms() throws {
        try #require(ANERuntime.available())
        let r = try probe(outputDim: 5120, inputDim: 2560, sequenceLength: 1024, maxPrograms: 200)
        print("[count-probe] 5120x2560 S=1024 (26 MB weights): built \(r.built); first failure: \(r.error ?? "none")")
    }

    /// How big can ONE program be?
    ///
    /// The two tests above establish that the resident limit is a COUNT (126)
    /// and not a byte budget: 8 KB and 26 MB programs both fail at the 127th.
    /// That leaves the other half of "how much model fits on the ANE"
    /// unmeasured -- total capacity is 126 x (max bytes per program), and only
    /// the 126 has ever been measured. 26 MB is merely the largest size anyone
    /// happened to try, not a demonstrated ceiling.
    ///
    /// Sweeps a single program's weight matrix upward at a deliberately small
    /// sequence length, so weight bytes dominate and the result is not confounded
    /// by activation buffers. Prints the largest size that loads.
    ///
    /// HOW TO RUN THIS. The first attempt (2026-09-04) produced nothing, for
    /// reasons that were entirely in the invocation:
    ///
    ///   - Do NOT pipe the output through `grep`. It block-buffers, so every
    ///     completed size line sits in a buffer and is lost if the run is
    ///     interrupted. Redirect to a file and read the file.
    ///   - Give it a per-size wall-clock bound. Compiling a several-hundred-MB
    ///     program is slow, and without a bound one size consumes the run.
    ///   - To judge liveness, sample the `swiftpm-testing-helper` WORKER, not
    ///     the `swift-test` wrapper. The wrapper sits at 0 percent CPU by
    ///     design because it waits on its child; reading the wrapper made a
    ///     computing run look hung and it was killed at 39 minutes.
    ///
    ///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions \
    ///       --filter maxSingleProgramSize > /tmp/size-probe.log 2>&1 &
    @Test("max single program size", .enabled(if: enabled))
    func maxSingleProgramSize() throws {
        try #require(ANERuntime.available())
        var largestOK = 0
        for outputDim in [5120, 10240, 20480, 40960, 81920, 163_840] {
            let mb = Double(outputDim * 2560 * 2) / 1_000_000
            let r = try probe(
                outputDim: outputDim, inputDim: 2560, sequenceLength: 128, maxPrograms: 1)
            let ok = r.built == 1
            print(
                "[size-probe] \(outputDim)x2560 S=128 = \(String(format: "%.1f", mb)) MB -> "
                    + (ok ? "OK" : "FAILED: \(r.error ?? "unknown")"))
            if !ok { break }
            largestOK = outputDim
        }
        guard largestOK > 0 else { return }
        let mb = Double(largestOK * 2560 * 2) / 1_000_000
        print("[size-probe] largest single program that loads: \(largestOK)x2560 = \(String(format: "%.1f", mb)) MB")

        // Does an AGGREGATE byte ceiling exist that the count probe missed?
        // If N programs of this size fail well before 126, the binding limit is
        // bytes after all, at least at this size.
        let r = try probe(
            outputDim: largestOK, inputDim: 2560, sequenceLength: 128, maxPrograms: 130)
        print(
            "[size-probe] count at \(String(format: "%.1f", mb)) MB each: built \(r.built) "
                + "(= \(String(format: "%.2f", Double(r.built) * mb / 1000)) GB resident); "
                + "first failure: \(r.error ?? "none")")
    }
}
