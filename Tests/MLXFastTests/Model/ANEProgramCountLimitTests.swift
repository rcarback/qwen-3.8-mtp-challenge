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
}
