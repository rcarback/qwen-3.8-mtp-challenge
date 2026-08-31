import Foundation
import MLX
import MLXRandom
import Testing
@testable import MLXFastModel

/// Task 2b (ANE IOSurface / procedure-bank plan, redo of Task 2): in-memory
/// compile + load of the real conv MIL TEXT through `_ANEInMemoryModel`,
/// proving the ANE compiler accepts a real program unentitled. Task 2's
/// binary MIL protobuf failed compile with `InvalidCompilationParam`; the
/// open-source oMLX project proves the MIL TEXT format works. See
/// `.superpowers/sdd/2026-08-31-ane-iosurface-procedure-bank/task-2b-brief.md`.
@Suite(.serialized)
struct ANEInMemoryModelTests {
    @Test("malformed MIL text fails compile cleanly, no crash")
    func malformedMilTextRejected() throws {
        try #require(ANERuntime.available())
        let m = try ANEInMemoryModel(milText: "program(1.0){ func main() {} }", weightBlob: Data())
        #expect(throws: ANEInMemoryModel.ANEError.self) { try m.compile() }
    }

    @Test("real conv MIL text compiles and loads with a non-zero programHandle")
    func realMilTextCompilesLoads() throws {
        try #require(ANERuntime.available())
        let K = 512, F = 256, S = 32
        let w = MLXRandom.normal([F, K]).asType(.float16)
        eval(w)
        let milText = buildConvMILText(inputDim: K, outputDim: F, sequenceLength: S)
        let weightBlob = buildConvWeightBlob(f16Bytes(w))
        let m = try ANEInMemoryModel(milText: milText, weightBlob: weightBlob)
        try m.compile()
        try m.load()
        #expect(m.programHandle != 0)
        m.unload()
    }
}
