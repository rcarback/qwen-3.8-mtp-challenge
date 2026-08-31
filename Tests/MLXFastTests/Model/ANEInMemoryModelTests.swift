import Foundation
import MLX
import MLXRandom
import Testing
@testable import MLXFastModel

/// Task 2 (ANE IOSurface / procedure-bank plan): in-memory compile + load of
/// the real conv MIL through `_ANEInMemoryModel`, proving the ANE compiler
/// accepts a real program. See
/// `.superpowers/sdd/2026-08-31-ane-iosurface-procedure-bank/task-2-brief.md`.
@Suite(.serialized)
struct ANEInMemoryModelTests {
    @Test("empty MIL fails with InvalidCompilationParam (gate passed, MIL rejected)")
    func emptyMilRejected() throws {
        try #require(ANERuntime.available())
        let m = try ANEInMemoryModel(milProgram: "program(1.0){ func main() {} }".data(using: .utf8)!)
        #expect(throws: ANEInMemoryModel.ANEError.self) { try m.compile() }
    }

    @Test("real conv MIL compiles and loads with a non-zero programHandle")
    func realMilCompilesLoads() throws {
        try #require(ANERuntime.available())
        let K = 512, F = 256, S = 32
        let w = MLXRandom.normal([F, K]).asType(.float16)
        eval(w)
        let prog = buildConvMILProgram(K: K, F: F, S: S, weight: f16Bytes(w))
        let m = try ANEInMemoryModel(milProgram: prog)
        try m.compile()
        try m.load()
        #expect(m.programHandle != 0)
        m.unload()
    }
}
