import Foundation
import MLX
import MLXRandom
import Testing
@testable import MLXFastModel

/// Task A (ANE IOSurface / procedure-bank plan): proves a loaded conv
/// program actually EXECUTES on the Neural Engine and returns numerically
/// correct output, via CPU-side IOSurface I/O and a blocking evaluate. See
/// `.superpowers/sdd/2026-08-31-ane-iosurface-procedure-bank/task-A-brief.md`.
@Suite(.serialized)
struct ANEDirectDispatchTests {
    /// Compiles+loads a conv program for `w=[F,K]`, runs `ANEDirectDispatch
    /// .runConv` on `x=[S,K]`, and checks the result against `x @ w.T`
    /// within the shared ANEGemm gate (maxAbs<0.3, meanAbs<0.01).
    private func assertMatchesMatmul(K: Int, F: Int, S: Int) throws {
        try #require(ANERuntime.available())
        let w = MLXRandom.normal([F, K]).asType(.float16)
        let x = MLXRandom.normal([S, K]).asType(.float16)
        eval(w, x)

        let milText = buildConvMILText(inputDim: K, outputDim: F, sequenceLength: S)
        let weightBlob = buildConvWeightBlob(f16Bytes(w))
        let model = try ANEInMemoryModel(milText: milText, weightBlob: weightBlob)
        try model.compile()
        try model.load()
        defer { model.unload() }

        let y = try ANEDirectDispatch.runConv(model: model, x: x, inputDim: K, outputDim: F, sequenceLength: S)
        #expect(y.shape == [S, F])

        let expected = matmul(x.asType(.float32), w.transposed(1, 0).asType(.float32))
        let diff = MLX.abs(y.asType(.float32) - expected)
        eval(diff)
        let maxAbs = diff.max().item(Float.self)
        let meanAbs = diff.mean().item(Float.self)
        #expect(maxAbs < 0.3, "maxAbs=\(maxAbs)")
        #expect(meanAbs < 0.01, "meanAbs=\(meanAbs)")
    }

    @Test("runConv matches matmul at S=32 (the shape that compiles)")
    func runConvMatchesMatmulS32() throws {
        try assertMatchesMatmul(K: 512, F: 256, S: 32)
    }

    @Test("runConv matches matmul at S=1 (decode width)")
    func runConvMatchesMatmulS1() throws {
        try assertMatchesMatmul(K: 512, F: 256, S: 1)
    }
}
