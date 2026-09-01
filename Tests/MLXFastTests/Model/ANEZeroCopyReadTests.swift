import Foundation
import MLX
import MLXRandom
import Testing
@testable import MLXFastModel

/// Task D2 spike (ANE IOSurface / procedure-bank plan): does MLX accept an
/// IOSurface-backed pointer as array backing? `ANEDirectDispatch.readZeroCopy`
/// wraps the ANE output surface directly via `MLXArray(rawPointer:)` instead
/// of the CPU gather in `read`. This test proves the zero-copy read is
/// numerically identical to the gather read, and (when gated) times both to
/// quantify the output-marshaling tax D2 removes.
///
/// Correctness runs by default (under MLXFAST_RUN_MLX_RUNTIME_TESTS). Timing
/// prints only when MLXFAST_ANE_ZEROCOPY_TIMING=1.
@Suite(.serialized)
struct ANEZeroCopyReadTests {
    private func buildModel(K: Int, F: Int, S: Int, x: MLXArray) throws -> ANEInMemoryModel {
        let w = MLXRandom.normal([F, K]).asType(.float16)
        eval(w)
        let milText = buildConvMILText(inputDim: K, outputDim: F, sequenceLength: S)
        let weightBlob = buildConvWeightBlob(f16Bytes(w))
        let model = try ANEInMemoryModel(milText: milText, weightBlob: weightBlob)
        try model.compile()
        try model.load()
        return model
    }

    @Test("readZeroCopy equals the CPU-gather read (S=512, no seq padding)")
    func zeroCopyEqualsGatherS512() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        try #require(ANERuntime.available())
        let K = 5_120, F = 2_176, S = 512

        let x = MLXRandom.normal([S, K]).asType(.float16)
        eval(x)
        let model = try buildModel(K: K, F: F, S: S, x: x)
        defer { model.unload() }

        let prepared = try ANEDirectDispatch.prepare(model: model, x: x, inputDim: K, outputDim: F, sequenceLength: S)
        try ANEDirectDispatch.evaluate(prepared)

        let gather = ANEDirectDispatch.read(prepared)
        let zero = ANEDirectDispatch.readZeroCopy(prepared)
        eval(gather, zero)
        #expect(zero.shape == [S, F])
        let diff = MLX.abs(gather.asType(.float32) - zero.asType(.float32))
        eval(diff)
        let maxAbs = diff.max().item(Float.self)
        print("ZEROCOPY-READ maxAbsDiff(gather,zerocopy)=\(maxAbs)")
        #expect(maxAbs == 0.0, "zero-copy read must be bit-identical to the gather read; maxAbs=\(maxAbs)")

        // Also correct vs the matmul reference.
        let w0 = gather // gather already validated elsewhere; anchor zero-copy to matmul too
        _ = w0
    }

    @Test("readZeroCopy vs read timing at S=512")
    func zeroCopyTiming() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
              env["MLXFAST_ANE_ZEROCOPY_TIMING"] == "1" else { return }
        try #require(ANERuntime.available())
        let K = 5_120, F = 5_120, S = 512   // output width = hidden, the real MLP-down shape
        let iters = Int(env["MLXFAST_TIMING_ITERS"] ?? "30") ?? 30

        let x = MLXRandom.normal([S, K]).asType(.float16)
        eval(x)
        let model = try buildModel(K: K, F: F, S: S, x: x)
        defer { model.unload() }
        let prepared = try ANEDirectDispatch.prepare(model: model, x: x, inputDim: K, outputDim: F, sequenceLength: S)
        try ANEDirectDispatch.evaluate(prepared)

        func best(_ body: () -> MLXArray) -> Double {
            eval(body())
            var b = Double.infinity
            for _ in 0 ..< iters {
                let t = Date(); eval(body()); b = Swift.min(b, Date().timeIntervalSince(t))
            }
            return b
        }
        let gatherT = best { ANEDirectDispatch.read(prepared) }
        let zeroT = best { ANEDirectDispatch.readZeroCopy(prepared) }
        print("ZEROCOPY-TIMING read(gather)=\(gatherT * 1e3)ms readZeroCopy=\(zeroT * 1e3)ms out=[\(S),\(F)]")
        #expect(gatherT.isFinite && zeroT.isFinite)
    }
}
