import CoreML
import Foundation
import MLX
import MLXFastModel
import MLXRandom
import Testing

/// Task 3 (ANE+GPU concurrent offload plan): `ConcurrentEngines.run` must
/// join a real ANE closure (`ANEGemm`) and a real GPU closure (`quantizedMM`)
/// without cross-thread corruption. See
/// `.superpowers/sdd/2026-08-30-ane-gpu-concurrent-offload/task-3-brief.md`.
@Suite(.serialized)
struct ConcurrentEnginesTests {
    // Matches the proven ANEGemmTests shape (S=512, K=5120) so this test
    // exercises the same ANE compile/output path known to work, rather than
    // an unproven tiny shape.
    static let S = 512, K = 5120, N = 2048
    static let F = 1024 // ANE prefix channel count; N-F is the GPU suffix

    @Test("concurrent ANE-prefix + GPU-suffix matches sequential run and full dense reference")
    func concurrentMatchesSequential() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        MLXRandom.seed(3)
        let x = MLXRandom.normal([Self.S, Self.K]).asType(.float16)
        let w = MLXRandom.normal([Self.N, Self.K]).asType(.float16) // [out, in]
        eval(x, w)

        // Full dense reference: x @ w.T -> [S, N]
        let full = matmul(x, w.transposed(1, 0))
        eval(full)

        let wPrefix = w[0 ..< Self.F, 0...]
        let wSuffix = w[Self.F ..< Self.N, 0...]

        // ANE prefix: warm ANEGemm, built once outside the measured calls.
        let aneGemm = try ANEGemm(weight: wPrefix, sequenceLength: Self.S)
        func aneOp() throws -> MLXArray { try aneGemm(x) }

        // GPU suffix: real 4-bit affine group-64 quantized path (the actual
        // Qwen forward path), not bf16/fp16.
        let (wq, scales, biases0) = quantized(wSuffix, groupSize: 64, bits: 4)
        let biases = biases0 ?? scales
        eval(wq, scales, biases)
        func gpuSuffixOp() -> MLXArray {
            quantizedMM(x, wq, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4)
        }
        eval(gpuSuffixOp()) // warm

        // Sequential reference: the same two ops, one after the other, no
        // concurrency at all. This isolates "does concurrency corrupt
        // results" from "how accurate is fp16/4-bit vs dense" -- a broken
        // join (shared mutable state, a lost or duplicated write) shows up
        // as a divergence from THIS reference even though every input is
        // bit-identical to the concurrent run below.
        let seqAne = try aneOp()
        let seqGpu = gpuSuffixOp()
        eval(seqAne, seqGpu)
        let seqCombined = concatenated([seqAne, seqGpu], axis: 1)
        eval(seqCombined)

        // Several concurrent cycles -- corruption from a race is not
        // guaranteed to show up on the first call.
        for cycle in 0 ..< 5 {
            let (concAne, concGpu) = try ConcurrentEngines.run(ane: aneOp, gpu: gpuSuffixOp)
            eval(concAne, concGpu)
            let concCombined = concatenated([concAne, concGpu], axis: 1)
            eval(concCombined)

            let vsSeq = abs(concCombined.asType(.float32) - seqCombined.asType(.float32)).max().item(Float.self)
            #expect(vsSeq < 1e-4, "cycle \(cycle): concurrent run diverged from sequential run by \(vsSeq)")

            // Loose sanity bound only (not the concurrency-safety gate above):
            // the GPU suffix here is real 4-bit affine group-64 quantized,
            // and K=5120 std-normal activations/weights put per-element dot
            // products around sqrt(K)~72, so 4-bit quantization noise alone
            // can land tens of units away from the fp32 dense reference.
            let vsFull = abs(concCombined.asType(.float32) - full.asType(.float32)).max().item(Float.self)
            #expect(vsFull < 40.0, "cycle \(cycle): concurrent combined diverged from full dense reference by \(vsFull)")
        }
    }
}
