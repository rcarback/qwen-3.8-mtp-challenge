// Shape sweep comparing one fp16 ANE 1x1-conv program against the quantized
// GPU matmul it would actually replace. No model, no routing: this measures
// the two engines only, on the expert projection's own shapes.
//
// A note on this file's shape, for anyone tempted to "improve" it: several
// call patterns that look like reasonable amortisation -- reusing one
// staged `Prepared` across many `predict` calls, materialising the ANE's
// `readOutput` inside the timed loop, batching several independent
// `quantizedMM` calls into one `eval`, or running the ANE and GPU timed
// loops as two separate sequential blocks instead of interleaved -- were
// each found by bisection to reproducibly corrupt process memory under a
// debug (`swift test`) build. None of those are used here; see the
// fix-round section of the task-1 report for the bisection and why the
// shape below (interleaved, one dispatch per side per timed repeat, no
// materialised ANE read outside the warm-up) is the configuration that was
// verified stable.
import Foundation
import MLX

public struct ANEGemmSample: Codable {
    public let m: Int
    public let k: Int
    public let n: Int
    /// Per-op ANE seconds: one full `makeInput` (fresh IOSurfaces,
    /// transpose-and-scatter) + `predict` cycle, minimum over `iterations`
    /// repeats. Does NOT include materialising `readOutput`: doing that on
    /// every repeat of this loop was one of the patterns found to corrupt
    /// memory (see the file header). The single warm-up call earlier in
    /// `measure` does materialise and validate one ANE output (for the
    /// correctness check below), so the read path itself is exercised and
    /// checked; it is only repeating that materialisation in a tight loop
    /// against fresh `Prepared` objects that is unsafe here. Because this
    /// number omits materialisation, it is a lower bound on the ANE's true
    /// per-dispatch cost, biased toward making the ANE look faster than it
    /// is -- carry that bias in mind when reading `rate` below.
    /// `ANEDirectDispatch.Prepared` is also a documented one-shot handoff
    /// object, so there is no supported way to amortise `makeInput` staging
    /// across repeats either -- every real dispatch pays that cost.
    public let aneSeconds: Double
    /// Per-op GPU seconds: one `quantizedMM` call (materialised via `eval`),
    /// minimum over `iterations` repeats, measured in the same interleaved
    /// loop as the ANE reading above.
    public let gpuSeconds: Double
    /// `gpuSeconds / aneSeconds`. Above 1.0 means the ANE is faster.
    public let rate: Double
}

public enum ANEGemmBench {
    /// Relative-error tolerance for the ANE-vs-fp32 correctness check,
    /// matching `Qwen4ExpANEDenseLaneTests.testProjectionMatchesGPUWithinFP16`.
    static let correctnessTolerance: Double = 2e-2

    /// The quantization the deployed Qwen4Exp routed expert weights actually
    /// use -- 4-bit affine, group size 32 (see
    /// `Sources/MLXFastModel/Qwen4ExpTransform.swift`'s `expertGroupSize`
    /// default, and docs/perf/qwen38-flash-2026-09.md: "routed experts
    /// affine 4-bit group 32"). Group 64 is the separate main-backbone
    /// dense-tensor conversion described elsewhere in that doc -- a
    /// different set of tensors, not the routed experts this bench
    /// characterises. The GPU leg is measured against this kernel, not a
    /// dense fp32 matmul, because the quantized gather-GEMM is the op an ANE
    /// offload would actually replace.
    static let expertGroupSize = 32
    static let expertBits = 4

    /// `m` tokens, `k` input features, `n` output features. Shapes the ANE
    /// cannot build, or whose output fails the correctness check, are
    /// omitted from the result rather than trapping or being reported at a
    /// speed nobody validated.
    public static func sweep(shapes: [(m: Int, k: Int, n: Int)], iterations: Int) -> [ANEGemmSample] {
        var out = [ANEGemmSample]()
        for shape in shapes {
            guard shape.m > 0, shape.k > 0, shape.n > 0 else { continue }
            if let sample = measure(shape: shape, iterations: max(1, iterations)) {
                out.append(sample)
            }
        }
        return out
    }

    private static func measure(shape: (m: Int, k: Int, n: Int), iterations: Int) -> ANEGemmSample? {
        // fp32 golden weight: the correctness reference is computed from
        // this, and the GPU's quantized weight is derived from this, so both
        // legs trace back to the same source values the ANE program itself
        // is built from.
        let wFull = MLXRandom.normal([shape.n, shape.k])
        let xBf16 = MLXRandom.normal([shape.m, shape.k]).asType(.bfloat16)
        guard let projection = try? Qwen4ExpANEProjection(weight: wFull, sequenceLength: shape.m) else {
            return nil
        }
        let (wq, scales, biases) = quantized(wFull, groupSize: expertGroupSize, bits: expertBits, mode: .affine)
        eval(wq, scales)
        if let biases { eval(biases) }

        // Warm both engines once (one full one-shot ANE cycle, materialised;
        // one GPU call) so neither pays first-dispatch cost during the timed
        // loop below, and keep the ANE's warm output for the correctness
        // check -- this is the only place `readOutput` is materialised.
        guard let warmPrepared = try? projection.makeInput(xBf16),
            (try? projection.predict(warmPrepared)) != nil
        else { return nil }
        let aneOut = projection.readOutput(warmPrepared).asType(.float32)
        eval(aneOut)
        eval(
            quantizedMM(
                xBf16, wq, scales: scales, biases: biases, transpose: true,
                groupSize: expertGroupSize, bits: expertBits, mode: .affine))

        // Correctness: the ANE program against the exact fp32 answer for
        // these operands, independent of which GPU kernel the throughput
        // comparison below uses. A shape whose output diverges is dropped
        // rather than reported at a speed nobody validated.
        let reference = matmul(xBf16.asType(.float32), wFull.transposed())
        eval(reference)
        let relErr = (abs(aneOut - reference).mean() / abs(reference).mean()).item(Float.self)
        guard relErr.isFinite, Double(relErr) < correctnessTolerance else {
            fputs(
                "[ane-gemm-bench] m=\(shape.m) k=\(shape.k) n=\(shape.n): "
                    + "ANE output diverged from the fp32 reference by \(relErr), dropping shape\n",
                stderr)
            return nil
        }

        // One interleaved loop, minimum of each side over `iterations`
        // repeats -- see the file header for what was tried and found
        // unsafe instead.
        var aneSeconds = Double.greatestFiniteMagnitude
        var gpuSeconds = Double.greatestFiniteMagnitude
        for _ in 0 ..< iterations {
            let a0 = CFAbsoluteTimeGetCurrent()
            guard let p = try? projection.makeInput(xBf16), (try? projection.predict(p)) != nil else {
                continue
            }
            _ = projection.readOutput(p)
            aneSeconds = min(aneSeconds, CFAbsoluteTimeGetCurrent() - a0)

            let g0 = CFAbsoluteTimeGetCurrent()
            let y = quantizedMM(
                xBf16, wq, scales: scales, biases: biases, transpose: true,
                groupSize: expertGroupSize, bits: expertBits, mode: .affine)
            eval(y)
            gpuSeconds = min(gpuSeconds, CFAbsoluteTimeGetCurrent() - g0)
        }
        guard aneSeconds < .greatestFiniteMagnitude, gpuSeconds < .greatestFiniteMagnitude else { return nil }

        // GFLOP/s is derivable from (m, k, n, aneSeconds/gpuSeconds); it is
        // logged here rather than stored on `ANEGemmSample` because adding
        // fields to that struct -- pure Swift Doubles, no MLX calls -- was
        // itself enough to move the corruption described in the file header
        // from silent to visible in the debug-build test. That is a heap-
        // layout-sensitivity symptom of a bug elsewhere (most likely in the
        // private-API `ANEDirectDispatch`/`ANEInMemoryModel` bridging this
        // file calls into, not in this file), not evidence that computing a
        // ratio of two already-returned Doubles is itself unsafe -- but the
        // struct is kept minimal here rather than relying on that read.
        let flops = 2.0 * Double(shape.m) * Double(shape.k) * Double(shape.n)
        fputs(
            "[ane-gemm-bench] m=\(shape.m) k=\(shape.k) n=\(shape.n): "
                + "relErr=\(relErr) aneGFLOPS=\(flops / aneSeconds / 1e9) gpuGFLOPS=\(flops / gpuSeconds / 1e9)\n",
            stderr)
        return ANEGemmSample(
            m: shape.m, k: shape.k, n: shape.n,
            aneSeconds: aneSeconds, gpuSeconds: gpuSeconds,
            rate: gpuSeconds / aneSeconds)
    }
}
