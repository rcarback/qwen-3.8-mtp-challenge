import Foundation
import MLX
import MLXRandom
import Testing

/// Times the 4-bit affine quantized matmul kernel at two accumulator-tile
/// heights, `bm = 32` (shipped) against `bm = 16`.
///
/// This is an instrument, not a gate. It loads no model and no weights: it
/// allocates random tensors at the projection shapes the Qwen 3.8 tower
/// dispatches and drives `quantizedMM` directly, so the measurement carries
/// none of the 64-layer confounds a serve-level sweep does.
///
/// The arm is selected by `MLX_QMM_BM`, which `qmm()` and `qmm_splitk()` read
/// fresh on every dispatch and fold into the JIT kernel name. That suffix is
/// load-bearing: the JIT library cache is keyed on the kernel name alone, so
/// without it the second arm would be served the first arm's compiled
/// pipeline and the probe would measure one kernel twice.
///
/// Opt in with `MLXFAST_RUN_QMM_BM_PROBE=1`.
@Suite(.serialized)
struct QuantizedMatmulTileBenchTests {

    private struct Shape {
        let label: String
        let k: Int
        let n: Int
    }

    private static let shapes = [
        Shape(label: "mlp gate_up", k: 5_120, n: 34_816),
        Shape(label: "gdn in_proj", k: 5_120, n: 16_480),
        Shape(label: "mlp down", k: 17_408, n: 5_120),
    ]

    private static let widths = [10, 12, 16, 24, 32]
    private static let arms = [32, 16]

    private static let groupSize = 64
    private static let bits = 4

    /// Distinct x operands, cycled so no two timed dispatches in a sample are
    /// the identical op on the identical inputs.
    private static let xVariants = 8
    /// Dispatches folded into one `eval`, to keep host graph-build cost small
    /// against the measured device time.
    private static let itersPerSample = 16
    /// Timed samples per cell per arm. Arms alternate sample by sample.
    private static let samples = 15

    private func setArm(_ bm: Int) {
        setenv("MLX_QMM_BM", String(bm), 1)
    }

    private func quantizedWeights(k: Int, n: Int) -> (MLXArray, MLXArray, MLXArray?) {
        let w = MLXRandom.normal([n, k], scale: 0.02).asType(.bfloat16)
        let q = quantized(
            w, groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
        eval(q.wq, q.scales)
        if let b = q.biases { eval(b) }
        return (q.wq, q.scales, q.biases)
    }

    private func project(
        _ x: MLXArray, _ wq: MLXArray, _ scales: MLXArray, _ biases: MLXArray?
    ) -> MLXArray {
        quantizedMM(
            x, wq, scales: scales, biases: biases, transpose: true,
            groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
    }

    /// Dequantize-and-matmul reference, in float32.
    private func referenceProduct(
        x: MLXArray, wq: MLXArray, scales: MLXArray, biases: MLXArray?
    ) -> MLXArray {
        let wFull = dequantized(
            wq, scales: scales, biases: biases,
            groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
        let reference = matmul(
            x.asType(.float32), wFull.asType(.float32).transposed())
        eval(reference)
        return reference
    }

    /// Largest deviation of `y` from the reference, normalised by the
    /// reference's own magnitude.
    private func relativeError(_ y: MLXArray, reference: MLXArray) -> Float {
        let scale = maximum(abs(reference).max(), MLXArray(Float(1e-6)))
        let err = (abs(y.asType(.float32) - reference) / scale).max()
        eval(err)
        return err.item(Float.self)
    }

    @Test("quantized matmul row tile: bm=32 versus bm=16")
    func rowTileSweep() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_QMM_BM_PROBE"] == "1",
            ProcessInfo.processInfo
                .environment["MLX_QMM_BM_COLLIDE"] != "1",
            ProcessInfo.processInfo
                .environment["MLX_QMM_BM_DEBUG"] != "1"
        else { return }

        MLXRandom.seed(0x5EED)

        var lines: [String] = []

        for shape in Self.shapes {
            let (wq, scales, biases) = quantizedWeights(k: shape.k, n: shape.n)
            let weightBytes =
                Double(shape.n) * Double(shape.k)
                * (Double(Self.bits) / 8.0
                    + 4.0 / Double(Self.groupSize))

            for m in Self.widths {
                let xs = (0 ..< Self.xVariants).map { _ -> MLXArray in
                    let x = MLXRandom.normal([m, shape.k], scale: 1.0)
                        .asType(.bfloat16)
                    eval(x)
                    return x
                }

                // Correctness first, both arms, before any timing is taken.
                let reference = referenceProduct(
                    x: xs[0], wq: wq, scales: scales, biases: biases)
                var errors: [Int: Float] = [:]
                for bm in Self.arms {
                    setArm(bm)
                    let y = project(xs[0], wq, scales, biases)
                    eval(y)
                    errors[bm] = relativeError(y, reference: reference)
                }
                for bm in Self.arms {
                    #expect(
                        errors[bm]! < 0.02,
                        "bm=\(bm) \(shape.label) M=\(m) relative error \(errors[bm]!)")
                }
                guard Self.arms.allSatisfy({ errors[$0]! < 0.02 }) else {
                    lines.append(
                        "  \(shape.label) M=\(m): CORRECTNESS FAILED, no timing")
                    continue
                }

                // Warm both pipelines so no timed sample pays JIT compilation.
                for bm in Self.arms {
                    setArm(bm)
                    for _ in 0 ..< 3 {
                        let y = project(xs[0], wq, scales, biases)
                        eval(y)
                    }
                }

                var timings: [Int: [Double]] = [32: [], 16: []]
                // ABAB: arms alternate sample by sample so thermal drift over
                // the cell lands on both equally.
                for _ in 0 ..< Self.samples {
                    for bm in Self.arms {
                        setArm(bm)
                        var outs: [MLXArray] = []
                        outs.reserveCapacity(Self.itersPerSample)
                        let t0 = DispatchTime.now().uptimeNanoseconds
                        for i in 0 ..< Self.itersPerSample {
                            outs.append(
                                project(
                                    xs[i % Self.xVariants], wq, scales, biases))
                        }
                        eval(outs)
                        let t1 = DispatchTime.now().uptimeNanoseconds
                        timings[bm]!.append(
                            Double(t1 - t0) / 1e9 / Double(Self.itersPerSample))
                    }
                }

                func median(_ v: [Double]) -> Double {
                    let s = v.sorted()
                    return s.count % 2 == 1
                        ? s[s.count / 2]
                        : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
                }

                let flops = 2.0 * Double(m) * Double(shape.n) * Double(shape.k)
                var cells: [String] = []
                for bm in Self.arms {
                    let v = timings[bm]!
                    let med = median(v)
                    cells.append(
                        "bm=\(bm) "
                            + String(
                                format:
                                    "%7.3f ms (min %7.3f, spread %5.1f%%) "
                                    + "%5.2f TFLOPS %6.1f GB/s",
                                1000 * med, 1000 * v.min()!,
                                100 * (v.max()! - v.min()!) / med,
                                flops / med / 1e12,
                                weightBytes / med / 1e9))
                    #expect(med > 0)
                }
                let ratio = median(timings[32]!) / median(timings[16]!)
                let label = shape.label.padding(
                    toLength: 12, withPad: " ", startingAt: 0)
                let tail = String(
                    format: "bm16 speedup %.3fx  (relerr 32:%.4f 16:%.4f)",
                    ratio, errors[32]!, errors[16]!)
                lines.append(
                    "  \(label) M=\(m)\t\(cells[0])  |  \(cells[1])  |  "
                        + tail)
            }
        }

        setenv("MLX_QMM_BM", "32", 1)

        print(
            """

            [qmm row tile probe] median of \(Self.samples) samples, \
            \(Self.itersPerSample) dispatches per sample, arms interleaved ABAB
            """)
        for line in lines { print(line) }
    }

    /// Falsification control for the kernel-name cache hazard.
    ///
    /// With `MLX_QMM_BM_COLLIDE=1` the `bm` suffix is dropped from the kernel
    /// name while `bm` still drives the launch grid, so the second arm is
    /// served the first arm's compiled pipeline — exactly the failure mode the
    /// suffix exists to prevent.
    ///
    /// Order matters. The bm=16 arm runs first, so the cache holds a BM=16
    /// pipeline; the bm=32 arm then launches `ceil(24/32) = 1` threadgroup
    /// tile into a kernel that fills only 16 rows, leaving output rows 16..23
    /// never written. (The reverse order would not falsify anything: a BM=32
    /// kernel launched on a 2-tile bm=16 grid computes the correct rows in
    /// tile 0 and clips tile 1 away, so it would still look right.)
    ///
    /// This test asserts the corruption happens. If it did not, the name
    /// suffix would not be separating the pipelines and the timing arms above
    /// could not be trusted to be two kernels.
    ///
    /// Runs only when both `MLXFAST_RUN_QMM_BM_PROBE=1` and
    /// `MLX_QMM_BM_COLLIDE=1` are set, in a process of its own.
    @Test("name collision without the bm suffix corrupts the result")
    func collisionControl() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_QMM_BM_PROBE"] == "1",
            ProcessInfo.processInfo
                .environment["MLX_QMM_BM_COLLIDE"] == "1"
        else { return }

        MLXRandom.seed(0x5EED)
        let k = 5_120
        let n = 16_480
        let m = 24  // 2 tiles at bm=16, 1 tile at bm=32.

        let (wq, scales, biases) = quantizedWeights(k: k, n: n)
        let x = MLXRandom.normal([m, k], scale: 1.0).asType(.bfloat16)
        eval(x)
        let reference = referenceProduct(
            x: x, wq: wq, scales: scales, biases: biases)

        setArm(16)
        let y16 = project(x, wq, scales, biases)
        eval(y16)
        let err16 = relativeError(y16, reference: reference)

        setArm(32)
        let y32 = project(x, wq, scales, biases)
        eval(y32)
        let err32 = relativeError(y32, reference: reference)

        print(
            String(
                format:
                    "\n[qmm collision control] shared kernel name, "
                    + "bm=16 first: relerr %.4f, then bm=32: relerr %.4f",
                err16, err32))

        #expect(err16 < 0.02, "the first arm compiles its own pipeline")
        #expect(
            err32 > 0.02,
            "with a shared kernel name the bm=32 arm must be served the cached bm=16 pipeline and leave rows 16..23 unwritten; it did not, so the two arms may not be distinct kernels")
    }

    /// Prints the compiled pipeline properties of both arms, one dispatch
    /// each, so the two kernels can be told apart by a property of the
    /// binaries rather than by their timings.
    ///
    /// `qmm_t` allocates `Xs[BM * BK_padded]` and `Ws[BN * BK_padded]` in
    /// threadgroup memory, bf16, `BK_padded = 40`. A BM=32 build therefore
    /// reserves 32*40*2 + 32*40*2 = 5120 bytes and a BM=16 build reserves
    /// 16*40*2 + 32*40*2 = 3840 bytes. `staticThreadgroupMemoryLength` reads
    /// that off the pipeline state, so a 1280-byte difference is direct
    /// evidence that the arms are two separately compiled kernels and not one
    /// cached pipeline measured twice.
    ///
    /// Run with `MLXFAST_RUN_QMM_BM_PROBE=1 MLX_QMM_BM_DEBUG=1` and read the
    /// `[bmprobe]` lines on stderr.
    @Test("both arms compile to distinct pipelines")
    func pipelineIdentity() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_QMM_BM_PROBE"] == "1",
            ProcessInfo.processInfo
                .environment["MLX_QMM_BM_DEBUG"] == "1",
            ProcessInfo.processInfo
                .environment["MLX_QMM_BM_COLLIDE"] != "1"
        else { return }

        MLXRandom.seed(0x5EED)
        let k = 5_120
        let n = 34_816
        let (wq, scales, biases) = quantizedWeights(k: k, n: n)
        let x = MLXRandom.normal([12, k], scale: 1.0).asType(.bfloat16)
        eval(x)
        let reference = referenceProduct(
            x: x, wq: wq, scales: scales, biases: biases)

        for bm in Self.arms {
            setArm(bm)
            let y = project(x, wq, scales, biases)
            eval(y)
            let err = relativeError(y, reference: reference)
            #expect(err < 0.02, "bm=\(bm) relative error \(err)")
        }
    }
}
