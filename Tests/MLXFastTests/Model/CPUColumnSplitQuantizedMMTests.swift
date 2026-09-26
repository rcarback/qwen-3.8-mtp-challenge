import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM

/// Correctness of CPU/GPU column-split assist for affine 4-bit group-64
/// quantized projections. These tests are the deliverable: each property
/// below is one that a wrong implementation would fail, and the report
/// records a mutation that made it fail.
@Suite(.serialized)
struct CPUColumnSplitQuantizedMMTests {

    private struct QuantizedProj {
        let x: MLXArray
        let w: MLXArray
        let scales: MLXArray
        let biases: MLXArray
    }

    private func makeProj(m: Int, n: Int, k: Int, seed: UInt64 = 7) -> QuantizedProj {
        MLXRandom.seed(seed)
        let x = MLXRandom.normal([1, m, k]).asType(.bfloat16)
        let dense = MLXRandom.normal([n, k]).asType(.bfloat16)
        let (w, scales, biases) = quantized(
            dense, groupSize: 64, bits: 4, mode: .affine)
        let z = biases!
        eval(x, w, scales, z)
        return QuantizedProj(x: x, w: w, scales: scales, biases: z)
    }

    private func gpuOnly(_ p: QuantizedProj) -> MLXArray {
        let y = quantizedMM(
            p.x, p.w, scales: p.scales, biases: p.biases,
            transpose: true, groupSize: 64, bits: 4, mode: .affine,
            stream: .gpu)
        eval(y)
        return y
    }

    private func cpuOnly(_ p: QuantizedProj, stream: MLX.Stream) -> MLXArray {
        let y = quantizedMM(
            p.x, p.w, scales: p.scales, biases: p.biases,
            transpose: true, groupSize: 64, bits: 4, mode: .affine,
            stream: .stream(stream))
        eval(y)
        return y
    }

    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// max |a-b| / max(|b|, eps). A single global ratio, not a per-element
    /// max: bf16 CPU and GPU kernels disagree on tiny outputs, which makes
    /// |d|/max(|b_i|, 1e-6) blow up without saying anything about the split.
    private func globalRel(_ a: MLXArray, _ b: MLXArray) -> Float {
        let absErr = maxAbs(a, b)
        let scale = abs(b.asType(.float32)).max().item(Float.self)
        let denom = scale >= 1e-6 ? scale : 1e-6
        return absErr / denom
    }

    private func loadProbe(stream: StreamOrDevice) throws -> (ok: Bool, error: String?) {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(
                "cpu-split-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("values.npy")
        try MLX.save(array: MLXArray([1, 2, 3, 4] as [Float]), url: url)
        do {
            let loaded = try withError { error in
                let array = try MLX.loadArray(url: url, stream: stream)
                array.eval()
                try error.check()
                return array.asArray(Float.self)
            }
            return (loaded == [1, 2, 3, 4], nil)
        } catch {
            return (false, String(describing: error))
        }
    }

    // MARK: - Fraction parsing and column counts

    @Test
    func unsetEnvFractionIsZero() throws {
        #expect(try qwen35ParsePrefillCPUColumnFraction(nil) == 0)
        #expect(try qwen35ParsePrefillCPUColumnFraction("") == 0)
        #expect(try qwen35ParsePrefillCPUColumnFraction("0") == 0)
        #expect(try qwen35ParsePrefillCPUColumnFraction("0.0") == 0)
    }

    @Test
    func validFractionsParse() throws {
        #expect(try qwen35ParsePrefillCPUColumnFraction("0.125") == 0.125)
        #expect(try qwen35ParsePrefillCPUColumnFraction("1") == 1)
        #expect(try qwen35ParsePrefillCPUColumnFraction("1.0") == 1)
    }

    @Test
    func invalidFractionRefuses() {
        #expect(throws: Qwen35CPUAssistError.self) {
            try qwen35ParsePrefillCPUColumnFraction("not-a-number")
        }
        #expect(throws: Qwen35CPUAssistError.fractionOutOfRange(-0.1)) {
            try qwen35ParsePrefillCPUColumnFraction("-0.1")
        }
        #expect(throws: Qwen35CPUAssistError.fractionOutOfRange(1.5)) {
            try qwen35ParsePrefillCPUColumnFraction("1.5")
        }
    }

    @Test
    func columnCountMatchesFractionsAndUnevenN() throws {
        let n = 50
        #expect(try qwen35CPUColumnCount(outputColumns: n, fraction: 0) == 0)
        #expect(try qwen35CPUColumnCount(outputColumns: n, fraction: 0.125) == 6)
        #expect(try qwen35CPUColumnCount(outputColumns: n, fraction: 0.25) == 12)
        #expect(try qwen35CPUColumnCount(outputColumns: n, fraction: 0.5) == 25)
        #expect(try qwen35CPUColumnCount(outputColumns: n, fraction: 0.75) == 37)
        #expect(try qwen35CPUColumnCount(outputColumns: n, fraction: 1.0) == 50)
        #expect(try qwen35CPUColumnCount(outputColumns: n, fraction: 0.3) == 15)
        let third = try qwen35CPUColumnCount(outputColumns: n, fraction: 1.0 / 3.0)
        #expect(third == 16)
        #expect(n % 32 != 0)
    }

    @Test
    func positiveFractionThatAssignsZeroColumnsRefuses() {
        #expect(
            throws: Qwen35CPUAssistError.silentGPUFallback(
                fraction: 0.01, columns: 50)
        ) {
            try qwen35CPUColumnCount(outputColumns: 50, fraction: 0.01)
        }
    }

    @Test
    func unsupportedQuantizationRefusesWhenSplitting() throws {
        let p = makeProj(m: 4, n: 64, k: 64)
        #expect(throws: Qwen35CPUAssistError.self) {
            try qwen35ColumnSplitQuantizedMM(
                p.x, p.w, scales: p.scales, biases: p.biases,
                groupSize: 32, bits: 4, mode: .affine,
                cpuColumnFraction: 0.5)
        }
    }

    // MARK: - CPU quantized matmul exists

    @Test
    func cpuAffine4Group64QuantizedMatmulEvaluates() throws {
        let cpuStream = MLX.Stream(Device.cpu)
        let p = makeProj(m: 8, n: 64, k: 64)
        let y = try withError { error in
            let out = cpuOnly(p, stream: cpuStream)
            try error.check()
            return out
        }
        #expect(y.shape == [1, 8, 64])
    }

    // MARK: - Equivalence and split invariance

    @Test
    func smallShapeHalvesMatchTheirDevicesAndGPUBoundIsMeasured() throws {
        let cpuStream = MLX.Stream(Device.cpu)
        let gpuStream = MLX.Stream.gpu
        let p = makeProj(m: 8, n: 50, k: 64)
        let gpu = gpuOnly(p)
        let cpu = cpuOnly(p, stream: cpuStream)
        let gpuVsCPUAbs = maxAbs(gpu, cpu)
        let gpuVsCPURel = globalRel(gpu, cpu)
        let gpuScale = abs(gpu.asType(.float32)).max().item(Float.self)
        print(
            "cpu-vs-gpu small M=8 N=50 K=64 maxAbs=\(gpuVsCPUAbs) globalRel=\(gpuVsCPURel) gpuMax=\(gpuScale)"
        )
        #expect(gpuVsCPUAbs > 0)

        let fractions: [Double] = [0, 0.125, 0.25, 0.3, 0.5, 0.75, 1.0 / 3.0, 1.0]
        var maxSplitVsGPURel: Float = 0
        var maxSplitVsGPUAbs: Float = 0
        for fraction in fractions {
            let split = try qwen35ColumnSplitQuantizedMM(
                p.x, p.w, scales: p.scales, biases: p.biases,
                groupSize: 64, bits: 4, mode: .affine,
                cpuColumnFraction: fraction,
                cpuStream: cpuStream, gpuStream: gpuStream)
            eval(split.output)
            #expect(split.output.shape == [1, 8, 50])
            let cpuN = try qwen35CPUColumnCount(
                outputColumns: 50, fraction: fraction)
            #expect(split.cpuColumns == cpuN)
            #expect(split.gpuColumns == 50 - cpuN)

            if cpuN == 0 {
                #expect(maxAbs(split.output, gpu) == 0)
            } else if cpuN == 50 {
                #expect(maxAbs(split.output, cpu) == 0)
            } else {
                let cpuSlice = cpu[.ellipsis, 0..<cpuN]
                let gpuSlice = gpu[.ellipsis, cpuN...]
                let splitCPU = split.output[.ellipsis, 0..<cpuN]
                let splitGPU = split.output[.ellipsis, cpuN...]
                eval(cpuSlice, gpuSlice, splitCPU, splitGPU)
                #expect(maxAbs(splitCPU, cpuSlice) == 0)
                #expect(maxAbs(splitGPU, gpuSlice) == 0)
            }

            let absErr = maxAbs(split.output, gpu)
            let relErr = globalRel(split.output, gpu)
            if absErr > maxSplitVsGPUAbs { maxSplitVsGPUAbs = absErr }
            if relErr > maxSplitVsGPURel { maxSplitVsGPURel = relErr }
            print(
                "split fraction=\(fraction) cpuN=\(cpuN) vsGPU maxAbs=\(absErr) globalRel=\(relErr)"
            )
        }

        // A mixed split's disagreement with GPU-only is exactly the CPU
        // kernel's disagreement on the columns it owns, so it cannot exceed
        // full CPU-vs-GPU.
        #expect(maxSplitVsGPUAbs <= gpuVsCPUAbs + 1e-5)
        #expect(maxSplitVsGPURel <= gpuVsCPURel + 1e-5)
        print(
            "small-shape measured max split-vs-GPU globalRel=\(maxSplitVsGPURel) abs=\(maxSplitVsGPUAbs)"
        )
    }

    @Test
    func splitInvarianceAcrossFractions() throws {
        let cpuStream = MLX.Stream(Device.cpu)
        let p = makeProj(m: 8, n: 50, k: 64)
        let gpu = gpuOnly(p)
        let bound = maxAbs(cpuOnly(p, stream: cpuStream), gpu)
        let fractions: [Double] = [0, 0.125, 0.25, 0.3, 0.5, 0.75, 1.0]
        var outputs: [MLXArray] = []
        for fraction in fractions {
            let split = try qwen35ColumnSplitQuantizedMM(
                p.x, p.w, scales: p.scales, biases: p.biases,
                groupSize: 64, bits: 4, mode: .affine,
                cpuColumnFraction: fraction,
                cpuStream: cpuStream, gpuStream: .gpu)
            eval(split.output)
            outputs.append(split.output)
        }
        for i in 0..<outputs.count {
            for j in i..<outputs.count {
                let absErr = maxAbs(outputs[i], outputs[j])
                #expect(absErr <= bound + 1e-5)
            }
        }
    }

    @Test
    func realGateUpShapeSplitMatchesHalves() throws {
        let cpuStream = MLX.Stream(Device.cpu)
        let p = makeProj(m: 1024, n: 17408, k: 5120)
        let gpu = gpuOnly(p)
        let fraction = 0.125
        let split = try qwen35ColumnSplitQuantizedMM(
            p.x, p.w, scales: p.scales, biases: p.biases,
            groupSize: 64, bits: 4, mode: .affine,
            cpuColumnFraction: fraction,
            cpuStream: cpuStream, gpuStream: .gpu)
        eval(split.output)
        let cpuN = try qwen35CPUColumnCount(
            outputColumns: 17408, fraction: fraction)
        #expect(split.cpuColumns == cpuN)
        #expect(split.output.shape == [1, 1024, 17408])

        let cpuDirect = quantizedMM(
            p.x, p.w[0..<cpuN], scales: p.scales[0..<cpuN],
            biases: p.biases[0..<cpuN],
            transpose: true, groupSize: 64, bits: 4, mode: .affine,
            stream: .stream(cpuStream))
        let gpuDirect = quantizedMM(
            p.x, p.w[cpuN...], scales: p.scales[cpuN...],
            biases: p.biases[cpuN...],
            transpose: true, groupSize: 64, bits: 4, mode: .affine,
            stream: .gpu)
        eval(cpuDirect, gpuDirect)
        #expect(maxAbs(split.output[.ellipsis, 0..<cpuN], cpuDirect) == 0)
        #expect(maxAbs(split.output[.ellipsis, cpuN...], gpuDirect) == 0)
        let rel = globalRel(split.output, gpu)
        let absErr = maxAbs(split.output, gpu)
        let gpuScale = abs(gpu.asType(.float32)).max().item(Float.self)
        print(
            "gate_up M=1024 N=17408 K=5120 fraction=0.125 vsGPU maxAbs=\(absErr) globalRel=\(rel) gpuMax=\(gpuScale)"
        )
        #expect(absErr.isFinite)
        #expect(gpuScale > 0)
    }

    @Test
    func realDownShapeSplitMatchesHalves() throws {
        let cpuStream = MLX.Stream(Device.cpu)
        let p = makeProj(m: 1024, n: 5120, k: 17408)
        let gpu = gpuOnly(p)
        let fraction = 0.25
        let split = try qwen35ColumnSplitQuantizedMM(
            p.x, p.w, scales: p.scales, biases: p.biases,
            groupSize: 64, bits: 4, mode: .affine,
            cpuColumnFraction: fraction,
            cpuStream: cpuStream, gpuStream: .gpu)
        eval(split.output)
        let cpuN = try qwen35CPUColumnCount(
            outputColumns: 5120, fraction: fraction)
        #expect(split.cpuColumns == cpuN)
        #expect(split.output.shape == [1, 1024, 5120])
        let cpuDirect = quantizedMM(
            p.x, p.w[0..<cpuN], scales: p.scales[0..<cpuN],
            biases: p.biases[0..<cpuN],
            transpose: true, groupSize: 64, bits: 4, mode: .affine,
            stream: .stream(cpuStream))
        let gpuDirect = quantizedMM(
            p.x, p.w[cpuN...], scales: p.scales[cpuN...],
            biases: p.biases[cpuN...],
            transpose: true, groupSize: 64, bits: 4, mode: .affine,
            stream: .gpu)
        eval(cpuDirect, gpuDirect)
        #expect(maxAbs(split.output[.ellipsis, 0..<cpuN], cpuDirect) == 0)
        #expect(maxAbs(split.output[.ellipsis, cpuN...], gpuDirect) == 0)
        let rel = globalRel(split.output, gpu)
        let absErr = maxAbs(split.output, gpu)
        let gpuScale = abs(gpu.asType(.float32)).max().item(Float.self)
        print(
            "down M=1024 N=5120 K=17408 fraction=0.25 vsGPU maxAbs=\(absErr) globalRel=\(rel) gpuMax=\(gpuScale)"
        )
        #expect(absErr.isFinite)
        #expect(gpuScale > 0)
    }

    // MARK: - Device placement

    @Test
    func cpuStreamIsObservablyCPUByLoadAndInverse() throws {
        let cpuStream = MLX.Stream(Device.cpu)
        let wrapped = StreamOrDevice.stream(cpuStream)
        #expect(wrapped.stream == cpuStream)
        #expect(wrapped.description.contains("cpu"))

        let cpuLoad = try loadProbe(stream: wrapped)
        #expect(cpuLoad.ok)

        let gpuLoad = try loadProbe(stream: .gpu)
        #expect(gpuLoad.ok == false)
        #expect(gpuLoad.error?.contains("eval_gpu") == true)

        // Inverse is the other CPU-only primitive in primitives.cpp. If this
        // MLX build still lacks a Metal inverse, the GPU call throws
        // eval_gpu; if it has one, Load above remains the witness.
        let eye = identity(2)
        let invCPU = try withError { error in
            let r = inv(eye, stream: wrapped)
            r.eval()
            try error.check()
            return r
        }
        #expect(invCPU.shape == [2, 2])

        var gpuInv: String?
        do {
            _ = try withError {
                let r = inv(eye, stream: .gpu)
                r.eval()
            }
        } catch {
            gpuInv = String(describing: error)
        }
        print("gpu inverse error: \(gpuInv ?? "none (GPU inverse exists)")")
        #expect(gpuInv != nil)
    }

    @Test
    func fractionOneUsesTheCPUStreamNotASilentGPUCopy() throws {
        let cpuStream = MLX.Stream(Device.cpu)
        let p = makeProj(m: 8, n: 64, k: 64)
        let gpu = gpuOnly(p)
        let cpu = cpuOnly(p, stream: cpuStream)
        let split = try qwen35ColumnSplitQuantizedMM(
            p.x, p.w, scales: p.scales, biases: p.biases,
            groupSize: 64, bits: 4, mode: .affine,
            cpuColumnFraction: 1.0,
            cpuStream: cpuStream, gpuStream: .gpu)
        eval(split.output)
        #expect(split.cpuColumns == 64)
        #expect(split.gpuColumns == 0)
        #expect(maxAbs(split.output, cpu) == 0)

        // If the helper silently ran the "CPU" half on the GPU, fraction 1
        // would match GPU-only and (when the kernels differ) miss CPU-only.
        let vsGPU = maxAbs(split.output, gpu)
        let vsCPU = maxAbs(split.output, cpu)
        print(
            "fraction=1 vsCPU=\(vsCPU) vsGPU=\(vsGPU) cpuVsGPU=\(maxAbs(cpu, gpu))"
        )
        #expect(vsCPU == 0)
        if maxAbs(cpu, gpu) > 0 {
            #expect(vsGPU > 0)
        }
    }

    @Test
    func helperSourcePlacesQuantizedMMOnStreamFactory() throws {
        let source = try String(
            contentsOfFile:
                "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift",
            encoding: .utf8)
        #expect(source.contains("stream: .stream(stream)"))
        #expect(source.contains("StreamOrDevice.stream(cpuStream)"))
        #expect(source.contains("StreamOrDevice.stream(gpuStream)"))
        #expect(source.contains("eval(cpuY, gpuY)"))
        #expect(
            source.contains("MLXFAST_PREFILL_CPU_COLUMN_FRACTION"))
        #expect(
            source.contains("refusing a silent GPU-only run"))
        #expect(
            source.contains(
                "if fraction > 0, x.dim(-2) > 16, let y = fusedGateUp(x)"))
    }

    // MARK: - Default off / routed path

    @Test
    func routedPathDefaultIsGPUOnly() throws {
        unsetenv(qwen35PrefillCPUColumnFractionEnv)
        let p = makeProj(m: 8, n: 64, k: 64)
        let routed = qwen35RoutedQuantizedMM(
            p.x, p.w, scales: p.scales, biases: p.biases,
            groupSize: 64, bits: 4, mode: .affine)
        let gpu = gpuOnly(p)
        eval(routed)
        #expect(maxAbs(routed, gpu) == 0)
    }

    @Test
    func routedPathHonoursEnvFraction() throws {
        let cpuStream = MLX.Stream(Device.cpu)
        let p = makeProj(m: 8, n: 64, k: 64)
        setenv(qwen35PrefillCPUColumnFractionEnv, "0.5", 1)
        defer { unsetenv(qwen35PrefillCPUColumnFractionEnv) }
        let routed = qwen35RoutedQuantizedMM(
            p.x, p.w, scales: p.scales, biases: p.biases,
            groupSize: 64, bits: 4, mode: .affine)
        eval(routed)
        let split = try qwen35ColumnSplitQuantizedMM(
            p.x, p.w, scales: p.scales, biases: p.biases,
            groupSize: 64, bits: 4, mode: .affine,
            cpuColumnFraction: 0.5,
            cpuStream: cpuStream, gpuStream: .gpu)
        eval(split.output)
        #expect(maxAbs(routed, split.output) == 0)
    }

    @Test
    func fusedMLPPrefillUsesSplitWhenEnabled() throws {
        unsetenv(qwen35PrefillCPUColumnFractionEnv)
        let mlp = Qwen35FusedMLP(dimensions: 64, hiddenDimensions: 128)
        quantize(model: mlp, groupSize: 64, bits: 4, mode: .affine)
        MLXRandom.seed(3)
        let x = MLXRandom.normal([1, 32, 64]).asType(.bfloat16)
        eval(x)
        let gpuY = mlp(x)
        eval(gpuY)

        setenv(qwen35PrefillCPUColumnFractionEnv, "0.5", 1)
        defer { unsetenv(qwen35PrefillCPUColumnFractionEnv) }
        let splitY = mlp(x)
        eval(splitY)
        #expect(splitY.shape == gpuY.shape)
        let rel = globalRel(splitY, gpuY)
        let absErr = maxAbs(splitY, gpuY)
        print("fusedMLP prefill M=32 vs GPU-only maxAbs=\(absErr) globalRel=\(rel)")
        #expect(rel < 0.05)
    }
}
