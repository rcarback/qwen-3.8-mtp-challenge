import Foundation
import MLX
import MLXRandom
import Testing

/// DECISIVE BASELINE (investigation, untracked): times 4-bit quantizedMM at the
/// gate/up projection shape (S=512, K=5120, N=17408) in a process that loads
/// NO ANE Core ML model. If this runs ~13.5 TF, the ANE model's mere residency
/// is what depresses splitBeatsGPU's gpu_alone leg to ~6.8 TF -- meaning the
/// channel-split's baseline is self-handicapped and the 1.41x ratio overstates
/// the real (no-ANE-resident) no-offload baseline. If this ALSO runs ~7 TF, the
/// depression is intrinsic to this shape/process and the offload baseline is honest.
@Suite(.serialized)
struct CleanGPUBaselineTests {
    @Test("clean GPU-only quantizedMM TF at gate shape, no ANE model resident")
    func cleanGpuOnly() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let S = 512, K = 5120, N = 17408
        let w = MLXRandom.normal([N, K]).asType(.bfloat16)
        let (wq, scales, biasesOpt) = quantized(w, groupSize: 64, bits: 4)
        let biases = biasesOpt ?? scales
        let x = MLXRandom.normal([1, S, K]).asType(.bfloat16)
        eval(x, wq, scales, biases)
        let flop = 2.0 * Double(S) * Double(K) * Double(N)

        func gpuTF(window: Double) -> Double {
            var n = 0
            let dl = Date().addingTimeInterval(window)
            var last = x
            while Date() < dl {
                last = quantizedMM(x, wq, scales: scales, biases: biases,
                                   transpose: true, groupSize: 64, bits: 4)
                eval(last)
                n += 1
            }
            return Double(n) * flop / window / 1e12
        }
        _ = gpuTF(window: 0.4) // warmup, discarded
        var tfs: [Double] = []
        var report = "# Clean GPU-only quantizedMM (no ANE model), gate shape S=512 K=5120 N=17408\n\n| cycle | TF |\n|---|---|\n"
        for c in 0 ..< 6 {
            let tf = gpuTF(window: 0.5)
            tfs.append(tf)
            report += "| \(c) | \(String(format: "%.2f", tf)) |\n"
        }
        let mean = tfs.reduce(0, +) / Double(tfs.count)
        let sorted = tfs.sorted()
        report += String(format: "\n- [MEASURED] clean GPU-only mean = %.2f TF (min %.2f, max %.2f)\n",
                         mean, sorted.first ?? 0, sorted.last ?? 0)
        report += "- Compare: splitBeatsGPU gpu_alone leg (ANE model resident) measured ~6.78 TF idle.\n"
        report += "- If this >> 6.78 (toward ~13.5): ANE residency taxes the GPU; offload baseline is self-handicapped.\n"
        report += "- If this ~= 6.78: depression is intrinsic; offload baseline is honest.\n"
        let path = ProcessInfo.processInfo.environment["CLEAN_GPU_REPORT"]
            ?? "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/clean-gpu-baseline.md"
        try? report.write(toFile: path, atomically: true, encoding: .utf8)
        print(report)
        #expect(mean > 0)
    }
}
