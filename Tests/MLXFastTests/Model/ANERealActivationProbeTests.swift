import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLLM

/// Model-free ANE numerics probe on CAPTURED real activations
/// (`ANERealActivationCaptureTests` writes them). For each captured layer:
///
///   1. The fused ANE program on the real `x` and real fp16 prefix weights,
///      against an fp32 reference and the GPU-fp16 ablation, under every
///      activation spelling (`ANEActivation`) and with no activation at all
///      (isolates the convs and mul). Found 2026-09-02: the MIL `silu` /
///      `sigmoid` ops carry 10-30x the fp16 floor; `expDiv` does not.
///   2. The PRODUCTION split object (ANE prefix + 4-bit GPU suffix) through
///      its concurrent and sequential paths against the all-GPU 4-bit MLP.
///   3. Every production program kept resident and re-run at the end -- the
///      regression check for the descriptor-identity collision that made
///      all but the first loaded program compute with the wrong weights.
///
/// Gated on MLXFAST_RUN_MLX_RUNTIME_TESTS and MLX_ANE_CAPTURE_DIR.
@Suite(.serialized)
struct ANERealActivationProbeTests {
    @Test("ANE error on real activations: activation spellings, production split, resident programs")
    func probe() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            let captureDir = env["MLX_ANE_CAPTURE_DIR"]
        else { return }
        try #require(ANERuntime.available())
        let fraction = Double(env["MLX_ANE_FRACTION"] ?? "0.3125") ?? 0.3125

        let files = try FileManager.default.contentsOfDirectory(atPath: captureDir)
            .filter { $0.hasPrefix("mlp-layer") && $0.hasSuffix(".safetensors") }
            .sorted { layerNumber($0) < layerNumber($1) }
        try #require(!files.isEmpty, "no captures under \(captureDir)")

        print("ANE-PROBE: layer  S  F | max|x| top-x-channels | max|gate| max|up| max|act| max|ref| | silu-ANE gpu16 | none-ANE | activation variants  (maxAbs)")
        // Round 4: every production program stays LOADED for the whole run
        // (as in the model, where all 64 layers' programs coexist) and is
        // re-run at the end. Programs whose descriptor identity collided
        // computed with each other's weights; this is the regression check.
        var resident: [(layer: Int, split: ANEFusedSplitMLP, x: MLXArray, full4: MLXArray)] = []
        for file in files {
            let layer = layerNumber(file)
            let arrays = try loadArrays(url: URL(fileURLWithPath: captureDir).appendingPathComponent(file))
            let x = try #require(arrays["x"])
            let S = x.dim(0), hidden = x.dim(1)
            let inter = try #require(arrays["gate.weight"]).dim(0)
            let F = ANEFusedSplitMLP.prefixChannels(inter: inter, aneFraction: fraction)

            func triple(_ name: String) throws -> (MLXArray, MLXArray, MLXArray) {
                (try #require(arrays["\(name).weight"]), try #require(arrays["\(name).scales"]),
                 try #require(arrays["\(name).biases"]))
            }
            let (gW, gS, gB) = try triple("gate")
            let (uW, uS, uB) = try triple("up")
            let (dW, dS, dB) = try triple("down")
            let gate = ANEWeightPrep.dequantizeFP16(wq: gW, scales: gS, biases: gB, channelStart: 0, channelEnd: F)
            let up = ANEWeightPrep.dequantizeFP16(wq: uW, scales: uS, biases: uB, channelStart: 0, channelEnd: F)
            let down = ANEWeightPrep.dequantizeFP16Columns(wq: dW, scales: dS, biases: dB, columnStart: 0, columnEnd: F)
            eval(gate, up, down)

            // fp32 reference on the same fp16 weights.
            let x32 = x.asType(.float32)
            let g32 = matmul(x32, gate.asType(.float32).transposed(1, 0))
            let u32 = matmul(x32, up.asType(.float32).transposed(1, 0))
            let act32 = silu(g32) * u32
            let ref = matmul(act32, down.asType(.float32).transposed(1, 0))
            eval(ref)

            // GPU fp16 (the healthy ablation).
            let x16 = x.asType(.float16)
            let gpu = matmul(silu(matmul(x16, gate.transposed(1, 0))) * matmul(x16, up.transposed(1, 0)),
                             down.transposed(1, 0)).asType(.float32)
            eval(gpu)

            func maxAbs(_ a: MLXArray) -> Float { let m = MLX.abs(a).max(); eval(m); return m.item(Float.self) }
            func err(_ a: MLXArray) -> Float { maxAbs(a - ref) }
            let xAbsMax = MLX.abs(x32).max(axis: 0)  // [hidden]
            eval(xAbsMax)
            let topX = argSort(xAbsMax)[(hidden - 4)...].asArray(Int32.self).reversed()
            let topXVals = topX.map { String(format: "%d:%.0f", $0, xAbsMax[Int($0)].item(Float.self)) }

            func ane(_ activation: ANEActivation) throws -> MLXArray {
                let program = try ANEFusedMLP(hidden: hidden, innerFraction: F, sequenceLength: S,
                                              gate: gate, up: up, down: down, activation: activation)
                let y = try program(x16).asType(.float32)
                eval(y)
                return y
            }
            // Round 1 (scaling folds E1-E4) left the error unchanged at 3
            // decimals on every layer, so magnitude is not the mechanism.
            // Round 2 swaps the activation spelling; `.none` compares against
            // its own reference (act = gate * up) to isolate the convs + mul.
            let A = try ane(.silu)
            let refNone = matmul(g32 * u32, down.asType(.float32).transposed(1, 0))
            let noneErr = maxAbs(try ane(.none) - refNone)
            let refNoneMax = maxAbs(refNone)
            var variantText = ""
            for variant in [ANEActivation.sigmoidMul, .expDiv, .tanhForm] {
                let e: Float
                do { e = err(try ane(variant)) } catch { e = .nan }
                variantText += String(format: " %@ %.4f", variant.rawValue, e)
            }

            // Round 3: the PRODUCTION object (ANE prefix + 4-bit GPU suffix)
            // through its concurrent path and its sequential twin, against
            // the all-GPU 4-bit reference over ALL channels. This is the
            // exact computation the hybrid substitutes for the MLP.
            let full4 = { () -> MLXArray in
                let g = quantizedMM(x, gW, scales: gS, biases: gB, transpose: true, groupSize: 64, bits: 4)
                let u = quantizedMM(x, uW, scales: uS, biases: uB, transpose: true, groupSize: 64, bits: 4)
                let a = (silu(g) * u).asType(.bfloat16)
                let y = quantizedMM(a, dW, scales: dS, biases: dB, transpose: true, groupSize: 64, bits: 4).asType(.float32)
                eval(y)
                return y
            }()
            let split = try ANEFusedSplitMLP(
                gateW: gW, gateScales: gS, gateBiases: gB, upW: uW, upScales: uS, upBiases: uB,
                downW: dW, downScales: dS, downBiases: dB,
                hidden: hidden, inter: inter, sequenceLength: S, aneFraction: fraction)
            let concurrent1 = try split(x).asType(.float32)
            let concurrent2 = try split(x).asType(.float32)
            let sequential = try split.sequentialCallAsFunctionForTesting(x).asType(.float32)
            eval(concurrent1, concurrent2, sequential)
            let full4Max = maxAbs(full4)
            print(String(
                format: "ANE-PROBE-SPLIT: layer %d | max|full4| %.2f | concurrent-vs-full4 %.4f | concurrent(2nd call) %.4f | sequential-vs-full4 %.4f | concurrent-vs-sequential %.4f",
                layer, full4Max, maxAbs(concurrent1 - full4), maxAbs(concurrent2 - full4),
                maxAbs(sequential - full4), maxAbs(concurrent1 - sequential)))

            resident.append((layer, split, x, full4))

            let refMax = maxAbs(ref)
            let row = String(
                format: "ANE-PROBE: %5d %4d %5d | %8.1f %@ | %8.1f %8.1f %10.1f %8.1f | silu-ANE %.4f gpu16 %.4f | none-ANE %.4f (of %.1f) |%@",
                layer, S, F, maxAbs(x32), topXVals.joined(separator: ","),
                maxAbs(g32), maxAbs(u32), maxAbs(act32), refMax,
                err(A), err(gpu), noneErr, refNoneMax, variantText)
            print(row)
            #expect(err(gpu).isFinite)
        }

        for entry in resident {
            let y = try entry.split(entry.x).asType(.float32)
            let d = MLX.abs(y - entry.full4)
            let scale = MLX.abs(entry.full4).max()
            eval(d, scale)
            let e = d.max().item(Float.self), m = scale.item(Float.self)
            print(String(format: "ANE-PROBE-RESIDENT: layer %d | all %d programs loaded | vs-full4 %.4f of %.2f", entry.layer, resident.count, e, m))
            // bf16 output rounding of the reference is the expected floor.
            #expect(e <= max(0.02 * m, 0.05), "layer \(entry.layer): resident program error \(e) on scale \(m)")
        }
    }

    private func layerNumber(_ file: String) -> Int {
        Int(file.dropFirst("mlp-layer".count).dropLast(".safetensors".count)) ?? -1
    }
}
