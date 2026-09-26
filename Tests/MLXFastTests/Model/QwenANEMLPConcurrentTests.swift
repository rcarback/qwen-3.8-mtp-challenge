import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANEMLPConcurrentTests {
    @Test("captured MLP rows run on ANE alongside independent GPU rows")
    func replay() throws {
        let env = ProcessInfo.processInfo.environment
        let stages = env["MLXFAST_ANE_MLP_STAGES"] == "1"
        let matchedMath = env["MLXFAST_ANE_MATCHED_MATH"] == "1"
        let preciseWeights = env["MLXFAST_ANE_FP32_EXPANSION"] == "1"
        guard stages || matchedMath || env["MLXFAST_ANE_MLP_CONCURRENT"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let arrays = try MLX.loadArrays(url: root.appendingPathComponent("mlp-capture.safetensors"))
        let metadata = try #require(try JSONSerialization.jsonObject(
            with: Data(contentsOf: root.appendingPathComponent("mlp-capture.json"))) as? [String: Any])
        var expanded: [String: MLXArray] = [:]
        func array(_ key: String) throws -> MLXArray {
            if preciseWeights && ["gate", "up", "down"].contains(key) {
                if let value = expanded[key] { return value }
                let config = try #require(metadata[key] as? [String: Int])
                let value = MLX.dequantized(try #require(arrays["\(key)_packed"]),
                    scales: try #require(arrays["\(key)_scales"]).asType(.float32),
                    biases: try #require(arrays["\(key)_biases"]).asType(.float32),
                    groupSize: try #require(config["group_size"]), bits: try #require(config["bits"]), mode: .affine)
                eval(value)
                expanded[key] = value
                return value
            }
            return try #require(arrays[key])
        }
        let input = try array("input")
        let reference = try array("reference")
        try #require(input.dim(0) >= 64 && input.shape == reference.shape)
        let gate = try #require(metadata["gate"] as? [String: Int])
        let up = try #require(metadata["up"] as? [String: Int])
        let down = try #require(metadata["down"] as? [String: Int])
        try #require(gate == up)
        let gateBits = try #require(gate["bits"])
        let gateGroup = try #require(gate["group_size"])
        let downBits = try #require(down["bits"])
        let downGroup = try #require(down["group_size"])
        let packed = try concatenated([array("gate_packed"), array("up_packed")], axis: 0)
        let scales = try concatenated([array("gate_scales"), array("up_scales")], axis: 0)
        let biases = try concatenated([array("gate_biases"), array("up_biases")], axis: 0)
        let downPacked = try array("down_packed")
        let downScales = try array("down_scales")
        let downBiases = try array("down_biases")
        let activation = compile { (y: MLXArray) -> MLXArray in
            let half = y.dim(-1) / 2
            return silu(y[.ellipsis, ..<half]) * y[.ellipsis, half...]
        }
        func gpu(_ rows: MLXArray) -> MLXArray {
            let y = qwen35RoutedQuantizedMM(rows.reshaped([1, rows.dim(0), 5120]),
                packed, scales: scales, biases: biases, groupSize: gateGroup, bits: gateBits, mode: .affine)
            return qwen35RoutedQuantizedMM(activation(y), downPacked, scales: downScales,
                biases: downBiases, groupSize: downGroup, bits: downBits, mode: .affine)
                .reshaped([rows.dim(0), 5120])
        }
        func errors(_ value: MLXArray, _ expected: MLXArray) -> [String: Double] {
            let difference = abs(value.asType(.float32) - expected.asType(.float32))
            return ["max_absolute": max(difference).item(Double.self),
                "mean_absolute": mean(difference).item(Double.self),
                "exact_fraction": mean((value .== expected).asType(.float32)).item(Double.self)]
        }
        var evidence: [String: Any] = ["phase": "gpu-control", "captured_rows": input.dim(0),
            "weight_expansion": preciseWeights ? "float32" : "captured-bfloat16"]
        func save() throws {
            try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent(matchedMath ? "mlp-matched-math.json" : (stages ? (preciseWeights ? "mlp-stages-fp32.json" : "mlp-stages.json") : "mlp-concurrent.json")), options: .atomic)
        }
        var outputs: [MLXArray] = []
        // MLX eval materializes tensors; it does not evaluate source code.
        for row in 0..<input.dim(0) {
            let result = gpu(input[row..<(row + 1)])
            eval(result)
            outputs.append(result)
        }
        let single = concatenated(outputs, axis: 0)
        evidence["single_gpu_error"] = errors(single, reference)
        try save()
        try #require(all(single .== reference).item(Bool.self))
        if matchedMath {
            let gw = try array("gate").asType(.float16)
            let uw = try array("up").asType(.float16)
            let dw = try array("down").asType(.float16)
            let program = try ANEFusedMLP(hidden: 5120, innerFraction: gw.dim(0), sequenceLength: 32,
                gate: gw, up: uw, down: dw, activation: .expDiv, weightForm: .fp16)
            let expDiv = compile { (gate: MLXArray, up: MLXArray) -> MLXArray in
                let negative = gate * MLXArray(Float16(-1))
                let denominator = exp(negative) + MLXArray(Float16(1))
                return (gate / denominator) * up
            }
            var aneRows: [MLXArray] = []
            var compiledRows: [MLXArray] = []
            var roundedRows: [MLXArray] = []
            func rounded(_ x: MLXArray) -> MLXArray {
                let value = x.asType(.float16)
                eval(value)
                return value
            }
            for start in stride(from: 0, to: input.dim(0), by: 32) {
                let x = input[start..<(start + 32)].asType(.float16)
                let gate = rounded(matmul(x, gw.T))
                let up = rounded(matmul(x, uw.T))
                let compiledOutput = rounded(matmul(expDiv(gate, up), dw.T))
                compiledRows.append(compiledOutput)
                let negative = rounded(gate * MLXArray(Float16(-1)))
                let exponential = rounded(exp(negative))
                let denominator = rounded(exponential + MLXArray(Float16(1)))
                let siluOutput = rounded(gate / denominator)
                let activated = rounded(siluOutput * up)
                roundedRows.append(rounded(matmul(activated, dw.T)))
                let ane = try program(x)
                let copied = rounded(ane + MLXArray(Float16(0)))
                aneRows.append(copied)
            }
            let ane = concatenated(aneRows, axis: 0)
            let compiled = concatenated(compiledRows, axis: 0)
            let stepwise = concatenated(roundedRows, axis: 0)
            evidence["compiled_gpu_vs_ane_fp16"] = errors(compiled, ane)
            evidence["stepwise_gpu_vs_ane_fp16"] = errors(stepwise, ane)
            evidence["compiled_gpu_vs_ane_bf16"] = errors(compiled.asType(.bfloat16), ane.asType(.bfloat16))
            evidence["stepwise_gpu_vs_ane_bf16"] = errors(stepwise.asType(.bfloat16), ane.asType(.bfloat16))
            evidence["compiled_gpu_vs_original"] = errors(compiled.asType(.bfloat16), reference)
            evidence["stepwise_gpu_vs_original"] = errors(stepwise.asType(.bfloat16), reference)
            evidence["ane_vs_original"] = errors(ane.asType(.bfloat16), reference)
            evidence["batch_rows"] = 32
            evidence["activation"] = "FP16 negative, exp, add one, divide, multiply up"
            evidence["phase"] = "complete"
            try save()
            return
        }
        if stages {
            func singleRows(_ x: MLXArray, _ operation: (MLXArray) -> MLXArray) -> MLXArray {
                var values: [MLXArray] = []
                for row in 0..<x.dim(0) {
                    let value = operation(x[row..<(row + 1)].reshaped([1, 1, x.dim(1)]))
                    eval(value)
                    values.append(value.reshaped([1, value.dim(-1)]))
                }
                return concatenated(values, axis: 0)
            }
            func aneProjection(_ name: String, _ x: MLXArray) throws -> MLXArray {
                let weight = try array(name)
                let blob = buildConvWeightBlob(f16Bytes(weight))
                let tag = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
                let text = buildConvMILText(inputDim: weight.dim(1), outputDim: weight.dim(0),
                    sequenceLength: 32, programTag: tag)
                let model = try ANEInMemoryModel(milText: text, weightBlob: blob, weightFileName: "weight_data.bin")
                try model.compile()
                try model.load()
                defer { model.unload() }
                try #require(x.dim(0) % 32 == 0)
                var values: [MLXArray] = []
                for start in stride(from: 0, to: x.dim(0), by: 32) {
                    let value = try ANEDirectDispatch.runConv(model: model, x: x[start..<(start + 32)].asType(.float16),
                        inputDim: weight.dim(1), outputDim: weight.dim(0), sequenceLength: 32)
                    let copied = (value + MLXArray(Float16(0))).asType(.bfloat16)
                    eval(copied)
                    values.append(copied)
                }
                return concatenated(values, axis: 0)
            }
            let quantizedGU = singleRows(input) { x in
                qwen35RoutedQuantizedMM(x, packed, scales: scales, biases: biases,
                    groupSize: gateGroup, bits: gateBits, mode: .affine)
            }
            let activated = singleRows(quantizedGU, activation)
            let denseGUWeight = try concatenated([array("gate"), array("up")], axis: 0)
            let denseGU = singleRows(input) { matmul($0, denseGUWeight.asType(.bfloat16).T) }
            evidence["dense_bf16_gate_up_error"] = errors(denseGU, quantizedGU)
            if preciseWeights {
                let denseFP32 = singleRows(input) { matmul($0.asType(.float32), denseGUWeight.T).asType(.bfloat16) }
                evidence["dense_fp32_gate_up_error"] = errors(denseFP32, quantizedGU)
            }
            let denseFP16 = singleRows(input) { matmul($0.asType(.float16), denseGUWeight.asType(.float16).T) }
            evidence["dense_fp16_gate_up_error"] = errors(denseFP16.asType(.bfloat16), quantizedGU)
            evidence["phase"] = "projection-controls"
            try save()
            let aneGate = try aneProjection("gate", input)
            let aneUp = try aneProjection("up", input)
            let aneGU = concatenated([aneGate, aneUp], axis: 1)
            evidence["ane_gate_up_error"] = errors(aneGU, quantizedGU)
            let downWeight = try array("down")
            let denseDown = singleRows(activated) { matmul($0, downWeight.asType(.bfloat16).T) }
            evidence["dense_bf16_down_on_reference_activation_error"] = errors(denseDown, reference)
            let aneDown = try aneProjection("down", activated)
            evidence["ane_down_on_reference_activation_error"] = errors(aneDown, reference)
            evidence["phase"] = "isolated-projections"
            try save()
            let hybridActivation = singleRows(aneGU, activation)
            let hybridGPUDown = singleRows(hybridActivation) { x in
                qwen35RoutedQuantizedMM(x, downPacked, scales: downScales, biases: downBiases,
                    groupSize: downGroup, bits: downBits, mode: .affine)
            }
            evidence["ane_gate_up_gpu_activation_down_error"] = errors(hybridGPUDown, reference)
            let hybridANEDown = try aneProjection("down", hybridActivation)
            evidence["ane_projections_gpu_activation_error"] = errors(hybridANEDown, reference)
            let fused = try ANEFusedMLP(hidden: 5120, innerFraction: array("gate").dim(0), sequenceLength: 32,
                gate: array("gate"), up: array("up"), down: array("down"), activation: .expDiv, weightForm: .fp16)
            var fusedRows: [MLXArray] = []
            for start in stride(from: 0, to: input.dim(0), by: 32) {
                let value = try fused(input[start..<(start + 32)].asType(.float16)).asType(.bfloat16)
                eval(value)
                fusedRows.append(value)
            }
            evidence["fused_ane_error"] = errors(concatenated(fusedRows, axis: 0), reference)
            evidence["phase"] = "complete"
            try save()
            return
        }
        let program = try ANEFusedMLP(hidden: 5120, innerFraction: array("gate").dim(0), sequenceLength: 32,
            gate: array("gate"), up: array("up"), down: array("down"), activation: .expDiv, weightForm: .fp16)
        let aneInput = input[0..<32].asType(.float16)
        func gpuWork() {
            for start in stride(from: 32, to: 64, by: 4) {
                eval(gpu(input[start..<(start + 4)]))
            }
        }
        var lastOutput = try program(aneInput)
        eval(lastOutput)
        gpuWork()
        let modes = ["ane-only", "gpu-only", "serial", "overlap"]
        var trials: [[String: Any]] = []
        for repeatIndex in 0..<3 {
            for offset in modes.indices {
                let mode = modes[(repeatIndex + offset) % modes.count]
                let start = Date()
                for _ in 0..<100 {
                    if mode == "gpu-only" {
                        gpuWork()
                        continue
                    }
                    let prepared = try program.makeInput(aneInput)
                    switch mode {
                    case "ane-only": try program.predict(prepared)
                    case "serial": try program.predict(prepared); gpuWork()
                    default:
                        _ = try ConcurrentEngines.run(ane: { try program.predict(prepared) }, gpu: { gpuWork() })
                    }
                    lastOutput = program.readOutput(prepared)
                    eval(lastOutput)
                }
                trials.append(["mode": mode, "repeat": repeatIndex,
                    "seconds": Date().timeIntervalSince(start), "iterations": 100])
                evidence["trials"] = trials
                evidence["phase"] = "concurrent-trials"
                try save()
            }
        }
        evidence["ane_error"] = errors(lastOutput.asType(.bfloat16), reference[0..<32])
        evidence["phase"] = "complete"
        try save()
    }
}
