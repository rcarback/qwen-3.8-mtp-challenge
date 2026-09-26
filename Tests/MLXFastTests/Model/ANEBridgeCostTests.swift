import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXLLM

/// Isolated phase diagnostics at Qwen prefill geometry, not request timings.
@Suite(.serialized, .enabled(if:
    ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        && ProcessInfo.processInfo.environment["MLXFAST_ANE_COST"] == "1"))
struct ANEBridgeCostTests {
    private static let hidden = 5_120
    private static let inter = 17_408
    private static let rows = 512

    private struct Projection {
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray

        init(output: Int, input: Int, seed: UInt64) throws {
            MLXRandom.seed(seed)
            let dense = (MLXRandom.normal([output, input]) / Float(input).squareRoot())
                .asType(.bfloat16)
            let (weight, scales, biases) = quantized(dense, groupSize: 64, bits: 4)
            self.weight = weight
            self.scales = scales
            self.biases = try #require(biases, "Affine q4 weights require biases")
            eval(self.weight, self.scales, self.biases)
        }

        var linear: QuantizedLinear {
            QuantizedLinear(weight: weight, scales: scales, biases: biases, groupSize: 64, bits: 4)
        }
    }

    private static func timed<T>(_ body: () throws -> T) rethrows -> (T, Double) {
        let start = DispatchTime.now().uptimeNanoseconds
        let value = try body()
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
        #expect(milliseconds.isFinite && milliseconds > 0)
        return (value, milliseconds)
    }

    private static func complete(_ body: () throws -> MLXArray) rethrows -> (MLXArray, Double) {
        try timed {
            let output = try body()
            eval(output)
            return output
        }
    }

    private static func checkOutput(_ output: MLXArray) throws {
        try #require(output.shape == [rows, hidden])
        try #require(MLX.isFinite(output).all().item(Bool.self), "A measured arm returned nonfinite values")
    }

    private static func errors(_ output: MLXArray, reference: MLXArray) -> (Float, Float) {
        let difference = MLX.abs(output.asType(.float32) - reference.asType(.float32))
        let largest = difference.max()
        let mean = difference.mean()
        eval(largest, mean)
        return (largest.item(Float.self), mean.item(Float.self))
    }

    private static func bridgePhases(
        gate: Projection, up: Projection, down: Projection, inputs: [MLXArray],
        channels: Int, pairs: Int
    ) throws {
        let gatePrefix = ANEWeightPrep.dequantizeFP16(wq: gate.weight, scales: gate.scales,
            biases: gate.biases, channelStart: 0, channelEnd: channels)
        let upPrefix = ANEWeightPrep.dequantizeFP16(wq: up.weight, scales: up.scales,
            biases: up.biases, channelStart: 0, channelEnd: channels)
        let downPrefix = ANEWeightPrep.dequantizeFP16Columns(wq: down.weight, scales: down.scales,
            biases: down.biases, columnStart: 0, columnEnd: channels)
        eval(gatePrefix, upPrefix, downPrefix)
        let (program, programMS) = try timed {
            try ANEFusedMLP(hidden: hidden, innerFraction: channels, sequenceLength: rows,
                gate: gatePrefix, up: upPrefix, down: downPrefix)
        }
        print("ANE-COST phase=program_init ms=\(programMS) includes=weight_blob_compile_load daemon_cache=uncontrolled")
        let (seed, seedMS) = try timed { try program.makeInput(inputs[0]) }
        try program.predict(seed)
        eval(program.readOutput(seed))
        print("ANE-COST phase=first_gpu_prepare ms=\(seedMS) includes=allocation_request_packer_init_submit")

        // prepare owns setup and staging together. Do not subtract warm staging
        // to manufacture a supposedly isolated allocation time.
        let (initial, setupMS) = try timed {
            try ANEDirectDispatch.prepare(model: seed.model, x: inputs[0], inputDim: hidden,
                outputDim: hidden, sequenceLength: rows)
        }
        var prepared = initial
        try ANEDirectDispatch.evaluate(prepared)
        eval(ANEDirectDispatch.read(prepared))
        print("ANE-COST phase=new_storage_cpu_prepare ms=\(setupMS) includes=surface_request_creation_cpu_staging")
        print("ANE-COST unavailable=allocation_request_only reason=prepare_also_stages_input")
        print("ANE-COST control=already_staged_ane present=1 phase=ane_evaluate excludes=staging_and_output_join")
        print("ANE-COST unavailable=gpu_stage_completion_only reason=transfer_is_private_evaluate_includes_its_wait")

        let addend = MLXArray.ones([rows, hidden], dtype: .bfloat16)
        eval(addend)
        // Both warmup arms use the same storage. Complete every published
        // output before timing the next restage, so draining is not hidden work.
        for gpuInput in [false, true] {
            prepared = try ANEDirectDispatch.restage(prepared, x: inputs[0],
                inputDim: hidden, gpuInput: gpuInput)
            try ANEDirectDispatch.evaluate(prepared)
            eval(ANEDirectDispatch.read(prepared).asType(.bfloat16) + addend)
            eval(ANEDirectDispatch.readZeroCopy(prepared).asType(.bfloat16) + addend)
            eval(ANEDirectDispatch.readJoined(prepared, gpu: addend, rows: rows, dtype: .bfloat16))
        }

        var completed = 0
        for pair in 0 ..< pairs {
            let x = inputs[pair % inputs.count]
            let order = pair.isMultiple(of: 2) ? [false, true] : [true, false]
            var cpuOutput: MLXArray?
            var gpuOutput: MLXArray?
            for gpuInput in order {
                let (next, stageMS) = try timed {
                    try ANEDirectDispatch.restage(prepared, x: x, inputDim: hidden, gpuInput: gpuInput)
                }
                let sameRequest = next.request === initial.request
                let sameInput = next.inputSurface === initial.inputSurface
                let sameOutput = next.outputSurface === initial.outputSurface
                try #require(sameRequest && sameInput && sameOutput, "Staging arms must reuse the same storage")
                prepared = next
                let (_, evaluateMS) = try timed { try ANEDirectDispatch.evaluate(prepared) }
                completed += 1
                let stage = gpuInput ? "gpu_submit" : "cpu_stage_complete"
                let execution = gpuInput ? "gpu_wait_and_ane_evaluate" : "ane_evaluate"
                print("ANE-COST pair=\(pair) phase=\(stage) ms=\(stageMS)")
                print("ANE-COST pair=\(pair) phase=\(execution) ms=\(evaluateMS) completed=1")

                // Each closure builds a fresh output from this completed ANE
                // call. The addend is a ready tensor, not an active GPU suffix.
                let reads: [(String, () -> MLXArray)] = [
                    ("host_read_cast_add", { ANEDirectDispatch.read(prepared).asType(.bfloat16) + addend }),
                    ("contiguous_read_cast_add", { ANEDirectDispatch.readZeroCopy(prepared).asType(.bfloat16) + addend }),
                    ("fused_join", { ANEDirectDispatch.readJoined(prepared, gpu: addend, rows: rows, dtype: .bfloat16) }),
                ]
                var outputs: [String: MLXArray] = [:]
                let readOrder = pair.isMultiple(of: 2) ? [0, 1, 2] : [2, 1, 0]
                for index in readOrder {
                    let (label, read) = reads[index]
                    let (output, readMS) = complete(read)
                    try checkOutput(output)
                    outputs[label] = output
                    print("ANE-COST pair=\(pair) staging=\(stage) phase=\(label) ms=\(readMS) includes=output_completion")
                }
                let owned = try #require(outputs["host_read_cast_add"])
                let joined = try #require(outputs["fused_join"])
                let (largest, mean) = errors(joined, reference: owned)
                print("ANE-COST-ERROR pair=\(pair) comparison=fused_vs_host max_abs=\(largest) mean_abs=\(mean)")
                if gpuInput { gpuOutput = joined } else { cpuOutput = joined }
            }
            let cpu = try #require(cpuOutput)
            let gpu = try #require(gpuOutput)
            let (largest, mean) = errors(gpu, reference: cpu)
            print("ANE-COST-ERROR pair=\(pair) comparison=gpu_vs_cpu_staging max_abs=\(largest) mean_abs=\(mean)")
        }
        try #require(completed == pairs * 2)
        print("ANE-COST completed_measured_ane=\(completed) phase_sum=not_request_latency")
    }

    @Test("Report completed bridge phases and paired production-layer controls")
    func measuredControls() throws {
        let env = ProcessInfo.processInfo.environment
        let fraction = try #require(Double(env["MLXFAST_ANE_FRACTION"] ?? "0.5"), "Invalid MLXFAST_ANE_FRACTION")
        let pairs = try #require(Int(env["MLXFAST_ANE_COST_PAIRS"] ?? "3"), "Invalid MLXFAST_ANE_COST_PAIRS")
        try #require(fraction.isFinite && fraction > 0 && fraction < 1)
        try #require((1 ... 12).contains(pairs), "MLXFAST_ANE_COST_PAIRS must be 1 through 12")
        let channels = ANEFusedSplitMLP.prefixChannels(inter: Self.inter, aneFraction: fraction)
        try #require(channels > 0 && channels < Self.inter, "The rounded fraction must leave both engines work")
        try #require(!ANESplitConfig.enabled, "Set MLX_ANE_DIRECT=0 for the production GPU control")
        try #require(!ANESplitConfig.verify && !ANESplitConfig.fp16GpuAblate, "Turn off ANE verification and GPU-prefix ablation")
        let cpuFraction = try qwen35PrefillCPUColumnFraction()
        try #require(cpuFraction == 0, "The production GPU control must not use CPU column splitting")
        try #require(env["MLX_ANE_CAPTURE_DIR"] == nil, "Turn off activation capture during measurement")
        try #require(ANEWeightForm(rawValue: env["MLX_ANE_WEIGHT_FORM"] ?? "fp16") != nil, "Invalid ANE weight form")
        try #require(ANEActivation(rawValue: env["MLX_ANE_ACTIVATION"] ?? "expDiv") != nil, "Invalid ANE activation")
        try #require(ANERuntime.available())

        print("ANE-COST geometry=H5120_I17408_S512 fraction=\(fraction) channels=\(channels) pairs=\(pairs) weight_form=\(ANESplitConfig.weightForm.rawValue) activation=\(ANESplitConfig.activation.rawValue) gpu_suffix=\(ANESplitConfig.fp16GpuSuffix ? "fp16" : "current")")
        print("ANE-COST scope=synthetic_single_layer thermal_gate=external phase_samples=isolated accuracy=reported_without_threshold")
        let gate = try Projection(output: Self.inter, input: Self.hidden, seed: 101)
        let up = try Projection(output: Self.inter, input: Self.hidden, seed: 102)
        let down = try Projection(output: Self.hidden, input: Self.inter, seed: 103)
        MLXRandom.seed(104)
        let inputs = [MLXRandom.normal([Self.rows, Self.hidden]).asType(.bfloat16),
                      MLXRandom.normal([Self.rows, Self.hidden]).asType(.bfloat16)]
        eval(inputs)
        try Self.bridgePhases(gate: gate, up: up, down: down, inputs: inputs, channels: channels, pairs: pairs)

        let production = Qwen35FusedMLP(dimensions: Self.hidden, hiddenDimensions: Self.inter)
        production.update(modules: ModuleChildren.unflattened([
            ("gate_proj", gate.linear), ("up_proj", up.linear), ("down_proj", down.linear),
        ]))
        let (split, initMS) = try Self.timed {
            try ANEFusedSplitMLP(gateW: gate.weight, gateScales: gate.scales, gateBiases: gate.biases,
                upW: up.weight, upScales: up.scales, upBiases: up.biases,
                downW: down.weight, downScales: down.scales, downBiases: down.biases,
                hidden: Self.hidden, inter: Self.inter, sequenceLength: Self.rows, aneFraction: fraction)
        }
        print("ANE-COST phase=split_init ms=\(initMS) daemon_cache=uncontrolled")
        eval(production(inputs[0]))
        eval(try split.sequentialCallAsFunctionForTesting(inputs[0]))
        eval(try split(inputs[0]))
        for concurrent in [false, true] {
            let mode = concurrent ? "concurrent" : "sequential"
            for pair in 0 ..< pairs {
                let x = inputs[pair % inputs.count]
                let baseline = { production(x) }
                let candidate = { () throws -> MLXArray in
                    if concurrent { return try split(x) }
                    return try split.sequentialCallAsFunctionForTesting(x)
                }
                let control: (MLXArray, Double)
                let treatment: (MLXArray, Double)
                if pair.isMultiple(of: 2) {
                    control = Self.complete(baseline)
                    treatment = try Self.complete(candidate)
                } else {
                    treatment = try Self.complete(candidate)
                    control = Self.complete(baseline)
                }
                try Self.checkOutput(control.0)
                try Self.checkOutput(treatment.0)
                let (largest, mean) = Self.errors(treatment.0, reference: control.0)
                let order = pair.isMultiple(of: 2) ? "AB" : "BA"
                print("ANE-COST-LAYER pair=\(pair) mode=\(mode) order=\(order) production_ms=\(control.1) split_ms=\(treatment.1) speedup=\(control.1 / treatment.1) max_abs=\(largest) mean_abs=\(mean)")
            }
        }
    }
}
