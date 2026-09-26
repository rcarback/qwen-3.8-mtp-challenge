import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM

@Suite(
    .serialized,
    .enabled(
        if: ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
            && ProcessInfo.processInfo.environment["MLX_ANE_FP16_SUFFIX"] == "1")
)
struct ANEFP16SuffixTests {
    private static let hidden = 512
    private static let inter = 256
    private static let prefix = 128
    private static let rows = 32
    private static let groupSize = 64
    private static let bits = 4

    private struct Projection {
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray
    }

    private struct Fixture {
        let gate: Projection
        let up: Projection
        let down: Projection
    }

    private struct Reference {
        let output: MLXArray
        let gateDType: DType
        let upDType: DType
        let activationDType: DType
        let downDType: DType
    }

    private enum SuffixPrecision {
        case current
        case fp16
        case fp16InputOnly
    }

    private static func quantize(_ dense: MLXArray) -> Projection {
        let (weight, scales, optionalBiases) = quantized(
            dense, groupSize: groupSize, bits: bits)
        let biases = optionalBiases ?? MLXArray.zeros(
            scales.shape, dtype: scales.dtype)
        eval(weight, scales, biases)
        return Projection(weight: weight, scales: scales, biases: biases)
    }

    private static func makeFixture() -> Fixture {
        MLXRandom.seed(8_101)
        let gate = concatenated([
            MLXArray.zeros([prefix, hidden], dtype: .bfloat16),
            (MLXRandom.normal([inter - prefix, hidden]) * 0.1)
                .asType(.bfloat16),
        ], axis: 0)
        let up = concatenated([
            MLXArray.zeros([prefix, hidden], dtype: .bfloat16),
            (MLXRandom.normal([inter - prefix, hidden]) * 0.1)
                .asType(.bfloat16),
        ], axis: 0)
        let down = concatenated([
            MLXArray.zeros([hidden, prefix], dtype: .bfloat16),
            (MLXRandom.normal([hidden, inter - prefix]) * 0.1)
                .asType(.bfloat16),
        ], axis: 1)
        eval(gate, up, down)
        return Fixture(
            gate: quantize(gate), up: quantize(up), down: quantize(down))
    }

    private static func suffixReference(
        _ x: MLXArray,
        fixture: Fixture,
        prefix: Int,
        precision: SuffixPrecision
    ) -> Reference {
        let gateWeight = fixture.gate.weight[prefix ..< inter, 0...]
        let upWeight = fixture.up.weight[prefix ..< inter, 0...]
        let gateScales0 = fixture.gate.scales[prefix ..< inter, 0...]
        let gateBiases0 = fixture.gate.biases[prefix ..< inter, 0...]
        let upScales0 = fixture.up.scales[prefix ..< inter, 0...]
        let upBiases0 = fixture.up.biases[prefix ..< inter, 0...]

        let packRatio = inter / fixture.down.weight.dim(1)
        let groupRatio = inter / fixture.down.scales.dim(1)
        let downWeight = fixture.down.weight[
            0..., (prefix / packRatio) ..< (inter / packRatio)]
        let downScales0 = fixture.down.scales[
            0..., (prefix / groupRatio) ..< (inter / groupRatio)]
        let downBiases0 = fixture.down.biases[
            0..., (prefix / groupRatio) ..< (inter / groupRatio)]

        let input = precision == .current ? x : x.asType(.float16)
        let fp16Metadata = precision == .fp16
        let gateScales = fp16Metadata
            ? gateScales0.asType(.float16) : gateScales0
        let gateBiases = fp16Metadata
            ? gateBiases0.asType(.float16) : gateBiases0
        let upScales = fp16Metadata
            ? upScales0.asType(.float16) : upScales0
        let upBiases = fp16Metadata
            ? upBiases0.asType(.float16) : upBiases0
        let downScales = fp16Metadata
            ? downScales0.asType(.float16) : downScales0
        let downBiases = fp16Metadata
            ? downBiases0.asType(.float16) : downBiases0

        let gate = quantizedMM(
            input, gateWeight, scales: gateScales, biases: gateBiases,
            transpose: true, groupSize: groupSize, bits: bits)
        let up = quantizedMM(
            input, upWeight, scales: upScales, biases: upBiases,
            transpose: true, groupSize: groupSize, bits: bits)
        let activationType: DType = precision == .current
            ? .bfloat16 : .float16
        let activation = (silu(gate) * up).asType(activationType)
        let down = quantizedMM(
            activation, downWeight, scales: downScales, biases: downBiases,
            transpose: true, groupSize: groupSize, bits: bits)
        let output = down.asType(.bfloat16)
        eval(output)
        return Reference(
            output: output,
            gateDType: gate.dtype,
            upDType: up.dtype,
            activationDType: activation.dtype,
            downDType: down.dtype)
    }

    private static func identical(_ lhs: MLXArray, _ rhs: MLXArray) -> Bool {
        MLX.all(MLX.equal(lhs, rhs)).item(Bool.self)
    }

    @Test("fp16 suffix uses fp16 input and metadata across reusable mixed calls")
    func fp16SuffixMatchesIndependentReferenceAcrossReuse() throws {
        try #require(ANERuntime.available())
        try #require(
            !ANESplitConfig.fp16GpuAblate,
            "MLX_ANE_FP16_GPU must be off so the test executes the ANE prefix")

        let fixture = Self.makeFixture()
        #expect(fixture.gate.scales.dtype == .bfloat16)
        #expect(fixture.gate.biases.dtype == .bfloat16)
        #expect(fixture.up.scales.dtype == .bfloat16)
        #expect(fixture.up.biases.dtype == .bfloat16)
        #expect(fixture.down.scales.dtype == .bfloat16)
        #expect(fixture.down.biases.dtype == .bfloat16)
        let gatePrefix = dequantized(
            fixture.gate.weight[0 ..< Self.prefix, 0...],
            scales: fixture.gate.scales[0 ..< Self.prefix, 0...],
            biases: fixture.gate.biases[0 ..< Self.prefix, 0...],
            groupSize: Self.groupSize, bits: Self.bits)
        let upPrefix = dequantized(
            fixture.up.weight[0 ..< Self.prefix, 0...],
            scales: fixture.up.scales[0 ..< Self.prefix, 0...],
            biases: fixture.up.biases[0 ..< Self.prefix, 0...],
            groupSize: Self.groupSize, bits: Self.bits)
        let downPrefix = dequantized(
            fixture.down.weight[0..., 0 ..< (Self.prefix / 8)],
            scales: fixture.down.scales[0..., 0 ..< (Self.prefix / Self.groupSize)],
            biases: fixture.down.biases[0..., 0 ..< (Self.prefix / Self.groupSize)],
            groupSize: Self.groupSize, bits: Self.bits)
        #expect(MLX.abs(gatePrefix).max().item(Float.self) == 0)
        #expect(MLX.abs(upPrefix).max().item(Float.self) == 0)
        #expect(MLX.abs(downPrefix).max().item(Float.self) == 0)
        MLXRandom.seed(8_102)
        let firstInput = (MLXRandom.normal([Self.rows, Self.hidden]) * 1.25)
            .asType(.bfloat16)
        MLXRandom.seed(8_103)
        let secondInput = (MLXRandom.normal([Self.rows, Self.hidden]) * 1.25)
            .asType(.bfloat16)
        eval(firstInput, secondInput)

        let firstFP16 = Self.suffixReference(
            firstInput, fixture: fixture, prefix: Self.prefix, precision: .fp16)
        let secondFP16 = Self.suffixReference(
            secondInput, fixture: fixture, prefix: Self.prefix, precision: .fp16)
        let firstCurrent = Self.suffixReference(
            firstInput, fixture: fixture, prefix: Self.prefix, precision: .current)
        let firstInputOnly = Self.suffixReference(
            firstInput, fixture: fixture, prefix: Self.prefix,
            precision: .fp16InputOnly)

        #expect(firstFP16.gateDType == .float16)
        #expect(firstFP16.upDType == .float16)
        #expect(firstFP16.activationDType == .float16)
        #expect(firstFP16.downDType == .float16)
        try #require(
            !Self.identical(firstFP16.output, firstCurrent.output),
            "fixture must distinguish fp16 suffix math from current bf16 math")
        try #require(
            !Self.identical(firstFP16.output, firstInputOnly.output),
            "fixture must detect bf16 scales or biases retained with fp16 input")

        let split = try ANEFusedSplitMLP(
            gateW: fixture.gate.weight,
            gateScales: fixture.gate.scales,
            gateBiases: fixture.gate.biases,
            upW: fixture.up.weight,
            upScales: fixture.up.scales,
            upBiases: fixture.up.biases,
            downW: fixture.down.weight,
            downScales: fixture.down.scales,
            downBiases: fixture.down.biases,
            hidden: Self.hidden,
            inter: Self.inter,
            sequenceLength: Self.rows,
            aneFraction: 0.5)

        let first = try split(firstInput)
        let second = try split(secondInput)
        eval(first, second)

        #expect(first.dtype == .bfloat16)
        #expect(second.dtype == .bfloat16)
        #expect(first.shape == [Self.rows, Self.hidden])
        #expect(second.shape == [Self.rows, Self.hidden])
        #expect(Self.identical(first, firstFP16.output))
        #expect(Self.identical(second, secondFP16.output))
    }

    @Test("fp16 suffix flag leaves the fraction-zero GPU path unchanged")
    func fp16SuffixLeavesZeroFractionUnchanged() throws {
        let fixture = Self.makeFixture()
        MLXRandom.seed(8_104)
        let input = MLXRandom.normal([Self.rows, Self.hidden])
            .asType(.bfloat16)
        eval(input)
        let reference = Self.suffixReference(
            input, fixture: fixture, prefix: 0, precision: .current)
        let split = try ANEFusedSplitMLP(
            gateW: fixture.gate.weight,
            gateScales: fixture.gate.scales,
            gateBiases: fixture.gate.biases,
            upW: fixture.up.weight,
            upScales: fixture.up.scales,
            upBiases: fixture.up.biases,
            downW: fixture.down.weight,
            downScales: fixture.down.scales,
            downBiases: fixture.down.biases,
            hidden: Self.hidden,
            inter: Self.inter,
            sequenceLength: Self.rows,
            aneFraction: 0)

        let output = try split(input)
        eval(output)
        #expect(output.dtype == .bfloat16)
        #expect(Self.identical(output, reference.output))
    }
}
