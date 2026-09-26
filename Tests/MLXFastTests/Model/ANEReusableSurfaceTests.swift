import Foundation
import MLX
import MLXRandom
import Testing
@testable import MLXLLM

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"))
struct ANEReusableSurfaceTests {
    @Test("One-column output owns its bytes after surface reuse")
    func singletonOutputSurvivesReuse() throws {
        try #require(ANERuntime.available())
        let width = 512, rows = 32
        let weights = MLXArray.ones([1, width], dtype: .float16) / Float(width)
        eval(weights)
        let model = try ANEInMemoryModel(
            milText: buildConvMILText(inputDim: width, outputDim: 1, sequenceLength: rows),
            weightBlob: buildConvWeightBlob(f16Bytes(weights)))
        try model.compile()
        try model.load()
        defer { model.unload() }
        let first = MLXArray.ones([rows, width], dtype: .float16)
        let a = try ANEDirectDispatch.prepare(model: model, x: first,
            inputDim: width, outputDim: 1, sequenceLength: rows)
        try ANEDirectDispatch.evaluate(a)
        let delayed = ANEDirectDispatch.readZeroCopy(a)
        let b = try ANEDirectDispatch.restage(a,
            x: MLXArray.zeros([rows, width], dtype: .float16), inputDim: width)
        try ANEDirectDispatch.evaluate(b)
        let zero = ANEDirectDispatch.readZeroCopy(b)
        eval(delayed, zero)
        #expect(MLX.abs(delayed - 1).max().item(Float.self) == 0)
        #expect(MLX.abs(zero).max().item(Float.self) == 0)
    }

    @Test("A fused program reuses its request without overwriting delayed outputs", arguments: [false, true])
    func delayedOutputSurvivesReuse(joined: Bool) throws {
        try #require(ANERuntime.available())
        let hidden = 512, inner = 256, rows = 32
        MLXRandom.seed(42)
        let gate = (MLXRandom.normal([inner, hidden]) / Float(hidden).squareRoot()).asType(.float16)
        let up = (MLXRandom.normal([inner, hidden]) / Float(hidden).squareRoot()).asType(.float16)
        let down = (MLXRandom.normal([hidden, inner]) / Float(inner).squareRoot()).asType(.float16)
        let first = MLXRandom.normal([rows, hidden]).asType(.float16)
        let second = MLXArray.zeros([rows, hidden], dtype: .float16)
        eval(gate, up, down, first, second)
        let mlp = try ANEFusedMLP(hidden: hidden, innerFraction: inner,
            sequenceLength: rows, gate: gate, up: up, down: down)
        let a = try mlp.makeInput(first)
        #expect(throws: (any Error).self) { try mlp.makeInput(second) }
        try mlp.predict(a)
        #expect(throws: (any Error).self) { try mlp.makeInput(second) }
        let reference = ANEDirectDispatch.read(a)
        let gpu = MLXArray.ones([rows, hidden], dtype: .bfloat16)
        let expected = joined ? reference.asType(.bfloat16) + gpu : reference
        eval(expected)
        let delayed = joined
            ? ANEDirectDispatch.readJoined(a, gpu: gpu, rows: rows, dtype: .bfloat16)
            : mlp.readOutput(a)
        let b = try mlp.makeInput(second)
        // Keep existential identity outside the macro: Swift 6.3 otherwise
        // crashes while lowering its generated AnyObject comparison thunk.
        let sameRequest = a.request === b.request
        let sameInput = a.inputSurface === b.inputSurface
        let sameOutput = a.outputSurface === b.outputSurface
        #expect(sameRequest, "Warm calls must reuse the request")
        #expect(sameInput)
        #expect(sameOutput)
        #expect(throws: (any Error).self) { try mlp.predict(a) }
        try mlp.predict(b)
        let zero = mlp.readOutput(b)
        eval(delayed, zero)
        #expect(MLX.abs(delayed - expected).max().item(Float.self) == 0)
        #expect(MLX.abs(zero).max().item(Float.self) == 0)
        #expect(throws: (any Error).self) {
            try mlp.makeInput(MLXArray.ones([rows, hidden], dtype: .int32))
        }
        let recovered = try mlp.makeInput(second)
        try mlp.predict(recovered)
        #expect(MLX.abs(mlp.readOutput(recovered)).max().item(Float.self) == 0)
    }
}
