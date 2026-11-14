import Foundation
import MLXFastCore
import Testing

@testable import MLXFastRuntimeWorkerSupport

struct EmbeddingRuntimeTests {
    @Test func normalizesFloat32Vector() throws {
        let vector = try EmbeddingRuntime.normalize([3, 4])
        #expect(abs(vector[0] - 0.6) < 1e-6)
        #expect(abs(vector[1] - 0.8) < 1e-6)
        for invalid: [Float] in [[0, 0], [.nan], [.infinity]] {
            #expect(throws: (any Error).self) { try EmbeddingRuntime.normalize(invalid) }
        }
    }

    @Test func validatesCompleteBatchBeforeInference() throws {
        let tokens = try EmbeddingRuntime.prepare(
            .texts(["a", "bbb"]), vocabularySize: 10, contextLimit: 8192,
            encode: { Array(repeating: 1, count: $0.count + 1) })
        #expect(tokens == [[1, 1], [1, 1, 1, 1]])
        #expect(tokens.reduce(0) { $0 + $1.count } == 6)
        #expect(throws: EmbeddingInputError.self) {
            try EmbeddingRuntime.prepare(
                .tokens([10]), vocabularySize: 10, contextLimit: 8192, encode: { _ in [] })
        }
        #expect(throws: EmbeddingInputError.self) {
            try EmbeddingRuntime.prepare(
                .text("a"), vocabularySize: 10, contextLimit: 1, encode: { _ in [1, 2] })
        }
    }

    @Test func enforcesRequestLimits() throws {
        try EmbeddingLimits.validateTokens(
            Array(repeating: Array(repeating: 1, count: 8192), count: 4))
        for invalid in [
            Array(repeating: [1], count: 33), [[Int]](), [[Int]()], [[-1]],
            [Array(repeating: 1, count: 8193)],
            Array(repeating: Array(repeating: 1, count: 8192), count: 5),
        ] {
            #expect(throws: EmbeddingInputError.self) {
                try EmbeddingLimits.validateTokens(invalid)
            }
        }
    }

    @Test func rejectsInvalidCheckpointBeforeLoading() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"model_type":"qwen3","hidden_size":4096}"#.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        await #expect(throws: (any Error).self) {
            try await EmbeddingRuntime.load(directory: directory)
        }
    }

    @Test(
        .enabled(if: ProcessInfo.processInfo.environment["MLXFAST_EMBEDDING_TEST_WEIGHTS"] != nil))
    func realModelBatchIsolation() async throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["MLXFAST_EMBEDDING_TEST_WEIGHTS"])
        let runtime = try await EmbeddingRuntime.load(directory: URL(fileURLWithPath: path))
        let single = try await runtime.embed(input: .text("Paris is the capital of France."))
        let batch = try await runtime.embed(
            input: .texts([
                "Paris is the capital of France.",
                "A longer sentence about apples and oranges on a kitchen table.",
            ]))
        #expect(single.vectors[0].count == 2560)
        #expect(zip(single.vectors[0], batch.vectors[0]).allSatisfy { abs($0 - $1) < 1e-5 })
        #expect(abs(single.vectors[0].reduce(Float(0)) { $0 + $1 * $1 } - 1) < 1e-4)
        let query = try await runtime.embed(
            input: .text(
                "Instruct: Retrieve the document that answers the query.\nQuery: What is the capital of France?"
            ))
        let scores = batch.vectors.map { vector in
            zip(query.vectors[0], vector).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        }
        #expect(scores[0] > scores[1])
    }
}
