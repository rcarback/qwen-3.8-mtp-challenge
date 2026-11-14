import Foundation
import MLXFastCore
import Testing

@testable import MLXFastHarness
@testable import MLXFastRuntimeWorkerSupport

struct EmbeddingWorkerProtocolTests {
    @Test func wireRoundTripAcrossModules() throws {
        let request = MLXFastHarness.RuntimeWorkerRequest(
            id: 1, kind: "embedding_encode", embeddingInput: .tokens([1, 2]))
        let decoded = try JSONDecoder().decode(
            MLXFastRuntimeWorkerSupport.RuntimeWorkerRequest.self,
            from: JSONEncoder().encode(request))
        #expect(try decoded.validatedEmbeddingInput() == .tokens([1, 2]))
        let result = EmbeddingBatchResult(
            vectors: [Array(repeating: 0, count: 2560)], promptTokens: 2)
        let response = MLXFastRuntimeWorkerSupport.RuntimeWorkerResponse(
            id: 1, nonce: "test", ok: true, embeddingResult: result)
        let parent = try JSONDecoder().decode(
            MLXFastHarness.RuntimeWorkerResponse.self, from: JSONEncoder().encode(response))
        #expect(parent.embeddingResult?.promptTokens == 2)
        let old = try JSONDecoder().decode(
            MLXFastHarness.RuntimeWorkerResponse.self,
            from: Data(#"{"id":0,"ok":true,"nonce":"test"}"#.utf8))
        #expect(old.embeddingResult == nil)
    }

    @Test func rejectsGenerationFields() throws {
        let request = MLXFastRuntimeWorkerSupport.RuntimeWorkerRequest(
            id: 1, kind: "embedding_encode", token: 7, embeddingInput: .text("hello"))
        #expect(throws: EmbeddingInputError.self) { try request.validatedEmbeddingInput() }
    }

    @Test func validatesWorkerResult() throws {
        try EmbeddingBatchResult(vectors: [Array(repeating: 1, count: 2560)], promptTokens: 2)
            .validate(inputCount: 1)
        for result in [
            EmbeddingBatchResult(vectors: [], promptTokens: 1),
            EmbeddingBatchResult(vectors: [[1]], promptTokens: 1),
            EmbeddingBatchResult(vectors: [Array(repeating: .nan, count: 2560)], promptTokens: 1),
            EmbeddingBatchResult(vectors: [Array(repeating: 1, count: 2560)], promptTokens: 0),
        ] {
            #expect(throws: (any Error).self) { try result.validate(inputCount: 1) }
        }
    }

    @Test func clientSurvivesInputFailureAndPassesEmbeddingPath() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let worker = directory.appendingPathComponent("worker")
        let response = MLXFastRuntimeWorkerSupport.RuntimeWorkerResponse(
            id: 2, nonce: "embedding-test", ok: true,
            embeddingResult: .init(vectors: [Array(repeating: 0.5, count: 2560)], promptTokens: 2))
        let json = String(decoding: try JSONEncoder().encode(response), as: UTF8.self)
        let script = """
            #!/bin/sh
            case "$*" in *'--embedding-weights /embedding'*) ;; *) exit 12 ;; esac
            printf '%s\\n' '{"id":0,"ok":true,"nonce":"embedding-test"}'
            read -r first
            printf '%s\\n' '{"id":1,"ok":false,"nonce":"embedding-test","error":"bad token","embedding_input_error":true}'
            read -r second
            printf '%s\\n' '\(json)'
            read -r closing
            """
        try Data(script.utf8).write(to: worker)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: worker.path)
        let client = try MLXFastHarness.RuntimeWorkerClient(
            options: MLXFastHarness.RuntimeWorkerOptions(
                executablePath: worker.path, requestTimeoutSeconds: 5),
            weightsPath: "/chat", mtpHeadPath: "/head", embeddingWeightsPath: "/embedding")
        defer { client.close() }
        #expect(throws: EmbeddingInputError.self) { try client.embed(input: .tokens([999999])) }
        let result = try client.embed(input: .text("hello"))
        #expect(result.promptTokens == 2)
        #expect(result.vectors[0].count == 2560)
    }
}
