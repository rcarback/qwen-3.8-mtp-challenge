import Foundation
import MLXFastCore
import Testing

@testable import MLXFastHarness

struct EmbeddingEndpointTests {
    @Test func serializesResponseAndPreservesOrder() throws {
        for format in ["float", "base64"] {
            let body = Data(
                "{\"model\":\"embed\",\"input\":[\"a\",\"b\"],\"encoding_format\":\"\(format)\"}"
                    .utf8)
            let response = try QwenRuntime.embeddingResponse(body: body, modelName: "embed") {
                input in
                #expect(input == .texts(["a", "b"]))
                return .init(
                    vectors: [
                        Array(repeating: 0.5, count: 2560), Array(repeating: -0.5, count: 2560),
                    ], promptTokens: 4)
            }
            let root = try #require(JSONSerialization.jsonObject(with: response) as? [String: Any])
            let rows = try #require(root["data"] as? [[String: Any]])
            #expect(rows.compactMap { $0["index"] as? Int } == [0, 1])
            if format == "float" {
                #expect((rows[0]["embedding"] as? [Float])?.first == 0.5)
            } else {
                let value = try #require(rows[0]["embedding"] as? String)
                #expect(Data(base64Encoded: value)?.count == 10240)
            }
        }
    }

    @Test func invalidRequestsNeverReachWorker() throws {
        for (json, model, status) in [
            (#"{"model":"embed","input":"a"}"#, nil as String?, 503),
            (#"{"model":"wrong","input":"a"}"#, "embed", 400),
            (#"{"model":"embed","input":[]}"#, "embed", 400),
            (#"{"model":"embed","input":"a","dimensions":10}"#, "embed", 400),
            ("not json", "embed", 400),
        ] {
            do {
                _ = try QwenRuntime.embeddingResponse(body: Data(json.utf8), modelName: model) {
                    _ in
                    Issue.record("Invalid request reached worker")
                    return .init(vectors: [], promptTokens: 0)
                }
                Issue.record("Expected HTTP failure")
            } catch let error as EmbeddingHTTPError {
                #expect(error.status == status)
            }
        }
    }

    @Test func distinguishesInputErrorsFromWorkerFailures() throws {
        let body = Data(#"{"model":"embed","input":"a"}"#.utf8)
        for status in [400, 500] {
            do {
                _ = try QwenRuntime.embeddingResponse(body: body, modelName: "embed") { _ in
                    if status == 400 { throw EmbeddingInputError("bad token") }
                    throw MLXFastError.invalidInput("worker exited")
                }
                Issue.record("Expected HTTP failure")
            } catch let error as EmbeddingHTTPError {
                #expect(error.status == status)
            }
        }
        #expect(throws: EmbeddingHTTPError.self) {
            try QwenRuntime.embeddingResponse(body: body, modelName: "embed") { _ in
                .init(vectors: [[1]], promptTokens: 1)
            }
        }
    }

    @Test func validatesConfigurationAndModelDiscovery() throws {
        func options(path: String?, name: String?) -> QwenRuntime.QwenServeOptions {
            .init(
                targetWeightsPath: "/chat", mtpHeadPath: "/head", depth: 0,
                maxNewTokens: 10, port: 8080, modelName: "chat",
                embeddingWeightsPath: path, embeddingModelName: name)
        }
        #expect(try options(path: nil, name: nil).resolvedEmbeddingModelName() == nil)
        #expect(
            try options(path: "/embedding", name: nil).resolvedEmbeddingModelName()
                == "qwen3-embedding-4b")
        #expect(throws: (any Error).self) {
            try options(path: nil, name: "embed").resolvedEmbeddingModelName()
        }
        #expect(throws: (any Error).self) {
            try options(path: "/embedding", name: "chat").resolvedEmbeddingModelName()
        }
        #expect(throws: (any Error).self) {
            try options(path: "", name: nil).resolvedEmbeddingModelName()
        }
        let list = try QwenRuntime.servedModelNames(
            options: options(path: "/embedding", name: "embed"))
        #expect(list == ["chat", "embed"])
    }

    @Test func rejectsOversizedWireRequestBeforeWorker() throws {
        let input = String(repeating: "x", count: 4 * 1024 * 1024)
        let body = try JSONSerialization.data(withJSONObject: ["model": "embed", "input": input])
        do {
            _ = try QwenRuntime.embeddingResponse(body: body, modelName: "embed") { _ in
                Issue.record("Oversized request reached worker")
                return .init(vectors: [], promptTokens: 0)
            }
            Issue.record("Expected size rejection")
        } catch let error as EmbeddingHTTPError {
            #expect(error.status == 400)
        }
    }
}
