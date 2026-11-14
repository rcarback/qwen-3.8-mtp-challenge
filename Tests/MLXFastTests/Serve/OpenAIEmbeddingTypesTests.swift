import Foundation
import MLXFastCore
import Testing

@testable import MLXFastHarness

struct OpenAIEmbeddingTypesTests {
    @Test func acceptsInputShapes() throws {
        for input in [#""hello""#, #"["hello","world"]"#, "[1,2]", "[[1],[2,3]]"] {
            let body = Data("{\"model\":\"embed\",\"input\":\(input)}".utf8)
            let request = try JSONDecoder().decode(EmbeddingRequest.self, from: body)
            try request.validate(modelName: "embed")
            let roundTrip = try JSONDecoder().decode(
                EmbeddingWorkerInput.self, from: JSONEncoder().encode(request.input))
            #expect(roundTrip == request.input)
        }
    }

    @Test(arguments: [
        "[]", #""""#, #"["ok",""]"#, "[[]]", "[true]", "[1.5]", "[-1]", #"[1,"x"]"#, "null",
    ])
    func rejectsInvalidInputs(_ input: String) {
        #expect(throws: (any Error).self) {
            let body = Data("{\"model\":\"embed\",\"input\":\(input)}".utf8)
            let request = try JSONDecoder().decode(EmbeddingRequest.self, from: body)
            try request.validate(modelName: "embed")
        }
    }

    @Test(arguments: [
        #"{"input":"ok"}"#,
        #"{"model":"wrong","input":"ok"}"#,
        #"{"model":"embed","input":"ok","dimensions":3}"#,
        #"{"model":"embed","input":"ok","encoding_format":"binary"}"#,
    ])
    func rejectsUnsupportedRequests(_ json: String) {
        #expect(throws: (any Error).self) {
            let request = try JSONDecoder().decode(EmbeddingRequest.self, from: Data(json.utf8))
            try request.validate(modelName: "embed")
        }
    }

    @Test func base64UsesLittleEndianFloat32() throws {
        let value = try encodeEmbedding([1, -2], format: .base64)
        guard case .base64(let encoded) = value else {
            Issue.record("Expected base64")
            return
        }
        #expect(Data(base64Encoded: encoded) == Data([0, 0, 128, 63, 0, 0, 0, 192]))
        #expect(throws: (any Error).self) { try encodeEmbedding([.nan], format: .float) }
    }

    @Test func responseHasIndicesAndTokenUsage() throws {
        let response = try EmbeddingResponse(
            result: .init(vectors: [[1, 0], [0, 1]], promptTokens: 5),
            model: "embed", format: .float)
        let json = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(response)) as? [String: Any])
        #expect(json["object"] as? String == "list")
        let rows = try #require(json["data"] as? [[String: Any]])
        #expect(rows.compactMap { $0["index"] as? Int } == [0, 1])
        let usage = try #require(json["usage"] as? [String: Int])
        #expect(usage == ["prompt_tokens": 5, "total_tokens": 5])
    }
}
