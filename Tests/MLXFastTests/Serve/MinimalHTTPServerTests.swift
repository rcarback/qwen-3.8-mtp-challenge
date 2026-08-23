import Foundation
import Testing

@testable import MLXFastHarness

@Suite("HTTP request parsing")
struct HTTPRequestParsingTests {
    private func buffer(_ text: String) -> Data { Data(text.utf8) }

    @Test("a complete POST parses into method, path, headers, and body")
    func parsesCompleteRequest() throws {
        let raw = "POST /v1/chat/completions HTTP/1.1\r\n"
            + "Host: localhost\r\nContent-Length: 2\r\n\r\n{}"
        let parsed = try #require(HTTPRequest.parse(buffer(raw)))
        #expect(parsed.request.method == "POST")
        #expect(parsed.request.path == "/v1/chat/completions")
        #expect(parsed.request.headers["content-length"] == "2")
        #expect(String(decoding: parsed.request.body, as: UTF8.self) == "{}")
        #expect(parsed.consumed == raw.utf8.count)
    }

    @Test("header names match case-insensitively")
    func lowercasesHeaderNames() throws {
        let raw = "GET /v1/models HTTP/1.1\r\nCONTENT-TYPE: application/json\r\n\r\n"
        let parsed = try #require(HTTPRequest.parse(buffer(raw)))
        #expect(parsed.request.headers["content-type"] == "application/json")
    }

    @Test("an incomplete header block parses to nil")
    func waitsForHeaders() {
        #expect(HTTPRequest.parse(buffer("POST /v1 HTTP/1.1\r\nHost: x")) == nil)
    }

    @Test("a short body parses to nil until the rest arrives")
    func waitsForBody() {
        let raw = "POST /v1 HTTP/1.1\r\nContent-Length: 10\r\n\r\n{}"
        #expect(HTTPRequest.parse(buffer(raw)) == nil)
    }

    @Test("a GET with no Content-Length parses with an empty body")
    func parsesBodylessGet() throws {
        let parsed = try #require(
            HTTPRequest.parse(buffer("GET /v1/models HTTP/1.1\r\n\r\n")))
        #expect(parsed.request.body.isEmpty)
        #expect(parsed.request.method == "GET")
    }

    @Test("a query string is stripped from the path")
    func stripsQuery() throws {
        let parsed = try #require(
            HTTPRequest.parse(buffer("GET /v1/models?limit=1 HTTP/1.1\r\n\r\n")))
        #expect(parsed.request.path == "/v1/models")
    }
}

@Suite("SSE framing")
struct SSEFramingTests {
    @Test("a payload becomes one data frame with a blank-line terminator")
    func framesPayload() {
        let frame = HTTPResponder.sseFrame(Data(#"{"a":1}"#.utf8))
        #expect(String(decoding: frame, as: UTF8.self) == "data: {\"a\":1}\n\n")
    }

    @Test("the terminator frame is the literal [DONE] sentinel")
    func framesTerminator() {
        #expect(
            String(decoding: HTTPResponder.sseTerminator, as: UTF8.self)
                == "data: [DONE]\n\n")
    }
}
