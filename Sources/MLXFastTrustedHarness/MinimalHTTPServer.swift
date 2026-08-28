import Foundation
import MLXFastCore
import Network

/// A loopback HTTP/1.1 server built on `Network.framework`.
///
/// WHY HAND-ROLLED. The package's dependency graph is frozen -- `Package.swift`
/// and `Package.resolved` cannot be edited -- so swift-nio is unavailable.
/// `Network` is a system framework and needs no manifest entry.
///
/// WHY EVERY RESPONSE CLOSES ITS CONNECTION. Keep-alive would need either a
/// `Content-Length` known before the first byte (impossible while streaming) or
/// chunked transfer encoding. Closing lets an SSE body end at EOF, which is
/// valid HTTP/1.1 and which every OpenAI client handles.
///
/// LOCAL DEVELOPER TOOLING. Outside `editablePaths`. Binds 127.0.0.1 and has no
/// authentication; do not expose it.
struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    /// Parse one request out of `buffer`, or return nil if more bytes are needed.
    static func parse(_ buffer: Data) -> (request: HTTPRequest, consumed: Int)? {
        let separator = Data("\r\n\r\n".utf8)
        guard let headerEnd = buffer.range(of: separator) else { return nil }
        let headerText = String(
            decoding: buffer[buffer.startIndex..<headerEnd.lowerBound],
            as: UTF8.self)
        var lines = headerText.components(separatedBy: "\r\n")
        guard !lines.isEmpty else { return nil }
        let requestLine = lines.removeFirst().split(separator: " ")
        guard requestLine.count >= 2 else { return nil }
        let method = String(requestLine[0])
        let target = String(requestLine[1])
        let path = target.split(separator: "?", maxSplits: 1).first.map(String.init)
            ?? target

        var headers: [String: String] = [:]
        for line in lines {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[line.startIndex..<colon]
                .trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...]
                .trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }

        let bodyStart = headerEnd.upperBound
        let expected = headers["content-length"].flatMap(Int.init) ?? 0
        let available = buffer.distance(from: bodyStart, to: buffer.endIndex)
        guard available >= expected else { return nil }
        let bodyEnd = buffer.index(bodyStart, offsetBy: expected)
        return (
            HTTPRequest(
                method: method, path: path, headers: headers,
                body: Data(buffer[bodyStart..<bodyEnd])),
            buffer.distance(from: buffer.startIndex, to: bodyEnd)
        )
    }
}

/// Writes one response on one connection, then closes it.
///
/// `@unchecked Sendable`: every mutation (`headersSent`) and every use of
/// `connection` happens serially on the connection's own dispatch queue, one
/// request per connection, so there is no concurrent access to guard against.
final class HTTPResponder: @unchecked Sendable {
    private let connection: NWConnection
    private var headersSent = false

    /// Whether a status line has already gone out. The streaming error path
    /// needs this: before the headers a real 500 still reaches the client,
    /// after them the only honest signal is an abrupt end of stream.
    var didSendHeaders: Bool { headersSent }

    init(connection: NWConnection) {
        self.connection = connection
    }

    static func sseFrame(_ payload: Data) -> Data {
        var frame = Data("data: ".utf8)
        frame.append(payload)
        frame.append(Data("\n\n".utf8))
        return frame
    }

    static let sseTerminator = Data("data: [DONE]\n\n".utf8)

    func sendJSON(status: Int, body: Data) {
        // A second status line written into an open SSE body is not a
        // response, it is corruption: the client already parsed the 200 and
        // reads these bytes as event data. End the stream instead.
        guard !headersSent else {
            endSSE()
            return
        }
        headersSent = true
        var out = Data(headLine(status: status, extra: [
            "Content-Type": "application/json",
            "Content-Length": "\(body.count)",
        ]).utf8)
        out.append(body)
        send(out, closing: true)
    }

    func sendError(status: Int, message: String) {
        // The OpenAI error envelope: clients surface `error.message` verbatim.
        let payload = OrderedJSON.object([
            ("error", OrderedJSON.object([
                ("message", .string(message)),
                ("type", .string("invalid_request_error")),
            ])),
        ])
        sendJSON(status: status, body: Data(payload.serialized().utf8))
    }

    func beginSSE() {
        guard !headersSent else { return }
        headersSent = true
        send(Data(headLine(status: 200, extra: [
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
        ]).utf8), closing: false)
    }

    func sendSSE(_ payload: Data) {
        // Idempotent, and mandatory: a frame written before the status line
        // makes "data: {...}" the first bytes on the wire, which every HTTP
        // client rejects as a malformed response rather than as a server
        // error. That masked a real worker fault as "invalid HTTP version
        // parsed" and sent the client into a retry loop.
        beginSSE()
        send(Self.sseFrame(payload), closing: false)
    }

    func endSSE() {
        beginSSE()
        send(Self.sseTerminator, closing: true)
    }

    private func headLine(status: Int, extra: [String: String]) -> String {
        var head = "HTTP/1.1 \(status) \(Self.reason(status))\r\n"
        for (name, value) in extra.sorted(by: { $0.key < $1.key }) {
            head += "\(name): \(value)\r\n"
        }
        head += "Connection: close\r\n\r\n"
        return head
    }

    private func send(_ data: Data, closing: Bool) {
        connection.send(content: data, completion: .contentProcessed { _ in
            if closing { self.connection.cancel() }
        })
    }

    private static func reason(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 500: return "Internal Server Error"
        default: return "Unknown"
        }
    }
}

final class MinimalHTTPServer {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "mlxfast.serve.accept")

    init(port: UInt16) throws {
        let parameters = NWParameters.tcp
        // Loopback only. This server has no authentication by design.
        parameters.requiredLocalEndpoint = .hostPort(
            host: .ipv4(.loopback), port: NWEndpoint.Port(rawValue: port)!)
        parameters.allowLocalEndpointReuse = true
        listener = try NWListener(using: parameters)
    }

    func start(handler: @escaping @Sendable (HTTPRequest, HTTPResponder) -> Void) throws {
        listener.newConnectionHandler = { [queue] connection in
            connection.start(queue: queue)
            Self.readRequest(on: connection, buffer: Data(), handler: handler)
        }
        listener.start(queue: queue)
    }

    private static func readRequest(
        on connection: NWConnection,
        buffer: Data,
        handler: @escaping @Sendable (HTTPRequest, HTTPResponder) -> Void
    ) {
        connection.receive(
            minimumIncompleteLength: 1, maximumLength: 1 << 20
        ) { chunk, _, isComplete, error in
            var buffer = buffer
            if let chunk { buffer.append(chunk) }
            if let parsed = HTTPRequest.parse(buffer) {
                handler(parsed.request, HTTPResponder(connection: connection))
                return
            }
            if error != nil || isComplete {
                connection.cancel()
                return
            }
            readRequest(on: connection, buffer: buffer, handler: handler)
        }
    }

    func waitForever() {
        dispatchMain()
    }
}
