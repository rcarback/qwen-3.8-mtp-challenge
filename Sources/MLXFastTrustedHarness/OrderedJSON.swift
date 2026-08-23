import Foundation
import MLXFastCore

/// A JSON value that remembers the order its object keys arrived in.
///
/// WHY NOT `JSONSerialization` OR A `[String: Any]`. The checkpoint's chat
/// template renders each tool with Jinja's `tojson`, which emits keys in the
/// order the incoming JSON carried them. Swift dictionaries are unordered, so
/// decoding a tool into one and re-encoding it would produce a different prompt
/// string on every process launch for a byte-identical request. That is a real
/// behavioural difference, not cosmetics: the prompt is the model's input.
///
/// LOCAL DEVELOPER TOOLING. This file lives outside `editablePaths`.
enum OrderedJSON: Equatable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([OrderedJSON])
    indirect case object([Pair])

    struct Pair: Equatable {
        let key: String
        let value: OrderedJSON
    }

    static func object(_ pairs: [(String, OrderedJSON)]) -> OrderedJSON {
        .object(pairs.map { Pair(key: $0.0, value: $0.1) })
    }

    subscript(_ key: String) -> OrderedJSON? {
        guard case .object(let pairs) = self else { return nil }
        return pairs.first { $0.key == key }?.value
    }

    var stringValue: String? {
        guard case .string(let text) = self else { return nil }
        return text
    }

    func serialized() -> String {
        switch self {
        case .null:
            return "null"
        case .bool(let flag):
            return flag ? "true" : "false"
        case .number(let value):
            // Integral values print as integers. A tool schema that says
            // `"maxItems": 5` should not reach the model as `5.0`.
            if value.rounded() == value, value.magnitude < 1e15 {
                return String(Int64(value))
            }
            return String(value)
        case .string(let text):
            return Self.quote(text)
        case .array(let items):
            return "[" + items.map { $0.serialized() }.joined(separator: ",") + "]"
        case .object(let pairs):
            let body = pairs
                .map { Self.quote($0.key) + ":" + $0.value.serialized() }
                .joined(separator: ",")
            return "{" + body + "}"
        }
    }

    private static func quote(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }
}

extension OrderedJSON: Codable {
    init(from decoder: Decoder) throws {
        // Object key order is not recoverable through `KeyedDecodingContainer`,
        // which hands back an unordered `allKeys`. The only way to keep it is to
        // read the raw bytes, so callers that need order use
        // `OrderedJSON.parse(_:)` on the raw request body instead of routing an
        // object through `JSONDecoder`. This conformance still exists so
        // `OrderedJSON` can be a `Decodable` field inside another struct (for
        // example a request's already-scalar or already-array fields).
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let flag = try? container.decode(Bool.self) {
            self = .bool(flag)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let text = try? container.decode(String.self) {
            self = .string(text)
        } else if let items = try? container.decode([OrderedJSON].self) {
            self = .array(items)
        } else if let raw = try? container.decode(RawObject.self) {
            self = .object(raw.pairs)
        } else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "not JSON"))
        }
    }

    func encode(to encoder: Encoder) throws {
        // Encoded through structural containers (not a re-quoted string) so a
        // value nested inside another `Encodable` type serializes as JSON, not
        // as a JSON string holding JSON text. `serialized()` / `parse(_:)`
        // remain the order-preserving round trip this type exists for; this
        // conformance only has to produce correct, structurally valid JSON.
        switch self {
        case .null:
            var container = encoder.singleValueContainer()
            try container.encodeNil()
        case .bool(let flag):
            var container = encoder.singleValueContainer()
            try container.encode(flag)
        case .number(let value):
            var container = encoder.singleValueContainer()
            try container.encode(value)
        case .string(let text):
            var container = encoder.singleValueContainer()
            try container.encode(text)
        case .array(let items):
            var container = encoder.unkeyedContainer()
            for item in items {
                try container.encode(item)
            }
        case .object(let pairs):
            var container = encoder.container(keyedBy: RawObject.AnyKey.self)
            for pair in pairs {
                guard let key = RawObject.AnyKey(stringValue: pair.key) else { continue }
                try container.encode(pair.value, forKey: key)
            }
        }
    }
}

/// Decodes a JSON object while keeping the container shape a `Decodable`
/// struct expects; key order is not recoverable this way (see the comment on
/// `OrderedJSON.init(from:)`), so the ordered parse is done by
/// `OrderedJSON.parse(_:)` below, which the server uses on the raw request
/// body.
private struct RawObject: Decodable {
    let pairs: [OrderedJSON.Pair]

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: AnyKey.self)
        var collected: [OrderedJSON.Pair] = []
        for key in container.allKeys {
            let value = try container.decode(OrderedJSON.self, forKey: key)
            collected.append(.init(key: key.stringValue, value: value))
        }
        pairs = collected
    }

    struct AnyKey: CodingKey {
        let stringValue: String
        let intValue: Int? = nil
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }
}

extension OrderedJSON {
    /// Parse JSON text while preserving object key order.
    ///
    /// `JSONDecoder` cannot do this (see `init(from:)` above), so the server
    /// parses tool definitions and any other order-sensitive JSON with this
    /// hand-rolled scanner instead.
    static func parse(_ text: String) throws -> OrderedJSON {
        var scanner = Scanner(text: Array(text.unicodeScalars))
        let value = try scanner.parseValue()
        scanner.skipWhitespace()
        guard scanner.isAtEnd else {
            throw MLXFastError.invalidInput("trailing bytes after JSON value")
        }
        return value
    }

    struct Scanner {
        let text: [Unicode.Scalar]
        var index = 0

        init(text: [Unicode.Scalar]) { self.text = text }

        var isAtEnd: Bool { index >= text.count }

        mutating func skipWhitespace() {
            while index < text.count,
                  text[index] == " " || text[index] == "\n"
                      || text[index] == "\r" || text[index] == "\t" {
                index += 1
            }
        }

        mutating func parseValue() throws -> OrderedJSON {
            skipWhitespace()
            guard index < text.count else {
                throw MLXFastError.invalidInput("JSON ended early")
            }
            switch text[index] {
            case "{": return try parseObject()
            case "[": return try parseArray()
            case "\"": return .string(try parseString())
            case "t":
                try expect("true"); return .bool(true)
            case "f":
                try expect("false"); return .bool(false)
            case "n":
                try expect("null"); return .null
            default: return .number(try parseNumber())
            }
        }

        mutating func parseObject() throws -> OrderedJSON {
            index += 1  // '{'
            var pairs: [Pair] = []
            skipWhitespace()
            if index < text.count, text[index] == "}" { index += 1; return .object(pairs) }
            while true {
                skipWhitespace()
                let key = try parseString()
                skipWhitespace()
                guard index < text.count, text[index] == ":" else {
                    throw MLXFastError.invalidInput("expected ':' in JSON object")
                }
                index += 1
                pairs.append(.init(key: key, value: try parseValue()))
                skipWhitespace()
                guard index < text.count else {
                    throw MLXFastError.invalidInput("unterminated JSON object")
                }
                if text[index] == "," { index += 1; continue }
                if text[index] == "}" { index += 1; return .object(pairs) }
                throw MLXFastError.invalidInput("expected ',' or '}' in JSON object")
            }
        }

        mutating func parseArray() throws -> OrderedJSON {
            index += 1  // '['
            var items: [OrderedJSON] = []
            skipWhitespace()
            if index < text.count, text[index] == "]" { index += 1; return .array(items) }
            while true {
                items.append(try parseValue())
                skipWhitespace()
                guard index < text.count else {
                    throw MLXFastError.invalidInput("unterminated JSON array")
                }
                if text[index] == "," { index += 1; continue }
                if text[index] == "]" { index += 1; return .array(items) }
                throw MLXFastError.invalidInput("expected ',' or ']' in JSON array")
            }
        }

        mutating func parseString() throws -> String {
            guard index < text.count, text[index] == "\"" else {
                throw MLXFastError.invalidInput("expected a JSON string")
            }
            index += 1
            var out = String.UnicodeScalarView()
            while index < text.count {
                let scalar = text[index]
                index += 1
                if scalar == "\"" { return String(out) }
                if scalar != "\\" { out.append(scalar); continue }
                guard index < text.count else { break }
                let escape = text[index]
                index += 1
                switch escape {
                case "n": out.append("\n")
                case "r": out.append("\r")
                case "t": out.append("\t")
                case "b": out.append(Unicode.Scalar(8)!)
                case "f": out.append(Unicode.Scalar(12)!)
                case "u":
                    guard index + 4 <= text.count else {
                        throw MLXFastError.invalidInput("truncated \\u escape")
                    }
                    let digits = String(String.UnicodeScalarView(text[index..<(index + 4)]))
                    index += 4
                    guard let code = UInt32(digits, radix: 16) else {
                        throw MLXFastError.invalidInput("bad \\u escape")
                    }
                    // Surrogate pairs: a high surrogate must be followed by its
                    // low half, which arrives as a second \u escape.
                    if code >= 0xD800, code <= 0xDBFF,
                       index + 6 <= text.count, text[index] == "\\",
                       text[index + 1] == "u",
                       let low = UInt32(
                           String(String.UnicodeScalarView(
                               text[(index + 2)..<(index + 6)])), radix: 16),
                       low >= 0xDC00, low <= 0xDFFF {
                        index += 6
                        let combined = 0x10000
                            + ((code - 0xD800) << 10) + (low - 0xDC00)
                        out.append(Unicode.Scalar(combined) ?? "\u{FFFD}")
                    } else {
                        out.append(Unicode.Scalar(code) ?? "\u{FFFD}")
                    }
                default: out.append(escape)
                }
            }
            throw MLXFastError.invalidInput("unterminated JSON string")
        }

        mutating func parseNumber() throws -> Double {
            let start = index
            while index < text.count,
                  "0123456789+-.eE".unicodeScalars.contains(text[index]) {
                index += 1
            }
            let literal = String(String.UnicodeScalarView(text[start..<index]))
            guard let value = Double(literal) else {
                throw MLXFastError.invalidInput("bad JSON number '\(literal)'")
            }
            return value
        }

        mutating func expect(_ word: String) throws {
            for scalar in word.unicodeScalars {
                guard index < text.count, text[index] == scalar else {
                    throw MLXFastError.invalidInput("expected '\(word)'")
                }
                index += 1
            }
        }
    }
}
