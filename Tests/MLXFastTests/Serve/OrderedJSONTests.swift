import Foundation
import Testing

@testable import MLXFastHarness

@Suite("OrderedJSON")
struct OrderedJSONTests {
    /// Key order is the whole point: the chat template feeds each tool through
    /// Jinja's `tojson`, which emits insertion order. Reordering would hand the
    /// model a different prompt string for the same request.
    @Test("decoding preserves object key order")
    func preservesKeyOrder() throws {
        let source = #"{"type":"function","function":{"name":"read","description":"d"}}"#
        let value = try OrderedJSON.parse(source)
        #expect(value.serialized() == source)
    }

    @Test("nested arrays and scalars round-trip")
    func roundTripsScalars() throws {
        let source = #"{"a":[1,true,null,"x"],"b":2.5}"#
        let value = try OrderedJSON.parse(source)
        #expect(value.serialized() == source)
    }

    @Test("integral numbers serialize without a decimal point")
    func serializesIntegers() throws {
        let value = try OrderedJSON.parse(#"{"n":3}"#)
        #expect(value.serialized() == #"{"n":3}"#)
    }

    @Test("strings escape the characters JSON requires")
    func escapesStrings() {
        let value = OrderedJSON.string("a\"b\\c\nd")
        #expect(value.serialized() == #""a\"b\\c\nd""#)
    }

    @Test("subscript reaches object members by key")
    func subscriptsObjects() throws {
        let value = try OrderedJSON.parse(#"{"function":{"name":"read"}}"#)
        #expect(value["function"]?["name"]?.stringValue == "read")
        #expect(value["missing"] == nil)
    }
}
