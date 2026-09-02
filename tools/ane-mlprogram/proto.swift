import Foundation

// Minimal protobuf writer. proto3: repeated scalars are packed; maps are
// repeated entry messages with key=1, value=2.
func varint(_ v: UInt64) -> Data {
    var v = v, out = Data()
    repeat { var b = UInt8(v & 0x7F); v >>= 7; if v != 0 { b |= 0x80 }; out.append(b) } while v != 0
    return out
}
func tag(_ f: Int, _ w: UInt8) -> Data { varint(UInt64(f) << 3 | UInt64(w)) }
func lenF(_ f: Int, _ p: Data) -> Data { tag(f, 2) + varint(UInt64(p.count)) + p }
func varF(_ f: Int, _ v: UInt64) -> Data { tag(f, 0) + varint(v) }
func strF(_ f: Int, _ s: String) -> Data { lenF(f, Data(s.utf8)) }
/// protobuf map entry: one submessage per pair, key field 1, value field 2.
func mapEntry(_ field: Int, key: String, value: Data) -> Data {
    lenF(field, strF(1, key) + lenF(2, value))
}

// MARK: - MIL value builders
// DataType: FLOAT16=10 FLOAT32=11 INT8=21 INT4=25 UINT8=31 UINT4=35 STRING=2
enum MILType: UInt64 { case bool = 1, fp16 = 10, fp32 = 11, int8 = 21, int4 = 25, uint8 = 31, uint4 = 35, string = 2 }

/// TensorType{dataType=1, rank=2, dimensions=3}; Dimension{constant=1{size=1}}
func tensorType(_ dt: MILType, _ shape: [Int]) -> Data {
    var d = varF(1, dt.rawValue) + varF(2, UInt64(shape.count))
    for s in shape { d += lenF(3, lenF(1, varF(1, UInt64(s)))) }
    return d
}
/// ValueType{tensorType=1}
func valueType(_ dt: MILType, _ shape: [Int]) -> Data { lenF(1, tensorType(dt, shape)) }

/// Value{type=2, immediateValue=3}; ImmediateValue{tensor=1}; TensorValue{bytes=7}
/// RepeatedBytes{values=1}
func tensorValueBytes(_ dt: MILType, _ shape: [Int], _ payload: Data) -> Data {
    let tv = lenF(7, lenF(1, payload))
    return lenF(2, valueType(dt, shape)) + lenF(3, lenF(1, tv))
}
/// Scalar string Value, used for the mandatory per-op "name" attribute.
/// TensorValue{strings=4}; RepeatedStrings{values=1}
func stringValue(_ s: String) -> Data {
    let tv = lenF(4, strF(1, s))
    return lenF(2, valueType(.string, [])) + lenF(3, lenF(1, tv))
}

// MARK: - MIL ops
// Operation{type=1, inputs=2(map), outputs=3, attributes=5(map)}
// NamedValueType{name=1, type=2}
func namedValue(_ name: String, _ dt: MILType, _ shape: [Int]) -> Data {
    strF(1, name) + lenF(2, valueType(dt, shape))
}
/// const: attributes {name, val}, one output. No inputs.
func constOp(name: String, dt: MILType, shape: [Int], payload: Data) -> Data {
    strF(1, "const")
        + lenF(3, namedValue(name, dt, shape))
        + mapEntry(5, key: "name", value: stringValue(name))
        + mapEntry(5, key: "val", value: tensorValueBytes(dt, shape, payload))
}
/// Argument{arguments=1}; Binding{name=1}
func inputBinding(_ field: Int, param: String, varName: String) -> Data {
    mapEntry(field, key: param, value: lenF(1, strF(1, varName)))
}
/// A generic op whose inputs are all references to existing variables.
func op(_ type: String, name: String, inputs: [(String, String)],
        outName: String, outType: MILType, outShape: [Int]) -> Data {
    var d = strF(1, type)
    for (p, v) in inputs { d += inputBinding(2, param: p, varName: v) }
    d += lenF(3, namedValue(outName, outType, outShape))
    d += mapEntry(5, key: "name", value: stringValue(name))
    return d
}

/// Scalar bool Value. TensorValue{bools=3}; RepeatedBools{values=1} (packed).
func boolValue(_ b: Bool) -> Data {
    let tv = lenF(3, lenF(1, varint(b ? 1 : 0)))
    return lenF(2, valueType(.bool, [])) + lenF(3, lenF(1, tv))
}
/// const op carrying a scalar bool, for op params like matmul's transpose flags.
func constBoolOp(name: String, value: Bool) -> Data {
    strF(1, "const")
        + lenF(3, namedValue(name, .bool, []))
        + mapEntry(5, key: "name", value: stringValue(name))
        + mapEntry(5, key: "val", value: boolValue(value))
}

/// Scalar const of an arbitrary dtype, for quantize/dequantize scales and
/// zero-points.
func constScalarOp(name: String, dt: MILType, payload: Data) -> Data {
    strF(1, "const")
        + lenF(3, namedValue(name, dt, []))
        + mapEntry(5, key: "name", value: stringValue(name))
        + mapEntry(5, key: "val", value: tensorValueBytes(dt, [], payload))
}
/// const carrying a string, for quantize's `output_dtype` parameter.
func constStringOp(name: String, value: String) -> Data {
    strF(1, "const")
        + lenF(3, namedValue(name, .string, []))
        + mapEntry(5, key: "name", value: stringValue(name))
        + mapEntry(5, key: "val", value: stringValue(value))
}
