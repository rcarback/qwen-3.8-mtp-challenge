// MIL protobuf builder for ANE offload (moved from the Task 1 PoC test
// target -- see `ANEGemm.swift`, which is the reusable warm primitive built
// on top of this builder).
//
// The free functions below (through `constIntsVecOp`) are copied verbatim
// from `tools/ane-gated-delta/proto.swift` (a hand-authored, self-contained,
// MLX-free MIL protobuf writer) plus `buildSpec` from
// `tools/ane-gated-delta/harness.swift`. Both are reference/investigation
// files outside `editablePaths` and are not modified; this file is the
// editable-surface copy those reference files informed.
import CoreML
import Foundation
import MLX

// MARK: - proto.swift (verbatim copy)

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
enum MILType: UInt64 { case int32 = 23, bool = 1, fp16 = 10, fp32 = 11, int8 = 21, int4 = 25, uint8 = 31, uint4 = 35, string = 2 }

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

/// const int32 vector, for conv's strides/pad/dilations/groups.
/// TensorValue{ints=2}; RepeatedInts{values=1} (packed varints).
func constIntsOp(name: String, values: [Int]) -> Data {
    let payload = values.reduce(Data()) { $0 + varint(UInt64(bitPattern: Int64($1))) }
    let tv = lenF(2, lenF(1, payload))
    let val = lenF(2, valueType(.int32, values.count == 1 ? [] : [values.count]))
        + lenF(3, lenF(1, tv))
    return strF(1, "const")
        + lenF(3, namedValue(name, .int32, values.count == 1 ? [] : [values.count]))
        + mapEntry(5, key: "name", value: stringValue(name))
        + mapEntry(5, key: "val", value: val)
}

/// Like constIntsOp but always a rank-1 tensor, for params (axes) that must be
/// a vector even with one element.
func constIntsVecOp(name: String, values: [Int]) -> Data {
    let payload = values.reduce(Data()) { $0 + varint(UInt64(bitPattern: Int64($1))) }
    let tv = lenF(2, lenF(1, payload))
    let val = lenF(2, valueType(.int32, [values.count])) + lenF(3, lenF(1, tv))
    return strF(1, "const") + lenF(3, namedValue(name, .int32, [values.count]))
        + mapEntry(5, key: "name", value: stringValue(name))
        + mapEntry(5, key: "val", value: val)
}

// MARK: - harness.swift (buildSpec only)

func fd(_ n: String, _ shape: [Int]) -> Data {
    let arr = lenF(1, shape.reduce(Data()) { $0 + varint(UInt64($1)) }) + varF(2, 65552) // fp16
    return strF(1, n) + lenF(3, lenF(5, arr))
}

func buildSpec(inputs: [(String, [Int])], outputs: [(String, [Int])], ops: Data) -> Data {
    var block = Data()
    for o in outputs { block += strF(2, o.0) }
    block += ops
    var fnInputs = Data()
    for i in inputs { fnInputs += lenF(1, namedValue(i.0, .fp16, i.1)) }
    let fn = fnInputs + strF(2, "CoreML8") + mapEntry(3, key: "CoreML8", value: block)
    let program = varF(1, 1) + mapEntry(2, key: "main", value: fn)
    var desc = Data()
    for i in inputs { desc += lenF(1, fd(i.0, i.1)) }
    for o in outputs { desc += lenF(10, fd(o.0, o.1)) }
    return varF(1, 9) + lenF(2, desc) + lenF(502, program)
}

// MARK: - One 1x1 conv == x @ w.T

/// Builds a single-op MIL program: input `a<fp16,[1,K,1,S]>`, a baked-in
/// fp16 weight const `[F,K,1,1]`, one `conv` (1x1, stride 1, valid pad,
/// groups 1) producing `y<fp16,[1,F,1,S]>`. A 1x1 conv over that layout is
/// exactly `x @ w.T` evaluated per sequence position -- see
/// `tools/ane-gated-delta/layer2.swift`'s `convW` for the same op shape used
/// in the gated-delta layer's real projections.
public func buildConvMatmul(K: Int, F: Int, S: Int, weight: Data) -> Data {
    var ops = Data()
    ops += lenF(3, constIntsOp(name: "st", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "dl", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0, 0, 0, 0]))
    ops += lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    ops += lenF(3, constOp(name: "w", dt: .fp16, shape: [F, K, 1, 1], payload: weight))
    ops += lenF(3, op("conv", name: "cv",
                       inputs: [("x", "a"), ("weight", "w"), ("strides", "st"),
                                ("pad_type", "pt"), ("pad", "pd"), ("dilations", "dl"),
                                ("groups", "gp")],
                       outName: "y", outType: .fp16, outShape: [1, F, 1, S]))
    let ins: [(String, [Int])] = [("a", [1, K, 1, S])]
    let outs: [(String, [Int])] = [("y", [1, F, 1, S])]
    return buildSpec(inputs: ins, outputs: outs, ops: ops)
}

// MARK: - MLX <-> MLMultiArray bridges

/// Materializes an [F,K] MLXArray as row-major fp16 bytes (the payload
/// layout the `const` op's TensorValue expects for a `[F,K,1,1]` weight).
public func f16Bytes(_ w: MLXArray) -> Data {
    let w16 = w.asType(.float16)
    eval(w16)
    return w16.asData().data
}

/// [S,K] fp16 MLXArray -> `MLMultiArray` shaped `[1,K,1,S]` (the ANE
/// activation layout: batch, channels, 1, sequence).
public func mlxToMultiArray_1C1S(_ x: MLXArray) throws -> MLMultiArray {
    let S = x.shape[0], K = x.shape[1]
    let xT = x.transposed(1, 0).asType(.float16)
    eval(xT)
    let arr = try MLMultiArray(shape: [1, K, 1, S].map { NSNumber(value: $0) }, dataType: .float16)
    let bytes = xT.asData().data
    arr.withUnsafeMutableBytes { raw, _ in
        _ = bytes.withUnsafeBytes { src in
            memcpy(raw.baseAddress!, src.baseAddress!, Swift.min(raw.count, bytes.count))
        }
    }
    return arr
}

/// `MLMultiArray` shaped `[1,F,1,S]` -> `[S,F]` MLXArray.
public func multiArray_1C1S_toMLX(_ a: MLMultiArray) -> MLXArray {
    let shape = a.shape.map(\.intValue) // [1,F,1,S]
    let F = shape[1], S = shape[3]
    let bytes = a.withUnsafeBytes { raw in Data(bytes: raw.baseAddress!, count: raw.count) }
    let arr = MLXArray(bytes, [F, S], type: Float16.self)
    return arr.transposed(1, 0) // [S,F]
}
