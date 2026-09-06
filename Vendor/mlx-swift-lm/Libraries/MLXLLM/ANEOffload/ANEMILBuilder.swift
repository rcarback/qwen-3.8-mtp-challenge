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

/// Builds just the `program` submessage (`Program{version=1, functions=2(map)}`)
/// for the given ops -- the same function block `buildSpec` wraps inside a
/// full CoreML `Model` proto. Shared by `buildSpec` (Model-envelope path,
/// used by `ANEGemm` via `MLModelAsset(specification:)`) and
/// `buildConvMILProgram` (bare-program path, used by `ANEInMemoryModel` via
/// `initWithNetworkText:weights:optionsPlist:isMILModel:`).
func buildProgram(inputs: [(String, [Int])], outputs: [(String, [Int])], ops: Data) -> Data {
    var block = Data()
    for o in outputs { block += strF(2, o.0) }
    block += ops
    var fnInputs = Data()
    for i in inputs { fnInputs += lenF(1, namedValue(i.0, .fp16, i.1)) }
    let fn = fnInputs + strF(2, "CoreML8") + mapEntry(3, key: "CoreML8", value: block)
    return varF(1, 1) + mapEntry(2, key: "main", value: fn)
}

func buildSpec(inputs: [(String, [Int])], outputs: [(String, [Int])], ops: Data) -> Data {
    let program = buildProgram(inputs: inputs, outputs: outputs, ops: ops)
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
private func convOps(K: Int, F: Int, S: Int, weight: Data) -> Data {
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
    return ops
}

private func convIO(K: Int, F: Int, S: Int) -> (ins: [(String, [Int])], outs: [(String, [Int])]) {
    ([("a", [1, K, 1, S])], [("y", [1, F, 1, S])])
}

public func buildConvMatmul(K: Int, F: Int, S: Int, weight: Data) -> Data {
    let ops = convOps(K: K, F: F, S: S, weight: weight)
    let io = convIO(K: K, F: F, S: S)
    return buildSpec(inputs: io.ins, outputs: io.outs, ops: ops)
}

/// One conv procedure of a MULTIFUNCTION Core ML `Model` proto: an `[F,K]`
/// 1x1 conv at a fixed `S`, its fp16 weight bytes baked in as a const. The
/// bare-MIL-text procedure bank (`buildBankMILText`) compiles and loads N
/// functions but only `main` is dispatchable, because the in-memory
/// descriptor carries no model description and so registers one procedure.
/// `buildMultiFunctionConvSpec` supplies that missing description.
public struct ANEMultiFunctionProc {
    public let name: String
    public let inputDim: Int
    public let outputDim: Int
    public let sequenceLength: Int
    public let weight: Data  // fp16, row-major [out, in]
    public init(name: String, inputDim: Int, outputDim: Int, sequenceLength: Int, weight: Data) {
        self.name = name
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.sequenceLength = sequenceLength
        self.weight = weight
    }
}

/// A MULTIFUNCTION Core ML `Model` proto: `procs.count` independent 1x1-conv
/// functions, each declared BOTH in the MIL program's `functions` map AND in
/// the model description's `functions` list (`ModelDescription.functions`,
/// field 20; iOS18/macOS15 multifunction models). The paired declaration is
/// what makes every function a runnable procedure -- the missing half of the
/// bare-text bank. Load through `MLModelAsset(specification:)` +
/// `MLModel.load(asset:configuration:)` with `MLModelConfiguration.functionName`
/// set to a proc name, then `prediction` runs that function on the ANE. Every
/// function takes an input feature `a` and produces an output feature `y`
/// (each function is its own MIL scope, so the names do not collide). The
/// first proc's name is also the `defaultFunctionName` (field 21).
public func buildMultiFunctionConvSpec(procs: [ANEMultiFunctionProc], includeTopLevelIO: Bool = false, opset: String = "CoreML9") -> Data {
    precondition(!procs.isEmpty, "buildMultiFunctionConvSpec needs at least one proc")

    // Program submessage: Program{version=1, functions=2(map name->Function)}.
    // `opset` gates which MIL ops are legal and which model features are
    // enabled: multifunction is iOS18 = `CoreML9` (the single-conv proto path
    // hardcodes the iOS17 `CoreML8`, which predates multifunction and makes
    // Core ML reject the description).
    var programFns = Data()
    for p in procs {
        let ops = convOps(K: p.inputDim, F: p.outputDim, S: p.sequenceLength, weight: p.weight)
        var block = Data()
        block += strF(2, "y")  // Block.outputs
        block += ops
        let fnInputs = lenF(1, namedValue("a", .fp16, [1, p.inputDim, 1, p.sequenceLength]))
        let fn = fnInputs + strF(2, opset) + mapEntry(3, key: opset, value: block)
        programFns += mapEntry(2, key: p.name, value: fn)
    }
    let program = varF(1, 1) + programFns

    // ModelDescription: one FunctionDescription per proc (field 20) plus the
    // default function name (field 21). FunctionDescription{name=1, input=2,
    // output=3} carrying FeatureDescriptions.
    var desc = Data()
    // Optional top-level default-function I/O (ModelDescription.input=1,
    // output=10), mirroring the first proc. Some Core ML validators expect
    // the default function's signature at the top level even for a
    // multifunction model; `includeTopLevelIO` toggles it for the probe.
    if includeTopLevelIO {
        desc += lenF(1, fd("a", [1, procs[0].inputDim, 1, procs[0].sequenceLength]))
        desc += lenF(10, fd("y", [1, procs[0].outputDim, 1, procs[0].sequenceLength]))
    }
    for p in procs {
        var fnd = strF(1, p.name)
        fnd += lenF(2, fd("a", [1, p.inputDim, 1, p.sequenceLength]))
        fnd += lenF(3, fd("y", [1, p.outputDim, 1, p.sequenceLength]))
        desc += lenF(20, fnd)
    }
    desc += strF(21, procs[0].name)

    // Model{specificationVersion=1, description=2, mlProgram=502}.
    return varF(1, 9) + lenF(2, desc) + lenF(502, program)
}

/// Same single-op conv program `buildConvMatmul` builds, but returns just the
/// `program` submessage bytes -- what `_ANEInMemoryModelDescriptor`'s
/// `initWithNetworkText:weights:optionsPlist:isMILModel:` wants for an
/// in-memory MIL model, rather than the full CoreML `Model` proto envelope
/// `buildConvMatmul` wraps it in for `MLModelAsset(specification:)`.
public func buildConvMILProgram(K: Int, F: Int, S: Int, weight: Data) -> Data {
    let ops = convOps(K: K, F: F, S: S, weight: weight)
    let io = convIO(K: K, F: F, S: S)
    return buildProgram(inputs: io.ins, outputs: io.outs, ops: ops)
}

// MARK: - MIL text (oMLX `fp16_linear_mil` format)

/// Same single-op 1x1 conv `buildConvMILProgram` builds as a binary MIL
/// protobuf, but emitted as MIL TEXT instead -- the format
/// `_ANEInMemoryModelDescriptor`'s `initWithNetworkText:weights:optionsPlist:isMILModel:`
/// actually expects. Task 2's binary-protobuf program failed compile with
/// `InvalidCompilationParam`; the open-source oMLX project's
/// `fp16_linear_mil` (`omlx/custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm`)
/// proves this text format compiles unentitled. Verbatim shape, with `w`'s
/// weight bytes referenced via `BLOBFILE` at `weights/weight_data.bin`
/// offset 64 (see `buildConvWeightBlob`).
/// `programTag` is stamped into the program's `buildInfo` so that two
/// programs with identical shapes but different weights get DIFFERENT
/// descriptor identities. `_ANEInMemoryModel` derives `hexStringIdentifier`
/// (its staging directory and compile-cache key) from the network text
/// alone -- the weight blob is staged as a side file and is not hashed.
/// Without the tag, all 64 Qwen layers collided on one identity: they
/// overwrote each other's staged `weight.bin`, and only the first loaded
/// program computed with its own weights (the 2026-09-02 hybrid collapse).
public func buildConvMILText(
    inputDim: Int, outputDim: Int, sequenceLength: Int,
    programTag: String = UUID().uuidString
) -> String {
    """
    program(1.3)
    [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}, {"mlxfast-program-tag", "\(programTag)"}})]
    {
      func main<ios18>(tensor<fp16, [1, \(inputDim), 1, \(sequenceLength)]> x) {
        tensor<fp16, [\(outputDim), \(inputDim), 1, 1]> w = const()[name=string("w"), val=tensor<fp16, [\(outputDim), \(inputDim), 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64(64)))];
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
        tensor<fp16, [1, \(outputDim), 1, \(sequenceLength)]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("conv")];
      } -> (y);
    }
    """
}

// MARK: - MIL text with COMPRESSED weights (int8 / int4), dequantized on-ANE

/// A 1x1-conv program whose weight is stored INT8 and dequantized to fp16 on
/// the ANE via `constexpr_affine_dequantize` (per-output-channel symmetric
/// scale, zero-point 0). Confirms the ANE consumes int8 weight storage rather
/// than requiring dense fp16. `scales` is one fp16 value per output channel
/// (length `outputDim`); the int8 weight bytes are the BLOBFILE payload.
public func buildConvMILTextInt8(
    inputDim: Int, outputDim: Int, sequenceLength: Int, scales: [Float],
    programTag: String = UUID().uuidString
) -> String {
    precondition(scales.count == outputDim, "int8 per-channel scales must be [outputDim]")
    let scaleList = scales.map { String(format: "%.8g", $0) }.joined(separator: ", ")
    return """
    program(1.3)
    [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}, {"mlxfast-program-tag", "\(programTag)"}})]
    {
      func main<ios18>(tensor<fp16, [1, \(inputDim), 1, \(sequenceLength)]> x) {
        tensor<fp16, [\(outputDim), \(inputDim), 1, 1]> w = constexpr_affine_dequantize()[axis = int32(0), name = string("wdeq"), quantized_data = tensor<int8, [\(outputDim), \(inputDim), 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight_data.bin"), offset = uint64(64))), scale = tensor<fp16, [\(outputDim)]>([\(scaleList)]), zero_point = int8(0)];
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
        tensor<fp16, [1, \(outputDim), 1, \(sequenceLength)]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("conv")];
      } -> (y);
    }
    """
}

/// A 1x1-conv program whose weight is stored INT4 and dequantized to fp16 on
/// the ANE via `constexpr_blockwise_shift_scale` (iOS18), the op that matches
/// MLX-style affine group quantization: a per-block scale along the input
/// dimension, block size `groupSize`, symmetric (offset 0). `scales` is
/// row-major `[outputDim, inputDim / groupSize]` fp16; the int4 weight is the
/// BLOBFILE payload, two signed nibbles per byte (low nibble first).
public func buildConvMILTextInt4Blockwise(
    inputDim: Int, outputDim: Int, sequenceLength: Int, groupSize: Int,
    dataOffset: UInt64, scaleOffset: UInt64,
    programTag: String = UUID().uuidString
) -> String {
    precondition(inputDim % groupSize == 0, "inputDim must be a multiple of groupSize")
    let groups = inputDim / groupSize
    return """
    program(1.3)
    [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}, {"mlxfast-program-tag", "\(programTag)"}})]
    {
      func main<ios18>(tensor<fp16, [1, \(inputDim), 1, \(sequenceLength)]> x) {
        tensor<fp16, [\(outputDim), \(inputDim), 1, 1]> w = constexpr_blockwise_shift_scale(data = tensor<int4, [\(outputDim), \(inputDim), 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(\(dataOffset)))), scale = tensor<fp16, [\(outputDim), \(groups), 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(\(scaleOffset)))))[name = string("wdeq")];
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
        tensor<fp16, [1, \(outputDim), 1, \(sequenceLength)]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("conv")];
      } -> (y);
    }
    """
}

/// A 1x1-conv program whose weight is stored as a 4-bit PALETTE and expanded
/// to fp16 on the ANE via `constexpr_lut_to_dense`: 4-bit indices into a
/// 16-entry fp16 lookup table. This is the int4 form the ANE architecture
/// paper measured streaming natively at about 2.37x fp16 bandwidth, and it is
/// an iOS16-vintage op, so the in-memory ANE compiler accepts it where the
/// iOS18 `constexpr_blockwise_shift_scale` does not. Matches the byte-exact
/// iOS16 MIL a real palettized model emits: `program(1.0)`, `func
/// main<ios16>`, `tensor<string, []>` / `tensor<int32, []>` scalar forms, all
/// op parameters inline in the attribute bracket. `indices` is the packed
/// `[outputDim*inputDim/2]` uint8 chunk (two 4-bit codes per byte, low first),
/// `lut` the 16 fp16 centroids; both are BLOBFILE chunks of `weights/weight.bin`
/// at the given header offsets (indices chunk type 3, lut chunk type 1).
public func buildConvMILTextInt4LUT(
    inputDim: Int, outputDim: Int, sequenceLength: Int,
    indicesOffset: UInt64, lutOffset: UInt64,
    programTag: String = UUID().uuidString
) -> String {
    let packed = outputDim * inputDim / 2
    return """
    program(1.0)
    [buildInfo = dict<tensor<string, []>, tensor<string, []>>({{"coremlc-component-MIL", "3520.4.1"}, {"coremlc-version", "3520.5.1"}, {"mlxfast-program-tag", "\(programTag)"}})]
    {
        func main<ios16>(tensor<fp16, [1, \(inputDim), 1, \(sequenceLength)]> x) {
            tensor<fp16, [\(outputDim), \(inputDim), 1, 1]> w = constexpr_lut_to_dense()[indices = tensor<uint8, [\(packed)]>(BLOBFILE(path = tensor<string, []>("@model_path/weights/weight.bin"), offset = tensor<uint64, []>(\(indicesOffset)))), lut = tensor<fp16, [16]>(BLOBFILE(path = tensor<string, []>("@model_path/weights/weight.bin"), offset = tensor<uint64, []>(\(lutOffset)))), name = tensor<string, []>("wdeq"), shape = tensor<uint32, [4]>([\(outputDim), \(inputDim), 1, 1])];
            tensor<int32, [2]> st = const()[name = tensor<string, []>("st"), val = tensor<int32, [2]>([1, 1])];
            tensor<string, []> pt = const()[name = tensor<string, []>("pt"), val = tensor<string, []>("valid")];
            tensor<int32, [2]> dl = const()[name = tensor<string, []>("dl"), val = tensor<int32, [2]>([1, 1])];
            tensor<int32, []> gr = const()[name = tensor<string, []>("gr"), val = tensor<int32, []>(1)];
            tensor<int32, [4]> pd = const()[name = tensor<string, []>("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];
            tensor<fp16, [1, \(outputDim), 1, \(sequenceLength)]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = tensor<string, []>("conv")];
        } -> (y);
    }
    """
}

// MARK: - MIL text (procedure bank: many convs, ONE loaded program)

/// One conv procedure in a `buildBankMILText` bank: an `[out, in]` 1x1 conv at
/// a fixed `sequenceLength`, reading its fp16 weight from `weightOffset` (a
/// `buildMultiWeightBlob` header offset into `weights/weight.bin`).
public struct ANEBankProcedure {
    public let inputDim: Int
    public let outputDim: Int
    public let sequenceLength: Int
    public let weightOffset: UInt64
    public init(inputDim: Int, outputDim: Int, sequenceLength: Int, weightOffset: UInt64) {
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.sequenceLength = sequenceLength
        self.weightOffset = weightOffset
    }
}

/// A procedure BANK: `procedures.count` independent 1x1-conv functions inside a
/// SINGLE MIL `program(1.3){}` block, so the ANE daemon loads them as ONE
/// program that holds ONE of the ~126 per-process program slots, yet exposes
/// each function to `evaluate` via its own `procedureIndex`. This is the
/// workaround for the 126-program count limit measured in
/// `ANEProgramCountLimitTests`: N fixed shapes cost 1 slot, not N.
///
/// The first function is named `main` (a program must define it); the rest are
/// `proc1`, `proc2`, ... in declaration order. The mapping from the numeric
/// `procedureIndex` passed at dispatch to these functions is NOT documented by
/// Apple and is resolved empirically by `ANEProcedureBankProbeTests` (it is
/// either declaration order, with `main` = 0, or alphabetical by function
/// name). Each function has its own input `x` and output `y`; all op and
/// const names are suffixed with the function index so nothing collides across
/// the shared program. Weight bytes are referenced from the shared
/// `weights/weight.bin` blob that `buildMultiWeightBlob` produces, at the
/// per-procedure `weightOffset`.
public func buildBankMILText(
    procedures: [ANEBankProcedure],
    functionName: (Int) -> String = { $0 == 0 ? "main" : "proc\($0)" },
    programTag: String = UUID().uuidString
) -> String {
    precondition(!procedures.isEmpty, "buildBankMILText needs at least one procedure")
    func function(_ i: Int, _ p: ANEBankProcedure) -> String {
        let name = functionName(i)
        return """
          func \(name)<ios18>(tensor<fp16, [1, \(p.inputDim), 1, \(p.sequenceLength)]> x\(i)) {
            tensor<fp16, [\(p.outputDim), \(p.inputDim), 1, 1]> w\(i) = const()[name=string("w\(i)"), val=tensor<fp16, [\(p.outputDim), \(p.inputDim), 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(\(p.weightOffset))))];
            string pt\(i) = const()[name=string("pt\(i)"), val=string("valid")];
            tensor<int32, [2]> st\(i) = const()[name=string("st\(i)"), val=tensor<int32, [2]>([1,1])];
            tensor<int32, [4]> pd\(i) = const()[name=string("pd\(i)"), val=tensor<int32, [4]>([0,0,0,0])];
            tensor<int32, [2]> dl\(i) = const()[name=string("dl\(i)"), val=tensor<int32, [2]>([1,1])];
            int32 gr\(i) = const()[name=string("gr\(i)"), val=int32(1)];
            tensor<fp16, [1, \(p.outputDim), 1, \(p.sequenceLength)]> y\(i) = conv(dilations=dl\(i), groups=gr\(i), pad=pd\(i), pad_type=pt\(i), strides=st\(i), weight=w\(i), x=x\(i))[name=string("conv\(i)")];
          } -> (y\(i));
        """
    }
    let functions = procedures.enumerated().map { function($0.offset, $0.element) }.joined(separator: "\n")
    return """
    program(1.3)
    [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}, {"mlxfast-program-tag", "\(programTag)"}})]
    {
    \(functions)
    }
    """
}

/// The on-disk weight blob `buildConvMILText`'s `BLOBFILE(offset=uint64(64))`
/// reference reads. `BLOBFILE` does not read a raw tensor payload directly
/// at that offset -- it reads oMLX's `make_blob` chunk-descriptor structure,
/// which the compiler parses to locate the actual payload. A 64-zero-byte
/// header (this function's first cut, matching the task brief's simplified
/// description) mmaps and reads without error but the compiler then rejects
/// the program with `InvalidMILProgram` ("Could not convert input MIL
/// program", confirmed via the ANECompilerService `log stream` trace) --
/// the chunk descriptor's magic/offset fields are load-bearing, not padding.
/// Layout (`make_blob` in oMLX's `qwen35_ane.mm`), matched byte-for-byte:
/// bytes `[0]=0x01`, `[4]=0x02` (blob-level header, unvalidated by a
/// single-chunk read but written for parity); at offset 64 (where the MIL
/// text's `BLOBFILE` offset points), a chunk header --
/// `[0..3]` = `EF BE AD DE` (magic, little-endian `0xDEADBEEF`), `[4]=0x01`
/// (chunk type), `[8..11]` = byte count of the payload (little-endian
/// `UInt32`), `[16..19]` = `128` (little-endian `UInt32`, the payload's
/// absolute byte offset in the file) -- followed by the payload itself
/// starting at absolute offset 128: the fp16 weight bytes, row-major
/// `[F,K,1,1]` (`outputDim` rows of `inputDim` fp16 values each).
public func buildConvWeightBlob(_ fp16Weight: Data) -> Data {
    let byteCount = fp16Weight.count
    var blob = [UInt8](repeating: 0, count: 128 + byteCount)
    blob[0] = 0x01
    blob[4] = 0x02
    blob[64] = 0xEF
    blob[65] = 0xBE
    blob[66] = 0xAD
    blob[67] = 0xDE
    blob[68] = 0x01
    withUnsafeBytes(of: UInt32(byteCount).littleEndian) { raw in
        for i in 0 ..< 4 { blob[64 + 8 + i] = raw[i] }
    }
    withUnsafeBytes(of: UInt32(128).littleEndian) { raw in
        for i in 0 ..< 4 { blob[64 + 16 + i] = raw[i] }
    }
    fp16Weight.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
        for i in 0 ..< byteCount { blob[128 + i] = src[i] }
    }
    return Data(blob)
}

// MARK: - MIL text (oMLX `fp16_swiglu_down_mil` format, fused SwiGLU-down)

/// The fused SwiGLU-down MLP as ONE MIL program: gate-conv, up-conv, silu,
/// mul, down-conv, no per-projection barrier -- oMLX's key optimization
/// (`fp16_swiglu_down_mil` in `qwen35_ane.mm`), matched verbatim. `inputDim`
/// is the model's hidden size (both the gate/up projections' input and the
/// down projection's output); `hiddenDim` is the ANE-fraction intermediate
/// width `F` (gate/up's output, down's input); `outputDim` is kept as its
/// own parameter, matching the reference signature, even though this MLP's
/// down-projection output is always `inputDim` again (down splits along the
/// *inter* dimension across ANE/GPU, not along hidden). Weight `BLOBFILE`
/// references point at `weights/weight.bin`, at the chunk-header offsets
/// `buildMultiWeightBlob` returns for the gate/up/down chunks in that order.
/// How the fused program computes `silu_out` from `gate`. All spellings are
/// algebraically SiLU; their ANE lowerings differ in precision. `.silu` and
/// `.sigmoidMul` go through the ANE's lookup-table sigmoid (measured error
/// on real activations 10-30x the fp16 floor -- the cause of the hybrid's
/// generation collapse). `.expDiv` (the production default, see
/// `ANESplitConfig.activation`) and `.tanhForm` use the ANE's accurate
/// `exp`/`tanh` and land at the conv rounding floor. `.none` skips the
/// activation entirely (diagnostic: isolates the convs and mul).
/// The two spellings of MIL text the ANE compiler accepts, and which one a
/// program must use throughout. `ios18` programs (`program(1.3)`, `func
/// main<ios18>`) write scalars bare: `string("x")`, `int32(1)`, `fp16(0.5)`.
/// `ios16` programs (`program(1.0)`, `func main<ios16>`) write every scalar as
/// a rank-0 tensor: `tensor<string, []>("x")`, `tensor<int32, []>(1)`. The
/// int4 palette op is confirmed only in the ios16 spelling and the int8 affine
/// op in the ios18 spelling, and mixing the two inside one program is an
/// `InvalidMILProgram`, so every line of a program goes through one dialect.
public enum MILDialect: Sendable {
    case ios18, ios16

    /// A string literal.
    func str(_ v: String) -> String { self == .ios18 ? "string(\"\(v)\")" : "tensor<string, []>(\"\(v)\")" }
    /// An fp16 scalar literal.
    func f16(_ v: Float) -> String { self == .ios18 ? "fp16(\(v))" : "tensor<fp16, []>(\(v))" }
    /// The declared type of an fp16 scalar variable.
    var f16Type: String { self == .ios18 ? "fp16" : "tensor<fp16, []>" }
    /// The declared type of a string variable.
    var stringType: String { self == .ios18 ? "string" : "tensor<string, []>" }
    /// The declared type of an int32 scalar variable.
    var int32Type: String { self == .ios18 ? "int32" : "tensor<int32, []>" }
    /// An int32 scalar literal.
    func i32(_ v: Int) -> String { self == .ios18 ? "int32(\(v))" : "tensor<int32, []>(\(v))" }
    /// The `name=` attribute of an op.
    func named(_ v: String) -> String { "[name=\(str(v))]" }

    /// The program header through the opening brace, with the tag stamped.
    func header(programTag: String) -> String {
        switch self {
        case .ios18:
            return """
            program(1.3)
            [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}, {"mlxfast-program-tag", "\(programTag)"}})]
            {
            """
        case .ios16:
            return """
            program(1.0)
            [buildInfo = dict<tensor<string, []>, tensor<string, []>>({{"coremlc-component-MIL", "3520.4.1"}, {"coremlc-version", "3520.5.1"}, {"mlxfast-program-tag", "\(programTag)"}})]
            {
            """
        }
    }

    var funcTarget: String { self == .ios18 ? "ios18" : "ios16" }

    /// The five conv attribute consts every 1x1-conv program declares.
    /// `groups` above 1 makes every conv in the program a grouped conv.
    var convConsts: String { convConsts(groups: 1) }
    func convConsts(groups: Int) -> String {
        """
        \(stringType) pt = const()[name=\(str("pt")), val=\(str("valid"))];
                tensor<int32, [2]> st = const()[name=\(str("st")), val=tensor<int32, [2]>([1, 1])];
                tensor<int32, [4]> pd = const()[name=\(str("pd")), val=tensor<int32, [4]>([0, 0, 0, 0])];
                tensor<int32, [2]> dl = const()[name=\(str("dl")), val=tensor<int32, [2]>([1, 1])];
                \(int32Type) gr = const()[name=\(str("gr")), val=\(i32(groups))];
        """
    }

    /// One 1x1 conv statement.
    func conv(out: String, outputDim: Int, sequenceLength: Int, weight: String, x: String, name: String) -> String {
        "tensor<fp16, [1, \(outputDim), 1, \(sequenceLength)]> \(out) = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=\(weight), x=\(x))\(named(name));"
    }

    /// A BLOBFILE reference into `weights/<file>` at a chunk header offset.
    func blob(file: String, offset: UInt64) -> String {
        "BLOBFILE(path=\(str("@model_path/weights/\(file)")), offset=\(self == .ios18 ? "uint64(\(offset))" : "tensor<uint64, []>(\(offset))"))"
    }
}

/// Which storage the ANE program keeps a projection weight in. The engine
/// multiplies in fp16 whatever the storage; `int8` is per-output-channel
/// affine (`constexpr_affine_dequantize`), `int4` is a 16-entry palette shared
/// by the whole tensor (`constexpr_lut_to_dense`) with a per-output-channel
/// scale applied to the conv OUTPUT, which is exact for a linear op and keeps
/// the weight a constexpr the ANE compiler accepts. Both are confirmed on the
/// ANE through the in-memory path; the MLX group-affine forms are not
/// ANE-executable at all (Core ML routes such a conv to the GPU), so no form
/// here is byte-identical to the GPU's tensor.
public enum ANEWeightForm: String, CaseIterable, Sendable {
    case fp16, int8, int4
}

public enum ANEActivation: String, CaseIterable, Sendable {
    case silu, sigmoidMul, expDiv, tanhForm, none

    /// MIL statements defining `silu_out` (`[1, F, 1, S]` fp16) from `gate`.
    func milLines(hiddenDim: Int, sequenceLength: Int, dialect d: MILDialect = .ios18) -> String {
        let t = "tensor<fp16, [1, \(hiddenDim), 1, \(sequenceLength)]>"
        switch self {
        case .silu:
            return "\(t) silu_out = silu(x=gate)\(d.named("silu"));"
        case .sigmoidMul:
            return """
            \(t) sig = sigmoid(x=gate)\(d.named("sig"));
                    \(t) silu_out = mul(x=gate, y=sig)\(d.named("silu"));
            """
        case .expDiv:
            return """
            \(d.f16Type) negone = const()[name=\(d.str("negone")), val=\(d.f16(-1.0))];
                    \(d.f16Type) one = const()[name=\(d.str("one")), val=\(d.f16(1.0))];
                    \(t) neg = mul(x=gate, y=negone)\(d.named("neg"));
                    \(t) e = exp(x=neg)\(d.named("e"));
                    \(t) den = add(x=e, y=one)\(d.named("den"));
                    \(t) silu_out = real_div(x=gate, y=den)\(d.named("silu"));
            """
        case .tanhForm:
            return """
            \(d.f16Type) half = const()[name=\(d.str("half")), val=\(d.f16(0.5))];
                    \(d.f16Type) one = const()[name=\(d.str("one")), val=\(d.f16(1.0))];
                    \(t) hg = mul(x=gate, y=half)\(d.named("hg"));
                    \(t) th = tanh(x=hg)\(d.named("th"));
                    \(t) thp = add(x=th, y=one)\(d.named("thp"));
                    \(t) sg = mul(x=thp, y=half)\(d.named("sg"));
                    \(t) silu_out = mul(x=gate, y=sg)\(d.named("silu"));
            """
        case .none:
            return "\(t) silu_out = identity(x=gate)\(d.named("silu"));"
        }
    }
}

/// Blob chunk references for a fused SwiGLU-down program in a compressed
/// weight form. Offsets are `buildMultiWeightBlob` HEADER offsets into
/// `weights/weight.bin`. Scales are fp16 chunks of one value per output
/// channel of their weight (`[F]` for gate and up, `[hidden]` for down).
public enum ANEFusedWeightRefs {
    /// int8 per-channel affine: three int8 data chunks and three scale chunks.
    case int8(gate: UInt64, up: UInt64, down: UInt64, gateScale: UInt64, upScale: UInt64, downScale: UInt64)
    /// int4 palette: three packed-index chunks (type 3), ONE shared 16-entry
    /// fp16 LUT chunk, and three per-channel output-scale chunks.
    case int4(gateIdx: UInt64, upIdx: UInt64, downIdx: UInt64, lut: UInt64, gateScale: UInt64, upScale: UInt64, downScale: UInt64)
}

/// The fused SwiGLU-down program with COMPRESSED weights. Same dataflow as
/// `buildSwiGLUDownMILText` (gate conv, up conv, activation, mul, down conv),
/// with the weights dequantized on the ANE. int8 is written in the ios18
/// dialect (the confirmed spelling of `constexpr_affine_dequantize`); int4 in
/// the ios16 dialect (the confirmed spelling of `constexpr_lut_to_dense`),
/// with each conv output multiplied by its per-channel `[1, O, 1, 1]` scale.
public func buildSwiGLUDownMILTextCompressed(
    inputDim: Int, hiddenDim: Int, outputDim: Int, sequenceLength: Int,
    weights: ANEFusedWeightRefs,
    activation: ANEActivation = ANESplitConfig.activation,
    groups: Int = 1,
    programTag: String = UUID().uuidString
) -> String {
    // With `groups > 1` (the grouped expert program) `inputDim`, `hiddenDim`
    // and `outputDim` are the TOTAL channel counts across groups; each conv's
    // weight is `[total_out, per_group_in, 1, 1]`.
    let inPerGroup = inputDim / groups
    let hidPerGroup = hiddenDim / groups
    let file = "weight.bin"
    switch weights {
    case let .int8(gate, up, down, gateScale, upScale, downScale):
        let d = MILDialect.ios18
        func w(_ name: String, _ o: Int, _ i: Int, data: UInt64, scale: UInt64) -> String {
            "tensor<fp16, [\(o), \(i), 1, 1]> \(name) = constexpr_affine_dequantize()[axis = int32(0), name = string(\"\(name)deq\"), quantized_data = tensor<int8, [\(o), \(i), 1, 1]>(\(d.blob(file: file, offset: data))), scale = tensor<fp16, [\(o)]>(\(d.blob(file: file, offset: scale))), zero_point = int8(0)];"
        }
        return """
        \(d.header(programTag: programTag))
          func main<\(d.funcTarget)>(tensor<fp16, [1, \(inputDim), 1, \(sequenceLength)]> x) {
            \(w("gw", hiddenDim, inPerGroup, data: gate, scale: gateScale))
            \(w("uw", hiddenDim, inPerGroup, data: up, scale: upScale))
            \(w("dw", outputDim, hidPerGroup, data: down, scale: downScale))
            \(d.convConsts(groups: groups))
            \(d.conv(out: "gate", outputDim: hiddenDim, sequenceLength: sequenceLength, weight: "gw", x: "x", name: "gate"))
            \(d.conv(out: "up", outputDim: hiddenDim, sequenceLength: sequenceLength, weight: "uw", x: "x", name: "up"))
            \(activation.milLines(hiddenDim: hiddenDim, sequenceLength: sequenceLength, dialect: d))
            tensor<fp16, [1, \(hiddenDim), 1, \(sequenceLength)]> act = mul(x=silu_out, y=up)\(d.named("swiglu"));
            \(d.conv(out: "y", outputDim: outputDim, sequenceLength: sequenceLength, weight: "dw", x: "act", name: "down"))
          } -> (y);
        }
        """
    case let .int4(gateIdx, upIdx, downIdx, lut, gateScale, upScale, downScale):
        let d = MILDialect.ios16
        func w(_ name: String, _ o: Int, _ i: Int, idx: UInt64) -> String {
            "tensor<fp16, [\(o), \(i), 1, 1]> \(name) = constexpr_lut_to_dense()[indices = tensor<uint8, [\(o * i / 2)]>(\(d.blob(file: file, offset: idx))), lut = tensor<fp16, [16]>(\(d.blob(file: file, offset: lut))), name = \(d.str("\(name)deq")), shape = tensor<uint32, [4]>([\(o), \(i), 1, 1])];"
        }
        func scale(_ name: String, _ o: Int, offset: UInt64) -> String {
            "tensor<fp16, [1, \(o), 1, 1]> \(name) = const()[name=\(d.str(name)), val=tensor<fp16, [1, \(o), 1, 1]>(\(d.blob(file: file, offset: offset)))];"
        }
        let th = "tensor<fp16, [1, \(hiddenDim), 1, \(sequenceLength)]>"
        return """
        \(d.header(programTag: programTag))
            func main<\(d.funcTarget)>(tensor<fp16, [1, \(inputDim), 1, \(sequenceLength)]> x) {
                \(w("gw", hiddenDim, inPerGroup, idx: gateIdx))
                \(w("uw", hiddenDim, inPerGroup, idx: upIdx))
                \(w("dw", outputDim, hidPerGroup, idx: downIdx))
                \(scale("gs", hiddenDim, offset: gateScale))
                \(scale("us", hiddenDim, offset: upScale))
                \(scale("ds", outputDim, offset: downScale))
                \(d.convConsts(groups: groups))
                \(d.conv(out: "gate_raw", outputDim: hiddenDim, sequenceLength: sequenceLength, weight: "gw", x: "x", name: "gate_raw"))
                \(th) gate = mul(x=gate_raw, y=gs)\(d.named("gate"));
                \(d.conv(out: "up_raw", outputDim: hiddenDim, sequenceLength: sequenceLength, weight: "uw", x: "x", name: "up_raw"))
                \(th) up = mul(x=up_raw, y=us)\(d.named("up"));
                \(activation.milLines(hiddenDim: hiddenDim, sequenceLength: sequenceLength, dialect: d))
                \(th) act = mul(x=silu_out, y=up)\(d.named("swiglu"));
                \(d.conv(out: "y_raw", outputDim: outputDim, sequenceLength: sequenceLength, weight: "dw", x: "act", name: "down_raw"))
                tensor<fp16, [1, \(outputDim), 1, \(sequenceLength)]> y = mul(x=y_raw, y=ds)\(d.named("down"));
            } -> (y);
        }
        """
    }
}

public func buildSwiGLUDownMILText(
    inputDim: Int, hiddenDim: Int, outputDim: Int, sequenceLength: Int,
    gateOffset: UInt64, upOffset: UInt64, downOffset: UInt64,
    activation: ANEActivation = ANESplitConfig.activation,
    programTag: String = UUID().uuidString
) -> String {
    """
    program(1.3)
    [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}, {"mlxfast-program-tag", "\(programTag)"}})]
    {
      func main<ios18>(tensor<fp16, [1, \(inputDim), 1, \(sequenceLength)]> x) {
        tensor<fp16, [\(hiddenDim), \(inputDim), 1, 1]> gw = const()[name=string("gw"), val=tensor<fp16, [\(hiddenDim), \(inputDim), 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(\(gateOffset))))];
        tensor<fp16, [\(hiddenDim), \(inputDim), 1, 1]> uw = const()[name=string("uw"), val=tensor<fp16, [\(hiddenDim), \(inputDim), 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(\(upOffset))))];
        tensor<fp16, [\(outputDim), \(hiddenDim), 1, 1]> dw = const()[name=string("dw"), val=tensor<fp16, [\(outputDim), \(hiddenDim), 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(\(downOffset))))];
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
        tensor<fp16, [1, \(hiddenDim), 1, \(sequenceLength)]> gate = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=gw, x=x)[name=string("gate")];
        tensor<fp16, [1, \(hiddenDim), 1, \(sequenceLength)]> up = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=uw, x=x)[name=string("up")];
        \(activation.milLines(hiddenDim: hiddenDim, sequenceLength: sequenceLength))
        tensor<fp16, [1, \(hiddenDim), 1, \(sequenceLength)]> act = mul(x=silu_out, y=up)[name=string("swiglu")];
        tensor<fp16, [1, \(outputDim), 1, \(sequenceLength)]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=dw, x=act)[name=string("down")];
      } -> (y);
    }
    """
}

/// Multi-chunk weight blob for a fused program (oMLX `append_blob_chunk` +
/// the swiglu constructor's blob assembly). A 64-byte zero prefix, then one
/// 64-byte-aligned chunk per entry in `chunks` -- each chunk is a 64-byte
/// header (`[0..3]` = `EF BE AD DE` magic, `uint32` type=1 at `+4`,
/// `uint64` byte count at `+8`, `uint64` absolute payload offset
/// (`chunkOffset+64`) at `+16`) followed immediately by the payload bytes.
/// After every chunk is appended, the prefix's first 8 bytes are set to
/// `uint32[0]=chunks.count` (chunk count), `uint32[1]=2` (version). Returns
/// each chunk's HEADER offset (not its payload offset) -- what a MIL text's
/// `BLOBFILE(offset=...)` reference expects, matching
/// `buildConvWeightBlob`'s single-chunk convention (offset 64 = the header,
/// not the payload at 128).
/// `chunkTypes`, when given, sets each chunk header's data-type field (byte
/// +4 of the chunk header) instead of the default `1`. The ANE blob format
/// encodes the payload width here: `1` for byte-and-wider payloads (fp16,
/// int8), `8` for packed int4. A weight.bin dumped from a coremltools int4
/// model has `8` there; fp16/int8 use `1`. Pass one entry per chunk, or omit
/// for all-`1`.
public func buildMultiWeightBlob(chunks: [Data], chunkTypes: [UInt32]? = nil) -> (blob: Data, offsets: [UInt64]) {
    if let chunkTypes { precondition(chunkTypes.count == chunks.count, "chunkTypes must match chunks") }
    var blob = Data(count: 64)
    var offsets: [UInt64] = []
    for (idx, chunk) in chunks.enumerated() {
        let aligned = ((blob.count + 63) / 64) * 64
        if aligned > blob.count {
            blob.append(Data(count: aligned - blob.count))
        }
        let chunkOffset = UInt64(blob.count)
        var header = [UInt8](repeating: 0, count: 64)
        header[0] = 0xEF
        header[1] = 0xBE
        header[2] = 0xAD
        header[3] = 0xDE
        withUnsafeBytes(of: (chunkTypes?[idx] ?? 1).littleEndian) { raw in
            for i in 0 ..< 4 { header[4 + i] = raw[i] }
        }
        withUnsafeBytes(of: UInt64(chunk.count).littleEndian) { raw in
            for i in 0 ..< 8 { header[8 + i] = raw[i] }
        }
        withUnsafeBytes(of: (chunkOffset + 64).littleEndian) { raw in
            for i in 0 ..< 8 { header[16 + i] = raw[i] }
        }
        blob.append(Data(header))
        blob.append(chunk)
        offsets.append(chunkOffset)
    }
    withUnsafeBytes(of: UInt32(chunks.count).littleEndian) { raw in
        for i in 0 ..< 4 { blob[i] = raw[i] }
    }
    withUnsafeBytes(of: UInt32(2).littleEndian) { raw in
        for i in 0 ..< 4 { blob[4 + i] = raw[i] }
    }
    return (blob, offsets)
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
///
/// Stride-aware: a freshly allocated `MLMultiArray(shape:dataType:)` is a
/// plain CPU allocation and is NOT guaranteed to be tightly packed -- Core
/// ML is free to pad the trailing sequence axis for its own layout reasons
/// (the same kind of padding `multiArray_1C1S_toMLX` below observes,
/// confirmed, on the ANE's *prediction output*), so `arr`'s `K`-axis stride
/// can exceed `S` for any `S` (S=1 decode, arbitrary prefill lengths, ...).
/// Writing via `memcpy` at the packed byte count either falls short of the
/// padded rows (silently leaving garbage/uninitialized padding, which is
/// harmless since the conv op never reads it) or -- for non-contiguous
/// strides -- corrupts the wrong elements entirely. Reading `arr.strides`
/// first and copying accordingly is correct for any padding Core ML
/// chooses; bulk `memcpy` (whole-buffer, or row-by-row when only the
/// between-row gap is padded) keeps this off the per-element scalar path
/// that would otherwise run F*S times on every ANE projection call.
public func mlxToMultiArray_1C1S(_ x: MLXArray) throws -> MLMultiArray {
    let S = x.shape[0], K = x.shape[1]
    // `transposed` is a lazy metadata op: the result [K,S] has strides [1,K],
    // which match no contiguous layout. `asData()` on such an array degrades to
    // a per-element (2-byte) scalar copy over K*S elements -- measured at ~2.4s
    // for K=5120,S=512 (and ~8.5s at K=17408). `contiguous(...)` forces a single
    // GPU kernel that writes a row-contiguous [K,S] buffer, after which
    // `asData()` takes the whole-buffer memcpy fast path (~0.2ms). Same bytes,
    // ~3500x faster makeInput.
    let xT = contiguous(x.transposed(1, 0).asType(.float16)) // [K,S], row-contiguous
    eval(xT)
    let arr = try MLMultiArray(shape: [1, K, 1, S].map { NSNumber(value: $0) }, dataType: .float16)
    let bytes = xT.asData().data
    let elementSize = MemoryLayout<Float16>.stride
    precondition(bytes.count == K * S * elementSize,
                 "mlxToMultiArray_1C1S: source is \(bytes.count) bytes, expected packed K*S*2 = \(K * S * elementSize)")

    let strides = arr.strides.map(\.intValue) // element strides for [1,K,1,S]
    let strideK = strides[1], strideS = strides[3]

    arr.withUnsafeMutableBytes { raw, _ in
        // Real bound: the strides above only describe the layout, not
        // whether the backing store is actually large enough for the
        // farthest element this scatter touches -- verify that directly
        // rather than trusting the nominal `raw.count`.
        precondition(((K - 1) * strideK + (S - 1) * strideS + 1) * elementSize <= raw.count,
                     "mlxToMultiArray_1C1S: destination MLMultiArray (\(raw.count) bytes) too small for a [K,S] scatter at strides (\(strideK), \(strideS))")
        bytes.withUnsafeBytes { src in
            // All offsets below are computed on the RAW (byte) pointers, so
            // every stride/index gets an explicit `* elementSize` -- MLX
            // and MLMultiArray strides are in elements, memcpy lengths and
            // pointer arithmetic on `UnsafeRawPointer` are in bytes.
            let dstBase = raw.baseAddress!
            let srcBase = src.baseAddress!
            if strideK == S, strideS == 1 {
                // Fully packed -- one bulk copy for the whole buffer.
                memcpy(dstBase, srcBase, bytes.count)
            } else if strideS == 1 {
                // Rows are packed but padded between each other -- one
                // memcpy per row instead of a per-element scalar loop.
                let rowBytes = S * elementSize
                for k in 0 ..< K {
                    memcpy(dstBase + k * strideK * elementSize, srcBase + k * rowBytes, rowBytes)
                }
            } else {
                // Non-unit last-axis stride -- no contiguous run to copy,
                // fall back to a scalar scatter.
                let dst = dstBase.assumingMemoryBound(to: Float16.self)
                let srcTyped = srcBase.assumingMemoryBound(to: Float16.self)
                for k in 0 ..< K {
                    let rowBase = k * strideK
                    for s in 0 ..< S {
                        dst[rowBase + s * strideS] = srcTyped[k * S + s]
                    }
                }
            }
        }
    }
    return arr
}

/// `MLMultiArray` shaped `[1,F,1,S]` -> `[S,F]` MLXArray.
///
/// Stride-aware for the same reason as `mlxToMultiArray_1C1S`'s write side:
/// Core ML's ANE `prediction` output pads the trailing sequence axis (`S`)
/// to a multiple of 32 elements in fp16, so `a`'s byte count is
/// `F * paddedS * 2`, not `F * S * 2`. Passing `raw.count` straight into
/// `MLXArray.init` (the prior implementation) preconditions on
/// `byteCount == F*S*2` and crashes for any `S` that is not already a
/// multiple of 32 (confirmed at S=8; guaranteed at decode's S=1). Reading
/// `a.strides` and gathering into a packed `[F,S]` buffer first is correct
/// for any padding; the gather uses bulk `memcpy` (whole-buffer, or
/// row-by-row when only the between-row gap is padded) rather than a
/// per-element scalar loop, since this runs on every ANE projection call.
public func multiArray_1C1S_toMLX(_ a: MLMultiArray) -> MLXArray {
    precondition(a.dataType == .float16, "multiArray_1C1S_toMLX expects a float16 MLMultiArray, got \(a.dataType)")
    let shape = a.shape.map(\.intValue) // [1,F,1,S]
    precondition(shape.count == 4 && shape[0] == 1 && shape[2] == 1,
                 "multiArray_1C1S_toMLX expects shape [1,F,1,S], got \(shape)")
    let F = shape[1], S = shape[3]
    let strides = a.strides.map(\.intValue) // element strides, matching `shape`
    let strideF = strides[1], strideS = strides[3]
    let elementSize = MemoryLayout<Float16>.stride

    var packed = [Float16](repeating: 0, count: F * S)
    a.withUnsafeBytes { raw in
        // Real bound, not the tautological "packed buffer is F*S by
        // construction" check: verify the SOURCE backing store actually
        // covers the farthest element this gather reads.
        precondition(((F - 1) * strideF + (S - 1) * strideS + 1) * elementSize <= raw.count,
                     "multiArray_1C1S_toMLX: source MLMultiArray (\(raw.count) bytes) too small for a [F,S] gather at strides (\(strideF), \(strideS))")
        let srcBase = raw.baseAddress!
        packed.withUnsafeMutableBytes { dstRaw in
            let dstBase = dstRaw.baseAddress!
            if strideS == 1, strideF == S {
                // Fully packed -- one bulk copy for the whole buffer.
                memcpy(dstBase, srcBase, F * S * elementSize)
            } else if strideS == 1 {
                // Rows are packed but padded between each other -- one
                // memcpy per row instead of a per-element scalar loop.
                let rowBytes = S * elementSize
                for f in 0 ..< F {
                    memcpy(dstBase + f * rowBytes, srcBase + f * strideF * elementSize, rowBytes)
                }
            } else {
                // Non-unit last-axis stride -- no contiguous run to copy,
                // fall back to a scalar gather.
                let dst = dstBase.assumingMemoryBound(to: Float16.self)
                let src = srcBase.assumingMemoryBound(to: Float16.self)
                for f in 0 ..< F {
                    let rowBase = f * strideF
                    for s in 0 ..< S {
                        dst[f * S + s] = src[rowBase + s * strideS]
                    }
                }
            }
        }
    }
    let bytes = packed.withUnsafeBufferPointer { Data(buffer: $0) }
    precondition(bytes.count == F * S * elementSize,
                 "multiArray_1C1S_toMLX: packed buffer is \(bytes.count) bytes, expected F*S*2 = \(F * S * elementSize)")
    let arr = MLXArray(bytes, [F, S], type: Float16.self)
    return arr.transposed(1, 0) // [S,F]
}

// MARK: - MIL text: one 1x1 conv in a compressed weight form

/// Blob chunk references for a single-projection program in a compressed form
/// (`buildMultiWeightBlob` header offsets into `weights/weight.bin`).
public enum ANEConvWeightRefs {
    /// int8 per-channel affine: the int8 data chunk and the fp16 `[out]` scale chunk.
    case int8(data: UInt64, scale: UInt64)
    /// int4 palette: packed indices (type 3), the 16-entry fp16 LUT, and the
    /// fp16 `[out]` scale applied to the conv output.
    case int4(indices: UInt64, lut: UInt64, scale: UInt64)
}

/// `buildConvMILText`'s compressed twin: `x[1,IN,1,S] @ w[OUT,IN]^T` with the
/// weight stored int8 (ios18 spelling) or as a 4-bit palette with a
/// per-channel output scale (ios16 spelling).
public func buildConvMILTextForm(
    inputDim: Int, outputDim: Int, sequenceLength: Int,
    weight: ANEConvWeightRefs, programTag: String = UUID().uuidString
) -> String {
    let file = "weight.bin"
    switch weight {
    case let .int8(data, scale):
        let d = MILDialect.ios18
        return """
        \(d.header(programTag: programTag))
          func main<\(d.funcTarget)>(tensor<fp16, [1, \(inputDim), 1, \(sequenceLength)]> x) {
            tensor<fp16, [\(outputDim), \(inputDim), 1, 1]> w = constexpr_affine_dequantize()[axis = int32(0), name = string("wdeq"), quantized_data = tensor<int8, [\(outputDim), \(inputDim), 1, 1]>(\(d.blob(file: file, offset: data))), scale = tensor<fp16, [\(outputDim)]>(\(d.blob(file: file, offset: scale))), zero_point = int8(0)];
            \(d.convConsts)
            \(d.conv(out: "y", outputDim: outputDim, sequenceLength: sequenceLength, weight: "w", x: "x", name: "conv"))
          } -> (y);
        }
        """
    case let .int4(indices, lut, scale):
        let d = MILDialect.ios16
        return """
        \(d.header(programTag: programTag))
            func main<\(d.funcTarget)>(tensor<fp16, [1, \(inputDim), 1, \(sequenceLength)]> x) {
                tensor<fp16, [\(outputDim), \(inputDim), 1, 1]> w = constexpr_lut_to_dense()[indices = tensor<uint8, [\(outputDim * inputDim / 2)]>(\(d.blob(file: file, offset: indices))), lut = tensor<fp16, [16]>(\(d.blob(file: file, offset: lut))), name = \(d.str("wdeq")), shape = tensor<uint32, [4]>([\(outputDim), \(inputDim), 1, 1])];
                tensor<fp16, [1, \(outputDim), 1, 1]> ws = const()[name=\(d.str("ws")), val=tensor<fp16, [1, \(outputDim), 1, 1]>(\(d.blob(file: file, offset: scale)))];
                \(d.convConsts)
                \(d.conv(out: "y_raw", outputDim: outputDim, sequenceLength: sequenceLength, weight: "w", x: "x", name: "conv"))
                tensor<fp16, [1, \(outputDim), 1, \(sequenceLength)]> y = mul(x=y_raw, y=ws)\(d.named("scale"));
            } -> (y);
        }
        """
    }
}
