// Does Apple's Core ML compiler accept an int8 MULTIPLY, and where does it place it?
//
// Prior conclusions leaned on coremltools' Python op type domains, which
// describe what THEIR CONVERTER emits -- not what Apple's compiler accepts.
// We hand-author the protobuf, so we are not bound by the converter. This asks
// the compiler directly.
//
// Each case builds a MIL program with int8 operands somewhere in the multiply
// and reports: does it compile, and what device does MLComputePlan assign?
// A rejection prints the compiler's own diagnostic, which is the most
// authoritative statement available about what the runtime supports.
import CoreML
import Foundation

let M = 256, K = 512, N = 512

func i8(_ n: Int) -> Data {
    var d = Data(count: n)
    d.withUnsafeMutableBytes { r in
        let p = r.bindMemory(to: Int8.self)
        for i in 0 ..< n { p[i] = Int8.random(in: -8 ... 8) }
    }
    return d
}
func f16(_ n: Int, _ v: Float) -> Data {
    var d = Data(count: n * 2)
    d.withUnsafeMutableBytes { r in
        let p = r.bindMemory(to: Float16.self)
        for i in 0 ..< n { p[i] = Float16(v) }
    }
    return d
}

/// dt codes: FLOAT16=10 FLOAT32=11 INT8=21 INT32=23 UINT8=31
func wrap(_ ops: Data, inDt: MILType, inShape: [Int], outDt: MILType, outShape: [Int],
          inFeat: UInt64, outFeat: UInt64) -> Data {
    let block = strF(2, "y") + ops
    let fn = lenF(1, namedValue("x", inDt, inShape)) + strF(2, "CoreML8")
        + mapEntry(3, key: "CoreML8", value: block)
    let program = varF(1, 1) + mapEntry(2, key: "main", value: fn)
    func fd(_ n: String, _ shape: [Int], _ ft: UInt64) -> Data {
        let arr = lenF(1, shape.reduce(Data()) { $0 + varint(UInt64($1)) }) + varF(2, ft)
        return strF(1, n) + lenF(3, lenF(5, arr))
    }
    let desc = lenF(1, fd("x", inShape, inFeat)) + lenF(10, fd("y", outShape, outFeat))
    return varF(1, 9) + lenF(2, desc) + lenF(502, program)
}

/// FLOAT32=65568 FLOAT16=65552 INT32=131104
func caseIntMatmul(outDt: MILType, outFeat: UInt64) -> Data {
    var ops = Data()
    ops += lenF(3, constOp(name: "w", dt: .int8, shape: [N, K], payload: i8(N * K)))
    ops += lenF(3, constBoolOp(name: "tx", value: false))
    ops += lenF(3, constBoolOp(name: "ty", value: true))
    ops += lenF(3, op("matmul", name: "mm",
                      inputs: [("x", "x"), ("y", "w"), ("transpose_x", "tx"), ("transpose_y", "ty")],
                      outName: "y", outType: .int8, outShape: [M, N]))
    return wrap(ops, inDt: .int8, inShape: [M, K], outDt: outDt, outShape: [M, N],
                inFeat: 131104, outFeat: outFeat)
}

func caseInt8Conv() -> Data {
    var ops = Data()
    ops += lenF(3, constIntsOp(name: "st", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "dl", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0, 0, 0, 0]))
    ops += lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    ops += lenF(3, constOp(name: "w", dt: .int8, shape: [N, K, 1, 1], payload: i8(N * K)))
    ops += lenF(3, op("conv", name: "cv",
                      inputs: [("x", "x"), ("weight", "w"), ("strides", "st"),
                               ("pad_type", "pt"), ("pad", "pd"),
                               ("dilations", "dl"), ("groups", "gp")],
                      outName: "y", outType: .int8, outShape: [1, N, 1, M]))
    return wrap(ops, inDt: .int8, inShape: [1, K, 1, M], outDt: .int8, outShape: [1, N, 1, M],
                inFeat: 131104, outFeat: 131104)
}

/// iOS15 conv_quantized: activation T (fp16), weight U (uint8), explicit
/// quant_scale / quant_bias. The one op built for quantized weights.
func caseConvQuantized() -> Data {
    var ops = Data()
    ops += lenF(3, constIntsOp(name: "st", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "dl", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0, 0, 0, 0]))
    ops += lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    ops += lenF(3, constStringOp(name: "qt", value: "linear"))
    ops += lenF(3, constIntsOp(name: "nb", values: [8]))
    ops += lenF(3, constOp(name: "w", dt: .uint8, shape: [N, K, 1, 1], payload: i8(N * K)))
    ops += lenF(3, constOp(name: "qs", dt: .fp16, shape: [1], payload: f16(1, 0.02)))
    ops += lenF(3, constOp(name: "qb", dt: .fp16, shape: [1], payload: f16(1, 0.0)))
    ops += lenF(3, op("conv_quantized", name: "cq",
                      inputs: [("x", "x"), ("weight", "w"), ("strides", "st"),
                               ("pad_type", "pt"), ("pad", "pd"), ("dilations", "dl"),
                               ("groups", "gp"), ("quantization_type", "qt"),
                               ("nbits", "nb"), ("quant_scale", "qs"), ("quant_bias", "qb")],
                      outName: "y", outType: .fp16, outShape: [1, N, 1, M]))
    return wrap(ops, inDt: .fp16, inShape: [1, K, 1, M], outDt: .fp16, outShape: [1, N, 1, M],
                inFeat: 65552, outFeat: 65552)
}


// ---------------------------------------------------------------------------
// The spelling the first three cases never tried.
//
// Core ML does not express int8 compute by handing int8 tensors to `conv` or
// `matmul`. It expresses it with the iOS17 `quantize` / `dequantize` ops
// around an fp16-typed graph: the tensors are declared fp16, the compiler
// recognises the quantize/dequantize pattern, and fuses it into int8 compute
// on hardware that has it (A17 Pro and M4 onward).
// Signatures from coremltools/converters/mil/mil/ops/defs/iOS17/quantization_ops.py:
//   quantize(input: SrcT, scale: SrcT const, zero_point: DstT const?,
//            axis: int32 const?, output_dtype: str const)   SrcT=fp16/fp32 DstT=int8/uint8
//   dequantize(input: SrcT, scale: DstT const, zero_point: SrcT const?,
//            axis: int32 const?)                            SrcT=int8/uint8 DstT=fp16/fp32
// ---------------------------------------------------------------------------

func i8v(_ values: [Int8]) -> Data {
    var d = Data(count: values.count)
    d.withUnsafeMutableBytes { r in
        let p = r.bindMemory(to: Int8.self)
        for (i, v) in values.enumerated() { p[i] = v }
    }
    return d
}

/// Isolation: is `quantize` accepted at all, and where is it placed? If this
/// is rejected the whole activation-quantization route is closed. If it is
/// accepted, the earlier "int8 is refused" conclusion was about the spelling.
func caseQuantizeOnly(scaleShape: [Int]) -> Data {
    var ops = Data()
    ops += lenF(3, constOp(name: "qs", dt: .fp16, shape: scaleShape, payload: f16(1, 0.02)))
    ops += lenF(3, constOp(name: "qz", dt: .int8, shape: scaleShape, payload: i8v([0])))
    ops += lenF(3, constStringOp(name: "od", value: "int8"))
    ops += lenF(3, op("quantize", name: "q",
                      inputs: [("input", "x"), ("scale", "qs"), ("zero_point", "qz"),
                               ("output_dtype", "od")],
                      outName: "y", outType: .int8, outShape: [1, K, 1, M]))
    return wrap(ops, inDt: .fp16, inShape: [1, K, 1, M], outDt: .int8, outShape: [1, K, 1, M],
                inFeat: 65552, outFeat: 131104)
}

/// W8A8, the shape Apple's own W8A8 numbers are measured at: the activation
/// is quantized then dequantized, the weight is a dequantized int8 const, and
/// the conv sits between them in fp16 types. The fusion is the compiler's job.
func caseW8A8Conv(scaleShape: [Int]) -> Data {
    var ops = Data()
    ops += lenF(3, constIntsOp(name: "st", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "dl", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0, 0, 0, 0]))
    ops += lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    ops += lenF(3, constOp(name: "qs", dt: .fp16, shape: scaleShape, payload: f16(1, 0.02)))
    ops += lenF(3, constOp(name: "qz", dt: .int8, shape: scaleShape, payload: i8v([0])))
    ops += lenF(3, constStringOp(name: "od", value: "int8"))
    ops += lenF(3, constOp(name: "wq", dt: .int8, shape: [N, K, 1, 1], payload: i8(N * K)))
    // activation: fp16 -> int8 -> fp16
    ops += lenF(3, op("quantize", name: "q",
                      inputs: [("input", "x"), ("scale", "qs"), ("zero_point", "qz"),
                               ("output_dtype", "od")],
                      outName: "xq", outType: .int8, outShape: [1, K, 1, M]))
    ops += lenF(3, op("dequantize", name: "dq",
                      inputs: [("input", "xq"), ("scale", "qs"), ("zero_point", "qz")],
                      outName: "xd", outType: .fp16, outShape: [1, K, 1, M]))
    // weight: int8 const -> fp16
    ops += lenF(3, op("dequantize", name: "dw",
                      inputs: [("input", "wq"), ("scale", "qs"), ("zero_point", "qz")],
                      outName: "wd", outType: .fp16, outShape: [N, K, 1, 1]))
    ops += lenF(3, op("conv", name: "cv",
                      inputs: [("x", "xd"), ("weight", "wd"), ("strides", "st"),
                               ("pad_type", "pt"), ("pad", "pd"),
                               ("dilations", "dl"), ("groups", "gp")],
                      outName: "y", outType: .fp16, outShape: [1, N, 1, M]))
    return wrap(ops, inDt: .fp16, inShape: [1, K, 1, M], outDt: .fp16, outShape: [1, N, 1, M],
                inFeat: 65552, outFeat: 65552)
}

/// W8A16: weight-only. The fallback if activation quantization is refused --
/// still halves weight bytes, and is what the ANE has accepted since iOS16.
func caseW8A16Conv(scaleShape: [Int]) -> Data {
    var ops = Data()
    ops += lenF(3, constIntsOp(name: "st", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "dl", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0, 0, 0, 0]))
    ops += lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    ops += lenF(3, constOp(name: "qs", dt: .fp16, shape: scaleShape, payload: f16(1, 0.02)))
    ops += lenF(3, constOp(name: "qz", dt: .int8, shape: scaleShape, payload: i8v([0])))
    ops += lenF(3, constOp(name: "wq", dt: .int8, shape: [N, K, 1, 1], payload: i8(N * K)))
    ops += lenF(3, op("dequantize", name: "dw",
                      inputs: [("input", "wq"), ("scale", "qs"), ("zero_point", "qz")],
                      outName: "wd", outType: .fp16, outShape: [N, K, 1, 1]))
    ops += lenF(3, op("conv", name: "cv",
                      inputs: [("x", "x"), ("weight", "wd"), ("strides", "st"),
                               ("pad_type", "pt"), ("pad", "pd"),
                               ("dilations", "dl"), ("groups", "gp")],
                      outName: "y", outType: .fp16, outShape: [1, N, 1, M]))
    return wrap(ops, inDt: .fp16, inShape: [1, K, 1, M], outDt: .fp16, outShape: [1, N, 1, M],
                inFeat: 65552, outFeat: 65552)
}


/// CONTROL. A plain fp16 conv in the same wrapper, same shapes, no
/// quantization anywhere. Without this, a CPU placement on the W8A8 case says
/// nothing: it could be int8 demoting to CPU, or it could be that this
/// probe's shape or wrapper never reaches the ANE for any dtype. This is the
/// case that tells those two apart.
func caseFP16ConvControl() -> Data {
    var ops = Data()
    ops += lenF(3, constIntsOp(name: "st", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "dl", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0, 0, 0, 0]))
    ops += lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    ops += lenF(3, constOp(name: "w", dt: .fp16, shape: [N, K, 1, 1], payload: f16(N * K, 0.01)))
    ops += lenF(3, op("conv", name: "cv",
                      inputs: [("x", "x"), ("weight", "w"), ("strides", "st"),
                               ("pad_type", "pt"), ("pad", "pd"),
                               ("dilations", "dl"), ("groups", "gp")],
                      outName: "y", outType: .fp16, outShape: [1, N, 1, M]))
    return wrap(ops, inDt: .fp16, inShape: [1, K, 1, M], outDt: .fp16, outShape: [1, N, 1, M],
                inFeat: 65552, outFeat: 65552)
}

/// Per-channel W8A8. The rank-1 rejection above says a vector `scale` needs an
/// explicit `axis`, so this supplies one. Per-channel is the scheme we would
/// actually want, since it is what the hybrid-collapse work already measured.
func caseW8A8PerChannel() -> Data {
    var ops = Data()
    ops += lenF(3, constIntsOp(name: "st", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "dl", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0, 0, 0, 0]))
    ops += lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    // Activation: one scale for the whole tensor, so no axis.
    ops += lenF(3, constScalarOp(name: "as", dt: .fp16, payload: f16(1, 0.02)))
    ops += lenF(3, constScalarOp(name: "az", dt: .int8, payload: i8v([0])))
    ops += lenF(3, constStringOp(name: "od", value: "int8"))
    // Weight: one scale per output channel, which is axis 0 of [N,K,1,1].
    ops += lenF(3, constOp(name: "ws", dt: .fp16, shape: [N], payload: f16(N, 0.01)))
    ops += lenF(3, constOp(name: "wz", dt: .int8, shape: [N], payload: i8v([Int8](repeating: 0, count: N))))
    ops += lenF(3, constIntsOp(name: "ax0", values: [0]))
    ops += lenF(3, constOp(name: "wq", dt: .int8, shape: [N, K, 1, 1], payload: i8(N * K)))
    ops += lenF(3, op("quantize", name: "q",
                      inputs: [("input", "x"), ("scale", "as"), ("zero_point", "az"),
                               ("output_dtype", "od")],
                      outName: "xq", outType: .int8, outShape: [1, K, 1, M]))
    ops += lenF(3, op("dequantize", name: "dq",
                      inputs: [("input", "xq"), ("scale", "as"), ("zero_point", "az")],
                      outName: "xd", outType: .fp16, outShape: [1, K, 1, M]))
    ops += lenF(3, op("dequantize", name: "dw",
                      inputs: [("input", "wq"), ("scale", "ws"), ("zero_point", "wz"),
                               ("axis", "ax0")],
                      outName: "wd", outType: .fp16, outShape: [N, K, 1, 1]))
    ops += lenF(3, op("conv", name: "cv",
                      inputs: [("x", "xd"), ("weight", "wd"), ("strides", "st"),
                               ("pad_type", "pt"), ("pad", "pd"),
                               ("dilations", "dl"), ("groups", "gp")],
                      outName: "y", outType: .fp16, outShape: [1, N, 1, M]))
    return wrap(ops, inDt: .fp16, inShape: [1, K, 1, M], outDt: .fp16, outShape: [1, N, 1, M],
                inFeat: 65552, outFeat: 65552)
}


/// The weight side done the way Core ML actually represents compressed
/// weights: `constexpr_affine_dequantize` (iOS16) is a COMPILE-TIME op, so the
/// int8 bytes are stored and expanded during compilation. A runtime
/// `dequantize` of a plain const is a real runtime op instead, which is the
/// most likely reason the cases above fell to CPU.
///   constexpr_affine_dequantize(quantized_data, zero_point, scale, axis)
func caseConstexprWeightOnly() -> Data {
    var ops = Data()
    ops += lenF(3, constIntsOp(name: "st", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "dl", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0, 0, 0, 0]))
    ops += lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    ops += lenF(3, constOp(name: "wq", dt: .int8, shape: [N, K, 1, 1], payload: i8(N * K)))
    ops += lenF(3, constOp(name: "ws", dt: .fp16, shape: [N], payload: f16(N, 0.01)))
    ops += lenF(3, constOp(name: "wz", dt: .int8, shape: [N], payload: i8v([Int8](repeating: 0, count: N))))
    ops += lenF(3, constIntsOp(name: "ax0", values: [0]))
    ops += lenF(3, op("constexpr_affine_dequantize", name: "cad",
                      inputs: [("quantized_data", "wq"), ("zero_point", "wz"),
                               ("scale", "ws"), ("axis", "ax0")],
                      outName: "wd", outType: .fp16, outShape: [N, K, 1, 1]))
    ops += lenF(3, op("conv", name: "cv",
                      inputs: [("x", "x"), ("weight", "wd"), ("strides", "st"),
                               ("pad_type", "pt"), ("pad", "pd"),
                               ("dilations", "dl"), ("groups", "gp")],
                      outName: "y", outType: .fp16, outShape: [1, N, 1, M]))
    return wrap(ops, inDt: .fp16, inShape: [1, K, 1, M], outDt: .fp16, outShape: [1, N, 1, M],
                inFeat: 65552, outFeat: 65552)
}

/// Full W8A8: constexpr weight plus a quantize/dequantize pair on the
/// activation. This is the combination Apple's own W8A8 numbers describe.
func caseConstexprW8A8() -> Data {
    var ops = Data()
    ops += lenF(3, constIntsOp(name: "st", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "dl", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0, 0, 0, 0]))
    ops += lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    ops += lenF(3, constScalarOp(name: "as", dt: .fp16, payload: f16(1, 0.02)))
    ops += lenF(3, constScalarOp(name: "az", dt: .int8, payload: i8v([0])))
    ops += lenF(3, constStringOp(name: "od", value: "int8"))
    ops += lenF(3, constOp(name: "wq", dt: .int8, shape: [N, K, 1, 1], payload: i8(N * K)))
    ops += lenF(3, constOp(name: "ws", dt: .fp16, shape: [N], payload: f16(N, 0.01)))
    ops += lenF(3, constOp(name: "wz", dt: .int8, shape: [N], payload: i8v([Int8](repeating: 0, count: N))))
    ops += lenF(3, constIntsOp(name: "ax0", values: [0]))
    ops += lenF(3, op("quantize", name: "q",
                      inputs: [("input", "x"), ("scale", "as"), ("zero_point", "az"),
                               ("output_dtype", "od")],
                      outName: "xq", outType: .int8, outShape: [1, K, 1, M]))
    ops += lenF(3, op("dequantize", name: "dq",
                      inputs: [("input", "xq"), ("scale", "as"), ("zero_point", "az")],
                      outName: "xd", outType: .fp16, outShape: [1, K, 1, M]))
    ops += lenF(3, op("constexpr_affine_dequantize", name: "cad",
                      inputs: [("quantized_data", "wq"), ("zero_point", "wz"),
                               ("scale", "ws"), ("axis", "ax0")],
                      outName: "wd", outType: .fp16, outShape: [N, K, 1, 1]))
    ops += lenF(3, op("conv", name: "cv",
                      inputs: [("x", "xd"), ("weight", "wd"), ("strides", "st"),
                               ("pad_type", "pt"), ("pad", "pd"),
                               ("dilations", "dl"), ("groups", "gp")],
                      outName: "y", outType: .fp16, outShape: [1, N, 1, M]))
    return wrap(ops, inDt: .fp16, inShape: [1, K, 1, M], outDt: .fp16, outShape: [1, N, 1, M],
                inFeat: 65552, outFeat: 65552)
}


/// The runtime called `quantized_data` undefined, so its parameter names are
/// not coremltools' Python names. Enumerate the plausible ones against the
/// compiler, which names the offending attribute in its diagnostic.
/// `constexpr_blockwise_shift_scale` is the iOS18 op a CoreML8 program should
/// prefer: data / scale / offset.
func caseConstexprNamed(_ opType: String, dataParam: String, zpParam: String?,
                        scaleShape: [Int], withAxis: Bool, quantAct: Bool) -> Data {
    var ops = Data()
    ops += lenF(3, constIntsOp(name: "st", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "dl", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0, 0, 0, 0]))
    ops += lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    ops += lenF(3, constOp(name: "wq", dt: .int8, shape: [N, K, 1, 1], payload: i8(N * K)))
    let scaleCount = scaleShape.reduce(1, *)
    ops += lenF(3, constOp(name: "ws", dt: .fp16, shape: scaleShape, payload: f16(scaleCount, 0.01)))
    ops += lenF(3, constOp(name: "wz", dt: .int8, shape: scaleShape,
                           payload: i8v([Int8](repeating: 0, count: scaleCount))))
    ops += lenF(3, constIntsOp(name: "ax0", values: [0]))

    var cadInputs: [(String, String)] = [(dataParam, "wq"), ("scale", "ws")]
    if let zpParam { cadInputs.append((zpParam, "wz")) }
    if withAxis { cadInputs.append(("axis", "ax0")) }
    ops += lenF(3, op(opType, name: "cad", inputs: cadInputs,
                      outName: "wd", outType: .fp16, outShape: [N, K, 1, 1]))

    var convX = "x"
    if quantAct {
        ops += lenF(3, constScalarOp(name: "as", dt: .fp16, payload: f16(1, 0.02)))
        ops += lenF(3, constScalarOp(name: "az", dt: .int8, payload: i8v([0])))
        ops += lenF(3, constStringOp(name: "od", value: "int8"))
        ops += lenF(3, op("quantize", name: "q",
                          inputs: [("input", "x"), ("scale", "as"), ("zero_point", "az"),
                                   ("output_dtype", "od")],
                          outName: "xq", outType: .int8, outShape: [1, K, 1, M]))
        ops += lenF(3, op("dequantize", name: "dq",
                          inputs: [("input", "xq"), ("scale", "as"), ("zero_point", "az")],
                          outName: "xd", outType: .fp16, outShape: [1, K, 1, M]))
        convX = "xd"
    }
    ops += lenF(3, op("conv", name: "cv",
                      inputs: [("x", convX), ("weight", "wd"), ("strides", "st"),
                               ("pad_type", "pt"), ("pad", "pd"),
                               ("dilations", "dl"), ("groups", "gp")],
                      outName: "y", outType: .fp16, outShape: [1, N, 1, M]))
    return wrap(ops, inDt: .fp16, inShape: [1, K, 1, M], outDt: .fp16, outShape: [1, N, 1, M],
                inFeat: 65552, outFeat: 65552)
}

@available(macOS 15.0, *)
func probe(_ name: String, _ spec: Data) async {
    do {
        let asset = try MLModelAsset(specification: spec)
        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
        let plan = try await MLComputePlan.load(asset: asset, configuration: cfg)
        var devices: [String] = []
        if case .program(let p) = plan.modelStructure, let f = p.functions["main"] {
            for o in f.block.operations where !o.operatorName.hasSuffix("const") {
                let d = String(describing: plan.deviceUsage(for: o)?.preferred)
                let short = d.contains("NeuralEngine") ? "ANE" : (d.contains("CPU") ? "CPU" : (d.contains("GPU") ? "GPU" : "?"))
                devices.append("\(o.operatorName)=\(short)")
            }
        }
        print("ACCEPTED  \(name)  \(devices.joined(separator: " "))")
    } catch {
        let msg = "\(error)".replacingOccurrences(of: "\n", with: " ")
        print("REJECTED  \(name)  \(String(msg.prefix(200)))")
    }
}

@available(macOS 15.0, *)
func run() async {
    setvbuf(stdout, nil, _IONBF, 0)
    await probe("matmul int8 x int8 -> int8", caseIntMatmul(outDt: .int8, outFeat: 131104))
    await probe("conv   int8 x int8 -> int8", caseInt8Conv())
    await probe("conv_quantized fp16 x uint8", caseConvQuantized())
    // Scalar quant params are spelled rank-0 by coremltools; try rank-1 too,
    // because this protobuf is hand-authored and the shape encoding is the
    // most likely thing to be wrong about an otherwise correct spelling.
    await probe("CONTROL fp16 conv, no quant  ", caseFP16ConvControl())
    await probe("W8A8 per-channel + axis      ", caseW8A8PerChannel())
    await probe("W8A16 constexpr weight       ", caseConstexprWeightOnly())
    await probe("W8A8  constexpr w + q/dq act ", caseConstexprW8A8())
    await probe("cbss data/scale/offset [N,1,1,1]", caseConstexprNamed(
        "constexpr_blockwise_shift_scale", dataParam: "data", zpParam: "offset",
        scaleShape: [N, 1, 1, 1], withAxis: false, quantAct: false))
    await probe("cbss data/scale only  [N,1,1,1]", caseConstexprNamed(
        "constexpr_blockwise_shift_scale", dataParam: "data", zpParam: nil,
        scaleShape: [N, 1, 1, 1], withAxis: false, quantAct: false))
    await probe("W8A8 cbss w + q/dq act        ", caseConstexprNamed(
        "constexpr_blockwise_shift_scale", dataParam: "data", zpParam: "offset",
        scaleShape: [N, 1, 1, 1], withAxis: false, quantAct: true))
    await probe("cad  data/scale/zp/axis  [N]   ", caseConstexprNamed(
        "constexpr_affine_dequantize", dataParam: "data", zpParam: "zero_point",
        scaleShape: [N], withAxis: true, quantAct: false))
    for shape in [[Int](), [1]] {
        let tag = shape.isEmpty ? "scalar" : "rank1"
        await probe("quantize only            (\(tag))", caseQuantizeOnly(scaleShape: shape))
        await probe("W8A8  q->dq->conv        (\(tag))", caseW8A8Conv(scaleShape: shape))
        await probe("W8A16 weight dq only     (\(tag))", caseW8A16Conv(scaleShape: shape))
    }
}
if #available(macOS 15.0, *) { await run() } else { exit(3) }
