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
}
if #available(macOS 15.0, *) { await run() } else { exit(3) }
