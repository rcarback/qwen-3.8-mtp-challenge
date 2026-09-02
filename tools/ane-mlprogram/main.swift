// ML Program (MIL) authored in pure Swift, to keep weights QUANTIZED at
// runtime instead of dequantizing to fp16.
//
// Why this exists: the NeuralNetwork format dequantizes weights at model LOAD,
// so int8 and fp16 measured identically (26.28 vs 26.30 ms at N=12288) and
// int8 bought nothing but a smaller file. E1/E2/E3 saw int8 run 1.6x faster,
// and their traces show why: `ios18.constexpr_blockwise_shift_scale`, an ML
// Program op that dequantizes at RUNTIME.
//
// Program shape:
//   const w_data  : int8  [N, K]
//   const w_scale : fp16  [N, K/64]
//   const w_offset: int8  [N, K/64]
//   w = constexpr_blockwise_shift_scale(data, scale, offset)   -> fp16 [N, K]
//   y = matmul(x, w, transpose_y=true)                         -> fp16 [M, N]
//
// scale*(data-offset) is exactly our affine group-64 form with
// offset = -bias/scale.
//
// env: MLP_M MLP_N MLP_K MLP_UNITS=ane|gpu|cpu MLP_REPS
import CoreML
import Foundation

func env(_ k: String, _ d: Int) -> Int { Int(ProcessInfo.processInfo.environment[k] ?? "") ?? d }

func halfBytes(_ v: Float) -> Data {
    var d = Data(); let h = Float16(v)
    withUnsafeBytes(of: h) { d.append(contentsOf: $0) }
    return d
}

func buildProgram(M: Int, N: Int, K: Int, group: Int, w8a8: Bool,
                  data: Data, scale: Data, offset: Data) -> Data {
    let gk = K / group
    var ops = Data()
    ops += lenF(3, constOp(name: "w_data", dt: .int8, shape: [N, K], payload: data))
    ops += lenF(3, constOp(name: "w_scale", dt: .fp16, shape: [N, gk], payload: scale))
    ops += lenF(3, constOp(name: "w_offset", dt: .int8, shape: [N, gk], payload: offset))
    if w8a8 {
        // Canonical fake-quant: BOTH operands arrive through dequantize,
        // which is the pattern a compiler fuses into an int8 matmul.
        ops += lenF(3, constScalarOp(name: "ws", dt: .fp16, payload: halfBytes(0.0004)))
        ops += lenF(3, constScalarOp(name: "wz", dt: .int8, payload: Data([0])))
        ops += lenF(3, op("dequantize", name: "dqw",
                          inputs: [("input", "w_data"), ("scale", "ws"), ("zero_point", "wz")],
                          outName: "w", outType: .fp16, outShape: [N, K]))
    } else {
    ops += lenF(3, op("constexpr_blockwise_shift_scale", name: "dq",
                      inputs: [("data", "w_data"), ("scale", "w_scale"), ("offset", "w_offset")],
                      outName: "w", outType: .fp16, outShape: [N, K]))
    }
    ops += lenF(3, constBoolOp(name: "tx", value: false))
    ops += lenF(3, constBoolOp(name: "ty", value: true))
    if w8a8 {
        // Documented W8A8 pattern: the op must be SURROUNDED by a
        // quantize/dequantize pair before and after, which Core ML
        // then fuses into an int8 compute op.
        ops += lenF(3, constScalarOp(name: "qs", dt: .fp16, payload: halfBytes(0.02)))
        ops += lenF(3, constScalarOp(name: "qz", dt: .int8, payload: Data([0])))
        ops += lenF(3, constStringOp(name: "odt", value: "int8"))
        ops += lenF(3, op("quantize", name: "qx",
                          inputs: [("input", "x"), ("scale", "qs"), ("zero_point", "qz"), ("output_dtype", "odt")],
                          outName: "x_q", outType: .int8, outShape: [M, K]))
        ops += lenF(3, op("dequantize", name: "dqx",
                          inputs: [("input", "x_q"), ("scale", "qs"), ("zero_point", "qz")],
                          outName: "x_d", outType: .fp16, outShape: [M, K]))
        ops += lenF(3, op("matmul", name: "mm",
                          inputs: [("x", "x_d"), ("y", "w"), ("transpose_x", "tx"), ("transpose_y", "ty")],
                          outName: "y0", outType: .fp16, outShape: [M, N]))
        ops += lenF(3, op("quantize", name: "qy",
                          inputs: [("input", "y0"), ("scale", "qs"), ("zero_point", "qz"), ("output_dtype", "odt")],
                          outName: "y_q", outType: .int8, outShape: [M, N]))
        ops += lenF(3, op("dequantize", name: "dqy",
                          inputs: [("input", "y_q"), ("scale", "qs"), ("zero_point", "qz")],
                          outName: "y", outType: .fp16, outShape: [M, N]))
    } else {
    ops += lenF(3, op("matmul", name: "mm",
                      inputs: [("x", "x"), ("y", "w"), ("transpose_x", "tx"), ("transpose_y", "ty")],
                      outName: "y", outType: .fp16, outShape: [M, N]))
    }

    // Block{inputs=1, outputs=2, operations=3}
    let block = strF(2, "y") + ops
    // Function{inputs=1, opset=2, block_specializations=3}
    let fn = lenF(1, namedValue("x", .fp16, [M, K])) + strF(2, "CoreML8")
        + mapEntry(3, key: "CoreML8", value: block)
    // Program{version=1, functions=2}
    let program = varF(1, 1) + mapEntry(2, key: "main", value: fn)

    // Model{specificationVersion=1, description=2, mlProgram=502}
    let desc = lenF(1, featureDesc("x", [M, K])) + lenF(10, featureDesc("y", [M, N]))
    return varF(1, 9) + lenF(2, desc) + lenF(502, program)
}

/// Model-level FeatureDescription with a FLOAT16 multiarray.
func featureDesc(_ name: String, _ shape: [Int]) -> Data {
    let arr = lenF(1, shape.reduce(Data()) { $0 + varint(UInt64($1)) }) + varF(2, 65552) // FLOAT16
    return strF(1, name) + lenF(3, lenF(5, arr))
}

@available(macOS 15.0, *)
func run() async {
    setvbuf(stdout, nil, _IONBF, 0)
    let M = env("MLP_M", 1024), N = env("MLP_N", 8192), K = env("MLP_K", 5120)
    let group = 64, gk = K / group

    var data = Data(count: N * K)
    data.withUnsafeMutableBytes { r in
        let p = r.bindMemory(to: Int8.self)
        for i in 0 ..< N * K { p[i] = Int8.random(in: -127 ... 127) }
    }
    var scale = Data(count: N * gk * 2)
    scale.withUnsafeMutableBytes { r in
        let p = r.bindMemory(to: Float16.self)
        for i in 0 ..< N * gk { p[i] = Float16(0.0004) }
    }
    var offset = Data(count: N * gk)
    offset.withUnsafeMutableBytes { r in
        let p = r.bindMemory(to: Int8.self)
        for i in 0 ..< N * gk { p[i] = 0 }
    }

    let w8a8 = ProcessInfo.processInfo.environment["MLP_W8A8"] == "1"
    let spec = buildProgram(M: M, N: N, K: K, group: group, w8a8: w8a8,
                            data: data, scale: scale, offset: offset)
    print("spec bytes: \(spec.count)  (int8 weights = \(N*K) bytes)")

    let asset: MLModelAsset
    do { asset = try MLModelAsset(specification: spec) }
    catch { print("ASSET FAILED: \(error)"); exit(1) }
    print("asset ok")

    let cfg = MLModelConfiguration()
    switch ProcessInfo.processInfo.environment["MLP_UNITS"] ?? "ane" {
    case "gpu": cfg.computeUnits = .cpuAndGPU
    case "cpu": cfg.computeUnits = .cpuOnly
    default: cfg.computeUnits = .cpuAndNeuralEngine
    }

    do {
        let plan = try await MLComputePlan.load(asset: asset, configuration: cfg)
        if case .program(let p) = plan.modelStructure, let f = p.functions["main"] {
            for o in f.block.operations {
                let u = plan.deviceUsage(for: o)
                print("op \(o.operatorName) -> \(String(describing: u?.preferred))")
            }
        } else { print("compute plan: structure not a program") }
    } catch { print("compute plan unavailable: \(error)") }

    let model: MLModel
    do { model = try await MLModel.load(asset: asset, configuration: cfg) }
    catch { print("LOAD FAILED: \(error)"); exit(1) }
    print("model loaded")

    guard let x = try? MLMultiArray(shape: [NSNumber(value: M), NSNumber(value: K)],
                                    dataType: .float16) else { exit(1) }
    x.withUnsafeMutableBytes { r, _ in
        let p = r.bindMemory(to: Float16.self)
        for i in 0 ..< M * K { p[i] = Float16(0.01) }
    }
    let input = try! MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: x)])
    do {
        _ = try await model.prediction(from: input)
        var best = Double.infinity
        for _ in 0 ..< env("MLP_REPS", 5) {
            let t0 = Date()
            _ = try await model.prediction(from: input)
            best = min(best, Date().timeIntervalSince(t0))
        }
        let fl = 2.0 * Double(M) * Double(N) * Double(K)
        print(String(format: "MLPPOINT\t%d\t%d\t%d\t%.4f\t%.3f", M, N, K, 1000*best, fl/best/1e12))
    } catch { print("PREDICT FAILED: \(error)"); exit(1) }
}

if #available(macOS 15.0, *) { await run() } else { exit(3) }
