// 1x1 convolution on the ANE, in (B, C, 1, S) layout, optionally W8A8.
//
// Why conv and not matmul: every int8 fast-path reference is convolution
// shaped. The Bryngelson paper measures int8 on a 3x3 conv; Apple's ANE
// guidance says to express linear layers as 1x1 convs because convolution is
// the hardware's primary primitive; Apple's own W8A8 benchmark is ResNet50.
// Our earlier probes were all matmul/innerProduct, which the same paper
// reports running ~3x slower than the conv datapath.
//
// Why a STACK and not one op: coremltools issue 2432 raises the real
// possibility that Apple's W8A8 speedup comes from halved ACTIVATION traffic
// between layers rather than int8 arithmetic. A single op cannot show that, so
// a one-layer test could give a false negative.
//
// x: fp16 [1, C, 1, S]   weight: [C, C, 1, 1]   y: [1, C, 1, S]
// env: CV_C CV_S CV_LAYERS CV_W8A8=1 CV_UNITS=ane|gpu|cpu CV_REPS
import CoreML
import Foundation

func env(_ k: String, _ d: Int) -> Int { Int(ProcessInfo.processInfo.environment[k] ?? "") ?? d }
func halfBytes(_ v: Float) -> Data {
    var d = Data(); withUnsafeBytes(of: Float16(v)) { d.append(contentsOf: $0) }; return d
}
func featureDesc(_ name: String, _ shape: [Int]) -> Data {
    let arr = lenF(1, shape.reduce(Data()) { $0 + varint(UInt64($1)) }) + varF(2, 65552)
    return strF(1, name) + lenF(3, lenF(5, arr))
}

func buildProgram(C: Int, S: Int, layers: Int, w8a8: Bool, wBytes: Data) -> Data {
    var ops = Data()
    // conv hyper-parameters, shared by every layer
    ops += lenF(3, constIntsOp(name: "st", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "dl", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0, 0, 0, 0]))
    ops += lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    if w8a8 {
        ops += lenF(3, constScalarOp(name: "qs", dt: .fp16, payload: halfBytes(0.02)))
        ops += lenF(3, constScalarOp(name: "qz", dt: .int8, payload: Data([0])))
        ops += lenF(3, constStringOp(name: "odt", value: "int8"))
    }

    let tiles = env("CV_TILES", 0)
    let cin = env("CV_CIN", C)
    let cout = env("CV_COUT", C)
    if tiles > 0 {
        // Independent tiles of ONE projection, all reading x, outputs concatenated.
        // Each tile holds Cout x C weights; pick Cout so that stays on-chip.
        var names: [String] = []
        for t in 0 ..< tiles {
            ops += lenF(3, constOp(name: "wt\(t)", dt: .fp16, shape: [cout, cin, 1, 1], payload: wBytes))
            ops += lenF(3, op("conv", name: "cvt\(t)",
                              inputs: [("x", "x"), ("weight", "wt\(t)"), ("strides", "st"),
                                       ("pad_type", "pt"), ("pad", "pd"),
                                       ("dilations", "dl"), ("groups", "gp")],
                              outName: "t\(t)", outType: .fp16, outShape: [1, cout, 1, S]))
            names.append("t\(t)")
        }
        ops += lenF(3, constIntsOp(name: "ax", values: [1]))
        ops += lenF(3, constBoolOp(name: "il", value: false))
        var cc = strF(1, "concat")
        var bindings = Data()
        for n in names { bindings += lenF(1, strF(1, n)) }
        cc += mapEntry(2, key: "values", value: bindings)
        cc += mapEntry(2, key: "axis", value: lenF(1, strF(1, "ax")))
        cc += mapEntry(2, key: "interleave", value: lenF(1, strF(1, "il")))
        cc += lenF(3, namedValue("y", .fp16, [1, cout * tiles, 1, S]))
        cc += mapEntry(5, key: "name", value: stringValue("cat"))
        ops += lenF(3, cc)
        let block = strF(2, "y") + ops
        let fn = lenF(1, namedValue("x", .fp16, [1, cin, 1, S])) + strF(2, "CoreML8")
            + mapEntry(3, key: "CoreML8", value: block)
        let program = varF(1, 1) + mapEntry(2, key: "main", value: fn)
        func fdT(_ n: String, _ shape: [Int]) -> Data {
            let arr = lenF(1, shape.reduce(Data()) { $0 + varint(UInt64($1)) }) + varF(2, 65552)
            return strF(1, n) + lenF(3, lenF(5, arr))
        }
        let desc = lenF(1, fdT("x", [1, cin, 1, S])) + lenF(10, fdT("y", [1, cout * tiles, 1, S]))
        return varF(1, 9) + lenF(2, desc) + lenF(502, program)
    }
    var cur = "x"
    for l in 0 ..< layers {
        let wn = "w\(l)"
        if w8a8 {
            // int8 weight const -> dequantize, so both operands arrive through
            // the quantized path the compiler is meant to fuse.
            ops += lenF(3, constOp(name: "\(wn)_q", dt: .int8, shape: [C, C, 1, 1], payload: wBytes))
            ops += lenF(3, constScalarOp(name: "\(wn)s", dt: .fp16, payload: halfBytes(0.0004)))
            ops += lenF(3, constScalarOp(name: "\(wn)z", dt: .int8, payload: Data([0])))
            ops += lenF(3, op("dequantize", name: "dq\(wn)",
                              inputs: [("input", "\(wn)_q"), ("scale", "\(wn)s")],
                              outName: wn, outType: .fp16, outShape: [C, C, 1, 1]))
            // quantize/dequantize pair on the activation
            ops += lenF(3, op("quantize", name: "qx\(l)",
                              inputs: [("input", cur), ("scale", "qs"), ("output_dtype", "odt")],
                              outName: "xq\(l)", outType: .int8, outShape: [1, C, 1, S]))
            ops += lenF(3, op("dequantize", name: "dqx\(l)",
                              inputs: [("input", "xq\(l)"), ("scale", "qs")],
                              outName: "xd\(l)", outType: .fp16, outShape: [1, C, 1, S]))
            cur = "xd\(l)"
        } else {
            ops += lenF(3, constOp(name: wn, dt: .fp16, shape: [C, C, 1, 1], payload: wBytes))
        }
        let outName = (l == layers - 1) ? "y" : "c\(l)"
        ops += lenF(3, op("conv", name: "cv\(l)",
                          inputs: [("x", cur), ("weight", wn), ("strides", "st"),
                                   ("pad_type", "pt"), ("pad", "pd"),
                                   ("dilations", "dl"), ("groups", "gp")],
                          outName: outName, outType: .fp16, outShape: [1, C, 1, S]))
        cur = outName
    }

    let block = strF(2, "y") + ops
    let fn = lenF(1, namedValue("x", .fp16, [1, C, 1, S])) + strF(2, "CoreML8")
        + mapEntry(3, key: "CoreML8", value: block)
    let program = varF(1, 1) + mapEntry(2, key: "main", value: fn)
    let desc = lenF(1, featureDesc("x", [1, C, 1, S])) + lenF(10, featureDesc("y", [1, C, 1, S]))
    return varF(1, 9) + lenF(2, desc) + lenF(502, program)
}

@available(macOS 15.0, *)
func run() async {
    setvbuf(stdout, nil, _IONBF, 0)
    let C = env("CV_C", 2048), S = env("CV_S", 1024), layers = env("CV_LAYERS", 1)
    let w8a8 = ProcessInfo.processInfo.environment["CV_W8A8"] == "1"

    let cinE = env("CV_CIN", C), coutE = env("CV_COUT", C)
    let wCount = (env("CV_TILES", 0) > 0) ? cinE * coutE : C * C
    var w = Data(count: wCount * (w8a8 ? 1 : 2))
    w.withUnsafeMutableBytes { r in
        if w8a8 { let p = r.bindMemory(to: Int8.self); for i in 0 ..< C*C { p[i] = Int8.random(in: -127...127) } }
        else { let p = r.bindMemory(to: Float16.self); for i in 0 ..< wCount { p[i] = Float16(Float.random(in: -0.02...0.02)) } }
    }
    let spec = buildProgram(C: C, S: S, layers: layers, w8a8: w8a8, wBytes: w)
    print("spec bytes: \(spec.count)  C=\(C) S=\(S) layers=\(layers) w8a8=\(w8a8)")

    let asset: MLModelAsset
    do { asset = try MLModelAsset(specification: spec) } catch { print("ASSET FAILED: \(error)"); exit(1) }

    let cfg = MLModelConfiguration()
    switch ProcessInfo.processInfo.environment["CV_UNITS"] ?? "ane" {
    case "gpu": cfg.computeUnits = .cpuAndGPU
    case "cpu": cfg.computeUnits = .cpuOnly
    default: cfg.computeUnits = .cpuAndNeuralEngine
    }
    var aneCount = 0, cpuCount = 0, gpuCount = 0
    do {
        let plan = try await MLComputePlan.load(asset: asset, configuration: cfg)
        if case .program(let p) = plan.modelStructure, let f = p.functions["main"] {
            for o in f.block.operations {
                guard let u = plan.deviceUsage(for: o) else { continue }
                let d = String(describing: u.preferred)
                if d.contains("NeuralEngine") { aneCount += 1 }
                else if d.contains("CPU") { cpuCount += 1 } else { gpuCount += 1 }
                if o.operatorName.contains("conv") { print("  \(o.operatorName) -> \(d)") }
            }
        }
    } catch { print("plan unavailable: \(error)") }

    let model: MLModel
    do { model = try await MLModel.load(asset: asset, configuration: cfg) } catch { print("LOAD FAILED: \(error)"); exit(1) }
    let xC = (env("CV_TILES", 0) > 0) ? cinE : C
    guard let x = try? MLMultiArray(shape: [1, xC, 1, S].map { NSNumber(value: $0) }, dataType: .float16) else { exit(1) }
    x.withUnsafeMutableBytes { r, _ in let p = r.bindMemory(to: Float16.self); for i in 0 ..< xC*S { p[i] = Float16(0.01) } }
    let input = try! MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: x)])
    do {
        _ = try await model.prediction(from: input)
        var best = Double.infinity
        for _ in 0 ..< env("CV_REPS", 5) {
            let t0 = Date(); _ = try await model.prediction(from: input)
            best = min(best, Date().timeIntervalSince(t0))
        }
        let fl = 2.0 * Double(S) * Double(C) * Double(C) * Double(layers)
        print(String(format: "CONVPOINT\t%@\t%d\t%d\t%d\tANE=%d/CPU=%d/GPU=%d\t%.4f\t%.3f",
                     w8a8 ? "w8a8" : "fp16", C, S, layers, aneCount, cpuCount, gpuCount,
                     1000*best, fl/best/1e12))
    } catch { print("PREDICT FAILED: \(error)"); exit(1) }
}
if #available(macOS 15.0, *) { await run() } else { exit(3) }
