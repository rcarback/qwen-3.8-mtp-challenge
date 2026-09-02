// Full gated-delta (linear_attention) Qwen 3.8 layer at real geometry as one
// ANE MIL program: rmsnorm -> in_proj (qkv/z/b/a) -> depthwise causal conv4
// -> silu -> chunkwise gated-delta recurrence (8 chunks x 64) -> gated norm
// * silu(z) -> out_proj -> residual -> rmsnorm -> SwiGLU MLP -> residual.
// Random proxy weights; structure/placement/timing are what is measured.
import CoreML
import Foundation

func convW(_ nm: String, _ inV: String, _ w: String, _ out: String, _ Cout: Int, _ S: Int) -> Data {
    lenF(3, op("conv", name: nm, inputs: [("x",inV),("weight",w),("strides","st"),
        ("pad_type","pt"),("pad","pd"),("dilations","dl"),("groups","gp")],
        outName: out, outType: .fp16, outShape: [1,Cout,1,S]))
}
func rms4(_ pfx: String, _ inV: String, _ out: String, _ C: Int, _ S: Int) -> Data {
    var d = Data()
    d += ew("mul", "\(pfx)sq", inV, inV, out: "\(pfx)sq", shape: [1,C,1,S])
    d += lenF(3, constIntsVecOp(name: "\(pfx)ax", values: [1]))
    d += lenF(3, op("reduce_mean", name: "\(pfx)mn", inputs: [("x","\(pfx)sq"),("axes","\(pfx)ax"),("keep_dims","g_t")], outName: "\(pfx)mn", outType: .fp16, outShape: [1,1,1,S]))
    d += lenF(3, op("rsqrt", name: "\(pfx)rs", inputs: [("x","\(pfx)mn"),("epsilon","g_eps")], outName: "\(pfx)rs", outType: .fp16, outShape: [1,1,1,S]))
    d += ew("mul", "\(pfx)o", inV, "\(pfx)rs", out: out, shape: [1,C,1,S])
    return d
}
func reshapeTo(_ nm: String, _ x: String, _ shape: [Int], out: String) -> Data {
    lenF(3, constIntsOp(name: "\(nm)_s", values: shape))
        + lenF(3, op("reshape", name: nm, inputs: [("x",x),("shape","\(nm)_s")], outName: out, outType: .fp16, outShape: shape))
}
func transposeTo(_ nm: String, _ x: String, _ perm: [Int], outShape: [Int], out: String) -> Data {
    lenF(3, constIntsOp(name: "\(nm)_p", values: perm))
        + lenF(3, op("transpose", name: nm, inputs: [("x",x),("perm","\(nm)_p")], outName: out, outType: .fp16, outShape: outShape))
}
func slice4(_ nm: String, _ x: String, begin: [Int], end: [Int], outShape: [Int], out: String) -> Data {
    lenF(3, constIntsVecOp(name: "\(nm)_b", values: begin))
        + lenF(3, constIntsVecOp(name: "\(nm)_e", values: end))
        + lenF(3, constIntsVecOp(name: "\(nm)_t", values: [Int](repeating: 1, count: begin.count)))
        + lenF(3, op("slice_by_index", name: nm, inputs: [("x",x),("begin","\(nm)_b"),("end","\(nm)_e"),("stride","\(nm)_t")], outName: out, outType: .fp16, outShape: outShape))
}

@available(macOS 15.0, *)
func runLayerBench() async {
    let S = Int(ProcessInfo.processInfo.environment["GDL_S"] ?? "512")!
    let CL = Int(ProcessInfo.processInfo.environment["GDL_CL"] ?? "64")!   // chunk length
    let NC = S / CL
    let C = 5120, NKH = 16, NVH = 48, D = 128
    let Cq = NKH*D, Cv = NVH*D            // 2048, 6144
    let Cqkv = 2*Cq + Cv                  // 10240
    let inter = 17408

    var ops = Data()
    // shared hyper consts
    ops += lenF(3, constIntsOp(name: "st", values: [1,1])) + lenF(3, constIntsOp(name: "dl", values: [1,1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0,0,0,0])) + lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    ops += lenF(3, constIntsOp(name: "dpd", values: [0,0,3,0])) + lenF(3, constIntsOp(name: "dgp", values: [Cqkv]))
    ops += lenF(3, constStringOp(name: "dpt", value: "custom"))
    ops += lenF(3, constBoolOp(name: "g_f", value: false)) + lenF(3, constBoolOp(name: "g_t", value: true))
    ops += lenF(3, constScalarOp(name: "g_eps", dt: .fp16, payload: Data.f16(1, 1e-6)))
    ops += lenF(3, constScalarOp(name: "g_neg1", dt: .fp16, payload: Data.f16(1, -1)))
    ops += lenF(3, constScalarOp(name: "g_dk", dt: .fp16, payload: Data.f16(1, -0.1)))

    // 1. pre-norm + projections
    ops += rms4("n1", "x", "nx", C, S)
    ops += lenF(3, constOp(name: "wqkv", dt: .fp16, shape: [Cqkv,C,1,1], payload: weightData("wqkv", Cqkv*C, scale: 0.001)))
    ops += convW("pqkv", "nx", "wqkv", "qkv", Cqkv, S)
    ops += lenF(3, constOp(name: "wz", dt: .fp16, shape: [Cv,C,1,1], payload: weightData("wz", Cv*C, scale: 0.005)))
    ops += convW("pz", "nx", "wz", "zz", Cv, S)
    ops += lenF(3, constOp(name: "wb", dt: .fp16, shape: [NVH,C,1,1], payload: weightData("wb", NVH*C, scale: 0.005)))
    ops += convW("pb", "nx", "wb", "bb", NVH, S)
    ops += lenF(3, constOp(name: "wa", dt: .fp16, shape: [NVH,C,1,1], payload: weightData("wa", NVH*C, scale: 0.005)))
    ops += convW("pa", "nx", "wa", "aa", NVH, S)
    // 2. depthwise causal conv4 + silu on qkv stream
    ops += lenF(3, constOp(name: "wdw", dt: .fp16, shape: [Cqkv,1,1,4], payload: weightData("wdw", Cqkv*4, scale: 0.25)))
    ops += lenF(3, op("conv", name: "dw", inputs: [("x","qkv"),("weight","wdw"),("strides","st"),
        ("pad_type","dpt"),("pad","dpd"),("dilations","dl"),("groups","dgp")],
        outName: "qkvc", outType: .fp16, outShape: [1,Cqkv,1,S]))
    ops += un("silu", "dwsi", "qkvc", out: "qkvs", shape: [1,Cqkv,1,S])
    // 3. beta = sigmoid(b); log g = -0.1*sigmoid(a)  (decay proxy, same op shapes)
    ops += un("sigmoid", "sgb", "bb", out: "beta4", shape: [1,NVH,1,S])
    ops += un("sigmoid", "sga", "aa", out: "asig", shape: [1,NVH,1,S])
    ops += ew("mul", "lgm", "asig", "g_dk", out: "logg4", shape: [1,NVH,1,S])
    // 4. split q/k/v and lay out as [NC, NVH, CL, D] chunk-major
    ops += slice4("slq", "qkvs", begin: [0,0,0,0], end: [1,Cq,1,S], outShape: [1,Cq,1,S], out: "qs")
    ops += slice4("slk", "qkvs", begin: [0,Cq,0,0], end: [1,2*Cq,1,S], outShape: [1,Cq,1,S], out: "ks")
    ops += slice4("slv", "qkvs", begin: [0,2*Cq,0,0], end: [1,Cqkv,1,S], outShape: [1,Cv,1,S], out: "vs")
    ops += lenF(3, constIntsOp(name: "cc_ax", values: [1]))
    // q,k: [1,Cq,1,S] -> [NKH,D,NC,CL] -> perm [2,0,3,1] -> [NC,NKH,CL,D], tile x3 heads
    for (nm, src) in [("q","qs"),("k","ks")] {
        ops += reshapeTo("\(nm)r1", src, [NKH,D,NC,CL], out: "\(nm)r1")
        ops += transposeTo("\(nm)t1", "\(nm)r1", [2,0,3,1], outShape: [NC,NKH,CL,D], out: "\(nm)h16")
        // tile to 48 v-heads: concat 3 copies on axis 1
        var d = strF(1, "concat")
        d += inputBindingMulti(2, param: "values", varNames: ["\(nm)h16","\(nm)h16","\(nm)h16"])
        d += inputBinding(2, param: "axis", varName: "cc_ax")
        d += inputBinding(2, param: "interleave", varName: "g_f")
        d += lenF(3, namedValue("\(nm)h48", .fp16, [NC,NVH,CL,D]))
        d += mapEntry(5, key: "name", value: stringValue("\(nm)cc"))
        ops += lenF(3, d)
    }
    ops += reshapeTo("vr1", "vs", [NVH,D,NC,CL], out: "vr1")
    ops += transposeTo("vt1", "vr1", [2,0,3,1], outShape: [NC,NVH,CL,D], out: "vh48")
    // lg/bt: [1,NVH,1,S] -> [NVH,NC,CL] -> perm [1,0,2] -> [NC,NVH,CL]
    ops += reshapeTo("lgr", "logg4", [NVH,NC,CL], out: "lgr")
    ops += transposeTo("lgt", "lgr", [1,0,2], outShape: [NC,NVH,CL], out: "lgc")
    ops += reshapeTo("btr", "beta4", [NVH,NC,CL], out: "btr")
    ops += transposeTo("btt", "btr", [1,0,2], outShape: [NC,NVH,CL], out: "btc")
    var hCur = "h0"
    let noRec = (ProcessInfo.processInfo.environment["GDL_NOREC"] ?? "0") == "1"
    // 5. per-chunk recurrence
    if noRec {
        ops += un("silu", "bypass", "vs", out: "ostream", shape: [1,Cv,1,S])
    } else {
    // 5. per-chunk recurrence
    hCur = "h0"
    var oNames: [String] = []
    for c in 0..<NC {
        for (nm, src, cs) in [("q","qh48",[NC,NVH,CL,D]), ("k","kh48",[NC,NVH,CL,D]), ("v","vh48",[NC,NVH,CL,D])] {
            ops += slice4("c\(c)s\(nm)", src, begin: [c,0,0,0], end: [c+1,cs[1],cs[2],cs[3]], outShape: [1,cs[1],cs[2],cs[3]], out: "c\(c)\(nm)4")
            ops += reshapeTo("c\(c)r\(nm)", "c\(c)\(nm)4", [NVH,CL,D], out: "c\(c)\(nm)")
        }
        for (nm, src) in [("lg","lgc"), ("bt","btc")] {
            ops += slice4("c\(c)s\(nm)", src, begin: [c,0,0], end: [c+1,NVH,CL], outShape: [1,NVH,CL], out: "c\(c)\(nm)3")
            ops += reshapeTo("c\(c)r\(nm)", "c\(c)\(nm)3", [NVH,CL,1], out: "c\(c)\(nm)")
        }
        let hOut = "hst\(c+1)"
        ops += gdChunk2(p: "gd\(c)_", q: "c\(c)q", k: "c\(c)k", v: "c\(c)v", lg: "c\(c)lg", bt: "c\(c)bt",
                        hin: hCur, oOut: "od\(c)", hOut: hOut,
                        eye: "eye", trilS: "trilS", trilI: "trilI",
                        vb: [NVH], L: CL, D: D)
        hCur = hOut
        ops += reshapeTo("c\(c)or", "od\(c)", [1,NVH,CL,D], out: "od4_\(c)")
        oNames.append("od4_\(c)")
    }
    // 6. stitch chunks back to [1,Cv,1,S]
    do {
        var d = strF(1, "concat")
        d += inputBindingMulti(2, param: "values", varNames: oNames)
        d += inputBinding(2, param: "axis", varName: "cc0_ax")
        d += inputBinding(2, param: "interleave", varName: "g_f")
        d += lenF(3, namedValue("oall", .fp16, [NC,NVH,CL,D]))
        d += mapEntry(5, key: "name", value: stringValue("occ"))
        ops += lenF(3, constIntsOp(name: "cc0_ax", values: [0])) + lenF(3, d)
    }
    ops += transposeTo("ot1", "oall", [1,3,0,2], outShape: [NVH,D,NC,CL], out: "ot1")
    ops += reshapeTo("or1", "ot1", [1,Cv,1,S], out: "ostream")
    }
    // 7. gated norm * silu(z), out_proj, residual
    ops += rms4("gn", "ostream", "onorm", Cv, S)
    ops += un("silu", "zsi", "zz", out: "zs", shape: [1,Cv,1,S])
    ops += ew("mul", "gmul", "onorm", "zs", out: "ogated", shape: [1,Cv,1,S])
    ops += lenF(3, constOp(name: "wo", dt: .fp16, shape: [C,Cv,1,1], payload: weightData("wo", C*Cv, scale: 0.005)))
    ops += convW("po", "ogated", "wo", "oproj", C, S)
    ops += ew("add", "res1", "x", "oproj", out: "h1", shape: [1,C,1,S])
    // 8. MLP
    ops += rms4("n2", "h1", "mx", C, S)
    ops += lenF(3, constOp(name: "wg", dt: .fp16, shape: [inter,C,1,1], payload: weightData("wg", inter*C, scale: 0.005)))
    ops += convW("pg", "mx", "wg", "gg", inter, S)
    ops += lenF(3, constOp(name: "wu", dt: .fp16, shape: [inter,C,1,1], payload: weightData("wu", inter*C, scale: 0.005)))
    ops += convW("pu", "mx", "wu", "uu", inter, S)
    ops += un("silu", "msi", "gg", out: "sg", shape: [1,inter,1,S])
    ops += ew("mul", "mgm", "sg", "uu", out: "gu", shape: [1,inter,1,S])
    ops += lenF(3, constOp(name: "wd", dt: .fp16, shape: [C,inter,1,1], payload: weightData("wd", C*inter, scale: 0.005)))
    ops += convW("pdn", "gu", "wd", "dd", C, S)
    ops += ew("add", "res2", "h1", "dd", out: "y", shape: [1,C,1,S])

    let ins: [(String,[Int])] = noRec ? [("x",[1,C,1,S])] : [("x",[1,C,1,S]),("h0",[NVH,D,D]),
        ("eye",[1,CL,CL]),("trilS",[NVH,CL,CL]),("trilI",[NVH,CL,CL])]
    let outs: [(String,[Int])] = noRec ? [("y",[1,C,1,S])] : [("y",[1,C,1,S]),(hCur, [NVH,D,D])]
    let spec = buildSpec(inputs: ins, outputs: outs, ops: ops)
    print("layer: S=\(S) CL=\(CL) NC=\(NC) spec=\(Double(spec.count)/1e6) MB")
    let t0 = Date()
    let asset: MLModelAsset
    do { asset = try MLModelAsset(specification: spec) }
    catch { print("REJECTED: \(error)"); exit(1) }
    do {
        let r = try await planPlacement(asset, verbose: false)
        print(String(format: "placement: ANE=%d CPU=%d GPU=%d  (plan %.1fs)", r.ane, r.cpu, r.gpu, Date().timeIntervalSince(t0)))
        if r.cpu + r.gpu > 0 { for l in r.lines where !l.hasSuffix("ANE") { print(l) } }
    } catch { print("plan failed: \(error)") }
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
    let model: MLModel
    let t1 = Date()
    do { model = try await MLModel.load(asset: asset, configuration: cfg) }
    catch { print("LOAD FAILED \(error)"); exit(1) }
    print(String(format: "load %.1fs", Date().timeIntervalSince(t1)))
    var seed: UInt64 = 7
    var feats: [String: MLFeatureValue] = [:]
    for (n, s) in ins {
        let a: MLMultiArray
        switch n {
        case "eye", "trilS", "trilI":
            a = try! MLMultiArray(shape: s.map{NSNumber(value:$0)}, dataType: .float16)
            a.withUnsafeMutableBytes { r,_ in
                let pp = r.bindMemory(to: Float16.self)
                for bb in 0..<s[0] { for i in 0..<CL { for j in 0..<CL {
                    let on: Bool = n == "eye" ? i == j : (n == "trilS" ? i > j : i >= j)
                    pp[bb*CL*CL + i*CL + j] = on ? 1 : 0
                } } }
            }
        case "h0":
            a = try! MLMultiArray(shape: s.map{NSNumber(value:$0)}, dataType: .float16)
            a.withUnsafeMutableBytes { r,_ in let pp = r.bindMemory(to: Float16.self); for i in 0..<NVH*D*D { pp[i]=0 } }
        default:
            a = randF16Array(shape: s, range: -0.5 ... 0.5, seed: &seed)
        }
        feats[n] = MLFeatureValue(multiArray: a)
    }
    let input = try! MLDictionaryFeatureProvider(dictionary: feats)
    guard let out0 = try? await model.prediction(from: input) else { print("PREDICT FAILED"); exit(1) }
    if let dump = ProcessInfo.processInfo.environment["GDL_DUMPY"] {
        let ya = out0.featureValue(for: "y")!.multiArrayValue!
        var dat = Data()
        ya.withUnsafeBytes { r in dat.append(contentsOf: r) }
        try? dat.write(to: URL(fileURLWithPath: dump))
        print("dumped y (\(dat.count) B) to \(dump)")
    }
    var best = Double.infinity
    let reps = Int(ProcessInfo.processInfo.environment["GDL_REPS"] ?? "8")!
    for _ in 0..<reps { let tt = Date(); _ = try? await model.prediction(from: input); best = min(best, Date().timeIntervalSince(tt)) }
    // FLOPs (matmul/conv only)
    let projF = 2.0*Double(C)*Double(Cqkv+Cv+2*NVH)*Double(S)
    let dwF   = 2.0*Double(Cqkv)*4*Double(S)
    let outF  = 2.0*Double(C)*Double(Cv)*Double(S)
    let mlpF  = 2.0*Double(C)*Double(inter)*3*Double(S)
    var iters = 0, span = 1
    while span < CL - 1 { span *= 2; iters += 1 }
    let cl = Double(CL), dd = Double(D)
    let perHeadChunk = 2*cl*cl*1 + 2*cl*cl*dd /*kk*/ + Double(iters)*2*2*cl*cl*cl /*T*/
        + 2*cl*dd*dd /*kh*/ + 2*cl*cl*dd /*U*/ + 2*cl*dd*dd /*qh*/ + 2*cl*cl*dd /*qk*/
        + 2*cl*cl*dd /*o2*/ + 2*dd*cl*dd /*su*/
    let recF = Double(NVH*NC) * perHeadChunk
    let total = projF + dwF + outF + mlpF + recF
    print(String(format: "LAYER ms=%.2f  total=%.1f GF (rec %.1f GF)  TFLOPS=%.2f",
                 best*1000, total/1e9, recF/1e9, total/best/1e12))
}
