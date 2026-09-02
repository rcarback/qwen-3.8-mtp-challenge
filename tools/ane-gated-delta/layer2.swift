// Tiled gated-delta layer: whole 512-token layer as one MIL graph, processed
// as NT sequence tiles of St tokens with the recurrence state threaded across
// tiles, and the SwiGLU intermediate split into K column blocks with an
// accumulated down projection. Goal: max live intermediate activation < 2 MB.
import CoreML
import Foundation

struct TiledBuild { let spec: Data; let ins: [(String,[Int])]; let yNames: [String]
                    let usefulGF: Double; let overheadGF: Double; let maxLiveMB: Double }

func buildTiledLayer(S: Int, St: Int, K: Int) -> TiledBuild {
    let NT = S / St
    let C = 5120, NKH = 16, NVH = 48, D = 128
    let Cq = NKH*D, Cv = NVH*D, Cqkv = 2*Cq + Cv
    let inter = 17408, IB = inter / K

    var ops = Data()
    // shared consts
    ops += lenF(3, constIntsOp(name: "st", values: [1,1])) + lenF(3, constIntsOp(name: "dl", values: [1,1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0,0,0,0])) + lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    ops += lenF(3, constIntsOp(name: "dpd0", values: [0,0,3,0])) + lenF(3, constIntsOp(name: "dgp", values: [Cqkv]))
    ops += lenF(3, constStringOp(name: "dpt0", value: "custom"))
    ops += lenF(3, constBoolOp(name: "g_f", value: false)) + lenF(3, constBoolOp(name: "g_t", value: true))
    ops += lenF(3, constScalarOp(name: "g_eps", dt: .fp16, payload: Data.f16(1, 1e-6)))
    ops += lenF(3, constScalarOp(name: "g_neg1", dt: .fp16, payload: Data.f16(1, -1)))
    ops += lenF(3, constScalarOp(name: "g_dk", dt: .fp16, payload: Data.f16(1, -0.1)))
    ops += lenF(3, constIntsOp(name: "cc_ax", values: [0]))
    // weights (same tags/values as monolithic layer.swift)
    ops += lenF(3, constOp(name: "wqkv", dt: .fp16, shape: [Cqkv,C,1,1], payload: weightData("wqkv", Cqkv*C, scale: 0.001)))
    ops += lenF(3, constOp(name: "wz", dt: .fp16, shape: [Cv,C,1,1], payload: weightData("wz", Cv*C, scale: 0.005)))
    ops += lenF(3, constOp(name: "wb", dt: .fp16, shape: [NVH,C,1,1], payload: weightData("wb", NVH*C, scale: 0.005)))
    ops += lenF(3, constOp(name: "wa", dt: .fp16, shape: [NVH,C,1,1], payload: weightData("wa", NVH*C, scale: 0.005)))
    ops += lenF(3, constOp(name: "wdw", dt: .fp16, shape: [Cqkv,1,1,4], payload: weightData("wdw", Cqkv*4, scale: 0.25)))
    ops += lenF(3, constOp(name: "wo", dt: .fp16, shape: [C,Cv,1,1], payload: weightData("wo", C*Cv, scale: 0.005)))
    // MLP blocks: row-slices of wg/wu, column-slices of wd
    let wgF = weightFloats("wg", inter*C, scale: 0.005)
    let wuF = weightFloats("wu", inter*C, scale: 0.005)
    let wdF = weightFloats("wd", C*inter, scale: 0.005)
    for k in 0..<K {
        ops += lenF(3, constOp(name: "wg\(k)", dt: .fp16, shape: [IB,C,1,1], payload: Data.f16Arr(Array(wgF[(k*IB*C)..<((k+1)*IB*C)]))))
        ops += lenF(3, constOp(name: "wu\(k)", dt: .fp16, shape: [IB,C,1,1], payload: Data.f16Arr(Array(wuF[(k*IB*C)..<((k+1)*IB*C)]))))
        var wdk = [Float](repeating: 0, count: C*IB)
        for r in 0..<C { for c in 0..<IB { wdk[r*IB+c] = wdF[r*inter + k*IB + c] } }
        ops += lenF(3, constOp(name: "wd\(k)", dt: .fp16, shape: [C,IB,1,1], payload: Data.f16Arr(wdk)))
    }

    var yNames: [String] = []
    var hCur = "h0"
    for t in 0..<NT {
        let p = "t\(t)_"
        let pad = t == 0 ? 0 : 3
        let W = St + pad
        // slice input tile (with 3-col causal-conv context for t>0)
        ops += slice4("\(p)sx", "x", begin: [0,0,0,t*St - pad], end: [1,C,1,(t+1)*St], outShape: [1,C,1,W], out: "\(p)xp")
        ops += rms4("\(p)n1", "\(p)xp", "\(p)nx", C, W)
        // qkv on padded width; z/b/a on the unpadded tail
        ops += convW("\(p)pq", "\(p)nx", "wqkv", "\(p)qkv", Cqkv, W)
        let nxs: String
        if pad > 0 {
            ops += slice4("\(p)snx", "\(p)nx", begin: [0,0,0,pad], end: [1,C,1,W], outShape: [1,C,1,St], out: "\(p)nxs")
            nxs = "\(p)nxs"
        } else { nxs = "\(p)nx" }
        ops += convW("\(p)pz", nxs, "wz", "\(p)zz", Cv, St)
        ops += convW("\(p)pb", nxs, "wb", "\(p)bb", NVH, St)
        ops += convW("\(p)pa", nxs, "wa", "\(p)aa", NVH, St)
        // depthwise causal conv (tile 0: left pad 3; else valid over the overlap)
        let cpt = t == 0 ? "dpt0" : "pt", cpd = t == 0 ? "dpd0" : "pd"
        ops += lenF(3, op("conv", name: "\(p)dw", inputs: [("x","\(p)qkv"),("weight","wdw"),("strides","st"),
            ("pad_type",cpt),("pad",cpd),("dilations","dl"),("groups","dgp")],
            outName: "\(p)qkvc", outType: .fp16, outShape: [1,Cqkv,1,St]))
        ops += un("silu", "\(p)dws", "\(p)qkvc", out: "\(p)qkvs", shape: [1,Cqkv,1,St])
        // beta / log g
        ops += un("sigmoid", "\(p)sgb", "\(p)bb", out: "\(p)bt4", shape: [1,NVH,1,St])
        ops += un("sigmoid", "\(p)sga", "\(p)aa", out: "\(p)as", shape: [1,NVH,1,St])
        ops += ew("mul", "\(p)lgm", "\(p)as", "g_dk", out: "\(p)lg4", shape: [1,NVH,1,St])
        ops += reshapeTo("\(p)btr", "\(p)bt4", [NVH,St,1], out: "\(p)bt")
        ops += reshapeTo("\(p)lgr", "\(p)lg4", [NVH,St,1], out: "\(p)lg")
        // q/k/v head tensors
        for (nm, lo, hi, nh) in [("q",0,Cq,NKH), ("k",Cq,2*Cq,NKH), ("v",2*Cq,Cqkv,NVH)] {
            ops += slice4("\(p)s\(nm)", "\(p)qkvs", begin: [0,lo,0,0], end: [1,hi,1,St], outShape: [1,hi-lo,1,St], out: "\(p)\(nm)s")
            ops += reshapeTo("\(p)r\(nm)", "\(p)\(nm)s", [nh,D,St], out: "\(p)\(nm)r")
            ops += transposeTo("\(p)t\(nm)", "\(p)\(nm)r", [0,2,1], outShape: [nh,St,D], out: nh == NVH ? "\(p)\(nm)" : "\(p)\(nm)16")
        }
        for nm in ["q","k"] {
            var d = strF(1, "concat")
            d += inputBindingMulti(2, param: "values", varNames: ["\(p)\(nm)16","\(p)\(nm)16","\(p)\(nm)16"])
            d += inputBinding(2, param: "axis", varName: "cc_ax")
            d += inputBinding(2, param: "interleave", varName: "g_f")
            d += lenF(3, namedValue("\(p)\(nm)", .fp16, [NVH,St,D]))
            d += mapEntry(5, key: "name", value: stringValue("\(p)cc\(nm)"))
            ops += lenF(3, d)
        }
        // recurrence chunk (chunk length == tile length), state carried across tiles
        let hOut = "hst\(t+1)"
        ops += gdChunk2(p: "\(p)gd_", q: "\(p)q", k: "\(p)k", v: "\(p)v", lg: "\(p)lg", bt: "\(p)bt",
                        hin: hCur, oOut: "\(p)od", hOut: hOut,
                        eye: "eye", trilS: "trilS", trilI: "trilI",
                        vb: [NVH], L: St, D: D)
        hCur = hOut
        ops += transposeTo("\(p)ot", "\(p)od", [0,2,1], outShape: [NVH,D,St], out: "\(p)odT")
        ops += reshapeTo("\(p)or", "\(p)odT", [1,Cv,1,St], out: "\(p)ost")
        // gated norm * silu(z), out_proj, residual (vs unpadded x tile)
        ops += rms4("\(p)gn", "\(p)ost", "\(p)on", Cv, St)
        ops += un("silu", "\(p)zsi", "\(p)zz", out: "\(p)zs", shape: [1,Cv,1,St])
        ops += ew("mul", "\(p)gml", "\(p)on", "\(p)zs", out: "\(p)og", shape: [1,Cv,1,St])
        ops += convW("\(p)po", "\(p)og", "wo", "\(p)op", C, St)
        let xres: String
        if pad > 0 {
            ops += slice4("\(p)sxr", "\(p)xp", begin: [0,0,0,pad], end: [1,C,1,W], outShape: [1,C,1,St], out: "\(p)xr")
            xres = "\(p)xr"
        } else { xres = "\(p)xp" }
        ops += ew("add", "\(p)r1", xres, "\(p)op", out: "\(p)h1", shape: [1,C,1,St])
        // MLP in K feature blocks with accumulated down projection
        ops += rms4("\(p)n2", "\(p)h1", "\(p)mx", C, St)
        var acc = ""
        for k in 0..<K {
            ops += convW("\(p)g\(k)", "\(p)mx", "wg\(k)", "\(p)gg\(k)", IB, St)
            ops += convW("\(p)u\(k)", "\(p)mx", "wu\(k)", "\(p)uu\(k)", IB, St)
            ops += un("silu", "\(p)si\(k)", "\(p)gg\(k)", out: "\(p)sg\(k)", shape: [1,IB,1,St])
            ops += ew("mul", "\(p)gm\(k)", "\(p)sg\(k)", "\(p)uu\(k)", out: "\(p)gu\(k)", shape: [1,IB,1,St])
            ops += convW("\(p)d\(k)", "\(p)gu\(k)", "wd\(k)", "\(p)dd\(k)", C, St)
            if k == 0 { acc = "\(p)dd0" }
            else {
                ops += ew("add", "\(p)ac\(k)", acc, "\(p)dd\(k)", out: "\(p)a\(k)", shape: [1,C,1,St])
                acc = "\(p)a\(k)"
            }
        }
        ops += ew("add", "\(p)r2", "\(p)h1", acc, out: "y\(t)", shape: [1,C,1,St])
        yNames.append("y\(t)")
    }

    let ins: [(String,[Int])] = [("x",[1,C,1,S]),("h0",[NVH,D,D]),
        ("eye",[1,St,St]),("trilS",[NVH,St,St]),("trilI",[NVH,St,St])]
    var outs: [(String,[Int])] = yNames.map { ($0, [1,C,1,St]) }
    outs.append((hCur, [NVH,D,D]))
    let spec = buildSpec(inputs: ins, outputs: outs, ops: ops)
    // FLOPs: useful = monolithic formula; overhead = qkv proj on the 3 overlap cols
    let projF = 2.0*Double(C)*Double(Cqkv+Cv+2*NVH)*Double(S)
    let outF  = 2.0*Double(C)*Double(Cv)*Double(S)
    let mlpF  = 2.0*Double(C)*Double(inter)*3*Double(S)
    var iters = 0, span = 1
    while span < St - 1 { span *= 2; iters += 1 }
    let cl = Double(St), dd = Double(D)
    let perHeadChunk = 2*cl*cl*1 + 2*cl*cl*dd + Double(iters)*2*2*cl*cl*cl
        + 2*cl*dd*dd + 2*cl*cl*dd + 2*cl*dd*dd + 2*cl*cl*dd + 2*cl*cl*dd + 2*dd*cl*dd
    let recF = Double(NVH*NT) * perHeadChunk
    let useful = projF + outF + mlpF + recF
    let overhead = 2.0*Double(C)*Double(Cqkv)*3.0*Double(NT-1)
    // max live intermediate activation (largest single tensor, fp16), excluding model input x and per-tile outputs
    let cands: [Int] = [
        C*(St+3), Cqkv*(St+3), Cqkv*St, Cv*St, inter/K*St,
        NVH*St*St, NVH*St*D, NVH*D*D, C*St ]
    let maxLive = Double(cands.max()!) * 2 / 1e6
    return TiledBuild(spec: spec, ins: ins, yNames: yNames,
                      usefulGF: useful/1e9, overheadGF: overhead/1e9, maxLiveMB: maxLive)
}

@available(macOS 15.0, *)
func runTiledBench() async {
    let S = 512
    let tiles = (ProcessInfo.processInfo.environment["GDL_TILES"] ?? "32,48,64,96,128").split(separator: ",").map { Int($0)! }
    let ks = (ProcessInfo.processInfo.environment["GDL_KS"] ?? "1,2,4,8").split(separator: ",").map { Int($0)! }
    let reps = Int(ProcessInfo.processInfo.environment["GDL_REPS"] ?? "6")!
    print("St\tK\tliveMB\tANE\tCPU\tms\tGF\tTFLOPS")
    for St in tiles { for K in ks {
        if S % St != 0 { print("\(St)\t\(K)\tskip (512 %% St != 0)"); continue }
        let b = buildTiledLayer(S: S, St: St, K: K)
        let asset: MLModelAsset
        do { asset = try MLModelAsset(specification: b.spec) }
        catch { print("\(St)\t\(K)\tREJECTED \(String("\(error)".prefix(120)))"); continue }
        var ane = 0, cpu = 0, badLines: [String] = []
        do {
            let r = try await planPlacement(asset, verbose: false)
            ane = r.ane; cpu = r.cpu + r.gpu
            badLines = r.lines.filter { !$0.hasSuffix("ANE") }
        } catch { print("\(St)\t\(K)\tPLAN FAILED"); continue }
        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
        guard let model = try? await MLModel.load(asset: asset, configuration: cfg) else {
            print("\(St)\t\(K)\tLOAD FAILED"); continue }
        var seed: UInt64 = 7
        var feats: [String: MLFeatureValue] = [:]
        for (n, s) in b.ins {
            let a: MLMultiArray
            switch n {
            case "eye", "trilS", "trilI":
                a = try! MLMultiArray(shape: s.map{NSNumber(value:$0)}, dataType: .float16)
                a.withUnsafeMutableBytes { r,_ in
                    let pp = r.bindMemory(to: Float16.self)
                    for bb in 0..<s[0] { for i in 0..<St { for j in 0..<St {
                        let on: Bool = n == "eye" ? i == j : (n == "trilS" ? i > j : i >= j)
                        pp[bb*St*St + i*St + j] = on ? 1 : 0
                    } } }
                }
            case "h0":
                a = try! MLMultiArray(shape: s.map{NSNumber(value:$0)}, dataType: .float16)
                a.withUnsafeMutableBytes { r,_ in let pp = r.bindMemory(to: Float16.self); for i in 0..<48*128*128 { pp[i]=0 } }
            default:
                a = randF16Array(shape: s, range: -0.5 ... 0.5, seed: &seed)
            }
            feats[n] = MLFeatureValue(multiArray: a)
        }
        let input = try! MLDictionaryFeatureProvider(dictionary: feats)
        guard let out0 = try? await model.prediction(from: input) else { print("\(St)\t\(K)\tPREDICT FAILED"); continue }
        if let dump = ProcessInfo.processInfo.environment["GDL_DUMPY"] {
            var dat = Data()
            for yn in b.yNames {
                let ya = out0.featureValue(for: yn)!.multiArrayValue!
                ya.withUnsafeBytes { r in dat.append(contentsOf: r) }
            }
            try? dat.write(to: URL(fileURLWithPath: dump))
            print("dumped y (\(dat.count) B) to \(dump)")
        }
        var best = Double.infinity
        for _ in 0..<reps { let t0 = Date(); _ = try? await model.prediction(from: input); best = min(best, Date().timeIntervalSince(t0)) }
        let tf = (b.usefulGF*1e9)/best/1e12
        print(String(format: "%d\t%d\t%.2f\t%d\t%d\t%.2f\t%.1f\t%.2f", St, K, b.maxLiveMB, ane, cpu, best*1000, b.usefulGF, tf))
        if !badLines.isEmpty { print("  non-ANE ops (\(badLines.count)): " + badLines.prefix(6).map{$0.trimmingCharacters(in:.whitespaces)}.joined(separator: " | ")) }
    } }
}

@available(macOS 15.0, *)
func runYCompare() {
    let args = CommandLine.arguments
    // args: mono.bin tiled.bin St  (mono: [C,512] channel-major; tiled: NT
    // blocks of [C,St] in sequence order)
    guard args.count >= 5 else { print("usage: gdprobe ycmp mono.bin tiled.bin St"); exit(2) }
    let a = try! Data(contentsOf: URL(fileURLWithPath: args[2]))
    let b = try! Data(contentsOf: URL(fileURLWithPath: args[3]))
    let St = Int(args[4])!, C = 5120, S = 512, NT = S / St
    guard a.count == b.count, a.count == C*S*2 else { print("size mismatch \(a.count) vs \(b.count)"); exit(1) }
    var maxErr: Float = 0, maxRef: Float = 0, sumSq: Double = 0
    a.withUnsafeBytes { ra in b.withUnsafeBytes { rb in
        let pa = ra.bindMemory(to: Float16.self), pb = rb.bindMemory(to: Float16.self)
        for c in 0..<C { for t in 0..<NT { for s in 0..<St {
            let x = Float(pa[c*S + t*St + s])
            let y = Float(pb[t*C*St + c*St + s])
            maxErr = max(maxErr, abs(x-y)); maxRef = max(maxRef, abs(x)); sumSq += Double((x-y)*(x-y))
        } } }
        print(String(format: "ycmp: n=%d maxAbsErr=%.5f maxRef=%.3f rms=%.6f", pa.count, maxErr, maxRef, (sumSq/Double(pa.count)).squareRoot()))
    } }
}
