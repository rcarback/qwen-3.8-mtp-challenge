// gdChunk v2: ANE-placement-safe variant.
//  - tril/eye masks are model INPUTS (const-tensor elementwise never places).
//  - no matmul with the same variable on both sides without transpose
//    (M^2 uses a re-masked copy: M is strictly lower so M .* trilS == M).
//  - row/scalar extraction via slice_by_index, not one-hot matmuls.
//  - no identity ops (CPU-only).
import CoreML
import Foundation

/// Masks (inputs): eye [1,L,L], trilS [vb,L,L], trilI [vb,L,L].
func gdChunk2(p: String, q: String, k: String, v: String, lg: String, bt: String,
              hin: String, oOut: String, hOut: String,
              eye: String, trilS: String, trilI: String,
              vb: [Int], L: Int, D: Int) -> Data {
    let r = vb.count
    let LL = vb + [L,L], LD = vb + [L,D], L1 = vb + [L,1], DD = vb + [D,D]
    var d = Data()
    // cumulative log decay: c = trilI @ lg   [vb,L,1]
    d += mm("\(p)cs", trilI, lg, out: "\(p)c", shape: L1)
    var perm = Array(0..<(r+2)); perm.swapAt(r, r+1)
    d += lenF(3, constIntsOp(name: "\(p)pm", values: perm))
    d += lenF(3, op("transpose", name: "\(p)tr",
        inputs: [("x","\(p)c"),("perm","\(p)pm")], outName: "\(p)cT", outType: .fp16, outShape: vb + [1,L]))
    d += ew("sub", "\(p)df", "\(p)c", "\(p)cT", out: "\(p)df", shape: LL)
    d += ew("mul", "\(p)dfS", "\(p)df", trilS, out: "\(p)dfS", shape: LL)
    d += un("exp", "\(p)eS", "\(p)dfS", out: "\(p)eS", shape: LL)
    d += ew("mul", "\(p)DS", "\(p)eS", trilS, out: "\(p)DS", shape: LL)
    d += ew("mul", "\(p)dfI", "\(p)df", trilI, out: "\(p)dfI", shape: LL)
    d += un("exp", "\(p)eI", "\(p)dfI", out: "\(p)eI", shape: LL)
    d += ew("mul", "\(p)DI", "\(p)eI", trilI, out: "\(p)DI", shape: LL)
    // A = beta .* KK^T .* DS ; M = -A
    d += mm("\(p)kk", k, k, ty: true, out: "\(p)kk", shape: LL)
    d += ew("mul", "\(p)kd", "\(p)kk", "\(p)DS", out: "\(p)kd", shape: LL)
    d += ew("mul", "\(p)A", "\(p)kd", bt, out: "\(p)A", shape: LL)
    d += ew("mul", "\(p)M", "\(p)A", "g_neg1", out: "\(p)M0", shape: LL)
    // T = (I+A)^{-1}: T_{k+1} = T_k + T_k M^{2^{k+1}}
    d += ew("add", "\(p)T0", eye, "\(p)M0", out: "\(p)T0", shape: LL)
    var iters = 0, span = 1
    while span < L - 1 { span *= 2; iters += 1 }
    var tCur = "\(p)T0", mCur = "\(p)M0"
    for i in 1...max(iters,1) {
        d += ew("mul", "\(p)Mc\(i)", mCur, trilS, out: "\(p)Mc\(i)", shape: LL)  // fresh copy, exact
        d += mm("\(p)Msq\(i)", mCur, "\(p)Mc\(i)", out: "\(p)M\(i)", shape: LL)
        d += mm("\(p)TM\(i)", tCur, "\(p)M\(i)", out: "\(p)TM\(i)", shape: LL)
        d += ew("add", "\(p)Ta\(i)", tCur, "\(p)TM\(i)", out: "\(p)T\(i)", shape: LL)
        tCur = "\(p)T\(i)"; mCur = "\(p)M\(i)"
    }
    // U = T (beta .* (V - Lam .* (K H0)))
    d += un("exp", "\(p)lam", "\(p)c", out: "\(p)lam", shape: L1)
    d += mm("\(p)kh", k, hin, out: "\(p)kh", shape: LD)
    d += ew("mul", "\(p)lkh", "\(p)lam", "\(p)kh", out: "\(p)lkh", shape: LD)
    d += ew("sub", "\(p)vm", v, "\(p)lkh", out: "\(p)vm", shape: LD)
    d += ew("mul", "\(p)rhs", bt, "\(p)vm", out: "\(p)rhs", shape: LD)
    d += mm("\(p)U", tCur, "\(p)rhs", out: "\(p)U", shape: LD)
    // O = Lam .* (Q H0) + (DI .* QK^T) U
    d += mm("\(p)qh", q, hin, out: "\(p)qh", shape: LD)
    d += ew("mul", "\(p)o1", "\(p)lam", "\(p)qh", out: "\(p)o1", shape: LD)
    d += mm("\(p)qk", q, k, ty: true, out: "\(p)qk", shape: LL)
    d += ew("mul", "\(p)qkd", "\(p)qk", "\(p)DI", out: "\(p)qkd", shape: LL)
    d += mm("\(p)o2", "\(p)qkd", "\(p)U", out: "\(p)o2", shape: LD)
    d += ew("add", "\(p)o", "\(p)o1", "\(p)o2", out: oOut, shape: LD)
    // H_out = Lam_L * H0 + (K .* rowL(DI))^T U
    var bg = [Int](repeating: 0, count: r+2); bg[r] = L-1
    d += lenF(3, constIntsVecOp(name: "\(p)rb", values: bg))
    d += lenF(3, constIntsVecOp(name: "\(p)re", values: vb + [L,L]))
    d += lenF(3, constIntsVecOp(name: "\(p)rs", values: [Int](repeating: 1, count: r+2)))
    d += lenF(3, op("slice_by_index", name: "\(p)rw",
        inputs: [("x","\(p)DI"),("begin","\(p)rb"),("end","\(p)re"),("stride","\(p)rs")],
        outName: "\(p)rw", outType: .fp16, outShape: vb + [1,L]))
    d += lenF(3, op("transpose", name: "\(p)rwT",
        inputs: [("x","\(p)rw"),("perm","\(p)pm")], outName: "\(p)rwT", outType: .fp16, outShape: L1))
    d += ew("mul", "\(p)kr", k, "\(p)rwT", out: "\(p)kr", shape: LD)
    d += mm("\(p)su", "\(p)kr", "\(p)U", tx: true, out: "\(p)su", shape: DD)
    d += lenF(3, constIntsVecOp(name: "\(p)le", values: vb + [L,1]))
    d += lenF(3, op("slice_by_index", name: "\(p)lL",
        inputs: [("x","\(p)lam"),("begin","\(p)rb"),("end","\(p)le"),("stride","\(p)rs")],
        outName: "\(p)lL", outType: .fp16, outShape: vb + [1,1]))
    d += ew("mul", "\(p)lh", "\(p)lL", hin, out: "\(p)lh", shape: DD)
    d += ew("add", "\(p)ho", "\(p)lh", "\(p)su", out: hOut, shape: DD)
    return d
}

@available(macOS 15.0, *)
func runChunkProbe2() async {
    let NH = Int(ProcessInfo.processInfo.environment["GD_NH"] ?? "4")!
    let L = 64, D = 128
    let nChunks = Int(ProcessInfo.processInfo.environment["GD_CHUNKS"] ?? "2")!
    var ops = lenF(3, constScalarOp(name: "g_neg1", dt: .fp16, payload: Data.f16(1, -1)))
    ops += lenF(3, constBoolOp(name: "g_f", value: false))
    ops += lenF(3, constBoolOp(name: "g_t", value: true))
    var hCur = "h0"
    var outs: [(String,[Int])] = []
    for c in 0..<nChunks {
        let hOut = c == nChunks-1 ? "hout" : "h\(c+1)"
        ops += gdChunk2(p: "c\(c)_", q: "q\(c)", k: "k\(c)", v: "v\(c)", lg: "lg\(c)", bt: "bt\(c)",
                        hin: hCur, oOut: "o\(c)", hOut: hOut,
                        eye: "eye", trilS: "trilS", trilI: "trilI",
                        vb: [NH], L: L, D: D)
        outs.append(("o\(c)", [NH,L,D]))
        hCur = hOut
    }
    outs.append(("hout", [NH,D,D]))
    var ins: [(String,[Int])] = [("h0",[NH,D,D]),("eye",[1,L,L]),("trilS",[NH,L,L]),("trilI",[NH,L,L])]
    for c in 0..<nChunks {
        ins += [("q\(c)",[NH,L,D]),("k\(c)",[NH,L,D]),("v\(c)",[NH,L,D]),("lg\(c)",[NH,L,1]),("bt\(c)",[NH,L,1])]
    }
    let spec = buildSpec(inputs: ins, outputs: outs, ops: ops)
    print("chunk2 probe: NH=\(NH) L=\(L) D=\(D) chunks=\(nChunks) spec=\(spec.count)B")
    let asset: MLModelAsset
    do { asset = try MLModelAsset(specification: spec) }
    catch { print("REJECTED: \(error)"); exit(1) }
    do {
        let r = try await planPlacement(asset, verbose: false)
        print("placement: ANE=\(r.ane) CPU=\(r.cpu) GPU=\(r.gpu)")
        if r.cpu + r.gpu > 0 {
            for l in r.lines where !l.hasSuffix("ANE") { print(l) }
        }
    } catch { print("plan failed: \(error)") }

    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
    let model: MLModel
    do { model = try await MLModel.load(asset: asset, configuration: cfg) }
    catch { print("LOAD FAILED \(error)"); exit(1) }
    var seed: UInt64 = 42
    var feats: [String: MLFeatureValue] = [:]
    var arrs: [String: MLMultiArray] = [:]
    for (n, s) in ins {
        var a: MLMultiArray
        switch n {
        case "eye", "trilS", "trilI":
            a = try! MLMultiArray(shape: s.map{NSNumber(value:$0)}, dataType: .float16)
            a.withUnsafeMutableBytes { r,_ in
                let pp = r.bindMemory(to: Float16.self)
                let b = s[0]
                for bb in 0..<b { for i in 0..<L { for j in 0..<L {
                    let on: Bool = n == "eye" ? i == j : (n == "trilS" ? i > j : i >= j)
                    pp[bb*L*L + i*L + j] = on ? 1 : 0
                } } }
            }
        case let x where x.hasPrefix("lg"): a = randF16Array(shape: s, range: -0.12 ... -0.01, seed: &seed)
        case let x where x.hasPrefix("bt"): a = randF16Array(shape: s, range: 0.1 ... 0.9, seed: &seed)
        case "h0": a = randF16Array(shape: s, range: -0.05 ... 0.05, seed: &seed)
        default:
            let ks = Float(ProcessInfo.processInfo.environment["GD_KSCALE"] ?? "0.15") ?? 0.15
            a = randF16Array(shape: s, range: -ks ... ks, seed: &seed)
        }
        arrs[n] = a; feats[n] = MLFeatureValue(multiArray: a)
    }
    let input = try! MLDictionaryFeatureProvider(dictionary: feats)
    guard let outp = try? await model.prediction(from: input) else { print("PREDICT FAILED"); exit(1) }

    let S = nChunks * L
    var qq = [Float](repeating: 0, count: NH*S*D), kk = qq, vv = qq
    var lgg = [Float](repeating: 0, count: NH*S), btt = lgg
    for c in 0..<nChunks {
        let qf = floats(arrs["q\(c)"]!), kf = floats(arrs["k\(c)"]!), vf = floats(arrs["v\(c)"]!)
        let lf = floats(arrs["lg\(c)"]!), bf = floats(arrs["bt\(c)"]!)
        for h in 0..<NH { for t in 0..<L {
            let dst = h*S*D + (c*L+t)*D, src = h*L*D + t*D
            for j in 0..<D { qq[dst+j]=qf[src+j]; kk[dst+j]=kf[src+j]; vv[dst+j]=vf[src+j] }
            lgg[h*S + c*L+t] = lf[h*L+t]; btt[h*S + c*L+t] = bf[h*L+t]
        } }
    }
    let ref = gdSequentialRef(q: qq, k: kk, v: vv, lg: lgg, bt: btt,
                              h0: floats(arrs["h0"]!), NH: NH, L: S, D: D)
    var maxErr: Float = 0, maxRef: Float = 0
    for c in 0..<nChunks {
        let got = floats(outp.featureValue(for: "o\(c)")!.multiArrayValue!)
        for i in 0..<NH*L*D {
            let h = i/(L*D), t = (i%(L*D))/D, j = i%D
            let rv = ref.o[h*S*D + (c*L+t)*D + j]
            maxErr = max(maxErr, abs(got[i] - rv)); maxRef = max(maxRef, abs(rv))
        }
    }
    let gotH = floats(outp.featureValue(for: "hout")!.multiArrayValue!)
    var hErr: Float = 0
    for i in 0..<NH*D*D { hErr = max(hErr, abs(gotH[i] - ref.h[i])) }
    print(String(format: "numeric: o maxAbsErr=%.5f (maxRef=%.3f)  h maxAbsErr=%.5f", maxErr, maxRef, hErr))
    var best = Double.infinity
    for _ in 0..<10 { let t0 = Date(); _ = try? await model.prediction(from: input); best = min(best, Date().timeIntervalSince(t0)) }
    print(String(format: "latency: %.3f ms (chunks=%d NH=%d)", best*1000, nChunks, NH))
}
