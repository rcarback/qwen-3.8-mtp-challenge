// Chunkwise-parallel gated-delta recurrence as MIL ops.
//
// Recurrence (per value head, state H in R^{Dk x Dv}, o_t = q_t^T H_t):
//   u_t = beta_t * (v_t - g_t * (k_t^T H_{t-1}))
//   H_t = g_t * H_{t-1} + k_t u_t^T
// Chunkwise form (WY representation; Yang et al., "Parallelizing Linear
// Transformers with the Delta Rule over Sequence Length" 2024, gating as in
// Gated DeltaNet 2025):
//   c_i    = cumsum(log g)_i (inclusive), Lam_i = exp(c_i)
//   DS_ij  = exp(c_i - c_j) for i>j else 0;  DI same for i>=j
//   A      = beta_row .* (K K^T) .* DS      (strictly lower triangular)
//   T      = (I + A)^{-1} = prod_j (I + M^{2^j}), M = -A   (A nilpotent)
//   U      = T (beta .* (V - Lam .* (K H0)))
//   O      = Lam .* (Q H0) + (DI .* (Q K^T)) U
//   H_out  = Lam_L * H0 + (K .* rowL(DI))^T U
// Everything is matmul / elementwise / exp / cumsum: no sequential scan.
import CoreML
import Foundation

/// Shared consts for gd chunks. Emit once per program.
func gdSharedConsts(L: Int) -> Data {
    var eye = [Float](repeating: 0, count: L*L)
    var trilS = eye, trilI = eye
    for i in 0..<L { for j in 0..<L {
        if i == j { eye[i*L+j] = 1 }
        if i > j { trilS[i*L+j] = 1 }
        if i >= j { trilI[i*L+j] = 1 }
    } }
    var onehot = [Float](repeating: 0, count: L); onehot[L-1] = 1
    var d = Data()
    d += lenF(3, constOp(name: "g_eye", dt: .fp16, shape: [L,L], payload: Data.f16Arr(eye)))
    d += lenF(3, constOp(name: "g_trilS", dt: .fp16, shape: [L,L], payload: Data.f16Arr(trilS)))
    d += lenF(3, constOp(name: "g_trilI", dt: .fp16, shape: [L,L], payload: Data.f16Arr(trilI)))
    d += lenF(3, constOp(name: "g_oh", dt: .fp16, shape: [1,L], payload: Data.f16Arr(onehot)))
    d += lenF(3, constScalarOp(name: "g_neg1", dt: .fp16, payload: Data.f16(1, -1)))
    d += lenF(3, constBoolOp(name: "g_f", value: false))
    d += lenF(3, constBoolOp(name: "g_t", value: true))
    return d
}

func mm(_ nm: String, _ x: String, _ y: String, tx: Bool = false, ty: Bool = false,
        out: String, shape: [Int]) -> Data {
    lenF(3, op("matmul", name: nm,
        inputs: [("x",x),("y",y),("transpose_x", tx ? "g_t" : "g_f"),("transpose_y", ty ? "g_t" : "g_f")],
        outName: out, outType: .fp16, outShape: shape))
}
func ew(_ kind: String, _ nm: String, _ x: String, _ y: String, out: String, shape: [Int]) -> Data {
    lenF(3, op(kind, name: nm, inputs: [("x",x),("y",y)], outName: out, outType: .fp16, outShape: shape))
}
func un(_ kind: String, _ nm: String, _ x: String, out: String, shape: [Int]) -> Data {
    lenF(3, op(kind, name: nm, inputs: [("x",x)], outName: out, outType: .fp16, outShape: shape))
}

/// One chunk. qb = batch dims of q/k (e.g. [16,1]); vb = batch dims of
/// v/beta/logg/state (e.g. [16,3]); q/k: [qb,L,D], v: [vb,L,D],
/// logg/beta: [vb,L,1], hin/hout: [vb,D,D], o: [vb,L,D].
/// useCumsum=false replaces cumsum with a tril-ones matmul.
func gdChunk(p: String, q: String, k: String, v: String, lg: String, bt: String,
             hin: String, oOut: String, hOut: String,
             qb: [Int], vb: [Int], L: Int, D: Int, useCumsum: Bool) -> Data {
    let r = vb.count // rank prefix
    let LL = vb + [L,L], LD = vb + [L,D], L1 = vb + [L,1], DD = vb + [D,D]
    let qLL = qb + [L,L]
    var d = Data()
    // 1. cumulative log decay
    if useCumsum {
        d += lenF(3, constIntsOp(name: "\(p)ax", values: [r]))
        d += lenF(3, op("cumsum", name: "\(p)cs",
            inputs: [("x",lg),("axis","\(p)ax"),("exclusive","g_f"),("reverse","g_f")],
            outName: "\(p)c", outType: .fp16, outShape: L1))
    } else {
        d += mm("\(p)cs", "g_trilI", lg, out: "\(p)c", shape: L1)
    }
    // 2. decay matrices
    var perm = Array(0..<(r+2)); perm.swapAt(r, r+1)
    d += lenF(3, constIntsOp(name: "\(p)pm", values: perm))
    d += lenF(3, op("transpose", name: "\(p)tr",
        inputs: [("x","\(p)c"),("perm","\(p)pm")], outName: "\(p)cT", outType: .fp16, outShape: vb + [1,L]))
    d += ew("sub", "\(p)df", "\(p)c", "\(p)cT", out: "\(p)df", shape: LL)
    d += ew("mul", "\(p)dfS", "\(p)df", "g_trilS", out: "\(p)dfS", shape: LL)
    d += un("exp", "\(p)eS", "\(p)dfS", out: "\(p)eS", shape: LL)
    d += ew("mul", "\(p)DS", "\(p)eS", "g_trilS", out: "\(p)DS", shape: LL)
    d += ew("mul", "\(p)dfI", "\(p)df", "g_trilI", out: "\(p)dfI", shape: LL)
    d += un("exp", "\(p)eI", "\(p)dfI", out: "\(p)eI", shape: LL)
    d += ew("mul", "\(p)DI", "\(p)eI", "g_trilI", out: "\(p)DI", shape: LL)
    // 3. A = beta .* KK^T .* DS ; M = -A
    d += mm("\(p)kk", k, k, ty: true, out: "\(p)kk", shape: qLL)
    d += ew("mul", "\(p)kd", "\(p)kk", "\(p)DS", out: "\(p)kd", shape: LL)
    d += ew("mul", "\(p)A", "\(p)kd", bt, out: "\(p)A", shape: LL)
    d += ew("mul", "\(p)M", "\(p)A", "g_neg1", out: "\(p)M0", shape: LL)
    // 4. T = (I+A)^{-1} by nilpotent squaring: T_{k+1} = T_k + T_k M^{2^{k+1}}
    d += ew("add", "\(p)T0", "g_eye", "\(p)M0", out: "\(p)T0", shape: LL)
    var iters = 0, span = 1
    while span < L - 1 { span *= 2; iters += 1 }
    var tCur = "\(p)T0", mCur = "\(p)M0"
    for i in 1...max(iters,1) {
        d += mm("\(p)Msq\(i)", mCur, mCur, out: "\(p)M\(i)", shape: LL)
        d += mm("\(p)TM\(i)", tCur, "\(p)M\(i)", out: "\(p)TM\(i)", shape: LL)
        d += ew("add", "\(p)Ta\(i)", tCur, "\(p)TM\(i)", out: "\(p)T\(i)", shape: LL)
        tCur = "\(p)T\(i)"; mCur = "\(p)M\(i)"
    }
    // 5. U = T (beta .* (V - Lam .* (K H0)))
    d += un("exp", "\(p)lam", "\(p)c", out: "\(p)lam", shape: L1)
    d += mm("\(p)kh", k, hin, out: "\(p)kh", shape: LD)
    d += ew("mul", "\(p)lkh", "\(p)lam", "\(p)kh", out: "\(p)lkh", shape: LD)
    d += ew("sub", "\(p)vm", v, "\(p)lkh", out: "\(p)vm", shape: LD)
    d += ew("mul", "\(p)rhs", bt, "\(p)vm", out: "\(p)rhs", shape: LD)
    d += mm("\(p)U", tCur, "\(p)rhs", out: "\(p)U", shape: LD)
    // 6. O = Lam .* (Q H0) + (DI .* QK^T) U
    d += mm("\(p)qh", q, hin, out: "\(p)qh", shape: LD)
    d += ew("mul", "\(p)o1", "\(p)lam", "\(p)qh", out: "\(p)o1", shape: LD)
    d += mm("\(p)qk", q, k, ty: true, out: "\(p)qk", shape: qLL)
    d += ew("mul", "\(p)qkd", "\(p)qk", "\(p)DI", out: "\(p)qkd", shape: LL)
    d += mm("\(p)o2", "\(p)qkd", "\(p)U", out: "\(p)o2", shape: LD)
    d += ew("add", "\(p)o", "\(p)o1", "\(p)o2", out: oOut, shape: LD)
    // 7. H_out = Lam_L * H0 + (K .* rowL(DI))^T U
    d += mm("\(p)rw", "g_oh", "\(p)DI", out: "\(p)rw", shape: vb + [1,L])
    d += lenF(3, op("transpose", name: "\(p)rwT",
        inputs: [("x","\(p)rw"),("perm","\(p)pm")], outName: "\(p)rwT", outType: .fp16, outShape: L1))
    d += ew("mul", "\(p)kr", k, "\(p)rwT", out: "\(p)kr", shape: LD)
    d += mm("\(p)su", "\(p)kr", "\(p)U", tx: true, out: "\(p)su", shape: DD)
    d += mm("\(p)lL", "g_oh", "\(p)lam", out: "\(p)lL", shape: vb + [1,1])
    d += ew("mul", "\(p)lh", "\(p)lL", hin, out: "\(p)lh", shape: DD)
    d += ew("add", "\(p)ho", "\(p)lh", "\(p)su", out: hOut, shape: DD)
    return d
}

// MARK: - Sequential CPU reference (Float accumulation)
// q,k,v: [NH][L][D]; lg,bt: [NH][L]; h0: [NH][D][D] (H: Dk x Dv)
func gdSequentialRef(q: [Float], k: [Float], v: [Float], lg: [Float], bt: [Float],
                     h0: [Float], NH: Int, L: Int, D: Int) -> (o: [Float], h: [Float]) {
    var o = [Float](repeating: 0, count: NH*L*D)
    var hAll = [Float](repeating: 0, count: NH*D*D)
    for hd in 0..<NH {
        var H = Array(h0[(hd*D*D)..<((hd+1)*D*D)])
        for t in 0..<L {
            let g = exp(lg[hd*L+t]), b = bt[hd*L+t]
            let kOff = hd*L*D + t*D
            // pred_j = g * sum_a k_a H[a][j]
            var u = [Float](repeating: 0, count: D)
            for a in 0..<D { let ka = k[kOff+a]; if ka != 0 {
                for j in 0..<D { u[j] += ka * H[a*D+j] } } }
            for j in 0..<D { u[j] = b * (v[kOff+j] - g * u[j]) }
            // H = g*H + k u^T
            for a in 0..<D { let ka = k[kOff+a]
                for j in 0..<D { H[a*D+j] = g * H[a*D+j] + ka * u[j] } }
            // o_t = q^T H
            for j in 0..<D { var s: Float = 0
                for a in 0..<D { s += q[kOff+a] * H[a*D+j] }
                o[kOff+j] = s }
        }
        for i in 0..<D*D { hAll[hd*D*D+i] = H[i] }
    }
    return (o, hAll)
}

@available(macOS 15.0, *)
func runChunkProbe() async {
    let NH = 4, L = 64, D = 128
    let useCumsum = (ProcessInfo.processInfo.environment["GD_CUMSUM"] ?? "1") == "1"
    let nChunks = Int(ProcessInfo.processInfo.environment["GD_CHUNKS"] ?? "2") ?? 2
    // Program: nChunks chunks chained through the state (one carry per chunk).
    var ops = gdSharedConsts(L: L)
    var hCur = "h0"
    var outs: [(String,[Int])] = []
    for c in 0..<nChunks {
        let hOut = c == nChunks-1 ? "hout" : "h\(c+1)"
        ops += gdChunk(p: "c\(c)_", q: "q\(c)", k: "k\(c)", v: "v\(c)", lg: "lg\(c)", bt: "bt\(c)",
                       hin: hCur, oOut: "o\(c)", hOut: hOut,
                       qb: [NH], vb: [NH], L: L, D: D, useCumsum: useCumsum)
        outs.append(("o\(c)", [NH,L,D]))
        hCur = hOut
    }
    outs.append(("hout", [NH,D,D]))
    var ins: [(String,[Int])] = [("h0",[NH,D,D])]
    for c in 0..<nChunks {
        ins += [("q\(c)",[NH,L,D]),("k\(c)",[NH,L,D]),("v\(c)",[NH,L,D]),("lg\(c)",[NH,L,1]),("bt\(c)",[NH,L,1])]
    }
    let spec = buildSpec(inputs: ins, outputs: outs, ops: ops)
    print("chunk probe: NH=\(NH) L=\(L) D=\(D) chunks=\(nChunks) cumsum=\(useCumsum) spec=\(spec.count)B")
    let asset: MLModelAsset
    do { asset = try MLModelAsset(specification: spec) }
    catch { print("REJECTED: \(error)"); exit(1) }
    do {
        let r = try await planPlacement(asset, verbose: false)
        print("placement: ANE=\(r.ane) CPU=\(r.cpu) GPU=\(r.gpu)")
        if r.cpu + r.gpu > 0 { for l in r.lines where !l.contains("ANE") { print(l) } }
    } catch { print("plan failed: \(error)") }

    // numeric check vs sequential scan
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
    let model: MLModel
    do { model = try await MLModel.load(asset: asset, configuration: cfg) }
    catch { print("LOAD FAILED \(error)"); exit(1) }
    var seed: UInt64 = 42
    var feats: [String: MLFeatureValue] = [:]
    var arrs: [String: MLMultiArray] = [:]
    for (n, s) in ins {
        let range: ClosedRange<Float>
        if n.hasPrefix("lg") { range = -0.12 ... -0.01 }       // log g, g in (0.88,0.99)
        else if n.hasPrefix("bt") { range = 0.1 ... 0.9 }      // beta
        else if n == "h0" { range = -0.05 ... 0.05 }
        else { range = -0.15 ... 0.15 }                        // q,k,v modest so T series is tame in fp16
        let a = randF16Array(shape: s, range: range, seed: &seed)
        arrs[n] = a; feats[n] = MLFeatureValue(multiArray: a)
    }
    let input = try! MLDictionaryFeatureProvider(dictionary: feats)
    guard let outp = try? await model.prediction(from: input) else { print("PREDICT FAILED"); exit(1) }

    // build full-sequence reference across chunks
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
        for h in 0..<NH { for t in 0..<L { for j in 0..<D {
            let rv = ref.o[h*S*D + (c*L+t)*D + j]
            maxErr = max(maxErr, abs(got[h*L*D+t*D+j] - rv)); maxRef = max(maxRef, abs(rv))
        } } }
    }
    let gotH = floats(outp.featureValue(for: "hout")!.multiArrayValue!)
    var hErr: Float = 0, hRef: Float = 0
    for i in 0..<NH*D*D { hErr = max(hErr, abs(gotH[i] - ref.h[i])); hRef = max(hRef, abs(ref.h[i])) }
    print(String(format: "numeric: o maxAbsErr=%.5f (maxRef=%.3f)  h maxAbsErr=%.5f (maxRef=%.3f)",
                 maxErr, maxRef, hErr, hRef))

    // quick latency
    var best = Double.infinity
    for _ in 0..<10 { let t0 = Date(); _ = try? await model.prediction(from: input); best = min(best, Date().timeIntervalSince(t0)) }
    print(String(format: "latency: %.3f ms (chunks=%d)", best*1000, nChunks))
}
