import CoreML
import Foundation
import MLX
import MLXRandom
import Testing
@testable import MLXFastModel

/// DIAGNOSTIC (investigation, untracked). Two questions the Task-7 catastrophe
/// (ANE offload 100-500x SLOWER in the real forward) raised:
///  (A) Where do the ~2-4s per ANE forward go, and does ANEGemm.predict()
///      latency scale with output size (CPU-fallback signature) or stay ~fixed
///      (Core ML per-prediction dispatch overhead -- the kind omlx's private
///      procedure-bank + IOSurface zero-copy path avoids)?
///  (B) Why does quantizedMM at N=17408 read ~7.2 TF here when the doc's
///      one-per-process harness reads 13.53 at that exact shape? Is it a
///      per-eval sync-amortization artifact?
@Suite(.serialized)
struct OffloadDiagnosticTests {
    private static func report(_ s: String, _ name: String) {
        let path = ProcessInfo.processInfo.environment["DIAG_REPORT_DIR"].map { "\($0)/\(name)" }
            ?? "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/\(name)"
        try? s.write(toFile: path, atomically: true, encoding: .utf8)
        print(s)
    }

    /// (A) ANEGemm.makeInput / predict / readOutput isolated, per-call, vs output size.
    @Test("diag: ANE predict latency vs output size")
    func aneLatencyVsOut() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        guard ProcessInfo.processInfo.environment["DIAG_TIMING"] == "1" else { return }
        let K = 5120, S = 512
        var out = "# (A) ANE per-call latency isolated, S=\(S), K=\(K)\n\n"
        out += "| out (F) | build s | makeInput ms | predict ms (median of 5) | readOutput ms | predict TF |\n|---|---|---|---|---|---|\n"
        for F in [512, 2048, 6976, 17408] {
            let w = MLXRandom.normal([F, K]).asType(.float16); eval(w)
            let x = MLXRandom.normal([S, K]).asType(.float16); eval(x)
            let t0 = Date()
            let g = try ANEGemm(weight: w, sequenceLength: S)
            let buildS = Date().timeIntervalSince(t0)
            // warm
            let warmIn = try g.makeInput(x)
            _ = try g.predict(warmIn); _ = try g.predict(warmIn)
            // makeInput
            var tmk = [Double]()
            for _ in 0..<5 { let a = Date(); let inp = try g.makeInput(x); tmk.append(Date().timeIntervalSince(a)); _ = inp }
            let inp = try g.makeInput(x)
            // predict (Core ML only)
            var tp = [Double]()
            for _ in 0..<5 { let a = Date(); _ = try g.predict(inp); tp.append(Date().timeIntervalSince(a)) }
            let po = try g.predict(inp)
            // readOutput
            var tr = [Double]()
            for _ in 0..<5 { let a = Date(); _ = g.readOutput(po); tr.append(Date().timeIntervalSince(a)) }
            let mkMs = (tmk.sorted()[2])*1000, prMs = (tp.sorted()[2])*1000, rdMs = (tr.sorted()[2])*1000
            let tf = (2.0*Double(F)*Double(K)*Double(S)) / (tp.sorted()[2]) / 1e12
            out += String(format: "| %d | %.2f | %.2f | %.2f | %.2f | %.3f |\n", F, buildS, mkMs, prMs, rdMs, tf)
            OffloadDiagnosticTests.report(out, "diag-ane-latency.md")
        }
        out += "\nRead: predict ms ~CONSTANT across F => fixed Core ML per-prediction overhead (omlx zero-copy path would fix). "
        out += "predict ms scaling with F => CPU fallback / compute-bound (ANE not doing the work).\n"
        OffloadDiagnosticTests.report(out, "diag-ane-latency.md")
        #expect(true)
    }

    /// (B) quantizedMM N=17408 throughput: eval-every-iter (our contaminated way)
    /// vs chain-many-per-eval (amortized sync). Does the chained method reach ~13.5?
    @Test("diag: GPU quantizedMM sync-amortization")
    func gpuSyncAmortization() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        guard ProcessInfo.processInfo.environment["DIAG_TIMING"] == "1" else { return }
        let S = 512, K = 5120, N = 17408
        let w = MLXRandom.normal([N, K]).asType(.bfloat16)
        let (wq, scales, biasesOpt) = quantized(w, groupSize: 64, bits: 4)
        let biases = biasesOpt ?? scales
        let x = MLXRandom.normal([1, S, K]).asType(.bfloat16)
        eval(x, wq, scales, biases)
        let flop = 2.0*Double(S)*Double(K)*Double(N)
        func qmm(_ a: MLXArray) -> MLXArray { quantizedMM(a, wq, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4) }

        // Method 1: eval every iteration (our CleanGPUBaseline way)
        func evalEach(_ win: Double) -> Double {
            var n = 0; let dl = Date().addingTimeInterval(win)
            while Date() < dl { let r = qmm(x); eval(r); n += 1 }
            return Double(n)*flop/win/1e12
        }
        // Method 2: chain C independent qmms, ONE eval (amortize dispatch/sync)
        func chained(_ C: Int, reps: Int) -> Double {
            // warm
            var outs = (0..<C).map { _ in qmm(x) }; eval(outs)
            var best = Double.infinity
            for _ in 0..<reps {
                let t = Date()
                outs = (0..<C).map { _ in qmm(x) }
                eval(outs)
                best = Swift.min(best, Date().timeIntervalSince(t))
            }
            return Double(C)*flop/best/1e12
        }
        _ = evalEach(0.3)
        var rep = "# (B) quantizedMM N=17408 throughput (doc D3 says 13.53 one-per-process)\n\n"
        let m1 = evalEach(0.5)
        rep += String(format: "- eval-every-iter (our contaminated way): %.2f TF\n", m1)
        for C in [8, 32, 64] {
            let tf = chained(C, reps: 3)
            rep += String(format: "- chained %d-per-eval (sync amortized): %.2f TF\n", C, tf)
            OffloadDiagnosticTests.report(rep, "diag-gpu-sync.md")
        }
        rep += "\nRead: if chained >> eval-each and approaches ~13.5, our 7.2 was per-eval sync overhead, not real GPU limit.\n"
        OffloadDiagnosticTests.report(rep, "diag-gpu-sync.md")
        #expect(true)
    }

    /// (C) makeInput internal breakdown: which line of mlxToMultiArray_1C1S
    /// is the ~2.4s constant cost? Replicates the four steps in isolation,
    /// warmed, at both projection input widths (K=5120 gate/up, K=17408 down).
    @Test("diag: makeInput step breakdown")
    func makeInputBreakdown() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        guard ProcessInfo.processInfo.environment["DIAG_TIMING"] == "1" else { return }
        let S = 512
        var out = "# (C) mlxToMultiArray_1C1S step breakdown, S=\(S)\n\n"
        out += "| K | transpose+eval ms | asData ms | alloc ms | memcpy ms | total ms |\n|---|---|---|---|---|---|\n"
        func med(_ v: [Double]) -> Double { v.sorted()[v.count/2] * 1000 }
        for K in [5120, 17408] {
            let x = MLXRandom.normal([S, K]).asType(.float16); eval(x)
            // warm the whole path twice
            for _ in 0..<2 { _ = try mlxToMultiArray_1C1S(x) }
            var tT = [Double](), tD = [Double](), tA = [Double](), tM = [Double](), tTot = [Double]()
            for _ in 0..<7 {
                let a0 = Date()
                let xT = x.transposed(1, 0).asType(.float16); eval(xT)
                let a1 = Date(); tT.append(a1.timeIntervalSince(a0))
                let bytes = xT.asData().data
                let a2 = Date(); tD.append(a2.timeIntervalSince(a1))
                let arr = try MLMultiArray(shape: [1, K, 1, S].map { NSNumber(value: $0) }, dataType: .float16)
                let a3 = Date(); tA.append(a3.timeIntervalSince(a2))
                let elementSize = MemoryLayout<Float16>.stride
                let strides = arr.strides.map(\.intValue)
                let strideK = strides[1], strideS = strides[3]
                arr.withUnsafeMutableBytes { raw, _ in
                    bytes.withUnsafeBytes { src in
                        let dstBase = raw.baseAddress!, srcBase = src.baseAddress!
                        if strideK == S, strideS == 1 { memcpy(dstBase, srcBase, bytes.count) }
                        else if strideS == 1 { let rb = S*elementSize; for k in 0..<K { memcpy(dstBase + k*strideK*elementSize, srcBase + k*rb, rb) } }
                    }
                }
                let a4 = Date(); tM.append(a4.timeIntervalSince(a3)); tTot.append(a4.timeIntervalSince(a0))
                _ = arr
            }
            out += String(format: "| %d | %.2f | %.2f | %.2f | %.2f | %.2f |\n", K, med(tT), med(tD), med(tA), med(tM), med(tTot))
            OffloadDiagnosticTests.report(out, "diag-makeinput-breakdown.md")
        }
        out += "\nRead: the dominant column IS the 2.4s. transpose+eval => transpose materialization; asData => MLX->Data sync/copy; alloc => MLMultiArray; memcpy => bridge write.\n"
        OffloadDiagnosticTests.report(out, "diag-makeinput-breakdown.md")
        #expect(true)
    }
    /// (D) makeInput FIX candidate: does contiguous(xT) make asData() fast?
    @Test("diag: makeInput contiguous fix")
    func makeInputContiguousFix() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        guard ProcessInfo.processInfo.environment["DIAG_TIMING"] == "1" else { return }
        let S = 512
        func med(_ v: [Double]) -> Double { v.sorted()[v.count/2] * 1000 }
        var out = "# (D) asData() fast-path fix, S=\(S)\n\n"
        out += "| K | asData(xT strided) ms | asData(contiguous(xT)) ms | contiguous+eval ms |\n|---|---|---|---|\n"
        for K in [5120, 17408] {
            let x = MLXRandom.normal([S, K]).asType(.float16); eval(x)
            // warm both paths
            let wA = x.transposed(1,0).asType(.float16); eval(wA); _ = wA.asData().data
            let wB = contiguous(x.transposed(1,0).asType(.float16)); eval(wB); _ = wB.asData().data
            var tStrided = [Double](), tContig = [Double](), tCeval = [Double]()
            for _ in 0..<5 {
                let xT = x.transposed(1,0).asType(.float16); eval(xT)
                let a0 = Date(); _ = xT.asData().data; tStrided.append(Date().timeIntervalSince(a0))
            }
            for _ in 0..<5 {
                let a0 = Date()
                let xc = contiguous(x.transposed(1,0).asType(.float16)); eval(xc)
                let a1 = Date(); tCeval.append(a1.timeIntervalSince(a0))
                _ = xc.asData().data; tContig.append(Date().timeIntervalSince(a1))
            }
            out += String(format: "| %d | %.2f | %.2f | %.2f |\n", K, med(tStrided), med(tContig), med(tCeval))
            OffloadDiagnosticTests.report(out, "diag-makeinput-fix.md")
        }
        out += "\nRead: if asData(contiguous) << asData(strided), the fix is `contiguous(xT)` before asData in mlxToMultiArray_1C1S.\n"
        OffloadDiagnosticTests.report(out, "diag-makeinput-fix.md")
        #expect(true)
    }
}
