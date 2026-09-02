// Do the ANE and the CPU matrix unit actually run CONCURRENTLY, or does one
// block the other?
//
// Every engine figure in docs/compute-engine-map.md was measured with that
// engine running alone, so the "1.69x if overlapped" number is arithmetic, not
// an observation. This measures the overlap directly: run each engine in a
// loop for a fixed wall-clock window, count completed iterations, and compare
// the solo rate against the concurrent rate.
//
//   concurrent rate ~= solo rate for both  -> genuine parallelism, rates add
//   each drops to ~half                    -> serialized, nothing gained
//   one unaffected, other collapses        -> asymmetric contention
//
// No MLX, so Metal is not covered here; ANE via Core ML and CPU via Accelerate
// are both reachable standalone.
//
// env: OV_SECONDS OV_MODE=solo_ane|solo_cpu|both  OV_C OV_S OV_LAYERS
//      OV_GEMM_N OV_GEMM_K
import Accelerate
import CoreML
import Foundation

func env(_ k: String, _ d: Int) -> Int { Int(ProcessInfo.processInfo.environment[k] ?? "") ?? d }

// MARK: - ANE model (1x1 conv stack, the shape section 19.2 found fastest)

func buildConvProgram(C: Int, S: Int, layers: Int, wBytes: Data) -> Data {
    var ops = Data()
    ops += lenF(3, constIntsOp(name: "st", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "dl", values: [1, 1]))
    ops += lenF(3, constIntsOp(name: "pd", values: [0, 0, 0, 0]))
    ops += lenF(3, constIntsOp(name: "gp", values: [1]))
    ops += lenF(3, constStringOp(name: "pt", value: "valid"))
    var cur = "x"
    for l in 0 ..< layers {
        ops += lenF(3, constOp(name: "w\(l)", dt: .fp16, shape: [C, C, 1, 1], payload: wBytes))
        let outName = (l == layers - 1) ? "y" : "c\(l)"
        ops += lenF(3, op("conv", name: "cv\(l)",
                          inputs: [("x", cur), ("weight", "w\(l)"), ("strides", "st"),
                                   ("pad_type", "pt"), ("pad", "pd"),
                                   ("dilations", "dl"), ("groups", "gp")],
                          outName: outName, outType: .fp16, outShape: [1, C, 1, S]))
        cur = outName
    }
    let block = strF(2, "y") + ops
    let fn = lenF(1, namedValue("x", .fp16, [1, C, 1, S])) + strF(2, "CoreML8")
        + mapEntry(3, key: "CoreML8", value: block)
    let program = varF(1, 1) + mapEntry(2, key: "main", value: fn)
    func fd(_ n: String) -> Data {
        let arr = lenF(1, [1, C, 1, S].reduce(Data()) { $0 + varint(UInt64($1)) }) + varF(2, 65552)
        return strF(1, n) + lenF(3, lenF(5, arr))
    }
    let desc = lenF(1, fd("x")) + lenF(10, fd("y"))
    return varF(1, 9) + lenF(2, desc) + lenF(502, program)
}

let counterLock = NSLock()
var aneIters = 0
var cpuIters = 0

@available(macOS 15.0, *)
func aneLoop(deadline: Date, model: MLModel, input: MLFeatureProvider) async {
    while Date() < deadline {
        _ = try? await model.prediction(from: input)
        counterLock.lock(); aneIters += 1; counterLock.unlock()
    }
}

/// Accelerate sgemm on the P-cluster matrix unit. Runs on its own thread so it
/// cannot be starved by the Core ML async machinery.
func cpuLoop(deadline: Date, M: Int, N: Int, K: Int) {
    let a = UnsafeMutablePointer<Float>.allocate(capacity: M * K)
    let b = UnsafeMutablePointer<Float>.allocate(capacity: N * K)
    let c = UnsafeMutablePointer<Float>.allocate(capacity: M * N)
    defer { a.deallocate(); b.deallocate(); c.deallocate() }
    for i in 0 ..< M * K { a[i] = 0.01 }
    for i in 0 ..< N * K { b[i] = 0.02 }
    while Date() < deadline {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(M), Int32(N), Int32(K), 1.0,
                    a, Int32(K), b, Int32(K), 0.0, c, Int32(N))
        counterLock.lock(); cpuIters += 1; counterLock.unlock()
    }
}

@available(macOS 15.0, *)
func run() async {
    setvbuf(stdout, nil, _IONBF, 0)
    let mode = ProcessInfo.processInfo.environment["OV_MODE"] ?? "both"
    let seconds = Double(env("OV_SECONDS", 6))
    let C = env("OV_C", 3072), S = env("OV_S", 1024), layers = env("OV_LAYERS", 4)
    let gM = 1024, gN = env("OV_GEMM_N", 2176), gK = env("OV_GEMM_K", 5120)

    var w = Data(count: C * C * 2)
    w.withUnsafeMutableBytes { r in
        let p = r.bindMemory(to: Float16.self)
        for i in 0 ..< C * C { p[i] = Float16(0.001) }
    }
    let spec = buildConvProgram(C: C, S: S, layers: layers, wBytes: w)
    guard let asset = try? MLModelAsset(specification: spec) else { print("asset failed"); exit(1) }
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
    guard let model = try? await MLModel.load(asset: asset, configuration: cfg) else { print("load failed"); exit(1) }
    guard let x = try? MLMultiArray(shape: [1, C, 1, S].map { NSNumber(value: $0) }, dataType: .float16)
    else { exit(1) }
    x.withUnsafeMutableBytes { r, _ in
        let p = r.bindMemory(to: Float16.self); for i in 0 ..< C * S { p[i] = Float16(0.01) }
    }
    let input = try! MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: x)])
    _ = try? await model.prediction(from: input)   // warm

    let deadline = Date().addingTimeInterval(seconds)
    var cpuThread: Thread?
    if mode == "solo_cpu" || mode == "both" {
        let t = Thread { cpuLoop(deadline: deadline, M: gM, N: gN, K: gK) }
        t.qualityOfService = .userInitiated
        cpuThread = t
        t.start()
    }
    if mode == "solo_ane" || mode == "both" {
        await aneLoop(deadline: deadline, model: model, input: input)
    }
    while let t = cpuThread, !t.isFinished { usleep(20_000) }

    let aneFlops = 2.0 * Double(S) * Double(C) * Double(C) * Double(layers) * Double(aneIters)
    let cpuFlops = 2.0 * Double(gM) * Double(gN) * Double(gK) * Double(cpuIters)
    print(String(format: "OVERLAP\t%@\t%.1f\tane_iters=%d\tcpu_iters=%d\tane_TFLOPS=%.3f\tcpu_TFLOPS=%.3f",
                 mode, seconds, aneIters, cpuIters, aneFlops / seconds / 1e12, cpuFlops / seconds / 1e12))
}
if #available(macOS 15.0, *) { await run() } else { exit(3) }
