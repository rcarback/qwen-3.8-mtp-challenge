// Pure-Swift ANE / GPU / CPU matmul benchmark via Core ML MLTensor.
//
// v2. v1 built inputs with MLTensor(repeating:) and measured nothing: macmon
// showed ane_power flat 0.0 and gpu_power at its 0.035 W idle floor in every
// mode, and all three modes landed within 27% of each other. Constant tensors
// get folded or CPU-handled. Inputs here are Data-backed with random bytes so
// there is nothing to fold.
//
// Placement is NOT assumed. Run under `macmon pipe` and require a non-zero
// power rail for the engine you think you are testing, exactly as E1 did.
//
// env: ANE_MODE=ane|gpu|cpu  ANE_M ANE_N ANE_K  ANE_REPS
import CoreML
import Foundation

func env(_ k: String, _ d: Int) -> Int { Int(ProcessInfo.processInfo.environment[k] ?? "") ?? d }

@available(macOS 15.0, *)
func units(_ n: String) -> MLComputeUnits? {
    switch n {
    case "ane": return .cpuAndNeuralEngine
    case "gpu": return .cpuAndGPU
    case "cpu": return .cpuOnly
    default: return nil
    }
}

/// Random Float16 bytes. Random rather than patterned so no constant folding
/// or common-subexpression trick can elide the multiply.
func randomHalfData(count: Int) -> Data {
    var d = Data(count: count * 2)
    d.withUnsafeMutableBytes { raw in
        let p = raw.bindMemory(to: Float16.self)
        for i in 0 ..< count { p[i] = Float16(Float.random(in: -1 ... 1)) }
    }
    return d
}

@available(macOS 15.0, *)
func run() async {
    let mode = ProcessInfo.processInfo.environment["ANE_MODE"] ?? "ane"
    guard let u = units(mode) else { exit(2) }
    let M = env("ANE_M", 1024), N = env("ANE_N", 17408), K = env("ANE_K", 5120)
    let reps = env("ANE_REPS", 3)
    let policy = MLComputePolicy(u)

    let aData = randomHalfData(count: M * K)
    let bData = randomHalfData(count: K * N)
    let cData = randomHalfData(count: M * N)

    func once() async -> Double {
        let start = Date()
        let r = withMLTensorComputePolicy(policy) { () -> MLTensor in
            let a = MLTensor(shape: [M, K], data: aData, scalarType: Float16.self)
            let b = MLTensor(shape: [K, N], data: bData, scalarType: Float16.self)
            return a.matmul(b)
        }
        _ = await r.shapedArray(of: Float16.self)
        return Date().timeIntervalSince(start)
    }
    func copyOnce() async -> Double {
        let start = Date()
        let r = withMLTensorComputePolicy(policy) { () -> MLTensor in
            MLTensor(shape: [M, N], data: cData, scalarType: Float16.self)
        }
        _ = await r.shapedArray(of: Float16.self)
        return Date().timeIntervalSince(start)
    }

    _ = await once()
    var best = Double.infinity
    for _ in 0 ..< reps { best = min(best, await once()) }
    _ = await copyOnce()
    var bestCopy = Double.infinity
    for _ in 0 ..< reps { bestCopy = min(bestCopy, await copyOnce()) }

    let compute = max(best - bestCopy, 1e-9)
    let flops = 2.0 * Double(M) * Double(N) * Double(K)
    print(String(
        format: "ANEPOINT\t%@\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.3f",
        mode, M, N, K, 1000 * best, 1000 * bestCopy, 1000 * compute,
        flops / compute / 1e12))
}

if #available(macOS 15.0, *) { await run() } else { exit(3) }
