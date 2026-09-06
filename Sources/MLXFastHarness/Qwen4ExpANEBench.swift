// Local fork only: drives ANEGemmBench over the expert-shaped GEMMs and prints
// one JSON line per shape. Loads no model.
import Foundation
import MLXFastCore
import MLXLLM

public enum Qwen4ExpANEBench {
    /// Expert GEMM shapes for this tower: gate and up are [640, 2560], down is
    /// [2560, 640]. The token dimension sweeps the plausible bucket sizes,
    /// because top-10 of 512 experts puts only about 5 tokens on each expert
    /// per 256-token tile and the padding waste decides viability.
    ///
    /// `MLXFAST_ANE_SHAPES="1x2560x10240,1x640x2560"` overrides the sweep
    /// with explicit `m x k x n` shapes (the decode-time comparison uses
    /// m = 1..8 on the dense projection shapes); `MLXFAST_ANE_ITER` sets the
    /// repeat count (default 5).
    public static func run() throws {
        var shapes = [(m: Int, k: Int, n: Int)]()
        let env = ProcessInfo.processInfo.environment
        if let spec = env["MLXFAST_ANE_SHAPES"] {
            for item in spec.split(separator: ",") {
                let parts = item.split(separator: "x").compactMap { Int($0) }
                guard parts.count == 3 else {
                    throw MLXFastError.invalidInput("MLXFAST_ANE_SHAPES entry \(item) is not m x k x n")
                }
                shapes.append((m: parts[0], k: parts[1], n: parts[2]))
            }
        } else {
            for m in [8, 16, 32, 64, 128, 256, 512] {
                shapes.append((m: m, k: 2560, n: 640))
                shapes.append((m: m, k: 640, n: 2560))
            }
        }
        let iterations = Int(env["MLXFAST_ANE_ITER"] ?? "") ?? 5
        let samples = ANEGemmBench.sweep(shapes: shapes, iterations: iterations)
        let data = try JSONEncoder().encode(samples)
        print(String(decoding: data, as: UTF8.self))
    }
}
