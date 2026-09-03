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
    public static func run() throws {
        var shapes = [(m: Int, k: Int, n: Int)]()
        for m in [8, 16, 32, 64, 128, 256, 512] {
            shapes.append((m: m, k: 2560, n: 640))
            shapes.append((m: m, k: 640, n: 2560))
        }
        let samples = ANEGemmBench.sweep(shapes: shapes, iterations: 5)
        let data = try JSONEncoder().encode(samples)
        print(String(decoding: data, as: UTF8.self))
    }
}
