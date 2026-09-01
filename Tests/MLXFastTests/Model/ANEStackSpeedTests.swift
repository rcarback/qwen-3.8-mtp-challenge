import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXFastModel

/// Task D2 scale check: does the per-MLP ANE∥GPU win survive when N layers
/// run back-to-back, the way the real 64-layer serve path dispatches the ANE
/// every layer? Each layer has its OWN weights and its OWN compiled ANE
/// program (as in the real model), so this exercises repeated ANE dispatch,
/// per-layer thread hops, and any cross-layer ANE serialization -- the
/// contention a single-MLP micro-bench cannot see.
///
/// Gated: MLXFAST_RUN_MLX_RUNTIME_TESTS=1 AND MLXFAST_ANE_STACK=1. One leg
/// per process via MLXFAST_TIMING_LEG (candidate|baseline). Layer count via
/// MLXFAST_STACK_LAYERS (default 32), fraction via MLXFAST_ANE_FRACTION.
@Suite(.serialized)
struct ANEStackSpeedTests {
    private static let hidden = 5_120
    private static let inter = 17_408

    private struct Q { let wq: MLXArray, s: MLXArray, b: MLXArray }
    private static func qw(out: Int, inn: Int, seed: UInt64) -> Q {
        MLXRandom.seed(seed)
        let w = (MLXRandom.normal([out, inn]) * Float(1.0 / Double(inn).squareRoot())).asType(.bfloat16)
        let (wq, s, b0) = quantized(w, groupSize: 64, bits: 4)
        let b = b0 ?? s
        eval(wq, s, b)
        return Q(wq: wq, s: s, b: b)
    }

    @Test("N-layer stack: concurrent ANE-split vs all-GPU, back-to-back")
    func stackTiming() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
              env["MLXFAST_ANE_STACK"] == "1" else { return }
        try #require(ANERuntime.available())
        let hidden = Self.hidden, inter = Self.inter
        let S = Int(env["MLXFAST_SEQ_LEN"] ?? "512") ?? 512
        let N = Int(env["MLXFAST_STACK_LAYERS"] ?? "32") ?? 32
        let fraction = Double(env["MLXFAST_ANE_FRACTION"] ?? "0.3125") ?? 0.3125
        let iters = Int(env["MLXFAST_TIMING_ITERS"] ?? "8") ?? 8
        let leg = env["MLXFAST_TIMING_LEG"] ?? "candidate"

        // Distinct weights per layer (as in the real model).
        var gates = [Q](), ups = [Q](), downs = [Q]()
        for i in 0 ..< N {
            gates.append(Self.qw(out: inter, inn: hidden, seed: UInt64(100 + i * 3)))
            ups.append(Self.qw(out: inter, inn: hidden, seed: UInt64(101 + i * 3)))
            downs.append(Self.qw(out: hidden, inn: inter, seed: UInt64(102 + i * 3)))
        }
        let x = MLXRandom.normal([S, hidden]).asType(.bfloat16)
        eval(x)

        func gpuMLP(_ x: MLXArray, _ l: Int) -> MLXArray {
            let g = quantizedMM(x, gates[l].wq, scales: gates[l].s, biases: gates[l].b, transpose: true, groupSize: 64, bits: 4)
            let u = quantizedMM(x, ups[l].wq, scales: ups[l].s, biases: ups[l].b, transpose: true, groupSize: 64, bits: 4)
            let h = (silu(g) * u).asType(.bfloat16)
            return quantizedMM(h, downs[l].wq, scales: downs[l].s, biases: downs[l].b, transpose: true, groupSize: 64, bits: 4)
        }

        var splits: [ANEFusedSplitMLP] = []
        if leg == "candidate" {
            let t0 = Date()
            for l in 0 ..< N {
                splits.append(try ANEFusedSplitMLP(
                    gateW: gates[l].wq, gateScales: gates[l].s, gateBiases: gates[l].b,
                    upW: ups[l].wq, upScales: ups[l].s, upBiases: ups[l].b,
                    downW: downs[l].wq, downScales: downs[l].s, downBiases: downs[l].b,
                    hidden: hidden, inter: inter, sequenceLength: S, aneFraction: fraction))
            }
            print("ANE-STACK build \(N) programs in \(Date().timeIntervalSince(t0))s")
        }

        // One "forward" = residual chain through all N layers (x stays [S,hidden]).
        func runStack() throws -> MLXArray {
            var h = x
            for l in 0 ..< N {
                let y = (leg == "candidate") ? try splits[l](h) : gpuMLP(h, l)
                h = (h + y).asType(.bfloat16)   // residual keeps shape + a real dependency chain
            }
            return h
        }

        eval(try runStack()) // warmup
        var best = Double.infinity
        for _ in 0 ..< iters {
            let t = Date(); eval(try runStack()); best = Swift.min(best, Date().timeIntervalSince(t))
        }
        print("ANE-STACK leg=\(leg) N=\(N) S=\(S) fraction=\(fraction) total=\(best * 1e3)ms perLayer=\(best / Double(N) * 1e3)ms")
        #expect(best.isFinite)
    }
}
