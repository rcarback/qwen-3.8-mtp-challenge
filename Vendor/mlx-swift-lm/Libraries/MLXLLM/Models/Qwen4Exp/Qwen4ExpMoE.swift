// Qwen3.8-Flash-Next (qwen4_exp) sparse MoE block. Local fork port;
// reference: mlx-lm PR 1788 (mlx_lm/models/qwen4_exp.py).
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// 512 routed experts (top-10) through the stacked SwitchGLU gather-GEMM, plus
/// one sigmoid-gated shared expert. The routed experts are the only tensors the
/// transform quantizes; `quantize(model:)` at load converts `switch_mlp` because
/// its `.scales` tensors are present and leaves every other Linear alone.
final class Qwen4ExpSparseMoeBlock: Module {
    let topK: Int
    let numExperts: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    private let aneShared = Qwen4ExpANESharedExpertCache(label: "mlp.shared_expert")

    init(_ args: Qwen4ExpTextConfiguration) {
        topK = args.numExpertsPerTok
        numExperts = args.numExperts
        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize, numExperts: args.numExperts)
        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
        super.init()
    }

    /// Router: float32 logits, top-k by `argPartition`, weights = softmax over the
    /// SELECTED logits (equal to softmax-all followed by renormalisation).
    func route(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let logits = gate(x.asType(.float32))
        let kth = numExperts - topK
        let idx = MLX.argPartition(logits, kth: kth, axis: -1)[.ellipsis, kth...]
        let w = MLX.softmax(MLX.takeAlong(logits, idx, axis: -1), axis: -1, precise: true)
        return (idx, w)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if Qwen4ExpANEFused.sharedEnabled,
            !(sharedExpert.gateProj is QuantizedLinear),
            !(sharedExpert.upProj is QuantizedLinear),
            !(sharedExpert.downProj is QuantizedLinear),
            Qwen4ExpANEFused.armed(tokens: x.dim(1), batch: x.dim(0)),
            let program = aneShared.program(
                forTokens: x.dim(1),
                gate: { self.sharedExpert.gateProj.weight },
                up: { self.sharedExpert.upProj.weight },
                down: { self.sharedExpert.downProj.weight })
        {
            // Hoisted out of the guard chain so a run failure can log. A
            // `try?` inside an `if let` condition list cannot carry a `catch`.
            do {
                return try offloadedForward(x, program: program)
            } catch {
                if Qwen4ExpANEFused.log {
                    fputs("[qwen4exp-ane] shared expert run failed: \(error); GPU path for this call\n", stderr)
                }
            }
        }
        let (idx, w) = route(x)
        let y = switchMLP(x, idx)  // [B, S, k, D]
        let routed = (y * w.expandedDimensions(axis: -1).asType(y.dtype)).sum(axis: -2)
        return routed.asType(x.dtype) + sigmoid(sharedExpertGate(x)) * sharedExpert(x)
    }

    /// `sharedExpert(x)` does not depend on the routed sum, so it runs on the
    /// ANE while the GPU runs the router, the routed gather-GEMM and the
    /// `[B, S, 1]` shared-expert gate. Two barriers per layer (spec 0.4).
    func offloadedForward(_ x: MLXArray, program: Qwen4ExpANESharedExpert) throws -> MLXArray {
        let tokens = x.dim(1)
        // CALLER THREAD. BARRIER 1: staging evals inside.
        let prepared = try program.makeInput(x.reshaped(tokens, -1))
        let (_, gpu) = try ConcurrentEngines.run(
            ane: { try program.predict(prepared) },  // BACKGROUND, no MLX
            gpu: { () -> (MLXArray, MLXArray) in  // CALLER THREAD
                let (idx, w) = self.route(x)
                let y = self.switchMLP(x, idx)
                let routed = (y * w.expandedDimensions(axis: -1).asType(y.dtype))
                    .sum(axis: -2).asType(x.dtype)
                let gate = sigmoid(self.sharedExpertGate(x))  // [B, S, 1]
                eval(routed, gate)  // BARRIER 2, mandatory
                return (routed, gate)
            })
        // CALLER THREAD: read back, drop padded rows, combine.
        let shared = program.readOutput(prepared, tokens: tokens)  // [tokens, hidden] fp16
        let out = gpu.0 + gpu.1 * shared.asType(x.dtype).reshaped(x.shape)
        if Qwen4ExpANEFused.zeroCopyReadback { eval(out) }  // BARRIER 3, zero-copy only
        return out
    }
}
