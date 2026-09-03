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
        let (idx, w) = route(x)
        let y = switchMLP(x, idx)  // [B, S, k, D]
        let routed = (y * w.expandedDimensions(axis: -1).asType(y.dtype)).sum(axis: -2)
        return routed.asType(x.dtype) + sigmoid(sharedExpertGate(x)) * sharedExpert(x)
    }
}
