// Qwen3.8-Flash-Next (qwen4_exp) norms and hyper-connection gated residual.
// Local fork port; reference: mlx-lm PR 1788 (mlx_lm/models/qwen4_exp.py).
import Foundation
import MLX
import MLXFast
import MLXNN

/// Zero-centred RMSNorm: `y = rms_norm(x) * (1 + weight)`. The checkpoint stores
/// weights near zero; `1 + w` is formed in float32 at every call so bf16 never
/// rounds a small `w` away (folding `1 + w` into a bf16 weight would).
/// With `groupSize`, each group of `groupSize` features (one hyper-connection
/// stream) is normalised on its own statistic and the flat vector is scaled.
final class Qwen4ExpRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let groupSize: Int?

    init(dimensions: Int, groupSize: Int? = nil, eps: Float) {
        self.eps = eps
        self.groupSize = groupSize
        if let g = groupSize {
            precondition(dimensions % g == 0, "dimensions must divide by groupSize")
        }
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scale = (1.0 + weight.asType(.float32)).asType(x.dtype)
        guard let g = groupSize else {
            return MLXFast.rmsNorm(x, weight: scale, eps: eps)
        }
        let shape = x.shape
        let grouped = x.reshaped(Array(shape.dropLast()) + [-1, g])
        let normed = MLXFast.rmsNorm(grouped, weight: MLXArray.ones([g], dtype: x.dtype), eps: eps)
        return normed.reshaped(shape) * scale
    }
}

/// Conventional gated RMSNorm (`linear_attn.norm`): `act(gate) * rms_norm(x) * weight`.
final class Qwen4ExpRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let useSigmoid: Bool

    init(dimensions: Int, eps: Float, gate: String) {
        self.eps = eps
        self.useSigmoid = gate == "sigmoid"
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps).asType(.float32)
        let g = gate.asType(.float32)
        let act = useSigmoid ? sigmoid(g) : silu(g)
        return (act * normed).asType(x.dtype)
    }
}

/// Hyper-connection read/write gate ("gated residual"). The residual stream is
/// `hcCount` copies of the hidden width. `mix` reads one hidden vector out of
/// the wide stream with a data-dependent element-wise gate; `combine` writes a
/// branch output back into every stream with a per-stream scalar gate.
final class Qwen4ExpGatedResidual: Module {
    let hc: Int
    let d: Int

    @ModuleInfo(key: "hc_norm") var hcNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "input_mix_weight_down") var mixDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var mixUp: Linear
    @ModuleInfo(key: "block_inject_weight") var inject: Linear?

    init(_ args: Qwen4ExpTextConfiguration, combine: Bool) {
        hc = args.hcCount
        d = args.hiddenSize
        let hcDim = hc * d
        _hcNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: hcDim, groupSize: d, eps: args.rmsNormEps)
        _mixDown.wrappedValue = Linear(hcDim, args.hcLowrank, bias: false)
        _mixUp.wrappedValue = Linear(args.hcLowrank, hcDim, bias: false)
        _inject.wrappedValue = combine ? Linear(hcDim, hc, bias: false) : nil
        super.init()
    }

    func mix(_ hyper: MLXArray) -> (mixed: MLXArray, inject: MLXArray?) {
        let normed = hcNorm(hyper)
        let lead = Array(hyper.shape.dropLast())
        var w = silu(mixDown(normed) / Float(hc))
        w = sigmoid(mixUp(w)).reshaped(lead + [hc, d])
        let mixed = (w * normed.reshaped(lead + [hc, d])).mean(axis: -2)
        guard let inject else { return (mixed, nil) }
        let gate = 2 * sigmoid(inject(normed) / Float(hc))
        return (mixed, gate)
    }

    func combine(_ hyper: MLXArray, branch: MLXArray, inject: MLXArray) -> MLXArray {
        let lead = Array(branch.shape.dropLast())
        let written = branch.expandedDimensions(axis: -2) * inject.expandedDimensions(axis: -1)
        return hyper + written.reshaped(lead + [hc * d])
    }
}
