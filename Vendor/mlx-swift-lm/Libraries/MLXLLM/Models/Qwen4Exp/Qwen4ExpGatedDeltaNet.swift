// Qwen3.8-Flash-Next (qwen4_exp) gated DeltaNet block. Local fork port;
// reference: mlx-lm PR 1788 (mlx_lm/models/qwen4_exp.py).
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

/// Gated DeltaNet block. Unlike Qwen3-Next the checkpoint splits the input
/// projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`) and the
/// output gate is a sigmoid. The recurrence is the vendored `gatedDeltaUpdate`.
final class Qwen4ExpGatedDeltaNet: Module {
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Qwen4ExpRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen4ExpTextConfiguration) {
        numVHeads = args.linearNumValueHeads
        numKHeads = args.linearNumKeyHeads
        headKDim = args.linearKeyHeadDim
        headVDim = args.linearValueHeadDim
        keyDim = headKDim * numKHeads
        valueDim = headVDim * numVHeads
        convKernelSize = args.linearConvKernelDim
        convDim = keyDim * 2 + valueDim
        let d = args.hiddenSize
        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim, kernelSize: convKernelSize,
            stride: 1, padding: 0, dilation: 1, groups: convDim, bias: false)
        _inProjQKV.wrappedValue = Linear(d, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(d, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(d, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(d, numVHeads, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        _aLog.wrappedValue = MLXArray.zeros([numVHeads])
        _norm.wrappedValue = Qwen4ExpRMSNormGated(
            dimensions: headVDim, eps: args.rmsNormEps, gate: args.outputGateType)
        _outProj.wrappedValue = Linear(valueDim, d, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: ArraysCache?) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        var mixedQKV = inProjQKV(x)
        let z = inProjZ(x).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(x)
        let a = inProjA(x)

        let convState = cache?[0] ?? MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: x.dtype)
        if let mask {
            mixedQKV = MLX.where(mask.expandedDimensions(axis: -1), mixedQKV, MLXArray.zeros(like: mixedQKV))
        }
        let convInput = concatenated([convState, mixedQKV], axis: 1)
        if let cache {
            cache[0] = convInput[0..., (convInput.dim(1) - (convKernelSize - 1))..., 0...]
        }
        let convOut = silu(conv1d(convInput))
        let parts = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        var q = parts[0].reshaped(B, S, numKHeads, headKDim)
        var k = parts[1].reshaped(B, S, numKHeads, headKDim)
        let v = parts[2].reshaped(B, S, numVHeads, headVDim)

        let invScale = pow(Float(headKDim), -0.5)
        q = MLXArray(invScale * invScale).asType(x.dtype) * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        k = MLXArray(invScale).asType(x.dtype) * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        let (out, newState) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: cache?[1], mask: mask)
        if let cache {
            cache[1] = newState
            cache.offset += S
        }
        return outProj(norm(out, gate: z).reshaped(B, S, valueDim))
    }
}
