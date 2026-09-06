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
///
/// The forward is split in two phases so the ANE dense lane can run the fused
/// `[in_proj_qkv; in_proj_z]` projection of the next micro-batch while the GPU
/// finishes this one: `inProjection` then `finish`.
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

    private let aneInProj = Qwen4ExpANEProjectionCache(label: "linear_attn.in_proj")
    private let aneSplitInProj = Qwen4ExpANESplitProjectionCache(label: "linear_attn.in_proj_qkv")

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

    // MARK: phase 1: the fused in-projection

    /// In-projection: `(mixedQKV [B, S, convDim], z [B, S, valueDim])`.
    ///
    /// With the fused split lane armed, the ANE computes `in_proj_qkv` rows
    /// `[0, F)` while the GPU computes `in_proj_qkv` rows `[F, convDim)` and all
    /// of `in_proj_z` in the same window. `F` is a share of the LOGICAL
    /// `convDim + valueDim` phase-1 rows, clamped to `convDim`, so the split
    /// point stays inside `in_proj_qkv` and the fused
    /// `[in_proj_qkv; in_proj_z]` weight is never materialized. At the default
    /// fraction `F = 5120` of the logical 16384 rows. At fraction 0.625 and
    /// above `F` clamps to `convDim`, the split is degenerate, and the build is
    /// refused; the layer then stays on the GPU.
    func inProjection(_ x: MLXArray) -> (MLXArray, MLXArray) {
        if Qwen4ExpANEFused.splitEnabled,
            !(inProjQKV is QuantizedLinear), !(inProjZ is QuantizedLinear),
            Qwen4ExpANEFused.armed(tokens: x.dim(1), batch: x.dim(0)),
            let program = aneSplitInProj.program(
                forTokens: x.dim(1),
                logicalOut: convDim + valueDim,
                weight: { self.inProjQKV.weight })
        {
            let tokens = x.dim(1)
            let x2 = x.reshaped(tokens, -1)
            do {
                let (qkv, extra) = try program.run(x2, gpuExtra: { [self.inProjZ($0)] })
                return (qkv.reshaped(1, tokens, -1), extra[0].reshaped(1, tokens, -1))
            } catch {
                if Qwen4ExpANEFused.log {
                    fputs("[qwen4exp-ane] split in_proj run failed: \(error); GPU path for this call\n", stderr)
                }
            }
        }
        if Qwen4ExpProjectionCompile.enabled,
            x.shape.dropLast().reduce(1, *) <= Qwen4ExpGatedResidual.compileMaxRows
        {
            if compiledInProj == nil {
                compiledInProj = compile { [self] v in (inProjQKV(v), inProjZ(v)) }
            }
            return compiledInProj!(x)
        }
        return (inProjQKV(x), inProjZ(x))
    }

    private var compiledInProj: (@Sendable (MLXArray) -> (MLXArray, MLXArray))?

    /// The ANE program for the fused `[in_proj_qkv; in_proj_z]` at `sequenceLength`,
    /// or nil when the lane is off or the program failed to build.
    func aneInProjection(sequenceLength: Int) -> Qwen4ExpANEProjection? {
        guard Qwen4ExpANELane.enabled else { return nil }
        return aneInProj.program(
            weight: { concatenated([inProjQKV.weight, inProjZ.weight], axis: 0) },
            sequenceLength: sequenceLength)
    }

    /// Split a fused projection `[B, S, convDim + valueDim]` into `(mixedQKV, z)`.
    func splitFused(_ y: MLXArray) -> (MLXArray, MLXArray) {
        (y[.ellipsis, 0 ..< convDim], y[.ellipsis, convDim...])
    }

    // MARK: phase 2: conv, recurrence, gate, out-projection

    func finish(_ x: MLXArray, mixedQKV mixedIn: MLXArray, z zFlat: MLXArray, mask: MLXArray?, cache: ArraysCache?)
        -> MLXArray
    {
        let B = x.dim(0)
        let (out, newConv, newState) = finishFunctional(
            x, mixedQKV: mixedIn, z: zFlat, mask: mask,
            convState: cache?[0] ?? MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: x.dtype),
            state: cache?[1])
        if let cache {
            cache[0] = newConv
            cache[1] = newState
            cache.offset += x.dim(1)
        }
        return out
    }

    /// The layer body with its recurrent state as explicit values: nothing is
    /// read from or written to a cache, so a compiled trace can take the state
    /// as inputs and hand the new state back as outputs. Same arithmetic as the
    /// cache-based `finish`, which is a thin wrapper over this.
    func finishFunctional(
        _ x: MLXArray, mixedQKV mixedIn: MLXArray, z zFlat: MLXArray, mask: MLXArray?,
        convState: MLXArray, state: MLXArray?
    ) -> (out: MLXArray, convState: MLXArray, state: MLXArray) {
        let B = x.dim(0)
        let S = x.dim(1)
        var mixedQKV = mixedIn
        let z = zFlat.reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(x)
        let a = inProjA(x)

        if let mask {
            mixedQKV = MLX.where(mask.expandedDimensions(axis: -1), mixedQKV, MLXArray.zeros(like: mixedQKV))
        }
        let convInput = concatenated([convState, mixedQKV], axis: 1)
        let newConv = convInput[0..., (convInput.dim(1) - (convKernelSize - 1))..., 0...]
        let convOut = silu(conv1d(convInput))
        let parts = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        var q = parts[0].reshaped(B, S, numKHeads, headKDim)
        var k = parts[1].reshaped(B, S, numKHeads, headKDim)
        let v = parts[2].reshaped(B, S, numVHeads, headVDim)

        let invScale = pow(Float(headKDim), -0.5)
        q = MLXArray(invScale * invScale).asType(x.dtype) * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        k = MLXArray(invScale).asType(x.dtype) * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        let (out, newState) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: state, mask: mask)
        return (outProj(norm(out, gate: z).reshaped(B, S, valueDim)), newConv, newState)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: ArraysCache?) -> MLXArray {
        let (mixedQKV, z) = inProjection(x)
        return finish(x, mixedQKV: mixedQKV, z: z, mask: mask, cache: cache)
    }
}
