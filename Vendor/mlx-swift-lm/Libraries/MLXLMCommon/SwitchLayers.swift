import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

/// Compiled SiLU-gated product (`silu(gate) * up`) for the common MoE GLU path.
/// Fusing activation + product into one compiled, shapeless kernel cuts kernel
/// dispatches and intermediates on the hot decode path. Upstream ef85ed0.
///
/// Gated by `MLXHardwareInfo.isCompiledDecodeSupported` (env `MLX_COMPILED_DECODE`,
/// default on) like the sibling `compiledSwiGLU` / `safeGeluApproximate` fusions.
/// The default SiLU `SwitchGLU` path wires this in as `activationProduct` (the
/// highest-precedence branch in `callAsFunction`) and `LFM2MoE` calls it directly,
/// so without the gate both would keep hitting compiled kernels on the very M1/M2 +
/// macOS Tahoe machines the opt-out (MLX #3329) is meant to protect. Falls back to
/// the plain uncompiled closure when off; the default (env unset) stays compiled.
public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, up in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Compiled weighted expert-output combine (`(outputs * weights[..., None]).sum(-2)`).
/// Shared by MoE routers (e.g. Gemma 4) to fuse the scale + reduce. Upstream ef85ed0.
public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}

// MARK: - Compiled activation fusions (vMLX / osaurus-main port)

/// Approximate (tanh) GELU written with `x * x * x` instead of the Power
/// primitive (`x ** 3`). The Power primitive returns zero results under the
/// macOS Tahoe Metal JIT (MLX #3329), so the explicit multiplies keep it safe
/// under `compile(shapeless: true)`. Numerically identical to
/// `MLXNN.geluApproximate`.
///
/// Gated by `MLXHardwareInfo.isCompiledDecodeSupported` (env `MLX_COMPILED_DECODE`,
/// default on); falls back to the plain closure when compiled fusions are off.
public let safeGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Drop-in replacement for `MLXNN.GELU(approximation: .tanh)` that avoids the
/// Power primitive crash. Use anywhere a tanh-approx GELU unary layer is needed.
public class SafeGELU: Module, UnaryLayer {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        safeGeluApproximate(x)
    }
}

/// Compiled SiLU-gated GLU product (`silu(gate) * up`). Same math as
/// `compiledSiluProduct` above, but gated by `MLXHardwareInfo` so M1/M2 + macOS
/// Tahoe can opt out. Used by `SwitchGLU` when a SiLU activation is supplied via
/// the custom-activation initializer (where `activationProduct` is nil).
private let compiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Compiled GELU-gated GLU product (`geluApprox(gate) * up`), fusing the tanh
/// GELU and the element-wise multiply into one shapeless kernel. Uses the
/// Power-free `x * x * x` GELU so it is safe under `compile(shapeless: true)`.
private let compiledGeGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}


// MARK: - Fused gate/up gather-GEMM

/// `MLX_SWITCH_FUSE_GATE_UP`. When on, a ``SwitchGLU`` that loaded SEPARATE
/// `gate_proj` / `up_proj` expert stacks concatenates them once, on first
/// forward, into a single `[E, 2 * hiddenDims, inputDims]` stack and issues ONE
/// gather-GEMM per MoE layer instead of two (SonicMoE, arXiv 2512.14080: one
/// pass over the gathered input tiles, activation applied on the split halves).
///
/// The concatenation is along the OUTPUT-channel axis. Affine group
/// quantization groups along the INPUT axis, so every group stays inside the row
/// it was quantized in and the packed weight, `scales` and `biases` concatenate
/// row-wise with no regrouping. The arithmetic per output row is therefore the
/// same dot product over the same input groups in the same order as the unfused
/// call.
///
/// The unfused stacks are released once the fused stack is materialized, so the
/// steady-state expert-weight footprint is unchanged; the transient peak during
/// the build is one layer's `gate + up`.
public enum SwitchGLUFusion {
    /// Default ON since 2026-09-05: at one decode row it measured -0.6 ms per
    /// step (-1.0 percent, counterbalanced) on the Qwen4Exp tower, and the
    /// loader builds the stack eagerly so the concat is not charged to the
    /// first request. `MLX_SWITCH_FUSE_GATE_UP=0` disables it.
    public static let fuseGateUp: Bool =
        ProcessInfo.processInfo.environment["MLX_SWITCH_FUSE_GATE_UP"] != "0"
}

/// Holds the lazily built fused `gate_up` stack for one ``SwitchGLU``.
/// Deliberately NOT a `@ModuleInfo` child: the checkpoint has no `gate_up_proj`
/// key, so a real child would break weight loading and quantization discovery.
final class FusedGateUpStore: @unchecked Sendable {
    private let lock = NSLock()
    private var attempted = false
    private var fused: SwitchLinear?

    /// Builds the fused stack once. Returns nil (and never retries) when the
    /// two stacks cannot be concatenated compatibly.
    func get(gate: SwitchLinear, up: SwitchLinear) -> SwitchLinear? {
        lock.lock()
        defer { lock.unlock() }
        if attempted { return fused }
        attempted = true
        fused = Self.build(gate: gate, up: up)
        return fused
    }

    var built: SwitchLinear? {
        lock.lock()
        defer { lock.unlock() }
        return fused
    }

    private static func build(gate: SwitchLinear, up: SwitchLinear) -> SwitchLinear? {
        guard gate.inputDims == up.inputDims, gate.outputDims == up.outputDims,
            gate.numExperts == up.numExperts
        else { return nil }

        let outputDims = gate.outputDims + up.outputDims
        // The additive bias, when present, must be present on both sides.
        let addBias: MLXArray?
        switch (gate.bias, up.bias) {
        case (nil, nil): addBias = nil
        case let (g?, u?): addBias = concatenated([g, u], axis: -1)
        default: return nil
        }

        if let qg = gate as? QuantizedSwitchLinear {
            guard let qu = up as? QuantizedSwitchLinear,
                qg.groupSize == qu.groupSize, qg.bits == qu.bits, qg.mode == qu.mode
            else { return nil }
            // Row-wise concat: axis -2 is the output-channel axis for the packed
            // weight and for scales/biases alike, and quantization groups run
            // along the (untouched) input axis.
            guard qg.weight.ndim == 3, qg.scales.ndim == 3 else { return nil }
            let w = concatenated([qg.weight, qu.weight], axis: -2)
            let sc = concatenated([qg.scales, qu.scales], axis: -2)
            let bi: MLXArray?
            switch (qg.biases, qu.biases) {
            case (nil, nil): bi = nil
            case let (g?, u?): bi = concatenated([g, u], axis: -2)
            default: return nil
            }
            var toEval = [w, sc]
            if let bi { toEval.append(bi) }
            if let addBias { toEval.append(addBias) }
            eval(toEval)
            return QuantizedSwitchLinear(
                inputDims: qg.inputDims, outputDims: outputDims, numExperts: qg.numExperts,
                weight: w, scales: sc, biases: bi, bias: addBias,
                groupSize: qg.groupSize, bits: qg.bits, mode: qg.mode)
        }

        guard !(up is QuantizedSwitchLinear), gate.weight.ndim == 3 else { return nil }
        let w = concatenated([gate.weight, up.weight], axis: -2)
        var toEval = [w]
        if let addBias { toEval.append(addBias) }
        eval(toEval)
        return SwitchLinear(
            inputDims: gate.inputDims, outputDims: outputDims, numExperts: gate.numExperts,
            weight: w, bias: addBias)
    }
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: SwitchLinear?
    @ModuleInfo(key: "up_proj") public var upProj: SwitchLinear?
    @ModuleInfo(key: "gate_up_proj") public var gateUpProj: SwitchLinear?
    @ModuleInfo(key: "down_proj") public var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    /// Optional fused (activation * up) kernel. Set for the default SiLU path so
    /// the GLU product runs as one compiled op; nil when a custom activation is
    /// supplied (we then fall back to `activation(gate) * up`). Upstream ef85ed0.
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    /// Activation-type flags detected once at init from a tiny test input (vMLX
    /// approach — no per-token check). Only consulted when `activationProduct` is
    /// nil (the custom-activation path): they let SiLU/GELU custom activations use
    /// the compiled `compiledSwiGLU` / `compiledGeGLU` fusions instead of the
    /// uncompiled `activation(gate) * up`. On any mismatch we fall back to that
    /// exact uncompiled path, so detection only ever enables a numerically
    /// equivalent fast path — it can never change results.
    let isSiluActivation: Bool
    let isGeluActivation: Bool

    /// Lazily built fused `[E, 2 * hiddenDims, inputDims]` stack for the
    /// separate-`gate_proj`/`up_proj` checkpoint layout. See ``SwitchGLUFusion``.
    private let fusedGateUpStore = FusedGateUpStore()

    /// Per-instance override for ``SwitchGLUFusion/fuseGateUp``. The env knob is
    /// a `static let` read once, so tests that need both arms in one process
    /// set this before the first forward. Nil uses the process default.
    /// False retains the unfused stacks; true builds the fused stack.
    public var forceFuseGateUp: Bool? = nil

    /// Default SiLU GLU path -- uses the compiled fused (silu * up) kernel.
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false,
        fuseGateUp: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct
        // Default path is SiLU and `activationProduct` is non-nil, so these are
        // not consulted on the hot path; set them accurately for completeness
        // (and to avoid a needless probe eval at load for every MoE layer).
        self.isSiluActivation = true
        self.isGeluActivation = false

        if fuseGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2, numExperts: numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        }
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Custom-activation GLU path -- runs `activation(gate) * up` uncompiled.
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false,
        fuseGateUp: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil
        // Detect SiLU/GELU once via a tiny test input (vMLX approach) so the hot
        // path can select the compiled fusion without a per-token check. Exact
        // equality is intentional: a match means the supplied closure computes
        // that exact function; any non-match falls back to `activation(gate) * up`
        // in callAsFunction, so this can only ever enable an equivalent fast path.
        let probe = MLXArray([Float(1.0)])
        let probeOut = activation(probe)
        let detectedSilu = (probeOut .== MLXNN.silu(probe)).all().item(Bool.self)
        self.isSiluActivation = detectedSilu
        self.isGeluActivation =
            !detectedSilu && (probeOut .== safeGeluApproximate(probe)).all().item(Bool.self)

        if fuseGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2, numExperts: numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        }
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// The fused `gate_up` stack, built on first use and cached, or nil when the
    /// knob is off, the checkpoint already ships a fused stack, or the two
    /// stacks are not concatenation-compatible.
    ///
    /// Releases the unfused `gate_proj` / `up_proj` children on success so the
    /// steady-state footprint is unchanged. ``denseGateUpDown(expert:)`` reads
    /// the released halves back out of the fused stack.
    public func fusedGateUp() -> SwitchLinear? {
        guard forceFuseGateUp ?? SwitchGLUFusion.fuseGateUp, gateUpProj == nil else { return nil }
        if let already = fusedGateUpStore.built { return already }
        guard let gateProj, let upProj else { return nil }
        guard let fused = fusedGateUpStore.get(gate: gateProj, up: upProj) else { return nil }
        // Release the big unfused stacks. A `@ModuleInfo` child cannot be set
        // to nil directly (Module.swift refuses the direct mutation), so both
        // are replaced with 1x1x1 placeholders; the checkpoint-sized arrays go
        // away with their last reference.
        var released = ModuleChildren()
        released["gate_proj"] = .value(Self.placeholderStack())
        released["up_proj"] = .value(Self.placeholderStack())
        update(modules: released)
        return fused
    }

    private static func placeholderStack() -> SwitchLinear {
        SwitchLinear(
            inputDims: 1, outputDims: 1, numExperts: 1, weight: MLXArray.zeros([1, 1, 1]))
    }

    /// One expert's dense `gate`, `up` and `down` weights, in the
    /// `[hiddenDims, inputDims]` / `[inputDims, hiddenDims]` shapes an offload
    /// lane wants. Works whether or not the gate/up stacks have been fused.
    public func denseGateUpDown(expert: Int) -> (MLXArray, MLXArray, MLXArray)? {
        let down = downProj.denseExpertWeight(expert)
        if let gateProj, let upProj, fusedGateUpStore.built == nil {
            return (gateProj.denseExpertWeight(expert), upProj.denseExpertWeight(expert), down)
        }
        guard let fused = gateUpProj ?? fusedGateUpStore.built else { return nil }
        let both = fused.denseExpertWeight(expert)  // [2 * hiddenDims, inputDims]
        return (both[0 ..< hiddenDims, 0...], both[hiddenDims ..< (2 * hiddenDims), 0...], down)
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xGate: MLXArray
        let xUp: MLXArray
        if let gateUpProj {
            // Pre-fused gate_up_proj weight from checkpoint — one gathered
            // matmul via the polymorphic SwitchLinear call, then split.
            let xGateUp = gateUpProj(x, idx, sortedIndices: doSort)
            xGate = xGateUp[.ellipsis, ..<hiddenDims]
            xUp = xGateUp[.ellipsis, hiddenDims...]
        } else if let fused = fusedGateUp() {
            // Separate gate_proj / up_proj checkpoints, concatenated once into a
            // single stack — ONE gathered matmul, split into halves.
            let xGateUp = fused(x, idx, sortedIndices: doSort)
            xGate = xGateUp[.ellipsis, ..<hiddenDims]
            xUp = xGateUp[.ellipsis, hiddenDims...]
        } else {
            // Separate gate_proj / up_proj checkpoints — two gathered matmuls.
            guard let gateProj, let upProj else {
                fatalError("SwitchGLU requires either gate_up_proj or gate_proj/up_proj")
            }
            xUp = upProj(x, idx, sortedIndices: doSort)
            xGate = gateProj(x, idx, sortedIndices: doSort)
        }

        let activated: MLXArray
        if let activationProduct {
            activated = activationProduct(xGate, xUp)
        } else if isSiluActivation {
            activated = compiledSwiGLU(xGate, xUp)
        } else if isGeluActivation {
            activated = compiledGeGLU(xGate, xUp)
        } else {
            activated = activation(xGate) * xUp
        }

        x = downProj(activated, idx, sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    /// Expert `index`'s weight as a dense `[outputDims, inputDims]` array.
    ///
    /// The stacked gather-GEMM never needs one expert on its own, but an
    /// offload lane that compiles a fixed-shape program per expert does.
    /// ``QuantizedSwitchLinear`` overrides this to dequantize; here the stack
    /// is already dense and the expert is a slice.
    public func denseExpertWeight(_ index: Int) -> MLXArray {
        weight[index]
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    /// Init from already-quantized arrays (no re-quantization).
    ///
    /// Used to build a fused `gate_up` stack by concatenating two loaded stacks
    /// along the output-channel axis; re-quantizing there would change the
    /// numbers, so the arrays are adopted as they are.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, scales: MLXArray, biases: MLXArray?, bias: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: inputDims, outputDims: outputDims, numExperts: numExperts,
            weight: weight, bias: bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    override public func denseExpertWeight(_ index: Int) -> MLXArray {
        MLX.dequantized(
            weight[index], scales: scales[index], biases: biases?[index],
            groupSize: groupSize, bits: bits, mode: mode)
    }

    /// The raw packed weight, scales and biases, for cross-module call sites
    /// that need to hand the quantized buffers straight to a kernel (e.g. a
    /// fused routed-MoE Metal kernel) instead of going through
    /// `callAsFunction` or `denseExpertWeight`'s dequantizing gather. `weight`
    /// (inherited from `SwitchLinear`) and `scales`/`biases` stay `internal`
    /// -- `Quantized`'s own contract only requires `groupSize`/`bits`/`mode`
    /// to be public -- so these three computed properties are the sanctioned
    /// public surface for a caller outside `MLXLMCommon`.
    public var quantizedWeight: MLXArray { weight }
    public var quantizedScales: MLXArray { scales }
    public var quantizedBiases: MLXArray? { biases }
}

