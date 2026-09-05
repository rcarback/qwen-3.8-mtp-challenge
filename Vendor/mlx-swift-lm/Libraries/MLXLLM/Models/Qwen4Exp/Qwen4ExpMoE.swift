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
    let hiddenSize: Int
    let moeIntermediateSize: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    private let aneShared = Qwen4ExpANESharedExpertCache(label: "mlp.shared_expert")
    private let aneExperts = Qwen4ExpANEExpertLane(label: "mlp.switch_mlp")

    init(_ args: Qwen4ExpTextConfiguration) {
        topK = args.numExpertsPerTok
        numExperts = args.numExperts
        hiddenSize = args.hiddenSize
        moeIntermediateSize = args.moeIntermediateSize
        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize, numExperts: args.numExperts)
        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
        super.init()
        if Self.fusedRoutedMoE {
            // Ruling C1 part 1: MLX_QWEN4EXP_FUSED_MOE=1 must IMPLY the fused
            // gate_up layout the kernel requires, rather than depend on the
            // operator also remembering MLX_SWITCH_FUSE_GATE_UP=1. Repeated,
            // idempotently, in `fusedRoutedForward` itself so the debug-only
            // per-instance override (`forceFusedRoutedMoE`, set after this
            // initializer has already run) stays honest too.
            switchMLP.forceFuseGateUp = true
        }
    }

    /// `MLX_QWEN4EXP_ROUTE_STATS=1`. Diagnostic only: on the first prefill,
    /// report how unevenly tokens land on experts. The capacity-buffer design
    /// picks its per-expert capacity C from this number, so it has to be
    /// measured rather than assumed. Off by default and never on the hot path.
    static let routeStats: Bool =
        ProcessInfo.processInfo.environment["MLX_QWEN4EXP_ROUTE_STATS"] == "1"
    nonisolated(unsafe) static var routeStatsSeen = 0

    static func reportRouteStats(_ idx: MLXArray, experts: Int) {
        guard routeStats, routeStatsSeen < 480 else { return }
        let flat = idx.flattened()
        guard flat.size >= 512 else { return }  // decode step, not a prefill
        eval(flat)
        var counts = [Int](repeating: 0, count: experts)
        for v in flat.asArray(Int32.self) { counts[Int(v)] += 1 }
        let total = counts.reduce(0, +)
        let mean = Double(total) / Double(experts)
        let maxN = counts.max() ?? 0
        let zero = counts.filter { $0 == 0 }.count
        func overflow(_ c: Int) -> Int { counts.reduce(0) { $0 + max(0, $1 - c) } }
        fputs(String(
            format: "[route-stats] call=%3d rows=%5d mean=%6.2f max=%3d ratio=%5.2f zero=%3d "
                + "overflow@16=%5d(%4.1f%%) @24=%5d(%4.1f%%) @32=%5d(%4.1f%%)\n",
            routeStatsSeen, total, mean, maxN, Double(maxN) / mean, zero,
            overflow(16), 100.0 * Double(overflow(16)) / Double(total),
            overflow(24), 100.0 * Double(overflow(24)) / Double(total),
            overflow(32), 100.0 * Double(overflow(32)) / Double(total)), stderr)
        routeStatsSeen += 1
    }

    /// `MLX_QWEN4EXP_TOPK`. Routes to this many experts per token instead of
    /// the checkpoint's `num_experts_per_tok`. Cutting k is the one way to
    /// reduce routed-expert work that needs no restructuring of the forward:
    /// the gather-GEMM keeps its shape, only narrower. The softmax below
    /// renormalises over whatever is selected, so the weights still sum to
    /// one. This changes model output and is a speed against accuracy dial,
    /// not a free win.
    static let topKOverride: Int? = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_QWEN4EXP_TOPK"],
            let v = Int(raw), v > 0
        else { return nil }
        return v
    }()

    /// `MLX_QWEN4EXP_ROUTER_NATIVE=1`. Run the router GEMM in the activation's
    /// own dtype and cast only the [tokens, 512] logits to float32, instead of
    /// casting the [tokens, 2560] activation first. Selection and the weight
    /// softmax still run in float32. Saves a wide upcast per layer; can change
    /// which experts win a near-tie, so it is off by default.
    static let routerNative: Bool =
        ProcessInfo.processInfo.environment["MLX_QWEN4EXP_ROUTER_NATIVE"] == "1"

    /// `MLX_QWEN4EXP_POOL`. Restrict routing to the first N experts, leaving
    /// top-k unchanged. This does not reduce multiply-accumulates at all: the
    /// same k experts run per token. It only concentrates those rows onto
    /// fewer distinct experts. It therefore isolates one question the pruning
    /// and merging literature depends on: does the gather kernel cost follow
    /// the number of DISTINCT experts touched, or only the number of rows? A
    /// speed change here is the whole speed case for pruning; no change means
    /// pruning is a memory argument only. Quality is destroyed, so this is a
    /// diagnostic, never a shipping mode.
    static let poolLimit: Int? = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_QWEN4EXP_POOL"],
            let v = Int(raw), v > 0
        else { return nil }
        return v
    }()

    /// `MLX_QWEN4EXP_POOL_CALIBRATED`. Keep only the N most POPULAR experts of
    /// this layer, learned from the first prefill this block sees, instead of
    /// the first N by index. This is what the pruning literature actually
    /// specifies, and it is the practical half of expert merging: same speed
    /// mechanism as the naive pool knob, far better selection.
    static let calibratedPool: Int? = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_QWEN4EXP_POOL_CALIBRATED"],
            let v = Int(raw), v > 0
        else { return nil }
        return v
    }()

    /// Per-layer keep-mask, learned once. Additive: 0 for kept, -1e9 for dropped.
    private var calibratedMask: MLXArray?

    /// Learn the mask from one prefill's own routing, then reuse it forever.
    private func calibratedKeepMask(_ idx: MLXArray) -> MLXArray? {
        guard let keep = Self.calibratedPool, keep < numExperts else { return nil }
        if let m = calibratedMask { return m }
        let flat = idx.flattened()
        guard flat.size >= numExperts else { return nil }  // decode step; wait for a prefill
        eval(flat)
        var counts = [Int](repeating: 0, count: numExperts)
        for v in flat.asArray(Int32.self) { counts[Int(v)] += 1 }
        let ranked = counts.enumerated().sorted { $0.element > $1.element }.prefix(keep)
        var allow = Set<Int>(); for r in ranked { allow.insert(r.offset) }
        let m = MLXArray((0 ..< numExperts).map { Float(allow.contains($0) ? 0 : -1e9) })
        eval(m)
        calibratedMask = m
        return m
    }

    /// `MLX_QWEN4EXP_FUSED_MOE`. Routes the gate/up/silu/down quartet through
    /// one persistent Metal kernel (`FusedRoutedMoE`, in `MLXLMCommon` next to
    /// `SwitchLayers.swift`). Default off. Requires the fused `gate_up` stack
    /// and 4-bit affine group-32 expert weights; falls back to the control
    /// path when either is absent, logging the specific reason once (see
    /// `fusedRoutedForward` and Ruling C1: this env var must IMPLY the fused
    /// gate_up layout rather than depend on a second one).
    static let fusedRoutedMoE: Bool =
        ProcessInfo.processInfo.environment["MLX_QWEN4EXP_FUSED_MOE"] == "1"

    #if DEBUG
        /// Per-instance override for ``fusedRoutedMoE``, mirroring
        /// `SwitchGLU.forceFuseGateUp`. `fusedRoutedMoE` is a `static let`
        /// read once per process, so a test that wants to exercise the fused
        /// path in the same process as the control path -- without relying on
        /// environment-variable timing against every other test in this
        /// target that also touches this static -- sets this instead.
        /// Debug-only: never read by the release worker binary.
        var forceFusedRoutedMoE = false
    #endif

    /// One-shot stderr log for a fused-routed-MoE decline. `MLX_QWEN4EXP_FUSED_MOE=1`
    /// makes the fused path mandatory in intent, so silently falling back to
    /// the control path when a guard declines would be indistinguishable from
    /// an honest negative measurement result (Ruling C1 part 2). Logged once
    /// per PROCESS, not once per layer per token -- 48 layers x hundreds of
    /// forwards would flood stderr for a condition that, once true, stays true.
    nonisolated(unsafe) private static var fusedRoutedFallbackLogged = false

    private static func logFusedRoutedFallbackOnce(_ reason: String) {
        guard !fusedRoutedFallbackLogged else { return }
        fusedRoutedFallbackLogged = true
        fputs(
            "[qwen4exp-fused-moe] MLX_QWEN4EXP_FUSED_MOE=1 but \(reason); "
                + "falling back to the stock SwitchGLU path for this and every later "
                + "call -- note this gate already forced the fused gate_up stack, so a "
                + "declining process is NOT a clean control for timing purposes\n",
            stderr)
    }

    /// `MLX_QWEN4EXP_COMPILE_SHARED=0` disables. The shared expert is
    /// `sigmoid(gate(x)) * down(silu(gate_proj(x)) * up(x))`: four matmuls with
    /// a silu, a sigmoid and two products threaded between them, once per layer.
    /// A quantized matmul is already one tuned kernel and compile cannot fuse
    /// inside it, so whatever this wins comes from the elementwise glue only --
    /// which is why it is a separate switch from the read gate and is measured
    /// on its own.
    static let compileShared: Bool =
        ProcessInfo.processInfo.environment["MLX_QWEN4EXP_COMPILE_SHARED"] != "0"

    private var compiledShared: (@Sendable (MLXArray) -> MLXArray)?

    /// The gated shared expert. Same restriction to the decode regime, and for
    /// the same reason, as `Qwen4ExpGatedResidual.mix`.
    func sharedBranch(_ x: MLXArray) -> MLXArray {
        guard Self.compileShared,
            x.shape.dropLast().reduce(1, *) <= Qwen4ExpGatedResidual.compileMaxRows
        else { return sharedBody(x) }
        if compiledShared == nil {
            compiledShared = compile { [self] v in sharedBody(v) }
        }
        return compiledShared!(x)
    }

    private func sharedBody(_ x: MLXArray) -> MLXArray {
        sigmoid(sharedExpertGate(x)) * sharedExpert(x)
    }

    /// `MLX_QWEN4EXP_FUSED_ROUTER=1`. Replace the router's selection tail --
    /// `argPartition`, the slice, `takeAlong` and the precise softmax -- with the
    /// single `FusedRouterSelect` kernel. Off by default: it is a hand-written
    /// kernel standing in for library primitives, and it breaks an exact logit
    /// tie toward the lower expert index where MLX promises nothing, so it has
    /// to earn its way on before it is on.
    static let fusedRouterSelect: Bool =
        ProcessInfo.processInfo.environment["MLX_QWEN4EXP_FUSED_ROUTER"] == "1"

    /// Router: float32 logits, top-k by `argPartition`, weights = softmax over the
    /// SELECTED logits (equal to softmax-all followed by renormalisation).
    func route(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let logits = Self.routerNative ? gate(x).asType(.float32) : gate(x.asType(.float32))
        var logitsMasked = logits
        if Self.calibratedPool != nil {
            // First prefill routes unmasked and teaches the mask; later ones use it.
            let raw = MLX.argPartition(logits, kth: numExperts - min(Self.topKOverride ?? topK, numExperts), axis: -1)[
                .ellipsis, (numExperts - min(Self.topKOverride ?? topK, numExperts))...]
            if let m = calibratedKeepMask(raw) { logitsMasked = logits + m }
        }
        if let pool = Self.poolLimit, pool < numExperts {
            // Push everything outside the pool below any real logit.
            let keep = MLXArray((0 ..< numExperts).map { Float($0 < pool ? 0 : -1e9) })
            logitsMasked = logits + keep
        }
        let k = min(Self.topKOverride ?? topK, numExperts)
        if Self.fusedRouterSelect {
            let lead = Array(logitsMasked.shape.dropLast())
            let (i, w) = FusedRouterSelect.forward(
                logits: logitsMasked.reshaped([-1, numExperts]), topK: k, numExperts: numExperts)
            let idx = i.reshaped(lead + [k])
            Self.reportRouteStats(idx, experts: numExperts)
            return (idx, w.reshaped(lead + [k]))
        }
        let kth = numExperts - k
        let idx = MLX.argPartition(logitsMasked, kth: kth, axis: -1)[.ellipsis, kth...]
        let w = MLX.softmax(MLX.takeAlong(logitsMasked, idx, axis: -1), axis: -1, precise: true)
        Self.reportRouteStats(idx, experts: numExperts)
        return (idx, w)
    }

    /// The fused routed-expert path: one persistent Metal kernel replaces the
    /// gate/up/silu/down launch quartet (`FusedRoutedMoE.forward`, in
    /// `MLXLMCommon`). Returns `nil` -- silently -- when the gate is off, and
    /// -- loudly, once per process via `logFusedRoutedFallbackOnce` -- when
    /// the gate is on but a guard declines, per Ruling C1 parts 2 and 3.
    func fusedRoutedForward(_ x: MLXArray) -> MLXArray? {
        #if DEBUG
            let gateOn = Self.fusedRoutedMoE || forceFusedRoutedMoE
        #else
            let gateOn = Self.fusedRoutedMoE
        #endif
        guard gateOn else { return nil }

        // Idempotent belt-and-suspenders: the initializer already sets this
        // when the process-wide static gate is on (Ruling C1 part 1). Setting
        // it again here, cheaply, also covers the debug-only per-instance
        // override above, which by construction is set AFTER `init` has
        // already run and so could not see it there.
        switchMLP.forceFuseGateUp = true

        guard let gateUpBase = switchMLP.gateUpProj ?? switchMLP.fusedGateUp() else {
            Self.logFusedRoutedFallbackOnce(
                "the fused gate_up stack is unavailable (checkpoint ships separate "
                    + "gate_proj/up_proj and they could not be concatenated -- are the "
                    + "expert stacks quantized?)")
            return nil
        }
        guard let gateUp = gateUpBase as? QuantizedSwitchLinear else {
            Self.logFusedRoutedFallbackOnce(
                "the gate_up stack is not quantized (expected QuantizedSwitchLinear, "
                    + "got \(type(of: gateUpBase)))")
            return nil
        }
        guard let down = switchMLP.downProj as? QuantizedSwitchLinear else {
            Self.logFusedRoutedFallbackOnce(
                "down_proj is not quantized (expected QuantizedSwitchLinear, "
                    + "got \(type(of: switchMLP.downProj)))")
            return nil
        }
        guard gateUp.groupSize == 32, gateUp.bits == 4, gateUp.mode == .affine,
            down.groupSize == 32, down.bits == 4, down.mode == .affine
        else {
            Self.logFusedRoutedFallbackOnce(
                "expert stacks are not 4-bit affine group-32 (gate_up: group=\(gateUp.groupSize) "
                    + "bits=\(gateUp.bits) mode=\(gateUp.mode); down: group=\(down.groupSize) "
                    + "bits=\(down.bits) mode=\(down.mode))")
            return nil
        }
        guard let gb = gateUp.quantizedBiases, let db = down.quantizedBiases else {
            Self.logFusedRoutedFallbackOnce("expert stacks are missing quantization biases")
            return nil
        }
        // The kernel's Metal source hard-codes `half` for the activation and
        // for every scale/bias pointer, while MLX generates the signature from
        // the ACTUAL input dtypes (`get_type_string` in compiled.cpp maps
        // bfloat16 to the distinct type `bfloat16_t`). A bf16 checkpoint --
        // which the reference Qwen3.8-Flash-Next tree is, `config.json` says
        // `"dtype": "bfloat16"` and the switch_mlp scales/biases are BF16 --
        // therefore does not decline here without this guard, it aborts the
        // process inside the Metal JIT. Decline loudly and by name instead.
        guard x.dtype == .float16, gateUp.quantizedScales.dtype == .float16,
            gb.dtype == .float16, down.quantizedScales.dtype == .float16,
            db.dtype == .float16
        else {
            Self.logFusedRoutedFallbackOnce(
                "the kernel requires fp16 activations and fp16 scales/biases, but got "
                    + "x=\(x.dtype), gate_up scales=\(gateUp.quantizedScales.dtype)/"
                    + "biases=\(gb.dtype), down scales=\(down.quantizedScales.dtype)/"
                    + "biases=\(db.dtype) -- a bf16 checkpoint needs a kernel templated "
                    + "on the element type, not a cast")
            return nil
        }

        let (idx, w) = route(x)
        // Match SwitchGLU.callAsFunction's own expand-then-gatherSort shape
        // exactly (SwitchLayers.swift): expand to [..., 1, 1, D] first so
        // gatherSort's `flattened(start:0,end:-3)` collapses batch/sequence
        // into one row axis of length rows = tokens * topK, one row per
        // (token, selected expert) pair, in the same order SwitchGLU uses.
        let xExpanded = MLX.expandedDimensions(x, axes: [-2, -3])
        let (xSorted, sortedIdx, invOrder) = gatherSort(x: xExpanded, indices: idx)
        let flat = xSorted.reshaped(xSorted.dim(0), -1)

        let rowOffsets = MoEWorkQueue.rowOffsets(sortedIndices: sortedIdx, numExperts: numExperts)
        let blockOffsets = MoEWorkQueue.blockOffsets(
            rowOffsets: rowOffsets, blockRows: FusedRoutedMoE.blockRows)

        let ySorted = FusedRoutedMoE.forward(
            xSorted: flat,
            gateUpWeight: gateUp.quantizedWeight, gateUpScales: gateUp.quantizedScales,
            gateUpBiases: gb,
            downWeight: down.quantizedWeight, downScales: down.quantizedScales, downBiases: db,
            rowOffsets: rowOffsets, blockOffsets: blockOffsets,
            hiddenDim: moeIntermediateSize, inDim: hiddenSize, numExperts: numExperts)

        let y = scatterUnsort(x: ySorted, invOrder: invOrder, shape: idx.shape)
        let routed = (y * w.expandedDimensions(axis: -1).asType(y.dtype)).sum(axis: -2)
        return routed.asType(x.dtype) + sharedBranch(x)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let out = fusedRoutedForward(x) { return out }
        if Qwen4ExpANEFused.expertsEnabled, Qwen4ExpANEFused.armed(tokens: x.dim(1), batch: x.dim(0)) {
            do {
                if let out = try groupedExpertForward(x) { return out }
            } catch {
                // Unconditional, like the shared lane's: a repeating run
                // failure otherwise silently doubles the routed work.
                fputs("[qwen4exp-ane] grouped expert run failed: \(error); GPU path for this call\n", stderr)
            }
        }
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
                // Unconditional: a repeating run failure recomputes the routed
                // sum below on every call, and that doubled work must be visible
                // without the log flag.
                fputs("[qwen4exp-ane] shared expert run failed: \(error); GPU path for this call\n", stderr)
            }
        }
        let (idx, w) = route(x)
        let y = switchMLP(x, idx)  // [B, S, k, D]
        let routed = (y * w.expandedDimensions(axis: -1).asType(y.dtype)).sum(axis: -2)
        return routed.asType(x.dtype) + sharedBranch(x)
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
        let sharedWide = shared.asType(x.dtype).reshaped(x.shape)
        let out = gpu.0 + gpu.1 * sharedWide
        if Qwen4ExpANEFused.zeroCopyReadback { eval(out) }  // BARRIER 3, zero-copy only

        if Qwen4ExpANEFused.verify {
            // MLX_QWEN4EXP_ANE_VERIFY=1 (diagnostic): the pure-GPU shared
            // expert this call substitutes for, logged as a per-call max-abs
            // error against the ANE-produced partial. Doubles the shared
            // expert's work; never use for timing.
            let ref = sharedExpert(x).asType(.float32)
            let diff = MLX.abs(sharedWide.asType(.float32) - ref)
            let maxAbs = diff.max()
            let maxRef = MLX.abs(ref).max()
            eval(maxAbs, maxRef)
            aneLog(String(
                format: "shared verify T=%d: maxAbs=%.4f max|ref|=%.2f",
                tokens, maxAbs.item(Float.self), maxRef.item(Float.self)))
        }

        return out
    }

    // MARK: - Grouped routed-expert ANE lane

    /// The hot experts' SwiGLU runs on the ANE as `programs` grouped
    /// fixed-shape programs while the GPU runs the router's gather-GEMM and
    /// the shared expert. ONE join per layer.
    ///
    /// What this does and does not save. The ANE takes real expert arithmetic
    /// off the critical path, but the GPU's `switchMLP` gather keeps its
    /// `[B, S, topK]` shape: the assignments the ANE served are neutralised by
    /// zeroing their combine weight, not by shrinking the gather. Removing
    /// them from the gather needs a variable-length compaction, whose dynamic
    /// shapes fragment exactly the fused lazy graph the micro-batch arm was
    /// measured losing 25% to. So this lane overlaps work; it does not yet
    /// delete GPU work.
    func groupedExpertForward(_ x: MLXArray) throws -> MLXArray? {
        let tokens = x.dim(1)
        let hidden = x.dim(2)
        let (idx, w) = route(x)
        let indices2D = idx.reshaped(tokens, topK)

        guard
            let resolved = aneExperts.resolve(
                indices: indices2D, numExperts: numExperts,
                dequantizedExpert: { expert in
                    // Reads the gate/up halves out of the fused stack when
                    // MLX_SWITCH_FUSE_GATE_UP released the unfused children.
                    self.switchMLP.denseGateUpDown(expert: expert)!
                })
        else { return nil }

        let group = resolved.programs[0].groupSize
        let capacity = resolved.programs[0].capacity
        let plan = Qwen4ExpANEExpertPlanner.plan(
            indices: indices2D, slotOf: resolved.slotOf, hotCount: resolved.hotCount,
            capacity: capacity)
        let packed = Qwen4ExpANEExpertPlanner.gather(x.reshaped(tokens, hidden), plan: plan)

        // CALLER THREAD. BARRIER 1: staging evals inside `prepare`.
        var prepared: [ANEDirectDispatch.Prepared] = []
        prepared.reserveCapacity(resolved.programs.count)
        for (p, program) in resolved.programs.enumerated() {
            // [G*C, hidden] expert-major -> [C, G*hidden], the grouped conv's
            // channel-block layout.
            let lo = p * group * capacity
            let hi = lo + group * capacity
            let slice: MLXArray = packed[lo ..< hi, 0...]
            let laid: MLXArray = slice.reshaped(group, capacity, hidden).transposed(1, 0, 2)
                .reshaped(capacity, group * hidden)
            prepared.append(try program.makeInput(laid))
        }

        let (_, gpu) = try ConcurrentEngines.run(
            ane: { () -> Bool in  // BACKGROUND, no MLX
                for (p, program) in resolved.programs.enumerated() {
                    try program.predict(prepared[p])
                }
                return true
            },
            gpu: { () -> (MLXArray, MLXArray) in  // CALLER THREAD
                let masked = w * (1 - plan.handled.reshaped(w.shape).asType(w.dtype))
                let y = self.switchMLP(x, idx)
                let routed = (y * masked.expandedDimensions(axis: -1).asType(y.dtype))
                    .sum(axis: -2).asType(x.dtype)
                let shared = sigmoid(self.sharedExpertGate(x)) * self.sharedExpert(x)
                eval(routed, shared)  // BARRIER 2, mandatory
                return (routed, shared)
            })

        // CALLER THREAD: read back, undo the channel-block layout, combine.
        var blocks: [MLXArray] = []
        blocks.reserveCapacity(resolved.programs.count)
        for (p, program) in resolved.programs.enumerated() {
            let out = program.readOutput(prepared[p])  // [C, G*hidden]
            blocks.append(
                out.reshaped(capacity, group, hidden).transposed(1, 0, 2)
                    .reshaped(group * capacity, hidden))
        }
        let packedOut = blocks.count == 1 ? blocks[0] : concatenated(blocks, axis: 0)
        let aneSum = Qwen4ExpANEExpertPlanner.combine(
            packedOut, plan: plan, weights: w.reshaped(tokens, topK), tokens: tokens)
        return gpu.0 + aneSum.asType(x.dtype).reshaped(x.shape) + gpu.1
    }
}
