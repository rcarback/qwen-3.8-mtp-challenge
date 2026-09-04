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
    private let aneExperts = Qwen4ExpANEExpertLane(label: "mlp.switch_mlp")

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

    /// Router: float32 logits, top-k by `argPartition`, weights = softmax over the
    /// SELECTED logits (equal to softmax-all followed by renormalisation).
    func route(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let logits = Self.routerNative ? gate(x).asType(.float32) : gate(x.asType(.float32))
        let k = min(Self.topKOverride ?? topK, numExperts)
        let kth = numExperts - k
        let idx = MLX.argPartition(logits, kth: kth, axis: -1)[.ellipsis, kth...]
        let w = MLX.softmax(MLX.takeAlong(logits, idx, axis: -1), axis: -1, precise: true)
        Self.reportRouteStats(idx, experts: numExperts)
        return (idx, w)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
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
