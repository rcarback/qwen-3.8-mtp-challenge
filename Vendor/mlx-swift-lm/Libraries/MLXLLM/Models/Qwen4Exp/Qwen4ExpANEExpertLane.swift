// LOCAL M4 FORK ONLY. Private Apple frameworks; breaks on macOS updates.
// NPUMoE-style grouped routed-expert execution for the Qwen4Exp MoE block.
// Gated by MLX_ANE_DIRECT=1 plus MLX_QWEN4EXP_ANE_MODE=experts at the call site.
//
// Reference: "Efficient Mixture-of-Experts LLM Inference with Apple Silicon
// NPUs" (arXiv 2604.18788). Three ideas are taken from it:
//
//  1. STATIC CAPACITY. The ANE cannot take a dynamic per-expert token count,
//     so every hot expert gets a fixed capacity `C` (one tier, not several).
//  2. GROUPED EXECUTION. `G` experts share one fixed-shape program over a
//     `[G*C, hidden]` buffer viewed as `G` contiguous capacity slices.
//     `ANEGroupedExpertMLP` implements that as one grouped conv per stage.
//  3. LOAD-AWARE RESIDENCY. Only the hottest experts of a layer are made
//     ANE-resident; every other expert stays on the GPU.
//
// Two deliberate departures from the paper:
//
//  * Overflow past `C` is NOT pruned by activation saliency. It stays on the
//    GPU path. This tower has no accuracy budget to spend and a spill is both
//    simpler and exact.
//  * Hot-expert selection is ONLINE, not calibrated offline. The first armed
//    prefill chunk's own routing histogram picks the hot set, and that choice
//    is then frozen for the process. This is an approximation of the paper's
//    calibration pass: it uses one chunk of real traffic instead of a
//    calibration corpus, and it is legitimate because the choice depends only
//    on the routing the model itself produced, never on hidden prompts or on
//    the harness protocol, and because it can only change WHICH engine an
//    expert runs on, never the arithmetic that expert computes.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The static-shape gather/scatter plan for one grouped MoE forward. Every
/// field is an index array; no CPU synchronisation is involved, so the plan
/// stays inside the layer's lazy graph.
struct Qwen4ExpGroupedRouting {
    /// Hot experts, i.e. `programs * groupSize`.
    let hotCount: Int
    /// Rows per hot expert.
    let capacity: Int
    /// `[hotCount * capacity]` int32. Packed row -> token index, or `tokens`
    /// (the zero sink row) for a capacity slot no token reached.
    let source: MLXArray
    /// `[tokens * topK]` int32. Assignment -> packed row, or
    /// `hotCount * capacity` (the sink) for an assignment the ANE does not
    /// serve: a cold expert, or a hot expert past its capacity.
    let destination: MLXArray
    /// `[tokens, topK]` bool. True where the ANE serves the assignment, so the
    /// GPU leg must zero that combine weight.
    let handled: MLXArray
}

enum Qwen4ExpANEExpertPlanner {
    /// Builds the plan. `indices` is `[tokens, topK]` expert ids; `slotOf` is
    /// `[numExperts]` int32 mapping an expert id to its position in the hot
    /// list, or `-1` when the expert is cold.
    ///
    /// The whole plan is sort-and-gather, never scatter: assignments are
    /// argsorted by hot slot, a slot's rank inside its run gives the capacity
    /// row, and both directions are then read back with `take`.
    static func plan(
        indices: MLXArray, slotOf: MLXArray, hotCount: Int, capacity: Int
    ) -> Qwen4ExpGroupedRouting {
        precondition(indices.ndim == 2, "plan expects indices [tokens, topK], got \(indices.shape)")
        let tokens = indices.dim(0)
        let topK = indices.dim(1)
        let n = tokens * topK
        let sink = Int32(hotCount * capacity)

        let slot = take(slotOf, indices.flattened().asType(.int32))  // [N], -1 when cold
        // Cold assignments are keyed to the one-past-the-end bin so the sort
        // pushes them behind every hot run.
        let key = MLX.which(slot .>= MLXArray(Int32(0)), slot, MLXArray(Int32(hotCount)))
        let order = argSort(key).asType(.int32)  // [N]
        let sortedKey = take(key, order)  // [N], ascending

        let bins = MLXArray.arange(hotCount + 1).asType(.int32)  // [hotCount+1]
        let counts = (key.expandedDimensions(axis: 1) .== bins.expandedDimensions(axis: 0))
            .asType(.int32).sum(axis: 0)  // [hotCount+1]
        let start = cumsum(counts, axis: 0) - counts  // exclusive prefix

        let positions = MLXArray.arange(n).asType(.int32)
        let rankSorted = positions - take(start, sortedKey)  // rank inside the run
        let servedSorted =
            (sortedKey .< MLXArray(Int32(hotCount))) .&& (rankSorted .< MLXArray(Int32(capacity)))
        let destinationSorted = MLX.which(
            servedSorted, sortedKey * MLXArray(Int32(capacity)) + rankSorted, MLXArray(sink))

        let inverseOrder = argSort(order).asType(.int32)
        let destination = take(destinationSorted, inverseOrder)  // [N], original order
        let handled = (destination .< MLXArray(sink)).reshaped(tokens, topK)

        // Reverse direction: packed row -> token. Row `d` of slot `j` at rank
        // `r` was filled by sorted position `start[j] + r`, and exists only
        // while `r` is below that slot's count.
        let rows = MLXArray.arange(hotCount * capacity).asType(.int32)
        let slotOfRow = rows.floorDivide(MLXArray(Int32(capacity)))
        let rankOfRow = rows - slotOfRow * MLXArray(Int32(capacity))
        let sortedPosition = take(start, slotOfRow) + rankOfRow
        let filled = rankOfRow .< take(counts, slotOfRow)
        let clamped = clip(sortedPosition, min: MLXArray(Int32(0)), max: MLXArray(Int32(max(n - 1, 0))))
        let tokenOfRow = take(order, clamped).floorDivide(MLXArray(Int32(topK)))
        let source = MLX.which(filled, tokenOfRow, MLXArray(Int32(tokens)))

        return Qwen4ExpGroupedRouting(
            hotCount: hotCount, capacity: capacity, source: source, destination: destination,
            handled: handled)
    }

    /// `[hotCount * capacity, hidden]`, expert-major: rows
    /// `[j*capacity, (j+1)*capacity)` are hot slot `j`'s tokens, zero-padded.
    static func gather(_ xFlat: MLXArray, plan: Qwen4ExpGroupedRouting) -> MLXArray {
        let hidden = xFlat.dim(1)
        let padded = concatenated([xFlat, MLXArray.zeros([1, hidden], dtype: xFlat.dtype)], axis: 0)
        return take(padded, plan.source, axis: 0)
    }

    /// Weighted scatter back, done as a gather: every assignment reads its own
    /// packed row, is scaled by its router weight, and the `topK` axis is
    /// summed. `packedOut` is `[hotCount * capacity, hidden]`, `weights` is
    /// `[tokens, topK]`. Returns `[tokens, hidden]`.
    static func combine(
        _ packedOut: MLXArray, plan: Qwen4ExpGroupedRouting, weights: MLXArray, tokens: Int
    ) -> MLXArray {
        let hidden = packedOut.dim(1)
        let topK = weights.dim(1)
        let padded = concatenated(
            [packedOut, MLXArray.zeros([1, hidden], dtype: packedOut.dtype)], axis: 0)
        let rows = take(padded, plan.destination, axis: 0)  // [N, hidden]
        let scale = (weights * plan.handled.asType(weights.dtype)).flattened()
        return (rows * scale.expandedDimensions(axis: -1).asType(rows.dtype))
            .reshaped(tokens, topK, hidden).sum(axis: 1)
    }
}

/// Per-MoE-block state: the frozen hot-expert choice and its grouped programs.
/// `NSLock` guarded, matching the sibling lanes.
final class Qwen4ExpANEExpertLane: @unchecked Sendable {
    private let lock = NSLock()
    private var built = false
    /// `[numExperts]` int32 expert id -> hot slot, `-1` when cold.
    private var slotOf: MLXArray?
    private var programs: [ANEGroupedExpertMLP] = []
    private var hotCount = 0
    let label: String

    init(label: String) { self.label = label }

    /// Hot slots this lane serves; 0 until the first successful build.
    var servedExperts: Int {
        lock.lock()
        defer { lock.unlock() }
        return hotCount
    }

    /// Returns the frozen plan inputs, building the programs on the first
    /// armed call. `nil` means "stay on the GPU" and is cached: a refusal is
    /// paid once, not on every forward.
    ///
    /// `dequantizedExpert` returns `(gate, up, down)` for one expert id, as
    /// `[inter, hidden]` / `[inter, hidden]` / `[hidden, inter]`.
    func resolve(
        indices: MLXArray, numExperts: Int,
        dequantizedExpert: (Int) -> (MLXArray, MLXArray, MLXArray)
    ) -> (slotOf: MLXArray, programs: [ANEGroupedExpertMLP], hotCount: Int)? {
        lock.lock()
        defer { lock.unlock() }
        if built {
            guard let slotOf, !programs.isEmpty else { return nil }
            return (slotOf, programs, hotCount)
        }
        built = true

        let group = Qwen4ExpANEFused.expertGroupSize
        let wanted = (Qwen4ExpANEFused.expertHotPerLayer / group) * group
        guard wanted > 0, wanted <= numExperts else {
            aneLog("expert lane \(label): hot=\(wanted) group=\(group) is not a usable split; GPU path")
            return nil
        }
        guard Qwen4ExpANEFused.admitExpertLayer() else {
            fputs(
                "[qwen4exp-ane] expert lane \(label): layer cap \(Qwen4ExpANEFused.expertMaxLayers) reached; GPU path\n",
                stderr)
            return nil
        }

        // ONLINE hot-expert selection (see the file header). One histogram of
        // this chunk's own routing, then the `wanted` busiest experts. The
        // `item`/`asArray` read below is the only CPU synchronisation in the
        // lane and it happens once per layer per process.
        let flat = indices.flattened().asType(.int32)
        let bins = MLXArray.arange(numExperts).asType(.int32)
        let counts = (flat.expandedDimensions(axis: 1) .== bins.expandedDimensions(axis: 0))
            .asType(.int32).sum(axis: 0)  // [numExperts]
        let kth = numExperts - wanted
        let hottest = MLX.argPartition(counts, kth: kth, axis: -1)[kth...]
        eval(hottest)
        let hotIds = hottest.asArray(Int32.self).map(Int.init)

        var slotTable = [Int32](repeating: -1, count: numExperts)
        for (slot, expert) in hotIds.enumerated() { slotTable[expert] = Int32(slot) }

        let capacity = Qwen4ExpANEFused.expertCapacity
        var builtPrograms: [ANEGroupedExpertMLP] = []
        for base in stride(from: 0, to: wanted, by: group) {
            var gate: [MLXArray] = []
            var up: [MLXArray] = []
            var down: [MLXArray] = []
            for offset in 0 ..< group {
                let (g, u, d) = dequantizedExpert(hotIds[base + offset])
                gate.append(g)
                up.append(u)
                down.append(d)
            }
            let bytes = 3 * group * gate[0].dim(0) * gate[0].dim(1) * 2
            guard Qwen4ExpANEFused.reserveProgram(bytes: bytes, label: "\(label)#\(base / group)") else {
                break  // reserveProgram already logged
            }
            do {
                builtPrograms.append(
                    try ANEGroupedExpertMLP(gate: gate, up: up, down: down, capacity: capacity))
            } catch {
                Qwen4ExpANEFused.releaseProgram(bytes: bytes)
                fputs(
                    "[qwen4exp-ane] EXPERT BUILD FAILED \(label) group=\(base / group): \(error); GPU path stays\n",
                    stderr)
                break
            }
        }

        guard !builtPrograms.isEmpty else { return nil }
        // A partial build is honoured at the groups that did build: the hot
        // experts past that point go back to cold so the plan never routes a
        // token to a program that does not exist.
        hotCount = builtPrograms.count * group
        for (slot, expert) in hotIds.enumerated() where slot >= hotCount { slotTable[expert] = -1 }
        let table = MLXArray(slotTable)
        eval(table)
        slotOf = table
        programs = builtPrograms
        if Qwen4ExpANEFused.log {
            fputs(
                "[qwen4exp-ane] expert lane \(label): \(builtPrograms.count) program(s), hot=\(hotCount)/\(numExperts), C=\(capacity)\n",
                stderr)
        }
        return (table, builtPrograms, hotCount)
    }
}
