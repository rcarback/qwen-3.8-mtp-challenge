import Foundation
import MLX
import MLXLLM
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel

/// Where does prefill and decode time actually go?  One opt-in instrument,
/// three tables:
///
///   1. Per-layer attribution of a 1024-token prefill chunk (depth 0 and
///      ~8k), grouped 48 linear vs 16 full-attention blocks, via the
///      profiling seam in the vendored model.  The seam is unfused and
///      synced per layer, so its SHARES are the signal and the separately
///      measured fused total is the truth they scale to.
///   2. Decode-shaped verify-width sweep: cost of one forward of M rows,
///      M in 1..32, with the host graph-build and GPU eval halves split so
///      a debug-built host does not masquerade as GPU time.
///   3. The MTP head draft chain and the lm_head projection, the two
///      non-backbone costs a decode round pays.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
///     MLXFAST_QWEN_PREFILL_WEIGHTS=weights \
///     MLXFAST_QWEN_PREFILL_HEAD=<head dir> \
///     swift test --force-resolved-versions --filter phaseBreakdown
@Suite(.serialized)
struct QwenPhaseBreakdownTests {
    @Test("prefill and decode phase breakdown")
    func phaseBreakdown() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            let weights = env["MLXFAST_QWEN_PREFILL_WEIGHTS"],
            let head = env["MLXFAST_QWEN_PREFILL_HEAD"]
        else { return }

        let targetURL = URL(fileURLWithPath: weights)
        let headURL = URL(fileURLWithPath: head)
        let context = try Qwen36MTPHeadAttachment.withHeadAttached(
            backboneDirectory: targetURL, headDirectory: headURL
        ) { _ in
            let box = UnsafeSendableBox<ModelContext>()
            let semaphore = DispatchSemaphore(value: 0)
            Task {
                box.value = try? await LLMModelFactory.shared.load(
                    from: targetURL, using: #huggingFaceTokenizerLoader())
                semaphore.signal()
            }
            semaphore.wait()
            guard let loaded = box.value else {
                throw MLXFastError.invalidInput("failed to load backbone")
            }
            return loaded
        }
        guard let model = context.model as? any Qwen36MTPTarget else {
            Issue.record("backbone is not an MTP target")
            return
        }

        func tokens(_ count: Int, offset: Int) -> MLXArray {
            MLXArray((0 ..< count).map {
                Int32((($0 + offset) * 7919) % 100_000 + 10)
            }).reshaped([1, count])
        }

        /// Fused production forward of one appended chunk, one eval at the
        /// end: the truth the profiled shares scale to.
        @discardableResult
        func fusedChunk(
            cache: [any KVCache], width: Int, offset: Int
        ) -> Double {
            let start = Date()
            let (_, hidden) = model.callWithHidden(
                input: LMInput.Text(tokens: tokens(width, offset: offset)),
                cache: cache, nConfirmed: 0)
            eval(hidden)
            return Date().timeIntervalSince(start)
        }

        func profiledChunk(
            cache: [any KVCache], width: Int, offset: Int
        ) -> Qwen35LayerProfile? {
            let input = tokens(width, offset: offset)
            if let top = context.model as? MLXLLM.Qwen35Model {
                return top.profiledLayerForward(input, cache: cache)
            }
            if let text = context.model as? Qwen35TextModel {
                return text.profiledLayerForward(input, cache: cache)
            }
            return nil
        }

        func report(
            _ label: String, _ profile: Qwen35LayerProfile,
            fusedSeconds: Double
        ) {
            let linearTotal = zip(profile.layerSeconds, profile.layerIsLinear)
                .filter { $0.1 }.map { $0.0 }.reduce(0, +)
            let fullTotal = zip(profile.layerSeconds, profile.layerIsLinear)
                .filter { !$0.1 }.map { $0.0 }.reduce(0, +)
            let synced = profile.layerSeconds.reduce(0, +)
                + profile.embedSeconds
            print("\n[\(label)] per-layer attribution (synced, unfused)")
            print(String(
                format: "  embed              %8.1f ms",
                1000 * profile.embedSeconds))
            print(String(
                format: "  48 linear blocks   %8.1f ms  (%4.1f%%)  mean %6.2f ms",
                1000 * linearTotal, 100 * linearTotal / synced,
                1000 * linearTotal / 48))
            print(String(
                format: "  16 full blocks     %8.1f ms  (%4.1f%%)  mean %6.2f ms",
                1000 * fullTotal, 100 * fullTotal / synced,
                1000 * fullTotal / 16))
            print(String(
                format: "  synced total       %8.1f ms   fused truth %8.1f ms"
                    + "  (sync tax %+.1f%%)",
                1000 * synced, 1000 * fusedSeconds,
                100 * (synced - fusedSeconds) / fusedSeconds))
            let ranked = profile.layerSeconds.enumerated()
                .sorted { $0.element > $1.element }.prefix(6)
            let rows = ranked.map { pair in
                String(
                    format: "L%02d/%@ %.1fms", pair.offset,
                    profile.layerIsLinear[pair.offset] ? "lin" : "FUL",
                    1000 * pair.element)
            }.joined(separator: "  ")
            print("  slowest: \(rows)")
        }

        func widthSweep(
            _ label: String, cache: [any KVCache], widths: [Int],
            filled: inout Int
        ) {
            print("\n[\(label)] verify-width sweep, 3 reps, best shown")
            print("     M   build_ms   eval_ms   total_ms    ms/row    vs M=1")
            var baseline = 0.0
            for m in widths {
                var best = (
                    build: 0.0, eval: 0.0,
                    total: Double.greatestFiniteMagnitude)
                for _ in 0 ..< 3 {
                    let t0 = Date()
                    let (logits, _) = model.callWithHidden(
                        input: LMInput.Text(tokens: tokens(m, offset: filled)),
                        cache: cache, nConfirmed: 0)
                    let t1 = Date()
                    eval(logits)
                    let t2 = Date()
                    filled += m
                    let total = t2.timeIntervalSince(t0)
                    if total < best.total {
                        best = (
                            t1.timeIntervalSince(t0),
                            t2.timeIntervalSince(t1), total)
                    }
                }
                if m == widths.first { baseline = best.total }
                print(String(
                    format: "  %4d  %9.1f  %8.1f  %9.1f  %8.1f  %7.2fx",
                    m, 1000 * best.build, 1000 * best.eval,
                    1000 * best.total, 1000 * best.total / Double(m),
                    best.total / baseline))
            }
        }

        // ---- depth 0: fused truth, then per-layer shares, fresh caches ----
        let fused0 = fusedChunk(
            cache: model.newCache(parameters: nil), width: 1024, offset: 0)
        if let p = profiledChunk(
            cache: model.newCache(parameters: nil), width: 1024, offset: 0) {
            report("prefill 1024 @ depth 0", p, fusedSeconds: fused0)
        }

        // ---- one shared cache from here on; depth grows as we measure ----
        let cache = model.newCache(parameters: nil)
        var filled = 0
        while filled < 2048 {
            fusedChunk(cache: cache, width: 1024, offset: filled)
            filled += 1024
        }

        // Per-layer attribution at the verify widths this plan optimizes.
        // The seam is unfused and synced per layer (Qwen35.swift:6306), so
        // treat the SHARES as the signal, not the totals: it runs the plain
        // layer call rather than the boundary-fused chain the production
        // forward uses.
        for m in [16, 32] {
            let fused = fusedChunk(cache: cache, width: m, offset: filled)
            filled += m
            if let p = profiledChunk(cache: cache, width: m, offset: filled) {
                filled += m
                report("verify width \(m) @ depth ~2k", p, fusedSeconds: fused)
            }
        }

        // Mask hypothesis: `createSSMMask` returns nil without left padding or
        // lengths (KVCache.swift:1397-1405), and the full-attention mask is
        // symbolic `.causal` unless a subclass overrides `makeMask`
        // (KVCache.swift:160-174). Name the concrete types so the reading is
        // checked against the objects the serve path actually builds.
        do {
            let probe = MLXArray.zeros([1, 32], dtype: .int32)
            let ssm = createSSMMask(h: probe, cache: cache[0] as? MambaCache)
            let fa = createAttentionMask(h: probe, cache: cache[3])
            print("\n[masks at width 32] "
                + "ssm=\(ssm == nil ? "nil" : "array\(ssm!.shape)")  "
                + "fa=\(fa)  "
                + "ssm cache=\(type(of: cache[0]))  "
                + "fa cache=\(type(of: cache[3]))")
        }

        // Tape hypothesis: one verify-shaped forward, then total the bytes the
        // gated-delta caches retain for replay.
        do {
            let (logits, _) = model.callWithHidden(
                input: LMInput.Text(tokens: tokens(32, offset: filled)),
                cache: cache, nConfirmed: 1)
            eval(logits)
            filled += 32
            var tapeBytes = 0
            var tapedLayers = 0
            for entry in cache {
                guard let mamba = entry as? MambaCache,
                      let tape = mamba.prefixReplayTape
                else { continue }
                tapedLayers += 1
                for array in [
                    tape.convInput, tape.q, tape.k, tape.v,
                    tape.a, tape.b, tape.g, tape.beta,
                ] {
                    tapeBytes += array.nbytes
                }
                if let pre = tape.ssmPre { tapeBytes += pre.nbytes }
            }
            print(String(
                format: "[replay tape at width 32] %d layers, %.1f MB retained",
                tapedLayers, Double(tapeBytes) / 1_048_576))
        }

        let ladderBand = ProcessInfo.processInfo
            .environment["MLX_QWEN_MTP_LADDER_MAXWIDTH"] ?? "<unset, 9>"
        print("\n[qmv arm] MLX_E120_QMV_ARM="
            + (ProcessInfo.processInfo.environment["MLX_E120_QMV_ARM"]
                ?? "<unset, shipped sumtable>")
            + "  [ladder band] MLX_QWEN_MTP_LADDER_MAXWIDTH=\(ladderBand)")

        widthSweep(
            "decode @ depth ~2k", cache: cache,
            widths: [1, 2, 3, 4, 8, 9, 12, 16, 24, 32, 33, 48],
            filled: &filled)

        // ---- MTP head draft chain, decomposed ----
        if model.hasMTPHead {
            let (logits, hidden) = model.callWithHidden(
                input: LMInput.Text(tokens: tokens(1, offset: filled)),
                cache: cache, nConfirmed: 0)
            filled += 1
            let d = logits.dim(1)
            var next = argMax(logits[0..., (d - 1) ..< d, 0...], axis: -1)
                .asType(.int32)
            var h = model.applyFinalNorm(hidden)
            eval(h, next)

            let mtpCache = model.makeMTPCache()
            let cost = Qwen36MTPHeadCost.pinnedQwen38Head
            let projectionBytes = Qwen36MTPHeadCost.projectionBytes(
                rows: 98_336, hiddenSize: 5_120, bits: 4, groupSize: 64)

            // Arm 1: the flush step the serve path takes, through the
            // key/value-only history call, with a three-row flush.  The head
            // cache is empty here, so this is also what a fresh round pays.
            let flushRows = 3
            let flushHidden = concatenated(
                (0 ..< flushRows).map { _ in h }, axis: 1)
            let flushTokens = concatenated(
                (0 ..< flushRows).map { _ in next }, axis: 1)
            let tFlush0 = Date()
            let flushOut = model.mtpHeadLastHiddenWithKVOnlyHistory(
                hidden: flushHidden, nextTokenIds: flushTokens,
                cache: mtpCache)
            let usedKVOnlyPath = flushOut != nil
            let flushResult = flushOut
                ?? model.mtpHeadHiddenForward(
                    hidden: flushHidden, nextTokenIds: flushTokens,
                    cache: mtpCache)
            let tFlush1 = Date()
            eval(flushResult)
            let tFlush2 = Date()
            h = flushResult[0..., (flushResult.dim(1) - 1)..., 0...]
            next = model.draftTokenID(h)
            eval(next)

            // Arm 2: seven pure one-row chain steps, each with its own build
            // and eval boundary, head forward and draft projection separated.
            var headBuilds: [Double] = []
            var headEvals: [Double] = []
            var projectionBuilds: [Double] = []
            var projectionEvals: [Double] = []
            for _ in 0 ..< 7 {
                let t0 = Date()
                let stepHidden = model.mtpHeadHiddenForward(
                    hidden: h, nextTokenIds: next, cache: mtpCache)
                let t1 = Date()
                eval(stepHidden)
                let t2 = Date()
                let id = model.draftTokenID(stepHidden)
                let t3 = Date()
                eval(id)
                let t4 = Date()
                headBuilds.append(t1.timeIntervalSince(t0))
                headEvals.append(t2.timeIntervalSince(t1))
                projectionBuilds.append(t3.timeIntervalSince(t2))
                projectionEvals.append(t4.timeIntervalSince(t3))
                h = stepHidden
                next = id
            }

            // Arm 3: the shipped shape, eight steps built lazily and evaluated
            // once, so this decomposition stays comparable to the number the
            // earlier revision of this instrument reported.
            let chainCache = model.makeMTPCache()
            var chainHidden = h
            var chainID = next
            let tChain0 = Date()
            for _ in 0 ..< 8 {
                chainHidden = model.mtpHeadHiddenForward(
                    hidden: chainHidden, nextTokenIds: chainID,
                    cache: chainCache)
                chainID = model.draftTokenID(chainHidden)
            }
            eval(chainID)
            let chainSeconds = Date().timeIntervalSince(tChain0)

            func median(_ values: [Double]) -> Double {
                let sorted = values.sorted()
                return sorted[sorted.count / 2]
            }
            let headBuild = median(headBuilds)
            let headEval = median(headEvals)
            let projectionBuild = median(projectionBuilds)
            let projectionEval = median(projectionEvals)
            let stepSeconds =
                headBuild + headEval + projectionBuild + projectionEval
            let historyRows = mtpCache.first?.offset ?? 0
            let headBytes = cost.headModuleBytes
                + historyRows * cost.headKVBytesPerRow

            func rate(_ bytes: Int, _ seconds: Double) -> Double {
                Double(bytes) / seconds / 1_000_000_000
            }

            print(String(
                format: """

                    [MTP head @ depth ~2k] head history %d rows, \
                    key/value-only path %@
                      flush step (%d rows)  build %6.2f ms   eval %6.2f ms
                      head module          build %6.2f ms   eval %6.2f ms   \
                    %6.1f GB/s
                      draft projection     build %6.2f ms   eval %6.2f ms   \
                    %6.1f GB/s
                      one step             %6.2f ms   host share %4.1f%%   \
                    %6.1f GB/s
                      8-step lazy chain    %6.2f ms   (%.2f ms/draft)
                    """,
                historyRows, usedKVOnlyPath ? "taken" : "NOT taken",
                flushRows,
                1000 * tFlush1.timeIntervalSince(tFlush0),
                1000 * tFlush2.timeIntervalSince(tFlush1),
                1000 * headBuild, 1000 * headEval,
                rate(headBytes, headEval),
                1000 * projectionBuild, 1000 * projectionEval,
                rate(projectionBytes, projectionEval),
                1000 * stepSeconds,
                100 * (headBuild + projectionBuild) / stepSeconds,
                rate(headBytes + projectionBytes, stepSeconds),
                1000 * chainSeconds, 1000 * chainSeconds / 8))
        }

        // ---- lm_head projection + argmax, the verify's sampling cost ----
        do {
            let (_, hidden) = model.callWithHidden(
                input: LMInput.Text(tokens: tokens(9, offset: filled)),
                cache: cache, nConfirmed: 0)
            filled += 9
            let normed = model.applyFinalNorm(hidden)
            eval(normed)
            print("")
            for rows in [1, 9] {
                let x = normed[0..., 0 ..< rows, 0...]
                eval(x)
                var best = Double.greatestFiniteMagnitude
                for _ in 0 ..< 3 {
                    let t0 = Date()
                    let ids = argMax(model.applyLMHead(x), axis: -1)
                    eval(ids)
                    best = min(best, Date().timeIntervalSince(t0))
                }
                print(String(
                    format: "  lm_head+argmax %d row(s): %.1f ms",
                    rows, 1000 * best))
            }
        }

        // ---- deepen to ~8k and repeat the key measurements ----
        while filled < 8192 {
            let w = min(1024, 8192 - filled)
            fusedChunk(cache: cache, width: w, offset: filled)
            filled += w
        }
        let fused8k = fusedChunk(cache: cache, width: 1024, offset: filled)
        filled += 1024
        if let p = profiledChunk(cache: cache, width: 1024, offset: filled) {
            filled += 1024
            report("prefill 1024 @ depth ~9k", p, fusedSeconds: fused8k)
        }

        widthSweep(
            "decode @ depth ~10k", cache: cache, widths: [1, 9, 32],
            filled: &filled)

        // ---- verify-shaped arm LAST: nConfirmed 1 engages the session's
        // checkpoint tape machinery, and running it outside a session is the
        // least-charted call in this instrument.  Anything it breaks can only
        // lose this one number. ----
        do {
            var best = Double.greatestFiniteMagnitude
            for _ in 0 ..< 3 {
                let t0 = Date()
                let (logits, _) = model.callWithHidden(
                    input: LMInput.Text(tokens: tokens(9, offset: filled)),
                    cache: cache, nConfirmed: 1)
                eval(logits)
                filled += 9
                best = min(best, Date().timeIntervalSince(t0))
            }
            print(String(
                format: "\n  M=9 verify-shaped (nConfirmed 1) @ ~10k: %.1f ms",
                1000 * best))
        }
        print("")
    }
}
