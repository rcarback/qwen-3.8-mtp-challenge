import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel

/// What a LOOKUP-ladder round actually costs, at the verify shape a round
/// really dispatches.
///
/// `QwenPhaseBreakdownTests` swept widths at `nConfirmed: 0`. A verify runs at
/// `nConfirmed: 1`, which selects `processChunkStashingPrefix` in every one of
/// the 48 gated-delta layers and retains a per-layer replay tape whose size is
/// linear in the row count. This measures three things that sweep did not:
///
///   1. the verify at `nConfirmed: 1` for the four ladder widths, split into
///      host graph build and GPU eval so a debug host cannot masquerade as GPU
///      time;
///   2. the memory the retained tapes hold at each width;
///   3. the wall time of a partial-acceptance repair -- the prefix replay plus
///      the attention trim -- which is what a rejected wide round pays.
///
/// The fill depth defaults to ~2k, matching the original sweep, but is
/// overridable through `MLXFAST_QWEN_LOOKUP_COST_FILL_DEPTH`: this fork exists
/// for large (20k+ token) serve contexts, and a 32-row verify's attention term
/// scales with the live KV length, so the ladder's top rung should also be
/// checked at a serve-realistic depth before it is trusted there. Setting the
/// variable runs the identical instrument further into the context -- no
/// shipped ladder value moves here, and the two Task 7 Step 3 assertions
/// that compare width 32 against width 16 (`total_ms`, `repair_ms`) still
/// apply verbatim at any depth; only the absolute `tape_MiB` ceiling is a
/// property of the model's per-row state, not of the fill depth.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
///     MLXFAST_QWEN_PREFILL_WEIGHTS=weights \
///     MLXFAST_QWEN_PREFILL_HEAD=<head dir> \
///     MLXFAST_QWEN_LOOKUP_COST_FILL_DEPTH=20480 \
///     swift test --force-resolved-versions --filter lookupWidthCost
@Suite(.serialized)
struct QwenLookupWidthCostTests {
    @Test("lookupWidthCost")
    func lookupWidthCost() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
              let weights = env["MLXFAST_QWEN_PREFILL_WEIGHTS"],
              let head = env["MLXFAST_QWEN_PREFILL_HEAD"]
        else { return }
        let fillDepth = env["MLXFAST_QWEN_LOOKUP_COST_FILL_DEPTH"]
            .flatMap(Int.init) ?? 2048

        let targetURL = URL(fileURLWithPath: weights)
        let context = try Qwen36MTPHeadAttachment.withHeadAttached(
            backboneDirectory: targetURL,
            headDirectory: URL(fileURLWithPath: head)
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
                Int32((($0 + offset) * 7919) % 90_000 + 10)
            }).reshaped([1, count])
        }

        let cache = model.newCache(parameters: nil)
        var filled = 0
        while filled < fillDepth {
            let (_, hidden) = model.callWithHidden(
                input: LMInput.Text(tokens: tokens(512, offset: filled)),
                cache: cache, nConfirmed: 0)
            eval(hidden)
            filled += 512
        }

        func tapeBytes(_ caches: [any KVCache]) -> Int {
            caches.reduce(0) { total, entry in
                guard let arrays = entry as? ArraysCache,
                      let tape = arrays.prefixReplayTape
                else { return total }
                let parts = [
                    tape.convInput, tape.q, tape.k, tape.v,
                    tape.a, tape.b, tape.g, tape.beta,
                ]
                return total + parts.reduce(0) { $0 + $1.nbytes }
                    + (tape.ssmPre?.nbytes ?? 0)
            }
        }

        struct Row {
            let width: Int
            let totalMS: Double
            let repairMS: Double
            let tapeMiB: Double
        }
        var rows: [Row] = []

        print("\n  lookup ladder verify cost at nConfirmed 1, fill depth \(fillDepth)")
        print("  width  build_ms   eval_ms  total_ms   ms/row  tape_MiB"
            + "  repair_ms  repair_path")
        for width in [4, 9, 16, 32] {
            var best = (
                build: 0.0, eval: 0.0,
                total: Double.greatestFiniteMagnitude)
            var tape = 0
            var repair = Double.greatestFiniteMagnitude
            // The two repair paths cost materially different amounts, so the
            // branch actually taken has to be legible in the table rather than
            // folded into one `repair_ms` number: `replayRecurrentPrefix` is
            // the promoted path production uses (`restoreAfterPrefixReject`);
            // the generic `rollbackAfterVerify` fallback only fires when the
            // gated-delta dimension checks in `canReplayPrefix` fail, and is
            // NOT what a live rejected round pays under the promoted path.
            var tookReplayPath = true
            for _ in 0 ..< 3 {
                let base = Qwen36MTPBlockSession.trimmableOffset(cache)
                let snapshot = Qwen36MTPBlockSession.snapshotRecurrent(cache)
                let t0 = Date()
                let (logits, _, normed) = model.callWithHiddenAndNormed(
                    input: LMInput.Text(tokens: tokens(width, offset: filled)),
                    cache: cache, nConfirmed: 1)
                let t1 = Date()
                var bundle: [MLXArray] = [logits]
                if let normed { bundle.append(normed) }
                eval(bundle)
                eval(cache.flatMap { $0.state })
                let t2 = Date()
                tape = max(tape, tapeBytes(cache))

                // A partial acceptance of half the block: replay the committed
                // prefix and trim the rejected attention rows. This is the
                // whole cost a rejected wide round pays under the promoted
                // path; the generic snapshot restore is the fallback below.
                let committedRows = max(1, width / 2)
                let t3 = Date()
                let replayed = model.replayRecurrentPrefix(
                    cache: cache, committedRows: committedRows)
                tookReplayPath = replayed
                if replayed {
                    for entry in cache where !(entry is ArraysCache) {
                        if entry.isTrimmable, entry.offset > base + committedRows {
                            _ = entry.trim(entry.offset - base - committedRows)
                        }
                    }
                } else {
                    Qwen36MTPBlockSession.rollbackAfterVerify(
                        cache, snapshot, verifiedTokens: width, to: base)
                }
                eval(cache.flatMap { $0.state })
                let t4 = Date()
                repair = min(repair, t4.timeIntervalSince(t3))

                // Put the cache back where the next repetition expects it.
                Qwen36MTPBlockSession.rollbackAfterVerify(
                    cache, snapshot,
                    verifiedTokens: width,
                    to: base)
                eval(cache.flatMap { $0.state })

                let total = t2.timeIntervalSince(t0)
                if total < best.total {
                    best = (
                        t1.timeIntervalSince(t0),
                        t2.timeIntervalSince(t1), total)
                }
            }
            // A missing tape must fail the instrument, not print 0.0 and read
            // as "no memory pressure." `processChunkStashingPrefix` stashes a
            // tape at `nConfirmed == 1 && S >= 3 && mask == nil` for every one
            // of these widths, so a zero here means the retained-tape path was
            // not exercised -- the Task 7 Step 3 `tape_MiB` reading would then
            // describe a path production never takes.
            #expect(
                tape > 0,
                Comment(rawValue: "width \(width) retained no prefix-replay "
                    + "tape; the verify did not take the tape-stashing path "
                    + "this instrument means to measure"))
            let tapeMiB = Double(tape) / (1024 * 1024)
            let repairMS = 1000 * repair
            let totalMS = 1000 * best.total
            print(String(
                format: "  %5d  %8.1f  %8.1f  %8.1f  %7.1f  %8.1f  %9.1f  %@",
                width, 1000 * best.build, 1000 * best.eval, totalMS,
                totalMS / Double(width), tapeMiB, repairMS,
                tookReplayPath ? "replay" : "rollback(generic)"))
            rows.append(Row(
                width: width, totalMS: totalMS, repairMS: repairMS,
                tapeMiB: tapeMiB))
        }
        print("")

        // Task 7 Step 3's three conditions, asserted rather than left to a
        // human reading the printed table. If none fires, the shipped ladder
        // stands; if one does, the plan names the exact remedy (see the
        // doc comment on `NGramPromptLookupConfiguration.shipped` and this
        // task's report).
        guard let width16 = rows.first(where: { $0.width == 16 }),
              let width32 = rows.first(where: { $0.width == 32 })
        else {
            Issue.record("the width sweep did not produce rows for 16 and 32")
            return
        }
        #expect(
            width32.totalMS <= 1.5 * width16.totalMS,
            Comment(rawValue: "width 32 total_ms \(width32.totalMS) exceeds "
                + "1.5x width 16's \(width16.totalMS); Task 7 Step 3 says "
                + "drop the top rung"))
        #expect(
            width32.repairMS <= 0.25 * width32.totalMS,
            Comment(rawValue: "width 32 repair_ms \(width32.repairMS) exceeds "
                + "25% of its own total_ms \(width32.totalMS); Task 7 Step 3 "
                + "says raise the top two thresholds"))
        #expect(
            width32.tapeMiB <= 2048,
            Comment(rawValue: "width 32 tape_MiB \(width32.tapeMiB) exceeds "
                + "2048; Task 7 Step 3 says drop the top rung"))
    }
}
