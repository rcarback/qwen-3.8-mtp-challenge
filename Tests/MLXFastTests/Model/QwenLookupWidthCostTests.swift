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
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
///     MLXFAST_QWEN_PREFILL_WEIGHTS=weights \
///     MLXFAST_QWEN_PREFILL_HEAD=<head dir> \
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
        while filled < 2048 {
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

        print("\n  lookup ladder verify cost at nConfirmed 1, depth ~2k")
        print("  width  build_ms   eval_ms  total_ms   ms/row  tape_MiB"
            + "  repair_ms")
        for width in [4, 9, 16, 32] {
            var best = (
                build: 0.0, eval: 0.0,
                total: Double.greatestFiniteMagnitude)
            var tape = 0
            var repair = Double.greatestFiniteMagnitude
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
                if model.replayRecurrentPrefix(
                    cache: cache, committedRows: committedRows)
                {
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
            print(String(
                format: "  %5d  %8.1f  %8.1f  %8.1f  %7.1f  %8.1f  %9.1f",
                width, 1000 * best.build, 1000 * best.eval, 1000 * best.total,
                1000 * best.total / Double(width),
                Double(tape) / (1024 * 1024), 1000 * repair))
        }
        print("")
    }
}
