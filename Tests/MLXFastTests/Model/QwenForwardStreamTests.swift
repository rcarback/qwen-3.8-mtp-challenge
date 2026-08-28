import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel
@testable import MLXLLM  // `qwen35EpsScalar` is internal; see GatedDeltaScanCostTests

/// Width-1 forward stream: what the host builds per decode step, and the
/// receipts for each launch reduction that removes part of it.
///
/// The first test needs no model. The rest are opt-in behind
/// `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` following the house pattern.
@Suite(.serialized)
struct QwenForwardStreamTests {

    /// Characterization, not a goal: compiled decode cannot take this model's
    /// cache shape, and this records why so the question is settled in code.
    ///
    /// `Qwen35TextModel.newCache` hands back a `MambaCache` for each of the 48
    /// linear-attention layers and a `KVCacheSimple` for each of the 16
    /// full-attention layers (Qwen35.swift:5348-5355).
    /// `CompiledDecode.setupCompiledDecode` refuses any layer that is neither
    /// `KVCacheSimple` nor `RotatingKVCache` (CompiledDecode.swift:132-138),
    /// and `eligible` accepts only the promoted `Compilable*` types
    /// (CompiledDecode.swift:50-55). There is no compilable `ArraysCache`.
    @Test("compiled decode rejects the Qwen cache shape")
    func compiledDecodeRejectsTheQwenCacheShape() {
        let qwenShaped: [KVCache] = [MambaCache(), KVCacheSimple()]
        #expect(!CompiledDecode.eligible(qwenShaped))

        // The specific layer that blocks it, isolated from its neighbour.
        let recurrent = MambaCache()
        #expect(!(recurrent is KVCacheSimple))
        #expect(!(recurrent is RotatingKVCache))
        #expect(!CompiledDecode.eligible([recurrent]))

        // The full-attention layer alone is also rejected, because `eligible`
        // wants the PROMOTED type, not the promotable one.
        #expect(!CompiledDecode.eligible([KVCacheSimple()]))
    }

    /// Release-mode width-1 bare-forward baseline at decode depth, plus the
    /// per-layer split. MEASUREMENT INSTRUMENT, not a gate: wall-clock has no
    /// thermal gate and no pairing here, so the numbers are directional and
    /// only comparisons taken in the same session are meaningful.
    ///
    ///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
    ///     MLXFAST_QWEN_PREFILL_WEIGHTS="$PWD/weights" \
    ///     MLXFAST_QWEN_PREFILL_HEAD="$HOME/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head" \
    ///     swift test -c release --force-resolved-versions \
    ///       --filter forwardStreamBaseline
    @Test("width-1 forward baseline")
    func forwardStreamBaseline() throws {
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

        // Fill to decode depth with 1024-token chunks, the same shape the
        // phase-breakdown instrument uses.
        let cache = model.newCache(parameters: nil)
        var filled = 0
        while filled < 2048 {
            let (_, hidden) = model.callWithHidden(
                input: LMInput.Text(tokens: tokens(1024, offset: filled)),
                cache: cache, nConfirmed: 0)
            eval(hidden)
            filled += 1024
        }

        let ladder = ProcessInfo.processInfo
            .environment["MLX_QWEN_MTP_LADDER"] ?? "default"
        print("\n[width-1 forward @ depth ~2k] ladder=\(ladder)")
        print("     rep   build_ms    eval_ms   total_ms")
        var best = Double.greatestFiniteMagnitude
        var bestBuild = 0.0
        var bestEval = 0.0
        for rep in 0 ..< 8 {
            let t0 = Date()
            let (logits, _) = model.callWithHidden(
                input: LMInput.Text(tokens: tokens(1, offset: filled)),
                cache: cache, nConfirmed: 0)
            let t1 = Date()
            eval(logits)
            let t2 = Date()
            filled += 1
            let build = t1.timeIntervalSince(t0)
            let evalTime = t2.timeIntervalSince(t1)
            let total = t2.timeIntervalSince(t0)
            print(String(
                format: "  %6d  %9.2f  %9.2f  %9.2f",
                rep, 1000 * build, 1000 * evalTime, 1000 * total))
            if total < best {
                best = total
                bestBuild = build
                bestEval = evalTime
            }
        }
        print(String(
            format: "  BEST    %9.2f  %9.2f  %9.2f",
            1000 * bestBuild, 1000 * bestEval, 1000 * best))

        // Per-layer split at width 1. The seam synchronizes after every layer
        // and runs the UNFUSED chain (Qwen35.swift:6293-6298), so at width 1
        // the sync tax dominates the absolute numbers: read the linear-versus-
        // full SHARE and the ranking, never the total.
        let input = tokens(1, offset: filled)
        var profile: Qwen35LayerProfile?
        if let top = context.model as? MLXLLM.Qwen35Model {
            profile = top.profiledLayerForward(input, cache: cache)
        } else if let text = context.model as? Qwen35TextModel {
            profile = text.profiledLayerForward(input, cache: cache)
        }
        if let profile {
            let linear = zip(profile.layerSeconds, profile.layerIsLinear)
                .filter { $0.1 }.map { $0.0 }.reduce(0, +)
            let full = zip(profile.layerSeconds, profile.layerIsLinear)
                .filter { !$0.1 }.map { $0.0 }.reduce(0, +)
            let synced = linear + full + profile.embedSeconds
            print(String(
                format: "\n  48 linear blocks  %7.1f ms  (%4.1f%%)"
                    + "  mean %5.2f ms",
                1000 * linear, 100 * linear / synced, 1000 * linear / 48))
            print(String(
                format: "  16 full blocks    %7.1f ms  (%4.1f%%)"
                    + "  mean %5.2f ms",
                1000 * full, 100 * full / synced, 1000 * full / 16))
            print(String(
                format: "  synced total      %7.1f ms  (sync tax is large at"
                    + " width 1; shares only)",
                1000 * synced))
        }
        print("")
    }
}
