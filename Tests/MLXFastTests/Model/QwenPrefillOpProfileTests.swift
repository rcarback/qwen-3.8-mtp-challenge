import Foundation
import MLX
import MLXLLM
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel

/// WHERE DOES ONE PREFILL BLOCK'S TIME GO, OP BY OP?
///
/// `QwenPhaseBreakdownTests` attributes a prefill chunk down to the layer:
/// 48 gated-delta linear blocks at ~131 ms and 16 full-attention blocks at
/// ~126 ms (depth 0) to ~170 ms (depth ~9k). This instrument goes one level
/// below that and splits a single block into its individual operations.
///
/// TWO MODES, BECAUSE ONE MODE CANNOT BE TRUSTED ALONE. MLX is lazy: a wall
/// clock around an expression measures graph construction on the host, not
/// GPU execution. The two honest ways out disagree in opposite directions and
/// the instrument prints both:
///
///  * `serialized` evaluates after every op. Attribution is direct, but the
///    command stream is serialised and the kernel fusion the production graph
///    gets is forbidden, so the TOTAL inflates. The per-layer seam measures
///    this tax at +1.7% at prefill width 1024 and -15.1% at verify width 16.
///  * `ladder` evaluates ONCE per pass, after op k, and charges op k the
///    increment `T(k) - T(k-1)`. Every op up to k keeps its production
///    fusion, and the increments telescope exactly to the one-eval whole-block
///    time, so the total is the fused truth by construction and only the SPLIT
///    between ops carries error. Its bias: an op that enables a fusion is
///    charged for it, and an op whose work overlaps its predecessor's is
///    charged near zero.
///
/// Where the two columns agree, the attribution is solid. Where they diverge,
/// the divergence IS the uncertainty and must be reported as such.
///
/// The instrument also verifies its own honesty: the op chain is a replica of
/// the production block expression, and every profile checks the replica's
/// output for bit-equality against the unmodified `callAsFunction`. A
/// `replica NOT exact` line invalidates that table.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
///     MLXFAST_QWEN_PREFILL_WEIGHTS=weights \
///     MLXFAST_QWEN_PREFILL_HEAD=<head dir> \
///     swift test --force-resolved-versions --filter prefillOpProfile
///
/// Optional knobs: `MLXFAST_QWEN_OP_ROWS` (chunk width, default 1024),
/// `MLXFAST_QWEN_OP_LAYERS` (comma-separated block indices, default `0,3` —
/// the first gated-delta block and the first full-attention block),
/// `MLXFAST_QWEN_OP_DEPTHS` (comma-separated cached depths for the
/// full-attention blocks, default `0,9216`).
@Suite(.serialized)
struct QwenPrefillOpProfileTests {
    @Test("prefill op profile")
    func prefillOpProfile() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            let weights = env["MLXFAST_QWEN_PREFILL_WEIGHTS"],
            let head = env["MLXFAST_QWEN_PREFILL_HEAD"]
        else { return }

        func intList(_ key: String, _ fallback: [Int]) -> [Int] {
            guard let raw = env[key], !raw.isEmpty else { return fallback }
            let parsed = raw.split(separator: ",").compactMap { Int($0) }
            return parsed.isEmpty ? fallback : parsed
        }
        let rows = intList("MLXFAST_QWEN_OP_ROWS", [1024])[0]
        let layerIndices = intList("MLXFAST_QWEN_OP_LAYERS", [0, 3])
        let depths = intList("MLXFAST_QWEN_OP_DEPTHS", [0, 9216])

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
        guard let mtp = context.model as? any Qwen36MTPTarget else {
            Issue.record("backbone is not an MTP target")
            return
        }

        func tokens(_ count: Int) -> MLXArray {
            MLXArray((0 ..< count).map { Int32(($0 * 7919) % 100_000 + 10) })
                .reshaped([1, count])
        }

        func report(_ profile: Qwen35OpProfile) {
            print("\n[op profile] \(profile.label)")
            if !profile.replicaExact {
                print("  *** replica NOT exact against callAsFunction — "
                    + "this table is INVALID ***")
            }
            print(String(
                format: "  fused truth (ladder total) %8.2f ms   "
                    + "production callAsFunction %8.2f ms",
                1000 * profile.ladderTotalSeconds,
                1000 * profile.productionSeconds))
            print(String(
                format: "  serialized total %8.2f ms   sync tax %+.1f%%",
                1000 * profile.serializedTotalSeconds, 100 * profile.syncTax))
            print("  op                        kind"
                + "              ladder_ms  share   serial_ms  share"
                + "    TFLOPS    GB/s   FLOP/B")
            let ladderTotal = max(profile.ladderTotalSeconds, 1e-12)
            let serialTotal = max(profile.serializedTotalSeconds, 1e-12)
            for (ladder, serial) in zip(profile.ladder, profile.serialized) {
                print(String(
                    format: "  %-24@  %-16@  %9.3f %5.1f%%  %9.3f %5.1f%%"
                        + "  %8.2f %7.1f %8.1f",
                    ladder.name as NSString, ladder.kind as NSString,
                    1000 * ladder.seconds,
                    100 * ladder.seconds / ladderTotal,
                    1000 * serial.seconds,
                    100 * serial.seconds / serialTotal,
                    serial.tflops, serial.gigabytesPerSecond,
                    serial.intensity))
            }
        }

        let input = tokens(rows)
        for index in layerIndices {
            let isLinear: Bool
            if let top = context.model as? MLXLLM.Qwen35Model {
                isLinear = top.layerIsLinear(index)
            } else if let text = context.model as? Qwen35TextModel {
                isLinear = text.layerIsLinear(index)
            } else {
                Issue.record("model exposes no Qwen35 op-profile seam")
                return
            }
            // A gated-delta block's cost is depth-independent (measured 131.16
            // ms at depth 0 against 130.83 ms at ~9k), so profile it once; the
            // full-attention block carries the whole depth term, so sweep it.
            for depth in isLinear ? [0] : depths {
                let prefixCache = mtp.newCache(parameters: nil)
                let hidden: MLXArray
                if let top = context.model as? MLXLLM.Qwen35Model {
                    hidden = top.hiddenBefore(
                        layer: index, inputs: input, cache: prefixCache)
                } else if let text = context.model as? Qwen35TextModel {
                    hidden = text.hiddenBefore(
                        layer: index, inputs: input, cache: prefixCache)
                } else {
                    return
                }
                let profile: Qwen35OpProfile
                if let top = context.model as? MLXLLM.Qwen35Model {
                    profile = top.profiledPrefillOpProfile(
                        hidden: hidden, layerIndex: index, cacheDepth: depth,
                        makeCache: { KVCacheSimple() })
                } else if let text = context.model as? Qwen35TextModel {
                    profile = text.profiledPrefillOpProfile(
                        hidden: hidden, layerIndex: index, cacheDepth: depth,
                        makeCache: { KVCacheSimple() })
                } else {
                    return
                }
                report(profile)
                #expect(
                    profile.replicaExact,
                    "profiled op chain diverged from the production block")
            }
        }
        print("")
    }
}
