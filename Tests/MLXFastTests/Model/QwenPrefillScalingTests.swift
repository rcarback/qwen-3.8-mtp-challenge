import Foundation
import MLX
import MLXLLM
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel

/// Does prefill amortize with chunk size?
///
/// WHY THIS EXISTS. Serve measures a flat ~12-16 ms per prompt token from 106
/// tokens to 20,022, which is not what a batched prefill does -- the 15 GB
/// weight read should amortize across the chunk and drive per-token cost down
/// as T grows. If per-token cost is genuinely flat, the prompt is bound by the
/// 48 gated-delta layers' sequential T-step recurrence and no chunk size helps.
/// If it falls with T, `prefillChunkProductBudget` (64 Mi, which caps the chunk
/// at 3,355 tokens once 20k sits in cache) is leaving throughput on the table.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
///     MLXFAST_QWEN_PREFILL_WEIGHTS=weights \
///     MLXFAST_QWEN_PREFILL_HEAD=<head dir> \
///     swift test --force-resolved-versions --filter QwenPrefillScalingTests
/// One-shot box so the async loader's result can cross back to this thread.
/// Single write, single read, ordered by the semaphore.
final class UnsafeSendableBox<T>: @unchecked Sendable {
    var value: T?
}

@Suite(.serialized)
struct QwenPrefillScalingTests {
    @Test("prefill cost per token versus chunk size")
    func prefillScaling() throws {
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

        // Cumulative cost through layer i, using the layer-capture seam:
        // evaluating ONLY the capture forces layers 0...i and leaves the rest
        // of the graph dead, so successive differences price one layer.
        // layer_types repeats [linear, linear, linear, full], so layers 3 and 7
        // are full_attention and 0,1,2,4,5,6 are gated-delta.
        let T = 1024
        print("\nPer-layer prefill cost at T=\(T), fresh cache each point")
        print("  through_layer   seconds   delta_ms/token   type")
        var previous = 0.0
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 15, 31, 63] {
            let cache = model.newCache(parameters: nil)
            let tokens = MLXArray(
                (0 ..< T).map { Int32(($0 * 7919) % 100_000 + 10) }
            ).reshaped([1, T])
            let start = Date()
            let out = model.callWithHiddenNormedAndLayers(
                input: LMInput.Text(tokens: tokens), cache: cache,
                nConfirmed: 0, layerIDs: [i])
            guard let captured = out.layerHidden else {
                Issue.record("no capture at layer \(i)"); return
            }
            eval(captured)
            let dt = Date().timeIntervalSince(start)
            let kind = (i % 4 == 3) ? "full_attention" : "gated_delta"
            let delta = i == 0 ? dt : dt - previous
            print(String(
                format: "  %13d  %8.3f  %14.4f   %@",
                i, dt, 1000 * delta / Double(T), kind))
            previous = dt
        }
        print("")
    }
}
