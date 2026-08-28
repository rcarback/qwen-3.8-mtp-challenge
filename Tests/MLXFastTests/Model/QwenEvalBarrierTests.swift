import Foundation
import MLX
import MLXLLM
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel

/// The eval barrier in a decode round exists to keep the cache's lazy graph
/// from growing. It does not need the trimmed VIEW of the cache, only the
/// roots -- and asking for the view builds two slice operations per
/// full-attention layer, every round, that are evaluated and discarded.
@Suite
struct QwenEvalBarrierTests {
    @Test("innerState returns the roots while state returns trimmed slices")
    func innerStateSkipsTheSlice() {
        let cache = KVCacheSimple()
        // One appended row against a step of 256 leaves offset far below the
        // allocated depth, which is the branch every decode round takes.
        let keys = MLXArray.zeros([1, 8, 1, 128], dtype: .float32)
        let values = MLXArray.zeros([1, 8, 1, 128], dtype: .float32)
        _ = cache.update(keys: keys, values: values)
        #expect(cache.offset == 1)
        #expect(cache.state[0].dim(2) == 1, "state is trimmed to the offset")
        #expect(
            cache.innerState()[0].dim(2) == 256,
            "innerState is the untrimmed root")
    }

    @Test("every cache class the session builds overrides innerState")
    func everyCacheClassCarriesRoots() {
        // BaseKVCache.innerState() returns an empty array, so a cache class
        // that forgot the override would make the barrier evaluate nothing at
        // all -- silently, and only under load. Pin both classes the Qwen
        // tower builds (Qwen35.swift newCache: MambaCache for linear layers,
        // KVCacheSimple for the rest).
        let attention = KVCacheSimple()
        _ = attention.update(
            keys: MLXArray.zeros([1, 8, 1, 128], dtype: .float32),
            values: MLXArray.zeros([1, 8, 1, 128], dtype: .float32))
        #expect(attention.innerState().count == 2)

        let recurrent = MambaCache()
        recurrent[0] = MLXArray.zeros([1, 4], dtype: .float32)
        recurrent[1] = MLXArray.zeros([1, 4], dtype: .float32)
        #expect(recurrent.innerState().count == 2)
        #expect(recurrent.innerState().count == recurrent.state.count)
    }

    /// The barrier change must not move a single token. This run prints the
    /// first tokens of each depth for eyeball comparison across builds; the
    /// position-by-position machine check is the serve A/B against the frozen
    /// trajectories, not this test. Opt-in, because it loads the real
    /// backbone and head.
    ///
    /// It also prints the per-round wall time, which is the only measurement
    /// of this change that exists before ship point S1.
    @Test("rounds stay identical and report their cost")
    func roundsAreUnchanged() throws {
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
            Issue.record("backbone is not an MTP target"); return
        }

        let seed = (0 ..< 512).map { ($0 * 7919) % 90_000 + 10 }
        for depth in [0, 2] {
            let session = try Qwen36MTPBlockSession(
                model: model, stopTokens: [])
            try session.warmAllDepths(maxDepth: Qwen36MTPLimits.maxDepth)
            _ = try session.begin(seedTokens: seed)
            var tokens: [Int] = []
            var best = Double.greatestFiniteMagnitude
            var total = 0.0
            for _ in 0 ..< 32 {
                let started = Date()
                let round = try session.generateRound(depth: depth)
                let seconds = Date().timeIntervalSince(started)
                best = Swift.min(best, seconds)
                total += seconds
                tokens.append(contentsOf: round.tokens)
            }
            print(String(
                format: "\n[eval barrier] depth %d: %d tokens in 32 rounds, "
                    + "best round %.1f ms, mean %.1f ms",
                depth, tokens.count, 1000 * best, 1000 * total / 32))
            print("  first 16 tokens: \(Array(tokens.prefix(16)))")
        }
    }
}
