import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel
@testable import MLXLLM

/// One real layer, real weights, chained N times under a single eval: what does
/// a decode layer cost when nothing but the layer is being measured?
///
/// The whole-forward per-layer timing evals after every layer and reports
/// ~1.15 ms per layer, but that number carries a CPU round trip per layer. The
/// bandwidth benchmark says a linear-attention layer's ~187 MB of bf16 weights
/// cost 0.42 ms at the 441 GB/s the gemv achieves. This closes the gap between
/// those two by chaining the SAME layer through its own output N times in one
/// graph: the weights are re-read from memory every pass (187 MB x N is far
/// past any cache), the op chain is the real one, and the only sync is at the
/// end. What is left over the byte floor is the layer's own op-chain cost.
///
/// A second arm times a chain of N trivial dependent elementwise ops at the
/// hidden width, to price one launch on this box.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_QWEN4EXP_WEIGHTS=<weights> \
///       swift test -c release --force-resolved-versions --filter decodeLayerCost
@Suite(.serialized)
struct DecodeLayerCostTests {
    private final class Box<T>: @unchecked Sendable { var value: T? }

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    @Test("one real layer chained, one eval", .enabled(if: enabled), .timeLimit(.minutes(30)))
    func decodeLayerCost() throws {
        let env = ProcessInfo.processInfo.environment
        guard let weights = env["MLXFAST_QWEN4EXP_WEIGHTS"] else { return }
        let weightsURL = URL(fileURLWithPath: weights)
        Qwen4ExpRuntime.weightsDirectory = weightsURL

        let box = Box<ModelContext>()
        let failure = Box<String>()
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            do {
                box.value = try await LLMModelFactory.shared.load(
                    from: weightsURL, using: #huggingFaceTokenizerLoader())
            } catch {
                failure.value = String(describing: error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        guard let context = box.value, let model = context.model as? Qwen4ExpModel else {
            throw MLXFastError.invalidInput("failed to load model: \(failure.value ?? "?")")
        }

        // Seed so caches exist and residency is warm, as the decode arms do.
        let cache = model.newCache(parameters: nil)
        let seed = (0 ..< 64).map { Int32(1000 + $0) }
        let prefill = model(
            LMInput.Text(tokens: MLXArray(seed).reshaped([1, 64])), cache: cache, state: nil)
        eval(prefill.logits)

        let text = model.model
        let args = text.args
        let hcDim = args.hcDim
        let h0 = MLXRandom.normal([1, 1, hcDim]).asType(.bfloat16)
        let ids = MLXArray([Int32(7)]).reshaped([1, 1])
        eval(h0)

        let mask = createAttentionMask(h: h0, cache: nil, returnArray: false)
        let N = 64
        func chain(_ layerIndex: Int) -> Double {
            let layer = text.layers[layerIndex]
            let c = cache[layerIndex]
            // A residual-stream layer maps [1,1,hcDim] -> [1,1,hcDim], so its
            // output feeds the next pass directly. The cache offset advances N
            // per timed chain, which is a normal decode.
            func run() -> MLXArray {
                var h = h0
                for _ in 0 ..< N {
                    h = layer(h, rope: text.rope, mask: mask, ssmMask: nil, cache: c, ids: ids, prevContext: nil)
                }
                return h
            }
            eval(run())
            var best = Double.greatestFiniteMagnitude
            for _ in 0 ..< 5 {
                let t0 = DispatchTime.now().uptimeNanoseconds
                eval(run())
                best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 / Double(N))
            }
            return best
        }

        let linearIdx = text.layers.firstIndex { $0.isLinear && $0.ple == nil }!
        let fullIdx = text.layers.firstIndex { !$0.isLinear }!
        let dense = env["DARKBLOOM_DENSE_QUANT_BITS"] ?? "bf16"
        print("[layer-cost] dense=\(dense)  N=\(N) chained passes of one real layer, one eval")
        let lin = chain(linearIdx), full = chain(fullIdx)
        print(String(format: "  linear-attention layer %2d   %.3f ms per pass   x36 = %6.2f ms", linearIdx, lin, lin * 36))
        print(String(format: "  full-attention   layer %2d   %.3f ms per pass   x12 = %6.2f ms", fullIdx, full, full * 12))
        print(String(format: "  layers total               %6.2f ms of a ~64 ms step", lin * 36 + full * 12))

        // Price one launch: a chain of dependent trivial ops at the hidden width.
        let x = MLXRandom.normal([1, 1, args.hiddenSize]).asType(.bfloat16)
        eval(x)
        for ops in [64, 256, 1024] {
            func run() -> MLXArray {
                var y = x
                for i in 0 ..< ops { y = (i & 1 == 0) ? y * 1.0001 : y + 0.0001 }
                return y
            }
            eval(run())
            var best = Double.greatestFiniteMagnitude
            for _ in 0 ..< 5 {
                let t0 = DispatchTime.now().uptimeNanoseconds
                eval(run())
                best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e3)
            }
            print(String(format: "  %4d dependent elementwise ops  %7.1f us  = %5.2f us per op", ops, best, best / Double(ops)))
        }
    }
}
