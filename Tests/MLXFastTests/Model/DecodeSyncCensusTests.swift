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

/// Where does a decode step wait on the host, and how much of it is graph
/// construction rather than GPU time?
///
/// MLX is lazy: the model forward builds a graph and returns, and the step's
/// single `eval` submits and waits. The forward contains two host readbacks
/// (the PLE n-gram embedding reads the token ids back, before layer 0 and at
/// layer 2), but both depend only on token ids, so they force tiny graphs and
/// cannot stall on earlier layers. What they DO is break a compile trace.
///
/// This instrument splits each step into `build` (forward returned) and `eval`
/// (GPU submitted and drained), and reports the n-gram gather's own time from
/// its stats box. If `build` is a large share, the launch term is CPU graph
/// construction and compile removes it; if `eval` dominates, it is GPU-side
/// launch gaps and only fusion or fewer kernels remove it.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_QWEN4EXP_WEIGHTS=... \
///       swift test -c release --force-resolved-versions --filter decodeSyncCensus
@Suite(.serialized)
struct DecodeSyncCensusTests {
    private final class Box<T>: @unchecked Sendable { var value: T? }

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    @Test("decode sync census", .enabled(if: enabled), .timeLimit(.minutes(60)))
    func decodeSyncCensus() throws {
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
        guard let context = box.value else {
            throw MLXFastError.invalidInput("failed to load model: \(failure.value ?? "?")")
        }

        let cache = context.model.newCache(parameters: nil)
        let seed = (0 ..< 64).map { Int32(1000 + $0) }
        let prefill = context.model(
            LMInput.Text(tokens: MLXArray(seed).reshaped([1, seed.count])), cache: cache,
            state: nil)
        eval(prefill.logits)

        var tokenId = Int32(42)
        func step() -> (build: Double, eval: Double) {
            let tok = MLXArray([tokenId]).reshaped([1, 1])
            tokenId = (tokenId + 17) % 900 + 100
            let t0 = DispatchTime.now().uptimeNanoseconds
            let o = context.model(LMInput.Text(tokens: tok), cache: cache, state: nil)
            let t1 = DispatchTime.now().uptimeNanoseconds
            eval(o.logits)
            let t2 = DispatchTime.now().uptimeNanoseconds
            return (Double(t1 - t0) / 1e6, Double(t2 - t1) / 1e6)
        }

        let warm = 220, timed = 200
        for _ in 0 ..< warm { _ = step() }
        Qwen4ExpNGramTable.stats.reset()
        var builds = [Double](), evals = [Double]()
        for _ in 0 ..< timed {
            let s = step()
            builds.append(s.build)
            evals.append(s.eval)
        }
        let ng = Qwen4ExpNGramTable.stats.snapshot()
        func median(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
        let b = median(builds), e = median(evals)
        let ngMs = Double(ng.nanos) / 1e6 / Double(timed)
        print("[sync-census] depth-0 step, \(timed) warm steps, medians")
        print(String(format: "  build (forward returned, graph only): %.3f ms", b))
        print(String(format: "  eval  (submit + drain):               %.3f ms", e))
        print(String(format: "  step:                                 %.3f ms   build share %.1f%%", b + e, b / (b + e) * 100))
        print(String(
            format: "  n-gram gather inside build: %d calls, %d rows, %.3f ms per step",
            ng.calls, ng.rows, ngMs))
        print("  host readbacks per step: 2 (n-gram ids before layer 0 and at layer 2), both token-id only; blocking evals: 1")
    }
}
