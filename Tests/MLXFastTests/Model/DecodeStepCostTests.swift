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

/// How long is one decode step, so per-token overheads have a denominator?
///
/// The router measures 0.729 ms per layer at a single row, which is 35.0 ms per
/// token across 48 layers. Whether that matters depends entirely on what a
/// decode step costs in total, and nothing in this repository had measured it.
/// Every per-token optimisation estimate needs this number.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_QWEN4EXP_WEIGHTS=<weights> \
///       swift test -c release --force-resolved-versions --filter decodeStepCost
@Suite(.serialized)
struct DecodeStepCostTests {
    private final class Box<T>: @unchecked Sendable { var value: T? }

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    @Test("one decode step, real model", .enabled(if: enabled))
    func decodeStepCost() throws {
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

        // Seed with a short prefill so the cache is populated, then time decode
        // steps one token at a time, which is the regime per-token overheads
        // are paid in.
        let seed = (0 ..< 64).map { Int32(1000 + $0) }
        let cache = context.model.newCache(parameters: nil)
        let prefill = context.model(
            LMInput.Text(tokens: MLXArray(seed).reshaped([1, seed.count])), cache: cache,
            state: nil)
        eval(prefill.logits)

        // Sweep rows per forward on the REAL model. The microbenchmarks that
        // suggested component cost is flat in row count turned out to be
        // measuring an isolated-eval floor of about 0.22 ms, so the
        // speculative-decode conclusion drawn from them needs the real model
        // to stand. If a forward carrying K rows costs about what one row
        // costs, a K-row speculative block costs roughly one forward and every
        // accepted token past the first is nearly free.
        var next = Int32(42)
        var perRows = [(Int, Double)]()
        for rows in [1, 2, 4, 8] {
            let tok = MLXArray((0 ..< rows).map { Int32(next) + Int32($0) })
                .reshaped([1, rows])
            // Warm this shape before timing it.
            for _ in 0 ..< 3 {
                let o = context.model(LMInput.Text(tokens: tok), cache: cache, state: nil)
                eval(o.logits)
            }
            var best = Double.greatestFiniteMagnitude
            for _ in 0 ..< 3 {
                let t0 = DispatchTime.now().uptimeNanoseconds
                for _ in 0 ..< 8 {
                    let o = context.model(LMInput.Text(tokens: tok), cache: cache, state: nil)
                    eval(o.logits)
                }
                best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 / 8)
            }
            perRows.append((rows, best))
            next = (next + Int32(rows)) % 1000
        }
        let one = perRows[0].1
        print("[decode-rows] forward cost against rows carried, real model")
        for (rows, ms) in perRows {
            print(
                "  rows=\(rows)  \(String(format: "%7.2f", ms))ms  "
                    + "\(String(format: "%5.2f", ms / one))x of one row  "
                    + "per token \(String(format: "%6.2f", ms / Double(rows)))ms")
        }

        // Per-layer split from INSIDE the forward. This forces an eval per
        // layer, serialising what the graph pipelines, so the total overstates
        // the unmeasured step. The gap between them is how much the graph
        // amortises, and it is the number a microbenchmark cannot reach.
        if Qwen4ExpLayerTiming.enabled {
            Qwen4ExpLayerTiming.reset()
            let tok = MLXArray([Int32(7)]).reshaped([1, 1])
            let o = context.model(LMInput.Text(tokens: tok), cache: cache, state: nil)
            eval(o.logits)
            let m = Qwen4ExpLayerTiming.snapshotMs()
            let c = Qwen4ExpLayerTiming.counts()
            let sum = m.full + m.linear + m.tail
            print(
                "[decode-layers] serialised, one row\n"
                    + "  full attention   \(c.full) layers  "
                    + "\(String(format: "%7.2f", m.full))ms  "
                    + "\(String(format: "%.3f", m.full / Double(max(c.full, 1))))ms each\n"
                    + "  linear attention \(c.linear) layers  "
                    + "\(String(format: "%7.2f", m.linear))ms  "
                    + "\(String(format: "%.3f", m.linear / Double(max(c.linear, 1))))ms each\n"
                    + "  tail (mixer)                 "
                    + "\(String(format: "%7.2f", m.tail))ms\n"
                    + "  serialised total             \(String(format: "%7.2f", sum))ms  "
                    + "against \(String(format: "%.2f", one))ms pipelined  "
                    + "= \(String(format: "%.2f", sum / one))x")
        }
    }
}
