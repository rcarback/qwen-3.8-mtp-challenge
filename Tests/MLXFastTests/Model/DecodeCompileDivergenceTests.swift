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

/// Does wrapping the decode glue in `compile` change what the model decides?
///
/// `Qwen4ExpNGramEncodingDivergenceTests` cannot answer this: it compares a
/// 512-token PREFILL, and the compiled path is restricted to the decode regime,
/// so that harness would exercise the eager code in both arms and report a
/// trivial zero. This one prefills a real prompt and then greedily decodes,
/// which is the only regime where the compiled trace runs.
///
/// The reason to ask at all is that `compile` fuses elementwise chains and can
/// keep intermediates in registers instead of round-tripping them through a
/// bf16 buffer. That is a precision change, not just a scheduling one.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_QWEN4EXP_WEIGHTS=<weights> \
///       MLXFAST_DECODE_AB_PROMPT=<prompt.txt> MLXFAST_DECODE_AB_OUT=<out.json> \
///       swift test -c release --force-resolved-versions --filter decodeCompileDivergence
@Suite(.serialized)
struct DecodeCompileDivergenceTests {
    private final class Box<T>: @unchecked Sendable { var value: T? }

    @Test("greedy decode stream, one arm", .timeLimit(.minutes(60)))
    func decodeCompileDivergence() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            let weights = env["MLXFAST_QWEN4EXP_WEIGHTS"],
            let promptPath = env["MLXFAST_DECODE_AB_PROMPT"],
            let outPath = env["MLXFAST_DECODE_AB_OUT"]
        else { return }

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

        let text = try String(contentsOfFile: promptPath, encoding: .utf8)
        let all = context.tokenizer.encode(text: text, addSpecialTokens: false)
        try #require(all.count >= 256, "prompt has \(all.count) tokens, need 256")
        let seed = Array(all[0 ..< 256])

        let cache = context.model.newCache(parameters: nil)
        var out = context.model(
            LMInput.Text(tokens: MLXArray(seed.map { Int32($0) }).reshaped([1, 256])),
            cache: cache, state: nil)
        eval(out.logits)

        // Greedy decode, recording the decision and how close it was. A gap is
        // what separates "the arms disagree because the fusion moved a near-tie"
        // from "the arms disagree because the model changed its mind".
        let steps = 256
        var rows: [[String: Any]] = []
        rows.reserveCapacity(steps)
        for p in 0 ..< steps {
            let last = out.logits[0..., -1, 0...].asType(.float32)
            let vocab = last.dim(1)
            let order = MLX.argSort(last, axis: -1)[0..., (vocab - 2)...]
            let vals = MLX.top(last, k: 2, axis: -1)
            eval(order, vals)
            let oi = order.asArray(Int32.self), ov = vals.asArray(Float.self)
            let best = Int(oi[1]), second = Int(oi[0])
            let bestV = Swift.max(ov[0], ov[1]), secondV = Swift.min(ov[0], ov[1])
            rows.append([
                "step": p, "token": best, "runner_up": second,
                "top1": bestV, "top2": secondV, "gap": bestV - secondV,
            ])
            out = context.model(
                LMInput.Text(tokens: MLXArray([Int32(best)]).reshaped([1, 1])),
                cache: cache, state: nil)
            eval(out.logits)
        }

        let payload: [String: Any] = [
            "compile_glue": env["MLX_QWEN4EXP_COMPILE_GLUE"] ?? "1",
            "compile_shared": env["MLX_QWEN4EXP_COMPILE_SHARED"] ?? "1",
            "prompt": promptPath,
            "steps": rows,
        ]
        try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: outPath))
        print("[decode-ab] wrote \(outPath) for glue="
            + "\(env["MLX_QWEN4EXP_COMPILE_GLUE"] ?? "1") "
            + "shared=\(env["MLX_QWEN4EXP_COMPILE_SHARED"] ?? "1")")
    }
}
