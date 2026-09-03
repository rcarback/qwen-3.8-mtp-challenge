import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXFastModel
@testable import MLXLLM

/// Opt-in parity check against the mlx-lm reference implementation
/// (`tools/qwen38-flash/qwen4_exp.py`) on the tiny synthetic source tree.
/// Needs `uv` and network access on the first run; enable with
/// `MLXFAST_RUN_QWEN4EXP_PY_PARITY=1`.
final class Qwen4ExpParityTests: XCTestCase {
    func testTinyTreeLogitsMatchReference() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_QWEN4EXP_PY_PARITY"] == "1" else { return }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen4exp-parity-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let src = root.appendingPathComponent("source")
        let dst = root.appendingPathComponent("weights")
        try Qwen4ExpTransformTests.writeTinySource(to: src)

        // Reference logits and the hash constants the reference derives from its config.
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let script = repoRoot.appendingPathComponent("tools/qwen38-flash/parity.py").path
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = ["uv", "run", "--with", "mlx==0.32.0", "--with", "mlx-lm", "python3", script, src.path]
        let pipe = Pipe()
        proc.standardOutput = pipe
        try proc.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        XCTAssertEqual(proc.terminationStatus, 0, "reference run failed; see stderr")
        let ref = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let mults = (ref["multipliers"] as! [NSNumber]).map { Int64(truncating: $0) }
        let sizes = (ref["sizes"] as! [NSNumber]).map { Int64(truncating: $0) }
        let offsets = (ref["offsets"] as! [NSNumber]).map { Int64(truncating: $0) }
        let refLogits = (ref["logits"] as! [[[NSNumber]]])[0].map { $0.map { Float(truncating: $0) } }

        // Our tree: transform, then load in float32 with the reference's hash constants.
        try Qwen4ExpTransform.run(
            .init(source: src, destination: dst, expertGroupSize: 32, expertBits: 4, shardBytes: 32 << 10))
        Qwen4ExpRuntime.weightsDirectory = dst
        let cfgData = try Data(contentsOf: dst.appendingPathComponent("config.json"))
        let model = Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: cfgData))
        let index =
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: dst.appendingPathComponent("model.safetensors.index.json")))
            as! [String: Any]
        var weights = [String: MLXArray]()
        for f in Set((index["weight_map"] as! [String: String]).values) {
            weights.merge(try MLX.loadArrays(url: dst.appendingPathComponent(f))) { a, _ in a }
        }
        weights = model.sanitize(weights: weights)
        // Both sides must compute in the same precision. The reference runs
        // float32; the transform writes bf16, so upcast every float tensor and
        // leave the packed 4-bit expert payload (uint32) alone.
        for (key, value) in weights where value.dtype == .bfloat16 {
            weights[key] = value.asType(.float32)
        }
        weights["model.layers.1.ple.ple_embedding.layer_multipliers"] = MLXArray(mults)
        weights["model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"] = MLXArray(sizes)
        weights["model.layers.1.ple.ple_embedding.ngram_heads_offsets"] = MLXArray(offsets)
        quantize(model: model) { path, _ in weights["\(path).scales"] != nil ? (32, 4, .affine) : nil }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        let logits = model(MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5), cache: model.newCache(parameters: nil))
            .asType(.float32)
        let ours = MLXArray(logits.asArray(Float.self), logits.shape)
        let want = MLXArray(refLogits.flatMap { $0 }, [1, 5, refLogits[0].count])
        let maxDiff = abs(ours - want).max().item(Float.self)
        XCTAssertLessThan(maxDiff, 1e-2, "max |logit delta| vs mlx-lm reference = \(maxDiff)")
    }
}
