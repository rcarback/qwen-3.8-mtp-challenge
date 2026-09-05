import Foundation
import MLX
import XCTest

@testable import MLXFastCore

/// Is affine group-64 an acceptable representation for the routed experts?
///
/// The checkpoint ships 4-bit affine group-32. Group-64 halves the scale and
/// bias overhead: a 2560-wide row carries 40 groups instead of 80. Across the
/// routed experts that is about 15.1 GB of metadata down to about 7.5 GB, a
/// 7.6 GB saving with no change in weight bit width.
///
/// This measures against the ORIGINAL bf16 source, not against the shipped
/// group-32 tensors. Requantizing an already-quantized tensor is not
/// idempotent even at the same group size: affine quantization fits its grid
/// to the values it is given, so a second pass fits a finer grid to the
/// attained range and the old levels do not land on it. Measured that way,
/// group-32 "re-quantization" showed a 3.3 percent relative error that has
/// nothing to do with group size and would have contaminated the comparison.
/// Both arms here quantize the same bf16 originals, so the only difference
/// between them is the group size.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_QWEN4EXP_SOURCE=<source> \
///       swift test -c release --force-resolved-versions --filter ExpertGroupSize
final class ExpertGroupSizeTests: XCTestCase {
    /// Experts sliced off the front. The full tensor is [512, 1280, 2560] bf16,
    /// about 3.35 GB; 32 experts is ~210 MB and still covers 32 independent
    /// weight matrices.
    private let expertCount = 32

    func testGroup64AgainstGroup32OnTheOriginalWeights() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "needs a GPU")
        guard let source = env["MLXFAST_QWEN4EXP_SOURCE"] else {
            throw XCTSkip("set MLXFAST_QWEN4EXP_SOURCE to the bf16 source directory")
        }
        let root = URL(fileURLWithPath: source)
        let key = "model.language_model.layers.0.mlp.experts.gate_up_proj"

        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        guard
            let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL))
                as? [String: Any],
            let map = index["weight_map"] as? [String: String],
            let shard = map[key]
        else { throw XCTSkip("no weight_map entry for \(key)") }

        let arrays = try MLX.loadArrays(url: root.appendingPathComponent(shard))
        guard let full = arrays[key] else { throw XCTSkip("\(key) missing from \(shard)") }

        // gate_up_proj fuses gate and up, so this covers both projections.
        let ref = full[0 ..< min(expertCount, full.dim(0))].asType(.float32)
        ref.eval()
        let signal = sqrt((ref * ref).mean().item(Float.self))

        var rel = [Int: Float]()
        var line = "[expert-gs] gate_up experts=\(expertCount) shape=\(ref.shape) "
        for gs in [32, 64] {
            let (wq, sc, bi) = quantized(ref, groupSize: gs, bits: 4)
            let back = dequantized(
                wq, scales: sc, biases: bi, groupSize: gs, bits: 4, dtype: .float32)
            back.eval()
            let d = back - ref
            let e = sqrt((d * d).mean().item(Float.self)) / signal
            rel[gs] = e
            // scales + biases, f16 each, for this slice.
            let metaKiB = sc.size * 2 * 2 / 1024
            line += "g\(gs): rel_rms=\(String(format: "%.4f", e)) meta=\(metaKiB)KiB  "
        }
        let ratio = rel[64]! / rel[32]!
        line += "ratio=\(String(format: "%.3f", ratio))x signal=\(String(format: "%.4e", signal))"
        print(line)

        // Both arms quantize the same originals, so this ratio isolates group
        // size. The plan's gate was 1.5x.
        XCTAssertLessThan(
            ratio, 1.5,
            "group-64 relative rms is \(ratio)x group-32's, over the plan's 1.5x gate")
    }
}
