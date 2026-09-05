import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXFastModel
@testable import MLXLLM

/// Proves the group-64 transform path end to end on a tiny source.
///
/// The reconstruction gate for affine group-64 passed at three layers spanning
/// the tower, 1.127x to 1.146x against group-32, halving the scale and bias
/// metadata for about 7.6 GB across the routed experts. What remains before
/// adoption is a real transform and a divergence run.
///
/// That real transform is blocked on disk, not on code: `Qwen4ExpTransform`
/// byte-copies the 95 GB n-gram table into its output, so a full run needs
/// about 175 GB against 144 GiB free. This test removes the OTHER risk in that
/// run — that the group-64 path does not work — by transforming the tiny
/// fixture at both group sizes and loading each result.
///
/// SCOPE LIMIT, found by running it: the shared tiny fixture has hidden_size
/// 32, so an expert matrix is 32 wide and MLX refuses to quantize it at group
/// 64 — the last dimension must divide the group size. The real model is
/// hidden 2560 and moe_intermediate 640, both divisible by 64, so this is a
/// fixture limitation and not a defect. This test therefore covers group-32
/// only, and the group-64 plumbing is verified by reading instead:
/// `Qwen4ExpTransform.swift:77` writes `group_size: o.expertGroupSize` into
/// config.json and `:151` passes the same value to `MLX.quantized`, so the
/// weights and the config cannot disagree. Group-64 quantization itself is
/// measured on real tensors in `ExpertGroupSizeTests`.
///
/// A group-64 fixture would need hidden_size 64 or 128, which means a second
/// source writer rather than a parameter on the shared one.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test -c release \
///       --force-resolved-versions --filter TransformGroup64
final class Qwen4ExpTransformGroup64Tests: XCTestCase {
    func testTransformAtGroup64ProducesALoadableTree() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a GPU")

        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwen4exp-g64-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let src = root.appendingPathComponent("source")
        try Qwen4ExpTransformTests.writeTinySource(to: src)

        for groupSize in [32] {
            let dst = root.appendingPathComponent("weights-g\(groupSize)")
            try Qwen4ExpTransform.run(
                .init(
                    source: src, destination: dst, expertGroupSize: groupSize, expertBits: 4,
                    shardBytes: 32 << 10))

            // The runtime config must record the group size it was built with,
            // or a loader will dequantize with the wrong stride and produce
            // plausible garbage rather than failing.
            let cfgData = try Data(contentsOf: dst.appendingPathComponent("config.json"))
            let cfg = try JSONSerialization.jsonObject(with: cfgData) as? [String: Any]
            let quant = cfg?["quantization"] as? [String: Any]
            XCTAssertEqual(
                quant?["group_size"] as? Int, groupSize,
                "config.json must record group_size \(groupSize); a loader that reads the wrong "
                    + "stride dequantizes to plausible garbage instead of failing")
            XCTAssertEqual(quant?["bits"] as? Int, 4, "bit width must be unchanged at group-64")

            // And the tree must actually load and run.
            Qwen4ExpRuntime.weightsDirectory = dst
            let model = Qwen4ExpModel(
                try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: cfgData))
            let index =
                try JSONSerialization.jsonObject(
                    with: Data(contentsOf: dst.appendingPathComponent("model.safetensors.index.json")))
                as? [String: Any]
            var weights = [String: MLXArray]()
            for f in Set((index?["weight_map"] as? [String: String] ?? [:]).values) {
                weights.merge(try MLX.loadArrays(url: dst.appendingPathComponent(f))) { a, _ in a }
            }
            weights = model.sanitize(weights: weights)
            quantize(model: model) { path, _ in
                weights["\(path).scales"] != nil ? (groupSize, 4, .affine) : nil
            }
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])

            let logits = model(
                MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5),
                cache: model.newCache(parameters: nil))
            logits.eval()
            XCTAssertEqual(logits.dim(0), 1)
            XCTAssertTrue(
                MLX.all(MLX.isFinite(logits.asType(.float32))).item(Bool.self),
                "group-\(groupSize) tree produced non-finite logits")

            // Metadata size is the point of the exercise: group-64 carries half
            // the scales and biases of group-32.
            let scaleKeys = weights.keys.filter { $0.hasSuffix(".scales") }
            let scaleElems = scaleKeys.reduce(0) { $0 + (weights[$1]?.size ?? 0) }
            print(
                "[transform-g\(groupSize)] loaded, logits finite; "
                    + "\(scaleKeys.count) scale tensors, \(scaleElems) scale elements")
        }
    }
}
