import Foundation
import MLX
import XCTest

@testable import MLXFastCore
@testable import MLXFastModel
@testable import MLXLLM

/// The transform's `denseBits` option: dense Linears leave with scales, the
/// tensors upstream keeps in full precision leave as bf16, and the config
/// says what was done -- globally when the parameters match the experts',
/// per layer when they do not.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test -c release \
///       --force-resolved-versions --filter TransformDense
final class Qwen4ExpTransformDenseTests: XCTestCase {
    private func transform(denseBits: Int?, denseGroupSize: Int = 32) throws -> (URL, [String: Any], [String: String]) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwen4exp-dense-\(UUID().uuidString)")
        let src = root.appendingPathComponent("source")
        try Qwen4ExpTransformTests.writeTinySource(to: src)
        let dst = root.appendingPathComponent("weights")
        try Qwen4ExpTransform.run(
            .init(
                source: src, destination: dst, expertGroupSize: 32, expertBits: 4,
                shardBytes: 32 << 10, denseBits: denseBits, denseGroupSize: denseGroupSize))
        let cfg = try JSONSerialization.jsonObject(
            with: Data(contentsOf: dst.appendingPathComponent("config.json"))) as! [String: Any]
        let index = try JSONSerialization.jsonObject(
            with: Data(contentsOf: dst.appendingPathComponent("model.safetensors.index.json"))) as! [String: Any]
        return (root, cfg, index["weight_map"] as! [String: String])
    }

    func testDefaultLeavesDenseAsBF16() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1")
        let (root, cfg, wm) = try transform(denseBits: nil)
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertNil(wm["model.layers.0.linear_attn.in_proj_qkv.scales"], "default must be byte-identical to the old tree")
        XCTAssertNotNil(wm["model.layers.0.mlp.switch_mlp.gate_proj.scales"])
        XCTAssertEqual((cfg["quantization"] as? [String: Any])?.count, 3, "no per-layer entries by default")
    }

    func testDenseQ4MatchingExpertsUsesTheGlobalBlock() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1")
        let (root, cfg, wm) = try transform(denseBits: 4)
        defer { try? FileManager.default.removeItem(at: root) }
        // quantized: the dense Linears
        for k in [
            "model.layers.0.linear_attn.in_proj_qkv", "model.layers.0.linear_attn.in_proj_z",
            "model.layers.0.mlp.shared_expert.gate_proj", "lm_head",
            "model.layers.0.attn_hyper_connection.input_mix_weight_down",  // in-width HC=64
        ] {
            XCTAssertNotNil(wm[k + ".scales"], "\(k) should be quantized")
            XCTAssertNotNil(wm[k + ".biases"], "\(k) should carry biases")
        }
        // kept: what upstream keeps, plus the embedding
        for k in [
            "model.layers.0.mlp.gate", "model.layers.0.mlp.shared_expert_gate", "model.embed_tokens",
            "model.layers.0.linear_attn.conv1d", "model.layers.0.attn_hyper_connection.hc_norm",
        ] {
            XCTAssertNil(wm[k + ".scales"], "\(k) must stay bf16")
            XCTAssertNotNil(wm[k + ".weight"], "\(k) must still be present")
        }
        let q = cfg["quantization"] as? [String: Any]
        XCTAssertEqual(q?["bits"] as? Int, 4)
        XCTAssertEqual(q?.count, 3, "matching parameters need no per-layer entries")
    }

    func testDenseQ8WithQ4ExpertsWritesPerLayerEntries() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1")
        let (root, cfg, wm) = try transform(denseBits: 8)
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertNotNil(wm["model.layers.0.linear_attn.in_proj_qkv.scales"])
        let q = cfg["quantization"] as? [String: Any]
        XCTAssertEqual(q?["bits"] as? Int, 4, "the global block still describes the experts")
        let entry = q?["model.layers.0.linear_attn.in_proj_qkv"] as? [String: Any]
        XCTAssertEqual(entry?["bits"] as? Int, 8, "a mixed tree names each dense path with its own bits")
        XCTAssertNil(q?["model.layers.0.mlp.gate"], "unquantized paths carry no entry; absence means the global block, which the loader only applies where .scales exists")
    }
}
