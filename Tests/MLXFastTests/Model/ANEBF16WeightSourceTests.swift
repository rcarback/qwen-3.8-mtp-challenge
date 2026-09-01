import Foundation
import MLX
import MLXRandom
import Testing
@testable import MLXLLM

/// `ANEBF16WeightSource` reads the ANE fp16 MLP prefix of a layer out of a
/// Hugging Face bf16 safetensors snapshot. Model-free: writes a two-shard
/// synthetic snapshot (gate/up in one shard, down in another, mirroring a
/// layer that straddles a shard boundary) and checks the slices, the
/// bf16->fp16 cast, and the failure modes. Always runs.
@Suite(.serialized)
struct ANEBF16WeightSourceTests {
    private static let hidden = 128
    private static let inter = 256

    /// Builds a snapshot for `layers` layers; returns its URL plus the
    /// full bf16 tensors so the test can compute expected slices.
    private static func makeSnapshot(layers: Int) throws -> (URL, [String: MLXArray]) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-bf16-source-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        var shardA: [String: MLXArray] = [:]
        var shardB: [String: MLXArray] = [:]
        var weightMap: [String: String] = [:]
        var all: [String: MLXArray] = [:]
        MLXRandom.seed(7)
        for layer in 0 ..< layers {
            let gate = MLXRandom.normal([inter, hidden]).asType(.bfloat16)
            let up = MLXRandom.normal([inter, hidden]).asType(.bfloat16)
            let down = MLXRandom.normal([hidden, inter]).asType(.bfloat16)
            eval(gate, up, down)
            let g = ANEBF16WeightSource.tensorName(layer: layer, projection: "gate")
            let u = ANEBF16WeightSource.tensorName(layer: layer, projection: "up")
            let d = ANEBF16WeightSource.tensorName(layer: layer, projection: "down")
            shardA[g] = gate; shardA[u] = up; shardB[d] = down
            weightMap[g] = "model-00001-of-00002.safetensors"
            weightMap[u] = "model-00001-of-00002.safetensors"
            weightMap[d] = "model-00002-of-00002.safetensors"
            all[g] = gate; all[u] = up; all[d] = down
        }
        try save(arrays: shardA, url: root.appendingPathComponent("model-00001-of-00002.safetensors"))
        try save(arrays: shardB, url: root.appendingPathComponent("model-00002-of-00002.safetensors"))
        let index: [String: Any] = ["metadata": ["total_size": 0], "weight_map": weightMap]
        let data = try JSONSerialization.data(withJSONObject: index)
        try data.write(to: root.appendingPathComponent("model.safetensors.index.json"))
        return (root, all)
    }

    private static func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = MLX.abs(a.asType(.float32) - b.asType(.float32))
        eval(d)
        return d.max().item(Float.self)
    }

    @Test("mlpPrefix returns the exact fp16 row/column slices across shards")
    func prefixSlices() throws {
        let (root, all) = try Self.makeSnapshot(layers: 2)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try ANEBF16WeightSource(snapshot: root)
        #expect(source.weightMap.count == 6)

        let f = 64
        let p = try source.mlpPrefix(layer: 1, hidden: Self.hidden, inter: Self.inter, f: f)
        #expect(p.gate.shape == [f, Self.hidden])
        #expect(p.up.shape == [f, Self.hidden])
        #expect(p.down.shape == [Self.hidden, f])
        #expect(p.gate.dtype == .float16 && p.up.dtype == .float16 && p.down.dtype == .float16)

        // bf16 -> fp16 is exact for N(0,1) weights: the slices must match to 0.
        let g = all[ANEBF16WeightSource.tensorName(layer: 1, projection: "gate")]!
        let u = all[ANEBF16WeightSource.tensorName(layer: 1, projection: "up")]!
        let d = all[ANEBF16WeightSource.tensorName(layer: 1, projection: "down")]!
        #expect(Self.maxAbs(p.gate, g[0 ..< f, 0...]) == 0)
        #expect(Self.maxAbs(p.up, u[0 ..< f, 0...]) == 0)
        #expect(Self.maxAbs(p.down, d[0..., 0 ..< f]) == 0)
        // And they are layer 1's weights, not layer 0's.
        let g0 = all[ANEBF16WeightSource.tensorName(layer: 0, projection: "gate")]!
        #expect(Self.maxAbs(p.gate, g0[0 ..< f, 0...]) > 0)
    }

    @Test("a missing layer, a wrong shape, and a missing index all throw")
    func failureModes() throws {
        let (root, _) = try Self.makeSnapshot(layers: 1)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try ANEBF16WeightSource(snapshot: root)

        #expect(throws: ANEBF16WeightSource.Error.self) {
            _ = try source.mlpPrefix(layer: 5, hidden: Self.hidden, inter: Self.inter, f: 64)
        }
        #expect(throws: ANEBF16WeightSource.Error.self) {
            _ = try source.mlpPrefix(layer: 0, hidden: Self.hidden + 64, inter: Self.inter, f: 64)
        }
        #expect(throws: ANEBF16WeightSource.Error.self) {
            _ = try ANEBF16WeightSource(snapshot: root.appendingPathComponent("nope"))
        }
    }

    @Test("ANEFusedSplitMLP with a supplied fp16 prefix uses it (GPU-fp16 ablation form)")
    func splitUsesSuppliedPrefix() throws {
        // Only checkable model-free under the GPU-fp16 ablation, where the
        // prefix runs as a plain matmul and no ANE program is built.
        guard ANESplitConfig.fp16GpuAblate else { return }
        let hidden = Self.hidden, inter = Self.inter, S = 32
        MLXRandom.seed(11)
        func q(_ out: Int, _ inn: Int) -> (MLXArray, MLXArray, MLXArray) {
            let w = (MLXRandom.normal([out, inn]) * 0.1).asType(.bfloat16)
            let (wq, s, b) = quantized(w, groupSize: 64, bits: 4)
            return (wq, s, b ?? s)
        }
        let (gW, gS, gB) = q(inter, hidden), (uW, uS, uB) = q(inter, hidden), (dW, dS, dB) = q(hidden, inter)
        let f = ANEFusedSplitMLP.prefixChannels(inter: inter, aneFraction: 0.25)
        let prefix = ANEPrefixWeights(
            gate: (MLXRandom.normal([f, hidden]) * 0.1).asType(.float16),
            up: (MLXRandom.normal([f, hidden]) * 0.1).asType(.float16),
            down: (MLXRandom.normal([hidden, f]) * 0.1).asType(.float16))
        let withPrefix = try ANEFusedSplitMLP(
            gateW: gW, gateScales: gS, gateBiases: gB, upW: uW, upScales: uS, upBiases: uB,
            downW: dW, downScales: dS, downBiases: dB, hidden: hidden, inter: inter,
            sequenceLength: S, aneFraction: 0.25, prefixFP16: prefix)
        let dequant = try ANEFusedSplitMLP(
            gateW: gW, gateScales: gS, gateBiases: gB, upW: uW, upScales: uS, upBiases: uB,
            downW: dW, downScales: dS, downBiases: dB, hidden: hidden, inter: inter,
            sequenceLength: S, aneFraction: 0.25)
        let x = MLXRandom.normal([S, hidden]).asType(.bfloat16)
        let a = try withPrefix(x), b = try dequant(x)
        #expect(Self.maxAbs(a, b) > 0, "a different prefix must change the output")
    }
}
