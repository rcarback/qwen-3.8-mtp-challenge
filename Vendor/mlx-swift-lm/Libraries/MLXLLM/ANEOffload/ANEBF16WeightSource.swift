// LOCAL FORK ONLY. Sources the ANE fp16 MLP slices from the original bf16
// base checkpoint instead of dequantizing the shipped 4-bit weights.
import Foundation
import MLX

/// The fp16 ANE-side MLP prefix for one layer: `gate`/`up` are
/// `[F, hidden]` output-channel slices, `down` is the `[hidden, F]`
/// input-channel (column) slice. Together with the GPU's 4-bit
/// `[F..<inter]` suffix they sum to the full SwiGLU MLP.
public struct ANEPrefixWeights {
    public let gate: MLXArray
    public let up: MLXArray
    public let down: MLXArray

    public init(gate: MLXArray, up: MLXArray, down: MLXArray) {
        self.gate = gate
        self.up = up
        self.down = down
    }
}

/// Reads the ANE fp16 MLP prefix of a layer straight out of a Hugging Face
/// bf16 safetensors snapshot (`MLX_ANE_BF16_WEIGHTS=<snapshot dir>`, the
/// directory holding `model.safetensors.index.json`).
///
/// Why: the ANE runs its fraction in fp16 anyway, and bf16 converts to fp16
/// exactly for weights of this magnitude (fp16 carries 3 more mantissa bits;
/// only the range differs, and MLP weights sit around 1e-2). Dequantizing the
/// 4-bit affine group-64 weights instead bakes the quantization error into
/// the ANE slice for no benefit. With the true base weights the ANE fraction
/// becomes a partial de-quantization of the model: the GPU keeps its fast
/// 4-bit `quantizedMM` on the remaining channels, and resident memory is
/// unchanged (only the slices are materialized, never the whole shard).
///
/// Output therefore diverges from the pure-4-bit golden by design; validate
/// with a self-consistent golden (generated with the same flags).
public final class ANEBF16WeightSource: @unchecked Sendable {
    public enum Error: Swift.Error, CustomStringConvertible {
        case indexUnreadable(URL)
        case tensorMissing(String)
        case shardUnreadable(URL, Swift.Error)
        case tensorNotInShard(String, URL)
        case shapeMismatch(String, expected: [Int], actual: [Int])
        case dtypeMismatch(String, DType)

        public var description: String {
            switch self {
            case .indexUnreadable(let url): return "cannot read safetensors index at \(url.path)"
            case .tensorMissing(let name): return "index has no entry for \(name)"
            case .shardUnreadable(let url, let e): return "cannot load shard \(url.lastPathComponent): \(e)"
            case .tensorNotInShard(let name, let url): return "\(name) not found in \(url.lastPathComponent)"
            case .shapeMismatch(let name, let expected, let actual):
                return "\(name) shape \(actual) != expected \(expected)"
            case .dtypeMismatch(let name, let dtype): return "\(name) is \(dtype), expected bfloat16"
            }
        }
    }

    /// Process-wide instance built from `MLX_ANE_BF16_WEIGHTS`, or nil when
    /// the flag is unset. A set-but-unreadable path logs and yields nil so
    /// the offload falls back to the dequantized 4-bit slices, matching the
    /// "any failure -> GPU path" contract of the rest of the offload.
    public static let shared: ANEBF16WeightSource? = {
        guard let path = ANESplitConfig.bf16WeightsPath else { return nil }
        do {
            let source = try ANEBF16WeightSource(snapshot: URL(fileURLWithPath: path))
            aneLog("bf16 weight source: \(path) (\(source.weightMap.count) tensors indexed)")
            return source
        } catch {
            aneLog("bf16 weight source UNAVAILABLE, using dequantized 4-bit slices: \(error)")
            return nil
        }
    }()

    public let snapshot: URL
    /// tensor name -> shard file name, from `model.safetensors.index.json`.
    public let weightMap: [String: String]

    public init(snapshot: URL) throws {
        self.snapshot = snapshot
        let indexURL = snapshot.appendingPathComponent("model.safetensors.index.json")
        guard let data = FileManager.default.contents(atPath: indexURL.path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let map = json["weight_map"] as? [String: String]
        else { throw Error.indexUnreadable(indexURL) }
        self.weightMap = map
    }

    /// Hugging Face tensor name for the Qwen3.x text-tower MLP projections.
    public static func tensorName(layer: Int, projection: String) -> String {
        "model.language_model.layers.\(layer).mlp.\(projection)_proj.weight"
    }

    /// The fp16 prefix for `layer`: gate/up rows `[0..<f]`, down columns
    /// `[0..<f]`. Loads each tensor lazily from its shard, slices, casts to
    /// fp16 and evaluates; the shard handle is dropped on return so only
    /// the `3 * f * hidden` fp16 values stay resident.
    public func mlpPrefix(layer: Int, hidden: Int, inter: Int, f: Int) throws -> ANEPrefixWeights {
        let gate = try load(layer: layer, projection: "gate", expected: [inter, hidden])[0 ..< f, 0...]
            .asType(.float16)
        let up = try load(layer: layer, projection: "up", expected: [inter, hidden])[0 ..< f, 0...]
            .asType(.float16)
        let down = try load(layer: layer, projection: "down", expected: [hidden, inter])[0..., 0 ..< f]
            .asType(.float16)
        eval(gate, up, down)
        return ANEPrefixWeights(gate: gate, up: up, down: down)
    }

    private func load(layer: Int, projection: String, expected: [Int]) throws -> MLXArray {
        let name = Self.tensorName(layer: layer, projection: projection)
        guard let shard = weightMap[name] else { throw Error.tensorMissing(name) }
        let url = snapshot.appendingPathComponent(shard)
        let arrays: [String: MLXArray]
        do { arrays = try loadArrays(url: url) } catch { throw Error.shardUnreadable(url, error) }
        guard let w = arrays[name] else { throw Error.tensorNotInShard(name, url) }
        guard w.shape == expected else { throw Error.shapeMismatch(name, expected: expected, actual: w.shape) }
        guard w.dtype == .bfloat16 else { throw Error.dtypeMismatch(name, w.dtype) }
        return w
    }
}
