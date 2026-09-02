// LOCAL FORK diagnostic. Captures REAL post-norm MLP inputs so ANE numerics
// can be validated on the activations the model actually produces, not on
// synthetic N(0,1) tensors (which passed at 1 ULP while real prompts collapsed).
//
// `MLX_ANE_CAPTURE_DIR=<dir>` enables it. For each layer index listed in
// `MLX_ANE_CAPTURE_LAYERS` (comma separated, default "0,1,15,31,47,63") the
// first prefill chunk with more than 16 tokens writes
// `<dir>/mlp-layer<N>.safetensors` holding `x` (`[S, hidden]`, the MLP input
// as the model computed it) and that layer's 4-bit gate/up/down triples.
// Each layer is written once per process. Off unless the env is set.
import Foundation
import MLX

public enum ANEActivationCapture {
    public static let directory: String? = {
        guard let cString = getenv("MLX_ANE_CAPTURE_DIR") else { return nil }
        let value = String(cString: cString)
        return value.isEmpty ? nil : value
    }()

    public static let layers: Set<Int> = {
        let text = getenv("MLX_ANE_CAPTURE_LAYERS").map { String(cString: $0) } ?? "0,1,15,31,47,63"
        return Set(text.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) })
    }()

    private static let lock = NSLock()
    // Guarded by `lock` on every access.
    nonisolated(unsafe) private static var written: Set<Int> = []

    public static func shouldCapture(layer: Int, tokens: Int) -> Bool {
        guard directory != nil, tokens > 16, layers.contains(layer) else { return false }
        lock.lock(); defer { lock.unlock() }
        return !written.contains(layer)
    }

    /// Writes one layer's capture. `x` is `[S, hidden]`. Each triple is
    /// `(weight, scales, biases)` in the model's own 4-bit affine layout.
    public static func write(
        layer: Int, x: MLXArray,
        gate: (MLXArray, MLXArray, MLXArray),
        up: (MLXArray, MLXArray, MLXArray),
        down: (MLXArray, MLXArray, MLXArray)
    ) {
        guard let directory else { return }
        lock.lock()
        let first = written.insert(layer).inserted
        lock.unlock()
        guard first else { return }
        let arrays: [String: MLXArray] = [
            "x": x,
            "gate.weight": gate.0, "gate.scales": gate.1, "gate.biases": gate.2,
            "up.weight": up.0, "up.scales": up.1, "up.biases": up.2,
            "down.weight": down.0, "down.scales": down.1, "down.biases": down.2,
        ]
        eval(Array(arrays.values))
        let url = URL(fileURLWithPath: directory).appendingPathComponent("mlp-layer\(layer).safetensors")
        do {
            try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
            try save(arrays: arrays, url: url)
            aneLog("capture: layer \(layer) x=\(x.shape) -> \(url.path)")
        } catch {
            aneLog("capture: layer \(layer) FAILED: \(error)")
        }
    }
}
