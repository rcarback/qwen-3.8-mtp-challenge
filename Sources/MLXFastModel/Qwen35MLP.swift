import MLX
import MLXFastCore
import MLXNN

public struct Qwen35MLPWeights {
    public let gateProjection: Qwen35LinearWeight
    public let upProjection: Qwen35LinearWeight
    public let downProjection: Qwen35LinearWeight
    /// Opt-in ANE∥GPU prefill offload cache (LOCAL FORK). A reference type
    /// shared across copies of this value; inert unless `MLXFAST_ANE_DIRECT=1`.
    public let aneCache = ANESplitMLPCache()

    public init(
        gateProjection: Qwen35LinearWeight,
        upProjection: Qwen35LinearWeight,
        downProjection: Qwen35LinearWeight
    ) {
        self.gateProjection = gateProjection
        self.upProjection = upProjection
        self.downProjection = downProjection
    }
}

/// Dense Qwen35 SwiGLU copied from pinned `Qwen3NextMLP`:
/// `down_proj(silu(gate_proj(x)) * up_proj(x))`.
public enum Qwen35MLP {
    public static func forward(
        _ input: MLXArray,
        weights: Qwen35MLPWeights
    ) -> MLXArray {
        if let y = aneForward(input, weights: weights) { return y }
        let gate = silu(Qwen35Ops.linear(input, weights.gateProjection))
        let up = Qwen35Ops.linear(input, weights.upProjection)
        return Qwen35Ops.linear(gate * up, weights.downProjection)
    }

    /// Opt-in ANE∥GPU prefill path (LOCAL FORK, `MLXFAST_ANE_DIRECT=1`).
    /// Returns nil to fall back to the all-GPU SwiGLU above whenever the ANE
    /// split is disabled, the shape is not a supported prefill tile, or the
    /// dispatch throws -- so model output never depends on the ANE.
    /// `input` is `[..., hidden]`; the leading dims flatten to the token
    /// count `S` the fixed-shape ANE program is compiled for.
    private static func aneForward(
        _ input: MLXArray,
        weights: Qwen35MLPWeights
    ) -> MLXArray? {
        guard ANESplitConfig.enabled else { return nil }
        let hidden = weights.gateProjection.logicalShape[1]
        guard input.ndim >= 2, input.shape.last == hidden else { return nil }
        let tokens = input.size / hidden
        guard let split = weights.aneCache.program(
            forSequenceLength: tokens,
            gate: weights.gateProjection,
            up: weights.upProjection,
            down: weights.downProjection)
        else { return nil }
        do {
            let x2 = input.reshaped([tokens, hidden])
            let y2 = try split(x2)                 // [S, hidden]
            return y2.reshaped(input.shape)
        } catch {
            return nil
        }
    }

    static func validateContract(
        weights: Qwen35MLPWeights,
        hiddenSize: Int
    ) throws {
        try weights.gateProjection.validate()
        try weights.upProjection.validate()
        try weights.downProjection.validate()
        guard weights.gateProjection.shape.count == 2,
              weights.upProjection.shape == weights.gateProjection.shape,
              weights.gateProjection.shape[1] == hiddenSize,
              weights.downProjection.shape
                == [hiddenSize, weights.gateProjection.shape[0]]
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 MLP weight shapes are invalid"
            )
        }
    }
}
