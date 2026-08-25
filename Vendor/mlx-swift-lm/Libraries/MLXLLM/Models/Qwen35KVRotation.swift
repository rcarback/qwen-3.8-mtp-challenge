import Foundation
import MLX

/// A randomized Walsh-Hadamard rotation applied to key and value vectors
/// before they enter a quantized KV cache.
///
/// WHY THIS EXISTS. Affine group quantization spends its bit budget on the
/// widest coordinate in each group of sixty-four. Attention key and value
/// vectors carry persistent outlier channels, so a few coordinates fix the
/// scale for every group they fall in and the rest lose resolution. An
/// orthogonal rotation spreads each outlier across all two hundred fifty-six
/// coordinates of the head, which flattens the per-group range and recovers
/// the wasted bits. This is the first and dominant stage of TurboQuant
/// (arXiv 2504.19874).
///
/// WHY IT IS FREE AT READ TIME. Attention scores are `Q Kᵀ`. Rotating both
/// sides by the same orthogonal `R` leaves the product unchanged, because
/// `(Q R)(K R)ᵀ = Q R Rᵀ Kᵀ = Q Kᵀ`. Values need one inverse rotation on the
/// attention output instead: with `V' = V R`, `A V' = A V R`, so
/// `A V = (A V') Rᵀ`. No dequantization step and no new kernel are involved,
/// and the mask, RoPE, and speculative rollback paths are untouched.
///
/// WHY HADAMARD. `hadamardTransform` is an existing MLX primitive costing
/// `O(d log d)` where a dense random rotation costs `O(d squared)`, and the
/// head dimension here is two hundred fifty-six, which is two to the eighth
/// power and therefore an exact Hadamard size. MLX scales the transform by
/// one over the square root of `d` by default, which makes the matrix
/// orthonormal. The Walsh-Hadamard matrix is also symmetric, so the scaled
/// matrix is its own inverse.
///
/// WHY THE SIGN VECTOR. A bare Hadamard matrix is fixed and public, so a
/// coordinate pattern aligned with one of its rows survives the transform
/// unspread. Multiplying by a random sign vector first removes that
/// alignment. The signs come from a fixed seed, so a cache written by one
/// process stays readable by the next. The session snapshot store depends on
/// that property.
public struct Qwen35KVRotation {
    /// The head dimension this rotation was built for. Applying it to a
    /// different final axis is a programming error.
    public let headDimension: Int

    /// Plus or minus one per coordinate, shaped `[1, 1, 1, headDimension]` so
    /// it broadcasts against `[batch, heads, length, headDimension]`.
    public let signs: MLXArray

    /// Fixed so that the rotation is stable across processes and machines.
    /// The value is the ASCII bytes of "QWENKVRO".
    public static let defaultSeed: UInt64 = 0x5157_454E_4B56_524F

    public init?(headDimension: Int, seed: UInt64 = Qwen35KVRotation.defaultSeed) {
        // hadamardTransform needs a supported size, and every supported size
        // this model can produce is a power of two.
        guard headDimension > 0, headDimension.nonzeroBitCount == 1 else {
            return nil
        }
        self.headDimension = headDimension
        self.signs = MLXArray(
            Qwen35KVRotation.signValues(count: headDimension, seed: seed)
        ).reshaped(1, 1, 1, headDimension)
    }

    /// SplitMix64. A fixed, dependency-free generator, so the sign vector is
    /// identical on every machine and in every process. Do not replace this
    /// with a system random source: a cache written under one sign vector is
    /// unreadable under another.
    public static func signValues(count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0 ..< count).map { _ in
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z = z ^ (z >> 31)
            return (z & 1) == 0 ? 1.0 : -1.0
        }
    }

    /// Rotate into the quantization basis. As a matrix acting on a row
    /// vector this is `x D H`, where `D` is the diagonal sign matrix and `H`
    /// is the orthonormal Walsh-Hadamard matrix.
    public func forward(_ x: MLXArray) -> MLXArray {
        hadamardTransform(x * signs.asType(x.dtype))
    }

    /// Rotate back out. The transpose of `D H` is `H D`, because `H` is
    /// symmetric, so the inverse applies the transform first and the signs
    /// second.
    public func inverse(_ x: MLXArray) -> MLXArray {
        hadamardTransform(x) * signs.asType(x.dtype)
    }
}
