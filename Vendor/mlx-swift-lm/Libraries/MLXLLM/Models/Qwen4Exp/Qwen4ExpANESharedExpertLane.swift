// LOCAL M4 FORK ONLY. Private Apple frameworks; breaks on macOS updates.
// The Qwen4Exp shared expert as ONE fused ANE program, run concurrently with
// the GPU's router and routed gather-GEMM. Gated by MLX_ANE_DIRECT=1 plus
// MLX_QWEN4EXP_ANE_MODE=shared|both at the call site.
import Foundation
import MLX
import MLXNN

/// The whole shared-expert SwiGLU (`down(silu(gate(x)) * up(x))`) as one fused
/// fp16 ANE program. This is `ANEFusedSplitMLP`'s `F == inter` shape without its
/// nine unused quantized arguments: `ANEFusedMLP` is the construction path both
/// use.
final class Qwen4ExpANESharedExpert {
    let hidden: Int
    let inter: Int
    /// Compiled row count (the bucket).
    let sequenceLength: Int
    /// fp16 weight bytes this program holds resident.
    let programBytes: Int

    private let mlp: ANEFusedMLP

    /// `gate` and `up` are `[inter, hidden]`; `down` is `[hidden, inter]`. Any
    /// float dtype; they are cast to fp16 inside. Throws when the program
    /// cannot be built.
    init(gate: MLXArray, up: MLXArray, down: MLXArray, sequenceLength: Int) throws {
        precondition(gate.ndim == 2 && up.ndim == 2 && down.ndim == 2,
                     "Qwen4ExpANESharedExpert weights must be 2D")
        inter = gate.dim(0)
        hidden = gate.dim(1)
        precondition(
            up.shape == [inter, hidden] && down.shape == [hidden, inter],
            "Qwen4ExpANESharedExpert: up expected [\(inter), \(hidden)], got \(up.shape); "
                + "down expected [\(hidden), \(inter)], got \(down.shape)")
        self.sequenceLength = sequenceLength
        programBytes = (2 * inter * hidden + hidden * inter) * 2
        mlp = try ANEFusedMLP(
            hidden: hidden, innerFraction: inter, sequenceLength: sequenceLength,
            gate: gate.asType(.float16), up: up.asType(.float16), down: down.asType(.float16))
    }

    /// CALLER THREAD ONLY. Pads `[T, hidden]` to the compiled row count in the
    /// SOURCE dtype and stages the ANE request. Contains BARRIER 1.
    func makeInput(_ x: MLXArray) throws -> ANEDirectDispatch.Prepared {
        let tokens = x.dim(0)
        precondition(
            x.ndim == 2 && x.dim(1) == hidden,
            "Qwen4ExpANESharedExpert.makeInput expected x [T, \(hidden)], got \(x.shape)")
        precondition(
            tokens <= sequenceLength,
            "Qwen4ExpANESharedExpert.makeInput: T=\(tokens) > compiled \(sequenceLength)")
        let padded =
            tokens == sequenceLength
            ? x
            : concatenated([x, MLXArray.zeros([sequenceLength - tokens, hidden], dtype: x.dtype)], axis: 0)
        return try mlp.makeInput(padded)
    }

    /// BACKGROUND SAFE. No MLX.
    func predict(_ p: ANEDirectDispatch.Prepared) throws { try mlp.predict(p) }

    /// CALLER THREAD ONLY. Reads the surface and drops the padded rows.
    /// Returns `[tokens, hidden]` fp16. Uses `ANEDirectDispatch.read` by
    /// default and `readZeroCopy` under `MLX_QWEN4EXP_ANE_ZEROCOPY=1`; it does
    /// NOT call `ANEFusedMLP.readOutput`, which is hardwired to `readZeroCopy`.
    func readOutput(_ p: ANEDirectDispatch.Prepared, tokens: Int) -> MLXArray {
        let full = Qwen4ExpANEFused.zeroCopyReadback ? ANEDirectDispatch.readZeroCopy(p) : ANEDirectDispatch.read(p)
        return tokens == sequenceLength ? full : full[0 ..< tokens, 0...]
    }

    /// CALLER THREAD ONLY. Stage, run, read. Used by tests only; the model call
    /// site uses the three phases so the GPU leg can overlap.
    func callAsFunction(_ x: MLXArray) throws -> MLXArray {
        let tokens = x.dim(0)
        let p = try makeInput(x)
        try predict(p)
        return readOutput(p, tokens: tokens)
    }
}

/// Per-block cache keyed by `Qwen4ExpANEFused.bucket(tokens)`. `NSLock`
/// guarded, matching `Qwen4ExpANESplitProjectionCache` and `ANESplitMLPCache`.
final class Qwen4ExpANESharedExpertCache: @unchecked Sendable {
    private let lock = NSLock()
    private var programs: [Int: Qwen4ExpANESharedExpert] = [:]
    /// Buckets already tried and found unbuildable (or budget-refused), so the
    /// failure is paid once instead of on every forward.
    private var failed: Set<Int> = []
    let label: String

    init(label: String) { self.label = label }

    func program(
        forTokens tokens: Int,
        gate: () -> MLXArray, up: () -> MLXArray, down: () -> MLXArray
    ) -> Qwen4ExpANESharedExpert? {
        let key = Qwen4ExpANEFused.bucket(tokens)
        lock.lock()
        defer { lock.unlock() }
        if let p = programs[key] { return p }
        if failed.contains(key) { return nil }
        let g = gate()
        let u = up()
        let d = down()
        let bytes = (2 * g.dim(0) * g.dim(1) + d.dim(0) * d.dim(1)) * 2
        guard Qwen4ExpANEFused.reserveProgram(bytes: bytes, label: label) else {
            failed.insert(key)
            return nil  // reserveProgram already logged
        }
        do {
            let p = try Qwen4ExpANESharedExpert(gate: g, up: u, down: d, sequenceLength: key)
            programs[key] = p
            if Qwen4ExpANEFused.log {
                fputs("[qwen4exp-ane] shared \(label) hidden=\(p.hidden) inter=\(p.inter) bucket=\(key)\n", stderr)
            }
            return p
        } catch {
            // Give the reservation back. A build that failed after reserving
            // would otherwise hold budget for a program that does not exist.
            Qwen4ExpANEFused.releaseProgram(bytes: bytes)
            failed.insert(key)
            fputs("[qwen4exp-ane] SHARED BUILD FAILED \(label) bucket=\(key): \(error); GPU path stays\n", stderr)
            return nil
        }
    }
}
