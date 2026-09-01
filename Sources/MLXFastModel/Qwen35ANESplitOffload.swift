import Foundation
import MLX
import MLXFastCore

/// Opt-in ANE∥GPU MLP offload for the Qwen35 prefill path (LOCAL FORK ONLY).
///
/// Off by default: unless `MLXFAST_ANE_DIRECT=1` is set, `Qwen35MLP.forward`
/// runs the unchanged all-GPU SwiGLU and this type is never touched. When
/// enabled, the first prefill forward at a given sequence length `S` lazily
/// builds an `ANEFusedSplitMLP` (fused ANE fraction + GPU 4-bit complement,
/// see `ANEFusedSplitMLP`) for that layer and caches it keyed by `S`;
/// subsequent forwards at that `S` reuse it. Any failure -- ANE unavailable,
/// non-64-aligned or non-4-bit weights, a build/dispatch error -- falls back
/// to the all-GPU path, so correctness never depends on the ANE succeeding.
///
/// This targets large-context COLD PREFILL, where the ANE overlap measured a
/// ~1.17-1.21x MLP speedup at S=512. It is never part of a ranked
/// submission; it is a local serve-performance experiment gated behind an
/// environment flag.
public enum ANESplitConfig {
    /// Read once at process start.
    public static let enabled: Bool = ProcessInfo.processInfo.environment["MLXFAST_ANE_DIRECT"] == "1"
    public static let fraction: Double = Double(ProcessInfo.processInfo.environment["MLXFAST_ANE_FRACTION"] ?? "0.3125") ?? 0.3125
    /// Below this token count the fixed-shape ANE program + marshaling is not
    /// worth it; decode (S=1) and short prefills stay on the GPU.
    public static let minSequenceLength: Int = Int(ProcessInfo.processInfo.environment["MLXFAST_ANE_MIN_SEQ"] ?? "128") ?? 128
}

/// Per-layer cache of fixed-shape ANE split programs, keyed by sequence
/// length. A reference type so it can live inside the value-type
/// `Qwen35MLPWeights` and be shared across its copies. Not thread-safe
/// against concurrent forwards of the SAME layer; the Qwen35 prefill path
/// runs layers sequentially on one caller thread, and the internal lock only
/// guards the dictionary against incidental races.
public final class ANESplitMLPCache: @unchecked Sendable {
    private let lock = NSLock()
    private var programs: [Int: ANEFusedSplitMLP] = [:]
    /// Sequence lengths already tried and found unbuildable, so we do not pay
    /// the failed build (dequant + compile attempt) on every forward.
    private var failed: Set<Int> = []

    public init() {}

    /// Returns a cached/newly-built split program for this `S`, or nil to
    /// signal "use the GPU path" (disabled, too short, unsupported weights,
    /// or a prior/again build failure).
    func program(
        forSequenceLength s: Int,
        gate: Qwen35LinearWeight,
        up: Qwen35LinearWeight,
        down: Qwen35LinearWeight
    ) -> ANEFusedSplitMLP? {
        guard ANESplitConfig.enabled,
              s >= ANESplitConfig.minSequenceLength,
              s % 32 == 0,
              gate.bits == 4, gate.groupSize == 64,
              let gs = gate.scales, let gb = gate.biases,
              let us = up.scales, let ub = up.biases,
              let ds = down.scales, let db = down.biases
        else { return nil }

        lock.lock()
        defer { lock.unlock() }
        if let cached = programs[s] { return cached }
        if failed.contains(s) { return nil }

        let hidden = gate.logicalShape[1]
        let inter = gate.logicalShape[0]
        do {
            let split = try ANEFusedSplitMLP(
                gateW: gate.weight, gateScales: gs, gateBiases: gb,
                upW: up.weight, upScales: us, upBiases: ub,
                downW: down.weight, downScales: ds, downBiases: db,
                hidden: hidden, inter: inter, sequenceLength: s,
                aneFraction: ANESplitConfig.fraction)
            programs[s] = split
            return split
        } catch {
            failed.insert(s)
            return nil
        }
    }
}
