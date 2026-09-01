// LOCAL M4 FORK ONLY. Private Apple frameworks; breaks on macOS updates.
// Never wired into the ranked forward; gated by MLX_ANE_DIRECT=1 at call sites.
import Foundation
import MLX

/// Opt-in ANE∥GPU MLP offload for the Qwen35 prefill path (LOCAL FORK ONLY).
///
/// Off by default: unless `MLX_ANE_DIRECT=1` is set, `Qwen35FusedMLP`
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
    /// Read from the environment on first access, then fixed for the process
    /// lifetime (a lazy `static let`), matching the sibling
    /// `qwen35PrefillCPUColumnFraction` style in `Qwen35.swift` (`getenv`,
    /// not `ProcessInfo`).
    public static let enabled: Bool = {
        guard let cString = getenv("MLX_ANE_DIRECT") else { return false }
        return String(cString: cString) == "1"
    }()
    public static let fraction: Double = {
        guard let cString = getenv("MLX_ANE_FRACTION") else { return 0.3125 }
        return Double(String(cString: cString)) ?? 0.3125
    }()
    /// Below this token count the fixed-shape ANE program + marshaling is not
    /// worth it; decode (S=1) and short prefills stay on the GPU.
    public static let minSequenceLength: Int = {
        guard let cString = getenv("MLX_ANE_MIN_SEQ") else { return 128 }
        return Int(String(cString: cString)) ?? 128
    }()
}

/// Diagnostic sink for the ANE offload. Writes to the file named by
/// `MLX_ANE_LOG` (appended, so it survives into the spawned runtime worker
/// where the model actually runs) and falls back to stderr. Local-fork only.
public func aneLog(_ message: String) {
    let line = "[ANE-DIRECT] \(message)\n"
    // Always echo to stderr (forwarded by benchmark --local-iterate) AND, if
    // MLX_ANE_LOG is set and writable, append there too.
    FileHandle.standardError.write(Data(line.utf8))
    if let path = getenv("MLX_ANE_LOG").map({ String(cString: $0) }), !path.isEmpty,
       let fp = fopen(path, "a") {
        fputs(line, fp); fclose(fp)
    }
}

/// Per-layer cache of fixed-shape ANE split programs, keyed by sequence
/// length. A reference type so it can live as a stored property on
/// `Qwen35FusedMLP` and be shared across forwards of that layer. Not
/// thread-safe against concurrent forwards of the SAME layer; the Qwen35
/// prefill path runs layers sequentially on one caller thread, and the
/// internal lock only guards the dictionary against incidental races.
public final class ANESplitMLPCache: @unchecked Sendable {
    private let lock = NSLock()
    private var programs: [Int: ANEFusedSplitMLP] = [:]
    /// Sequence lengths already tried and found unbuildable, so we do not pay
    /// the failed build (dequant + compile attempt) on every forward.
    private var failed: Set<Int> = []

    public init() {}

    /// Returns a cached/newly-built split program for this `S`, or nil to
    /// signal "use the GPU path" (disabled, too short, unsupported weights,
    /// or a prior/again build failure). `gate`/`up` are 4-bit affine
    /// group-64 triples with logical shape `[inter, hidden]`; `down` is a
    /// 4-bit affine group-64 triple with logical shape `[hidden, inter]`.
    public func program(
        forSequenceLength s: Int,
        gateW: MLXArray, gateScales: MLXArray, gateBiases: MLXArray,
        gateBits: Int, gateGroupSize: Int,
        upW: MLXArray, upScales: MLXArray, upBiases: MLXArray,
        upBits: Int, upGroupSize: Int,
        downW: MLXArray, downScales: MLXArray, downBiases: MLXArray,
        downBits: Int, downGroupSize: Int,
        hidden: Int, inter: Int
    ) -> ANEFusedSplitMLP? {
        guard ANESplitConfig.enabled,
              s >= ANESplitConfig.minSequenceLength,
              gateBits == 4, gateGroupSize == 64,
              upBits == 4, upGroupSize == 64,
              downBits == 4, downGroupSize == 64
        else { return nil }

        lock.lock()
        defer { lock.unlock() }
        if let cached = programs[s] { return cached }
        if failed.contains(s) { return nil }

        do {
            let split = try ANEFusedSplitMLP(
                gateW: gateW, gateScales: gateScales, gateBiases: gateBiases,
                upW: upW, upScales: upScales, upBiases: upBiases,
                downW: downW, downScales: downScales, downBiases: downBiases,
                hidden: hidden, inter: inter, sequenceLength: s,
                aneFraction: ANESplitConfig.fraction)
            programs[s] = split
            aneLog("built split program S=\(s) fraction=\(ANESplitConfig.fraction) hidden=\(hidden) inter=\(inter)")
            return split
        } catch {
            failed.insert(s)
            aneLog("BUILD FAILED S=\(s) hidden=\(hidden) inter=\(inter) error=\(error)")
            return nil
        }
    }
}
