// LOCAL M4 FORK ONLY. Private Apple frameworks; breaks on macOS updates.
// Mode selection and the shared program budget for the Qwen4Exp fused ANE
// lanes. Read by both the projection-split lane and the shared-expert lane.
import Foundation

/// Which Qwen4Exp ANE arm runs. `off` and `microbatch` leave both fused lanes
/// inert; `microbatch` is the legacy arm and is the default so an existing
/// `MLX_ANE_DIRECT=1` run is unchanged.
public enum Qwen4ExpANEFusedMode: String, Sendable, CaseIterable {
    case off
    case microbatch
    case split
    case shared
    case both
}

public enum Qwen4ExpANEFused {
    /// `MLX_QWEN4EXP_ANE_MODE`. Default `microbatch`. An unrecognised value
    /// falls back to `microbatch` and logs once.
    public static let mode: Qwen4ExpANEFusedMode = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_QWEN4EXP_ANE_MODE"] else { return .microbatch }
        if let m = Qwen4ExpANEFusedMode(rawValue: raw) { return m }
        fputs("[qwen4exp-ane] unrecognised MLX_QWEN4EXP_ANE_MODE=\(raw); falling back to microbatch\n", stderr)
        return .microbatch
    }()

    /// `MLX_QWEN4EXP_ANE_SPLIT_FRACTION`, default 0.3125. Deliberately not
    /// `MLX_ANE_FRACTION`: that knob tunes the dense Qwen35 tower and the two
    /// must be tunable apart.
    public static let splitFraction: Double = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_QWEN4EXP_ANE_SPLIT_FRACTION"],
            let v = Double(raw)
        else { return 0.3125 }
        return v
    }()

    /// `MLX_QWEN4EXP_ANE_MIN_SEQ`, default 128. Falls back to `MLX_ANE_MIN_SEQ`
    /// when unset, then to 128.
    public static let minSequenceLength: Int = {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["MLX_QWEN4EXP_ANE_MIN_SEQ"], let v = Int(raw) { return v }
        if let raw = env["MLX_ANE_MIN_SEQ"], let v = Int(raw) { return v }
        return 128
    }()

    /// `MLX_QWEN4EXP_ANE_BUDGET_MB`, default 4096.
    public static let programBudgetBytes: Int = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_QWEN4EXP_ANE_BUDGET_MB"],
            let v = Int(raw)
        else { return 4096 * 1024 * 1024 }
        return v * 1024 * 1024
    }()

    /// `MLX_QWEN4EXP_ANE_MAX_PROGRAMS`, default 256. Mode `both` wants 96
    /// programs per bucket, so 128 would refuse the second bucket mid-tower.
    public static let programCountLimit: Int = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_QWEN4EXP_ANE_MAX_PROGRAMS"],
            let v = Int(raw)
        else { return 256 }
        return v
    }()

    /// `MLX_QWEN4EXP_ANE_LOG=1`. A private variable, NOT `MLX_ANE_LOG`:
    /// `aneLog` in `Qwen35ANESplitOffload.swift:112` reads `MLX_ANE_LOG` as a
    /// FILE PATH and does `fopen(path, "a")`, so setting it to `1` creates a
    /// file named `1` in the working directory.
    public static let log: Bool = ProcessInfo.processInfo.environment["MLX_QWEN4EXP_ANE_LOG"] == "1"

    /// `MLX_QWEN4EXP_ANE_ZEROCOPY=1`. Reads the ANE output surface with
    /// `readZeroCopy` plus a trailing `eval` (three barriers per layer)
    /// instead of the default `read` gather (two barriers). See spec 0.5.
    public static let zeroCopyReadback: Bool =
        ProcessInfo.processInfo.environment["MLX_QWEN4EXP_ANE_ZEROCOPY"] == "1"

    /// `MLX_QWEN4EXP_ANE_FP16_GPU=1` (diagnostic). Feature A computes the
    /// prefix rows as a GPU fp16 `matmul` instead of on the ANE. Isolates an
    /// fp16 REPRESENTATION divergence from an ANE-specific one. No ANE, so it
    /// runs without the private frameworks. See spec 8.5.
    public static let fp16GpuAblate: Bool =
        ProcessInfo.processInfo.environment["MLX_QWEN4EXP_ANE_FP16_GPU"] == "1"

    /// `MLX_QWEN4EXP_ANE_VERIFY=1` (diagnostic). At every offloaded call, also
    /// compute the pure-GPU result and log the max absolute difference.
    /// Doubles the work of the offloaded piece. Never use it for timing.
    public static let verify: Bool = ProcessInfo.processInfo.environment["MLX_QWEN4EXP_ANE_VERIFY"] == "1"

    /// True while the legacy structural probe is armed. Both fused lanes
    /// disarm, so `MLX_QWEN4EXP_FORCE_MICROBATCH` still measures pure GPU work
    /// even when a split or shared mode is left in the environment.
    public static let forcedMicroBatchProbe: Bool =
        ProcessInfo.processInfo.environment["MLX_QWEN4EXP_FORCE_MICROBATCH"] != nil

    /// True when the projection-split lane may build and run.
    public static var splitEnabled: Bool {
        Qwen4ExpANELane.enabled && !forcedMicroBatchProbe && (mode == .split || mode == .both)
    }

    /// True when the shared-expert lane may build and run.
    public static var sharedEnabled: Bool {
        Qwen4ExpANELane.enabled && !forcedMicroBatchProbe && (mode == .shared || mode == .both)
    }

    /// True when the legacy micro-batch arm may run.
    public static var microBatchEnabled: Bool { Qwen4ExpANELane.enabled && mode == .microbatch }

    /// Power-of-two program bucket. Delegates to the dense tower's rule so both
    /// towers key programs the same way.
    public static func bucket(_ s: Int) -> Int { ANESplitMLPCache.bucketedSequenceLength(s) }

    /// `F = round(fraction * logicalOut / 64) * 64`, clamped to
    /// `[0, physicalOut]`. `logicalOut` is the whole phase-1 output row count
    /// the fraction is a share of; `physicalOut` is the row count of the one
    /// weight actually being split. See spec 0.3. The 64 rounding is a coarse
    /// tuning grid, NOT a correctness constraint: it is inherited from
    /// `ANEFusedSplitMLP.prefixChannels`, where the group-64 and pack-ratio
    /// preconditions require it. These weights are bf16, and
    /// `ANEDirectDispatch` pads only the sequence axis, never channels. Change
    /// the grid freely.
    public static func prefixChannels(logicalOut: Int, physicalOut: Int, fraction: Double) -> Int {
        let clamped = min(max(fraction, 0.0), 1.0)
        let raw = Int((clamped * Double(logicalOut) / 64.0).rounded()) * 64
        return min(max(raw, 0), physicalOut)
    }

    /// Prefill gate shared by both fused lanes.
    public static func armed(tokens: Int, batch: Int) -> Bool {
        batch == 1 && tokens >= minSequenceLength
    }

    private static let budgetLock = NSLock()
    nonisolated(unsafe) private static var reservedBytesCount = 0
    nonisolated(unsafe) private static var reservedProgramsCount = 0

    /// Reserves headroom for one program before it is built. Returns false when
    /// either the byte budget or the count limit is already spent, in which case
    /// the caller must stay on the GPU. Process wide, thread safe. A successful
    /// build never releases; a FAILED build must call `releaseProgram` so a
    /// program that does not exist does not hold budget forever.
    public static func reserveProgram(bytes: Int, label: String) -> Bool {
        budgetLock.lock()
        defer { budgetLock.unlock() }
        guard reservedBytesCount + bytes <= programBudgetBytes,
            reservedProgramsCount + 1 <= programCountLimit
        else {
            fputs(
                "[qwen4exp-ane] BUDGET REFUSED \(label) bytes=\(bytes) reserved=\(reservedBytesCount)/\(programBudgetBytes) programs=\(reservedProgramsCount)/\(programCountLimit)\n",
                stderr)
            return false
        }
        reservedBytesCount += bytes
        reservedProgramsCount += 1
        return true
    }

    /// Returns the reservation `reserveProgram` granted. Call it only when the
    /// build that reservation was for threw.
    public static func releaseProgram(bytes: Int) {
        budgetLock.lock()
        defer { budgetLock.unlock() }
        reservedBytesCount = max(0, reservedBytesCount - bytes)
        reservedProgramsCount = max(0, reservedProgramsCount - 1)
    }

    /// Diagnostic: bytes and programs reserved so far.
    public static func reservedBytes() -> Int {
        budgetLock.lock()
        defer { budgetLock.unlock() }
        return reservedBytesCount
    }

    public static func reservedPrograms() -> Int {
        budgetLock.lock()
        defer { budgetLock.unlock() }
        return reservedProgramsCount
    }

    /// Test-only. Zeroes both counters. Not for production paths.
    public static func resetForTesting() {
        budgetLock.lock()
        defer { budgetLock.unlock() }
        reservedBytesCount = 0
        reservedProgramsCount = 0
    }
}
