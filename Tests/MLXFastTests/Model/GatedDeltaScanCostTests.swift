import Foundation
import MLX
import Testing

@testable import MLXFastCore
@testable import MLXLLM

/// Prices the gated-delta recurrence on its own, at the Qwen 3.8 geometry.
///
/// The whole-model prefill curve is flat in T, which points at a per-token
/// serial term. This isolates the candidate: if the recurrence alone costs
/// ~T x constant with no amortization, it is the term, and a chunked parallel
/// scan is the fix. 48 of the 64 layers are gated-delta.
@Suite(.serialized)
struct GatedDeltaScanCostTests {
    @Test("gated-delta recurrence cost versus T")
    func scanCost() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let Hk = 16, Hv = 48, Dk = 128, Dv = 128, B = 1
        print("\nGatedDelta recurrence, one layer (Hk 16, Hv 48, Dk/Dv 128)")
        print("      T   seconds   ms/token   x48 layers ms/token")
        for T in [128, 512, 2048, 8192] {
            let q = MLXRandom.normal([B, T, Hk, Dk]).asType(.bfloat16)
            let k = MLXRandom.normal([B, T, Hk, Dk]).asType(.bfloat16)
            let v = MLXRandom.normal([B, T, Hv, Dv]).asType(.bfloat16)
            let a = MLXRandom.normal([B, T, Hv])
            let bb = MLXRandom.normal([B, T, Hv])
            let aLog = MLXRandom.normal([Hv])
            let dtBias = MLXRandom.normal([Hv])
            eval(q, k, v, a, bb, aLog, dtBias)
            // warm
            var (y, st) = gatedDeltaUpdate(
                q: q, k: k, v: v, a: a, b: bb, aLog: aLog, dtBias: dtBias)
            eval(y, st)
            let start = Date()
            (y, st) = gatedDeltaUpdate(
                q: q, k: k, v: v, a: a, b: bb, aLog: aLog, dtBias: dtBias)
            eval(y, st)
            let dt = Date().timeIntervalSince(start)
            print(String(
                format: "  %5d  %8.3f  %9.4f  %18.3f",
                T, dt, 1000 * dt / Double(T), 48 * 1000 * dt / Double(T)))
        }
        print("")
    }
}
