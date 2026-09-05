import Foundation
import MLX
import MLXFast
import MLXRandom
import XCTest

@testable import MLXFastCore

/// Corrects the decode breakdown's rms_norm figure, and tests a hoist.
///
/// `DecodeBreakdownTests` timed a HAND-WRITTEN norm — multiply, mean, rsqrt,
/// multiply — and reported it at 17 percent of a decode step. Production does
/// not do that. `Qwen4ExpRMSNorm.callAsFunction` calls `MLXFast.rmsNorm`, one
/// fused kernel, so the breakdown measured an unfused shape the model never
/// runs and that figure is wrong.
///
/// But production is not one launch either. Every call recomputes
///
///     let scale = (1.0 + weight.asType(.float32)).asType(x.dtype)
///
/// which is an add and two casts over a weight that never changes, on every
/// forward. At 48 layers and two norms per layer that is 96 norms and roughly
/// 288 redundant ops per decode step, in a regime this repository has measured
/// to be launch-bound rather than arithmetic-bound.
///
/// Arms: production as written, production with the scale precomputed once,
/// and the hand-written unfused version the breakdown actually timed.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test -c release \
///       --force-resolved-versions --filter RMSNormScaleHoist
final class RMSNormScaleHoistTests: XCTestCase {
    private func time(_ body: () -> Void) -> Double {
        for _ in 0 ..< 20 { body() }  // warm this shape before timing it
        var best = Double.greatestFiniteMagnitude
        for _ in 0 ..< 5 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< 50 { body() }
            best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 / 50)
        }
        return best
    }

    func testScaleHoistAgainstProductionNorm() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "needs a GPU")

        let hidden = 2560
        let eps: Float = 1e-6
        let decodeStepMs = 59.41
        let normsPerToken = 96  // 48 layers x 2

        let x = MLXRandom.normal([1, hidden]).asType(.float16)
        let weight = MLXRandom.normal([hidden]).asType(.float16)
        eval(x, weight)

        // Arm A: exactly what Qwen4ExpRMSNorm does today.
        let asWritten = time {
            let scale = (1.0 + weight.asType(.float32)).asType(x.dtype)
            eval(MLXFast.rmsNorm(x, weight: scale, eps: eps))
        }

        // Arm B: the same result with the scale computed once. The weight is a
        // parameter and does not change between forwards, so this is the
        // hoist a real implementation would cache in the module.
        let hoisted = (1.0 + weight.asType(.float32)).asType(x.dtype)
        eval(hoisted)
        let withHoist = time { eval(MLXFast.rmsNorm(x, weight: hoisted, eps: eps)) }

        // Arm C: what DecodeBreakdownTests actually timed, for the correction.
        let unfused = time {
            eval(x * rsqrt((x * x).mean(axis: -1, keepDims: true) + eps) * weight)
        }

        // The hoist must not change the result.
        let a = MLXFast.rmsNorm(x, weight: (1.0 + weight.asType(.float32)).asType(x.dtype), eps: eps)
        let b = MLXFast.rmsNorm(x, weight: hoisted, eps: eps)
        a.eval(); b.eval()
        XCTAssertEqual(
            MLX.abs(a - b).max().item(Float.self), 0,
            "hoisting the scale changed the norm's output")

        func perToken(_ ms: Double) -> Double { ms * Double(normsPerToken) }
        print(
            "[rms-norm] one row, per call / per token / share of a \(decodeStepMs)ms step\n"
                + "  production as written  \(String(format: "%.4f", asWritten))ms  "
                + "\(String(format: "%6.2f", perToken(asWritten)))ms  "
                + "\(String(format: "%5.1f", perToken(asWritten) / decodeStepMs * 100))%\n"
                + "  scale hoisted          \(String(format: "%.4f", withHoist))ms  "
                + "\(String(format: "%6.2f", perToken(withHoist)))ms  "
                + "\(String(format: "%5.1f", perToken(withHoist) / decodeStepMs * 100))%\n"
                + "  hand-written unfused   \(String(format: "%.4f", unfused))ms  "
                + "\(String(format: "%6.2f", perToken(unfused)))ms  "
                + "\(String(format: "%5.1f", perToken(unfused) / decodeStepMs * 100))%  "
                + "<- what the breakdown timed\n"
                + "  hoist saves            "
                + "\(String(format: "%.2f", perToken(asWritten) - perToken(withHoist)))ms/token, "
                + "\(String(format: "%.1f", (perToken(asWritten) - perToken(withHoist)) / decodeStepMs * 100))% of the step")
    }
}
