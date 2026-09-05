import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import XCTest

@testable import MLXFastCore

/// Does the fused router selection kernel agree with the eager router, and is
/// it faster?
///
/// The eager tail is `argPartition` + slice + `takeAlong` + precise `softmax`:
/// four launches over a `[rows, 512]` tensor to turn expert logits into ten
/// indices and ten weights. `FusedRouterSelect` does it in one.
///
/// Agreement is checked as a SET, not as a sequence. The caller gathers the
/// experts and reduces with `sum(axis: -2)`, so index order is not observable;
/// only membership is. The one way membership can legitimately differ is an
/// exact float32 tie astride the k-boundary, so this counts those directly
/// rather than assuming they never happen.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test -c release \
///       --force-resolved-versions --filter FusedRouterSelect
final class FusedRouterSelectTests: XCTestCase {
    private let experts = 512
    private let k = 10

    private func eager(_ logits: MLXArray) -> (MLXArray, MLXArray) {
        let kth = experts - k
        let idx = MLX.argPartition(logits, kth: kth, axis: -1)[.ellipsis, kth...]
        let w = MLX.softmax(MLX.takeAlong(logits, idx, axis: -1), axis: -1, precise: true)
        return (idx, w)
    }

    func testFusedSelectionMatchesEager() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "needs a GPU")

        for rows in [1, 7, 64, 1024] {
            let logits = MLXRandom.normal([rows, experts]).asType(.float32)
            eval(logits)
            let (ei, ew) = eager(logits)
            let (fi, fw) = FusedRouterSelect.forward(
                logits: logits, topK: k, numExperts: experts)
            eval(ei, ew, fi, fw)

            let eIdx = ei.asArray(Int32.self), fIdx = fi.asArray(Int32.self)
            let eW = ew.asArray(Float.self), fW = fw.asArray(Float.self)
            for r in 0 ..< rows {
                let a = Set(eIdx[(r * k) ..< ((r + 1) * k)])
                let b = Set(fIdx[(r * k) ..< ((r + 1) * k)])
                XCTAssertEqual(a, b, "row \(r) of \(rows) selected a different expert set")
            }
            // Weights are compared as a mapping from expert to weight, because
            // the two paths emit the same ten pairs in different orders.
            for r in 0 ..< rows {
                var want = [Int32: Float]()
                for j in 0 ..< k { want[eIdx[r * k + j]] = eW[r * k + j] }
                for j in 0 ..< k {
                    let got = fW[r * k + j]
                    guard let expected = want[fIdx[r * k + j]] else { continue }
                    XCTAssertEqual(got, expected, accuracy: 1e-6, "row \(r) weight")
                }
            }
        }
    }

    /// How often are the k-th and (k+1)-th logits exactly equal in float32? That
    /// boundary tie is the only way the two paths can pick different sets, so it
    /// is the whole numerical risk of the kernel expressed as a number.
    func testBoundaryTieFrequency() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "needs a GPU")

        let rows = 20000
        let logits = MLXRandom.normal([rows, experts]).asType(.float32)
        eval(logits)
        let sortedL = MLX.sorted(logits, axis: -1)
        eval(sortedL)
        let flat = sortedL.asArray(Float.self)
        var ties = 0
        var minGap = Float.greatestFiniteMagnitude
        for r in 0 ..< rows {
            // Ascending, so the k-boundary sits between positions E-k-1 and E-k.
            let below = flat[r * experts + (experts - k - 1)]
            let above = flat[r * experts + (experts - k)]
            let gap = above - below
            if gap == 0 { ties += 1 }
            minGap = Swift.min(minGap, gap)
        }
        print("[router-tie] \(ties) exact k-boundary ties in \(rows) rows; "
            + "smallest gap \(minGap)")
    }

    func testFusedSelectionSpeed() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "needs a GPU")

        func time(_ body: () -> Void) -> Double {
            for _ in 0 ..< 10 { body() }
            var best = Double.greatestFiniteMagnitude
            for _ in 0 ..< 5 {
                let t0 = DispatchTime.now().uptimeNanoseconds
                for _ in 0 ..< 50 { body() }
                best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 / 50)
            }
            return best
        }

        print("[router-fuse] eager tail against one fused kernel")
        // 1 row is decode; 7000 rows is the measured real prefill width.
        for rows in [1, 512, 7000] {
            let logits = MLXRandom.normal([rows, experts]).asType(.float32)
            eval(logits)
            let e = time {
                let (i, w) = eager(logits)
                eval(i, w)
            }
            let f = time {
                let (i, w) = FusedRouterSelect.forward(
                    logits: logits, topK: k, numExperts: experts)
                eval(i, w)
            }
            let layers = 48
            print(String(
                format: "  rows=%5d  eager %7.3f ms  fused %7.3f ms  %+6.1f%%  "
                    + "x%d layers saves %7.2f ms",
                rows, e, f, (f - e) / e * 100, layers, (e - f) * Double(layers)))
        }
    }
}
