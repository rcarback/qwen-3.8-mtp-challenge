import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXFastCore

/// Does rotating a row before quantizing it reduce int4 error?
///
/// The literature reports that with Hadamard rotation, INT4 can surpass NVFP4
/// at matched block size (arXiv 2603.28765). Rotation spreads outliers across
/// channels so no single value dominates its group's range, which is what
/// forces a coarse affine scale.
///
/// This fork's n-gram table is per-ROW affine int4: one scale and bias per 160
/// values. That is far coarser than the group-16 the papers discuss, and coarse
/// groups are exactly where one large value wastes the range for the other 159,
/// so rotation should help MORE here, not less.
///
/// This measures the principle with a random orthogonal rotation rather than a
/// Hadamard matrix. 160 is not a power of two, so a structured transform needs
/// a Kronecker construction (160 = 8 x 20, and Hadamard orders 8 and 20 both
/// exist) or padding. A random orthogonal matrix answers "does rotation help at
/// all" without that work; if it does, the structured version is worth building
/// because it is the one that is cheap at runtime.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_QWEN4EXP_NGRAM_SHARD=<path> \
///       swift test -c release --force-resolved-versions --filter NGramRotation
final class NGramRotationTests: XCTestCase {
    /// Per-row affine int4: one scale and bias per row, as the table stores it.
    private func roundTripInt4(_ rows: MLXArray) -> MLXArray {
        let lo = rows.min(axis: -1, keepDims: true)
        let hi = rows.max(axis: -1, keepDims: true)
        let scale = (hi - lo) / 15.0
        let safe = MLX.where(scale .== 0, MLXArray(Float(1)), scale)
        let q = MLX.clip(MLX.round((rows - lo) / safe), min: 0, max: 15)
        return q * scale + lo
    }

    func testRotationBeforeInt4OnRealRows() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "needs a GPU")
        guard let shard = env["MLXFAST_QWEN4EXP_NGRAM_SHARD"] else {
            throw XCTSkip("set MLXFAST_QWEN4EXP_NGRAM_SHARD to a real ngram shard")
        }

        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: shard))
        guard let weight = arrays["weight"] else { throw XCTSkip("shard has no `weight`") }
        let dim = weight.dim(1)
        // A sample large enough that the row-max statistics are stable.
        let rows = weight[0 ..< 20000].asType(.float32)
        rows.eval()
        let signal = sqrt((rows * rows).mean().item(Float.self))

        // A fixed random orthogonal matrix, from QR of a seeded normal matrix.
        MLXRandom.seed(20260905)
        // qr is CPU-only in MLX; the stream must be explicit or it traps.
        let (qMat, _) = MLX.qr(
            MLXRandom.normal([dim, dim]).asType(.float32), stream: .cpu)
        qMat.eval()
        // Sanity: the rotation must be orthogonal or the inverse is not the
        // transpose and the whole comparison is meaningless.
        let ortho = matmul(qMat, qMat.transposed(1, 0))
        let identity = MLXArray.eye(dim).asType(.float32)
        ortho.eval()
        XCTAssertLessThan(
            MLX.abs(ortho - identity).max().item(Float.self), 1e-3,
            "QR did not produce an orthogonal matrix; the inverse is not the transpose")

        // Arm A: quantize as the table does today.
        let plain = roundTripInt4(rows)
        plain.eval()
        let ePlain = sqrt(((plain - rows) * (plain - rows)).mean().item(Float.self)) / signal

        // Arm B: rotate, quantize, rotate back.
        let rotated = matmul(rows, qMat)
        let rtq = roundTripInt4(rotated)
        let restored = matmul(rtq, qMat.transposed(1, 0))
        restored.eval()
        let eRot =
            sqrt(((restored - rows) * (restored - rows)).mean().item(Float.self)) / signal

        // Row-max error is where the affine range is wasted, so report it too.
        func rowMax(_ a: MLXArray) -> Float {
            let d = MLX.abs(a - rows).max(axis: -1)
            let r = MLX.abs(rows).max(axis: -1)
            let v = (d / MLX.maximum(r, MLXArray(Float(1e-12)))).mean()
            v.eval()
            return v.item(Float.self)
        }

        print(
            "[ngram-rot] \(rows.dim(0)) real rows, dim \(dim), per-row affine int4\n"
                + "  as shipped        rel_rms \(String(format: "%.5f", ePlain))  "
                + "rel row-max \(String(format: "%.5f", rowMax(plain)))\n"
                + "  rotated first     rel_rms \(String(format: "%.5f", eRot))  "
                + "rel row-max \(String(format: "%.5f", rowMax(restored)))\n"
                + "  rotation changes error by "
                + "\(String(format: "%+.1f", (eRot - ePlain) / ePlain * 100)) percent")
    }
}
