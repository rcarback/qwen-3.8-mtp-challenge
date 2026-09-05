import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXFastCore

/// Are the routed experts similar enough to each other to merge?
///
/// Merging 512 experts down to 128 by weight averaging assumes the experts
/// cluster: that groups of four are close enough that their mean stands in for
/// each of them. If the experts are close to mutually orthogonal, averaging
/// four of them produces a matrix resembling none of the four, and the 1.35x
/// the method table estimates is unreachable at any quality.
///
/// This measures the premise before anyone builds the merge. It flattens each
/// expert's gate_up_proj into a vector, projects onto a random subspace to keep
/// the Gram matrix cheap, and reports the off-diagonal cosine similarity
/// distribution. A random projection preserves pairwise cosines closely enough
/// for a clustering question (Johnson-Lindenstrauss), and the alternative is a
/// 512 by 3.3M Gram matrix.
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_QWEN4EXP_SOURCE=<source> \
///       swift test -c release --force-resolved-versions --filter ExpertSimilarity
final class ExpertSimilarityTests: XCTestCase {
    func testExpertPairwiseSimilarity() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "needs a GPU")
        guard let source = env["MLXFAST_QWEN4EXP_SOURCE"] else {
            throw XCTSkip("set MLXFAST_QWEN4EXP_SOURCE to the bf16 source directory")
        }
        // Layer 0 alone proves nothing about layer 47, and a merge has to hold
        // everywhere. These span the tower.
        for layer in [0, 23, 47] {
            try measureLayer(source: source, layer: layer)
        }
    }

    private func measureLayer(source: String, layer: Int) throws {
        let root = URL(fileURLWithPath: source)
        let key = "model.language_model.layers.\(layer).mlp.experts.gate_up_proj"

        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        guard
            let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL))
                as? [String: Any],
            let map = index["weight_map"] as? [String: String],
            let shard = map[key]
        else { throw XCTSkip("no weight_map entry for \(key)") }
        let arrays = try MLX.loadArrays(url: root.appendingPathComponent(shard))
        guard let full = arrays[key] else { throw XCTSkip("\(key) missing from \(shard)") }

        let experts = min(128, full.dim(0))
        let flatDim = full.dim(1) * full.dim(2)
        let projDim = 4096

        // Random projection: [experts, flatDim] @ [flatDim, projDim].
        // Done expert by expert so the full [128, 3.3M] matrix never exists.
        let proj = MLXRandom.normal([flatDim, projDim]).asType(.float16)
        eval(proj)
        var rows = [MLXArray]()
        for e in 0 ..< experts {
            let v = full[e].reshaped([1, flatDim]).asType(.float16)
            let p = matmul(v, proj).asType(.float32)
            p.eval()
            rows.append(p)
        }
        let embedded = concatenated(rows, axis: 0)  // [experts, projDim]
        let norms = sqrt((embedded * embedded).sum(axis: 1, keepDims: true))
        let unit = embedded / norms
        let gram = matmul(unit, unit.transposed(1, 0))
        gram.eval()

        // Off-diagonal statistics.
        let g = gram.asArray(Float.self)
        var offs = [Float]()
        offs.reserveCapacity(experts * (experts - 1) / 2)
        for i in 0 ..< experts {
            for j in (i + 1) ..< experts { offs.append(g[i * experts + j]) }
        }
        let mean = offs.reduce(0, +) / Float(offs.count)
        let sorted = offs.sorted()
        let p50 = sorted[sorted.count / 2]
        let p99 = sorted[Int(Double(sorted.count) * 0.99)]
        let maxOff = sorted[sorted.count - 1]
        // A 512 -> 128 merge needs THREE close partners per expert, not one.
        // The first, second and third best partner separate "pairs" from
        // "clusters of four": if the second drops sharply below the first,
        // only pairing is supported and the method table's 1.35x, which is for
        // 512 -> 128, does not apply.
        var best1 = [Float](), best2 = [Float](), best3 = [Float]()
        for i in 0 ..< experts {
            var row = [Float]()
            row.reserveCapacity(experts - 1)
            for j in 0 ..< experts where j != i { row.append(g[i * experts + j]) }
            row.sort(by: >)
            best1.append(row[0])
            best2.append(row.count > 1 ? row[1] : -1)
            best3.append(row.count > 2 ? row[2] : -1)
        }
        func avg(_ a: [Float]) -> Float { a.reduce(0, +) / Float(a.count) }
        let m1 = avg(best1), m2 = avg(best2), m3 = avg(best3)

        print(
            "[expert-sim] layer=\(layer) experts=\(experts) proj=\(projDim): off-diagonal cosine "
                + "mean=\(String(format: "%.4f", mean)) p50=\(String(format: "%.4f", p50)) "
                + "p99=\(String(format: "%.4f", p99)) max=\(String(format: "%.4f", maxOff)); "
                + "best1=\(String(format: "%.4f", m1)) best2=\(String(format: "%.4f", m2)) "
                + "best3=\(String(format: "%.4f", m3)) "
                + "drop1to2=\(String(format: "%.4f", m1 - m2))")

        // No assertion on the value: this is a measurement that decides whether
        // to build a merge, not a property the checkpoint must satisfy. It only
        // guards against a broken projection producing a degenerate Gram.
        XCTAssertLessThan(maxOff, 0.999, "projection collapsed: experts appear identical")
    }
}
