import Foundation
import MLX
import XCTest

@testable import MLXFastCore
@testable import MLXFastTransform
@testable import MLXLLM

final class Qwen4ExpNGramGatherTests: XCTestCase {
    /// The gather must report how much work it did. Without this, no claim
    /// about the n-gram table's cost is measurable -- the path has never had
    /// any instrumentation, and the table's 102.4 GB size has been used as a
    /// proxy for its cost, which the 3.58 MB-per-forward arithmetic refutes.
    func testGatherStatsCountRowsAndPages() throws {
        Qwen4ExpNGramTable.stats.reset()
        let table = try Qwen4ExpNGramTable.inMemoryFixture(rowsPerShard: 1000, dim: 160, shards: 2)

        // Two tokens, 16 heads each -- the real per-token head count.
        let gids: [[Int64]] = [
            (0 ..< 16).map { Int64($0 * 37) },
            (0 ..< 16).map { Int64(1000 + $0 * 41) },
        ]
        let out = table.gather(gids)
        out.eval()

        XCTAssertEqual(out.shape, [2, 16 * 160])
        let s = Qwen4ExpNGramTable.stats.snapshot()
        XCTAssertEqual(s.calls, 1)
        XCTAssertEqual(s.rows, 32, "16 heads x 2 tokens")
        XCTAssertGreaterThan(s.nanos, 0, "the gather must record elapsed time")
        if Qwen4ExpNGramTable.statsEnabled {
            XCTAssertGreaterThan(s.distinctPages, 0, "page accounting must be populated when enabled")
            XCTAssertLessThanOrEqual(s.distinctPages, 32, "at most one page per row")
        }
    }

    /// Real per-forward geometry: 700 tokens x 16 heads = 11,200 rows of 160
    /// bf16 values. Reports gather wall time so the table's cost can be stated
    /// as a fraction of a forward pass instead of guessed from its file size.
    ///
    /// Baseline for interpretation: one layer's routed MoE measures 22.0 ms at
    /// real geometry, and there are 48 layers, so a whole forward's MoE work is
    /// on the order of 1.06 s. The gather runs ONCE per forward (PLE is at
    /// layer 2 only). Anything under ~10 ms here is under 1 percent of the
    /// forward and is not worth optimizing further.
    func testGatherTimingAtRealGeometry() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a real GPU")
        let rowsPerShard = 2_500_012 / 1000  // scaled fixture, same row width
        let table = try Qwen4ExpNGramTable.inMemoryFixture(
            rowsPerShard: rowsPerShard, dim: 160, shards: 8)

        var rng = SystemRandomNumberGenerator()
        let total = Int64(rowsPerShard * 8)
        let gids: [[Int64]] = (0 ..< 700).map { _ in
            (0 ..< 16).map { _ in Int64.random(in: 0 ..< total, using: &rng) }
        }

        _ = table.gather(gids)  // warm: fault the pages in
        Qwen4ExpNGramTable.stats.reset()
        var best = Double.greatestFiniteMagnitude
        for _ in 0 ..< 5 {
            let t0 = Date()
            let out = table.gather(gids)
            out.eval()
            best = min(best, Date().timeIntervalSince(t0))
        }
        let s = Qwen4ExpNGramTable.stats.snapshot()
        print(
            "[ngram] 700 tok x 16 heads = \(s.rows / 5) rows; best \(String(format: "%.3f", best * 1000)) ms; "
                + "distinct 16KiB pages \(s.distinctPages / 5)")
        XCTAssertEqual(s.rows / 5, 11_200)
    }
    /// The page counter must equal an independently computed page set, for
    /// every encoding.
    ///
    /// This exists because the counter was wrong in a way that hid a real
    /// performance bug. It computed `dataOffsets[shard] + r * stride` using the
    /// weight-tensor stride alone. Under the old split layout a quantized row
    /// also read a scale and a bias from two far-away regions, so the true
    /// fault count was about 3x what the counter reported. The three encodings
    /// therefore reported near-identical page counts, which read as "locality
    /// is the same" when quantized rows were in fact faulting three pages each
    /// and measuring 2.3x slower than bf16.
    ///
    /// The row is one contiguous record now, so the counter is exact. This test
    /// pins that: it recomputes the expected `(shard, page)` set from the gids
    /// in the test itself, rather than trusting the implementation's own
    /// arithmetic.
    func testDistinctPagesMatchesAnIndependentCount() throws {
        guard Qwen4ExpNGramTable.statsEnabled else {
            throw XCTSkip("needs MLX_QWEN4EXP_NGRAM_STATS=1")
        }
        let rowsPerShard = 512, dim = 160, shards = 3
        let src = try Qwen4ExpNGramTable.fixtureDirectory(
            rowsPerShard: rowsPerShard, dim: dim, shards: shards)
        let spec = Qwen4ExpNGramTableSpec(
            directory: src.lastPathComponent, shards: shards, rowsPerShard: rowsPerShard,
            dim: dim, dtype: "bfloat16")

        // Spread the gids over all three shards and across each shard's rows,
        // so pages are genuinely distinct rather than clustered in one block.
        let gids: [[Int64]] = (0 ..< 12).map { t in
            (0 ..< 8).map { h in Int64((t * 8 + h) * 37 % (rowsPerShard * shards)) }
        }

        for (label, dir) in try encodings(from: src) {
            let table = try Qwen4ExpNGramTable(directory: dir, spec: spec)
            let bytesPerRow = try rowWidth(of: dir, spec: spec)
            Qwen4ExpNGramTable.stats.reset()
            _ = table.gather(gids)
            let got = Qwen4ExpNGramTable.stats.snapshot().distinctPages

            // Independent recomputation: one record per row, so a row occupies
            // the page its start offset falls in.
            let base = try weightOffset(of: dir)
            var expected = Set<Int>()
            for row in gids {
                for gid in row {
                    let shard = Int(gid) / rowsPerShard
                    let r = Int(gid) % rowsPerShard
                    expected.insert(shard << 40 | ((base + r * bytesPerRow) / 16384))
                }
            }
            XCTAssertEqual(
                got, expected.count,
                "\(label): counter said \(got) pages, independent count says \(expected.count)")
            XCTAssertGreaterThan(expected.count, 1, "\(label): fixture must span several pages")
        }
    }

    /// bf16 source plus one converted directory per quantized encoding.
    private func encodings(from src: URL) throws -> [(String, URL)] {
        var out: [(String, URL)] = [("bf16", src)]
        for (label, bits) in [
            ("int8", 8), ("int4", 4), ("nvfp4", NGramTableQuantize.nvfp4Bits),
        ] {
            let dst = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("ngram-pages-\(label)-\(UUID().uuidString)")
            try NGramTableQuantize.convert(sourceDir: src, destDir: dst, bits: bits)
            out.append((label, dst))
        }
        return out
    }

    /// Reads `weight`'s row width straight from shard 0's header.
    private func rowWidth(of dir: URL, spec: Qwen4ExpNGramTableSpec) throws -> Int {
        let (header, _) = try shardHeader(of: dir)
        guard let info = header["weight"] as? [String: Any],
            let shape = info["shape"] as? [Int], shape.count == 2
        else { throw MLXFastError.invalidInput("bad fixture header") }
        // bf16 rows are dim values of 2 bytes; quantized rows are byte-shaped.
        return (info["dtype"] as? String) == "BF16" ? shape[1] * 2 : shape[1]
    }

    /// Byte offset of `weight`'s payload within shard 0.
    private func weightOffset(of dir: URL) throws -> Int {
        let (header, headerLength) = try shardHeader(of: dir)
        guard let info = header["weight"] as? [String: Any],
            let range = info["data_offsets"] as? [Int], range.count == 2
        else { throw MLXFastError.invalidInput("bad fixture header") }
        return 8 + headerLength + range[0]
    }

    private func shardHeader(of dir: URL) throws -> ([String: Any], Int) {
        let data = try Data(contentsOf: dir.appendingPathComponent("shard_000.safetensors"))
        let headerLength = Int(
            data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian })
        guard
            let header = try JSONSerialization.jsonObject(
                with: data.subdata(in: 8 ..< (8 + headerLength))) as? [String: Any]
        else { throw MLXFastError.invalidInput("bad fixture header") }
        return (header, headerLength)
    }

}
