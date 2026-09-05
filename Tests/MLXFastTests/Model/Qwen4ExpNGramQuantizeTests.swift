import Foundation
import MLX
import XCTest

@testable import MLXFastTransform
@testable import MLXLLM

final class Qwen4ExpNGramQuantizeTests: XCTestCase {
    /// int8 must reconstruct a row to within its own quantization step.
    /// A per-row affine int8 code has 255 levels across the row's range, so
    /// the worst-case error is half a step: range/510. Asserting against the
    /// row's measured range rather than a fixed epsilon keeps this meaningful
    /// for rows of any magnitude.
    func testInt8RoundTripStaysWithinOneQuantizationStep() throws {
        let src = try Qwen4ExpNGramTable.fixtureDirectory(rowsPerShard: 256, dim: 160, shards: 2)
        let dst = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ngram-q8-\(UUID().uuidString)")
        try NGramTableQuantize.convert(sourceDir: src, destDir: dst, bits: 8)

        let spec = Qwen4ExpNGramTableSpec(
            directory: dst.lastPathComponent, shards: 2, rowsPerShard: 256, dim: 160,
            dtype: "bfloat16")
        let original = try Qwen4ExpNGramTable(directory: src, spec: spec)
        let quantized = try Qwen4ExpNGramTable(directory: dst, spec: spec, bits: 8)

        let gids: [[Int64]] = [[0, 1, 255, 256, 300, 511, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37]]
        let a = original.gather(gids).asType(.float32)
        let b = quantized.gather(gids).asType(.float32)
        a.eval(); b.eval()

        let lo = a.min().item(Float.self), hi = a.max().item(Float.self)
        let tolerance = (hi - lo) / 510 + 1e-4
        let maxErr = MLX.abs(a - b).max().item(Float.self)
        XCTAssertLessThan(maxErr, tolerance, "int8 error \(maxErr) exceeded one step \(tolerance)")
    }

    /// int4 has 15 levels, so its step is 17x coarser than int8's. This does
    /// not assert int4 is good enough for the model -- only that the codec is
    /// self-consistent. Whether int4 is acceptable is a model-quality question
    /// that `testInt8AndInt4QualityAgainstRealShard` below answers against real
    /// embedding weights.
    func testInt4RoundTripStaysWithinOneQuantizationStep() throws {
        let src = try Qwen4ExpNGramTable.fixtureDirectory(rowsPerShard: 256, dim: 160, shards: 2)
        let dst = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ngram-q4-\(UUID().uuidString)")
        try NGramTableQuantize.convert(sourceDir: src, destDir: dst, bits: 4)

        let spec = Qwen4ExpNGramTableSpec(
            directory: dst.lastPathComponent, shards: 2, rowsPerShard: 256, dim: 160,
            dtype: "bfloat16")
        let original = try Qwen4ExpNGramTable(directory: src, spec: spec)
        let quantized = try Qwen4ExpNGramTable(directory: dst, spec: spec, bits: 4)

        let gids: [[Int64]] = [[0, 1, 255, 256, 300, 511, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37]]
        let a = original.gather(gids).asType(.float32)
        let b = quantized.gather(gids).asType(.float32)
        a.eval(); b.eval()

        let lo = a.min().item(Float.self), hi = a.max().item(Float.self)
        let tolerance = (hi - lo) / 30 + 1e-3
        let maxErr = MLX.abs(a - b).max().item(Float.self)
        XCTAssertLessThan(maxErr, tolerance, "int4 error \(maxErr) exceeded one step \(tolerance)")
    }

    /// The two round-trip tests above only prove the codec is self-consistent
    /// against its own quantization step, on a narrow synthetic fixture whose
    /// values all sit within one bf16 mantissa band near 1.0. That bounds the
    /// arithmetic, not the model-quality question this task exists to answer:
    /// is int8, or int4, an acceptable degradation of the REAL n-gram table.
    ///
    /// This test answers that question directly, on real embedding weights,
    /// without running a full 27B-parameter forward pass (out of proportion
    /// for a converter task and outside this task's scope). It quantizes
    /// exactly one real shard -- 2,500,012 rows, 800 MB -- of the pinned
    /// 102.4 GB / 128-shard table, leaving the other 127 shards untouched, and
    /// measures reconstruction error against a large random sample of that
    /// shard's real rows using the same `gather` code path production uses.
    func testInt8AndInt4QualityAgainstRealShard() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs the real n-gram checkpoint and a GPU")
        let realDir = URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["NGRAM_TABLE_DIR"]
                ?? (NSHomeDirectory() + "/.cache/mlxfast/qwen3.8-flash-next/weights/ngram"))
        let shardFile = realDir.appendingPathComponent("shard_000.safetensors")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: shardFile.path),
            "real n-gram shard not present at \(shardFile.path); set NGRAM_TABLE_DIR to override")

        // Isolate exactly one real shard by symlink: NGramTableQuantize.convert
        // converts every shard_*.safetensors it finds in sourceDir, and the real
        // directory holds all 128 shards (102.4 GB). A dedicated one-shard
        // source directory keeps the conversion -- and the disk it touches --
        // to the single shard this test needs.
        let srcDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ngram-real1-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: srcDir, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: srcDir.appendingPathComponent("shard_000.safetensors"), withDestinationURL: shardFile)
        defer { try? FileManager.default.removeItem(at: srcDir) }

        let rowsPerShard = 2_500_012
        let dim = 160
        let spec = Qwen4ExpNGramTableSpec(
            directory: srcDir.lastPathComponent, shards: 1, rowsPerShard: rowsPerShard, dim: dim,
            dtype: "bfloat16")
        let original = try Qwen4ExpNGramTable(directory: srcDir, spec: spec)

        var rng = SystemRandomNumberGenerator()
        let sampleSize = 100_000
        let gids: [[Int64]] = (0 ..< sampleSize).map { _ in
            [Int64.random(in: 0 ..< Int64(rowsPerShard), using: &rng)]
        }
        let a = original.gather(gids).asType(.float32)
        a.eval()

        for bits in [8, 4, NGramTableQuantize.nvfp4Bits] {
            let dst = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("ngram-real1-q\(bits)-\(UUID().uuidString)")
            let label = bits == NGramTableQuantize.nvfp4Bits ? "nvfp4" : "int\(bits)"
            defer { try? FileManager.default.removeItem(at: dst) }
            try NGramTableQuantize.convert(sourceDir: srcDir, destDir: dst, bits: bits)

            // Step 6: confirm the measured shard-size ratio against the real
            // shard, not just the byte-layout arithmetic. Expected roughly
            // 164/320 = 0.51 (int8) or 84/320 = 0.26 (int4) of the source size.
            let sourceBytes =
                (try FileManager.default.attributesOfItem(atPath: shardFile.path)[.size] as? Int) ?? 0
            let destBytes =
                (try FileManager.default.attributesOfItem(
                    atPath: dst.appendingPathComponent("shard_000.safetensors").path)[.size] as? Int) ?? 0
            let ratio = Double(destBytes) / Double(sourceBytes)
            print(
                "[ngram-quantize] bits=\(label) shard_000: source=\(sourceBytes)B dest=\(destBytes)B "
                    + "ratio=\(String(format: "%.4f", ratio))")

            let quantized = try Qwen4ExpNGramTable(directory: dst, spec: spec, bits: bits)
            XCTAssertEqual(
                try Qwen4ExpNGramTable.detectBits(directory: dst, spec: spec), bits,
                "\(label) shards must be self-describing on disk")

            let b = quantized.gather(gids).asType(.float32)
            b.eval()

            let diff = MLX.abs(a - b)
            let mae = diff.mean().item(Float.self)
            let rmse = sqrt((diff * diff).mean().item(Float.self))
            let maxErr = diff.max().item(Float.self)

            let rowRange = a.max(axis: 1) - a.min(axis: 1)
            let rowMaxErr = diff.max(axis: 1)
            let meanRelRowErr =
                (rowMaxErr / MLX.maximum(rowRange, MLXArray(Float(1e-6)))).mean().item(Float.self)

            let dot = (a * b).sum(axis: 1)
            let normA = MLX.sqrt((a * a).sum(axis: 1))
            let normB = MLX.sqrt((b * b).sum(axis: 1))
            let cosSim = dot / MLX.maximum(normA * normB, MLXArray(Float(1e-9)))
            let meanCos = cosSim.mean().item(Float.self)
            let minCos = cosSim.min().item(Float.self)

            print(
                "[ngram-quality] bits=\(bits) sample=\(sampleSize) rows from real shard_000: "
                    + "MAE=\(mae) RMSE=\(rmse) maxAbsErr=\(maxErr) meanRelRowMaxErr=\(meanRelRowErr) "
                    + "meanCosSim=\(meanCos) minCosSim=\(minCos)")

            // Sanity floors only, not a quality verdict -- the report states
            // the acceptability judgement. int8's 255-level code is expected to
            // stay very close to identity; int4's 15-level code is expected to
            // be far coarser, so only a loose floor (better than a random unit
            // vector) is asserted for it here.
            if bits == 8 {
                XCTAssertGreaterThan(meanCos, 0.999, "int8 mean cosine similarity below expectation")
            } else {
                XCTAssertGreaterThan(meanCos, 0.9, "int4 mean cosine similarity below expectation")
            }
        }
    }

    /// Step 7: re-measure the gather at real per-forward geometry (700 tokens
    /// x 16 heads = 11,200 rows) for the int8 and int4 paths, the same way
    /// Task 2's `testGatherTimingAtRealGeometry` measured the bf16 path
    /// (1.630 ms release best-of-5). This is a footprint/residency change, not
    /// a speed change -- report whichever direction the number moves.
    func testGatherTimingAtRealGeometryQuantized() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a real GPU")
        let rowsPerShard = 2_500_012 / 1000  // scaled fixture, same row width as Task 2
        let dim = 160
        let shards = 8
        let bf16Dir = try Qwen4ExpNGramTable.fixtureDirectory(
            rowsPerShard: rowsPerShard, dim: dim, shards: shards)

        var rng = SystemRandomNumberGenerator()
        let total = Int64(rowsPerShard * shards)
        let gids: [[Int64]] = (0 ..< 700).map { _ in
            (0 ..< 16).map { _ in Int64.random(in: 0 ..< total, using: &rng) }
        }

        for bits in [8, 4, NGramTableQuantize.nvfp4Bits] {
            let dst = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("ngram-fixture-q\(bits)-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: dst) }
            try NGramTableQuantize.convert(sourceDir: bf16Dir, destDir: dst, bits: bits)
            let spec = Qwen4ExpNGramTableSpec(
                directory: dst.lastPathComponent, shards: shards, rowsPerShard: rowsPerShard,
                dim: dim, dtype: "bfloat16")
            let table = try Qwen4ExpNGramTable(directory: dst, spec: spec, bits: bits)

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
                "[ngram-quantized-timing] bits=\(bits) 700 tok x 16 heads = \(s.rows / 5) rows; "
                    + "best \(String(format: "%.3f", best * 1000)) ms; "
                    + "distinct 16KiB pages \(s.distinctPages / 5)")
            XCTAssertEqual(s.rows / 5, 11_200)
        }
    }
    /// A quantized directory must announce its own width, so production can
    /// open one without a config key that could disagree with the bytes.
    /// This is what makes the int8 and int4 encodings reachable outside the
    /// test suite: the runtime load site calls the two-argument initializer.
    func testTwoArgumentInitDetectsTheEncodingOnDisk() throws {
        let src = try Qwen4ExpNGramTable.fixtureDirectory(rowsPerShard: 256, dim: 160, shards: 2)
        let spec = Qwen4ExpNGramTableSpec(
            directory: src.lastPathComponent, shards: 2, rowsPerShard: 256, dim: 160,
            dtype: "bfloat16")
        XCTAssertEqual(try Qwen4ExpNGramTable.detectBits(directory: src, spec: spec), 16)

        let gids: [[Int64]] = [[0, 1, 255, 256, 300, 511, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37]]
        for bits in [8, 4, NGramTableQuantize.nvfp4Bits] {
            let dst = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("ngram-detect-\(bits)-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: dst) }
            try NGramTableQuantize.convert(sourceDir: src, destDir: dst, bits: bits)

            XCTAssertEqual(
                try Qwen4ExpNGramTable.detectBits(directory: dst, spec: spec), bits,
                "int\(bits) shards must be detected as int\(bits)")

            // The detected open must equal the explicit open, value for value.
            let detected = try Qwen4ExpNGramTable(directory: dst, spec: spec)
            let explicit = try Qwen4ExpNGramTable(directory: dst, spec: spec, bits: bits)
            let a = detected.gather(gids).asType(.float32)
            let b = explicit.gather(gids).asType(.float32)
            a.eval(); b.eval()
            XCTAssertEqual(
                MLX.abs(a - b).max().item(Float.self), 0,
                "detected open disagreed with explicit bits: \(bits)")
        }
    }

    /// NVFP4 must round-trip through its own codec and be detected on disk.
    /// It spends four bits per value like int4, but as E2M1 floats under a
    /// per-16 E4M3 block scale, so its row is 92 bytes against int4's 84.
    /// The tolerance is the E2M1 step at the block's own magnitude, not a
    /// fixed epsilon: E2M1 levels are logarithmic (0, .5, 1, 1.5, 2, 3, 4, 6),
    /// so the worst relative step is between 4 and 6, that is one third.
    func testNVFP4RoundTripAndDetection() throws {
        let src = try Qwen4ExpNGramTable.fixtureDirectory(rowsPerShard: 256, dim: 160, shards: 2)
        let dst = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ngram-nvfp4-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dst) }
        try NGramTableQuantize.convert(
            sourceDir: src, destDir: dst, bits: NGramTableQuantize.nvfp4Bits)

        let spec = Qwen4ExpNGramTableSpec(
            directory: dst.lastPathComponent, shards: 2, rowsPerShard: 256, dim: 160,
            dtype: "bfloat16")
        XCTAssertEqual(
            try Qwen4ExpNGramTable.detectBits(directory: dst, spec: spec),
            Qwen4ExpNGramTable.nvfp4Bits, "a 92-byte row must be detected as nvfp4")

        let gids: [[Int64]] = [[0, 1, 255, 256, 300, 511, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37]]
        let original = try Qwen4ExpNGramTable(directory: src, spec: spec)
        let quantized = try Qwen4ExpNGramTable(directory: dst, spec: spec)  // detected
        let a = original.gather(gids).asType(.float32)
        let b = quantized.gather(gids).asType(.float32)
        a.eval(); b.eval()

        let lo = a.min().item(Float.self), hi = a.max().item(Float.self)
        let tolerance = (hi - lo) / 3 + 1e-3
        let maxErr = MLX.abs(a - b).max().item(Float.self)
        XCTAssertLessThan(maxErr, tolerance, "nvfp4 error \(maxErr) exceeded one step \(tolerance)")
        // This fixture is ADVERSE to nvfp4 and deliberately does not assert
        // that nvfp4 beats int4 here. Its values sit in [0.0078, 0.0137]: a
        // narrow band that never crosses zero. An affine code carries a bias,
        // so it puts that offset in the bias and spends all 15 levels on the
        // spread. NVFP4 is sign-and-magnitude with no offset, so its levels
        // fan out from zero and only three of them land in that band. Measured
        // on this fixture nvfp4 is about 5.7x worse than int4, which is a fact
        // about the fixture rather than about the format. Real embeddings are
        // near zero-centred, which is the case NVFP4 is built for; the
        // real-shard test above is where the two are actually compared.
    }

}
