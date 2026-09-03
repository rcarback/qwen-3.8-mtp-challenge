import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen4ExpNGramTests: XCTestCase {
    // Constants read from the checkpoint tensors on 2026-09-02; vectors from a
    // pure-Python reference of the mlx-lm PR 1788 hashing with these constants.
    static let mults: [Int64] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    static let sizes: [Int64] = [
        20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069, 20_000_077,
        20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159, 20_000_161, 20_000_171,
    ]
    static let offsets: [Int64] = [
        0, 20_000_003, 40_000_026, 60_000_059, 80_000_106, 100_000_165, 120_000_228, 140_000_297,
        160_000_374, 180_000_455, 200_000_548, 220_000_655, 240_000_802, 260_000_955, 280_001_114,
        300_001_275,
    ]
    static let eos: Int64 = 248044

    func hasher() -> Qwen4ExpNGramHasher {
        Qwen4ExpNGramHasher(
            multipliers: Self.mults, sizes: Self.sizes, offsets: Self.offsets,
            eos: Self.eos, ngramSize: 3, headsPerNgram: 8)
    }

    func testHashVectorsMatchReference() {
        let ids: [Int64] = [151644, 872, 198, 9707, 1879, 0]
        let rows = hasher().gids(history: [Self.eos, Self.eos] + ids)
        XCTAssertEqual(rows.count, 6)
        XCTAssertEqual(
            rows[0],
            [
                9_213_577, 25_503_567, 59_963_735, 67_280_845, 97_263_283, 108_604_657, 126_879_742,
                153_604_170, 163_991_649, 192_963_728, 210_734_887, 224_850_479, 244_505_954,
                265_606_139, 292_960_741, 312_140_635,
            ])
        XCTAssertEqual(
            rows[1],
            [
                6_954_764, 37_428_000, 54_939_688, 74_004_108, 95_568_253, 116_575_022, 138_540_160,
                142_009_604, 165_349_151, 193_458_441, 202_840_010, 231_812_024, 252_625_381,
                274_082_424, 294_711_253, 318_927_564,
            ])
        XCTAssertEqual(
            rows[5],
            [
                11_899_280, 26_999_030, 54_577_278, 65_218_531, 94_369_136, 117_425_429, 122_015_444,
                148_146_117, 179_478_088, 187_107_196, 216_111_291, 222_450_264, 246_479_657,
                270_529_504, 291_884_105, 318_690_556,
            ])
    }

    func testEOSInsideSequenceResetsContext() {
        let rows = hasher().gids(history: [Self.eos, Self.eos, 9707, 248044, 1879])
        XCTAssertEqual(
            rows[0],
            [
                16_410_909, 39_682_429, 55_103_279, 60_931_720, 87_006_904, 116_506_179, 131_512_017,
                152_932_897, 169_641_436, 182_022_480, 209_277_433, 237_891_023, 256_841_529,
                277_007_269, 290_665_954, 300_984_276,
            ])
        XCTAssertEqual(
            rows[2],
            [
                4_534_223, 34_461_195, 53_188_831, 67_623_329, 89_624_755, 104_428_266, 137_386_396,
                149_402_401, 160_426_620, 196_479_308, 211_378_993, 239_143_795, 249_988_146,
                262_053_263, 299_679_851, 309_845_962,
            ])
    }

    func testMixedIntermediateWrapsLikeInt64() {
        XCTAssertEqual(Qwen4ExpNGramHasher.mix([151644, Self.eos], Self.mults), 8_420_194_656_258_222_560)
        XCTAssertEqual(
            Qwen4ExpNGramHasher.mix([151644, Self.eos, Self.eos], Self.mults), 8_026_254_874_704_583_700)
        XCTAssertEqual(Qwen4ExpNGramHasher.pythonMod(-7, 5), 3)
        XCTAssertEqual(Qwen4ExpNGramHasher.pythonMod(7, 5), 2)
    }

    func testTableGatherReadsMappedRows() throws {
        // Two shards of 4 rows x 2 dims, bf16; row r of shard s holds value (s*4 + r) in every column.
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen4exp-ngram-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        for s in 0 ..< 2 {
            var values = [Float]()
            for r in 0 ..< 4 {
                let v = Float(s * 4 + r)
                values.append(v)
                values.append(v)
            }
            let w = MLXArray(values, [4, 2]).asType(.bfloat16)
            try MLX.save(
                arrays: ["weight": w],
                url: dir.appendingPathComponent(String(format: "shard_%03d.safetensors", s)))
        }
        let spec = Qwen4ExpNGramTableSpec(directory: ".", shards: 2, rowsPerShard: 4, dim: 2, dtype: "bfloat16")
        let table = try Qwen4ExpNGramTable(directory: dir, spec: spec)
        let out = table.gather([[0, 5], [7, 3]])  // [T=2, heads=2] -> [2, 4]
        XCTAssertEqual(out.shape, [2, 4])
        XCTAssertEqual(out.asType(.float32).asArray(Float.self), [0, 0, 5, 5, 7, 7, 3, 3])
    }

    func testPLEForwardIsStepwiseConsistent() throws {
        // hidden 8, hc 2, ple_embed 8, headsPerNgram 2 -> 4 heads x dim 2.
        var args = Qwen4ExpTextConfiguration()
        args.hiddenSize = 8
        args.hcCount = 2
        args.hcLowrank = 4
        args.pleEmbedDim = 8
        args.headsPerNgram = 2
        args.ngramSize = 3
        args.pleConvKernelSize = 4
        args.splitNgramParts = 2
        args.eosTokenId = 7
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen4exp-ple-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let rows = 16
        for s in 0 ..< 2 {
            try MLX.save(
                arrays: ["weight": MLXRandom.normal([rows, 2]).asType(.bfloat16)],
                url: dir.appendingPathComponent(String(format: "shard_%03d.safetensors", s)))
        }
        args.ngramTable = Qwen4ExpNGramTableSpec(
            directory: ".", shards: 2, rowsPerShard: rows, dim: 2, dtype: "bfloat16")
        Qwen4ExpRuntime.weightsDirectory = dir
        let ple = Qwen4ExpPLELayer(args)
        // tiny hash constants: 4 heads with vocab sizes that fit the 32-row table
        ple.update(
            parameters: ModuleParameters.unflattened([
                "ple_embedding.layer_multipliers": MLXArray([Int64(3), 5, 7]),
                "ple_embedding.ngram_heads_vocab_sizes": MLXArray([Int64(7), 7, 8, 8]),
                "ple_embedding.ngram_heads_offsets": MLXArray([Int64(0), 7, 14, 22]),
            ]))
        let ids = MLXArray([Int32(1), 2, 3, 4, 5, 6]).reshaped(1, 6)
        let hidden = MLXRandom.normal([1, 6, 16])
        let prev = MLXArray([Int32(7), 7]).reshaped(1, 2)
        let full = ple(hidden: hidden, ids: ids, prevContext: prev, cache: nil)
        XCTAssertEqual(full.shape, [1, 6, 16])
        // step-wise with a cache must match the single pass
        let cache = ArraysCache(size: 4)
        var outs = [MLXArray]()
        var ctx = prev
        for t in 0 ..< 6 {
            outs.append(
                ple(
                    hidden: hidden[0..., t ..< (t + 1), 0...], ids: ids[0..., t ..< (t + 1)],
                    prevContext: ctx, cache: cache))
            ctx = concatenated([ctx, ids[0..., t ..< (t + 1)]], axis: 1)[0..., 1...]
        }
        XCTAssertTrue(allClose(concatenated(outs, axis: 1), full, rtol: 1e-3, atol: 1e-4).item())
    }
}
