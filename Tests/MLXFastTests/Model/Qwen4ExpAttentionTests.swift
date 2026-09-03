import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen4ExpAttentionTests: XCTestCase {
    func tinyArgs(budget: Int) -> Qwen4ExpTextConfiguration {
        var a = Qwen4ExpTextConfiguration()
        a.hiddenSize = 16
        a.attentionHeads = 2
        a.kvHeads = 1
        a.headDim = 8
        a.partialRotaryFactor = 0.5
        a.indexerNHeads = 2
        a.indexerKVHeads = 1
        a.indexerHeadDim = 4
        a.indexerCompressRatio = 2
        a.indexerBudget = budget
        a.ropeTheta = 10_000
        return a
    }

    func testBelowBudgetIsDenseAndStepwiseMatchesPrefill() {
        let args = tinyArgs(budget: 64)
        let attn = Qwen4ExpAttention(args)
        let rope = Qwen4ExpRotary(dims: args.rotaryDims, base: args.ropeTheta)
        let x = MLXRandom.normal([1, 7, 16])
        let full = attn(x, rope: rope, mask: .causal, cache: nil)
        let cache = Qwen4ExpAttnCache()
        var outs = [MLXArray]()
        for t in 0 ..< 7 {
            outs.append(attn(x[0..., t ..< (t + 1), 0...], rope: rope, mask: .none, cache: cache))
        }
        XCTAssertEqual(cache.offset, 7)
        XCTAssertTrue(allClose(concatenated(outs, axis: 1), full, rtol: 1e-2, atol: 1e-3).item())
        XCTAssertEqual(cache.state.count, 3)  // keys, values, indexer keys
        XCTAssertEqual(cache.state[2].shape, [1, 7, 4])
    }

    func testSparseKeepMaskKeepsTopBlocksAndOwnTail() {
        // budget 4, compress 2 -> block_topk 2. With 9 cached tokens: 4 complete blocks, 1 tail token.
        let args = tinyArgs(budget: 4)
        let idx = Qwen4ExpQSAIndexer(args)
        let rope = Qwen4ExpRotary(dims: args.rotaryDims, base: args.ropeTheta)
        let cache = Qwen4ExpAttnCache()
        _ = idx.keepMask(MLXRandom.normal([1, 8, 16]), rope: rope, cache: cache, offset: 0)
        cache.offset = 8  // stand in for the K/V update
        let keep = idx.keepMask(MLXRandom.normal([1, 1, 16]), rope: rope, cache: cache, offset: 8)!
        XCTAssertEqual(keep.shape, [1, 1, 1, 9])
        let row = keep[0, 0, 0].asArray(Bool.self)
        // own tail: the query at position 8 sits in the partial block [8]
        XCTAssertTrue(row[8])
        let keptBlocks = stride(from: 0, to: 8, by: 2).filter { row[$0] }.count
        XCTAssertEqual(keptBlocks, 2)  // exactly block_topk complete blocks
        for b in stride(from: 0, to: 8, by: 2) {
            XCTAssertEqual(row[b], row[b + 1])  // whole blocks
        }
    }

    func testTrimDropsIndexerRows() {
        let args = tinyArgs(budget: 64)
        let attn = Qwen4ExpAttention(args)
        let rope = Qwen4ExpRotary(dims: args.rotaryDims, base: args.ropeTheta)
        let cache = Qwen4ExpAttnCache()
        _ = attn(MLXRandom.normal([1, 5, 16]), rope: rope, mask: .causal, cache: cache)
        XCTAssertEqual(cache.trim(2), 2)
        XCTAssertEqual(cache.offset, 3)
        XCTAssertEqual(cache.state[2].shape, [1, 3, 4])
        _ = attn(MLXRandom.normal([1, 1, 16]), rope: rope, mask: .none, cache: cache)
        XCTAssertEqual(cache.state[2].shape, [1, 4, 4])
    }
}
