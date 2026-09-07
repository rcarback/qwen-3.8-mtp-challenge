import MLX
import MLXLMCommon
import Testing

/// The eval barrier in a decode round exists to keep the cache's lazy graph
/// from growing. It does not need the trimmed VIEW of the cache, only the
/// roots -- and asking for the view builds two slice operations per
/// full-attention layer, every round, that are evaluated and discarded.
@Suite
struct QwenEvalBarrierTests {
    @Test("innerState returns the roots while state returns trimmed slices")
    func innerStateSkipsTheSlice() {
        let cache = KVCacheSimple()
        // One appended row against a step of 256 leaves offset far below the
        // allocated depth, which is the branch every decode round takes.
        let keys = MLXArray.zeros([1, 8, 1, 128], dtype: .float32)
        let values = MLXArray.zeros([1, 8, 1, 128], dtype: .float32)
        _ = cache.update(keys: keys, values: values)
        #expect(cache.offset == 1)
        #expect(cache.state[0].dim(2) == 1, "state is trimmed to the offset")
        #expect(
            cache.innerState()[0].dim(2) == 256,
            "innerState is the untrimmed root")
    }

    @Test("every cache class the session builds overrides innerState")
    func everyCacheClassCarriesRoots() {
        // BaseKVCache.innerState() returns an empty array, so a cache class
        // that forgot the override would make the barrier evaluate nothing at
        // all -- silently, and only under load. Pin both classes the Qwen
        // tower builds (Qwen35.swift newCache: MambaCache for linear layers,
        // KVCacheSimple for the rest).
        let attention = KVCacheSimple()
        _ = attention.update(
            keys: MLXArray.zeros([1, 8, 1, 128], dtype: .float32),
            values: MLXArray.zeros([1, 8, 1, 128], dtype: .float32))
        #expect(attention.innerState().count == 2)

        let recurrent = MambaCache()
        recurrent[0] = MLXArray.zeros([1, 4], dtype: .float32)
        recurrent[1] = MLXArray.zeros([1, 4], dtype: .float32)
        #expect(recurrent.innerState().count == 2)
        #expect(recurrent.innerState().count == recurrent.state.count)
    }
}
