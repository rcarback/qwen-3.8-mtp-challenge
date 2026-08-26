import Foundation
import MLX
import MLXFastCore
import MLXLMCommon
import Testing

@testable import MLXFastModel

@Suite(.serialized)
struct QwenPrefillDiskCacheTests {
    private func scratch() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("prefill-disk-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }

    private func sampleEntry() -> QwenPrefillDiskCache.CacheEntry {
        QwenPrefillDiskCache.CacheEntry(
            tokens: [11, 22, 33, 44],
            layerTags: ["MambaCache", "KVCacheSimple"],
            stateCounts: [1, 2],
            offsets: [4, 4],
            arrays: [
                "L0.S0": MLXArray(converting: [1.0, 2.0, 3.0]).reshaped([1, 3]),
                "L1.S0": MLXArray(converting: [4.0, 5.0]).reshaped([1, 1, 2, 1]),
                "L1.S1": MLXArray(converting: [6.0, 7.0]).reshaped([1, 1, 2, 1]),
            ],
            kvBytes: 4096,
            recurrentBytes: 8192,
            seedTokenCount: 4,
            committedTokenCount: 4)
    }

    @Test("a written entry reads back unchanged")
    func roundTrip() throws {
        let root = scratch()
        defer { try? FileManager.default.removeItem(at: root) }
        let fingerprint = QwenPrefillDiskCache.Fingerprint(
            weightsIdentity: "w1", chunkSize: 4096, kvPolicy: "bf16")
        let entry = sampleEntry()
        try QwenPrefillDiskCache.write(
            entry, key: "abc", fingerprint: fingerprint, root: root)
        let back = try #require(
            try QwenPrefillDiskCache.read(
                key: "abc", fingerprint: fingerprint, root: root))
        #expect(back.tokens == entry.tokens)
        #expect(back.layerTags == entry.layerTags)
        #expect(back.stateCounts == entry.stateCounts)
        #expect(back.offsets == entry.offsets)
        #expect(back.kvBytes == entry.kvBytes)
        #expect(back.recurrentBytes == entry.recurrentBytes)
        #expect(back.seedTokenCount == entry.seedTokenCount)
        #expect(back.committedTokenCount == entry.committedTokenCount)
        #expect(back.arrays.keys.sorted() == entry.arrays.keys.sorted())
        #expect(back.arrays["L1.S1"]!.asArray(Float.self) == [6.0, 7.0])
    }

    @Test("a different fingerprint refuses the restore")
    func fingerprintRefusal() throws {
        let root = scratch()
        defer { try? FileManager.default.removeItem(at: root) }
        let written = QwenPrefillDiskCache.Fingerprint(
            weightsIdentity: "w1", chunkSize: 4096, kvPolicy: "bf16")
        // A restore across a weight change would reinterpret cache rows under
        // the wrong basis and produce plausible numerical garbage rather than
        // an error, so it must return nil rather than the stored entry.
        let other = QwenPrefillDiskCache.Fingerprint(
            weightsIdentity: "w2", chunkSize: 4096, kvPolicy: "bf16")
        try QwenPrefillDiskCache.write(
            sampleEntry(), key: "abc", fingerprint: written, root: root)
        #expect(try QwenPrefillDiskCache.read(
            key: "abc", fingerprint: other, root: root) == nil)
    }

    @Test("an absent key reads as nil rather than throwing")
    func missingKey() throws {
        let root = scratch()
        defer { try? FileManager.default.removeItem(at: root) }
        let fingerprint = QwenPrefillDiskCache.Fingerprint(
            weightsIdentity: "w1", chunkSize: 4096, kvPolicy: "bf16")
        #expect(try QwenPrefillDiskCache.read(
            key: "nope", fingerprint: fingerprint, root: root) == nil)
    }

    @Test("a MambaCache tags as itself, not as its ArraysCache base")
    func subclassOrdering() throws {
        // The pinned model builds 48 MambaCache layers, and
        // `MambaCache: ArraysCache`. A switch that tested the base class first
        // would tag every one of them "ArraysCache" and rebuild them as the
        // wrong class -- silently, because the shapes still line up.
        #expect(try QwenPrefillDiskCache.tag(for: MambaCache()) == "MambaCache")
        #expect(try QwenPrefillDiskCache.tag(for: ChunkedKVCache())
            == "ChunkedKVCache")
        #expect(try QwenPrefillDiskCache.tag(for: KVCacheSimple())
            == "KVCacheSimple")
        let rebuilt = try QwenPrefillDiskCache.makeCache(
            tag: "MambaCache", state: [], offset: 17)
        #expect(rebuilt is MambaCache)
        #expect(rebuilt.offset == 17)
    }

    @Test("a QuantizedKVCache without a policy is refused rather than guessed")
    func quantizedNeedsPolicy() {
        // Defaulting to group 64 / 8 bits would read rows written at another
        // width as plausible numbers rather than failing.
        #expect(throws: MLXFastError.self) {
            _ = try QwenPrefillDiskCache.makeCache(
                tag: "QuantizedKVCache", state: [], offset: 0)
        }
    }

    @Test("an unknown cache class is refused rather than guessed")
    func unknownTag() {
        #expect(throws: MLXFastError.self) {
            _ = try QwenPrefillDiskCache.makeCache(
                tag: "SomeFutureCache", state: [], offset: 0)
        }
    }
}
