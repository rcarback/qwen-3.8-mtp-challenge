import Foundation
import MLX
import MLXFastCore
import MLXLMCommon

/// On-disk form of a prefill checkpoint.
///
/// WHY A FINGERPRINT GUARDS EVERY READ. A checkpoint is raw cache rows. Read
/// back under different weights, a different cache layout, or a different KV
/// quantization basis, those rows do not fail -- they decode into plausible
/// numbers and produce silently wrong output. The fingerprint is compared
/// before any array is touched, and a mismatch is a miss, never a fallback.
public enum QwenPrefillDiskCache {
    /// Bump when the on-disk layout changes in a way older files cannot satisfy.
    static let formatVersion = 1

    public struct Fingerprint: Equatable {
        let weightsIdentity: String
        let chunkSize: Int
        let kvPolicy: String

        public init(weightsIdentity: String, chunkSize: Int, kvPolicy: String) {
            self.weightsIdentity = weightsIdentity
            self.chunkSize = chunkSize
            self.kvPolicy = kvPolicy
        }

        /// Identity string stored in the file and compared on read.
        public var token: String {
            "v\(QwenPrefillDiskCache.formatVersion)"
                + "|w:\(weightsIdentity)|c:\(chunkSize)|q:\(kvPolicy)"
        }
    }

    public struct CacheEntry {
        public let tokens: [Int]
        public let layerTags: [String]
        public let stateCounts: [Int]
        /// Per-layer `offset`. Carried explicitly because `ArraysCache`'s
        /// `state` setter does not recompute it the way `KVCacheSimple`'s
        /// does, and a gated-delta layer restored without it reads as empty.
        public let offsets: [Int]
        public let arrays: [String: MLXArray]
        public let kvBytes: Int
        public let recurrentBytes: Int
        public let seedTokenCount: Int
        public let committedTokenCount: Int

        public init(
            tokens: [Int], layerTags: [String], stateCounts: [Int],
            offsets: [Int], arrays: [String: MLXArray], kvBytes: Int,
            recurrentBytes: Int, seedTokenCount: Int, committedTokenCount: Int
        ) {
            self.tokens = tokens
            self.layerTags = layerTags
            self.stateCounts = stateCounts
            self.offsets = offsets
            self.arrays = arrays
            self.kvBytes = kvBytes
            self.recurrentBytes = recurrentBytes
            self.seedTokenCount = seedTokenCount
            self.committedTokenCount = committedTokenCount
        }
    }

    /// File for `key`. The key is already a base-36 FNV-1a digest, so it is
    /// filename-safe, but it is filtered anyway: a key is data, and data that
    /// reaches a path must not be able to name a parent directory.
    static func url(key: String, root: URL) -> URL {
        let safe = key.filter { $0.isLetter || $0.isNumber }
        return root.appendingPathComponent("\(safe).safetensors")
    }

    /// Persistence tag for a cache class.
    ///
    /// Fail closed on an unrecognized class. Restoring rows into the wrong
    /// cache class reinterprets them silently, which is the one failure mode
    /// this whole file exists to prevent.
    /// ORDER MATTERS: most-derived first. `MambaCache: ArraysCache` and
    /// `ChunkedKVCache: KVCacheSimple`, so an `is ArraysCache` case placed
    /// ahead of `is MambaCache` would match every gated-delta layer and
    /// rebuild it as the wrong class. The pinned model builds 48 MambaCache
    /// and 16 KVCacheSimple (see `Qwen35Cache.swift:18`).
    public static func tag(for cache: any KVCache) throws -> String {
        switch cache {
        case is MambaCache: return "MambaCache"
        case is ChunkedKVCache: return "ChunkedKVCache"
        case is ArraysCache: return "ArraysCache"
        case is QuantizedKVCache: return "QuantizedKVCache"
        case is RotatingKVCache: return "RotatingKVCache"
        case is KVCacheSimple: return "KVCacheSimple"
        default:
            throw MLXFastError.invalidInput(
                "prefill checkpoint cannot persist cache class "
                    + "\(type(of: cache))")
        }
    }

    /// Rebuild a cache of `tag` holding `state` at `offset`.
    ///
    /// `offset` is passed rather than derived. `KVCacheSimple`'s state setter
    /// recomputes it from the key tensor, but `ArraysCache`'s does not, so a
    /// gated-delta layer restored without it believes it is empty.
    ///
    /// `quantization` supplies the group size and bit width for a
    /// `QuantizedKVCache`. Reading rows written at one bit width as another
    /// yields plausible numbers rather than an error, so the defaults are not
    /// safe here; the caller reads the live policy, and the fingerprint has
    /// already refused any checkpoint written under a different one.
    public static func makeCache(
        tag: String, state: [MLXArray], offset: Int,
        quantization: (groupSize: Int, bits: Int)? = nil
    ) throws -> any KVCache {
        let cache: BaseKVCache
        switch tag {
        case "MambaCache": cache = MambaCache()
        case "ChunkedKVCache": cache = ChunkedKVCache()
        case "ArraysCache": cache = ArraysCache(size: state.count)
        case "QuantizedKVCache":
            guard let quantization else {
                throw MLXFastError.invalidInput(
                    "prefill checkpoint holds a QuantizedKVCache but no KV "
                        + "quantization policy was supplied")
            }
            cache = QuantizedKVCache(
                groupSize: quantization.groupSize, bits: quantization.bits)
        case "RotatingKVCache": cache = RotatingKVCache(maxSize: 512)
        case "KVCacheSimple": cache = KVCacheSimple()
        default:
            throw MLXFastError.invalidInput(
                "prefill checkpoint holds unknown cache class \(tag)")
        }
        if !state.isEmpty {
            cache.state = state
        }
        cache.offset = offset
        return cache
    }

    public static func write(
        _ entry: CacheEntry, key: String, fingerprint: Fingerprint, root: URL
    ) throws {
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        var arrays = entry.arrays
        arrays["tokens"] = MLXArray(entry.tokens.map { Int32($0) })
        let metadata: [String: String] = [
            "fingerprint": fingerprint.token,
            "layerTags": entry.layerTags.joined(separator: ","),
            "stateCounts": entry.stateCounts.map(String.init)
                .joined(separator: ","),
            "offsets": entry.offsets.map(String.init).joined(separator: ","),
            "kvBytes": String(entry.kvBytes),
            "recurrentBytes": String(entry.recurrentBytes),
            "seedTokenCount": String(entry.seedTokenCount),
            "committedTokenCount": String(entry.committedTokenCount),
        ]
        // Write beside the target and move into place. A reader that opens a
        // half-written checkpoint would see a valid fingerprint over truncated
        // rows, which is exactly the silent corruption this guards against.
        let target = url(key: key, root: root)
        let staging = target.deletingPathExtension()
            .appendingPathExtension("partial.safetensors")
        try MLX.save(arrays: arrays, metadata: metadata, url: staging)
        _ = try? FileManager.default.removeItem(at: target)
        try FileManager.default.moveItem(at: staging, to: target)
    }

    public static func read(
        key: String, fingerprint: Fingerprint, root: URL
    ) throws -> CacheEntry? {
        let target = url(key: key, root: root)
        guard FileManager.default.fileExists(atPath: target.path) else {
            return nil
        }
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: target)
        guard metadata["fingerprint"] == fingerprint.token else { return nil }
        guard let tokensArray = arrays["tokens"],
              let tags = metadata["layerTags"],
              let counts = metadata["stateCounts"],
              let offsets = metadata["offsets"],
              let kvBytes = metadata["kvBytes"].flatMap(Int.init),
              let recurrentBytes = metadata["recurrentBytes"].flatMap(Int.init),
              let seed = metadata["seedTokenCount"].flatMap(Int.init),
              let committed = metadata["committedTokenCount"].flatMap(Int.init)
        else { return nil }
        var payload = arrays
        payload.removeValue(forKey: "tokens")
        return CacheEntry(
            tokens: tokensArray.asArray(Int32.self).map(Int.init),
            layerTags: tags.isEmpty ? [] : tags.components(separatedBy: ","),
            stateCounts: counts.isEmpty
                ? []
                : counts.components(separatedBy: ",").compactMap(Int.init),
            offsets: offsets.isEmpty
                ? []
                : offsets.components(separatedBy: ",").compactMap(Int.init),
            arrays: payload,
            kvBytes: kvBytes,
            recurrentBytes: recurrentBytes,
            seedTokenCount: seed,
            committedTokenCount: committed)
    }
}
