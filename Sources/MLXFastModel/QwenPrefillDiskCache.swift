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
        case "ChunkedKVCache":
            // `chunkSize` and `startPosition` are construction parameters,
            // not restorable state -- guessing them here is exactly the
            // silent mis-restore this file exists to prevent. Unreachable
            // for the pinned model (it never builds one), so refuse rather
            // than guess.
            throw MLXFastError.invalidInput(
                "prefill checkpoint holds a ChunkedKVCache, which cannot be "
                    + "rebuilt without its original chunkSize/startPosition")
        case "ArraysCache": cache = ArraysCache(size: state.count)
        case "QuantizedKVCache":
            guard let quantization else {
                throw MLXFastError.invalidInput(
                    "prefill checkpoint holds a QuantizedKVCache but no KV "
                        + "quantization policy was supplied")
            }
            cache = QuantizedKVCache(
                groupSize: quantization.groupSize, bits: quantization.bits)
        case "RotatingKVCache":
            // `keep` and `step` are construction parameters, not restorable
            // state -- same reasoning as `ChunkedKVCache` above. Unreachable
            // for the pinned model.
            throw MLXFastError.invalidInput(
                "prefill checkpoint holds a RotatingKVCache, which cannot be "
                    + "rebuilt without its original keep/step")
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

    /// Rebuild every layer's cache from a restored `CacheEntry`.
    ///
    /// Pulled out of the worker's request dispatch so the length-validation
    /// and failure paths are unit-testable without a live session: a missing
    /// array, an unknown tag, or a layer-count disagreement throws here
    /// rather than trapping or silently mis-restoring, and the caller decides
    /// what to do about it (fall back to a full prefill).
    public static func restoreCaches(
        from entry: CacheEntry, quantization: (groupSize: Int, bits: Int)?
    ) throws -> [any KVCache] {
        var caches: [any KVCache] = []
        for (layer, tag) in entry.layerTags.enumerated() {
            var state: [MLXArray] = []
            for slot in 0 ..< entry.stateCounts[layer] {
                guard let array = entry.arrays["L\(layer).S\(slot)"] else {
                    throw MLXFastError.invalidInput(
                        "prefill checkpoint is missing array "
                            + "L\(layer).S\(slot)")
                }
                state.append(array)
            }
            caches.append(
                try makeCache(
                    tag: tag, state: state, offset: entry.offsets[layer],
                    quantization: quantization))
        }
        return caches
    }

    public static func write(
        _ entry: CacheEntry, key: String, fingerprint: Fingerprint, root: URL,
        budgetBytes: Int = QwenPrefillDiskBudget.clampedDefault()
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
        // Write beside the target and swap into place. A reader that opens a
        // half-written checkpoint would see a valid fingerprint over truncated
        // rows, which is exactly the silent corruption this guards against.
        //
        // The staging name carries a UUID: two processes sharing one cache
        // root derive the SAME key for the same tokens, and a name derived
        // only from the key would let them collide on the same staging file
        // mid-write. `replaceItemAt` (rather than `removeItem` + `moveItem`)
        // makes the swap atomic, so a reader never observes a moment with
        // neither file present.
        let target = url(key: key, root: root)
        let staging = root.appendingPathComponent(
            "\(key).\(UUID().uuidString).partial.safetensors")
        do {
            try MLX.save(arrays: arrays, metadata: metadata, url: staging)
            _ = try FileManager.default.replaceItemAt(target, withItemAt: staging)
        } catch {
            // `MLX.save` throwing mid-write (disk full) or a failed swap must
            // not leave the staging file behind forever.
            _ = try? FileManager.default.removeItem(at: staging)
            throw error
        }
        evictToBudget(root: root, budgetBytes: budgetBytes)
    }

    /// Evict least-recently-modified checkpoints until `root` is back under
    /// `budgetBytes`.
    ///
    /// Best-effort: a directory listing or delete failure just leaves the
    /// root over budget until the next successful write retries it, rather
    /// than turning a housekeeping sweep into a request failure.
    static func evictToBudget(root: URL, budgetBytes: Int) {
        let manager = FileManager.default
        guard let entries = try? manager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey])
        else { return }
        var files = entries.compactMap {
            fileURL -> (url: URL, size: Int, modified: Date)? in
            guard fileURL.pathExtension == "safetensors",
                  !fileURL.lastPathComponent.contains(".partial.")
            else { return nil }
            let values = try? fileURL.resourceValues(
                forKeys: [.contentModificationDateKey, .fileSizeKey])
            return (
                fileURL, values?.fileSize ?? 0,
                values?.contentModificationDate ?? .distantPast)
        }
        var total = files.reduce(0) { $0 + $1.size }
        guard total > budgetBytes else { return }
        files.sort { $0.modified < $1.modified }
        for file in files where total > budgetBytes {
            if (try? manager.removeItem(at: file.url)) != nil {
                total -= file.size
            }
        }
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
        let layerTags = tags.isEmpty ? [] : tags.components(separatedBy: ",")
        let stateCounts = counts.isEmpty
            ? []
            : counts.components(separatedBy: ",").compactMap(Int.init)
        let layerOffsets = offsets.isEmpty
            ? []
            : offsets.components(separatedBy: ",").compactMap(Int.init)
        // `stateCounts` and `offsets` are parsed with `compactMap(Int.init)`,
        // which silently drops an unparseable entry -- so a corrupt metadata
        // string can produce arrays SHORTER than `layerTags`. The worker
        // indexes both by the `layerTags` enumeration; a short array there is
        // an out-of-bounds trap, not a catchable error. Treat any length
        // disagreement as a miss, the same way a fingerprint mismatch is.
        guard stateCounts.count == layerTags.count,
              layerOffsets.count == layerTags.count
        else { return nil }
        var payload = arrays
        payload.removeValue(forKey: "tokens")
        return CacheEntry(
            tokens: tokensArray.asArray(Int32.self).map(Int.init),
            layerTags: layerTags,
            stateCounts: stateCounts,
            offsets: layerOffsets,
            arrays: payload,
            kvBytes: kvBytes,
            recurrentBytes: recurrentBytes,
            seedTokenCount: seed,
            committedTokenCount: committed)
    }
}

/// Disk budget for persisted prefill checkpoints, and its override.
///
/// WHY A BUDGET AT ALL. `QwenSessionCacheStore`'s in-memory pool is clamped
/// to a quarter of physical RAM; its disk twin has no natural backstop --
/// every distinct chunk boundary the server ever prefills writes another
/// checkpoint, forever, at full KV-plus-recurrent size. Left unbounded that
/// fills the volume out from under whatever else uses it.
///
/// 32 GiB DEFAULT. A checkpoint is NOT the flat 144 MiB the in-memory store
/// budgets against. That figure counts only the gated-delta recurrent state,
/// which is constant in the prompt; on disk the attention KV rides along as
/// real bytes rather than copy-on-write, and it grows with the context. A
/// measured checkpoint at 5972 tokens is 520 MiB -- 144 MiB recurrent plus
/// 373 MiB of KV -- so the KV term dominates everything past a few thousand
/// tokens and a 50k-token checkpoint approaches 3 GiB.
///
/// So 32 GiB is on the order of 60 checkpoints of a mid-length session, or
/// roughly 10 of a long one, not the 200 a flat-144-MiB reading suggests. It
/// is still an actual bound rather than "however much disk happens to be
/// free," which is the property that matters, but size it against the context
/// lengths a given server actually sees.
///
/// `DARKBLOOM_PREFILL_CACHE_MAX_BYTES` overrides it. It is the ONLY prefix
/// that survives `sanitizedRuntimeWorkerEnvironment`'s allowlist -- an
/// `MLXFAST_`-prefixed name would be stripped at spawn and look applied
/// while silently doing nothing.
public enum QwenPrefillDiskBudget {
    public static let defaultBytes = 32 * 1024 * 1024 * 1024

    public static func clampedDefault(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        guard let raw = environment["DARKBLOOM_PREFILL_CACHE_MAX_BYTES"],
              let value = Int(raw), value >= 0
        else { return defaultBytes }
        return value
    }
}
