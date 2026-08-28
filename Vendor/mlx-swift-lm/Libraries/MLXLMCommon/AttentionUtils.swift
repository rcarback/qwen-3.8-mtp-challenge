import Foundation
import MLX

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Diagnostic counters for the quantized attention dispatch.
///
/// Set `DARKBLOOM_KV_DISPATCH_STATS` to a file path to count how often the
/// fused kernel is actually reached and why it is skipped when it is not. The
/// counts are written when the process exits. `serve` does not forward worker
/// stderr, so a file is the only way these become visible. Off by default, and
/// when off the whole block costs one already-loaded Bool test.
public enum QuantizedDispatchStats {
    nonisolated(unsafe) private static var quantizedCalls = 0
    nonisolated(unsafe) private static var fusedTaken = 0
    nonisolated(unsafe) private static var skippedNotCausal = 0
    nonisolated(unsafe) private static var skippedUnsupported = 0
    nonisolated(unsafe) private static var rowHistogram: [Int: Int] = [:]
    private static let lock = NSLock()

    public static let path: String? =
        ProcessInfo.processInfo.environment["DARKBLOOM_KV_DISPATCH_STATS"]
    public static let enabled: Bool = path != nil

    public static func record(queryRows: Int, causal: Bool, supported: Bool) {
        lock.lock()
        defer { lock.unlock() }
        quantizedCalls += 1
        rowHistogram[queryRows, default: 0] += 1
        if !causal {
            skippedNotCausal += 1
        } else if !supported {
            skippedUnsupported += 1
        } else {
            fusedTaken += 1
        }
    }

    /// Registered once, on first use, so the counts survive process exit.
    public static func installWriterIfNeeded() {
        guard enabled else { return }
        lock.lock()
        defer { lock.unlock() }
        if installed { return }
        installed = true
        atexit {
            QuantizedDispatchStats.write()
        }
    }

    nonisolated(unsafe) private static var installed = false

    private static func write() {
        guard let path else { return }
        lock.lock()
        let rows = rowHistogram
            .sorted { $0.key < $1.key }
            .map { "\"\($0.key)\": \($0.value)" }
            .joined(separator: ", ")
        let json = """
            {"quantized_calls": \(quantizedCalls), "fused_taken": \(fusedTaken), \
            "skipped_not_causal": \(skippedNotCausal), \
            "skipped_unsupported": \(skippedUnsupported), \
            "query_rows": {\(rows)}}
            """
        lock.unlock()
        try? json.write(toFile: path, atomically: true, encoding: .utf8)
    }
}

/// Whether the fused quantized decode kernel is enabled.
///
/// Reads `DARKBLOOM_KV_FUSED_SDPA` once. Set it to `0` to force the
/// decomposed path. This only ever applies to a cache that is already
/// quantized, which is itself opt-in, so a default run never reaches it.
public let fusedQuantizedSDPADefault: Bool =
    ProcessInfo.processInfo.environment["DARKBLOOM_KV_FUSED_SDPA"] != "0"

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
///
/// LIMITATION (ContinuousBatchingV2 caches): when `cache` is a
/// `CBv2AttendingLayerCache`, the layer cache owns BOTH the KV update and
/// the attention computation, INCLUDING masking — the `mask` parameter is
/// DISCARDED on that path (v2 derives causal/window masks from per-row
/// absolute positions), and `sinks` are passed as nil. A non-adapted model
/// driven with CBv2 caches therefore silently loses any CUSTOM array mask
/// (e.g. bidirectional/prefix-LM or padding masks) and any attention
/// sinks; sinks-bearing or custom-mask models must call `updateAndAttend`
/// directly instead. Array masks fail HARD here — in release builds too
/// (`preconditionFailure`, not a debug-only assertion): silently swapping
/// the model's required mask for v2's causal/window mask would corrupt
/// output in exactly the builds users run (PR#62 review).
///
/// MULTI-ROW LIMITATION: this compatibility path is B == 1 ONLY. Legacy
/// (non-v2-adapted) models apply scalar RoPE via `KVCache.offset` BEFORE
/// calling this helper; a CBv2 cache's legacy `offset` is the MAX row
/// offset, so at B > 1 every shorter row would be silently mis-rotated.
/// B == 1 stays allowed (the scalar offset is exact for a single row).
/// Generic models must be v2-adapted — capture `positionOffsets` before
/// dispatch and call `updateAndAttend` directly — before they can serve
/// multi-row CBv2 batches. This fails loudly rather than mis-rotating.
/// Shipped upper bound of the wide-decode exactness chunk: the widest verify
/// block the shipped draft-depth ceiling of 8 can ask for is 9 rows.
public let wideDecodeExactnessDefaultMaxQueryRows = 9

/// Largest bound the override may select: the widest verify block the runtime
/// has a QMV width plan for is 16 rows.
public let wideDecodeExactnessHardCapQueryRows = 16

/// Pure, total parser for the wide-decode chunk bound.
///
/// MIRROR, DELIBERATELY. `MLXFastConstants.parseMaxDraftDepth` owns the
/// trusted ceiling `d`; this owns the query-row bound `d + 1` that the chunk
/// above must cover. The two live in packages that cannot import each other --
/// `MLXLMCommon` is a vendored dependency of the harness, not a client of it
/// -- so the arithmetic is restated here and pinned by a test that runs both
/// parsers over the same inputs and asserts `rows == depth + 1`.
///
/// Both read the same `MLX_`-prefixed name, which
/// `sanitizedRuntimeWorkerEnvironment` forwards to the runtime worker, so the
/// parent process and the worker process resolve the same number.
public func parseWideDecodeExactnessMaxQueryRows(_ raw: String?) -> Int {
    guard let raw else { return wideDecodeExactnessDefaultMaxQueryRows }
    let trimmed = raw.trimmingCharacters(in: .whitespaces)
    guard let depth = Int(trimmed), depth >= 1,
          depth + 1 <= wideDecodeExactnessHardCapQueryRows
    else { return wideDecodeExactnessDefaultMaxQueryRows }
    return depth + 1
}

/// Read once at process start. Never varies with the request or the prompt.
public let wideDecodeExactnessMaxQueryRows = parseWideDecodeExactnessMaxQueryRows(
    ProcessInfo.processInfo.environment["MLX_QWEN_MTP_MAX_DRAFT_DEPTH"])

public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    fusedQuantizedEnabled: Bool = fusedQuantizedSDPADefault
) -> MLXArray {
    // ContinuousBatchingV2 hook — see the LIMITATION notes above.
    if let v2 = cache as? CBv2AttendingLayerCache {
        if let violation = cbv2CustomMaskViolation(mask: mask, layerIndex: v2.layerIndex) {
            preconditionFailure(violation)
        }
        if let violation = cbv2LegacyAttentionBatchViolation(
            batch: queries.dim(0), layerIndex: v2.layerIndex)
        {
            preconditionFailure(violation)
        }
        return v2.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
    }
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        // FUSED DECODE PATH. The decomposed call below materializes a
        // [B, heads, L, N] score matrix, which is nine to twelve times slower
        // than the fused bfloat16 kernel at decode shapes and gets worse as
        // context grows. The fused kernel computes the same thing without
        // writing scores to memory. Anything it does not support -- prefill
        // widths, unsupported bit widths, sinks, a non-causal array mask --
        // falls through unchanged.
        var causal = false
        if case .causal = mask { causal = true }
        let supportedShape = FusedQuantizedSDPA.isSupported(
            headDim: queries.dim(3),
            valueHeadDim: values.dim(3),
            queryRows: queries.dim(2),
            bits: quantizedKVCache.bits,
            groupSize: quantizedKVCache.groupSize,
            mode: quantizedKVCache.mode,
            hasSinks: false,
            hasBiases: quantizedKeys.2 != nil && quantizedValues.2 != nil)
        if QuantizedDispatchStats.enabled {
            QuantizedDispatchStats.installWriterIfNeeded()
            QuantizedDispatchStats.record(
                queryRows: queries.dim(2), causal: causal, supported: supportedShape)
        }
        if fusedQuantizedEnabled, causal, supportedShape
        {
            return FusedQuantizedSDPA.attention(
                queries: queries,
                quantizedKeys: quantizedKeys,
                quantizedValues: quantizedValues,
                scale: scale, causal: true,
                groupSize: quantizedKVCache.groupSize,
                bits: quantizedKVCache.bits)
        }
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        // WIDE-DECODE EXACTNESS CHUNK (B == 1, causal,
        // 6 <= qL <= wideDecodeExactnessMaxQueryRows): the fused sdpa vector
        // path serves qL * gqa <= 32; above it the dispatch changes kernel
        // family and the accumulation order of every score — the measured
        // source of the MTP width wall's top-2 VALUE drift.
        // Splitting the queries every 5 rows keeps every segment on the fused
        // vector path with windows that are BYTE-IDENTICAL to consecutive
        // <= 5-row rounds at the same offsets: with bottom-right causal
        // alignment, the segment [start, end) taken over
        // keys[..<kL - (qL - end)] gives absolute row i the window
        // kL - qL + 1 + i that a serial round at that position would see, for
        // every segmentation. Keys/values are re-sliced, not recomputed — the
        // only extra cost is one more pass over the KV rows (a few MB), never
        // over weights. Serial (qL == 1), the <= 5 verify widths, and prefill
        // (qL > wideDecodeExactnessMaxQueryRows) are untouched.
        // The cache update above happens exactly once; every segment below is
        // a read-only view of that single committed candidate window.
        //
        // THE UPPER GUARD IS A DEPTH-CEILING CONSUMER, not a free constant.
        // A verify round of depth d is qL = d + 1 rows, so a guard below the
        // trusted ceiling + 1 would send the widest legal rounds down the
        // unchunked path and reintroduce exactly the drift this chunk exists
        // to remove — silently. `wideDecodeExactnessMaxQueryRows` therefore
        // tracks the same environment override the ceiling does.
        let qL = queries.dim(2)
        let kL = cachedKeys.dim(2)
        if queries.dim(0) == 1, qL >= 6, qL <= wideDecodeExactnessMaxQueryRows,
           kL >= qL, case .causal = mask
        {
            let split = 5
            var segments: [MLXArray] = []
            var start = 0
            while start < qL {
                let end = min(start + split, qL)
                let kEnd = kL - (qL - end)
                // The final segment spans the whole cached window; pass the
                // arrays themselves rather than a full-range slice so the
                // two-segment case stays the graph the shipped code built.
                let k = kEnd == kL ? cachedKeys : cachedKeys[0..., 0..., 0 ..< kEnd, 0...]
                let v = kEnd == kL ? cachedValues : cachedValues[0..., 0..., 0 ..< kEnd, 0...]
                segments.append(
                    MLXFast.scaledDotProductAttention(
                        queries: queries[0..., 0..., start ..< end, 0...],
                        keys: k,
                        values: v,
                        scale: scale,
                        mask: .causal
                    ))
                start = end
            }
            return concatenated(segments, axis: 2)
        }
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
    }
}

/// Custom-mask guard for the CBv2 branch of `attentionWithCacheUpdate` (see
/// the LIMITATION doc there). `.none`/`.causal` are subsumed by v2's own
/// position-derived masks; any other mode (custom `.array`/`.arrays`) would
/// be silently DISCARDED by `updateAndAttend`, so it must fail in ALL build
/// configurations — a debug-only `assertionFailure` compiles out of release
/// builds and lets the wrong mask ship (PR#62 review). Returns the failure
/// description for an illegal call, or nil when the call is allowed.
/// Internal (not private) so tests can pin the exact condition without
/// tripping the precondition.
func cbv2CustomMaskViolation(
    mask: MLXFast.ScaledDotProductAttentionMaskMode, layerIndex: Int
) -> String? {
    switch mask {
    case .none, .causal:
        return nil
    default:
        return """
            attentionWithCacheUpdate: a custom array mask was passed with a \
            CBv2 layer cache (layer \(layerIndex)). CBv2 caches own their \
            masks and DISCARD this parameter — the model must be v2-adapted \
            (call updateAndAttend with its own semantics) instead.
            """
    }
}

/// Multi-row guard for the CBv2 branch of `attentionWithCacheUpdate` (see
/// the MULTI-ROW LIMITATION doc there). Returns the failure description for
/// an illegal call, or nil when the call is allowed. Internal (not private)
/// so tests can pin the exact condition without tripping the precondition.
func cbv2LegacyAttentionBatchViolation(batch: Int, layerIndex: Int) -> String? {
    guard batch > 1 else { return nil }
    return """
        attentionWithCacheUpdate: a multi-row batch (B=\(batch)) reached the \
        legacy CBv2 compatibility path (layer \(layerIndex)). Legacy models \
        apply scalar RoPE via `KVCache.offset` — the MAX row offset — so \
        shorter rows would be silently mis-rotated at B > 1. This model must \
        be v2-adapted (read `positionOffsets` before dispatch and call \
        `updateAndAttend` directly) before multi-row CBv2 serving; B == 1 \
        remains supported.
        """
}
