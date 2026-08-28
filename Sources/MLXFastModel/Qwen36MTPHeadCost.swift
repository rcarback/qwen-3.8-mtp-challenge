import Foundation

/// Bytes one MTP head draft step reads, derived from geometry rather than
/// measured.
///
/// WHY THIS IS A TYPE. The draft chain is compared against a bandwidth floor in
/// two places that must agree: the phase instrument, which divides measured
/// seconds into these bytes to report an achieved rate, and the depth-price
/// re-fit, which prices a head step against a verify forward. Deriving the same
/// number twice by hand is how the two drift apart.
///
/// WHY THE QUERY PROJECTION IS DOUBLE WIDTH. The Qwen 3.8 attention block packs
/// the per-head output gate beside the queries in one projection and splits the
/// result in two (`Qwen35Attention.callAsFunction`,
/// Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:3350). Counting it
/// once yields 786.4 MB and misses the published artifact size by 63 MB, which
/// is the check the accompanying test performs.
public struct Qwen36MTPHeadCost: Equatable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    /// Bytes per stored element of the head module. The pinned head is
    /// bfloat16, so 2.
    public let headElementBytes: Int

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        kvHeads: Int,
        headDim: Int,
        headElementBytes: Int = 2
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.headElementBytes = headElementBytes
    }

    /// The geometry of the pinned Qwen 3.8 27B head, from `weights/config.json`.
    public static let pinnedQwen38Head = Qwen36MTPHeadCost(
        hiddenSize: 5_120,
        intermediateSize: 17_408,
        attentionHeads: 24,
        kvHeads: 4,
        headDim: 256)

    /// Rows the query projection emits: queries and the packed output gate.
    public var queryProjectionRows: Int { 2 * attentionHeads * headDim }
    /// Rows the output projection consumes.
    public var attentionOutputRows: Int { attentionHeads * headDim }
    /// Rows each of the key and value projections emits.
    public var keyValueProjectionRows: Int { kvHeads * headDim }

    /// `fc` maps the concatenated `[embedding | hidden]` pair back to one hidden
    /// width, so its input width is twice the hidden size.
    public var fcBytes: Int {
        hiddenSize * 2 * hiddenSize * headElementBytes
    }

    public var attentionBytes: Int {
        let projections =
            queryProjectionRows + 2 * keyValueProjectionRows
            + attentionOutputRows
        return projections * hiddenSize * headElementBytes
    }

    /// Gate, up and down.
    public var mlpBytes: Int {
        3 * intermediateSize * hiddenSize * headElementBytes
    }

    /// Five hidden-width RMS norm weights (`pre_fc_norm_hidden`,
    /// `pre_fc_norm_embedding`, `input_layernorm`,
    /// `post_attention_layernorm`, the head's own `norm`) and two head-dim ones
    /// (the query and key norms inside attention).
    public var normBytes: Int {
        5 * hiddenSize * headElementBytes + 2 * headDim * headElementBytes
    }

    public var headModuleBytes: Int {
        fcBytes + attentionBytes + mlpBytes + normBytes
    }

    /// One cached key row plus one cached value row.
    public var headKVBytesPerRow: Int {
        2 * kvHeads * headDim * headElementBytes
    }

    /// Bytes an affine-quantized vocabulary projection reads: packed weight
    /// rows plus one scale and one zero point per group, both bfloat16.
    public static func projectionBytes(
        rows: Int,
        hiddenSize: Int,
        bits: Int,
        groupSize: Int,
        scaleElementBytes: Int = 2
    ) -> Int {
        let weight = rows * hiddenSize * bits / 8
        let groups = rows * (hiddenSize / groupSize)
        return weight + 2 * groups * scaleElementBytes
    }

    /// Total bytes one draft step reads: the head module, the draft vocabulary
    /// projection, and the head key/value history it attends over.
    public func stepBytes(projectionBytes: Int, historyRows: Int) -> Int {
        headModuleBytes + projectionBytes + historyRows * headKVBytesPerRow
    }
}
