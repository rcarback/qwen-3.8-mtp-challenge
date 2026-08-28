import Testing

@testable import MLXFastModel

/// The byte model is arithmetic over the pinned geometry, so it is checkable
/// against a number nobody in this repository chose: the published size of the
/// MTP head artifact. `EigenLabs/Qwen3.8-27B-MTP-bf16` at revision
/// `26a328e070875b0314d652a039b6b59902690f03` carries 849,398,784 tensor bytes
/// across 15 bfloat16 tensors.
/// A model that reproduces that total from hidden size, intermediate size and
/// the head counts has the right tensor inventory, including the detail that
/// the query projection carries the per-head output gate packed beside the
/// queries.
@Suite
struct Qwen36MTPHeadCostTests {
    @Test("head module bytes reproduce the published artifact size")
    func headModuleBytesMatchPublishedArtifact() {
        #expect(Qwen36MTPHeadCost.pinnedQwen38Head.headModuleBytes == 849_398_784)
    }

    @Test("head module byte blocks split as measured")
    func headModuleByteBlocks() {
        let cost = Qwen36MTPHeadCost.pinnedQwen38Head
        #expect(cost.fcBytes == 104_857_600)
        #expect(cost.attentionBytes == 209_715_200)
        #expect(cost.mlpBytes == 534_773_760)
        #expect(cost.normBytes == 52_224)
    }

    @Test("compact draft projection is 283,207,680 bytes")
    func compactProjectionBytes() {
        #expect(
            Qwen36MTPHeadCost.projectionBytes(
                rows: 98_336, hiddenSize: 5_120, bits: 4, groupSize: 64)
                == 283_207_680)
    }

    @Test("exact lm_head projection is 715,161,600 bytes")
    func exactProjectionBytes() {
        #expect(
            Qwen36MTPHeadCost.projectionBytes(
                rows: 248_320, hiddenSize: 5_120, bits: 4, groupSize: 64)
                == 715_161_600)
    }

    @Test("head key/value history costs 4,096 bytes per row")
    func headKVBytesPerRow() {
        #expect(Qwen36MTPHeadCost.pinnedQwen38Head.headKVBytesPerRow == 4_096)
    }

    @Test("one draft step at 2,048 history rows reads 1,140,995,072 bytes")
    func stepBytesAtBenchmarkDepth() {
        let cost = Qwen36MTPHeadCost.pinnedQwen38Head
        let projection = Qwen36MTPHeadCost.projectionBytes(
            rows: 98_336, hiddenSize: 5_120, bits: 4, groupSize: 64)
        #expect(
            cost.stepBytes(projectionBytes: projection, historyRows: 2_048)
                == 1_140_995_072)
    }
}
