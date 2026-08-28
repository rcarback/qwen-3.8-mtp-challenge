import Foundation
import Testing

@testable import MLXFastCore
@testable import MLXLLM
@testable import MLXLMCommon

/// Contract tests for the two runtime bounds that were literals before:
/// the custom-QMV upper width and the trusted draft-depth ceiling.
///
/// Both are read from the environment once at process start, so the tests
/// here exercise the PURE parsers and the tables derived from them. Nothing
/// in this file loads a model or touches a GPU stream.
@Suite
struct QwenDepthAndWidthBoundTests {

    // MARK: - custom QMV width plan

    @Test("the shipped width bound is 9 and every malformed override lands there")
    func widthParserDefaults() {
        #expect(Qwen35CustomQMV.maxWidthDefault == 9)
        #expect(Qwen35CustomQMV.maxWidthHardCap == 16)
        for raw in [nil, "", " ", "nine", "1", "0", "-4", "17", "9.5", "12x"] {
            #expect(Qwen35CustomQMV.parseMaxWidth(raw) == 9,
                    "\(String(describing: raw)) should fall back to the default")
        }
    }

    @Test("a well-formed width override is taken verbatim")
    func widthParserAccepts() {
        for value in 2 ... 16 {
            #expect(Qwen35CustomQMV.parseMaxWidth("\(value)") == value)
        }
        #expect(Qwen35CustomQMV.parseMaxWidth(" 12 ") == 12)
    }

    /// The kernel's `static_assert(M % IPG != 1)`: a one-row tail group
    /// instantiates `qwen_e120_qmv_wide<2>` and would read a row past `M`.
    @Test("every planned width has a legal inputs-per-group")
    func inputGroupPlanIsLegal() {
        for (m, ipg) in Qwen35CustomQMV.inputGroupPlan {
            #expect(ipg >= 2, "width \(m): a group of one input is not built")
            #expect(ipg <= m, "width \(m): group wider than the block")
            #expect(m % ipg != 1, "width \(m): a one-input tail group is illegal")
            // NA sizes acc[4], partial[4], a0..a3 and sums, ~13 * NA floats
            // per thread. 5 is the footprint the shipped m = 5 entry pays.
            #expect(ipg <= 5, "width \(m): untested register footprint")
        }
    }

    @Test("the plan covers 2 through 16 exactly once, in order")
    func inputGroupPlanIsComplete() {
        #expect(Qwen35CustomQMV.inputGroupPlan.map(\.0) == Array(2 ... 16))
    }

    @Test("widths 2 through 9 keep the shipped groups")
    func shippedWidthsUnchanged() {
        let shipped: [Int: Int] = [2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3]
        for (m, ipg) in Qwen35CustomQMV.inputGroupPlan where m <= 9 {
            #expect(ipg == shipped[m])
        }
    }

    /// The launch witness must equal `ceil(m / ipg)`, which is what the Metal
    /// body's `first_m = group * IPG` early return assumes.
    @Test("the launch witness matches the plan")
    func activeInputGroupsMatchesPlan() {
        for (m, ipg) in Qwen35CustomQMV.enabledInputGroupPlan {
            #expect(Qwen35CustomQMV.activeInputGroups(m) == (m + ipg - 1) / ipg)
        }
    }

    /// With the override unset the generated Metal source and the kernel names
    /// are the shipped ones: the dispatch table is the same eight widths and
    /// the name suffix is empty.
    @Test("the default bound reproduces the shipped dispatch exactly")
    func defaultsReproduceShippedDispatch() {
        #expect(Qwen35CustomQMV.maxWidth == 9)
        #expect(Qwen35CustomQMV.widths == 2 ... 9)
        #expect(Qwen35CustomQMV.kernelNameWidthSuffix == "")
        #expect(
            Qwen35CustomQMV.enabledInputGroupPlan.map(\.0) == Array(2 ... 9))
        #expect(
            Qwen35CustomQMV.enabledInputGroupPlan.map(\.1)
                == [2, 3, 4, 5, 3, 4, 4, 3])
    }

    // MARK: - trusted draft-depth ceiling

    @Test("the shipped ceiling is 8 and every malformed override lands there")
    func depthParserDefaults() {
        #expect(MLXFastConstants.qwenMTPMaxDraftDepthDefault == 8)
        #expect(MLXFastConstants.qwenMTPMaxDraftDepthHardCap == 15)
        for raw in [nil, "", "  ", "eight", "0", "-1", "16", "99", "8.0"] {
            #expect(MLXFastConstants.parseMaxDraftDepth(raw) == 8,
                    "\(String(describing: raw)) should fall back to the default")
        }
    }

    @Test("a well-formed depth override is taken verbatim")
    func depthParserAccepts() {
        for value in 1 ... 15 {
            #expect(MLXFastConstants.parseMaxDraftDepth("\(value)") == value)
        }
        #expect(MLXFastConstants.parseMaxDraftDepth(" 12 ") == 12)
    }

    /// The whole point of the change: one source, so every consumer moves
    /// together. `Qwen36MTPLimits.maxDepth` is the worker's mirror and it is
    /// defined as `MLXFastConstants.qwenMTPMaxDepth`, which is an alias of
    /// the draft-depth ceiling.
    @Test("the wire alias is the same number as the trusted ceiling")
    func aliasTracksTheCeiling() {
        #expect(MLXFastConstants.qwenMTPMaxDepth
            == MLXFastConstants.qwenMTPMaxDraftDepth)
        #expect(MLXFastConstants.qwenMTPMaxDraftDepth == 8)
    }

    /// The vendored attention chunk cannot import `MLXFastCore`, so it
    /// restates the arithmetic. This pins the restatement: the chunk must
    /// cover a verify block of `ceiling + 1` rows for every input, including
    /// the malformed ones, or the widest legal round silently falls off the
    /// exact path.
    @Test("the attention chunk bound is always the ceiling plus one row")
    func chunkBoundTracksTheCeiling() {
        let inputs: [String?] = [
            nil, "", "  ", "eight", "0", "-1", "1", "2", "5", "8", "9",
            "12", "15", "16", "99", " 12 ", "8.0",
        ]
        for raw in inputs {
            let depth = MLXFastConstants.parseMaxDraftDepth(raw)
            let rows = parseWideDecodeExactnessMaxQueryRows(raw)
            let detail = "\(String(describing: raw)): chunk covers \(rows) "
                + "rows but the ceiling admits a \(depth + 1)-row block"
            #expect(rows == depth + 1, "\(detail)")
        }
    }

    @Test("the chunk's own defaults match the shipped ceiling")
    func chunkDefaults() {
        #expect(wideDecodeExactnessDefaultMaxQueryRows
            == MLXFastConstants.qwenMTPMaxDraftDepthDefault + 1)
        #expect(wideDecodeExactnessHardCapQueryRows
            == MLXFastConstants.qwenMTPMaxDraftDepthHardCap + 1)
        #expect(wideDecodeExactnessMaxQueryRows == 9)
    }

    /// The generalized segmentation must reproduce the shipped two-way split
    /// for every width the shipped code handled. Segment `[start, end)` reads
    /// `keys[..<kL - (qL - end)]`; at qL = 6...9 that is exactly rows 0..<5
    /// over `kL - (qL - 5)` keys followed by rows 5..<qL over all `kL`.
    @Test("the segmentation reproduces the shipped split at widths 6 through 9")
    func segmentationMatchesShippedSplit() {
        let kL = 4096
        for qL in 6 ... 9 {
            var segments: [(Int, Int, Int)] = []
            var start = 0
            while start < qL {
                let end = min(start + 5, qL)
                segments.append((start, end, kL - (qL - end)))
                start = end
            }
            #expect(segments.count == 2)
            #expect(segments[0] == (0, 5, kL - (qL - 5)))
            #expect(segments[1] == (5, qL, kL))
        }
    }

    /// Bottom-right causal alignment, for any segmentation: absolute row `i`
    /// of a `qL`-row block must see `kL - qL + 1 + i` keys.
    @Test("every segment gives each row its serial causal window")
    func segmentationPreservesCausalWindows() {
        let kL = 1024
        for qL in 6 ... 16 {
            var start = 0
            while start < qL {
                let end = min(start + 5, qL)
                let kEnd = kL - (qL - end)
                let length = end - start
                for i in start ..< end {
                    // Bottom-right alignment inside the segment.
                    let window = kEnd - (length - 1 - (i - start))
                    #expect(window == kL - qL + 1 + i,
                            "qL \(qL) row \(i) sees \(window) keys")
                }
                start = end
            }
        }
    }
}
