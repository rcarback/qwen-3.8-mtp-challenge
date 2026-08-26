import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

@testable import MLXFastCore
@testable import MLXFastModel

/// Per-component cost profile of the block drafter.
///
/// WHY THIS EXISTS. Paired end-to-end runs put the block drafter 10.2% behind
/// the declared autoregressive head per round even after the body was halved to
/// affine 4-bit, and byte accounting alone does not explain it: at 4 bits the
/// two heads read comparable weight volume per round. This splits one drafting
/// round into its four parts so the residual can be attributed instead of
/// guessed.
///
/// It builds SYNTHETIC inputs rather than driving a real target: the drafter
/// borrows the target's 2.5 GB embedding table and its vocabulary projection,
/// and neither is needed to price the drafter's own work. The vocabulary
/// projections are stand-ins with the real shapes and quantization, which is
/// what their cost depends on at batch 1.
///
///     MLXFAST_DFLASH2_HEAD_PATH=<drafter directory> \
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions \
///       --filter Qwen38DFlash2ProfileTests
@Suite(.serialized)
struct Qwen38DFlash2ProfileTests {
    /// Rows measured in the paired runs: 3.74 drafts per round plus the anchor,
    /// against 4.4 committed context rows.
    private static let blockRows = 5
    private static let contextRows = 5
    private static let cacheWarmRows = 128

    @Test("DFlash2 per-component cost profile")
    func componentProfile() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            let headPath = ProcessInfo.processInfo
                .environment["MLXFAST_DFLASH2_HEAD_PATH"]
        else { return }

        let head = try Qwen38DFlash2Head.load(
            from: URL(fileURLWithPath: headPath))
        let config = head.config
        let hiddenSize = config.hiddenSize

        // Stand-ins for what the drafter borrows from the target.
        let embedded = MLXRandom.normal([1, Self.blockRows, hiddenSize])
            .asType(.bfloat16)
        let targetHidden = MLXRandom
            .normal([1, Self.contextRows, config.contextWidth])
            .asType(.bfloat16)

        // A quantized vocabulary projection with the real shape. affine 4-bit
        // group-64 is the target checkpoint's scheme, so the packed widths are
        // hiddenSize * 4 / 32 and hiddenSize / 64.
        func projection(rows: Int) -> (MLXArray, MLXArray, MLXArray) {
            let packed = MLXRandom
                .randInt(0 ..< Int32.max, [rows, hiddenSize * 4 / 32])
                .asType(.uint32)
            let scales = MLXRandom.normal([rows, hiddenSize / 64])
                .asType(.bfloat16)
            let biases = MLXRandom.normal([rows, hiddenSize / 64])
                .asType(.bfloat16)
            eval(packed, scales, biases)
            return (packed, scales, biases)
        }
        let exactHead = projection(rows: config.vocabSize)
        let compactHead = projection(rows: 98_336)

        func project(
            _ x: MLXArray, _ w: (MLXArray, MLXArray, MLXArray)
        ) -> MLXArray {
            quantizedMM(
                x, w.0, scales: w.1, biases: w.2, transpose: true,
                groupSize: 64, bits: 4, mode: .affine)
        }

        // One cache, warmed to a realistic depth. Attention over a few hundred
        // rows at five query rows is compute-trivial next to the 936 MB layer
        // read, so the drift as iterations append is not material.
        let cache = head.makeCache()
        let warm = MLXRandom
            .normal([1, Self.cacheWarmRows, config.contextWidth])
            .asType(.bfloat16)
        let warmContext = head.hiddenNorm(head.contextProjection(warm))
        var warmH = MLXRandom
            .normal([1, Self.cacheWarmRows, hiddenSize]).asType(.bfloat16)
        for (layer, layerCache) in zip(head.layers, cache) {
            warmH = layer(
                warmH, context: warmContext, rope: head.ropeForTesting,
                cache: layerCache)
        }
        eval(warmH)

        func measure(
            _ label: String, iterations: Int = 25, _ body: () -> MLXArray
        ) -> Double {
            for _ in 0 ..< 5 { eval(body()) }
            var samples = [Double]()
            samples.reserveCapacity(iterations)
            for _ in 0 ..< iterations {
                let start = Date()
                eval(body())
                samples.append(Date().timeIntervalSince(start) * 1000)
            }
            samples.sort()
            let median = samples[samples.count / 2]
            let padded = label.padding(
                toLength: 30, withPad: " ", startingAt: 0)
            print(
                "  \(padded) median \(String(format: "%7.3f", median)) ms"
                    + "   min \(String(format: "%7.3f", samples[0])) ms")
            return median
        }

        print("\nDFlash2 component profile"
            + " (block \(Self.blockRows) rows, context \(Self.contextRows) rows)")

        let fcCost = measure("fc + hidden_norm") {
            head.hiddenNorm(head.contextProjection(targetHidden))
        }

        let context = head.hiddenNorm(head.contextProjection(targetHidden))
        eval(context)

        let layerCost = measure("5 decoder layers") {
            var h = embedded
            for (layer, layerCache) in zip(head.layers, cache) {
                h = layer(
                    h, context: context, rope: head.ropeForTesting,
                    cache: layerCache)
            }
            return head.norm(h[0..., 1..., 0...])
        }

        var hidden = embedded
        for (layer, layerCache) in zip(head.layers, cache) {
            hidden = layer(
                hidden, context: context, rope: head.ropeForTesting,
                cache: layerCache)
        }
        hidden = head.norm(hidden[0..., 1..., 0...])
        eval(hidden)

        let exactCost = measure("lm_head exact (248320)") {
            project(hidden, exactHead)
        }
        let compactCost = measure("lm_head compact (98336)") {
            project(hidden, compactHead)
        }

        let exactLogits = project(hidden, exactHead)
        let anchors = MLXArray([Int32(1234)]).reshaped([1, 1])[0..., 0]
        eval(exactLogits, anchors)

        let selectorCost = measure("selector walk") {
            head.candidateSelector.select(
                hidden: hidden, logits: exactLogits, anchorIDs: anchors).0
        }

        let total = fcCost + layerCost + exactCost + selectorCost
        print("  " + String(repeating: "-", count: 58))
        print("  round total \(String(format: "%7.3f", total)) ms")
        for (name, cost) in [
            ("fc + hidden_norm", fcCost), ("5 decoder layers", layerCost),
            ("lm_head exact", exactCost), ("selector walk", selectorCost),
        ] {
            let padded = name.padding(
                toLength: 30, withPad: " ", startingAt: 0)
            print("  \(padded) \(String(format: "%5.1f", 100 * cost / total))%")
        }
        print("  compact projection would save "
            + "\(String(format: "%.3f", exactCost - compactCost)) ms"
            + " (\(String(format: "%.1f", 100 * (exactCost - compactCost) / total))% of the round)\n")
    }
}
