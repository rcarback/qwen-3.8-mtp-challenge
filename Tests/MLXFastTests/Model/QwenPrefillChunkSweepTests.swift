import Foundation
import MLX
import MLXLLM
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

@testable import MLXFastCore
@testable import MLXFastModel

/// A measured cost surface over (chunk width, already-cached positions), and
/// the schedule integrator that turns it into a whole-prompt prediction.
///
/// Measuring every cap end to end costs minutes per sample and a cool-down.
/// Measuring a grid of single appended chunks costs seconds per point, and the
/// real chunk schedule is a pure function of the cap, so the whole-prompt cost
/// follows by integration.
struct ChunkCostGrid {
    struct Point {
        let chunk: Int
        let cached: Int
        let seconds: Double
    }

    let points: [Point]
    private let chunks: [Int]
    private let depths: [Int]

    init(points: [Point]) {
        self.points = points
        chunks = Array(Set(points.map(\.chunk))).sorted()
        depths = Array(Set(points.map(\.cached))).sorted()
    }

    /// Bilinear interpolation over the measured grid, clamped at the edges.
    ///
    /// Clamping rather than extrapolating is deliberate. Outside the measured
    /// box the surface is unknown, and a linear extrapolation of a term that
    /// grows with the product of its axes would report confident nonsense.
    func seconds(chunk: Int, cached: Int) -> Double {
        guard !points.isEmpty else { return 0 }
        let (c0, c1, ct) = Self.bracket(chunk, in: chunks)
        let (d0, d1, dt) = Self.bracket(cached, in: depths)
        let s00 = raw(chunk: c0, cached: d0)
        let s10 = raw(chunk: c1, cached: d0)
        let s01 = raw(chunk: c0, cached: d1)
        let s11 = raw(chunk: c1, cached: d1)
        let low = s00 + (s10 - s00) * ct
        let high = s01 + (s11 - s01) * ct
        return low + (high - low) * dt
    }

    /// Looks up a point the axes promise exists.
    ///
    /// `bracket` only ever returns coordinates drawn from `chunks` and
    /// `depths`, which are built from the points themselves, so a miss means
    /// the grid is ragged rather than rectangular. That is a programming
    /// error, and it traps: returning zero instead would under-price the
    /// missing corner and quietly bias every sum that crosses it.
    private func raw(chunk: Int, cached: Int) -> Double {
        guard let point = points.first(where: {
            $0.chunk == chunk && $0.cached == cached
        }) else {
            fatalError("cost grid has no point at chunk \(chunk), "
                + "cached \(cached); the grid must be rectangular")
        }
        return point.seconds
    }

    /// Returns the two grid values bracketing `value` and the fraction between
    /// them. Clamps to the end values outside the measured span.
    private static func bracket(
        _ value: Int, in axis: [Int]
    ) -> (Int, Int, Double) {
        guard let first = axis.first, let last = axis.last else {
            return (0, 0, 0)
        }
        if value <= first { return (first, first, 0) }
        if value >= last { return (last, last, 0) }
        for index in 1 ..< axis.count where value <= axis[index] {
            let low = axis[index - 1]
            let high = axis[index]
            let span = Double(high - low)
            let fraction = span == 0 ? 0 : Double(value - low) / span
            return (low, high, fraction)
        }
        return (last, last, 0)
    }

    /// The exact chunk schedule the session would take for this cap.
    static func schedule(
        tokens: Int, cap: Int, budget: Int
    ) -> [(chunk: Int, cached: Int)] {
        var out: [(chunk: Int, cached: Int)] = []
        var index = 0
        while index < tokens {
            let size = Qwen36MTPBlockSession.prefillChunkSize(
                cached: index, cap: cap, budget: budget)
            let end = min(index + size, tokens)
            out.append((chunk: end - index, cached: index))
            index = end
        }
        return out
    }

    /// Whole-prompt cost for a cap, by integrating the measured surface over
    /// that cap's schedule.
    func predictedSeconds(tokens: Int, cap: Int, budget: Int) -> Double {
        Self.schedule(tokens: tokens, cap: cap, budget: budget)
            .reduce(0) { $0 + seconds(chunk: $1.chunk, cached: $1.cached) }
    }

    /// Whether a cap's schedule asks the surface for a point it never
    /// measured.
    ///
    /// This matters because `seconds(chunk:cached:)` clamps outside the grid
    /// instead of extrapolating. A clamped lookup is not an error, but summing
    /// clamped lookups IS: a schedule of 8192-wide chunks priced against a
    /// grid that stops at 4096 charges 8192 tokens the cost of 4096 and halves
    /// the total by construction. The bias runs toward wide chunks, which is
    /// the direction a chunk-width hypothesis wants to be true, so the
    /// prediction must say when it has left the measured box rather than
    /// return a confident number.
    func predictionLeavesGrid(tokens: Int, cap: Int, budget: Int) -> Bool {
        guard let widest = chunks.last, let deepest = depths.last else {
            return true
        }
        return Self.schedule(tokens: tokens, cap: cap, budget: budget)
            .contains { $0.chunk > widest || $0.cached > deepest }
    }
}

@Suite
struct ChunkCostGridTests {
    private static let budget = 64 << 20

    @Test("the schedule matches the session's own chunk derivation")
    func scheduleShape() {
        let flat = ChunkCostGrid.schedule(
            tokens: 3000, cap: 1024, budget: Self.budget)
        // 64 Mi over any depth below 65536 exceeds 1024, so the cap binds all
        // the way: 1024, 1024, 952.
        #expect(flat.map(\.chunk) == [1024, 1024, 952])
        #expect(flat.map(\.cached) == [0, 1024, 2048])
        // Every schedule covers the prompt exactly once, whatever the cap.
        for cap in [512, 1024, 2048, 4096, 8192] {
            let s = ChunkCostGrid.schedule(
                tokens: 11682, cap: cap, budget: Self.budget)
            #expect(s.reduce(0) { $0 + $1.chunk } == 11682)
            #expect(s.first?.cached == 0)
        }
        // A wider cap takes strictly fewer chunks over the same prompt.
        let narrow = ChunkCostGrid.schedule(
            tokens: 11682, cap: 1024, budget: Self.budget).count
        let wide = ChunkCostGrid.schedule(
            tokens: 11682, cap: 4096, budget: Self.budget).count
        #expect(wide < narrow)
    }

    @Test("interpolation returns measured points exactly")
    func interpolationAtGridPoints() {
        let grid = ChunkCostGrid(points: [
            .init(chunk: 256, cached: 0, seconds: 1.0),
            .init(chunk: 1024, cached: 0, seconds: 2.0),
            .init(chunk: 256, cached: 8192, seconds: 3.0),
            .init(chunk: 1024, cached: 8192, seconds: 6.0),
        ])
        #expect(grid.seconds(chunk: 256, cached: 0) == 1.0)
        #expect(grid.seconds(chunk: 1024, cached: 8192) == 6.0)
    }

    @Test("interpolation is bilinear between points and clamps outside")
    func interpolationBetweenAndOutside() {
        let grid = ChunkCostGrid(points: [
            .init(chunk: 256, cached: 0, seconds: 1.0),
            .init(chunk: 1024, cached: 0, seconds: 2.0),
            .init(chunk: 256, cached: 8192, seconds: 3.0),
            .init(chunk: 1024, cached: 8192, seconds: 6.0),
        ])
        // Halfway on the chunk axis at depth 0: midway between 1.0 and 2.0.
        #expect(abs(grid.seconds(chunk: 640, cached: 0) - 1.5) < 1e-9)
        // Halfway on both axes: midway between the two edge midpoints, 1.5
        // and 4.5.
        #expect(abs(grid.seconds(chunk: 640, cached: 4096) - 3.0) < 1e-9)
        // Outside the box, clamp rather than extrapolate.
        #expect(grid.seconds(chunk: 8192, cached: 0) == 2.0)
        #expect(grid.seconds(chunk: 64, cached: 0) == 1.0)
        #expect(grid.seconds(chunk: 1024, cached: 99999) == 6.0)
    }

    /// The twenty points measured on a quiet host on 2026-08-26, seconds per
    /// chunk, transcribed from the run recorded in
    /// `docs/qwen-prefill-research-plan.md`.
    ///
    /// These are pinned so the conclusion drawn in that document is checkable
    /// without a 29-minute re-measurement, and so a later edit to the surface
    /// or the schedule cannot quietly change what the recorded numbers imply.
    static let measured2026_08_26: [ChunkCostGrid.Point] = [
        .init(chunk: 256, cached: 0, seconds: 2.345),
        .init(chunk: 512, cached: 0, seconds: 4.758),
        .init(chunk: 1024, cached: 0, seconds: 9.813),
        .init(chunk: 2048, cached: 0, seconds: 20.253),
        .init(chunk: 4096, cached: 0, seconds: 43.688),
        .init(chunk: 256, cached: 2048, seconds: 2.470),
        .init(chunk: 512, cached: 2048, seconds: 4.960),
        .init(chunk: 1024, cached: 2048, seconds: 10.238),
        .init(chunk: 2048, cached: 2048, seconds: 21.224),
        .init(chunk: 4096, cached: 2048, seconds: 45.203),
        .init(chunk: 256, cached: 8192, seconds: 2.633),
        .init(chunk: 512, cached: 8192, seconds: 5.230),
        .init(chunk: 1024, cached: 8192, seconds: 10.945),
        .init(chunk: 2048, cached: 8192, seconds: 24.994),
        .init(chunk: 4096, cached: 8192, seconds: 47.831),
        .init(chunk: 256, cached: 16384, seconds: 2.820),
        .init(chunk: 512, cached: 16384, seconds: 5.775),
        .init(chunk: 1024, cached: 16384, seconds: 12.313),
        .init(chunk: 2048, cached: 16384, seconds: 25.203),
        .init(chunk: 4096, cached: 16384, seconds: 53.147),
    ]

    @Test("the recorded measurement still says what the document claims")
    func recordedMeasurementSupportsTheConclusion() {
        let grid = ChunkCostGrid(points: Self.measured2026_08_26)
        let tokens = 11682

        // The document's claim is that the widest chunk costs more per token
        // than the narrowest at every measured depth. Check it on the points
        // themselves rather than on the prose.
        //
        // The claim is deliberately about the endpoints and not about strict
        // monotonicity, because the surface is not strictly monotonic: at
        // depth 8192 chunk 512 reads slightly under chunk 256, and chunk 4096
        // reads under chunk 2048. Asserting a clean rise here would assert
        // something the measurement does not show.
        for depth in [0, 2048, 8192, 16384] {
            let perToken = [256, 512, 1024, 2048, 4096].map { width in
                grid.seconds(chunk: width, cached: depth) / Double(width)
            }
            #expect(perToken.last! > perToken.first!)
            #expect(perToken.max()! == perToken.dropFirst(3).max()!)
        }

        // The three caps the document asks the reader to act on stay inside
        // the measured box, so their predictions are real sums, not clamped
        // ones.
        for cap in [1024, 2048, 4096] {
            #expect(!grid.predictionLeavesGrid(
                tokens: tokens, cap: cap, budget: Self.budget))
        }
        // Cap 8192 does not, which is exactly the row the document refuses to
        // quote. Before `predictionLeavesGrid` existed the sum reported it as
        // 84.76 s, or 0.702x, because two 8192-wide chunks were each priced at
        // the 4096 edge. The flag is what stops that number being printed.
        #expect(grid.predictionLeavesGrid(
            tokens: tokens, cap: 8192, budget: Self.budget))

        // The whole-prompt figures quoted in the document, to the two decimals
        // it quotes them at.
        let predicted = [1024, 2048, 4096].map { cap in
            (grid.predictedSeconds(
                tokens: tokens, cap: cap, budget: Self.budget) * 100)
                .rounded() / 100
        }
        #expect(predicted == [120.81, 129.59, 130.84])
        // Narrow wins: that is the finding, and it is what closed the sweep.
        #expect(predicted[0] < predicted[1])
        #expect(predicted[1] < predicted[2])
    }

    @Test("a prediction reports when its schedule leaves the measured box")
    func predictionOutsideGridIsFlagged() {
        let grid = ChunkCostGrid(points: [
            .init(chunk: 256, cached: 0, seconds: 1.0),
            .init(chunk: 4096, cached: 0, seconds: 16.0),
            .init(chunk: 256, cached: 16384, seconds: 2.0),
            .init(chunk: 4096, cached: 16384, seconds: 32.0),
        ])
        // Caps inside the measured chunk span stay inside the box: the
        // deepest prompt below is 11682 tokens, under the 16384 depth edge.
        for cap in [1024, 2048, 4096] {
            #expect(!grid.predictionLeavesGrid(
                tokens: 11682, cap: cap, budget: Self.budget))
        }
        // A cap above the widest measured chunk does not, and the clamp would
        // otherwise price those chunks at the 4096 cost.
        #expect(grid.predictionLeavesGrid(
            tokens: 11682, cap: 8192, budget: Self.budget))
        // Depth counts too: a prompt longer than the deepest measured row
        // leaves the box even at an in-range cap.
        #expect(grid.predictionLeavesGrid(
            tokens: 40000, cap: 1024, budget: Self.budget))
    }

    @Test("a whole-prompt prediction sums the schedule over the surface")
    func predictionSumsSchedule() {
        // A surface that is flat in depth and exactly linear in chunk means
        // every cap must predict the same whole-prompt cost, because total
        // work is conserved. That is the invariant a prediction bug breaks.
        let grid = ChunkCostGrid(points: [
            .init(chunk: 256, cached: 0, seconds: 0.256),
            .init(chunk: 4096, cached: 0, seconds: 4.096),
            .init(chunk: 256, cached: 16384, seconds: 0.256),
            .init(chunk: 4096, cached: 16384, seconds: 4.096),
        ])
        let narrow = grid.predictedSeconds(
            tokens: 8192, cap: 1024, budget: Self.budget)
        let wide = grid.predictedSeconds(
            tokens: 8192, cap: 4096, budget: Self.budget)
        #expect(abs(narrow - 8.192) < 1e-6)
        #expect(abs(wide - 8.192) < 1e-6)
    }
}

/// Fills the cost grid from the real model. Opt-in, because it loads the
/// 21.6 GB checkpoint.
///
///     ./tools/host-quiet-gate.sh && \
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
///     MLXFAST_QWEN_PREFILL_WEIGHTS=weights \
///     MLXFAST_QWEN_PREFILL_HEAD=<head dir> \
///     swift test --force-resolved-versions --filter chunkCostSurface
@Suite(.serialized)
struct QwenPrefillChunkSweepTests {
    @Test("chunk cost surface against the fused attention path")
    func chunkCostSurface() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            let weights = env["MLXFAST_QWEN_PREFILL_WEIGHTS"],
            let head = env["MLXFAST_QWEN_PREFILL_HEAD"]
        else { return }

        let targetURL = URL(fileURLWithPath: weights)
        let headURL = URL(fileURLWithPath: head)
        let context = try Qwen36MTPHeadAttachment.withHeadAttached(
            backboneDirectory: targetURL, headDirectory: headURL
        ) { _ in
            let box = UnsafeSendableBox<ModelContext>()
            let semaphore = DispatchSemaphore(value: 0)
            Task {
                box.value = try? await LLMModelFactory.shared.load(
                    from: targetURL, using: #huggingFaceTokenizerLoader())
                semaphore.signal()
            }
            semaphore.wait()
            guard let loaded = box.value else {
                throw MLXFastError.invalidInput("failed to load backbone")
            }
            return loaded
        }
        guard let model = context.model as? any Qwen36MTPTarget else {
            Issue.record("backbone is not an MTP target")
            return
        }

        func tokens(_ count: Int, offset: Int) -> MLXArray {
            MLXArray((0 ..< count).map {
                Int32((($0 + offset) * 7919) % 100_000 + 10)
            }).reshaped([1, count])
        }

        // Production `forwardPrefill` calls `callWithHidden`, so this
        // measurement calls it too. The protocol returns an UNLABELED
        // `(logits, pre-norm hidden)` pair, which is why `.hidden` does not
        // resolve; the element is `.1`. The sibling
        // `callWithHiddenNormedAndLayers` would also work and is what
        // `QwenPrefillScalingTests.swift` uses, but it returns the POST-norm
        // hidden, and pricing a forward the server does not run is exactly the
        // ambiguity this instrument exists to remove.
        //
        // Evaluating the hidden forces the 64-layer backbone and leaves the
        // vocabulary projection as dead graph, which is what production
        // prefill does as well: MLX is lazy, and no prefill chunk evaluates
        // logits it will not read.
        func forward(
            input: LMInput.Text, cache: [any KVCache]
        ) -> MLXArray {
            model.callWithHidden(
                input: input, cache: cache, nConfirmed: 0
            ).1
        }

        /// Seconds for ONE appended chunk of `chunk` tokens onto a cache that
        /// already holds `cached` positions. The warm-up prefill is untimed.
        func appendSeconds(chunk: Int, cached: Int) -> Double {
            let cache = model.newCache(parameters: nil)
            if cached > 0 {
                var filled = 0
                while filled < cached {
                    let width = min(1024, cached - filled)
                    let hidden = forward(
                        input: LMInput.Text(tokens: tokens(width, offset: filled)),
                        cache: cache)
                    eval(hidden)
                    filled += width
                }
            }
            // ONE timed sample, not a best-of-N. The timed call advances
            // the cache, so a second sample would price a deeper position and
            // silently answer a different question.
            //
            // Know what that single sample does and does not control. The
            // quiet gate holds host contention down, which is real but
            // partial. It does nothing about position inside this process,
            // and this loop walks chunk width and depth in the same
            // direction it walks position. The nearest independent reading
            // is the separate-process pair in the `prefillChunkRange` doc
            // comment. The reversed-order control that settles the position
            // question is built in below: set
            // MLXFAST_QWEN_CHUNK_SWEEP_REVERSED=1 and compare the two
            // surfaces.
            let start = Date()
            let hidden = forward(
                input: LMInput.Text(tokens: tokens(chunk, offset: cached)),
                cache: cache)
            eval(hidden)
            return Date().timeIntervalSince(start)
        }

        // THE REVERSED-ORDER CONTROL. Position inside the process biases
        // every point by an amount that grows with when it ran, and the
        // forward loop walks chunk width in the same direction it walks
        // position, so the bias and the conclusion point the same way.
        // Reversing the chunk loop measures the SAME shapes and moves the
        // position tax to the other end of the sweep. The two orders
        // therefore bracket the truth: if narrow chunks still win when they
        // are measured last, position did not produce the result; if the
        // ordering flips, the surface was artifact. Set
        // MLXFAST_QWEN_CHUNK_SWEEP_REVERSED=1 for the control pass. The
        // header line records which order produced the numbers, because a
        // surface whose order is not on its face cannot be compared later.
        let reversed = env["MLXFAST_QWEN_CHUNK_SWEEP_REVERSED"] == "1"
        var chunkWidths = [256, 512, 1024, 2048, 4096]
        if reversed { chunkWidths.reverse() }
        let depths = [0, 2048, 8192, 16384]
        var points: [ChunkCostGrid.Point] = []
        print("\nAppended-chunk cost, fused attention path"
            + " (chunk order: \(reversed ? "descending" : "ascending"))")
        print("     chunk    cached   seconds   ms/token")
        for cached in depths {
            for chunk in chunkWidths {
                let dt = appendSeconds(chunk: chunk, cached: cached)
                points.append(.init(chunk: chunk, cached: cached, seconds: dt))
                print(String(
                    format: "  %8d  %8d  %8.3f  %9.4f",
                    chunk, cached, dt, 1000 * dt / Double(chunk)))
            }
        }

        let grid = ChunkCostGrid(points: points)
        let budget = Qwen36MTPBlockSession.prefillChunkProductBudget
        print("\nPredicted whole-prompt prefill by cap")
        print("      cap   chunks   predicted_s   versus_1024")
        var baseline = 0.0
        for cap in [1024, 2048, 4096, 8192] {
            let schedule = ChunkCostGrid.schedule(
                tokens: 11682, cap: cap, budget: budget)
            let predicted = grid.predictedSeconds(
                tokens: 11682, cap: cap, budget: budget)
            if cap == 1024 { baseline = predicted }
            if grid.predictionLeavesGrid(
                tokens: 11682, cap: cap, budget: budget) {
                // Numeric columns keep the numeric format. Only the two cost
                // columns change, because there is no cost to report.
                print(String(format: "  %7d  %7d", cap, schedule.count)
                    + "    unmeasured          --")
                continue
            }
            print(String(
                format: "  %7d  %7d  %11.2f  %10.3fx",
                cap, schedule.count, predicted,
                baseline == 0 ? 1 : predicted / baseline))
        }
        print("")
    }
}
