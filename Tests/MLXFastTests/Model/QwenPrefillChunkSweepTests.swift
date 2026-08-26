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

    private func raw(chunk: Int, cached: Int) -> Double {
        points.first { $0.chunk == chunk && $0.cached == cached }?.seconds ?? 0
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

        // `callWithHidden` returns an unlabeled tuple with no `.hidden`
        // member, so this measurement uses `callWithHiddenNormedAndLayers`
        // with an empty layer capture list instead, exactly as
        // `QwenPrefillScalingTests.swift` does: it returns
        // `Qwen35ForwardOutput`, which does publish `.hidden`.
        func forward(
            input: LMInput.Text, cache: [any KVCache]
        ) -> MLXArray {
            model.callWithHiddenNormedAndLayers(
                input: input, cache: cache, nConfirmed: 0, layerIDs: []
            ).hidden
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
            // silently answer a different question. The quiet gate is what
            // makes a single sample usable, and Task 4 confirms the conclusion
            // end to end rather than trusting this surface alone.
            let start = Date()
            let hidden = forward(
                input: LMInput.Text(tokens: tokens(chunk, offset: cached)),
                cache: cache)
            eval(hidden)
            return Date().timeIntervalSince(start)
        }

        let chunkWidths = [256, 512, 1024, 2048, 4096]
        let depths = [0, 2048, 8192, 16384]
        var points: [ChunkCostGrid.Point] = []
        print("\nAppended-chunk cost, fused attention path")
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
            print(String(
                format: "  %7d  %7d  %11.2f  %10.3fx",
                cap, schedule.count, predicted,
                baseline == 0 ? 1 : predicted / baseline))
        }
        print("")
    }
}
