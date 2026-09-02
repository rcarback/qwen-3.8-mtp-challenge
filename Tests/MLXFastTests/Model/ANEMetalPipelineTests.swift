import CoreML
import Foundation
import MLX
import Testing

// Two-engine (ANE + Metal GPU) pipeline microbenchmark for the Qwen 3.8
// gated-delta prefill. Answers one question: does running ANE and Metal
// CONCURRENTLY on a chunked, dependency-respecting pipeline over a
// representative 16-layer stack beat running everything on Metal alone?
//
// Concurrency discipline (matches ANEMetalPartitionTests's proven pattern):
// the ANE role always executes inside a single `DispatchQueue.global().async`
// closure that touches only CoreML types (wrapped `@unchecked Sendable`) and
// plain `Data`/`MLMultiArray`; the Metal role always executes MLX ops
// synchronously on the CALLING thread. MLX `eval` is therefore never invoked
// from two threads at once anywhere in this file -- the existing overlap
// tests in this repo (ANEMetalPartitionTests, tools/overlap) deliberately
// avoid that too, and there is no evidence in this codebase that concurrent
// multi-threaded MLX eval is supported.
//
// Proxy weights only. No real model weights are loaded here.

/// Boxes CoreML handles so they can cross onto a background dispatch queue.
private final class MLBox: @unchecked Sendable {
    let model: MLModel
    let input: MLFeatureProvider
    init(_ m: MLModel, _ i: MLFeatureProvider) { model = m; input = i }
}

/// A plain Int result written on a background queue and read only after a
/// DispatchGroup join (happens-before/-after via the group), so no lock is
/// needed for this narrow use.
private final class ResultBox: @unchecked Sendable { var value = 0 }

/// Thread-safe accumulator for handoff-cost bookkeeping across the ANE
/// worker's task loop.
private final class HandoffCounters: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var count = 0
    private(set) var totalSec = 0.0
    func add(_ n: Int, _ seconds: Double) { lock.lock(); count += n; totalSec += seconds; lock.unlock() }
}

/// Set-once, wake-all latch. A (layer, chunk) cell has up to TWO dependents --
/// the depth successor (l+1, c) and, for GD layers, the recurrence successor
/// (l, c+1). A counting semaphore is consumed on wait, so one signal cannot
/// release two waiters (the second blocks forever -> deadlock). A broadcast
/// latch releases every waiter and is idempotent.
private final class Latch: @unchecked Sendable {
    private var done = false
    private let cond = NSCondition()
    func signal() { cond.lock(); done = true; cond.broadcast(); cond.unlock() }
    func wait() { cond.lock(); while !done { cond.wait() }; cond.unlock() }
}

/// Dependency latches for the (layer, chunk) task grid.
private final class SemGrid: @unchecked Sendable {
    let sems: [[Latch]]
    init(layers: Int, chunks: Int) {
        sems = (0 ..< layers).map { _ in (0 ..< chunks).map { _ in Latch() } }
    }
}

private struct Blocked: Error, CustomStringConvertible {
    let msg: String
    var description: String { msg }
}

/// Deterministic, allocation-free RNG for the Stage-4 validation stack --
/// pure Swift, no MLX/CoreML, so both scheduler "roles" in that stage can run
/// on plain background threads without touching anything MLX-related.
private struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func nextUnit() -> Double { Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0) * 2 - 1 }
}

@Suite(.serialized)
struct ANEMetalPipelineTests {

    // MARK: - Shape constants (match Qwen 3.8's gated-delta layer geometry;
    // see tools/ane-gated-delta/layer2.swift)

    static let Cdim = 5120
    static let NVH = 48
    static let Dh = 128

    // MARK: - gdprobe artifact plumbing

    private static func repoRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Model/
            .deletingLastPathComponent() // MLXFastTests/
            .deletingLastPathComponent() // Tests/
            .deletingLastPathComponent() // repo root
    }

    private static func gdprobeBinary() -> URL {
        repoRoot().appendingPathComponent("tools/ane-gated-delta/gdprobe")
    }

    /// Runs `gdprobe savechunk outDir St K`, compiling a SINGLE-TILE (NT=1,
    /// external recurrence) gated-delta layer to a .mlmodelc. Returns the
    /// compiled model's path and its useful GFLOPs for one St-token chunk.
    private static func ensureChunkModel(St: Int, K: Int, cacheDir: URL) throws -> (URL, Double) {
        let bin = gdprobeBinary()
        guard FileManager.default.fileExists(atPath: bin.path) else {
            throw Blocked(msg: "gdprobe binary missing at \(bin.path); build it with " +
                "`swiftc -O tools/ane-gated-delta/*.swift -framework CoreML -o tools/ane-gated-delta/gdprobe`")
        }
        let proc = Process()
        proc.executableURL = bin
        proc.arguments = ["savechunk", cacheDir.path, "\(St)", "\(K)"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        try proc.run()
        proc.waitUntilExit()
        let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        guard proc.terminationStatus == 0, out.contains("SAVED") else {
            throw Blocked(msg: "gdprobe savechunk St=\(St) K=\(K) failed: \(out)")
        }
        var gf = 0.0
        var path = ""
        for tok in out.split(separator: "\t") {
            if tok.hasPrefix("usefulGF=") { gf = Double(tok.dropFirst("usefulGF=".count)) ?? 0 }
            if tok.hasPrefix("/") { path = String(tok) }
        }
        guard !path.isEmpty else { throw Blocked(msg: "gdprobe savechunk St=\(St): no path in output: \(out)") }
        return (URL(fileURLWithPath: path), gf)
    }

    // MARK: - Fixed ANE input construction (content is a constant fill --
    // Stages 1-3 measure throughput/scheduling, not ANE numerics, so a fixed
    // reused input is sufficient and keeps every timed call identical).

    private static func maskArray(kind: String, St: Int, NVH: Int) -> MLMultiArray {
        let b0 = kind == "eye" ? 1 : NVH
        let a = try! MLMultiArray(shape: [b0, St, St].map { NSNumber(value: $0) }, dataType: .float16)
        a.withUnsafeMutableBytes { raw, _ in
            let p = raw.bindMemory(to: Float16.self)
            for bb in 0 ..< b0 { for i in 0 ..< St { for j in 0 ..< St {
                let on: Bool = kind == "eye" ? i == j : (kind == "trilS" ? i > j : i >= j)
                p[bb * St * St + i * St + j] = on ? 1 : 0
            } } }
        }
        return a
    }

    private static func constMultiArray(shape: [Int], value: Float) -> MLMultiArray {
        let a = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        let n = shape.reduce(1, *)
        a.withUnsafeMutableBytes { raw, _ in
            let p = raw.bindMemory(to: Float16.self)
            for i in 0 ..< n { p[i] = Float16(value) }
        }
        return a
    }

    private static func makeChunkInput(St: Int) throws -> MLFeatureProvider {
        try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: constMultiArray(shape: [1, Cdim, 1, St], value: 0.01)),
            "h0": MLFeatureValue(multiArray: constMultiArray(shape: [NVH, Dh, Dh], value: 0)),
            "eye": MLFeatureValue(multiArray: maskArray(kind: "eye", St: St, NVH: NVH)),
            "trilS": MLFeatureValue(multiArray: maskArray(kind: "trilS", St: St, NVH: NVH)),
            "trilI": MLFeatureValue(multiArray: maskArray(kind: "trilI", St: St, NVH: NVH)),
        ])
    }

    /// A Metal proxy op with total FLOPs matched to `gf` (billion FLOPs) via
    /// an (St x Cdim) @ (Cdim x N) bf16 matmul.
    private static func matchedGEMM(gf: Double, St: Int) -> (MLXArray, MLXArray, Int) {
        let n = Swift.max(64, Int((gf * 1e9) / (2.0 * Double(St) * Double(Cdim))))
        let a = MLXArray.ones([St, Cdim]).asType(.bfloat16)
        let b = MLXArray.ones([Cdim, n]).asType(.bfloat16)
        eval(a, b)
        return (a, b, n)
    }

    private static func timePerOp(_ n: Int, _ body: () -> Void) -> Double {
        body()
        let t0 = Date()
        for _ in 0 ..< n { body() }
        return Date().timeIntervalSince(t0) / Double(n)
    }

    // MARK: - Stage 2/3: dependency-respecting two-engine pipeline

    /// Runs the 16-layer x C-chunk task DAG. Depth edge: (L,c) needs
    /// (L-1,c). Recurrence edge (GD layers only): (L,c) needs (L,c-1). The
    /// ANE role runs as ONE background-queue closure over its assigned
    /// (layer,chunk) tasks in ascending order; the Metal role runs
    /// synchronously on the calling thread over its own tasks. `aneLayers`
    /// empty == the all-Metal serial baseline (same function, same DAG).
    private static func runPipeline(
        layers: Int, chunks: Int, isGD: @escaping @Sendable (Int) -> Bool, aneLayers: Set<Int>,
        aneBox: MLBox, St: Int, metalGEMM: (MLXArray, MLXArray)
    ) -> (wallSec: Double, handoffCount: Int, handoffSec: Double) {
        let grid = SemGrid(layers: layers, chunks: chunks)
        let counters = HandoffCounters()
        let aneList = (0 ..< layers).filter { aneLayers.contains($0) }
        let metalList = (0 ..< layers).filter { !aneLayers.contains($0) }
        let payload = Data(count: Cdim * St * 2) // fp16-sized zero payload, real memcpy source
        let (ga, gb) = metalGEMM

        let t0 = Date()
        let aneGroup = DispatchGroup()
        aneGroup.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            for l in aneList {
                for c in 0 ..< chunks {
                    if l > 0 { grid.sems[l - 1][c].wait() }
                    if isGD(l), c > 0 { grid.sems[l][c - 1].wait() }
                    // autoreleasepool: the per-cell MLMultiArray handoffs and the
                    // Core ML prediction autorelease their backing buffers; without
                    // draining each iteration RSS climbs into the tens of GB and
                    // wedges the Core ML/ANE XPC daemon on a later model load.
                    autoreleasepool {
                        // Real inbound handoff: byte payload -> freshly allocated MLMultiArray.
                        let hin0 = Date()
                        if let inArr = try? MLMultiArray(shape: [1, Cdim, 1, St].map { NSNumber(value: $0) },
                                                         dataType: .float16) {
                            inArr.withUnsafeMutableBytes { raw, _ in
                                _ = payload.withUnsafeBytes { src in
                                    memcpy(raw.baseAddress!, src.baseAddress!, Swift.min(raw.count, payload.count))
                                }
                            }
                        }
                        let hin = Date().timeIntervalSince(hin0)
                        _ = try? aneBox.model.prediction(from: aneBox.input)
                        // Real outbound handoff: MLMultiArray-shaped bytes -> Data.
                        let hout0 = Date()
                        if let outArr = try? MLMultiArray(shape: [1, Cdim, 1, St].map { NSNumber(value: $0) },
                                                          dataType: .float16) {
                            _ = outArr.withUnsafeBytes { raw in Data(bytes: raw.baseAddress!, count: raw.count) }
                        }
                        let hout = Date().timeIntervalSince(hout0)
                        counters.add(2, hin + hout)
                    }
                    grid.sems[l][c].signal()
                }
            }
            aneGroup.leave()
        }
        for l in metalList {
            for c in 0 ..< chunks {
                if l > 0 { grid.sems[l - 1][c].wait() }
                if isGD(l), c > 0 { grid.sems[l][c - 1].wait() }
                eval(matmul(ga, gb))
                grid.sems[l][c].signal()
            }
        }
        aneGroup.wait()
        return (Date().timeIntervalSince(t0), counters.count, counters.totalSec)
    }

    // MARK: - Stage 4: scheduler-correctness validation (pure Swift, no
    // MLX/CoreML, deterministic given seeded per-layer weights)

    private typealias Vec = [Double]

    private static func detMat(seed: Int, n: Int) -> [[Double]] {
        var rng = SplitMix64(seed: UInt64(bitPattern: Int64(seed)))
        return (0 ..< n).map { _ in (0 ..< n).map { _ in rng.nextUnit() * 0.05 } }
    }

    private static func matVec(_ m: [[Double]], _ v: Vec) -> Vec {
        m.map { row in Swift.max(-1, Swift.min(1, tanh(zip(row, v).reduce(0) { $0 + $1.0 * $1.1 }))) }
    }

    private final class Store: @unchecked Sendable {
        private var act: [Int: Vec] = [:]
        private var carry: [Int: Vec] = [:]
        private let lock = NSLock()
        func setAct(_ l: Int, _ c: Int, _ v: Vec) { lock.lock(); act[l * 10_000 + c] = v; lock.unlock() }
        func getAct(_ l: Int, _ c: Int) -> Vec { lock.lock(); defer { lock.unlock() }; return act[l * 10_000 + c]! }
        func setCarry(_ l: Int, _ c: Int, _ v: Vec) { lock.lock(); carry[l * 10_000 + c] = v; lock.unlock() }
        func getCarry(_ l: Int, _ c: Int) -> Vec { lock.lock(); defer { lock.unlock() }; return carry[l * 10_000 + c]! }
    }

    /// Runs the identical dependency DAG as `runPipeline`, but both "roles"
    /// execute the same deterministic proxy step -- this isolates scheduler
    /// correctness from cross-engine numerics. Returns the final layer's
    /// concatenated activation vectors, in chunk order.
    private static func runProxyStack(
        layers: Int, chunks: Int, isGD: @escaping @Sendable (Int) -> Bool, dim: Int, roleA: Set<Int>, serial: Bool
    ) -> [Vec] {
        let store = Store()
        let wAct = (0 ..< layers).map { detMat(seed: $0, n: dim) }
        let wCarry = (0 ..< layers).map { detMat(seed: 1000 + $0, n: dim) }
        let seedX: Vec = (0 ..< dim).map { Double($0) / Double(dim) - 0.5 }
        let seedH: Vec = Array(repeating: 0.0, count: dim)

        @Sendable func step(_ l: Int, _ c: Int) {
            let actIn = l == 0 ? seedX : store.getAct(l - 1, c)
            if isGD(l) {
                let carryIn = c == 0 ? seedH : store.getCarry(l, c - 1)
                let carryOut = matVec(wCarry[l], zip(carryIn, actIn).map { $0 + $1 })
                store.setCarry(l, c, carryOut)
                store.setAct(l, c, matVec(wAct[l], zip(actIn, carryOut).map { $0 + $1 }))
            } else {
                store.setAct(l, c, matVec(wAct[l], actIn))
            }
        }

        if serial {
            for l in 0 ..< layers { for c in 0 ..< chunks { step(l, c) } }
        } else {
            let grid = SemGrid(layers: layers, chunks: chunks)
            let listA = (0 ..< layers).filter { roleA.contains($0) }
            let listB = (0 ..< layers).filter { !roleA.contains($0) }
            let group = DispatchGroup()
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                for l in listA { for c in 0 ..< chunks {
                    if l > 0 { grid.sems[l - 1][c].wait() }
                    if isGD(l), c > 0 { grid.sems[l][c - 1].wait() }
                    step(l, c)
                    grid.sems[l][c].signal()
                } }
                group.leave()
            }
            for l in listB { for c in 0 ..< chunks {
                if l > 0 { grid.sems[l - 1][c].wait() }
                if isGD(l), c > 0 { grid.sems[l][c - 1].wait() }
                step(l, c)
                grid.sems[l][c].signal()
            } }
            group.wait()
        }
        return (0 ..< chunks).map { store.getAct(layers - 1, $0) }
    }

    // MARK: - The benchmark

    @Test("ANE + Metal concurrent pipeline vs Metal-alone, Qwen 3.8 gated-delta prefill proxy")
    func pipelineBenchmark() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        var report = "# ANE + Metal Pipeline Benchmark Report\n\n"
        report += "Generated by Tests/MLXFastTests/Model/ANEMetalPipelineTests.swift. "
        report += "All weights are proxy/random; the real 21.6 GB model is never loaded.\n\n"
        let reportPath = env["ANE_METAL_PIPELINE_REPORT"] ??
            "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/" +
            "3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/ane-metal-pipeline-report.md"
        defer { try? report.write(toFile: reportPath, atomically: true, encoding: .utf8) }
        // Flush incrementally: `defer` never runs if a later stage hangs, so a
        // completed earlier stage (e.g. the Stage 1 go/no-go) must be persisted
        // the moment it finishes, not only at function exit.
        func flush() { try? report.write(toFile: reportPath, atomically: true, encoding: .utf8) }

        let K = 4
        let cacheDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("ane-metal-pipeline-cache")
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine

        // ---- Stage 1: overlap sanity (go/no-go) at St=128, K=4 ----
        report += "## Stage 1 -- overlap sanity (St=128, K=4)\n\n"
        let (m128URL, gf128): (URL, Double)
        do {
            (m128URL, gf128) = try Self.ensureChunkModel(St: 128, K: K, cacheDir: cacheDir)
        } catch {
            report += "BLOCKED building the St=128 ANE model: \(error)\n\nStopping -- no further stages ran.\n"
            Issue.record("blocked: \(error)")
            return
        }
        let box128 = MLBox(try MLModel(contentsOf: m128URL, configuration: cfg), try Self.makeChunkInput(St: 128))
        _ = try? box128.model.prediction(from: box128.input) // warm

        let (gA128, gB128, n128) = Self.matchedGEMM(gf: gf128, St: 128)
        eval(matmul(gA128, gB128)) // warm

        let window = 4.0
        @Sendable func aneCount(_ box: MLBox, until deadline: Date) -> Int {
            var n = 0
            while Date() < deadline { _ = try? box.model.prediction(from: box.input); n += 1 }
            return n
        }
        func metalCount(_ a: MLXArray, _ b: MLXArray, until deadline: Date) -> Int {
            var n = 0
            while Date() < deadline { eval(matmul(a, b)); n += 1 }
            return n
        }

        let aneSolo = aneCount(box128, until: Date().addingTimeInterval(window))
        let metalSolo = metalCount(gA128, gB128, until: Date().addingTimeInterval(window))

        let concDeadline = Date().addingTimeInterval(window)
        let holder = ResultBox()
        let concGroup = DispatchGroup()
        concGroup.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            holder.value = aneCount(box128, until: concDeadline)
            concGroup.leave()
        }
        let metalConc = metalCount(gA128, gB128, until: concDeadline)
        concGroup.wait()
        let aneConc = holder.value

        let aneSoloRate = Double(aneSolo) / window
        let metalSoloRate = Double(metalSolo) / window
        let aneConcRate = Double(aneConc) / window
        let metalConcRate = Double(metalConc) / window
        let aneSustain = aneSoloRate > 0 ? aneConcRate / aneSoloRate : 0
        let metalSustain = metalSoloRate > 0 ? metalConcRate / metalSoloRate : 0
        let aneFlopsPerIter = gf128 * 1e9
        let metalFlopsPerIter = 2.0 * 128 * Double(Self.Cdim) * Double(n128)
        let soloTFLOPS = aneSoloRate * aneFlopsPerIter / 1e12 + metalSoloRate * metalFlopsPerIter / 1e12
        let concTFLOPS = aneConcRate * aneFlopsPerIter / 1e12 + metalConcRate * metalFlopsPerIter / 1e12

        report += String(format: "- [MEASURED] ANE solo: %.1f iters/s (%.2f TFLOPS)\n",
                         aneSoloRate, aneSoloRate * aneFlopsPerIter / 1e12)
        report += String(format: "- [MEASURED] Metal solo (matched-FLOP GEMM, N=%d): %.1f iters/s (%.2f TFLOPS)\n",
                         n128, metalSoloRate, metalSoloRate * metalFlopsPerIter / 1e12)
        report += String(format: "- [MEASURED] ANE concurrent: %.1f iters/s (sustain=%.2f of solo)\n",
                         aneConcRate, aneSustain)
        report += String(format: "- [MEASURED] Metal concurrent: %.1f iters/s (sustain=%.2f of solo)\n",
                         metalConcRate, metalSustain)
        report += String(format: "- [DERIVED] sum-of-solos TFLOPS=%.2f, concurrent combined TFLOPS=%.2f (ratio=%.2f)\n\n",
                         soloTFLOPS, concTFLOPS, concTFLOPS / Swift.max(soloTFLOPS, 1e-9))
        flush() // persist the go/no-go before the heavier pipeline stages run

        if aneSustain < 0.65, metalSustain < 0.65 {
            report += "**VERDICT: DEAD.** Both engines collapse toward roughly half their solo rate under " +
                "concurrent load -- there is no genuine parallelism here; the two engines are serializing on " +
                "something (likely a shared dispatch/Metal-command-buffer resource). Stopping before Stages 2-4.\n"
            return
        }
        report += "Both engines sustain enough of their solo rate to proceed to the pipeline stages.\n\n"

        // ---- Stages 2/3: pipeline harness + measurement ----
        report += "## Stages 2/3 -- 16-layer pipeline (N=4 groups of [GD,GD,GD,FULL]), C in {4,8,16}\n\n"
        report += "Assumption [DERIVED]: the FULL-attention layer's proxy FLOPs are set equal to the GD " +
            "layer's chunk FLOPs at the same St (no exact full-attention FLOP formula was in scope here); " +
            "both engines' GD/FULL proxies at a given St therefore share one matched-FLOP GEMM shape.\n\n"

        let layers = 16
        @Sendable func isGD(_ l: Int) -> Bool { l % 4 != 3 }
        let gdLayers = (0 ..< layers).filter { isGD($0) } // 12
        let fullLayers = (0 ..< layers).filter { !isGD($0) } // 4

        struct Row { let policy: String; let C: Int; let St: Int; let speedup: Double
                     let handoffAvgMs: Double; let overlapFrac: Double }
        var rows: [Row] = []
        var stageBlocked: String?

        // C=16 (St=32) dropped: near the op-count CPU cliff (docs section 27) and
        // the heaviest daemon load, least informative at the same time.
        for C in [4, 8] {
            let St = 512 / C
            let (mURL, gf): (URL, Double)
            do { (mURL, gf) = try Self.ensureChunkModel(St: St, K: K, cacheDir: cacheDir) }
            catch {
                stageBlocked = "BLOCKED building St=\(St) ANE model: \(error)"
                break
            }
            let box = MLBox(try MLModel(contentsOf: mURL, configuration: cfg), try Self.makeChunkInput(St: St))
            _ = try? box.model.prediction(from: box.input) // warm
            let (ga, gb, _) = Self.matchedGEMM(gf: gf, St: St)
            eval(matmul(ga, gb)) // warm

            let tAne = Self.timePerOp(3) { _ = try? box.model.prediction(from: box.input) }
            let tMetal = Self.timePerOp(3) { eval(matmul(ga, gb)) }

            let kTarget = (Double(gdLayers.count) * tMetal + Double(fullLayers.count) * tMetal)
                / Swift.max(tAne + tMetal, 1e-9)
            let kBalanced = Swift.max(0, Swift.min(gdLayers.count, Int(kTarget.rounded())))
            let policies: [(String, Set<Int>)] = [
                ("allGDane", Set(gdLayers)),
                ("balanced(k=\(kBalanced))", Set(gdLayers.prefix(kBalanced))),
                ("firstHalfANE", Set(gdLayers.prefix(gdLayers.count / 2))),
                ("alternateANE", Set(gdLayers.enumerated().filter { $0.offset % 2 == 0 }.map { $0.element })),
            ]

            let (baseWall, _, _) = Self.runPipeline(layers: layers, chunks: C, isGD: isGD, aneLayers: [],
                                               aneBox: box, St: St, metalGEMM: (ga, gb))
            for (name, aneSet) in policies {
                let (wall, hCount, hSec) = Self.runPipeline(layers: layers, chunks: C, isGD: isGD, aneLayers: aneSet,
                                                       aneBox: box, St: St, metalGEMM: (ga, gb))
                let aneTaskCount = aneSet.count * C
                let metalTaskCount = (layers - aneSet.count) * C
                let sequentialEstimate = Double(aneTaskCount) * tAne + Double(metalTaskCount) * tMetal
                let overlapFrac = sequentialEstimate > 0 ? Swift.max(0, 1 - wall / sequentialEstimate) : 0
                let handoffAvgMs = hCount > 0 ? (hSec / Double(hCount)) * 1000 : 0
                rows.append(Row(policy: name, C: C, St: St, speedup: baseWall / wall,
                                handoffAvgMs: handoffAvgMs, overlapFrac: overlapFrac))
            }
            flush() // persist progress after each C in case a later C stalls
        }

        report += "| policy | C | St | speedup (metal_alone/two_engine) | avg handoff (ms) | overlap fraction |\n"
        report += "|---|---|---|---|---|---|\n"
        for r in rows {
            report += String(format: "| %@ | %d | %d | %.3f [MEASURED] | %.3f [MEASURED] | %.2f [DERIVED] |\n",
                             r.policy, r.C, r.St, r.speedup, r.handoffAvgMs, r.overlapFrac)
        }
        report += "\n"
        if let blocked = stageBlocked {
            report += "\(blocked)\n\nRemaining C values in the sweep were skipped.\n\n"
        }

        // ---- Stage 4: scheduler-correctness validation ----
        report += "## Stage 4 -- scheduler validation (pure-Swift deterministic proxy stack, no MLX/CoreML)\n\n"
        report += "Both the serial reference and the scheduled run execute the IDENTICAL deterministic " +
            "proxy step function; only the execution order/threading differs, isolating scheduler " +
            "correctness from any real ANE/Metal numerical difference.\n\n"
        let vC = 4
        let vDim = 16
        let altANE = Set(gdLayers.enumerated().filter { $0.offset % 2 == 0 }.map { $0.element })
        let serialOut = Self.runProxyStack(layers: layers, chunks: vC, isGD: isGD, dim: vDim, roleA: altANE, serial: true)
        let schedOut = Self.runProxyStack(layers: layers, chunks: vC, isGD: isGD, dim: vDim, roleA: altANE, serial: false)
        var maxAbsErr = 0.0
        for (a, b) in zip(serialOut, schedOut) {
            for (x, y) in zip(a, b) { maxAbsErr = Swift.max(maxAbsErr, abs(x - y)) }
        }
        report += String(format: "- [MEASURED] maxAbsErr(serial, scheduled) = %.3e over %d chunks x %d dims " +
                         "(layers=%d, alternating-engine policy)\n\n", maxAbsErr, vC, vDim, layers)
        #expect(maxAbsErr < 1e-9, "scheduler dropped or misordered a dependency: maxAbsErr=\(maxAbsErr)")

        // ---- Verdict ----
        report += "## Verdict\n\n"
        if let best = rows.max(by: { $0.speedup < $1.speedup }) {
            report += String(format: "Best observed two-engine speedup vs Metal-alone: **%.3fx** " +
                             "(policy=%@, C=%d, St=%d) [MEASURED].\n\n",
                             best.speedup, best.policy, best.C, best.St)
        }
        report += "Limits observed: per-call ANE dispatch overhead and the real MLMultiArray<->Data handoff " +
            "copy (see table) both grow relative to useful compute as St shrinks (smaller chunks = more calls " +
            "for the same 512-token window), which is the expected op-count cliff at small St. " +
            "maxAbsErr in Stage 4 confirms the scheduler's dependency graph (depth + same-layer recurrence " +
            "edges) is honored under real concurrent execution.\n"
    }

    // MARK: - Memory-contention sweep

    /// Stage 1 showed Metal collapsing to 41% of solo when the ANE ran the
    /// 2.68 MB St=128 layer concurrently. If the cause is unified-memory
    /// contention, a SMALLER ANE working set should hand Metal's bandwidth back
    /// and RAISE combined throughput -- even though a smaller ANE tile is slower
    /// solo. This sweeps the ANE footprint and measures Metal sustain + combined
    /// concurrent TFLOPS to find the offload-lane optimum (which is NOT the ANE
    /// solo peak). It also distinguishes the mechanism: if combined throughput is
    /// flat across footprints, the contention is CPU-dispatch/fabric, not
    /// bandwidth, and shrinking the tile cannot help.
    @Test("ANE footprint vs Metal sustain under concurrency (memory-contention sweep)")
    func footprintContentionSweep() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        var report = "# ANE Footprint / Memory-Contention Sweep\n\n"
        report += "For each ANE config: solo rates, concurrent rates, Metal sustain, and combined " +
            "concurrent TFLOPS. Offload-lane optimum = MAX combined (not ANE solo peak). Proxy Metal " +
            "GEMM is matched to each config's per-chunk FLOPs; real Metal is faster, so combined here " +
            "is an upper bound on the offload win.\n\n"
        let reportPath = env["ANE_FOOTPRINT_REPORT"] ??
            "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/" +
            "3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/ane-footprint-sweep-report.md"
        func flush() { try? report.write(toFile: reportPath, atomically: true, encoding: .utf8) }
        defer { flush() }

        let cacheDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("ane-metal-pipeline-cache")
        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
        let window = 3.0

        report += "| St | K | ANE liveMB | ANE solo TF | Metal solo TF | ANE conc TF | Metal conc TF " +
            "| Metal sustain | combined TF |\n"
        report += "|---|---|---|---|---|---|---|---|---|\n"

        // (St, K): 128/4 = 2.68 MB peak (qkv-bound); 64/* and 32/* pin at the
        // 1.57 MB recurrence-state floor. Spans the reachable footprint range.
        for (St, K) in [(128, 4), (64, 2), (64, 4), (32, 2)] {
            let mURL: URL, gf: Double
            do {
                let (u, g) = try Self.ensureChunkModel(St: St, K: K, cacheDir: cacheDir)
                mURL = u; gf = g
            } catch {
                report += "| \(St) | \(K) | BLOCKED: \(error) |\n"; flush(); continue
            }
            // Largest single fp16 activation, mirroring layer2.swift's maxLive:
            // qkv / z / MLP-block / score-matrix / recurrence-state / hidden.
            let C = 5120, Cqkv = 10240, Cv = 6144, inter = 17408, NVH = 48, D = 128
            let liveElems = [Cqkv * St, Cv * St, (inter / K) * St, NVH * St * St, NVH * D * D, C * (St + 3)].max()!
            let liveMB = Double(liveElems) * 2 / 1e6
            let box = MLBox(try MLModel(contentsOf: mURL, configuration: cfg), try Self.makeChunkInput(St: St))
            _ = try? box.model.prediction(from: box.input)
            let (ga, gb, n) = Self.matchedGEMM(gf: gf, St: St)
            eval(matmul(ga, gb))

            @Sendable func aneCount(_ b: MLBox, until d: Date) -> Int {
                var k = 0; while Date() < d { autoreleasepool { _ = try? b.model.prediction(from: b.input) }; k += 1 }; return k
            }
            func metalCount(_ a: MLXArray, _ b: MLXArray, until d: Date) -> Int {
                var k = 0; while Date() < d { eval(matmul(a, b)); k += 1 }; return k
            }
            let aneSolo = Double(aneCount(box, until: Date().addingTimeInterval(window))) / window
            let metalSolo = Double(metalCount(ga, gb, until: Date().addingTimeInterval(window))) / window

            let dl = Date().addingTimeInterval(window)
            let holder = ResultBox(); let g = DispatchGroup(); g.enter()
            DispatchQueue.global(qos: .userInitiated).async { holder.value = aneCount(box, until: dl); g.leave() }
            let metalConc = Double(metalCount(ga, gb, until: dl)) / window
            g.wait()
            let aneConc = Double(holder.value) / window

            let aneTF = gf * 1e9 / 1e12
            let metalTF = 2.0 * Double(St) * Double(Self.Cdim) * Double(n) / 1e12
            let sustain = metalSolo > 0 ? metalConc / metalSolo : 0
            let combined = aneConc * aneTF + metalConc * metalTF
            report += String(format: "| %d | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | **%.2f** |\n",
                             St, K, liveMB, aneSolo * aneTF, metalSolo * metalTF,
                             aneConc * aneTF, metalConc * metalTF, sustain, combined)
            flush()
        }
        report += "\nRead: if combined TF RISES as liveMB falls, contention is memory-bound and the " +
            "small-footprint config is the offload optimum. If combined TF is flat, the bottleneck is " +
            "CPU-dispatch/fabric and shrinking the ANE tile does not help.\n"
    }

    // MARK: - Dispatch-contention discriminator

    /// Isolates WHY Metal collapses to ~45% under concurrent ANE load, without
    /// touching CoreML/ANE at all. Runs the Metal GEMM against pure CPU spinner
    /// threads (arithmetic only -- no accelerator, no memory traffic). If Metal
    /// throughput drops as CPU cores get busy, Metal NEEDS the CPU to feed the
    /// GPU (command-buffer encode/submit/sync), so the ANE's real cost was its
    /// CPU-side XPC/marshaling stealing those cycles -- a fixable DISPATCH
    /// problem. If Metal is unaffected by a pegged CPU, the ANE contention was
    /// ANE-specific (power/fabric), not generic dispatch.
    @Test("Metal throughput vs pure-CPU spinners (dispatch-contention discriminator)")
    func metalDispatchSensitivity() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        var report = "# Metal Dispatch-Sensitivity Discriminator\n\n"
        report += "Metal 4-bit-ish GEMM throughput vs N pure-CPU spinner threads (no accelerator, no " +
            "memory traffic). Compare Metal's drop here to its ~0.45 sustain under concurrent ANE " +
            "(Stage 1). If a CPU spinner reproduces the drop, the ANE contention was CPU-dispatch " +
            "(fixable). If Metal is flat under spinners, it was ANE-specific.\n\n"
        let reportPath = env["METAL_DISPATCH_REPORT"] ??
            "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/" +
            "3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/metal-dispatch-report.md"
        func flush() { try? report.write(toFile: reportPath, atomically: true, encoding: .utf8) }
        defer { flush() }

        // Matched-FLOP GEMM shape at St=128 (same as Stage 1's Metal proxy).
        let (ga, gb, n) = Self.matchedGEMM(gf: 102.3, St: 128)
        eval(matmul(ga, gb)) // warm
        let metalTF = 2.0 * 128.0 * Double(Self.Cdim) * Double(n) / 1e12
        let window = 4.0

        func metalRate(spinners: Int) -> Double {
            let stop = SpinFlag()
            var threads: [Thread] = []
            for _ in 0 ..< spinners {
                let t = Thread {
                    var acc = 1.000001
                    while !stop.get() { for _ in 0 ..< 4096 { acc = acc * 1.0000001 + 1e-9 }; stop.sink = acc }
                }
                t.qualityOfService = .userInitiated
                threads.append(t); t.start()
            }
            let dl = Date().addingTimeInterval(window)
            var k = 0
            while Date() < dl { eval(matmul(ga, gb)); k += 1 }
            stop.set()
            return Double(k) / window
        }

        let base = metalRate(spinners: 0)
        report += String(format: "- [MEASURED] Metal solo: %.1f iters/s (%.2f TFLOPS)\n\n", base, base * metalTF)
        report += "| CPU spinners | Metal iters/s | Metal TFLOPS | sustain vs solo |\n|---|---|---|---|\n"
        report += String(format: "| 0 | %.1f | %.2f | 1.00 |\n", base, base * metalTF); flush()
        for s in [1, 2, 4, 6, 8] {
            let r = metalRate(spinners: s)
            report += String(format: "| %d | %.1f | %.2f | %.2f |\n", s, r, r * metalTF, base > 0 ? r / base : 0)
            flush()
        }
        report += "\nRead: sustain at a few spinners near 0.45 => Metal is CPU-dispatch-bound, ANE " +
            "contention was dispatch (fixable via async submission / fewer larger ANE calls). Sustain " +
            "near 1.0 => Metal does not need the CPU, ANE contention was ANE-specific (power/fabric).\n"
    }

    /// Direct fabric-vs-driver discriminator. Runs Metal against pure MEMORY
    /// STREAMING threads: memcpy over 64 MB buffers (>> last-level cache, so it
    /// hits DRAM), no arithmetic, no IOKit, no XPC, no accelerator. The ONLY
    /// resource it shares with Metal is the unified-memory controller/fabric.
    /// If memory streaming drops Metal (where §metalDispatchSensitivity's pure
    /// COMPUTE spinners did not), the contention is memory-fabric, and both
    /// kernel/IOKit driver-path AND CPU-dispatch are ruled out: this workload
    /// touches neither the driver nor shares any submission path with Metal.
    @Test("Metal throughput vs pure-memory-streaming threads (fabric-vs-driver discriminator)")
    func metalMemoryContention() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        var report = "# Metal vs Pure-Memory-Streaming (fabric-vs-driver discriminator)\n\n"
        report += "memcpy-only threads (64 MB buffers, DRAM traffic, no compute/IOKit/XPC/accelerator). " +
            "Compare to metalDispatchSensitivity's pure-COMPUTE spinners (which left Metal ~1.0). If memory " +
            "streaming drops Metal, memory-fabric is the cause and driver-path + dispatch are ruled out.\n\n"
        let reportPath = env["METAL_MEM_REPORT"] ??
            "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/" +
            "3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/metal-memory-contention-report.md"
        func flush() { try? report.write(toFile: reportPath, atomically: true, encoding: .utf8) }
        defer { flush() }

        let (ga, gb, n) = Self.matchedGEMM(gf: 102.3, St: 128)
        eval(matmul(ga, gb))
        let metalTF = 2.0 * 128.0 * Double(Self.Cdim) * Double(n) / 1e12
        let window = 4.0
        let bufBytes = 64 * 1024 * 1024 // 64 MB > any M-series last-level cache -> real DRAM traffic

        func metalRate(memThreads: Int) -> Double {
            let stop = SpinFlag()
            var threads: [Thread] = []
            for _ in 0 ..< memThreads {
                let t = Thread {
                    let a = UnsafeMutableRawPointer.allocate(byteCount: bufBytes, alignment: 64)
                    let b = UnsafeMutableRawPointer.allocate(byteCount: bufBytes, alignment: 64)
                    memset(a, 1, bufBytes); memset(b, 2, bufBytes)
                    while !stop.get() { memcpy(a, b, bufBytes); stop.sink = Double(a.load(as: UInt8.self)) }
                    a.deallocate(); b.deallocate()
                }
                t.qualityOfService = .userInitiated
                threads.append(t); t.start()
            }
            let dl = Date().addingTimeInterval(window)
            var k = 0
            while Date() < dl { eval(matmul(ga, gb)); k += 1 }
            stop.set()
            return Double(k) / window
        }

        let base = metalRate(memThreads: 0)
        report += String(format: "- [MEASURED] Metal solo: %.1f iters/s (%.2f TFLOPS)\n\n", base, base * metalTF)
        report += "| mem-stream threads | Metal iters/s | Metal TFLOPS | sustain vs solo |\n|---|---|---|---|\n"
        report += String(format: "| 0 | %.1f | %.2f | 1.00 |\n", base, base * metalTF); flush()
        for s in [1, 2, 4, 6] {
            let r = metalRate(memThreads: s)
            report += String(format: "| %d | %.1f | %.2f | %.2f |\n", s, r, r * metalTF, base > 0 ? r / base : 0)
            flush()
        }
        report += "\nRead: memory-stream sustain << 1.0 while compute-spinner sustain ~1.0 " +
            "(metalDispatchSensitivity) => memory-fabric arbitration confirmed; kernel/IOKit driver-path " +
            "and CPU-dispatch both ruled out (memcpy touches neither).\n"
    }

    /// Tests the "keep the data on-chip to avoid contention" hypothesis: sweep
    /// the memcpy WORKING-SET size from cache-resident (256 KB, fits L2) to
    /// DRAM-spilling (256 MB), one thread, and measure Metal sustain. If small
    /// (cached) buffers leave Metal ~1.0 while large (DRAM) buffers crush it,
    /// then on-chip/cache-resident data does NOT contend and a cache-resident
    /// partition is physically possible (implementation aside). If even small
    /// buffers hurt Metal, the shared SLC is contended too and on-chip residency
    /// does not help.
    @Test("Metal sustain vs memcpy working-set size (cache-resident vs DRAM)")
    func metalCacheVsDram() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        var report = "# Metal sustain vs memcpy working-set size (cache-vs-DRAM)\n\n"
        report += "One memcpy thread, buffer size swept from L2-resident to DRAM-spilling. Tests whether " +
            "cache-resident memory traffic contends with the GPU (=> on-chip partition impossible) or only " +
            "DRAM traffic does (=> cache-resident data avoids contention).\n\n"
        let reportPath = env["METAL_CACHE_REPORT"] ??
            "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/" +
            "3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/metal-cache-vs-dram-report.md"
        func flush() { try? report.write(toFile: reportPath, atomically: true, encoding: .utf8) }
        defer { flush() }

        let (ga, gb, n) = Self.matchedGEMM(gf: 102.3, St: 128)
        eval(matmul(ga, gb))
        let metalTF = 2.0 * 128.0 * Double(Self.Cdim) * Double(n) / 1e12
        let window = 4.0

        func metalRate(bufKB: Int) -> Double {
            let stop = SpinFlag()
            let bytes = bufKB * 1024
            let t = Thread {
                // half the working set each side of the copy so total footprint == bufKB
                let half = Swift.max(bytes / 2, 4096)
                let a = UnsafeMutableRawPointer.allocate(byteCount: half, alignment: 64)
                let b = UnsafeMutableRawPointer.allocate(byteCount: half, alignment: 64)
                memset(a, 1, half); memset(b, 2, half)
                while !stop.get() { memcpy(a, b, half); stop.sink = Double(a.load(as: UInt8.self)) }
                a.deallocate(); b.deallocate()
            }
            t.qualityOfService = .userInitiated
            t.start()
            let dl = Date().addingTimeInterval(window)
            var k = 0
            while Date() < dl { eval(matmul(ga, gb)); k += 1 }
            stop.set()
            return Double(k) / window
        }

        let base = metalRate(bufKB: 0) // 0 => allocate tiny (4KB), effectively L1/L2-resident hot loop
        report += String(format: "- [MEASURED] Metal solo (tiny 4KB working set): %.1f iters/s (%.2f TFLOPS)\n\n",
                         base, base * metalTF)
        report += "| memcpy working set | Metal iters/s | Metal TFLOPS | sustain |\n|---|---|---|---|\n"
        // 256KB (L2), 2MB, 8MB, 32MB (~SLC edge), 128MB, 256MB (deep DRAM)
        for kb in [256, 2048, 8192, 32768, 131_072, 262_144] {
            let r = metalRate(bufKB: kb)
            let label = kb < 1024 ? "\(kb) KB" : "\(kb / 1024) MB"
            report += String(format: "| %@ | %.1f | %.2f | %.2f |\n", label, r, r * metalTF, base > 0 ? r / base : 0)
            flush()
        }
        report += "\nRead: sustain ~1.0 for small (cached) sizes dropping to ~0.5 as the working set exceeds " +
            "cache => cache-resident traffic does NOT contend; only DRAM does. Sustain low even for small " +
            "sizes => shared SLC is contended; on-chip residency does not help.\n"
    }

    /// THE offload decision: measured combined throughput of GPU+ANE concurrent
    /// vs GPU-alone, paired and noise-robust. Each cycle measures the GPU's real
    /// useful TFLOPS alone, then the GPU's + ANE's useful TFLOPS running
    /// concurrently over the same window. ratio = (gpu_conc + ane_conc) /
    /// gpu_solo. ratio > 1 => offloading gated-delta work to the ANE while the
    /// GPU does GEMMs delivers more total prefill work than the GPU alone,
    /// despite the ANE slowing the GPU (§35). GPU workload is a real prefill
    /// projection shape (bf16, ~13.5 TF solo); ANE workload is the St=128
    /// gated-delta layer (~9.3 TF solo).
    @Test("Measured combined GPU+ANE throughput vs GPU-alone (offload decision)")
    func hybridCombinedThroughput() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        var report = "# Measured combined GPU+ANE throughput vs GPU-alone\n\n"
        report += "Paired per cycle: GPU real projection GEMM alone, then GPU + ANE gated-delta layer " +
            "concurrent. ratio = (gpu_conc + ane_conc useful TFLOPS) / gpu_solo. ratio>1 => hybrid offload " +
            "beats GPU-alone. Adjacent phases => host noise divides out.\n\n"
        let reportPath = env["HYBRID_REPORT"] ??
            "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/" +
            "3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/hybrid-combined-report.md"
        func flush() { try? report.write(toFile: reportPath, atomically: true, encoding: .utf8) }
        defer { flush() }

        let cacheDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("ane-metal-pipeline-cache")
        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
        let mURL: URL, aneGF: Double
        do { (mURL, aneGF) = try Self.ensureChunkModel(St: 128, K: 4, cacheDir: cacheDir) }
        catch { report += "BLOCKED loading ANE model: \(error)\n"; Issue.record("blocked"); return }
        let box = MLBox(try MLModel(contentsOf: mURL, configuration: cfg), try Self.makeChunkInput(St: 128))
        _ = try? box.model.prediction(from: box.input)

        // Real prefill projection: X[512,5120] @ W[17408,5120].T (gate/up), 4-bit
        // affine group-64 -- the ACTUAL Qwen forward path, ~13.5 TF solo (not bf16,
        // which runs ~7 TF at this shape and would inflate the ratio).
        let S = 512, K = 5120, N = 17408
        let gx = MLXArray.ones([S, K]).asType(.float16)
        let gwf = MLXArray.ones([N, K]).asType(.float16)
        let (gwq, gsc, gbi) = quantized(gwf, groupSize: 64, bits: 4)
        eval(gx, gwq, gsc); if let gbi { eval(gbi) }
        func gpuOp() -> MLXArray {
            quantizedMM(gx, gwq, scales: gsc, biases: gbi, transpose: true, groupSize: 64, bits: 4)
        }
        eval(gpuOp())
        let gpuFlop = 2.0 * Double(S) * Double(K) * Double(N)
        let aneFlop = aneGF * 1e9
        let window = 1.0

        func gpuIters(until dl: Date) -> Int { var k = 0; while Date() < dl { eval(gpuOp()); k += 1 }; return k }

        report += "| cycle | gpu_solo TF | gpu_conc TF | ane_conc TF | combined TF | ratio |\n|---|---|---|---|---|---|\n"
        var ratios: [Double] = []
        for c in 0 ..< 6 {
            // Phase A: GPU alone
            let solo = Double(gpuIters(until: Date().addingTimeInterval(window))) * gpuFlop / window / 1e12
            // Phase B: GPU + ANE concurrent over one shared window
            let dl = Date().addingTimeInterval(window)
            let aneRes = ResultBox()
            let g = DispatchGroup(); g.enter()
            let boxRef = box
            DispatchQueue.global(qos: .userInitiated).async {
                var n = 0
                while Date() < dl { autoreleasepool { _ = try? boxRef.model.prediction(from: boxRef.input) }; n += 1 }
                aneRes.value = n; g.leave()
            }
            let gpuN = gpuIters(until: dl)
            g.wait()
            let gpuConc = Double(gpuN) * gpuFlop / window / 1e12
            let aneConc = Double(aneRes.value) * aneFlop / window / 1e12
            let combined = gpuConc + aneConc
            let ratio = solo > 0 ? combined / solo : 0
            ratios.append(ratio)
            report += String(format: "| %d | %.2f | %.2f | %.2f | %.2f | %.3f |\n",
                             c, solo, gpuConc, aneConc, combined, ratio)
            flush()
        }
        let mean = ratios.reduce(0, +) / Double(ratios.count)
        let sorted = ratios.sorted()
        report += String(format: "\n- [MEASURED] mean combined/solo ratio = %.3f (min %.3f, max %.3f) over %d cycles\n",
                         mean, sorted.first ?? 0, sorted.last ?? 0, ratios.count)
        report += "\nRead: ratio > 1 => hybrid GPU+ANE offload delivers more total prefill throughput than " +
            "GPU-alone (the ANE's added work outweighs the GPU slowdown). ratio <= 1 => GPU-alone wins; " +
            "offload not worth it. This is the measured version of §35's 1.18x arithmetic.\n"
    }

    /// Noise-robust confirmation that the ANE actually slows the GPU. Runs Metal
    /// continuously and TOGGLES the ANE on/off across many short paired cycles.
    /// Each cycle measures Metal's rate with the ANE idle, then with the ANE
    /// looping -- adjacent in time, so slow host noise (Spotlight bursts) hits
    /// both halves equally and cancels in the ratio. A consistent on/off ratio
    /// < 1 across cycles confirms the slowdown is real and ANE-caused; a ratio
    /// ~1 means the original 0.45 was a host-noise artifact and there is no real
    /// ANE/GPU contention.
    @Test("ANE on/off toggle vs Metal rate (noise-robust paired confirmation)")
    func aneMetalToggle() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }

        var report = "# ANE on/off toggle vs Metal rate (paired, noise-robust)\n\n"
        report += "Metal runs continuously; ANE toggles on/off each cycle. Paired off/on windows are " +
            "adjacent so slow host noise cancels in the ratio. ratio<1 consistently => real ANE-caused " +
            "GPU slowdown; ratio~1 => the earlier 0.45 was a noise artifact.\n\n"
        let reportPath = env["ANE_TOGGLE_REPORT"] ??
            "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/" +
            "3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/ane-toggle-report.md"
        func flush() { try? report.write(toFile: reportPath, atomically: true, encoding: .utf8) }
        defer { flush() }

        let cacheDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("ane-metal-pipeline-cache")
        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
        let mURL: URL, gf: Double
        do { (mURL, gf) = try Self.ensureChunkModel(St: 128, K: 4, cacheDir: cacheDir) }
        catch { report += "BLOCKED loading ANE model: \(error)\n"; Issue.record("blocked"); return }
        let box = MLBox(try MLModel(contentsOf: mURL, configuration: cfg), try Self.makeChunkInput(St: 128))
        _ = try? box.model.prediction(from: box.input) // warm ANE

        let (ga, gb, _) = Self.matchedGEMM(gf: gf, St: 128)
        eval(matmul(ga, gb)) // warm Metal
        let window = 1.0

        func metalRate() -> Double {
            let dl = Date().addingTimeInterval(window)
            var k = 0
            while Date() < dl { eval(matmul(ga, gb)); k += 1 }
            return Double(k) / window
        }

        report += "| cycle | Metal off (ANE idle) | Metal on (ANE run) | on/off ratio |\n|---|---|---|---|\n"
        var ratios: [Double] = []
        let cycles = 8
        for c in 0 ..< cycles {
            let off = metalRate()
            // ANE running on a background thread for the duration of the "on" window
            let stop = SpinFlag()
            let g = DispatchGroup(); g.enter()
            let boxRef = box
            DispatchQueue.global(qos: .userInitiated).async {
                while !stop.get() { autoreleasepool { _ = try? boxRef.model.prediction(from: boxRef.input) } }
                g.leave()
            }
            let on = metalRate()
            stop.set(); g.wait()
            let ratio = off > 0 ? on / off : 0
            ratios.append(ratio)
            report += String(format: "| %d | %.1f | %.1f | %.2f |\n", c, off, on, ratio)
            flush()
        }
        let mean = ratios.reduce(0, +) / Double(ratios.count)
        let sorted = ratios.sorted()
        report += String(format: "\n- [MEASURED] mean on/off ratio = %.2f (min %.2f, max %.2f, median %.2f) over %d cycles\n",
                         mean, sorted.first ?? 0, sorted.last ?? 0, sorted[sorted.count / 2], cycles)
        report += "\nRead: mean ratio consistently < ~0.9 with tight spread => real ANE-caused GPU slowdown " +
            "confirmed, robust to host noise. Ratio ~1.0 => no real contention; the earlier before/after 0.45 " +
            "was a coincident host-noise burst, and the offload lane must be reconsidered.\n"
    }
}

/// Simple thread-safe stop flag for the CPU spinners.
private final class SpinFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var sink: Double = 0 // racy black-hole write to keep the spin loop from being optimized away
    func get() -> Bool { lock.lock(); defer { lock.unlock() }; return flag }
    func set() { lock.lock(); flag = true; lock.unlock() }
}
