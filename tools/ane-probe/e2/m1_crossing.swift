import CoreML
import CoreVideo
import Foundation

/// Native Core ML round-trip timing for a [1, 1024, 5120] fp16 tensor.
///
/// Paths:
///   copy        — MLMultiArray(shape:) + memcpy into it
///   dataPointer — initWithDataPointer wrapping already-allocated memory
///   iosurface   — IOSurface-backed CVPixelBuffer wrapped by MLMultiArray
///
/// Each path times construction, predict, and output extraction separately,
/// plus a full round-trip. Repeated predicts on one loaded model vs reload.

struct Stats: Codable {
    let n: Int
    let best_ms: Double
    let median_ms: Double
    let mean_ms: Double
    let p90_ms: Double
    let worst_ms: Double
    let samples_ms: [Double]
}

struct PathReport: Codable {
    let path: String
    let construction: Stats?
    let predict_reused_input: Stats?
    let output_extract_memcpy: Stats?
    let round_trip: Stats?
    let notes: [String]
}

struct Report: Codable {
    let model: String
    let compute_units: String
    let shape: [Int]
    let nbytes: Int
    let load_s: Double
    let first_predict_ms: Double?
    let memcpy_baseline: Stats
    let reload_plus_one_predict: Stats?
    let paths: [PathReport]
    let errors: [String]
}

let shapeNums: [NSNumber] = [1 as NSNumber, 1024 as NSNumber, 5120 as NSNumber]
let shapeInts = [1, 1024, 5120]
let strides: [NSNumber] = [(1024 * 5120) as NSNumber, 5120 as NSNumber, 1 as NSNumber]
let count = 1 * 1024 * 5120
let nbytes = count * 2
let warmup = 8
let timed = 80

func nowNs() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }

func stats(from samplesMs: [Double]) -> Stats {
    let s = samplesMs.sorted()
    let n = s.count
    let median = s[n / 2]
    let p90 = s[Int(0.9 * Double(n - 1))]
    let mean = s.reduce(0, +) / Double(n)
    return Stats(
        n: n,
        best_ms: s[0],
        median_ms: median,
        mean_ms: mean,
        p90_ms: p90,
        worst_ms: s[n - 1],
        samples_ms: samplesMs
    )
}

func timeLoop(_ n: Int, warmup nWarm: Int, body: () throws -> Void) throws -> Stats {
    for _ in 0..<nWarm { try body() }
    var samples: [Double] = []
    samples.reserveCapacity(n)
    for _ in 0..<n {
        let t0 = nowNs()
        try body()
        samples.append(Double(nowNs() - t0) / 1_000_000.0)
    }
    return stats(from: samples)
}

func fillRandomFp16(_ ptr: UnsafeMutableRawPointer, count: Int, seed: UInt64) {
    var state = seed &+ 0x9E3779B97F4A7C15
    let dst = ptr.bindMemory(to: UInt16.self, capacity: count)
    for i in 0..<count {
        state = state &* 6364136223846793005 &+ 1
        // Pack a small fp16 value in (0.5, 1.5) so the graph is not all-zero.
        let u = UInt16((state >> 33) & 0x03FF) // 10 mantissa bits
        dst[i] = 0x3800 | u // exponent 14 (value ~0.5-1.5)
    }
}

func memcpyExtract(_ array: MLMultiArray, into dst: UnsafeMutableRawPointer) {
    _ = array.withUnsafeBytes { raw in
        memcpy(dst, raw.baseAddress!, min(nbytes, raw.count))
    }
}

func makeCopyArray(from src: UnsafeRawPointer) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shapeNums, dataType: .float16)
    _ = array.withUnsafeMutableBytes { buf, _ in
        memcpy(buf.baseAddress!, src, min(nbytes, buf.count))
    }
    return array
}

func makeWrappedArray(pointer: UnsafeMutableRawPointer) throws -> MLMultiArray {
    try MLMultiArray(
        dataPointer: pointer,
        shape: shapeNums,
        dataType: .float16,
        strides: strides,
        deallocator: nil
    )
}

func makeIOSurfaceArray(from src: UnsafeRawPointer) throws -> (MLMultiArray, CVPixelBuffer) {
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        kCVPixelBufferWidthKey as String: 5120,
        kCVPixelBufferHeightKey as String: 1024,
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_OneComponent16Half,
    ]
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        5120,
        1024,
        kCVPixelFormatType_OneComponent16Half,
        attrs as CFDictionary,
        &pb
    )
    guard status == kCVReturnSuccess, let pixelBuffer = pb else {
        throw NSError(
            domain: "e2.m1",
            code: Int(status),
            userInfo: [NSLocalizedDescriptionKey: "CVPixelBufferCreate failed \(status)"]
        )
    }
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
    let base = CVPixelBufferGetBaseAddress(pixelBuffer)!
    let bpr = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let srcBpr = 5120 * 2
    if bpr == srcBpr {
        memcpy(base, src, nbytes)
    } else {
        for row in 0..<1024 {
            memcpy(base.advanced(by: row * bpr), src.advanced(by: row * srcBpr), srcBpr)
        }
    }
    let array = MLMultiArray(pixelBuffer: pixelBuffer, shape: shapeNums)
    return (array, pixelBuffer)
}

func predict(_ model: MLModel, array: MLMultiArray) throws -> MLMultiArray {
    let value = MLFeatureValue(multiArray: array)
    let provider = try MLDictionaryFeatureProvider(dictionary: ["hidden_states": value])
    let out = try model.prediction(from: provider)
    guard let result = out.featureValue(for: "output_hidden")?.multiArrayValue else {
        let names = out.featureNames
        let first = names.first.flatMap { out.featureValue(for: $0)?.multiArrayValue }
        if let first { return first }
        throw NSError(
            domain: "e2.m1",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "no multiarray output, names=\(names)"]
        )
    }
    return result
}

func usageAndExit() -> Never {
    fputs("usage: m1_crossing <mlpackage> <out.json>\n", stderr)
    exit(2)
}

guard CommandLine.arguments.count >= 3 else { usageAndExit() }
let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
let outURL = URL(fileURLWithPath: CommandLine.arguments[2])

var errors: [String] = []
let src = UnsafeMutableRawPointer.allocate(byteCount: nbytes, alignment: 64)
defer { src.deallocate() }
fillRandomFp16(src, count: count, seed: 42)
let extractBuf = UnsafeMutableRawPointer.allocate(byteCount: nbytes, alignment: 64)
defer { extractBuf.deallocate() }

let memcpyBaseline = try! timeLoop(timed, warmup: warmup) {
    memcpy(extractBuf, src, nbytes)
}

let config = MLModelConfiguration()
config.computeUnits = .cpuAndNeuralEngine
config.allowLowPrecisionAccumulationOnGPU = false

let loadT0 = nowNs()
let compiledURL: URL
do {
    compiledURL = try MLModel.compileModel(at: modelURL)
} catch {
    // mlpackage may already be loadable.
    compiledURL = modelURL
}
let model: MLModel
do {
    model = try MLModel(contentsOf: compiledURL, configuration: config)
} catch {
    let report = Report(
        model: modelURL.path,
        compute_units: "cpuAndNeuralEngine",
        shape: shapeInts,
        nbytes: nbytes,
        load_s: Double(nowNs() - loadT0) / 1_000_000_000.0,
        first_predict_ms: nil,
        memcpy_baseline: memcpyBaseline,
        reload_plus_one_predict: nil,
        paths: [],
        errors: ["model load failed: \(error)"]
    )
    let data = try JSONEncoder().encode(report)
    try data.write(to: outURL)
    fputs("model load failed: \(error)\n", stderr)
    exit(1)
}
let loadS = Double(nowNs() - loadT0) / 1_000_000_000.0
print(String(format: "loaded in %.3fs", loadS))

var firstPredictMs: Double? = nil
var paths: [PathReport] = []

func runPath(
    name: String,
    construct: () throws -> MLMultiArray,
    keepAlive: Any? = nil
) -> PathReport {
    _ = keepAlive
    var notes: [String] = []
    var constructed: MLMultiArray?
    do {
        constructed = try construct()
    } catch {
        return PathReport(
            path: name,
            construction: nil,
            predict_reused_input: nil,
            output_extract_memcpy: nil,
            round_trip: nil,
            notes: ["construct failed: \(error)"]
        )
    }
    guard let input0 = constructed else {
        return PathReport(
            path: name, construction: nil, predict_reused_input: nil,
            output_extract_memcpy: nil, round_trip: nil,
            notes: ["construct returned nil"]
        )
    }

    if name == "copy" || firstPredictMs == nil {
        let t0 = nowNs()
        do {
            _ = try predict(model, array: input0)
            firstPredictMs = Double(nowNs() - t0) / 1_000_000.0
            print(String(format: "first predict (%@) %.3f ms", name, firstPredictMs!))
        } catch {
            notes.append("first predict failed: \(error)")
        }
    }

    var constructionStats: Stats?
    do {
        constructionStats = try timeLoop(timed, warmup: warmup) {
            _ = try construct()
        }
        print(
            String(
                format: "%@ construction best=%.3f median=%.3f ms n=%d",
                name,
                constructionStats!.best_ms,
                constructionStats!.median_ms,
                constructionStats!.n
            )
        )
    } catch {
        notes.append("construction timing failed: \(error)")
    }

    var predictStats: Stats?
    do {
        let reused = try construct()
        predictStats = try timeLoop(timed, warmup: warmup) {
            _ = try predict(model, array: reused)
        }
        print(
            String(
                format: "%@ predict-reused best=%.3f median=%.3f ms n=%d",
                name,
                predictStats!.best_ms,
                predictStats!.median_ms,
                predictStats!.n
            )
        )
    } catch {
        notes.append("predict timing failed: \(error)")
    }

    var extractStats: Stats?
    do {
        let reused = try construct()
        let output = try predict(model, array: reused)
        extractStats = try timeLoop(timed, warmup: warmup) {
            memcpyExtract(output, into: extractBuf)
        }
        print(
            String(
                format: "%@ extract best=%.3f median=%.3f ms n=%d",
                name,
                extractStats!.best_ms,
                extractStats!.median_ms,
                extractStats!.n
            )
        )
        notes.append(
            "output shape=\(output.shape) strides=\(output.strides) pixelBuffer=\(output.pixelBuffer != nil)"
        )
    } catch {
        notes.append("extract timing failed: \(error)")
    }

    var roundTripStats: Stats?
    do {
        roundTripStats = try timeLoop(timed, warmup: warmup) {
            let input = try construct()
            let output = try predict(model, array: input)
            memcpyExtract(output, into: extractBuf)
        }
        print(
            String(
                format: "%@ round-trip best=%.3f median=%.3f ms n=%d",
                name,
                roundTripStats!.best_ms,
                roundTripStats!.median_ms,
                roundTripStats!.n
            )
        )
    } catch {
        notes.append("round-trip timing failed: \(error)")
    }

    return PathReport(
        path: name,
        construction: constructionStats,
        predict_reused_input: predictStats,
        output_extract_memcpy: extractStats,
        round_trip: roundTripStats,
        notes: notes
    )
}

paths.append(runPath(name: "copy", construct: { try makeCopyArray(from: src) }))
paths.append(runPath(name: "dataPointer", construct: { try makeWrappedArray(pointer: src) }))

// Hold the pixel buffer so it is not released while MLMultiArray is used.
var held: CVPixelBuffer?
let iosurfacePath = runPath(name: "iosurface", construct: {
    let pair = try makeIOSurfaceArray(from: src)
    held = pair.1
    return pair.0
}, keepAlive: held as Any)
paths.append(iosurfacePath)

var reloadStats: Stats? = nil
do {
    // Reload is expensive; fewer samples.
    let nReload = 8
    var samples: [Double] = []
    for i in 0..<nReload {
        let t0 = nowNs()
        let compiled = try MLModel.compileModel(at: modelURL)
        let m = try MLModel(contentsOf: compiled, configuration: config)
        let input = try makeWrappedArray(pointer: src)
        _ = try predict(m, array: input)
        samples.append(Double(nowNs() - t0) / 1_000_000.0)
        print(String(format: "reload+predict[%d] %.1f ms", i, samples[i]))
    }
    reloadStats = stats(from: samples)
} catch {
    errors.append("reload path failed: \(error)")
}

let report = Report(
    model: modelURL.path,
    compute_units: "cpuAndNeuralEngine",
    shape: shapeInts,
    nbytes: nbytes,
    load_s: loadS,
    first_predict_ms: firstPredictMs,
    memcpy_baseline: memcpyBaseline,
    reload_plus_one_predict: reloadStats,
    paths: paths,
    errors: errors
)

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(report)
try data.write(to: outURL)
print("wrote \(outURL.path)")
