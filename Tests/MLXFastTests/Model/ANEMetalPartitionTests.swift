import CoreML
import Foundation
import MLX
import MLXRandom
import Testing

/// Boxes the Core ML handles so they can cross onto a dispatch queue.
private final class MLBox: @unchecked Sendable {
    let model: MLModel
    let input: MLFeatureProvider
    init(_ m: MLModel, _ i: MLFeatureProvider) { model = m; input = i }
}

/// A REAL partitioned projection: one affine 4-bit group-64 GEMM split by
/// output columns across Metal and the ANE, run concurrently, joined, and
/// checked against a Metal-only reference.
///
/// `cachedUInt4Partition` is the honest form of the comparison: weights stay
/// UINT4 in the model (dequantized at runtime by
/// constexpr_blockwise_shift_scale), and the Core ML model is compiled to
/// .mlmodelc ON DISK once and reused. Compilation is a build-time artifact
/// step like the existing weight transform, so charging it to inference was
/// wrong.
@Suite(.serialized)
struct ANEMetalPartitionTests {

    enum Enc {
        static func varint(_ v: UInt64) -> Data {
            var v = v, out = Data()
            repeat { var b = UInt8(v & 0x7F); v >>= 7; if v != 0 { b |= 0x80 }; out.append(b) } while v != 0
            return out
        }
        static func tag(_ f: Int, _ w: UInt8) -> Data { varint(UInt64(f) << 3 | UInt64(w)) }
        static func lenF(_ f: Int, _ p: Data) -> Data { tag(f, 2) + varint(UInt64(p.count)) + p }
        static func varF(_ f: Int, _ v: UInt64) -> Data { tag(f, 0) + varint(v) }
        static func strF(_ f: Int, _ s: String) -> Data { lenF(f, Data(s.utf8)) }
        static func mapEntry(_ f: Int, key: String, value: Data) -> Data {
            lenF(f, strF(1, key) + lenF(2, value))
        }
        static func tensorType(_ dt: UInt64, _ shape: [Int]) -> Data {
            var d = varF(1, dt) + varF(2, UInt64(shape.count))
            for s in shape { d += lenF(3, lenF(1, varF(1, UInt64(s)))) }
            return d
        }
        static func valueType(_ dt: UInt64, _ shape: [Int]) -> Data { lenF(1, tensorType(dt, shape)) }
        static func tensorValueBytes(_ dt: UInt64, _ shape: [Int], _ payload: Data) -> Data {
            lenF(2, valueType(dt, shape)) + lenF(3, lenF(1, lenF(7, lenF(1, payload))))
        }
        static func stringValue(_ s: String) -> Data {
            lenF(2, valueType(2, [])) + lenF(3, lenF(1, lenF(4, strF(1, s))))
        }
        static func namedValue(_ n: String, _ dt: UInt64, _ shape: [Int]) -> Data {
            strF(1, n) + lenF(2, valueType(dt, shape))
        }
        static func constOp(_ name: String, _ dt: UInt64, _ shape: [Int], _ payload: Data) -> Data {
            strF(1, "const") + lenF(3, namedValue(name, dt, shape))
                + mapEntry(5, key: "name", value: stringValue(name))
                + mapEntry(5, key: "val", value: tensorValueBytes(dt, shape, payload))
        }
        static func constInts(_ name: String, _ values: [Int]) -> Data {
            let payload = values.reduce(Data()) { $0 + varint(UInt64($1)) }
            let shape = values.count == 1 ? [] : [values.count]
            let val = lenF(2, valueType(23, shape)) + lenF(3, lenF(1, lenF(2, lenF(1, payload))))
            return strF(1, "const") + lenF(3, namedValue(name, 23, shape))
                + mapEntry(5, key: "name", value: stringValue(name))
                + mapEntry(5, key: "val", value: val)
        }
        static func constString(_ name: String, _ value: String) -> Data {
            strF(1, "const") + lenF(3, namedValue(name, 2, []))
                + mapEntry(5, key: "name", value: stringValue(name))
                + mapEntry(5, key: "val", value: stringValue(value))
        }
        static func op(_ type: String, _ name: String, _ inputs: [(String, String)],
                       _ outName: String, _ outShape: [Int]) -> Data {
            var d = strF(1, type)
            for (p, v) in inputs { d += mapEntry(2, key: p, value: lenF(1, strF(1, v))) }
            d += lenF(3, namedValue(outName, 10, outShape))
            d += mapEntry(5, key: "name", value: stringValue(name))
            return d
        }
        private static func featureDesc(_ n: String, _ shape: [Int]) -> Data {
            let arr = lenF(1, shape.reduce(Data()) { $0 + varint(UInt64($1)) }) + varF(2, 65552)
            return strF(1, n) + lenF(3, lenF(5, arr))
        }
        private static func wrap(_ M: Int, _ K: Int, _ N: Int, _ ops: Data) -> Data {
            let block = strF(2, "y") + ops
            let fn = lenF(1, namedValue("x", 10, [1, K, 1, M])) + strF(2, "CoreML8")
                + mapEntry(3, key: "CoreML8", value: block)
            let program = varF(1, 1) + mapEntry(2, key: "main", value: fn)
            let desc = lenF(1, featureDesc("x", [1, K, 1, M])) + lenF(10, featureDesc("y", [1, N, 1, M]))
            return varF(1, 9) + lenF(2, desc) + lenF(502, program)
        }
        private static func convHyper() -> Data {
            lenF(3, constInts("st", [1, 1])) + lenF(3, constInts("dl", [1, 1]))
                + lenF(3, constInts("pd", [0, 0, 0, 0])) + lenF(3, constInts("gp", [1]))
                + lenF(3, constString("pt", "valid"))
        }
        private static func convOp(_ M: Int, _ N: Int) -> Data {
            lenF(3, op("conv", "cv",
                       [("x", "x"), ("weight", "w"), ("strides", "st"), ("pad_type", "pt"),
                        ("pad", "pd"), ("dilations", "dl"), ("groups", "gp")],
                       "y", [1, N, 1, M]))
        }
        /// fp16 weights baked in.
        static func convModel(M: Int, K: Int, N: Int, weightsFP16: Data) -> Data {
            wrap(M, K, N, convHyper() + lenF(3, constOp("w", 10, [N, K, 1, 1], weightsFP16)) + convOp(M, N))
        }
        /// Weights stay UINT4; constexpr_blockwise_shift_scale expands at runtime.
        /// MIL: out = scale * (data - offset). MLX affine: w = scale*q + bias,
        /// so offset = -bias/scale. Block size is implied by data.shape/scale.shape.
        static func convModelUInt4(M: Int, K: Int, N: Int, group: Int,
                                   packed: Data, scaleFP16: Data, offsetFP16: Data) -> Data {
            let gk = K / group
            let ops = convHyper()
                + lenF(3, constOp("wq", 35, [N, K, 1, 1], packed))
                + lenF(3, constOp("ws", 10, [N, gk, 1, 1], scaleFP16))
                + lenF(3, constOp("wo", 10, [N, gk, 1, 1], offsetFP16))
                + lenF(3, op("constexpr_blockwise_shift_scale", "dq",
                             [("data", "wq"), ("scale", "ws"), ("offset", "wo")],
                             "w", [N, K, 1, 1]))
                + convOp(M, N)
            return wrap(M, K, N, ops)
        }
    }

    private static func timeIt(_ n: Int, _ body: () -> Void) -> Double {
        body()
        var best = Double.infinity
        for _ in 0 ..< n {
            let t0 = Date(); body(); best = Swift.min(best, Date().timeIntervalSince(t0))
        }
        return best
    }

    /// Synchronous so a semaphore is legal: dispatch the ANE prediction, run
    /// the Metal half on this thread, join.
    private static func timeSplit(_ box: MLBox, reps: Int, gpu: () -> Void) -> Double {
        var best = Double.infinity
        for _ in 0 ..< reps {
            let t0 = Date()
            let sem = DispatchSemaphore(value: 0)
            DispatchQueue.global(qos: .userInitiated).async {
                _ = try? box.model.prediction(from: box.input)
                sem.signal()
            }
            gpu()
            sem.wait()
            best = Swift.min(best, Date().timeIntervalSince(t0))
        }
        return best
    }

    /// Two 4-bit codes per byte, low nibble first -- the layout MLX uses inside
    /// its uint32 words. If MIL uint4 expects the other order, globalRel blows up.
    private static func packNibbles(_ v: [UInt8]) -> Data {
        var d = Data(count: (v.count + 1) / 2)
        d.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt8.self)
            for i in stride(from: 0, to: v.count, by: 2) {
                p[i / 2] = (v[i] & 0x0F) | (((i + 1 < v.count) ? v[i + 1] : 0) & 0x0F) << 4
            }
        }
        return d
    }

    @Test("gate_up split with a disk-cached uint4 ANE model")
    func cachedUInt4Partition() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let M = Int(env["PART_M"] ?? "1024") ?? 1024
        let K = Int(env["PART_K"] ?? "5120") ?? 5120
        let N = Int(env["PART_N"] ?? "17408") ?? 17408
        let frac = Double(env["PART_ANE_FRAC"] ?? "0.22") ?? 0.22
        let cacheRoot = env["PART_CACHE"] ?? (NSTemporaryDirectory() + "ane-cache")
        var aneN = Int(Double(N) * frac) / 64 * 64
        aneN = Swift.min(aneN, 16384)
        let group = 64, gk = K / group
        MLXRandom.seed(42)   // cached model must match the reference weights

        let x = MLXRandom.normal([1, M, K]).asType(.bfloat16)
        let w = MLXRandom.normal([N, K]).asType(.bfloat16)
        let (wq, scales, biasesOpt) = quantized(w, groupSize: group, bits: 4)
        guard let biases = biasesOpt else { Issue.record("no biases"); return }
        eval(x, wq, scales, biases)
        let reference = quantizedMM(x, wq, scales: scales, biases: biases, transpose: true,
                                    groupSize: group, bits: 4)
        eval(reference)
        let refMs = Self.timeIt(3) {
            eval([quantizedMM(x, wq, scales: scales, biases: biases, transpose: true,
                              groupSize: group, bits: 4)])
        }

        try FileManager.default.createDirectory(atPath: cacheRoot, withIntermediateDirectories: true)
        let key = "gateup_M\(M)_K\(K)_N\(aneN)_u4"
        let compiledURL = URL(fileURLWithPath: cacheRoot).appendingPathComponent("\(key).mlmodelc")

        var compileSec = 0.0
        if !FileManager.default.fileExists(atPath: compiledURL.path) {
            let t0 = Date()
            let ones = MLXArray.ones([aneN, gk]).asType(.bfloat16)
            let zeros = MLXArray.zeros([aneN, gk]).asType(.bfloat16)
            let codes = dequantized(wq[0 ..< aneN], scales: ones, biases: zeros,
                                    groupSize: group, bits: 4).asType(.uint8)
            let s = scales[0 ..< aneN]
            // Exact offset: OffsetT allows fp16, so no rounding into 0..15.
            let off = -biases[0 ..< aneN] / s
            eval(codes, s, off)
            let spec = Enc.convModelUInt4(
                M: M, K: K, N: aneN, group: group,
                packed: Self.packNibbles([UInt8](codes.asData().data)),
                scaleFP16: Data(s.asType(.float16).asData().data),
                offsetFP16: Data(off.asType(.float16).asData().data))
            let specURL = URL(fileURLWithPath: cacheRoot).appendingPathComponent("\(key).mlmodel")
            try spec.write(to: specURL)
            let tmp = try await MLModel.compileModel(at: specURL)
            try? FileManager.default.removeItem(at: compiledURL)
            try FileManager.default.moveItem(at: tmp, to: compiledURL)
            compileSec = Date().timeIntervalSince(t0)
        }

        let loadStart = Date()
        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
        let model = try MLModel(contentsOf: compiledURL, configuration: cfg)
        let loadSec = Date().timeIntervalSince(loadStart)

        let xT = x.reshaped([M, K]).transposed(1, 0).asType(.float16)
        eval(xT)
        guard let xArr = try? MLMultiArray(shape: [1, K, 1, M].map { NSNumber(value: $0) },
                                           dataType: .float16) else { Issue.record("alloc"); return }
        let xBytes = xT.asData().data
        xArr.withUnsafeMutableBytes { raw, _ in
            _ = xBytes.withUnsafeBytes { src in
                memcpy(raw.baseAddress!, src.baseAddress!, Swift.min(raw.count, xBytes.count))
            }
        }
        let input = try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: xArr)])
        _ = try await model.prediction(from: input)

        let gpuRows = wq[aneN ..< N], gpuScales = scales[aneN ..< N], gpuBiases = biases[aneN ..< N]
        let box = MLBox(model, input)
        let bestBoth = Self.timeSplit(box, reps: 4) {
            eval([quantizedMM(x, gpuRows, scales: gpuScales, biases: gpuBiases,
                              transpose: true, groupSize: group, bits: 4)])
        }

        let anePred = try await model.prediction(from: input)
        guard let aneArr = anePred.featureValue(for: "y")?.multiArrayValue else {
            Issue.record("no ANE output"); return
        }
        let aneBytes = aneArr.withUnsafeBytes { raw in Data(bytes: raw.baseAddress!, count: raw.count) }
        let aneOut = MLXArray(aneBytes, [aneN, M], type: Float16.self)
            .transposed(1, 0).reshaped([1, M, aneN]).asType(.bfloat16)
        let gpuOut = quantizedMM(x, gpuRows, scales: gpuScales, biases: gpuBiases,
                                 transpose: true, groupSize: group, bits: 4)
        let joined = concatenated([aneOut, gpuOut], axis: 2)
        eval(joined)
        let globalRel = abs(joined - reference).max().item(Float.self)
            / Swift.max(abs(reference).max().item(Float.self), 1e-6)
        let gpuOnlyMax = abs(gpuOut - reference[0..., 0..., aneN ..< N]).max().item(Float.self)

        let flops = 2.0 * Double(M) * Double(N) * Double(K)
        print(String(format:
            "CACHED\tK=%d N=%d aneN=%d\tcompile=%.2fs\tload=%.3fs\tmetal=%.2fms (%.2f TF)\tsplit=%.2fms (%.2f TF)\tspeedup=%.3f\tglobalRel=%.5f\tgpuExact=%.1e",
            K, N, aneN, compileSec, loadSec, 1000 * refMs, flops / refMs / 1e12,
            1000 * bestBoth, flops / bestBoth / 1e12, refMs / bestBoth, globalRel, gpuOnlyMax))
    }
}
