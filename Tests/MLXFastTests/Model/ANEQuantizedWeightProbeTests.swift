import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXFastModel

/// Hardware confirmation that the ANE consumes COMPRESSED weight storage, not
/// only dense fp16. Stores a 1x1-conv weight as int8 (affine per-channel) and
/// int4 (blockwise group-64) in the MIL program, and dispatches it on the real
/// ANE through the zero-copy direct path. If each compiles, loads, runs, and
/// lands near the fp32 reference (at the quant error, not garbage), the ANE
/// dequantizes the compressed weight on-chip and the "needs bf16" belief is
/// retired in hardware.
///
/// Needs the real ANE. Run with:
///   MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_NO_SANDBOX=1 \
///     swift test -c release --force-resolved-versions \
///     --filter ANEQuantizedWeightProbeTests
@Suite(.serialized)
struct ANEQuantizedWeightProbeTests {
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    /// Per-output-channel symmetric int8: scale[o] = max|w[o,:]| / 127.
    private func quantInt8(_ w: [Float], O: Int, K: Int) -> (bytes: Data, scales: [Float]) {
        var q = [Int8](repeating: 0, count: O * K)
        var scales = [Float](repeating: 0, count: O)
        for o in 0 ..< O {
            var m: Float = 1e-8
            for k in 0 ..< K { m = max(m, abs(w[o * K + k])) }
            let s = m / 127
            scales[o] = s
            for k in 0 ..< K {
                let v = (w[o * K + k] / s).rounded()
                q[o * K + k] = Int8(max(-127, min(127, v)))
            }
        }
        return (q.withUnsafeBytes { Data($0) }, scales)
    }

    /// Blockwise symmetric int4, group `G` along the input axis:
    /// scale[o,g] = max|block| / 7. Two signed nibbles per byte, low first,
    /// over the row-major [O,K] weight.
    private func quantInt4(_ w: [Float], O: Int, K: Int, G: Int) -> (bytes: Data, scales: [Float]) {
        let groups = K / G
        var scales = [Float](repeating: 0, count: O * groups)
        var nib = [Int](repeating: 0, count: O * K)
        for o in 0 ..< O {
            for g in 0 ..< groups {
                var m: Float = 1e-8
                for j in 0 ..< G { m = max(m, abs(w[o * K + g * G + j])) }
                let s = m / 7
                scales[o * groups + g] = s
                for j in 0 ..< G {
                    let v = (w[o * K + g * G + j] / s).rounded()
                    nib[o * K + g * G + j] = max(-8, min(7, Int(v)))
                }
            }
        }
        var bytes = [UInt8](repeating: 0, count: (O * K + 1) / 2)
        for i in 0 ..< (O * K) {
            let n = UInt8(nib[i] & 0xF)
            if i % 2 == 0 { bytes[i / 2] = n } else { bytes[i / 2] |= (n << 4) }
        }
        return (Data(bytes), scales)
    }

    private func runANE(text: String, blob: Data, weightFileName: String = "weight_data.bin", inputDim: Int, outputDim: Int, sequenceLength: Int, x: MLXArray) throws -> MLXArray {
        let m = try ANEInMemoryModel(milText: text, weightBlob: blob, weightFileName: weightFileName)
        defer { m.unload() }
        try m.compile()
        try m.load()
        return try ANEDirectDispatch.runConv(model: m, x: x, inputDim: inputDim, outputDim: outputDim, sequenceLength: sequenceLength)
    }

    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = MLX.abs(a.asType(.float32) - b.asType(.float32)).max()
        eval(d)
        return d.item(Float.self)
    }

    @Test("ANE accepts int8 and int4 weight storage", .enabled(if: enabled))
    func quantizedWeights() throws {
        try #require(ANERuntime.available())
        let K = 512, O = 512, S = 128, G = 64
        MLXRandom.seed(1)
        let w = MLXRandom.normal([O, K]).asType(.float16)
        let x = MLXRandom.normal([S, K]).asType(.float16)
        eval(w, x)
        let wf = w.asType(.float32).asArray(Float.self)  // row-major [O*K]
        let ref = matmul(x.asType(.float32), w.asType(.float32).transposed(1, 0))  // [S,O]
        let refMax = MLX.abs(ref).max(); eval(refMax)
        let scale = refMax.item(Float.self)

        // fp16 control.
        do {
            let y = try runANE(
                text: buildConvMILText(inputDim: K, outputDim: O, sequenceLength: S, programTag: "q-fp16-\(UUID().uuidString)"),
                blob: buildConvWeightBlob(f16Bytes(w)), inputDim: K, outputDim: O, sequenceLength: S, x: x)
            print("[q-probe] fp16: OK maxAbs=\(String(format: "%.4f", maxAbs(y, ref))) (ref max \(String(format: "%.1f", scale)))")
        } catch { print("[q-probe] fp16: FAILED \(error)") }

        // int8 affine per-channel.
        do {
            let (b8, sc8) = quantInt8(wf, O: O, K: K)
            let y = try runANE(
                text: buildConvMILTextInt8(inputDim: K, outputDim: O, sequenceLength: S, scales: sc8, programTag: "q-int8-\(UUID().uuidString)"),
                blob: buildConvWeightBlob(b8), inputDim: K, outputDim: O, sequenceLength: S, x: x)
            let e = maxAbs(y, ref)
            print("[q-probe] int8: OK maxAbs=\(String(format: "%.4f", e)) rel=\(String(format: "%.4f", e / scale))")
            #expect(e / scale < 0.1, "int8 ANE result diverged from reference by rel \(e / scale) — not a quant-level error")
        } catch {
            print("[q-probe] int8: FAILED \(error)")
            Issue.record("int8 weight storage not accepted: \(error)")
        }

        // int4 blockwise group-64. Data and scale both via BLOBFILE (a
        // two-chunk blob), matching the coremltools-emitted ground truth.
        do {
            let (b4, sc4) = quantInt4(wf, O: O, K: K, G: G)
            let scaleBytes = f16Bytes(MLXArray(sc4).asType(.float16))
            // int4 data chunk uses blob type 8 (packed 4-bit); fp16 scale uses 1.
            let (blob, offsets) = buildMultiWeightBlob(chunks: [b4, scaleBytes], chunkTypes: [8, 1])
            let y = try runANE(
                text: buildConvMILTextInt4Blockwise(
                    inputDim: K, outputDim: O, sequenceLength: S, groupSize: G,
                    dataOffset: offsets[0], scaleOffset: offsets[1], programTag: "q-int4-\(UUID().uuidString)"),
                blob: blob, weightFileName: "weight.bin", inputDim: K, outputDim: O, sequenceLength: S, x: x)
            let e = maxAbs(y, ref)
            print("[q-probe] int4: OK maxAbs=\(String(format: "%.4f", e)) rel=\(String(format: "%.4f", e / scale))")
            #expect(e / scale < 0.25, "int4 ANE result diverged from reference by rel \(e / scale) — not a quant-level error")
        } catch {
            print("[q-probe] int4: FAILED \(error)")
            Issue.record("int4 weight storage not accepted: \(error)")
        }
    }
}
