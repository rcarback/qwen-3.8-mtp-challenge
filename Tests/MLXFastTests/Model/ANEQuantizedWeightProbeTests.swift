import Foundation
import MLX
import MLXNN
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

    /// Per-tensor uniform 4-bit palette: 16 fp16 centroids `(i-8)*s`, `s =
    /// max|w| / 8`, codes `clamp(round(w/s) + 8, 0, 15)`, packed two per byte
    /// low first over the row-major [O,K] weight.
    private func quantInt4LUT(_ w: [Float], O: Int, K: Int) -> (indices: Data, lut: [Float]) {
        var m: Float = 1e-8
        for v in w { m = max(m, abs(v)) }
        let s = m / 8
        let lut = (0 ..< 16).map { Float($0 - 8) * s }
        var codes = [Int](repeating: 0, count: O * K)
        for i in 0 ..< (O * K) {
            codes[i] = max(0, min(15, Int((w[i] / s).rounded()) + 8))
        }
        var bytes = [UInt8](repeating: 0, count: (O * K + 1) / 2)
        for i in 0 ..< (O * K) {
            let n = UInt8(codes[i] & 0xF)
            if i % 2 == 0 { bytes[i / 2] = n } else { bytes[i / 2] |= (n << 4) }
        }
        return (Data(bytes), lut)
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
            // The iOS18 blockwise op is rejected by the in-memory ANE compiler;
            // recorded, not asserted. The palette arm below is the int4 path.
            print("[q-probe] int4-blockwise (iOS18 op): FAILED \(error)")
        }

        // int4 PALETTE (constexpr_lut_to_dense, iOS16): per-tensor 16-entry
        // LUT, 4-bit packed indices. The ANE-native compressed form (the
        // paper's 2.37x-bandwidth int4), and an iOS16 op the in-memory
        // compiler should accept like affine_dequantize.
        do {
            let (idx4, lut4) = quantInt4LUT(wf, O: O, K: K)
            let lutBytes = f16Bytes(MLXArray(lut4).asType(.float16))
            // indices chunk type 3 (uint8 packed), lut chunk type 1 (fp16).
            let (blob, offsets) = buildMultiWeightBlob(chunks: [idx4, lutBytes], chunkTypes: [3, 1])
            let y = try runANE(
                text: buildConvMILTextInt4LUT(
                    inputDim: K, outputDim: O, sequenceLength: S,
                    indicesOffset: offsets[0], lutOffset: offsets[1], programTag: "q-int4lut-\(UUID().uuidString)"),
                blob: blob, weightFileName: "weight.bin", inputDim: K, outputDim: O, sequenceLength: S, x: x)
            let e = maxAbs(y, ref)
            print("[q-probe] int4-lut: OK maxAbs=\(String(format: "%.4f", e)) rel=\(String(format: "%.4f", e / scale))")
            #expect(e / scale < 0.25, "int4 palette ANE result diverged from reference by rel \(e / scale) — not a quant-level error")
        } catch {
            print("[q-probe] int4-lut: FAILED \(error)")
            Issue.record("int4 palette weight storage not accepted: \(error)")
        }
    }
}

/// The fused SwiGLU-down program in each weight form, on the real ANE through
/// the in-memory path, against an fp32 reference of the ORIGINAL weights: the
/// error each form adds on top of the fp16 compute, and the per-call time at a
/// small shape. This is the unit gate before the dense-tower end-to-end sweep.
@Suite(.serialized)
struct ANEFusedFormProbeTests {
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    @Test("fused SwiGLU-down in fp16, int8 and int4 forms", .enabled(if: enabled))
    func fusedForms() throws {
        try #require(ANERuntime.available())
        let hidden = 1024, inner = 512, S = 256
        MLXRandom.seed(3)
        let gate = (MLXRandom.normal([inner, hidden]) * 0.02).asType(.bfloat16)
        let up = (MLXRandom.normal([inner, hidden]) * 0.02).asType(.bfloat16)
        let down = (MLXRandom.normal([hidden, inner]) * 0.02).asType(.bfloat16)
        let x = MLXRandom.normal([S, hidden]).asType(.float16)
        eval(gate, up, down, x)
        let x32 = x.asType(.float32)
        let g = matmul(x32, gate.asType(.float32).transposed(1, 0))
        let u = matmul(x32, up.asType(.float32).transposed(1, 0))
        let ref = matmul(silu(g) * u, down.asType(.float32).transposed(1, 0))
        let refMax = MLX.abs(ref).max(); eval(refMax)
        for form in ANEWeightForm.allCases {
            do {
                let t0 = DispatchTime.now().uptimeNanoseconds
                let mlp = try ANEFusedMLP(
                    hidden: hidden, innerFraction: inner, sequenceLength: S,
                    gate: gate.asType(.float16), up: up.asType(.float16), down: down.asType(.float16),
                    activation: .expDiv, weightForm: form)
                let buildMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
                let y = try mlp(x)
                let d = MLX.abs(y.asType(.float32) - ref)
                let maxAbs = d.max(); let meanAbs = d.mean(); let meanRef = MLX.abs(ref).mean()
                eval(maxAbs, meanAbs, meanRef)
                var t: [Double] = []
                for _ in 0 ..< 5 { _ = try mlp(x) }
                for _ in 0 ..< 20 {
                    let s = DispatchTime.now().uptimeNanoseconds
                    _ = try mlp(x)
                    t.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6)
                }
                t.sort()
                print(String(format: "[fused-form] %@: OK build %.0f ms | maxAbs %.4f rel %.4f meanRel %.4f | %.3f ms/call",
                             form.rawValue, buildMs, maxAbs.item(Float.self), maxAbs.item(Float.self) / refMax.item(Float.self),
                             meanAbs.item(Float.self) / meanRef.item(Float.self), t[t.count / 2]))
            } catch {
                print("[fused-form] \(form.rawValue): FAILED \(error)")
                Issue.record("fused \(form.rawValue) program failed: \(error)")
            }
        }
    }
}
