import Foundation
import MLX

/// Prepares MLX 4-bit affine-quantized projection weights for the ANE
/// (fp16) side of a concurrent ANE/GPU offload split.
///
/// The ANE primitive needs a plain fp16 `[out, in]` weight (or a
/// output-channel slice `[F, in]`), dequantized from the model's shipped
/// MLX 4-bit affine group-64 weights. This must match the GPU
/// `quantizedMM` path bit-for-bit, so it calls MLX's own `dequantized(...)`
/// op rather than hand-rolling the affine dequant math.
public enum ANEWeightPrep {
    /// Dequantize MLX 4-bit (wq,scales,biases,groupSize:64,bits:4) to fp16 [out,in],
    /// optionally only output-channel rows [channelStart..<channelEnd].
    public static func dequantizeFP16(
        wq: MLXArray, scales: MLXArray, biases: MLXArray,
        channelStart: Int, channelEnd: Int
    ) -> MLXArray {
        let full = dequantized(wq, scales: scales, biases: biases, groupSize: 64, bits: 4)
        return full[channelStart ..< channelEnd, 0...].asType(.float16)
    }

    /// Dequantize MLX 4-bit (wq,scales,biases,groupSize:64,bits:4) to fp16 [out,in],
    /// restricted to an INPUT-channel (column) range [columnStart..<columnEnd].
    /// Unlike the packed 4-bit operand, `dequantized(...)` returns the full
    /// logical `[out, in]` array, so this is a plain column slice -- no
    /// packed-axis arithmetic needed.
    public static func dequantizeFP16Columns(
        wq: MLXArray, scales: MLXArray, biases: MLXArray,
        columnStart: Int, columnEnd: Int
    ) -> MLXArray {
        let full = dequantized(wq, scales: scales, biases: biases, groupSize: 64, bits: 4)
        return full[0..., columnStart ..< columnEnd].asType(.float16)
    }
}

/// Re-quantizers for the ANE's two compressed weight forms. Both take a dense
/// `[out, in]` weight (any float dtype, typically the dequantized GPU tensor)
/// and return the byte chunks a `buildMultiWeightBlob` blob carries. Every
/// step runs as MLX ops on the GPU; only the finished bytes come to the host.
public enum ANEWeightQuant {
    /// Per-output-channel symmetric int8: `scale[o] = max|w[o,:]| / 127`,
    /// `q = clamp(round(w / scale), -127, 127)`. Returns the row-major int8
    /// bytes and the fp16 `[out]` scale bytes.
    public static func int8PerChannel(_ w: MLXArray) -> (data: Data, scale: Data) {
        let w32 = w.asType(.float32)
        let s = maximum(MLX.abs(w32).max(axis: 1, keepDims: true) / 127, MLXArray(Float(1e-8)))
        let q = clip(round(w32 / s), min: -127, max: 127).asType(.int8)
        let s16 = s.squeezed(axis: 1).asType(.float16)
        eval(q, s16)
        return (q.asData().data, s16.asData().data)
    }

    /// The Lloyd-Max 16-level quantizer of a unit Gaussian, the starting point
    /// for the codebook fit.
    private static let gaussian16: [Float] = [
        -2.733, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128,
        0.128, 0.388, 0.657, 0.942, 1.256, 1.618, 2.069, 2.733,
    ]

    /// Per-row RMS, the per-channel scale the palette form applies on the conv
    /// output. `[out, 1]` float32.
    private static func rowScale(_ w32: MLXArray) -> MLXArray {
        maximum(sqrt((w32 * w32).mean(axis: 1, keepDims: true)), MLXArray(Float(1e-8)))
    }

    /// Fits one 16-entry codebook to the pooled, per-row-normalized values of
    /// `weights` by ten Lloyd iterations on a strided sample of at most 2^18
    /// values per weight. Returns the sorted centroids.
    public static func fitCodebook(_ weights: [MLXArray]) -> [Float] {
        var samples: [MLXArray] = []
        for w in weights {
            let w32 = w.asType(.float32)
            let n = (w32 / rowScale(w32)).flattened()
            let total = n.dim(0)
            let stride = max(1, total / (1 << 18))
            let usable = (total / stride) * stride
            samples.append(n[0 ..< usable].reshaped([-1, stride])[0..., 0])
        }
        let sample = concatenated(samples, axis: 0)  // [m]
        var c = MLXArray(gaussian16)  // [16]
        let idx = MLXArray(Int32(0) ..< Int32(16))
        for _ in 0 ..< 10 {
            let dist = MLX.abs(sample.expandedDimensions(axis: 1) - c.expandedDimensions(axis: 0))  // [m,16]
            let assign = argMin(dist, axis: 1)  // [m] int32
            let oh = (assign.expandedDimensions(axis: 1) .== idx.expandedDimensions(axis: 0)).asType(.float32)  // [m,16]
            let sums = (oh * sample.expandedDimensions(axis: 1)).sum(axis: 0)
            let counts = oh.sum(axis: 0)
            c = MLX.where(counts .> 0, sums / maximum(counts, MLXArray(Float(1))), c)
            c = sorted(c)
            eval(c)
        }
        return c.asArray(Float.self)
    }

    /// int4 palette with a per-channel output scale: each row is divided by
    /// its RMS, every value takes the code of the nearest codebook centroid,
    /// and the codes are packed two per byte, low nibble first, over the
    /// row-major `[out, in]` weight. Returns the packed indices, the fp16
    /// `[out]` scale bytes and the fp16 16-entry LUT bytes.
    public static func int4Palette(_ w: MLXArray, codebook: [Float]) -> (indices: Data, scale: Data, lut: Data) {
        precondition(codebook.count == 16 && w.dim(1) % 2 == 0)
        let w32 = w.asType(.float32)
        let s = rowScale(w32)
        let c = MLXArray(codebook)
        // Codes from the 15 sorted midpoints: code = number of midpoints below the value.
        var mids: [Float] = []
        for i in 0 ..< 15 { mids.append((codebook[i] + codebook[i + 1]) / 2) }
        let bounds = MLXArray(mids)  // [15]
        let rows = w32.dim(0), cols = w32.dim(1)
        let block = max(1, min(rows, (64 << 20) / max(1, cols * 15)))  // ~64 MB of bool per block
        var packedParts: [MLXArray] = []
        var start = 0
        while start < rows {
            let end = min(rows, start + block)
            let n = w32[start ..< end] / s[start ..< end]  // [b, cols]
            let codes = (n.expandedDimensions(axis: 2) .> bounds.reshaped([1, 1, 15])).sum(axis: 2).asType(.uint8)  // [b, cols]
            let pairs = codes.reshaped([end - start, cols / 2, 2])
            let packed = pairs[0..., 0..., 0] + pairs[0..., 0..., 1] * MLXArray(UInt8(16))
            eval(packed)
            packedParts.append(packed)
            start = end
        }
        let packed = concatenated(packedParts, axis: 0)
        let s16 = s.squeezed(axis: 1).asType(.float16)
        let lut16 = c.asType(.float16)
        eval(packed, s16, lut16)
        return (packed.asData().data, s16.asData().data, lut16.asData().data)
    }
}
