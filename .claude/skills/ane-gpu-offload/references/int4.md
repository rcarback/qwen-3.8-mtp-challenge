# int4 weight programs (palette, and the rejected blockwise form)

**Confirmed in hardware, via the palette form.** `ANEQuantizedWeightProbeTests`
int4-lut arm: the ANE compiled, loaded and ran a 4-bit palettized weight
(`constexpr_lut_to_dense`) through the in-memory path at 15.9 percent
relative error, which is the expected error of a per-tensor uniform 16-level
palette on a random weight, not a dispatch fault. The architecture paper
establishes this palette form is the one that streams natively at about 2.37x
fp16 bandwidth. So int4 storage on the ANE is real; what remains is accuracy,
which is a quantizer choice, not a hardware question.

## The two int4 forms and their status

| form | MIL op | vintage | in-memory ANE compile | bandwidth (paper) |
| --- | --- | --- | --- | --- |
| palette (lookup table) | `constexpr_lut_to_dense` | iOS16 | **confirmed** | streams natively, ~2.37x fp16 |
| blockwise affine (group scale) | `constexpr_blockwise_shift_scale` | iOS18 | **rejected**, `InvalidMILProgram`, with a byte-identical blob | affine forms fold to fp16 |

The pattern behind the split: the in-memory ANE compiler accepts the iOS16
constexpr ops (`affine_dequantize`, `lut_to_dense`) and rejects the iOS18 one,
even though coremltools emits it and Core ML compiles it. Use the palette op.
It is also the form with the measured bandwidth win, so this is the right
answer, not a fallback.

## The MIL text (byte-exact, iOS16 conventions, from `buildConvMILTextInt4LUT`)

The palette op is iOS16, and a byte-exact match means using the iOS16 text
conventions throughout the program: `program(1.0)`, `func main<ios16>`,
strings as `tensor<string, []>("...")`, scalars as `tensor<int32, []>(1)`
(NOT the bare `string(...)` / `int32` forms the ios18 programs use). All op
parameters sit inline in the attribute bracket with an empty argument list.

```text
program(1.0)
[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{"coremlc-component-MIL", "3520.4.1"}, {"coremlc-version", "3520.5.1"}, {"mlxfast-program-tag", "<unique-tag>"}})]
{
    func main<ios16>(tensor<fp16, [1, IN, 1, S]> x) {
        tensor<fp16, [OUT, IN, 1, 1]> w = constexpr_lut_to_dense()[indices = tensor<uint8, [OUT*IN/2]>(BLOBFILE(path = tensor<string, []>("@model_path/weights/weight.bin"), offset = tensor<uint64, []>(IDX_OFF))), lut = tensor<fp16, [16]>(BLOBFILE(path = tensor<string, []>("@model_path/weights/weight.bin"), offset = tensor<uint64, []>(LUT_OFF))), name = tensor<string, []>("wdeq"), shape = tensor<uint32, [4]>([OUT, IN, 1, 1])];
        tensor<int32, [2]> st = const()[name = tensor<string, []>("st"), val = tensor<int32, [2]>([1, 1])];
        tensor<string, []> pt = const()[name = tensor<string, []>("pt"), val = tensor<string, []>("valid")];
        tensor<int32, [2]> dl = const()[name = tensor<string, []>("dl"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, []> gr = const()[name = tensor<string, []>("gr"), val = tensor<int32, []>(1)];
        tensor<int32, [4]> pd = const()[name = tensor<string, []>("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<fp16, [1, OUT, 1, S]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = tensor<string, []>("conv")];
    } -> (y);
}
```

- `indices` is typed as the PACKED byte count, `uint8 [OUT*IN/2]`, not the
  element count.
- `lut` is 16 fp16 centroids for 4 bits (the op supports 2, 4, 16, 64 or 256
  entries: 1, 2, 4, 6, 8 bits).
- `shape` is the dense output shape, inline.

## The blob (two chunks, `buildMultiWeightBlob(chunks:chunkTypes:)`)

- Chunk 0: packed indices, **two 4-bit codes per byte, low nibble first**,
  over the row-major `[OUT, IN]` weight. Verified identical to a real
  palettized model's payload byte for byte. Its header type field (byte +4)
  is **3** (uint8); pass `chunkTypes: [3, 1]`.
- Chunk 1: the 16 fp16 LUT entries, type 1.
- For reference, the blob type codes seen so far: fp16 and int8 payloads `1`,
  uint8 packed indices `3`, packed int4 (blockwise data) `8`.

## Quantizing to a palette

Per-tensor uniform, the form the probe confirmed:

```text
s       = max|w| / 8
lut[i]  = (i - 8) * s              i in 0..15   (values -8s .. 7s)
code    = clamp(round(w / s) + 8, 0, 15)
```

This is int4 uniform quantization expressed as a LUT, at one scale for the
whole tensor, which is why the error is 15.9 percent on a random weight. It
proves the hardware path; it is too coarse for a production projection.

## Getting production accuracy

The palette op decouples the codebook from uniform spacing, so accuracy
improves without leaving the confirmed op:

1. **Non-uniform centroids.** Fit the 16 entries to the weight distribution
   (k-means, or quantiles) instead of uniform steps. Same op, same blob,
   better error on the heavy-tailed weights a projection has.
2. **Per-channel scale plus a shared codebook.** Keep one 16-entry LUT and
   apply a per-output-channel fp16 scale afterward with a `mul` against a
   `[OUT, 1, 1, 1]` const; the `mul` is a confirmed accurate ANE op. This
   gives per-channel range at 4-bit storage.
3. **Grouped LUTs.** The iOS18 `constexpr_lut_to_dense` carries a per-group
   LUT shape; its in-memory compile is untested and, given the blockwise
   result, likely rejected. Try it only after 1 and 2.

Match the GPU's group-64 affine int4 exactly is not possible through the
palette op (a palette is a codebook, an affine group is a scale); the GPU and
ANE will hold the same tensor at different 4-bit representations. Since this
is not the ranked model, that is acceptable; measure the end-to-end
divergence, not byte identity.

## NVFP4 on the ANE

Not native. NVFP4's E2M1 codebook is exactly 16 values, so it is
representable as this 4-bit palette (centroids = the FP4 values, per-block
scales through option 2 above). Representation, not a native mode.
