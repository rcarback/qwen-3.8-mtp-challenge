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
| blockwise affine (group scale) | `constexpr_blockwise_shift_scale` | iOS18 | **rejected** in memory (`InvalidMILProgram`, byte-identical blob); through `.mlpackage` Core ML compiles it and places the conv on the **GPU** (`MLComputePlan`, 2026-09-06: int8 g32, int4 g32, int4 g64, with and without offset, all `supported=cpu/gpu`) | affine forms fold to fp16 |
| grouped palette (one LUT per row block) | `constexpr_lut_to_dense`, `lut` shaped `[O/64, 1, 1, 1, 16, 1]` | iOS18 | **confirmed on the ANE through `.mlpackage`** (0.27 ms, same as per-tensor; unprobed in memory) | palette |

The pattern behind the split is now settled: the blockwise op is not an ANE
op at all. Core ML compiles it and then routes the conv that consumes the
weight to the GPU, so the MLX group-affine tensor (q4 g32, q4 g64, q8 g32)
can never be the ANE's tensor, and no ANE weight is byte-identical to the
GPU's. The ANE's compressed forms are int8 per-channel affine and the
palette, per-tensor or grouped. Use the palette op. It is also the form with
the measured bandwidth win, so this is the right answer, not a fallback.

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
   LUT shape. Confirmed on the ANE through the `.mlpackage` path with one
   codebook per 64 output rows at no time cost; a per-row codebook
   (`[O, 1, 1, 1, 16, 1]`) costs 32 bytes per row. Untested in memory.
4. **Do not chain a per-channel `constexpr_blockwise_shift_scale` after an
   int8-entry LUT.** It stays on the ANE but runs 2.9x slower (0.77 ms
   against 0.27). The output-side `mul` of option 2 is free.
5. **Never apply a runtime op to the weight.** `mul(constexpr_lut_to_dense(...),
   scale)` compiles, the compute plan keeps the conv on the ANE, and Core ML
   then rebuilds the dense weight on every call: 0.1 to 3.8 seconds per conv
   at real shapes (2026-09-06, the void int4 rows of the `r` table). Scale the
   conv output (`mul(y, scale[1, O, 1, 1])`), which is what
   `buildConvMILTextForm` and the fused int4 program do.

Matching the GPU's group-64 affine int4 exactly is not possible on the ANE
at all (the blockwise op is GPU-only, above), so the GPU and ANE hold the
same tensor at different 4-bit representations. The production form in this
tree (`ANEWeightQuant.int4Palette`, `MLX_ANE_WEIGHT_FORM=int4`) is option 2:
each row divided by its RMS, a 16-entry codebook fitted by Lloyd iterations
on the pooled normalized values, and the per-row scale applied to the conv
output. Measure the end-to-end divergence, not byte identity.

## NVFP4 on the ANE

Not native. NVFP4's E2M1 codebook is exactly 16 values, so it is
representable as this 4-bit palette (centroids = the FP4 values, per-block
scales through option 2 above). Representation, not a native mode.

## Measured at real shape, direct path (2026-09-06)

The dense tower's fused MLP prefix (hidden 5120, F=5440) through the
in-memory dispatch: fp16 33.9 ms, int8 18.9, int4 15.0 per call at S=1024;
at S=128 fp16 3.25, int8 2.33, int4 2.59. The compressed forms are 1.8 to
2.3x faster at the bucket the lane uses because the engine re-streams its
weights per spatial tile (fp16's per-row cost rises with S, int4's falls).
End to end, cool, int4 at fraction 0.3125 prefilled +14.8 percent over the
GPU and int8 at 0.5 +16.6, against fp16's +4.8. Per-row codebooks are
lossless in perplexity (5.565 vs 5.566) but a per-row LUT program falls off
the ANE in Core ML's compute plan (every op `preferred=gpu`); one codebook
per 64 rows stays on it.

