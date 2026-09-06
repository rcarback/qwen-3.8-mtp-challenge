# int4 weight programs (blockwise, grouped)

**Status: exact format known; in-memory compile is an open item.** The
architecture paper establishes that the ANE dequantizes int4 to fp16 at the
multiplier input, and that the int4 palette form streams natively at about
2.37x the bandwidth of fp16. The MIL op, blob layout and nibble packing below
are byte-identical to what a real int4 model emits. But the in-memory ANE
compiler (`_ANEInMemoryModel`, `compileWithQoS`) returns `InvalidMILProgram`
for it, while accepting the int8 affine op from the same code path. The
distinguishing fact is that `constexpr_blockwise_shift_scale` is an iOS18-only
op and `constexpr_affine_dequantize` is iOS16; the working hypothesis is that
the in-memory compile path predates the op and the `.mlpackage` compile path
(`MLModel.compileModel`, which handles the multifunction descriptor) is
needed. That is the next experiment; do not report int4 as working until it
runs.

## Two int4 forms, and which gives the bandwidth win

| form | MIL op | what the paper measured |
| --- | --- | --- |
| int4 affine / blockwise (group scale) | `constexpr_blockwise_shift_scale` | accepted; affine forms "fold to dense fp16 in conversion" |
| int4 palette (lookup table) | `constexpr_lut_to_dense` (16-entry LUT) | streams natively at ~2.37x bandwidth, ~2.37x faster than fp16 |

The GPU's MLX 4-bit is affine group-64, so blockwise is the form that shares
the GPU's representation. The 2.37x streaming win is specific to the palette
form, which is a different quantization (a 16-value codebook per group). Since
this model is not the ranked target, palettizing the dense projections is an
open option; NVFP4's E2M1 codebook is exactly 16 values and can be expressed
as a grouped 4-bit palette, so an NVFP4 weight maps onto the ANE's LUT form
(not a native mode).

## The MIL text (byte-exact, blockwise)

Unlike the affine op, blockwise puts `data` and `scale` in the argument list
and only `name` in the attribute bracket. Both are BLOBFILE references (a
two-chunk blob). An inline rank-4 scale also fails through the in-memory path.

```text
func main<ios18>(tensor<fp16, [1, IN, 1, S]> x) {
  tensor<fp16, [OUT, IN, 1, 1]> w = constexpr_blockwise_shift_scale(data = tensor<int4, [OUT, IN, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(DATA_OFF))), scale = tensor<fp16, [OUT, IN/G, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(SCALE_OFF))))[name = string("wdeq")];
  ... pt/st/pd/dl/gr consts and conv with weight=w ...
} -> (y);
```

- The block size is inferred from the scale shape: `data [OUT, IN, 1, 1]`
  with `scale [OUT, IN/G, 1, 1]` means blocks of `G` along the input axis.
  `output = scale[block] * (data - offset[block])`; omit `offset` for
  symmetric.
- `data` may be int4, uint4, int8 or uint8. Using int8 data with int4-range
  values compiles and gives int4 values at int8 storage (no byte saving).

## The int4 blob (two chunks, `buildMultiWeightBlob(chunks:chunkTypes:)`)

- Chunk 0 is the packed int4 data: **two signed nibbles per byte, low nibble
  first**, over the row-major `[OUT, IN]` flattened weight. `byte[i/2] =
  (q[i] & 0xF) | ((q[i+1] & 0xF) << 4)`, two's complement in the nibble
  (`-8 = 0x8`, `-1 = 0xF`, `7 = 0x7`). This packing was verified identical to
  a real int4 model's payload byte for byte.
- Chunk 0's header type field (byte +4) is **8**, not 1. A real int4 model's
  `weight.bin` carries `08 00 00 00` there; fp16 and int8 carry `01`. Pass
  `chunkTypes: [8, 1]`.
- Chunk 0's byte count is `OUT*IN/2`, payload at header+64.
- Chunk 1 is the fp16 scale, `[OUT, IN/G]` row-major, type 1.
- The blob-level header at offset 0 is `chunks.count` (u32) then `2` (u32).

## Quantizing the weight

Blockwise symmetric int4, group `G` along the input axis:

```text
for each output row o, group g:
  scale[o,g] = max_{j<G} |w[o, g*G+j]| / 7
  q[o, g*G+j] = clamp(round(w / scale[o,g]), -8, 7)
```

## Confirming it

Run `ANEQuantizedWeightProbeTests` int4 arm. It builds the two-chunk blob
with the exact packing above and currently records the compile failure. The
next step is to route the same program through `MLModel.compileModel` on a
`.mlpackage` (the path `ANEMultiFunctionProbeTests` already uses) and check
whether the iOS18 op compiles there. If it does, the direct zero-copy dispatch
of an int4 program goes through the multifunction `.mlpackage`, which is also
how the program-count limit is relieved; see `program-limit.md`.
