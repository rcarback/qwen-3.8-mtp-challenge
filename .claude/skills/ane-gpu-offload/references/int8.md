# int8 weight programs (affine, per-channel)

**Confirmed in hardware.** `ANEQuantizedWeightProbeTests` int8 arm: the ANE
consumed int8 weight storage and produced the projection at 0.8 percent
relative error (max-abs 0.795 against a reference max of 98), which is the
int8 quantization error, not a dispatch fault. This retires the belief that the
ANE lane needs a dense fp16 or bf16 weight tree.

## Why it matters

The architecture paper establishes that the ANE datapath reconstructs an int8
weight to fp16 at the multiplier input. So the stored bytes are int8, the
multiply is fp16, and the ANE reads half the bytes of an fp16 program. An int8
model tree (the q8 dense projections) can feed the ANE directly. One caveat
from the same paper: int8 affine "folds to dense fp16 in the conversion", so
treat the bandwidth win as accepted-storage, not a measured streaming gain;
the 2.37x streaming win is the int4 palette form.

## The MIL text (byte-exact, from `buildConvMILTextInt8`)

The dequantize is a `constexpr_affine_dequantize` whose parameters are ALL
inline inside the op's attribute bracket, with an empty argument list. There
are no separate `const()` declarations for the quantized data, scale,
zero-point or axis. This is the form the ANE compiler accepts; declaring them
as separate consts and passing them in `(...)` fails with `InvalidMILProgram`.

```text
func main<ios18>(tensor<fp16, [1, IN, 1, S]> x) {
  tensor<fp16, [OUT, IN, 1, 1]> w = constexpr_affine_dequantize()[axis = int32(0), name = string("wdeq"), quantized_data = tensor<int8, [OUT, IN, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight_data.bin"), offset = uint64(64))), scale = tensor<fp16, [OUT]>([s0, s1, ..., s_OUT-1]), zero_point = int8(0)];
  ... same pt/st/pd/dl/gr consts and conv as fp16, with weight=w ...
} -> (y);
```

- `scale` must be a scalar or a rank-1 vector. With `axis = 0` it is one
  fp16 value per output channel, `[OUT]`. A rank-4 scale is rejected.
- `zero_point = int8(0)` for symmetric quantization. It is inline.
- The op is iOS16-vintage and compiles through the in-memory ANE path (unlike
  the iOS18-only blockwise op; see `int4.md`).

## Quantizing the weight

Per-output-channel symmetric int8, the form the probe used:

```text
for each output row o:
  scale[o] = max_k |w[o,k]| / 127
  q[o,k]   = clamp(round(w[o,k] / scale[o]), -127, 127)   as Int8
```

The int8 bytes are row-major `[OUT, IN]`, one byte per value, wrapped with
`buildConvWeightBlob` exactly as fp16 is (chunk type `1`, payload at 128, byte
count `OUT*IN`). The scale vector goes inline in the MIL text; for a large
`OUT` the text literal is long but accepted (`OUT=512` was tested inline).

## Matching the GPU tree's quantization

MLX's affine quantization is group-wise along the input axis (group 64 in this
model), with a scale and bias per group, not per output channel. The
per-channel affine op above cannot express a per-group scale; that is the
blockwise op in `int4.md`, which also takes int8 data. To share the exact GPU
representation on int8, use `constexpr_blockwise_shift_scale` with int8 data
and a `[OUT, IN/64, 1, 1]` scale, once that op's in-memory compile is
resolved. Until then, per-channel int8 is a re-quantization of the same tensor
at int8 precision, correct to int8 error, but not byte-identical to the GPU's
grouped form.

## Where it plugs in

`Qwen4ExpANEProjection` builds from a weight and today only emits fp16. The
plain micro-batch path (`aneQProjection`, `aneInProjection`) is guarded to
return nil for a `QuantizedLinear`, because feeding its packed `.weight` (q8
`dim(1) = in/4`) to the fp16 builder built the program at the wrong width and
crashed on the shape precondition. The fix path is to unpack the
`QuantizedLinear` to int8 values plus scales and emit this int8 program, so
the ANE lane runs on the q8 tree with no dense expansion.
