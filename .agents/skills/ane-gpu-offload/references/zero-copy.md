# Zero-copy I/O with the ANE

The ANE reads and writes IOSurfaces. MLX activations live in Metal buffers.
Every byte that crosses between them is either shared or copied, and on a
bandwidth-bound engine the copy is the whole margin. This file is the measured
cost structure and the pattern that keeps the hot path copy-free.

## The measurement that settles it

`in_proj_qkv [10240x2560]`, S=128, warm, median of 50, through
`MLModel.prediction` with a surface-backed input `MLMultiArray` and
`MLPredictionOptions.outputBackings` writing into an IOSurface that MLX wraps
directly (`ANEMultiFunctionProbeTests.realShapeDispatchCostSurface`):

| step | ms |
| --- | --- |
| stage: MLX activation to input surface | 0.220 |
| of which the GPU transpose plus `eval` alone | 0.226 |
| of which plus the actual GPU-to-CPU copy | 0.223 |
| predict | 0.632 |
| read: output surface to MLX | 0.005 |
| GPU 4-bit group-64 matmul, reads the MLX buffer, no staging | 0.684 |

Three facts:

- **The output is genuinely zero-copy.** `outputBackings` lets the ANE write
  its result into a surface you own, and `MLXArray(rawPointer:
  surface.baseAddress, ...)` wraps it with no copy. 0.005 ms.
- **The compute wins.** predict over the quantized GPU is 0.92x.
- **The input "stage" is a launch floor, not a copy.** The transpose plus
  `eval` is the entire 0.22 ms, and adding the real GPU-to-CPU move changes
  nothing. A 655 KB transpose is microseconds of bandwidth; the rest is the
  per-op GPU launch and sync, measured on a tiny isolated op. In a fused
  forward the activation is already materialized and the transpose is
  pipelined, so this is not a marginal cost there.

So the end-to-end 1.24x that the isolated probe shows is an artifact. The
honest ANE per-projection cost at a real prefill shape is the 0.632 ms
compute plus a cheap handoff, against 0.684 ms on the GPU.

## What "zero copy" requires

The activations must sit at an address both engines read. On unified memory an
IOSurface is that address: the GPU writes it through an `MTLBuffer` created
from the surface, the ANE reads the same bytes. MLX exposes the primitives:

- `MLXArray(rawPointer: ioSurface.baseAddress, shape, dtype:, finalizer:)`
  wraps a surface's memory as the array's backing store with no copy
  (ownership transferred; its own docstring uses an IOSurface).
- `asMTLBuffer(device:noCopy:)` hands MLX's backing to Metal without a copy.

Neither path in this tree is fully zero-copy on input yet: the direct
`_ANERequest` path does `asData()` (GPU to CPU) then a memcpy into the
surface, and its `readZeroCopy` still materializes the output with a
`contiguous`. Full zero copy means producing the activation into a
surface-backed buffer in the first place. That is the remaining engineering,
and it is orthogonal to multifunction and to the program count.

## The pattern

1. Allocate persistent IOSurfaces once per (program, bucket): input
   `[1, IN, 1, S]` and output `[1, OUT, 1, S]`, contiguous fp16 through the
   Core ML path (Core ML handles the engine's 32-element sequence padding);
   64 KB aligned (`makeSurface` in `ANEDirectDispatch`).
2. Wrap each as an `MLMultiArray(dataPointer: surface.baseAddress, shape:,
   dataType: .float16, strides:)` with contiguous strides
   `[IN*S, S, S, 1]`.
3. Build one `MLPredictionOptions` with `outputBackings = ["y": outArray]`
   and one `MLDictionaryFeatureProvider(["a": inArray])`, reused every call.
4. Per call: write the activation into the input surface (lock, memcpy,
   unlock), `model.prediction(from:options:)`, then wrap the output surface as
   an MLXArray and `contiguous(transposed)` it into the graph.
5. Never rebuild an `MLMultiArray` per call; `mlxToMultiArray_1C1S` allocates
   and memcpys each time and was the 0.177 ms "input copy" that mis-measured
   this path as a loss.

## The direct path, for comparison

`ANEDirectDispatch` (`_ANERequest` over `_ANEIOSurfaceObject`, evaluated by
`_ANEInMemoryModel`) is the raw form of the same handoff and carries a
`procedureIndex`. Its sequence axis must be written at the 32-padded stride on
both input and output (the Core ML path hides this). It dispatches only
procedure 0 of a bare in-memory program, so for a banked program the Core ML
multifunction path above is the one to use; see `program-limit.md`.

## What is not the cost

The host-side surface lock, write and read are not the dominant cost; the
field guide and the architecture paper both fit the per-call cost as a fixed
dispatch floor (about 95 to 190 us across M1 to M3 Max, about 98 percent
software and firmware) plus bytes over 78 to 85 GB/s. The interface is not
wrong; the engine is narrow and the dispatch is fixed-cost. That is why
decode, with a 5 to 20 KB activation and 48 or more crossings per token,
cannot amortize it, and prefill at S >= 128 can.
