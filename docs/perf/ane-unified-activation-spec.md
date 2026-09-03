# Unified activation path for the ANE dense lane

Status: design, not implemented. Written 2026-09-03.

## The proposal in one paragraph

Remove every per-dispatch copy between the GPU and the ANE by giving both
engines the same activation precision, the same layout, the same row stride,
and the same allocation. On this hardware the two engines address one pool of
DRAM, so a copy between them moves bytes from one region of that pool to
another region of the same pool. The copy exists because four software
conventions disagree, not because the data must travel. Make the conventions
agree and the copy has nothing to do.

Weights are explicitly out of scope. See "Why weights stay as they are".

## Scope: the dense lane, not the MoE lane

This specification targets the Qwen35 dense tower split-MLP lane
(`Qwen35ANESplitOffload.swift`), not the Qwen4Exp MoE projection lane.

Measured reason. The dense lane stages 15.0 MB per layer at S=732 and performs
122 GFLOP against it, a ratio of 8160 FLOP per staged byte. The MoE lane
reaches 2214. The dense lane therefore returns 3.7 times more compute for each
byte moved, which is the quantity this work improves.

Second reason. `docs/perf/ane-zero-transfer-spec.md` establishes a ceiling on
the MoE lane: it offloads one projection per layer per micro-batch, capping the
whole lane at 2.7 to 8.1 percent of prefill, while the micro-batching it
requires costs 2.6 to 15 times that ceiling. No transfer work can lift a lane
whose upside is smaller than its enabling cost.

## The four disagreements, and what each costs today

Dense tower geometry: 64 layers, hidden 5120, intermediate 17408, weights
4-bit affine group-64. The ANE takes a prefix of 31.25 percent of the
intermediate channels, so f = 5440.

| Convention | GPU today | ANE today | Cost of the mismatch |
|---|---|---|---|
| Precision | bf16 activations | fp16 | a cast per dispatch |
| Layout | `[S, IN]` token-major | `[IN, S]` channel-major | a transpose, which forces a fresh buffer and an `eval` |
| Row stride | packed | padded to 32-element multiples | a scatter of one memcpy per channel row |
| Allocation | Metal allocator pool | `IOSurfaceCreate` | a memcpy between two regions of the same DRAM |

At bucket 1024 the lane stages 21.0 MB per layer, so 1.25 GiB per prefill
across 64 layers. Every byte of that is subject to the four costs above.

## Why fp16 is safe here, with evidence

`docs/perf/ane-hybrid-collapse-2026-09.md` lines 46 to 49 record a direct test.
Four exact weight-folded rescalings, including per-channel SmoothQuant, left
the measured error unchanged to three decimals on every layer. The same
investigation recorded a maximum input value of 50 and a maximum intermediate
value of 209.

The fp16 maximum is 65504. The observed activations sit three orders of
magnitude below it. The outlier-channel hypothesis was tested against this
model and did not hold. The earlier hybrid collapse was traced to a descriptor
identity collision and a silu approximation, both corrected in commit
`37538ee`.

Conclusion: converting activations from bf16 to fp16 is a layout and range
question, and this model answers it safely. fp16 carries more mantissa than
bf16 (10 bits against 7), so the conversion adds precision and removes only
exponent range this model does not use.

## Why weights stay as they are

Two independent reasons, either sufficient.

Memory. The MoE tower holds 120.8 billion expert parameters. At 4 bits they
occupy 56.2 GiB. At fp16 they would occupy 225.0 GiB, against a 128 GB
machine whose observed serve peak already reaches 88 to 90 GB. The dense tower
is smaller but the argument holds in the same direction.

Relevance. Weights are not copied per dispatch. `buildConvWeightBlob` bakes
them into the ANE program once, at program build. They never move again.
Changing their precision therefore costs memory and saves no transfer at all.

The per-dispatch traffic is activations. Activations are small: 15.0 MB per
layer at S=732, against weights measured in gigabytes. Unifying the small
thing that moves is the entire opportunity.

## Variant A: fp16 unified path

Now. This is the primary variant and the one to build first.

1. Activations carry fp16 from the point they enter the offloaded region to
   the point they leave it.
2. Activations are stored channel-major, `[IN, S]`, so the ANE consumes them
   without a transpose.
3. Activations are allocated with rows padded to 32-element multiples, so the
   stride already matches what the ANE requires.
4. The backing allocation is one IOSurface, wrapped for MLX rather than copied
   into.

Point 4 carries the one unresolved dependency. MLX copy-on-writes when it
assigns into memory wrapped by `MLXArray(rawPointer:)`, which a spike in this
session confirmed directly: a poisoned buffer kept its poison while MLX's own
view showed the new values. Reading foreign memory zero-copy works and is what
`ANEDirectDispatch.readZeroCopy` already does. Writing into it does not.

Therefore point 4 requires one of:

- teaching the vendored MLX allocator to adopt a caller-supplied `MTLBuffer`,
  which `MTLDevice.makeBuffer(bytesNoCopy:)` can build over a page-aligned
  IOSurface. `Vendor/mlx-swift` is inside the editable surface.
- or accepting one bulk copy per dispatch, which is still far cheaper than
  today's transpose plus `eval` plus host copy plus per-row scatter.

Build the second first. It is small, it is safe, and it establishes the layout
and stride work independently of the allocator change.

## Variant B: int8 unified path

Next. Build only after Variant A is measured.

The ANE runs int8 well, and int8 halves every staged byte: 7.5 MB per layer at
S=732 instead of 15.0 MB, and the ratio rises from 8160 to 16320 FLOP per
staged byte.

Three differences from Variant A that must not be blurred together.

First, int8 is lossy and fp16 is not. For this model fp16 is effectively exact,
because the values sit far inside its range. int8 spans 256 levels and requires
a scale per tensor or per channel. It introduces real quantization error where
fp16 introduces none. Variant B therefore needs an accuracy gate that Variant A
does not.

Second, the scaling scheme is a decision, not a detail. Per-tensor scaling is
simpler and less accurate. Per-channel scaling is what the hybrid-collapse
investigation already implemented and measured, so the machinery exists.

Third, int8 activations are not weights, so the frozen quantization envelope
described in `CLAUDE.md` does not obviously apply. Confirm this before any
int8 work reaches a ranked path. int8 *weights* outside the attention
projections would violate that envelope. This lane is currently headed
`LOCAL M4 FORK ONLY` and is never wired into the ranked forward, so the
envelope does not bind today.

## What must be verified, and how

Every check below runs without loading the model.

1. Numerical equivalence, Variant A. Compare the unified path against the
   current GPU path at asymmetric, non-square shapes. Use a distinct value per
   (row, column) position drawn from a generator whose period exceeds the
   larger dimension. A symmetric or low-period test input hides a transposed or
   row-permuted result, which is the exact failure this change risks.
2. Stride correctness. Assert the staged bytes equal the expected bytes at a
   packed shape and at a padded shape. Reading back the staged input surface is
   the only check that proves the bytes are correct rather than merely
   self-consistent.
3. Coherency. Confirm the ANE observes GPU writes to the shared surface. This
   is the least understood risk in the design: CPU and GPU are coherent on this
   hardware, but ANE visibility is not automatic, and `IOSurfaceLock` and
   `IOSurfaceUnlock` exist for that handshake. A missed handshake yields stale
   data, not a crash, so no test that only checks for crashes will catch it.
4. Accuracy gate, Variant B only. Measure the error the int8 quantization
   introduces against the fp16 path, per layer, and state the tolerance before
   measuring rather than after.

## Interaction with Task 3

Task 3 buckets the dense lane's program key to a power of two floored at 128.
That fix is correct and independent, and it removes a real compile thrash.

It also creates a quantity this specification cares about. A 732-token prompt
buckets to 1024, so 28.5 percent of every staged byte and every ANE FLOP is
padding. Halving the bytes through int8 while discarding 28.5 percent of them
to padding is a smaller win than it appears. Rounding the bucket to a multiple
of 128 instead of a power of two would cut the waste to about 10 percent, at
the cost of more compiled programs. Task 3's report records the waste
explicitly so that this decision can be made on measurement.

## What this work does not do

It does not move a ranked score. The lane is headed `LOCAL M4 FORK ONLY` and
is gated behind `MLX_ANE_DIRECT=1`, so it never executes in the ranked forward.
Treat the value of this work as understanding and as groundwork, and do not
report it as a ranked improvement.

It does not rescue the MoE lane, whose ceiling is established elsewhere and is
lower than its enabling cost.

It does not remove the `eval` before staging. `asData(access:)` calls `eval`
unconditionally at `MLXArray+Bytes.swift:211`, and any CPU-visible read of
MLX-owned bytes must evaluate first. Removing the transpose removes a
materialisation, not the evaluation.

## Open questions

1. Does the ANE observe GPU writes to a shared IOSurface without an explicit
   lock cycle, and what does the correct handshake cost? This gates the whole
   allocation half of the design.
2. Can the vendored MLX allocator adopt a caller-supplied `MTLBuffer` without
   disturbing its pooling? `GPU+Metal.swift` returns freed buffers to an
   internal pool, so an adopted buffer must not enter it.
3. What does the GPU actually lose by working channel-major? `quantizedMM`
   places the quantized operand on the right and computes `x @ w.T` or
   `x @ w`, so a channel-major output is not directly expressible for
   quantized weights. Plain `matmul` has no such restriction, which is why
   this specification applies to activations, where the ANE side already holds
   fp16.
