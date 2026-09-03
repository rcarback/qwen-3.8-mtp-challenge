# Unified activation path for the ANE dense lane: final plan

Status: plan, after adversarial review. Written 2026-09-03.
Supersedes the recommendation in `docs/perf/ane-unified-activation-spec.md`.
That specification stays in the tree as the original design statement. Three
of its four claimed costs do not hold. This document records what was checked,
what survives, and what to build.

## Verdict: redirect

The design asks to remove four per-dispatch copies caused by four disagreeing
conventions. Reading the vendored sources shows that three of the four rows in
the specification's table cost nothing that a respelling can remove.

| Row | Claim in the specification | What the sources show |
|---|---|---|
| Precision | a cast per dispatch | True, and the cast is one required pass. No respelling removes it. |
| Layout | a transpose, which forces a fresh buffer and an `eval` | The transpose does not fuse with the cast. Both orders run the same two kernels and move the same bytes. |
| Row stride | one memcpy per channel row | The stride already matches on this lane. The loop wastes 5119 call overheads, not traffic. |
| Allocation | a memcpy between two regions of the same DRAM | True. This is the only cost a respelling removes, and it is host-side. |

What remains buildable inside the editable surface is worth an estimated 0.6
to 0.9 percent of dense-tower prefill. On the same lane, at the same time,
28.5 percent of every unit of MLP work at 732 tokens is bucket padding, and
49.6 percent at 1032 tokens. Both engines pay it. That is between 30 and 80
times the staging prize, and it is the target the work should move to.

The padding fix lives in `Qwen35ANESplitOffload.swift` and
`Qwen35.swift`. Another agent owned both files while this plan was written, so
this work did not edit them. Section "Handoff: the bucket padding" states the
arithmetic for that owner, and records that the first and cheapest part of it
has since been built in commit `00d4e270`.

Precision stays fp16. Variant B, int8 activations, is refused on first-hand
evidence, not deferred. See "Variant B is refused".

## What was checked, and how

Every finding below comes from the vendored sources or from a probe run in
this session. No timing was measured, and no timing number is reported.

### The transpose does not fuse into the cast

The draft plan claimed that `contiguous(x.transposed(1,0).asType(.float16))`
runs one kernel where `contiguous(x.transposed(1,0)).asType(.float16)` runs
two. It does not. The chain of four facts:

1. `Transpose::eval`
   (`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/common/common.cpp:272-303`)
   copies `flags` from the input and recomputes only `row_contiguous` and
   `col_contiguous`. The `contiguous` bit stays true, because a transposed
   view of a packed array has no gaps.
2. `AsType::eval_gpu` (`backend/gpu/primitives.cpp:28-33`) branches on exactly
   that bit: `inputs[0].flags().contiguous ? CopyType::Vector :
   CopyType::General`. A transposed view therefore takes the Vector path.
3. `set_copy_output_data` for Vector (`backend/common/copy.h:30-41`) gives the
   output `in.strides()` and `in.flags()`. The cast result keeps the
   transposed strides. It is not row-contiguous, and no transpose happened.
4. `Contiguous::eval_gpu` (`backend/gpu/primitives.cpp:49-61`) needs
   `row_contiguous`, or `col_contiguous` when `allow_col_major_` is set.
   Swift's `contiguous(_:allowColMajor:)` defaults that flag to false
   (`Vendor/mlx-swift/Source/MLX/Ops.swift:3279-3281`). So it runs
   `copy_gpu(in, out, CopyType::General)`, the full transposing copy.

Both orders run one Vector kernel and one General kernel, allocate two
buffers, and move the same bytes. The layout row of the specification's table
is not free, and it is not removable by reordering. Only a custom kernel
reaches one pass.

The same reasoning applies to the output side. `readZeroCopy` ends in
`contiguous(real.transposed(1,0))`, and the caller then runs
`anePartial.asType(gpu.dtype)`
(`ANEFusedSplitMLP.swift:229`). Respelling those as
`contiguous(real.transposed(1,0).asType(dtype))` gives a Vector kernel plus a
General kernel, which is what runs today.

A test in this plan asserts the underlying fact directly, so the refuted
mechanism cannot be re-asserted silently later.

### The row stride already matches

`rowStride = ((sequenceLength + 31) / 32) * 32`
(`ANEDirectDispatch.swift:131`). `ANESplitMLPCache.bucketedSequenceLength`
returns a power of two floored at 128
(`Qwen35ANESplitOffload.swift:140-145`). Every bucket is 32-aligned, so
`rowStride == sequenceLength` on every dense-lane dispatch. The 5120-iteration
scatter loop moves the correct bytes. Its waste is call overhead alone.

The comment at `ANEDirectDispatch.swift:148-150` claims a bulk fast path
already exists. No such branch is in the loop. Step S1 makes the comment true.

### fp16 is 2 bytes per element

The draft plan priced fp16 at 1 byte in every cell of its byte table, and both
reviewers repeated the error in at least one cell. The specification's own
figure settles it. At bucket 1024 with hidden 5120 the lane stages 21.0 MB per
layer, which is two surfaces of 5120 x 1024 x 2 bytes. The corrected
accounting is in "Byte accounting".

### int8 is rejected by the Core ML compiler

> **SUPERSEDED 2026-09-03, and the heading is wrong.** This section generalised
> from three spellings to "the opset refuses int8", and that inference does not
> hold. Core ML does not express int8 by handing int8 tensors to `conv` or
> `matmul`, which is all this section tested. Re-probed with the representation
> Core ML actually uses, `constexpr_blockwise_shift_scale` (`data` / `scale` /
> `offset`, scale shaped `[N,1,1,1]`), the compiler ACCEPTS int8 weights and
> places the conv on the ANE. See "int8, re-probed" below. The three rejections
> quoted here are real; the conclusion drawn from them was not.

`tools/int8-probe` asks Apple's compiler directly through hand-authored MIL.
Run in this session:

```text
REJECTED  matmul int8 x int8 -> int8   in operation mm: Param 'y' has incorrect
          type for operator 'ios18.matmul'. Expected { tensor<int32, ...
REJECTED  conv   int8 x int8 -> int8   in operation cv: Param 'weight' has
          incorrect type for operator 'ios18.conv'. Expected { tensor<bf16, ...
REJECTED  conv_quantized fp16 x uint8  Unknown operator 'conv_quantized'.
```

Variant B needs an int8 activation tensor entering the conv. The ios18 opset
that `ANEMILBuilder` emits refuses it.

## int8, re-probed (2026-09-03)

The earlier finding was drawn from a probe that never tested the supported
spelling. Re-run on this M4 Max, all cases hand-authored MIL asked directly of
Apple's compiler, with `MLComputePlan` reporting placement.

| Spelling | Compiler | Placement |
|---|---|---|
| CONTROL: fp16 conv, no quantization | accepted | `conv=ANE` |
| raw int8 operands to `matmul` | rejected | -- |
| raw int8 weight const to `conv` | rejected | -- |
| `conv_quantized` (iOS15) | rejected, unknown operator | -- |
| runtime `dequantize` of an int8 const | accepted | whole graph `CPU` |
| `constexpr_blockwise_shift_scale` weight | accepted | `conv=ANE` |
| the same plus activation `quantize`/`dequantize` | accepted | `conv=ANE`, `quantize`/`dequantize`=`CPU` |

Three things follow, and they are different from each other.

**int8 WEIGHTS reach the ANE.** `constexpr_blockwise_shift_scale` is a
compile-time op, so the int8 bytes are stored and expanded during compilation.
Its runtime parameter names are `data`, `scale` and `offset`, which are NOT
coremltools' Python names: `constexpr_affine_dequantize` with `quantized_data`
is rejected as an undefined attribute. Per-output-channel scale is shaped
`[N,1,1,1]` for a `[N,K,1,1]` weight, not `[N]`.

**A runtime `dequantize` of a const is the trap.** It is accepted, so it looks
like it works, and it silently moves the entire graph to the CPU -- including
a conv that runs on the ANE when the same weight is fp16. The CONTROL row is
what makes that visible; without it a CPU placement is unattributable.

**int8 ACTIVATIONS do not pay in this form.** The `quantize`/`dequantize` pair
compiles and the conv stays on the ANE, but the pair itself is placed on the
CPU, which buys a CPU round trip per dispatch instead of int8-int8 compute.
Apple documents int8-int8 ANE throughput from A17 Pro and M4 onward, and this
box is an M4 Max, so the capability exists and this spelling does not reach
it. Treat activation quantization as open, not closed: what is unresolved is
the pattern the compiler's fusion pass wants, not the hardware.

The error budget quoted further down is still unmeasured, and that criticism
stands independently of the placement result.

## Target and arithmetic

### The lane, and its size

The lane is the Qwen35 dense-tower split MLP: `ANEFusedSplitMLP`,
`ANEFusedMLP`, and `ANEDirectDispatch`. Geometry: 64 layers, hidden 5120,
intermediate 17408, `MLX_ANE_FRACTION` 0.3125, so f = 5440.

The ANE prefix per layer at 732 tokens is 3 x 2 x 732 x 5120 x 5440 = 122.33
GFLOP. Across 64 layers that is 7829 GFLOP. The draft plan's correction of the
earlier ceiling probe stands: the probe divided the per-layer figure by the
64-layer total and reported 0.34 percent, which is wrong by the layer count.

FLOP per staged byte on this lane is 3 x 2 x 5120 x 5440 / (2 x 5120 x 2) =
8160 exactly. The MoE projection lane reaches 2 x 2560 x 16384 / (2 x 2560 + 2
x 16384) = 2214. The dense lane returns 3.7 times more compute for each staged
byte, which is why the specification chose it.

### Byte accounting, corrected

Per dispatch at bucket 1024, hidden 5120. One payload is 1024 x 5120 x 2 =
10.49 MB. Both fp16 and bf16 are 2 bytes per element, so a cast moves the same
volume in and out.

Input path today:

| Step | Kernel | Traffic | Allocations |
|---|---|---|---|
| `x.asType(.float16)` at `ANEFusedSplitMLP.swift:219` | Vector | 20.97 MB | 1 pool |
| `contiguous(xT)` in `prepare` | General | 20.97 MB | 1 pool |
| `xT.asData()`, default `.copy` | host | about 31.46 MB | 1 heap, 10.49 MB |
| 5120-call scatter into the surface | host | 20.97 MB | none |

The `asData` figure counts the `Data(count:)` zero-fill plus the read and
write inside `asDataCopy` (`MLXArray+Bytes.swift:179-191`).

Input path after step S1:

| Step | Kernel | Traffic | Allocations |
|---|---|---|---|
| `x.asType(.float16)` | Vector | 20.97 MB | 1 pool |
| `contiguous(xT)` in `prepare` | General | 20.97 MB | 1 pool |
| `xT.asData(access: .noCopyIfContiguous)` | host | 0 | none |
| one bulk memcpy into the surface | host | 20.97 MB | none |

GPU traffic and GPU allocations do not change. Host traffic drops from about
52.4 MB to 20.97 MB, memcpy calls drop from 5121 to 1, and one 10.49 MB heap
allocation per dispatch disappears.

Across 64 layers at bucket 1024, step S1 removes about 2.01 GB of CPU-serial
traffic, 327,616 memcpy calls, and 671 MB of transient heap allocation.

Input path after step S4, the custom staging kernel:

| Step | Kernel | Traffic | Allocations |
|---|---|---|---|
| one transposing cast straight into the surface | custom Metal | 20.97 MB | none |
| host | none | 0 | none |

Step S4 removes a further 20.97 MB of GPU traffic, 20.97 MB of host traffic,
and two pool allocations per dispatch. Across 64 layers that is about 1.34 GB
of GPU traffic and 1.34 GB of host traffic.

Output path today, including the add that consumes it:

| Step | Kernel | Traffic | Allocations |
|---|---|---|---|
| `contiguous(real.transposed(1,0))` in `readZeroCopy` | General | 20.97 MB | 1 pool |
| `anePartial.asType(gpu.dtype)` | Vector | 20.97 MB | 1 pool |
| `anePartial + gpu` | binary | 31.46 MB | 1 pool |

Step S3 offers one measured candidate that removes 20.97 MB and one allocation
from this path. It is not a deterministic win, because it hands the add a
column-major operand. See step S3.

### Byte counts converted to time

Byte counts alone hid the size of this work in the draft plan. The conversion
below uses only numbers already recorded in the tree.

`docs/perf/ane-zero-transfer-spec.md:56-65` prices the same two fixes that step
S1 makes, at a 1.31 MB staged payload, at roughly 60 to 100 microseconds
combined. This lane stages 10.49 MB, which is 8 times as much, and the work is
linear in payload. That gives roughly 0.5 to 0.8 milliseconds per dispatch and
31 to 51 milliseconds across 64 layers.

`docs/perf/qwen38-flash-2026-09.md:359` records a cold dense prefill of 5.402
seconds at 732 tokens. Step S1 is therefore about 0.6 to 0.9 percent of that
prefill. Step S4 is expected to be about the same again on the input side.

Both figures are inherited estimates carried across a payload change. They are
not measurements of this lane. The microbench in step S9 exists to replace
them. State them with that label or not at all.

### Why the padding is the larger target

`Qwen35.swift:2520-2531` pads the MLP input up to
`ANESplitMLPCache.bucketedSequenceLength(tokens)` and passes the padded array
into `ANEFusedSplitMLP.callAsFunction`. Inside, `gpuPartial(x)`
(`ANEFusedSplitMLP.swift:223` and `:268-283`) runs on the same padded rows, and
the ANE program is compiled at the padded length. Both engines compute the
padding.

At 732 tokens the bucket is 1024, so 28.5 percent of all MLP work is
discarded. At 1032 tokens the bucket is 2048, so 49.6 percent is discarded.
Those are the two prompt lengths this project measures.
`docs/perf/qwen38-flash-2026-09.md` records the dense MLP as the dominant term
in the tower, so the tower-level waste is roughly 20 percent at 732 tokens and
roughly 35 percent at 1032 tokens.

The staging work above is worth under 1 percent of the same prefill. The
comparison decides the target.

### Why the lane's speed evidence does not support a value claim

The `1.17 to 1.21x MLP speedup` in the header of `Qwen35ANESplitOffload.swift`
predates the descriptor-identity fix, so it was measured on a build where 63 of
64 layers ran the wrong compiled program.
`docs/perf/ane-hybrid-collapse-2026-09.md` withdraws the companion prefill
claim and keeps it withdrawn. `ANEFusedSplitSpeedTests` defaults to
`MLXFAST_SEQ_LEN=512` and builds the split directly, so it never exercises
`ANESplitMLPCache` and never pays the padding. The only dense-lane prefill rows
in `docs/perf/qwen38-flash-2026-09.md:468-470` are compile-bound at 31.83 and
34.57 seconds, and the document says so.

There is no valid post-fix dense-lane comparison in the tree. Phase 2 of the
measurement protocol is what produces one. Until it exists, no claim that this
lane beats the GPU end to end is supported.

## Variant B is refused

Two independent reasons, either sufficient.

No dispatch path exists. The probe output above shows the ios18 opset refusing
int8 matmul operands, an int8 conv weight, and `conv_quantized` entirely.

The error budget is unmeasured, and the recorded statistics point the wrong
way. Symmetric per-tensor int8 has a relative RMS error of `crest / 439.9`,
from 127 levels and a step size divided by the square root of 12. The draft
plan assumed a crest factor of 5 to 15 and quoted 1.1 to 3.4 percent. The tree
records no crest factor. It records two tensor maxima, input 50 and
intermediate 209 (`docs/perf/ane-hybrid-collapse-2026-09.md:46-49`). The MLP
input is post-RMSNorm, so its per-element RMS is of order 1 times the learned
gain, and a maximum of 50 supports a crest factor far above 15. The honest
statement is that the cost is unknown and probably larger than the draft
claimed.

Reopening needs both of the following. First, an accepted and ANE-placed int8
activation path, found by extending `tools/int8-probe` across other MIL opset
versions and across the quantize and dequantize op family. Second, per-channel
activation statistics produced from the existing capture harness
(`ANEActivationCapture`, `MLX_ANE_CAPTURE_DIR`), which is model-free
post-processing over captures that already exist. Leave the error percentage
blank until that runs.

## Steps

Steps S1, S2, S5, S6 and S7 are the buildable core. Steps S3, S4 and S8 are
gated on evidence they produce themselves. Step S9 is written and not run.
Step S10 is a handoff.

### S1. Host-side staging cleanup in `prepare`

File: `Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEDirectDispatch.swift`

Replace lines 144 to 161. Keep line 142 exactly as it is, because the
reordering the draft plan proposed saves nothing.

```swift
        // `.noCopyIfContiguous` wraps the Metal buffer's contents pointer
        // instead of copying it. `xT` is row-contiguous by construction on
        // line 142, so this never falls back to `asDataCopy()`, which
        // allocated and zero-filled a 10.49 MB `Data` and then traversed the
        // staged buffer a second time to no end. The returned `Data` borrows
        // `xT`'s storage (`deallocator: .none`), so `xT` must outlive it.
        let srcData = xT.asData(access: .noCopyIfContiguous).data
        precondition(srcData.count == inputByteCount,
                     "ANEDirectDispatch.prepare: transposed input is \(srcData.count) bytes, expected \(inputByteCount)")

        // Scatter [IN,S] into the ANE's [IN,paddedS] row-padded input. When
        // the padded stride equals the packed row the two layouts are
        // byte-identical and the whole block is one memcpy. The per-row loop
        // is the fallback for a sequence length that is not 32-aligned. Both
        // consumers take the fast path today, but this is a runtime check,
        // not an assumption about them.
        let inputRowStrideBytes = rowStride * 2
        let inputRowBytes = sequenceLength * 2
        inputSurface.lock(options: [], seed: nil)
        let inputBase = inputSurface.baseAddress
        withExtendedLifetime(xT) {
            srcData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Void in
                let src = raw.baseAddress!
                if inputRowStrideBytes == inputRowBytes {
                    _ = memcpy(inputBase, src, inputDim * inputRowBytes)
                } else {
                    for c in 0 ..< inputDim {
                        _ = memcpy(inputBase + c * inputRowStrideBytes, src + c * inputRowBytes, inputRowBytes)
                    }
                }
            }
        }
        inputSurface.unlock(options: [], seed: nil)
```

Risk. The `withExtendedLifetime` is required, because the borrowed `Data` has
`deallocator: .none` and points into `xT`'s Metal buffer. The precondition on
`srcData.count` catches any future caller whose input is not contiguous, in
which case `.noCopyIfContiguous` silently falls back to a copy that is still
correct. The bulk branch is a strict superset of the loop when the strides are
equal.

Verification, model-free. Step S6 test 3 reads the staged surface back and
compares it against a reference the test computes on the CPU from `x` alone, at
S=1024 and at S=100. `MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test
--force-resolved-versions --filter ANEDirectDispatch` keeps the existing
conv-against-matmul gate. `swift build -c release --force-resolved-versions`
must stay clean.

### S2. The same bulk fast path in `read`

File: same.

In `read(_:)`, wrap the loop at line 237 in the same equal-stride branch.

Note the cost of this step, because it is not free in test power.
`ANEZeroCopyReadTests` validates `readZeroCopy` against `read` at S=512, where
the strides are equal. Today `read`'s per-row loop is a structurally
independent implementation of the stride arithmetic. After this step both sides
take a flat bulk path at equal stride and can agree while both are wrong. Step
S6 test 4 adds a padded case that restores the independence. Land S6 test 4
before or with this step, never after.

### S3. Output-side dtype forwarding, measured, not assumed

Files: `ANEDirectDispatch.swift`, `ANEFusedMLP.swift`.

This step is a candidate, not a saving. Add a `dtype` parameter to
`readZeroCopy` and to `ANEFusedMLP.readOutput`, defaulting to `.float16`:

```swift
    public static func readZeroCopy(_ prepared: Prepared, as dtype: DType = .float16) -> MLXArray {
        ...
        let real = wrapped[0 ..< outputDim, 0 ..< sequenceLength] // [OUT,S] view
        let t = real.transposed(1, 0)                             // [S,OUT] view
        // fp16 asks for no conversion, so `contiguous` is what materialises
        // the surface bytes into MLX-owned storage. A different dtype casts
        // instead, which is ONE Vector pass over the same bytes and which
        // also materialises them. The Vector output keeps the transposed
        // strides, so the caller's add then runs its strided variant.
        return dtype == .float16 ? contiguous(t) : t.asType(dtype)
    }
```

At the split call site, replace `anePartial.asType(gpu.dtype)` with
`aneMLP.readOutput(prepared, as: gpu.dtype)`. That removes one full pass over
10.49 MB and one pool allocation per dispatch.

Risk, stated plainly. The saving is real in traffic and is not obviously real
in time. A column-major operand makes the following elementwise add read one
side with a stride of 1024 elements, which coalesces badly. The strided add can
cost more than the removed pass. Land this step only if the microbench in step
S9 shows the whole lane is faster. Keep the default at `.float16` so every
other call site is unchanged.

Verification. Bit-exactness against today's spelling at [5120,1024], which is
pure MLX and needs no ANE. Then the existing `ANEFusedSplitMLPTests`
concurrent-against-sequential check at its shipped 1e-4 bar.

### S4. The custom staging kernel

Files: `ANEDirectDispatch.swift`, plus a new `ANEStagingKernel.swift` beside
it.

This is the only step that reaches one pass, and it is the real content of the
"make the conventions agree" idea. Build it behind `MLX_ANE_GPU_STAGE=1`, and
only after step S5 answers the coherency question.

Add `ANEDirectDispatch.prepare(..., stagingMode:)` with `.host`, the default,
and `.gpu`. In `.gpu` mode, replace lines 142 to 161 with one Metal dispatch
that reads `x`'s own bf16 buffer and writes the fp16 transpose straight into
the IOSurface.

```metal
// [S,K] bf16 row-major -> [K,rowStride] fp16 row-major, one pass.
// bf16 to fp32 is an exact 16-bit shift, because bf16 is the high half of
// fp32. fp32 to half then rounds to nearest even, which is what MLX's AsType
// applies. The staged bytes are expected to be bit-identical to the host
// path. The test asserts that instead of assuming it.
kernel void ane_stage_transpose_bf16_to_fp16(
    device const ushort *src [[buffer(0)]],
    device half *dst         [[buffer(1)]],
    constant uint &S         [[buffer(2)]],
    constant uint &K         [[buffer(3)]],
    constant uint &rowStride [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= S || gid.y >= K) { return; }
    uint bits = uint(src[gid.x * K + gid.y]) << 16;
    dst[gid.y * rowStride + gid.x] = half(as_type<float>(bits));
}
```

Swift side, in order. Call `eval(x)`. Guarantee contiguity by construction
rather than by a check, because no public check exists: pass
`contiguous(x)` and accept that this is a copy when `x` arrives strided, which
it does not on this lane. Take `x.asMTLBuffer(device: device, noCopy: true)`.
Build the destination with
`device.makeBuffer(bytesNoCopy: inputSurface.baseAddress, length: alloc,
options: [.storageModeShared], deallocator: nil)`, where `alloc` is
`max(65536, (inputDim * rowStride * 2 + 65535) & ~65535)`, matching
`makeSurface` at line 56. Verify both base pointers are 16384-aligned and that
`alloc` is a page multiple, and fall back to `.host` with a one-line log if
not. Encode, `commit()`, `waitUntilCompleted()`. Bracket the dispatch with
`lock` and `unlock` only if step S5 arm 1 fails and arm 2 passes.

Do not use `x.contiguousToDimension()` as a guard. It is declared without an
access modifier at `MLXArray+Bytes.swift:13` and is internal to module MLX, so
it does not compile from MLXLLM. There is also nothing to detect with it:
`asMTLBuffer(device:noCopy:)` does not return nil on a strided array. It falls
through to `asDataCopy()` and a fresh device buffer
(`MLXArray+Metal.swift:24-34`), which silently reinstates the copy this step
exists to delete while every correctness test still passes. Log once per
process whenever `.gpu` falls back to `.host`.

Correct the claim the draft plan made about this step. The caller-thread CPU
cost does not go to zero. It becomes `commit()` plus `waitUntilCompleted()`,
which is a full command-buffer round trip. At `ANEFusedSplitMLP.swift:219` that
lands before `ConcurrentEngines.run`, so it serialises a GPU round trip ahead of
the work it was supposed to overlap. Whether step S4 is a win is a measurement,
not an arithmetic result.

Pin the ordering invariant in a comment at the staging site. `eval(x)` before
encoding and `waitUntilCompleted()` before returning `Prepared` are the only
things that stop MLX recycling `x`'s buffer under an in-flight read, and the
only things that make the GPU write visible before the ANE reads. MLX allocates
with `MTL::ResourceHazardTrackingModeUntracked`
(`backend/metal/allocator.cpp:18-19`), and the `bytesNoCopy` alias is a separate
`MTLBuffer` object, so Metal performs no cross-object hazard tracking. Step S5's
evidence is scoped to that exact ordering. It does not transfer to any
asynchronous or fence-based variant, including the tiled transpose named below.

Add an always-on self-check, because the failure mode is silent stale data on a
private DMA boundary and 20 test iterations are not a guarantee. On the first
`.gpu` dispatch in a process, stage the same input by both routes and compare
the surface bytes. Refuse `.gpu` for the process lifetime on any mismatch. Add
`MLX_ANE_GPU_STAGE_VERIFY=1` to do that on every dispatch. `.gpu` must not
become the default on step S5 evidence alone.

Follow-up, not part of this step. The kernel above is uncoalesced on the write
side. A 32 by 32 threadgroup-tiled transpose through `threadgroup half
tile[32][33]` is the standard fix, and it must be measured against the naive
one rather than assumed better.

### S5. The coherency experiment

File: `Tests/MLXFastTests/Model/ANEGPUWriteCoherencyTests.swift`, new.

Model-free, correctness only, safe under GPU contention. Gate behind
`MLXFAST_ANE_COHERENCY=1` on top of `MLXFAST_RUN_MLX_RUNTIME_TESTS=1`. Import
`@testable import MLXLLM`, not `MLXFastModel`. `Prepared.inputSurface`,
`outputSurface` and `rowStride` carry no access modifier
(`ANEDirectDispatch.swift:89-93`), so they are internal to MLXLLM and a
`@testable import MLXFastModel` cannot reach them.

Protocol, two arms, 20 iterations each.

1. Build the synthetic conv program the way `ANEZeroCopyReadTests` does, at
   K=5120, F=2176, S=512, where `rowStride` equals S.
2. Compute the CPU-staged reference once through `ANEDirectDispatch.runConv`.
3. Per iteration, call `prepare`, then poison the whole input surface with a
   fresh byte pattern, `memset(base, UInt8(0xA0 &+ i), allocSize)`. Recompute
   `allocSize` as `max(65536, (K * rowStride * 2 + 65535) & ~65535)`, because
   `Prepared` does not store it.
4. Author the surface from the GPU instead, following step S4's ordering. Arm 1
   uses no `IOSurfaceLock` cycle. Arm 2 brackets the dispatch with `lock` and
   `unlock`.
5. Call `evaluate`, then `readZeroCopy`, and require a max absolute difference
   of exactly 0 against the reference.

Assert 16384-byte alignment of both base pointers before step 4 and skip loudly
rather than silently if it fails.

Decision rule, fixed before running. Arm 1 exact on 20 of 20 lets step S4
proceed with no lock cycle. Arm 1 fails and arm 2 exact on 20 of 20 lets step
S4 proceed with the lock cycle, whose cost is charged against the 20.97 MB of
host traffic it replaces. Both arms fail, and step S4 is abandoned. Step S1
then stands as the final state, and this document says so.

Record what the experiment does not cover. `readZeroCopy` already shows a GPU
kernel reading ANE-written surface memory through an MLX-adopted `bytesNoCopy`
buffer with no lock, so only the GPU-write to ANE-read direction is open.

### S6. Tests with power to detect this plan's own defects

File: `Tests/MLXFastTests/Model/ANEStagingEquivalenceTests.swift`, new. Import
`@testable import MLXLLM`.

Test 1, the negative control, ungated and needing no ANE. It asserts the fact
that the draft plan got backwards, so the refuted mechanism cannot return:

```swift
    @Test("a transposed array stays flagged contiguous, so the cast does not transpose")
    func castDoesNotFuseWithTranspose() {
        let x = probeInput(1024, 5120, seed: 7)          // [S,K] bf16
        let t = x.transposed(1, 0).asType(.float16)      // Vector copy
        eval(t)
        #expect(t.shape == [5120, 1024])
        // If MLX ever starts fusing, this flips and the byte accounting in
        // docs/perf/ane-unified-activation-plan.md must be rewritten.
        #expect(!isRowContiguous(t),
                "AsType now produces a row-contiguous transpose; re-derive the byte table")
    }
```

Spell `isRowContiguous` from the public surface, by comparing
`t.strides` against the contiguous strides for `t.shape`.

Test 2, bit-exactness of both spellings, ungated. Assert
`contiguous(x.transposed(1,0)).asType(.float16)` and
`contiguous(x.transposed(1,0).asType(.float16))` differ by exactly 0. This
records that the reordering is safe and worthless, which is the finding.

Test 3, staged bytes against an independent CPU reference. Gated on the ANE.
Call the production `ANEDirectDispatch.prepare`, then read the input surface
back and compare every element against `x.asType(.float16)` computed on the
CPU. Run at S=1024, which takes the bulk path, and at S=100, which takes the
per-row fallback. Extend it with a `stagingMode` argument when step S4 lands,
so the GPU-staged surface faces the same reference.

Test 4, the padded zero-copy read. Gated. At K=512, F=256, S=100, require
`read` and `readZeroCopy` to be bit-identical. This is the independence that
step S2 removes at equal stride.

Test 5, the silent-copy shape class. Gated. Choose a shape where `outputDim *
rowStride * 2` is not a multiple of 16384, for example F=100 and S=100, giving
12800 bytes. `MetalAllocator::make_buffer` returns a null Buffer when
`newBufferWithBytesNoCopy` refuses the length
(`backend/metal/allocator.cpp:258-269`), and `array::array(void*, ...)` then
mallocs, copies, and runs the finalizer at once
(`array.cpp:89-110`). Require `readZeroCopy` to equal `read` bit-exactly there.
Nothing in the tree tests that path today, and the output-side byte accounting
assumes it never happens.

The probe generator. Do not use the draft plan's `arange * 1e-4 + noise`. In
bf16 the ULP at magnitude v is v/256, so the ramp is swamped above about 0.03
and the noise above about 256. Simulation of that exact construction at
[1024,5120] gives 2766 distinct values across 5.24 million elements, with a
median of 6 distinct values per row. bf16 cannot carry a distinct value per
position at these sizes, so build discrimination structurally instead. Use
`value(t, c) = f(t) * g(c)` with `f` and `g` on well-separated bf16 grid points,
and keep the dynamic range near the measured one, where the maximum activation
is 50. Exact-equality assertions survive a degenerate generator. Any
tolerance-based reuse does not.

Every test must call the production entry points. The draft plan's first two
tests re-implemented both spellings inline, so an axis or offset error inside
`ANEDirectDispatch` would have passed them.

### S7. Memory diagnostics around every step

`docs/perf/qwen38-flash-2026-09.md:590-602` records reproducible process-memory
corruption under `MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test` in a debug build,
and concludes that the source looks like a pre-existing problem in the
`ANEDirectDispatch` and `ANEInMemoryModel` bridging, whose symptom moved with
unrelated allocation-size changes in the same function. Step S1 deletes a 10.49
MB per-dispatch heap allocation. Step S4 replaces a pool allocation with a
`bytesNoCopy` mapping. Both are exactly that class of change.

Bit-exactness of a returned tensor is not evidence of heap health. A corrupted
heap that misses the compared tensor passes every test above. So run, before
and after steps S1 through S3 and again after step S4:

```bash
MallocScribble=1 MallocPreScribble=1 MallocGuardEdges=1 \
MallocCheckHeapStart=1 MallocCheckHeapEach=1 \
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions --filter ANE

swift test --sanitize=address --force-resolved-versions --filter ANE
```

Record the result as a named artefact. A change in crash behaviour on either
side of the diff is a finding to investigate, not noise to retry past.

### S8. The token-major acceptance probe

File: `tools/int8-probe/main.swift`, extended. The harness already asks the
compiler directly and reports `MLComputePlan.deviceUsage`.

Add two cases. First, an fp16 token-major matmul at the real dense shape,
`x<fp16,[S,K]>` against `w<fp16,[F,K]>` transposed, at S=1024, K=5120, F=5440.
Second, the same as a SwiGLU-down chain of three matmuls and a multiply.

State the prize honestly, because the draft plan oversold it. Token-major does
not fix the padding. The caller zero-pads before the call, and the ANE program
is compiled at the padded length, so the padded rows are computed either way.
What token-major would remove is the output-side de-transpose, whose value is
the same measured question as step S3.
`docs/perf/ane-zero-transfer-spec.md:92-127` already records four unknowns
against a matmul spelling, and unknown 1, whether a matmul program compiles
through the bare `initWithNetworkText:` path, is exactly what this probe
answers.

Decision rule. Every non-const op reports `NeuralEngine` in both cases, and the
question moves to the tiling redesign as an input. Any op reports CPU or GPU,
and the conv spelling stays.

Read a positive as the compiler's preferred device, which is a plan and not an
execution trace. It licenses a measurement, not a conclusion.

### S9. The microbench, written and not run

File: `Tests/MLXFastTests/Model/ANEStagingSpeedTests.swift`, new.

Gate behind `MLXFAST_ANE_STAGING=1` on top of `MLXFAST_RUN_MLX_RUNTIME_TESTS=1`.
Read `MLXFAST_SEQ_LEN`, default 1024, and `MLXFAST_TIMING_ITERS`, default 15.
Report the median with the interquartile range beside it, because
`docs/thermal-variance-investigation.md` recommends the median over the
minimum, and this repository's gated coefficient of variation runs 3.5 to 14.4
percent.

Four arms, each a separate timed loop in the same process, interleaved:

- A, `prepare` alone at [S,5120] bf16, staging only.
- B, `prepare` plus `evaluate` plus `readZeroCopy(as: .bfloat16)`, one full
  dispatch.
- C, `ANEFusedSplitMLP.callAsFunction` at hidden 5120, intermediate 17408,
  fraction 0.3125, with synthetic weights.
- D, the same shape with `aneFraction` 0.0, the all-GPU reference.

Arms C and D are what decide step S3, because only they contain the add.

Write the hazard into the file header so a later editor does not reintroduce
it. Do not reuse a `Prepared` across timed repeats. Do not materialise an ANE
read inside a tight loop. Do not batch independent GPU calls into a single
`eval`. All three reproducibly corrupted process memory under this project's own
required debug-build test command.

Print the counted staged bytes per dispatch unconditionally and without timing,
so the byte-level claim stays checkable on a busy machine.

### S10. Handoff: the bucket padding

STATUS 2026-09-03, after this plan was written: partly done in commit
`00d4e270`. The controller took ownership of both files and found a cheaper
first step than either candidate below. Only the ANE program is fixed-shape,
so only the ANE leg needs padding. `gpuPartial` accepts any row count, and at
`MLX_ANE_FRACTION` 0.3125 the GPU owns 11968 of 17408 intermediate channels.
Moving the padding into `ANEFusedSplitMLP.padForANE`, scoped to the ANE leg,
cuts the discarded fraction from 28.5 to 11.1 percent at 732 tokens and from
49.6 to 23.5 percent at 1032, without retiling anything. The remaining
paragraphs of this step describe the retiling that would take it further, and
that work is still open.

Do not edit `Qwen35ANESplitOffload.swift` or `Qwen35.swift`. Hand the owner
these numbers.

`ANESplitMLPCache.bucketedSequenceLength` rounds to a power of two floored at
128. `Qwen35.swift:2520-2531` pads to that bucket and passes the padded array
to both partials, and the ANE program is compiled at the padded length. So both
engines compute the padding.

| Prompt tokens | Bucket | Discarded MLP work |
|---|---|---|
| 732 | 1024 | 28.5 percent |
| 1032 | 2048 | 49.6 percent |

Two candidate fixes, with their costs.

Rounding to a multiple of 128 instead of a power of two cuts the waste at 732
tokens to 4.7 percent. It raises the distinct bucket count from 5 to 16, and
each compiled program costs roughly 100 MB of ANE daemon cache, so 64 layers
times 16 buckets is on the order of 100 GB of root-owned cache. Probably not
affordable.

Fixed-tile keying is the alternative, and
`docs/perf/qwen38-flash-2026-09.md:455-475` already names it as the property to
keep. One program per layer over a fixed tile of 128 or 256 rows, dispatched
`ceil(S / tile)` times, caps the waste near 5 to 12 percent, removes the
per-length recompile entirely, and needs no larger program cache. The MoE lane
already works this way.

Note the coupling. Finer tiles mean more dispatches per layer, and every
per-dispatch fixed cost that steps S1 through S4 remove then applies more
times. At a 256-row tile the current per-row scatter would run 5120 memcpy
calls of 512 bytes each, six times per layer at 732 tokens. The staging work in
this plan is the supporting optimization sized to the tiling win, not a
standalone one.

## Accuracy plan

The tolerances below are fixed before any measurement and are not to be
renegotiated after seeing a result.

### Class 1: byte-movement changes. Bar: bit-exact, max absolute difference 0

Steps S1, S2, S3 and S4 move the same values through fewer passes.
`.noCopyIfContiguous` wraps the bytes `.copy` would have duplicated. A bulk
memcpy at an equal stride writes the same bytes as the per-row loop. A Vector
cast produces the same values as a General copy followed by a cast. Step S4's
bf16 to fp16 conversion is an exact 16-bit shift followed by the same
round-to-nearest-even. Any non-zero difference is a defect.

Two instruments enforce it. The pure-MLX equivalence tests in step S6 need
neither ANE nor model. The byte-level readback in step S6 test 3 proves the
staging is correct rather than merely self-consistent, which is the check the
specification asks for and which no existing test performs.

Step S4's rounding claim is an expectation about two independent
implementations. Step S6 test 3 asserts it. A tie-case mismatch means fall back
to `.host` and report it, not loosen the bar.

### Class 2: anything that changes arithmetic. Bar: relative, and no worse than today

No step in this plan changes arithmetic. If the step S8 follow-up ever
respells the ANE program, the accumulation order inside the ANE changes and
bit-exactness stops being available. Two pre-registered conditions, both
required.

First, mean relative error against an fp32 dense reference below 2e-2, which is
what `ANEGemmBench.correctnessTolerance` and `Qwen4ExpANEDenseLaneTests`
already enforce.

Second, against the existing conv-spelled program on the same captured
activations, per-layer max absolute error divided by per-layer max absolute
reference must not exceed 1.25 times what today's configuration produces. Use
the relative form, not an absolute constant. The recorded in-situ table
(`docs/perf/ane-hybrid-collapse-2026-09.md`) shows the absolute error tracking
each layer's own output scale: 0.03 on a scale of 2.0 at layer 1, 0.24 on 8.3
at layer 31, 6.0 on 552 at layer 63. That is about 1 percent at every depth. A
fixed absolute bar of 0.09375 is 0.9 percent of layer 0's scale and 0.017
percent of layer 63's, so it means a different thing at every depth.

Correct two misquotations the draft plan made. `ANEFusedSplitMLPTests`
`meanTolerance` is 0.02, not 0.01
(`Tests/MLXFastTests/Model/ANEFusedSplitMLPTests.swift:128`). Both of its
real-shape cases run at `aneFraction` 0.125, giving F=2176, and at S=512. They
hold a 40 percent scale proxy, not the shipped `ANESplitConfig.fraction` of
0.3125. Add a case at the shipped fraction and at a real bucket before citing
that file as the shipped path's gate.

There is no per-layer composition budget, and a per-layer bar satisfied 64
times does not supply one. State the end-to-end budget separately, and measure
it with the goldens below.

### Class 3: scheduling. Bar: the shipped 1e-4

`sequentialCallAsFunctionForTesting` against the concurrent path stays at 1e-4
and must keep passing after step S3 touches the concurrent path.

### The instruments that have caught real failures on this lane

Per-layer max-abs tolerances did not catch the descriptor-identity collision.
`docs/perf/ane-hybrid-collapse-2026-09.md` states it directly: the synthetic
single-program tests passed at 1 ULP while every layer after the first computed
with the wrong weights. Two instruments did catch it, and both are required
here.

`ANERealActivationProbeTests` runs the fused program on captured real
activations against an fp32 reference. It is model-free and needs only
`MLX_ANE_CAPTURE_DIR`.

The goldens table is the end-to-end signal: distinct-token ratio and repeated
8-gram count over 200 greedy tokens on the four public prompts.

Record the current baseline now, so a later change is not blamed for it and so
a healthy per-layer table is not mistaken for a healthy lane.

| Prompt | GPU distinct | hybrid distinct | repeated 8-grams |
|---|---|---|---|
| README | 0.41 | 0.40 | 13 |
| qwen-prefill-research-plan | 0.48 | 0.56 | 0 |
| qwen-mtp-go-live-runbook | 0.56 | 0.20 | 82 |
| private-benchmark-security | 0.55 | 0.54 | 0 |

The runbook row is a generation collapse on one of four prompts, present after
both fixes, in the configuration this plan invests in. The same document's
per-layer table calls that configuration healthy. Any accuracy argument for
this lane must state that row.

### What is true about fp16, and what is not

The bf16 to fp16 cast is exact for normal values, because bf16 carries 7
mantissa bits and fp16 carries 10, so the bf16 mantissa is a subset. The range
argument covers the high end only: the measured maximum activation is 209
against an fp16 maximum of 65504. At the low end fp16 has the narrower
exponent, so bf16 values below about 6.1e-5 become fp16 subnormal and values
below about 6e-8 flush to zero. That is negligible at the measured statistics
and it is the correct direction of the argument.

Do not claim the ANE prefix is more accurate than the 4-bit GPU path it
replaces. `ANESplitConfig.bf16WeightsPath` defaults to nil
(`Qwen35ANESplitOffload.swift:88-94`), so `ANEBF16WeightSource.shared` is nil
and `ANEFusedSplitMLP.init` takes the `else` branch at
`ANEFusedSplitMLP.swift:90-97`, building the prefix from
`ANEWeightPrep.dequantizeFP16` of the same 4-bit weights. The prefix carries
identical weight information in a wider container. The fp16 lane is also not
error-free: the recorded GPU fp16 column runs 0.0002 to 0.062 max absolute
error against fp32.

Step S3 makes one precision choice rather than preserving one, and it must not
be made silently. Reading the ANE partial back as bf16 rounds fp16's 10
mantissa bits to 7 before the add, which is what `anePartial.asType(gpu.dtype)`
already does today. Reading it back as fp32, or leaving it fp16 so MLX promotes
the mixed add, keeps those bits through the sum for one extra pass. Measure the
fp32 alternative on the real-activation probe before fixing bf16 as the API
default.

## Measurement protocol

Nothing below runs while another agent holds the GPU. Steps S5, S6 and S8
produce verdicts rather than durations and may run at any time.

### Phase 0, now, under contention. Correctness only, no numbers reported

1. `swift build -c release --force-resolved-versions` after each of steps S1
   through S3 individually.
2. `swift test --force-resolved-versions --filter ANEStagingEquivalence`, which
   runs tests 1 and 2 with no ANE and no model.
3. `MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions
   --filter 'ANEStagingEquivalence|ANEDirectDispatch|ANEZeroCopyRead|ANEFusedMLPTests|ANEFusedSplitMLPTests'`.
4. The step S7 guard-malloc and Address Sanitizer runs, before and after.
5. `MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_ANE_COHERENCY=1 swift test
   --force-resolved-versions --filter ANEGPUWriteCoherency`. Record which arm
   passes. That is the step S4 gate.
6. The step S8 probe. Record the placement lines.
7. `ANERealActivationProbeTests` on existing captures, before and after.

### Phase 1, when the machine is free. The timed run

1. Confirm no model-holding process is resident with `ps aux | grep -i
   mlxfast`. One at a time, always.
2. `./tools/host-quiet-gate.sh` must pass. It samples GPU power over an
   8-second window and separately asserts screensaver `idleTime 0`. The second
   check is what defends against the 300-second screensaver cycle, so a passing
   window alone is not sufficient.
3. `swift build -c release --force-resolved-versions`.
4. Run step S9, one arm per fresh process, at S in 128, 256, 512, 1024 and
   2048. The buckets 1024 and 2048 are what the two measured prompt lengths,
   732 and 1032, round up to. 128 is `ANESplitConfig.minSequenceLength`.
5. Report the median of 15 with the interquartile range beside it. If the
   observed spread is wider than about a 10 percent relative half-width at 95
   percent confidence, extend the repeat count rather than reporting the narrow
   claim.
6. Compare before against after by checking out the parent commit into a second
   working copy. Never interleave two builds in one tree, and never put the two
   arms in one process.
7. Re-run the quiet gate between the before and after sets. Discard a set that
   fails it afterwards.

Arms C and D decide step S3. If the strided add costs more than the removed
pass, drop step S3 and keep the default spelling.

### Phase 2, when a model-holding run is permitted

Prefill wall time is the only figure that turns the byte counts into a claim
about the model, and the tree contains no valid post-fix dense-lane comparison.

Run the dense-tower prefill with `MLX_ANE_DIRECT=1` at a prewarmed bucket, once
on the parent commit and once on this branch, on a cool machine. Instrument the
per-phase split: attention, gated delta, MLP prefix staging, MLP prefix ANE
evaluate, and MLP suffix GPU. That separates the staging share from the compute
share.

The same run answers two open questions. It gives the wall-time share of the
split-MLP prefix, of which only the FLOP share is known. It also gives the
first post-fix answer to whether this lane beats the all-GPU path at all.

Run the goldens afterwards and compare against the baseline table above.

## What was dropped, and why

**Variant B, int8 activations.** Refused on the probe output above. The ios18
opset that `ANEMILBuilder` emits rejects int8 matmul operands, an int8 conv
weight, and `conv_quantized`. The error budget is separately unmeasured, and
the recorded tensor maxima point to a larger cost than the draft plan assumed.
Reopening conditions are in "Variant B is refused".

**Reordering `contiguous` and `asType` to fuse the transpose into the cast.**
Refuted in "The transpose does not fuse into the cast". Both orders run the
same two kernels. The draft plan's headline, 73.40 MB down to 41.94 MB and a 43
percent cut, does not occur, and its byte table priced fp16 at 1 byte per
element.

**Dropping the caller-side `x.asType(.float16)` at
`ANEFusedSplitMLP.swift:219`.** No saving, and a real hazard. With that cast
removed and `prepare` unchanged, the General copy runs on bf16 and the Vector
cast runs after it, which is more traffic than today, not less. The draft plan
avoided this only through step ordering, while presenting the steps as
independently landable and verifying each for bit-exactness alone, which cannot
catch it.

**Teaching the vendored MLX allocator to adopt a caller-supplied
`MTLBuffer`.** Unnecessary rather than impossible. Adoption already exists at
`MetalAllocator::make_buffer(void*, size_t)`
(`backend/metal/allocator.cpp:258-269`) and is pool-safe, because adopted
buffers free through `release()`, which never touches `buffer_cache_`. What
does not exist is a way to make an MLX compute op write its output into a
chosen buffer. Donation is a per-primitive `use_count() == 1` heuristic, not an
addressable override. Step S4 sidesteps the question by writing the surface
with its own kernel through the public `asMTLBuffer(device:noCopy:)`.

**Making the GPU residual stream channel-major.** Not expressible for
quantized weights. `quantizedMM` always places the quantized operand on the
right and computes `x @ w.T` or `x @ w`, so a channel-major output has no
spelling.

**Further work on the Qwen4Exp MoE projection lane.** It dispatches one
projection per layer per micro-batch, which caps the lane at 2.7 to 8.1 percent
of prefill even for a free ANE, while the micro-batching that manufactures its
overlap costs 499 to 986 milliseconds against a 64 to 193 millisecond prize.
Steps S1 and S2 improve it incidentally, because it shares `ANEDirectDispatch`.
That is a side effect, not a reason.

**Making the whole-expert-MLP offload the first target.** Deferred, not
dropped. It is the larger ceiling at 44 to 48 percent of MoE-tower FLOP, with
`r` measured at the correct expert shapes. Four things put it second. Its own
`r` table excludes the ANE output-materialisation cost while including the
GPU's, so every reading is biased upward by an unmeasured amount.
`ANEMILBuilder` emits an fp16 weight const only, so int8 and int4 const
emission is unbuilt. At fp16 the expert split is memory-capped near f = 0.12,
giving a 1.14x ceiling. The per-token top-10 routing split across two engines is
undesigned.

## Open questions

1. Does the private ANE DMA path observe a GPU-authored IOSurface without a
   lock cycle? This gates step S4 alone. The failure mode is stale data, not a
   fault, so a single passing run is not an answer. Step S5 answers it at 20
   iterations with a fresh poison pattern each time, and step S4 keeps a
   first-dispatch self-check regardless of the result.
2. Does the strided add in step S3 cost more than the pass it removes? Arms C
   and D of step S9 answer it. A column-major operand reads with a stride of
   1024 elements.
3. Is step S4's bf16 to fp16 conversion bit-identical to MLX's `AsType`?
   Expected yes, asserted by step S6 test 3 rather than assumed.
4. What is the wall-time share of the dense split-MLP prefix in a real Qwen35
   prefill, and does the lane beat the all-GPU path at all after both fixes?
   Only the FLOP share is known. Phase 2 answers both.
5. Does `ios18.matmul` with fp16 token-major operands compile through the bare
   `initWithNetworkText:` path and place on the ANE? Step S8 asks. A positive
   earns a measurement, not a conclusion.
6. Is `MLX_ANE_FRACTION = 0.3125` still the right split once staging is
   cheaper, and once the lane is tiled? Sweep it after the staging work lands
   and after the tiling decision, never before, or it is tuned against a cost
   that no longer exists.
7. Which of the two padding fixes does the owner of
   `Qwen35ANESplitOffload.swift` take? The arithmetic is in step S10. The answer
   sets how many dispatches per layer the staging path must serve.
