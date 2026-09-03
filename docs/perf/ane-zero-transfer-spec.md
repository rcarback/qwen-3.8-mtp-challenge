# ANE staging-transfer fix: final specification

Status: **do-not-proceed** as performance work.
Date: 2026-09-03. Scope: `ANEDirectDispatch.prepare` staging on the Qwen4Exp
MoE ANE lane (FIX A, FIX B, FIX C).

## 1. The finding, first

The MoE ANE lane cannot beat `moe_plain` at any staging cost, including zero.
The prize is smaller than the entrance fee by a factor of 2.6 to 15.

**The prize.** `Qwen4ExpModel.forwardMicroBatched` dispatches exactly one
projection per layer per ANE-served micro-batch
(`Qwen4ExpModel.swift:207-268`). `aneServes` requires a micro-batch of the
full width `m`, so a ragged tail always goes to `gpuProjection`. Verified
shapes: `IN = hiddenSize = 2560`; `OUT = 16384` on the 36 gated-delta layers
(fused `in_proj` = `convDim` 10240 + `valueDim` 6144) and `OUT = 12288` on the
12 attention layers (`q_proj` = 24 heads x 256 headDim x 2). At S=732 with
m=256 the lane makes 96 dispatches carrying 1.93 TFLOP. On the GPU that work
costs 64 to 193 ms, depending on the quantized-matmul rate assumed. The plain
prefill is 2399 ms. **The entire lane is worth 2.7 to 8.1 percent of the
forward, and that is the ceiling a free, instantaneous ANE would reach.**

**The entrance fee.** Micro-batching is not incidental; single-sequence prefill
is strictly serial per layer, so micro-batching is what manufactures the
independent work the ANE overlaps against. `docs/perf/qwen38-flash-2026-09.md`
("Where the ANE lane's cost actually is") measures that structure with GPU
projections and zero ANE dispatches:

| Arm | Structure | Projections | cold A (732) | cold B (1032) |
|---|---|---|---|---|
| `moe_plain` | one fused lazy graph | GPU | 305.1 tok/s | 303.7 tok/s |
| `moe_mb_gpu` | micro-batched | GPU | 252.6 | 235.4 |
| `moe_mb_ane` | micro-batched | ANE | 267.5 | 233.8 |

`moe_mb_gpu` costs 499 ms on prompt A and 986 ms on prompt B against
`moe_plain`, and it never calls `prepare` once.

**Therefore.** FIX A, FIX B, FIX C and any future transfer fix optimize inside
a 64-193 ms box that sits inside a 499-986 ms hole. Staging appears in neither
the numerator nor the denominator of the lane's problem.

## 2. Why the brief's premise does not hold

The 195.6-versus-236.3 and 159.1-versus-221.9 comparisons match no arm recorded
in `docs/perf/qwen38-flash-2026-09.md` -- neither the three-arm table nor the
post-load pair (274.3 / 247.9 ANE against 307.3 / 328.3 GPU). Whatever their
provenance, they put the ANE lane against the plain fused-graph forward, which
attributes the micro-batching harness cost to the ANE. Against the fair
baseline the ANE arm is 161 ms **faster** on prompt A and 30 ms slower on
prompt B.

Only the doc's number set can settle this, because only that set carries an
ANE-free control arm through the same harness.

## 3. The two-sided effect of FIX A and FIX C, honestly stated

Per dispatch at the real shape (S=256, IN=2560, 1.31 MB staged):

- FIX A removes one full-buffer host copy of 1,310,720 bytes inside
  `asDataCopy()`: roughly 30 to 40 microseconds.
- FIX C removes 2,559 of 2,560 `memcpy` call overheads and lets the remaining
  copy stream one block instead of 2,560 fragments of 512 bytes: roughly 20 to
  60 microseconds.
- Combined: roughly 60 to 100 microseconds.

Scaled: 6 to 10 ms at S=732 (96 dispatches), 12 to 19 ms at S=1032 (192).

- **Prompt A:** the ANE arm is already 161 ms ahead of the identical harness on
  GPU. There is no ANE-attributable deficit to remove. FIX A and FIX C widen an
  existing gain by 6 to 10 ms, and the arm stays 499 ms behind `moe_plain`.
- **Prompt B:** the ANE arm's entire net loss is 30 ms over 192 dispatches,
  that is 0.156 ms per dispatch. FIX A and FIX C address 50 to 110 microseconds
  of it. Best case the arm improves by 12 to 19 ms of 4414 ms, about 0.4
  percent, and stays 986 ms behind `moe_plain`.

Neither result changes any ordering.

## 4. The output side, which this specification declines to touch

For completeness, because the choice of surface should be visible: the output
per dispatch is 8.39 MB (16384 x 256 x 2) against 1.31 MB in, on 36 of 48
layers. `readZeroCopy`'s `contiguous(real.transposed(1,0))` plus
`Qwen4ExpModel.readANE`'s `asType(dtype)` fp16-to-bf16 conversion move roughly
34 MB of GPU traffic per dispatch, about 85 microseconds at 400 GB/s. That is
the same order as everything FIX A and FIX C remove from the input side.

It is not specified here because removing it needs a precision change (running
downstream compute in fp16) or a layout change, and neither is proportionate to
a lane whose ceiling is 193 ms.

## 5. FIX B: dropped, twice over

**As literally stated it is impossible.** MIL `conv` fixes dim 1 as the
contraction and channel axis and ties the weight's channel count to `inputDim`.
No permutation of a `[F, K]` weight makes S the contraction axis, so
pre-transposing weights at initialization cannot let MLX's native `[S, IN]` be
fed to a conv. The brief's rationale -- weights are reused across tokens,
activations are not -- is sound but has no layout that realises it.

**The op-family substitute is mis-priced.** Replacing `conv` with `matmul` and
`transpose_y` does **not** remove the eval barrier: `asData(access:)` calls
`self.eval()` unconditionally (`MLXArray+Bytes.swift:211`), and any CPU read of
MLX-owned bytes must evaluate them first. The barrier is intrinsic to CPU-side
staging, not to the transpose. The whole payoff is one 1.31 MB GPU transpose
kernel and its launch, tens of microseconds.

Against that, four unresolved unknowns, each able to sink it:

1. No proof a matmul program compiles through the bare `initWithNetworkText:`
   path that conv was empirically forced onto after a binary-protobuf attempt
   failed with `InvalidCompilationParam` (`ANEMILBuilder.swift:210-220`).
   `tools/ane-mlprogram` uses the binary `MLModelAsset` path, so it does not
   answer the question.
2. No confirmation matmul lands on the ANE rather than falling back silently.
3. The 32-element row-stride IOSurface convention was reverse-engineered for
   conv's spatial tiling and would need fresh empirical rederivation.
4. A changed reduction order is the numeric-reassociation class the house rules
   flag as able to flip near-tie greedy argmaxes.

It also changes shared conventions in `ANEMILBuilder.swift` and
`ANEDirectDispatch.swift` that `Qwen35ANESplitOffload.swift` depends on and
that another agent is editing.

The ceiling in section 1 rules FIX B out regardless of what a probe would find,
so no probe is scheduled.

## 6. What to do instead, in order of measured size

1. **The micro-batch harness.** 499 to 986 ms, and the only thing standing
   between the lane and `moe_plain`. Nothing else on this lane is worth doing
   first.
2. **Measure `r`, the ANE-to-GPU rate on expert-shaped GEMMs at the 256-token
   tile.** The perf document states outright that nothing should be built
   before `r` is known. `ANEGemmBench.sweep` can produce it with no model, at
   the real shapes `(m: 256, k: 2560, n: 16384)` and `(m: 256, k: 2560,
   n: 12288)`. Read `rate` with the documented bias in mind:
   `aneSeconds` omits `readOutput` materialisation and is a lower bound
   (`ANEGemmBench.swift:24-38`).
3. **If transfer volume is genuinely the target, go to the dense lane.**
   `ANESplitMLPCache.bucketedSequenceLength` (`Qwen35ANESplitOffload.swift:
   140-145`) rounds up to a power of two floored at 128, so S=732 pads to a
   1024-row bucket and 28.5 percent of every staged byte and every ANE FLOP is
   discarded padding. That is an order of magnitude more transfer than FIX A
   and FIX C combined, on the lane with 3.7x better amortisation: FLOP per
   staged byte is 8160 on the dense lane (hidden 5120, `MLX_ANE_FRACTION`
   0.3125 giving f=5440, three matmuls against a fixed `[S,5120]` in and out)
   against 2214 on the MoE lane. That file has a concurrent owner; hand the
   finding over rather than editing it.

Separately: none of this can move a ranked score. The lane is headed `LOCAL M4
FORK ONLY ... Never wired into the ranked forward` and is gated by
`MLX_ANE_DIRECT=1`.

## 7. The residual hygiene edit (optional, not performance work)

`ANEDirectDispatch.swift:148-150` claims the per-row scatter "degenerates to
one bulk copy when `sequenceLength == rowStride`". That is false: the loop has
no such fast path, and every real dispatch from both consumers hits exactly
that condition. If the parent wants the comment made true, the steps below are
the correct way to do it. They are justified as correcting a false comment.
They are not justified as performance work and must not be reported as such.

### Step H1 -- FIX C, one bulk copy when the strides are already packed

`Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEDirectDispatch.swift`.

Add inside `public enum ANEDirectDispatch`, above `prepare`:

```swift
    /// Copies `rows` rows of `rowBytes` bytes between two row-strided
    /// layouts. When BOTH strides already equal `rowBytes` the layouts are
    /// byte-identical and the whole block is one `memcpy`; the per-row loop
    /// is the fallback for the ANE's padded 32-element row stride.
    ///
    /// The fast path is a runtime check, not an assumption. Both consumers
    /// happen to guarantee it today (`Qwen4ExpModel` dispatches only full
    /// `microBatch` chunks and routes the ragged tail to `gpuProjection`;
    /// `ANESplitMLPCache.bucketedSequenceLength` is a power of two floored
    /// at 128), but that guarantee lives in caller logic that can change
    /// independently of this file, and copying padded rows contiguously
    /// would produce wrong numerics rather than a crash. Both operands are
    /// checked because `read` has the strides the other way round.
    @inline(__always)
    static func copyRows(
        dst: UnsafeMutableRawPointer, dstRowStrideBytes: Int,
        src: UnsafeRawPointer, srcRowStrideBytes: Int,
        rows: Int, rowBytes: Int
    ) {
        if dstRowStrideBytes == rowBytes, srcRowStrideBytes == rowBytes {
            _ = memcpy(dst, src, rows * rowBytes)
            return
        }
        for r in 0 ..< rows {
            _ = memcpy(dst + r * dstRowStrideBytes, src + r * srcRowStrideBytes, rowBytes)
        }
    }
```

Route `prepare`'s scatter and `read`'s gather through it, and correct the
comment at lines 148-150 to describe what the code now does.

### Step H2 -- FIX A, with the borrow made structurally unable to escape

Same file. The `Data` returned by `.noCopyIfContiguous` is
`Data(bytesNoCopy:count:deallocator:.none)` over MLX's own backing
(`MLXArray+Bytes.swift:217-230`) and holds no reference back to the array, so
it must not outlive it. Declare it inside the closure so no variable exists
outside `xT`'s extended lifetime:

```swift
        // MIL conv input is tensor<fp16,[1,IN,1,S]> = [IN,S] row-major
        // (channel-major); our x is [S,IN], so transpose before writing.
        let xT = contiguous(x.transposed(1, 0)).asType(.float16) // [IN,S], row-contiguous
        // No separate `eval(xT)`: `asData(access:)` evals unconditionally
        // (MLXArray+Bytes.swift:211).
        let inputRowStrideBytes = rowStride * 2
        let inputRowBytes = sequenceLength * 2
        withExtendedLifetime(xT) {
            // `.noCopyIfContiguous` wraps xT's own backing rather than
            // duplicating `inputByteCount` bytes. `contiguous(_:)` (Ops.swift)
            // forces a row-contiguous buffer, so the wrapper branch is the one
            // normally taken. The fallback is `asDataCopy()`, which for a
            // strided array is the PER-ELEMENT scalar path -- measured at 2.4 s
            // for K=5120,S=512 (ANEMILBuilder.swift:460-466), not merely
            // "slower". It is also the branch actually taken at degenerate
            // shapes: `contiguousToDimension()` (MLXArray+Bytes.swift:13)
            // counts size-1 axes, so at S=1 the [IN,1] result reports
            // non-contiguous and copies. That is today's behaviour either way.
            //
            // The Data is `bytesNoCopy` with a `.none` deallocator and holds no
            // reference back to xT, so it MUST NOT escape this closure. It is
            // declared here so that it cannot.
            let srcData = xT.asData(access: .noCopyIfContiguous).data
            precondition(srcData.count == inputByteCount,
                         "ANEDirectDispatch.prepare: transposed input is \(srcData.count) bytes, expected \(inputByteCount)")
            inputSurface.lock(options: [], seed: nil)
            let inputBase = inputSurface.baseAddress
            srcData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Void in
                copyRows(dst: inputBase, dstRowStrideBytes: inputRowStrideBytes,
                         src: raw.baseAddress!, srcRowStrideBytes: inputRowBytes,
                         rows: inputDim, rowBytes: inputRowBytes)
            }
            inputSurface.unlock(options: [], seed: nil)
        }
```

Do not add any mutating use of `srcData`: `Data.withUnsafeBytes` is read-only
and does not trigger copy-on-write, which is what keeps the scatter unchanged.

### Step H3 -- tests with actual discriminating power

Acceptance criterion for both fixes: **bit-identity of the staged input surface
and of the read output.** Bit-identity of `read` against `readZeroCopy` is not
sufficient -- it compares two readers of the same surface and is blind to any
input-side staging error.

**`ANECopyRowsTests`** (new, pure memory, no ANE, no MLX, no model). Fill per
row and element, not by flat byte index, so a permuted, dropped or
off-by-one-row copy is detectable. Write the source as `UInt16` elements
`UInt16(truncatingIfNeeded: c &* 65_521 &+ k)` for row `c`, element `k`;
65,521 is prime and coprime with 65,536, so all rows differ. Poison the
inter-row gap in the source with a distinct byte when `srcStride > rowBytes`,
and assert the whole destination pad region between rows, not one byte.

Cases: packed (rows 2560, rowBytes 512, both strides 512); padded scatter
(rows 512, rowBytes 2, dst 64, src 2; and rows 256, rowBytes 96, dst 128,
src 96); padded gather (rows 256, rowBytes 2, dst 2, src 64).

The draft's fill `UInt8(truncatingIfNeeded: i &* 31 &+ 7)` has period 256 and
produces exactly **one** distinct row pattern across all 2560 rows of the
headline case, because 31 x 512 is congruent to 0 modulo 256. It would pass a
copy that permuted or dropped rows. Do not use it.

**`ANEStagedSurfaceTests`** (new, ANE-gated, model-free). The only test that
checks the staged bytes are the *correct* bytes rather than self-consistent.
Call `prepare`, lock `prepared.inputSurface` (an `internal let` on `Prepared`,
reachable under `@testable import MLXLLM`), and compare the first
`inputDim * rowStride * 2` bytes against `f16Bytes(contiguous(x.transposed(1,0)))`
scattered at the known row stride. Use random `x` with `S != IN` and `IN` not a
multiple of `S` -- for example S=256, IN=320 (packed) and S=48, IN=320
(rowStride 64, padded) -- so a swapped axis pair, a wrong row stride, an offset
error, or a stale source buffer all change the bytes.

Add the lifetime negative control to the same test: after `prepare` returns,
drop the local reference to the staged array, force several large MLX
allocations and `GPU.clearCache()`, then read the input surface again. The
content must be unchanged, because staging completed inside `prepare`.

**`MLXAsDataAccessTests`** (optional). If kept, it must pin *which* branch
runs, or it proves nothing about FIX A: take the base address of a `.noCopy`
`Data` and of a `.noCopyIfContiguous` `Data` on the same array and require they
are equal, and require the `.copy` base address differs from both. Byte
equality alone passes whether or not the wrapper silently degraded to a copy.

### What not to run

- **Delete `swift test -c release --filter ANEDirectDispatchTests`** from any
  verification list. Every test file in this target uses `@testable import`,
  and SwiftPM does not pass `-enable-testing` in release configuration, so the
  command fails at compile time before any test runs. It is usable only with
  `-Xswiftc -enable-testing`.
- **Do not use Address Sanitizer as the lifetime check.** `xT`'s bytes come
  from MLX's own Metal allocator cache. A premature release returns the buffer
  to MLX's pool (`GPU+Metal.swift:28-39`); the memory stays mapped and
  ASan-valid, and the bytes only go wrong once MLX reuses it. A green ASan run
  would read as proof and would not be one. The negative control in
  `ANEStagedSurfaceTests` exercises the pool-reuse mode instead.

### Step H4 -- verification, all model-free

```bash
swift build --build-tests --force-resolved-versions
swift test --force-resolved-versions --filter ANECopyRowsTests
```

On ANE hardware, additionally:

```bash
swift test --force-resolved-versions --filter 'ANEStagedSurfaceTests|ANEDirectDispatchTests'
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions \
  --filter 'ANEZeroCopyReadTests|Qwen4ExpANEDenseLaneTests|ANEFusedMLPTests'
```

`ANEDirectDispatchTests` covers both branches end to end already: S=32
exercises the new bulk path, S=1 the retained per-row loop, both against
`matmul`. None of these loads the model.

`Tests/` is absent from `benchmark.json` `editablePaths`, so none of these
files would ship in a submission. That is consistent with the lane itself never
being wired into the ranked forward.

## 8. Corrections to statements in the draft specification

- `ANEFusedMLP.makeInput` (`ANEFusedMLP.swift:80-82`) does **not** cast to
  fp16; it forwards `x` unchanged. Only `Qwen4ExpANEProjection.makeInput`
  (`Qwen4ExpANEDenseLane.swift:56-57`) casts. The contiguous branch should be
  justified from `prepare`'s own `contiguous(x.transposed(1,0))` and the copy
  fallback, not from caller dtypes.
- The draft's Step 1 gate used `OUT = 6144`, which is `valueDim` alone and is
  not any offloaded program's output. It understates the predict denominator by
  roughly 2.5x.
- The draft's gate ratio divided by `ANEGemmBench.aneSeconds`, which the bench
  documents as omitting `readOutput` materialisation and therefore as a lower
  bound on per-dispatch cost. Dividing a saving by an understated cost inflates
  the ratio. The decision needs `saved / per-dispatch deficit`
  (1.68 ms per dispatch on prompt A, 0.156 ms on prompt B), and no measurement
  of a standalone dispatch can license reopening FIX B.
- Neither FIX A nor FIX B removes the eval barrier.
