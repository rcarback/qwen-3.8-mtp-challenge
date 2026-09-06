---
name: ane-gpu-offload
description: "Patterns for offloading dense projections from the GPU to the Apple Neural Engine (ANE) on Apple Silicon, and for the GPU/ANE interaction that decides whether it wins. Covers the fp16/int8/int4 weight programs (exact MIL text and blob format), zero-copy IOSurface I/O, the 126-program per-process limit and the procedure-bank / multifunction workarounds, and the full ANE-dispatchable MIL operation set with a per-chip capability table. TRIGGERS: any work on the ANEOffload path, Qwen4ExpANE* / ANEDirectDispatch / ANEMILBuilder, MLX_ANE_DIRECT, deciding whether to run a projection on the ANE vs the GPU, ANE weight quantization, IOSurface handoff, the ANE program-count limit, or ANE/Core ML MIL programs. This is local M4-fork knowledge, verified in hardware on M4 Max / macOS 26.5.2; it is not part of the ranked contract."
---

# ANE / GPU offload

The Apple Neural Engine is a fixed-function fp16 matrix-multiply and
convolution accelerator. This skill is the working contract for putting model
projections on it and, more important, for deciding when that is a win against
the GPU. It is built from measurements in this repository, verified in
hardware, not from documentation alone.

## The three facts that govern everything

1. **The ANE computes in fp16 but stores compressed.** The datapath
   reconstructs a compressed weight to fp16 at the multiplier input. It
   accepts int8 and int4 (and palettized) weight storage and dequantizes
   on-chip. fp16 is the compute precision, not the storage requirement. So the
   ANE runs on the same quantized weights the GPU uses; there is no need to
   expand to a bf16 tree. See `references/int8.md` and `references/int4.md`.

2. **The bottleneck is bandwidth, and the handoff, not the multiply.** The
   ANE reads DRAM at roughly 78 to 110 GB/s, the narrowest engine on the chip.
   Its compute beats the quantized GPU on a real projection (measured
   0.92x at `in_proj_qkv [10240x2560]`, S=128), but only if the input and
   output do not cost a copy. Getting the weight bytes and activations to and
   from the ANE is the real work. See `references/zero-copy.md`.

3. **A process holds only ~126 loaded programs.** The ANE daemon fails the
   127th `load` with `Program load failure (0x50004)`, whatever each program's
   size. One shape is one program. A real model has far more than 126
   (shape x layer x dtype x sequence bucket), so the program budget is the
   binding design constraint. See `references/program-limit.md`.

## When the ANE wins, and when it does not

- **Prefill (compute-bound, S >= 128): candidate win.** The ANE compute beats
  the quantized GPU, and the dense projections (attention `q_proj`, gated-delta
  `in_proj`) are dispatchable. The win is bounded because the routed experts
  cannot run on the ANE (they need a dynamic gather) and dominate prefill
  FLOPs.
- **Decode (S=1, latency/bandwidth-bound): loses.** The ANE loses at one row
  by 5 to 10 times, with a per-dispatch floor near 0.30 ms. No decode shape
  is worth moving.
- **Concurrency is close to zero-sum.** Running the GPU and ANE at once does
  not beat the GPU alone on this box (GPU alone ~189 GB/s; GPU+ANE
  ~153+64 = 217, but the GPU chain there is host-bound). Overlap helps only
  when the ANE work is genuinely independent of the GPU's, which at decode it
  is not. Re-check this on the target silicon; it moves by generation.

## The program design that follows

Because the program budget binds, do not build one program per
(layer, projection, bucket). Instead:

- **One program per op-type and dtype and bucket, packing many layers as
  procedures.** Attention `q_proj` at fp16 S=512 is one program holding all 12
  attention layers as functions; gated-delta `in_proj` at int8 S=512 is
  another. This collapses the count from hundreds to a handful. The mechanism
  is the multifunction `.mlpackage`, not the bare in-memory bank (which loads
  many functions but dispatches only `main`). See `references/program-limit.md`.
- **Match the weight dtype to the GPU tree.** If the model is stored int8, the
  ANE program holds int8; do not dequantize to fp16 first. This shares the
  representation and keeps the ANE's read narrow. See the dtype sub-skills.
- **Keep every hot-path transfer zero-copy.** Output through
  `MLPredictionOptions.outputBackings` into an IOSurface wrapped straight into
  MLX; input through a surface-backed `MLMultiArray`. See
  `references/zero-copy.md`.

## Sub-skills

- `references/sources.md` — every source this skill rests on and what each
  established, plus the hardware probes. Read this to know what is measured
  and what is inferred.
- `references/fp16.md` — fp16 weight programs. The baseline, confirmed.
- `references/int8.md` — int8 affine weight programs. Confirmed in hardware.
- `references/int4.md` — int4 blockwise weight programs. Exact format; the
  in-memory compile of the iOS18 op is an open item.
- `references/zero-copy.md` — IOSurface input and output, the staging-cost
  analysis, the direct path versus `MLModel.prediction`.
- `references/program-limit.md` — the 126-program limit, the procedure bank,
  the multifunction descriptor, and the per-op-type program design.
- `references/gpu-interaction.md` — the split decision, measured numbers,
  concurrency, and the decode verdict.
- `references/operations.md` — every ANE-dispatchable MIL operation and its
  usage.
- `references/chip-support.md` — which Apple Silicon generation supports what.

## What this skill trusts, and what it does not

The authority on how the ANE behaves is independent reverse-engineering and
measurement, not Apple's tooling. Apple documents Core ML, not the engine; the
Core ML op set is a superset of what the ANE runs, and several ops that are
valid MIL are silently rejected or produce wrong results on the engine. The
sources this skill rests on, and what each established, are listed in
`references/sources.md`. Every quantitative claim here is either from one of
those sources or from a hardware probe in this repository, and the reference
files say which.

coremltools is used for exactly one thing: extracting the byte-exact MIL text
and `weight.bin` layout that the ANE compiler accepts, because the compiler is
opaque and reports only `InvalidMILProgram`. Treat its output as a syntax
sample to match, never as a statement about ANE behavior. Confirm every op and
format in hardware with the probes:
`MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_NO_SANDBOX=1 swift test -c release`.
