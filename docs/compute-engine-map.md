# Compute engine map: ANE, GPU, and CPU on Apple M4 Max

Scope: the Qwen 3.8 (`qwen3_5_text`) local serve stack. Hidden 5120,
intermediate 17408, 24 Q heads over 4 KV heads, head_dim 256, 48
gated-delta linear-attention layers, 16 full-attention layers, weights
affine 4-bit group-64.

---

## 0. CORRECTION 2026-08-29: the GPU baseline was contaminated

Everything below that compares an engine against "our Metal path" used a
Metal number measured **after other work in the same process**. That number
is wrong by about 2.2x, and correcting it removes the case for every
migration this document recommends.

### The measurement

`tools/gemm-point-sweep.sh`, one shape per process, real `quantizedMM`
(affine 4-bit, group 64), `eval`-forced timing, best of three. [M]

| shape | q4 TFLOPS | bf16 TFLOPS | q4/bf16 |
|---|---:|---:|---:|
| gate/up 1024 x 17408 x 5120 | **13.42** | 14.08 | 0.95 |
| down 1024 x 5120 x 17408 | **13.51** | 14.46 | 0.93 |
| gdn qkv 1024 x 10240 x 5120 | **13.49** | 14.38 | 0.94 |

### What this falsifies

1. **"Our 4-bit path runs at 6.2 - 6.9 TFLOPS."** It runs at 13.4 - 13.5.
2. **"There is a 1.79x quantization penalty."** There is a 1.05x penalty.
   The 4-bit kernel is within 5 - 7 percent of bf16 at the same shape.
3. **"There is a 1.64x aspect-ratio penalty."** There is none. bf16 at our
   real shapes reads 14.1 - 14.5, against 14.7 for the square reference.
4. **"About 2.2x is available by fixing the quantized GEMM."** At most 1.05x
   is available. The GPU kernel work is finished.

### Why the old numbers looked the way they did

`PrefillMatmulCostTests` records the mechanism in its own doc comment: the
square bf16 reference reads 14.69 / 14.60 / 14.66 TFLOPS when its block runs
first in a fresh process, and **6.35 to 6.62 when any other measurement block
ran before it**. Every superseded figure sits inside that second band.

### The cascade into E3

E3 measured all of its cells in one process. Its entire Metal column
therefore carries the same position term, biased consistently against Metal
and so **toward the ANE**. Correcting it inverts E3's headline result:

| cell | E3 as reported | corrected |
|---|---|---|
| tall GEMM, seq 2048 | ANE int8 50.8 ms vs Metal 55.7 ms; ANE wins 1.10x | 365 GFLOP at 13.4 TFLOPS is ~27 ms; **Metal wins ~1.9x** |

**No cell in E3 shows the ANE beating a correctly measured Metal.** E3's
partition recommendation is withdrawn. Its ANE column, its power column and
its sustained-load column stand; only the comparison baseline was wrong.

### Revised priority

| # | item | status |
|---|---|---|
| 1 | GPU 4-bit GEMM | **closed.** 1.05x remains, not worth the correctness risk |
| 2 | CPU SME | **diminished.** 2.0 TFLOPS against a real 13.4, so ~15% if overlapped |
| 3 | ANE | **closed.** Never beats correctly measured Metal at these shapes |
| 4 | Three-engine overlap | **diminished** with items 1 - 3 |
| **5** | **The position effect itself** | **open, and now the largest item** |

The full forward runs at about 6.11 TFLOPS while its own dominant kernel does
13.4 in isolation. That 2.2x now belongs to the forward pass, not to the
kernel. Two readings, and they are not both true:

- The position effect is a **measurement artifact**: 13.4 is the real rate,
  the forward should reach it, and ~2.2x is available with no kernel change.
- The position effect is **mechanical**: any kernel that is not first in its
  process is degraded, a forward pass is thousands of kernels deep, ~6 TFLOPS
  is the true sustained rate, and 13.4 is the artifact.

Known: the effect survives `Memory.clearCache()`, does not survive a process
boundary, and a fresh process reads full speed immediately after a heavy run
in another process, which rules out GPU clock and thermal state. The
mechanism is unidentified. `GPUPositionEffectTests` plus
`tools/position-effect-sweep.sh` probe it by crossing a prelude
(none / cpu / alloc / tinygpu / bandwidth / biggpu) with a compute-bound or
bandwidth-bound measurement, one pair per process.


Host: Apple M4 Max, 12 P-cores + 4 E-cores, 40 GPU cores, 16-core ANE,
128 GB unified memory, 546 GB/s, macOS 26.5.2.

Evidence labels: **[M]** measured by us, **[P]** published with a source,
**[I]** inferred. Numbers without a label are arithmetic on labelled ones.

---

## 1. Summary

There is no fastest engine. There is a fastest engine per operation
class, and the classes differ by more than 20x in how they rank.

| operation class | winner | margin | second |
|---|---|---|---|
| Tall GEMM, out >> inn (MLP gate/up) | **ANE int8** at seq>=2048 | 1.10x time, 4.0x power | Metal |
| Wide GEMM, out << inn (MLP down) | **Metal** | 6.7 - 9.4x | Core ML GPU |
| Attention projections | **either** | within 3-7% | ANE wins power 4.3x |
| Score matmuls, short KV (4096) | **either** | Metal 1.14x | ANE wins power 3.5x |
| Score matmuls, long KV (16384) | **Metal** | 3.2 - 15.6x | Core ML GPU |
| Elementwise, RMSNorm, SiLU | **Metal** | 7 - 8x | tie ANE/CoreML-GPU |
| Draft and decode GEMV (seq 1-8) | **Metal** | 2.4 - 5.7x | CPU (untested, plausible) |
| Gated-delta recurrence | **Metal only** | ANE cannot compile it | - |
| Ops below ~0.2 ms | **CPU** | accelerators are dispatch-bound | - |

The single most important structural fact: **GEMM orientation changes the
ANE rate by 4.7x at identical weight bytes.** [M] Tall GEMMs run at 4.4
TFLOPS fp16; wide GEMMs at 0.94 TFLOPS, same 178 MB of weights, same
FLOP count. This is not a working-set effect. It is the contraction
orientation.

---

## 2. Measured throughput on our shapes

| engine and path | TFLOPS | source |
|---|---:|---|
| GPU Metal, bf16 dense | **14.7** | [M] our GEMM probe |
| GPU Metal, bf16 dense (independent) | 14.8 at 2048^3 | [P] Rigel, same chip |
| GPU Metal, **our 4-bit group-64 path** | **13.4 - 13.5** | [M] one-per-process sweep. The 6.2-6.9 in E3 was position-contaminated |
| GPU via Core ML, fp16 | 4.4 - 5.9 | [M] E3 |
| ANE int8 (W8A16), tall GEMM | **6.7 - 8.25** | [M] E3 |
| ANE fp16, tall GEMM | 4.1 - 7.6 | [M] E3 |
| ANE fp16, attention projections | 3.7 - 7.0 | [M] E3 |
| ANE fp16, wide GEMM | **0.94** | [M] E3 |
| ANE fp16, scores at KV=16384 | 0.28 - 1.9 | [M] E3 |
| **CPU SME2, int8 -> int32** | **4.02** | [P] Jena microbenchmarks |
| **CPU SME2, bf16 or fp16 -> fp32** | **2.01** | [P] |
| CPU SME2, fp32 | 2.01 | [P] |
| CPU NEON, fp32 FMLA | 0.107 | [P] |
| CPU, MLX `_qmm_t_simd` (what we call today) | NEON class | [M] code read |

Two readings follow immediately.

**Our GPU path leaves about 2.2x on the table.** 14.7 bf16 against 6.2-6.9
on the same silicon. The cause is issue-slot dilution: the 4-bit kernel
spends about 0.32 of its issue slots on matrix instructions against about
0.62 for bf16. [M] On M4 there is no dedicated matrix datapath; Metal 4
`matmul2d` beats `simdgroup_matrix` by only 1.05 - 1.21x because both
lower to the same units. [P] Dedicated matrix hardware arrives with M5.

**The CPU is not negligible.** SME2 at 2.0 TFLOPS bf16 is roughly a third
of our Metal path, and 4.0 TOPS int8 is roughly two thirds of it. But we
do not reach it: MLX's CPU quantized matmul is a hand-written float SIMD
loop, which is NEON at about 0.107 TFLOPS. **We are19x below what this
CPU can do on matrix work.** [I]

---

## 3. Power, measured sustained rather than burst

Burst windows flatter the GPU. These are 60-second loops. [M] E3

| engine | sustained watts |
|---|---|
| ANE | 1.4 - 2.3 |
| Core ML GPU | 7.7 - 9.0 |
| Metal (our path) | 9.6 - 12.8 |

Over 60 seconds no engine collapsed. Metal crept 9.6% on the tall GEMM
(52.6 -> 57.7 ms) while drawing 12.8 W. ANE was flat. [M]

Time ranking does not flip under sustained load. What sustained load
changes is the cost of keeping Metal first: 10-13 W continuously against
about 2 W on the ANE. On a thermally limited laptop that is the ANE's
durable argument, and it is an argument about headroom for the work that
stays on the GPU, not about latency.

---

## 4. The ANE

### What it is

A fixed-function fp16 matrix accelerator with a wide fp32-class
accumulator, reachable only through Core ML in any supported way. [P]

### Faster units that exist

| unit | effect | do we use it? |
|---|---|---|
| **Double-int8 MAC mode** | Packs two int8 products per MAC; int8 runs at ~2x the fp16 rate. Distinct from int8 *weight* compression, where the weight dequantises to fp16 and enters an ordinary fp16 multiply | **No.** Requires W8A8. E1, E2 and E3 all used weight-only int8 (W8A16) |
| **Structured sparsity** | 1.55 - 1.64x faster at 0.43x dense weight bytes, output bit-faithful | No, untested |
| **Winograd path** | Auto-selected for eligible 3x3 stride-1 convolutions | Not applicable; we have 1x1/linear and a depthwise conv1d of kernel 4 |
| **Graph fusion depth** | One large matmul holds 4.8 TFLOP/s; a fused chain sustains 8.1 | **No.** Every probe was a single-layer graph |

Apple documents that W8A8 can use an int8 compute path on M4. [P] All
three of our experiments quantised weights only. **Every "ANE int8"
number we hold is an fp16-rate number with cheaper weight storage.**

### Hard constraints

| constraint | consequence |
|---|---|
| **Per-block quantisation is refused.** "ANE only support per-cout/per-tensor quantization" [P] | Our affine 4-bit **group-64** scheme has no ANE equivalent at all |
| Static or enumerated shapes only; `RangeDim` falls off the ANE [P] | KV cache append and slide cannot be dynamic |
| Gated-delta recurrence fails `ANECCompile` [M] | 48 layers keep a GPU-only component |
| Dispatch floor about 0.23 ms per eval regardless of size [P] | Anything smaller is overhead-bound |
| Standalone elementwise streams at 24 GB/s against the GPU's 230 [P] | Norms and activations must not cross alone |
| fp16 error grows with contraction dimension: 1.2e-2 at N=2048, 6.2e-2 at N=8192, against a flat GPU 3.6e-4 [P] | Our contractions are 5120 and 17408, inside and past that regime |

### Layout

The conducive format is 4D channels-first `(B, C, 1, S)`, with `nn.Linear`
swapped for a 1x1 convolution. The last axis is not packed: it must be
contiguous and 64-byte aligned, and a singleton last axis pads 32x in
fp16. [P] **None of our probes retiled to this layout.** E1 placed a
block 100% on ANE in rank-3 anyway, so layout was not required for
*placement*; whether it changes the *rate* is untested. [M]

### Ridge point

ANE 424 FLOP/byte against GPU 134 and CPU 15. [P] A layer needs roughly
3x the arithmetic intensity to be compute-bound on the ANE. Our chunk-1024
GEMMs sit near 2048 FLOP/byte, so we are compute-bound on the ANE, not
bandwidth-starved.

### The generation gap in our own numbers

```
M1 ANE: 11 TOPS int8 -> 5.5 TFLOPS fp16 theoretical; 4.8 measured (87%) [P]
M4 ANE: 38 TOPS int8 ->  19 TFLOPS fp16 theoretical; ~16.5 expected      [I]
Our best measured ANE cell on M4 Max:                    8.25            [M]
```

We get at best half, and typically a quarter, of what this engine should
do. Two named causes multiply to about the gap: weight-compressed int8
instead of true W8A8 (~2x) times single-layer instead of fused (~1.7x).
[I] The factors may not be independent.

---

## 5. The GPU

| property | value | source |
|---|---|---|
| Matrix hardware | **None dedicated on M4.** `simdgroup_matrix` competes for the same issue ports as every other instruction | [P] Rigel |
| Metal 4 `matmul2d` | 1.05 - 1.21x over `simdgroup_matrix`; same execution path | [P] |
| M5 difference | A Neural Accelerator per GPU core. Not this machine | [P] |
| fp8 | Emulated, 0.87 - 0.94x of fp16, therefore slower | [P] |
| Accumulator | fp32 or wider | [P] |
| fp16 accuracy | 3.6e-4 relative error, flat across size | [P] |
| Shapes | Fully dynamic, no recompilation | [M] |
| Achieved vs achievable | **13.4 - 13.5 of 14.4** at our shapes (q4 vs bf16, same shape); the GPU kernel path is effectively closed | [M] |

The GPU treats tall and wide GEMMs as nearly the same work, 6.2 to 6.9
TFLOPS either way. [M] That insensitivity is exactly what the ANE lacks.

---

## 6. The CPU

### 6.1 Full feature inventory for this processor

Read from `sysctl hw.optional` on the host. **1 means present.** [M]

**Matrix and dot-product extensions — the ones that matter**

| flag | present | what it gives | useful here? |
|---|---|---|---|
| `FEAT_SME` | **1** | Scalable Matrix Extension, ZA tile storage, streaming mode | **Yes, primary** |
| `FEAT_SME2` | **1** | Multi-vector instructions, wider ZA addressing | **Yes** |
| `sme_max_svl_b` | **64** | Streaming vector length 64 bytes = 512 bits | sets tile geometry |
| `SME_I8I32` | **1** | int8 inputs, int32 accumulate (`SMOPA`) | **Yes, fastest CPU path** |
| `SME_B16F32` | **1** | bf16 inputs, fp32 accumulate (`BFMOPA`) | **Yes, matches our dtype** |
| `SME_F16F32` | **1** | fp16 inputs, fp32 accumulate (`FMOPA`) | Yes |
| `SME_F32F32` | **1** | fp32 outer product | Yes |
| `SME_I16I32` | **1** | int16 to int32 | marginal |
| `SME_BI32I32` | **1** | binary/int32 | no |
| `FEAT_SME_I16I64` | **1** | int16 to int64 widening | no |
| `FEAT_SME_F64F64` | **1** | fp64 outer product | no |
| `FEAT_I8MM` | **1** | NEON `SMMLA`/`UMMLA`/`USMMLA`, 2x8 by 8x2 int8 matmul | fallback only |
| `FEAT_BF16` | **1** | NEON `BFMMLA`/`BFDOT`/`BFMLALB`/`BFMLALT` | fallback only |
| `FEAT_DotProd` | **1** | NEON `SDOT`/`UDOT`, 4-way int8 dot | **Yes, for 4-bit unpack paths** |
| `FEAT_FHM` | **1** | `FMLAL`/`FMLSL`, fp16 to fp32 widening multiply-add | situational |
| `FEAT_FP16` | **1** | Full half-precision arithmetic | yes |
| `FEAT_RDM` | **1** | `SQRDMLAH` rounding doubling multiply-add | no |
| `FEAT_FCMA` | **1** | Complex arithmetic | no |
| `FEAT_RPRES` | **1** | Higher-precision reciprocal estimate | situational |

**Absent, and each absence matters**

| flag | value | consequence |
|---|---|---|
| `FEAT_EBF16` | **0** | bf16 operations use the non-IEEE rounding mode. bf16 results will not bit-match an IEEE reference |
| `FEAT_SME_F16F16` | 0 | No fp16-to-fp16 SME. fp16 must accumulate to fp32 |
| `FEAT_SME_B16B16` | 0 | No bf16-to-bf16 SME |
| `FEAT_SME2p1` | 0 | No SME2.1 additions |
| `FEAT_SVE` | absent | **No non-streaming SVE.** Scalable vectors exist only inside SME streaming mode |
| `FEAT_CSSC` | 0 | No common short sequence compression |
| `FEAT_HBC` | 0 | No hinted conditional branches |
| `FEAT_MTE*` | 0 | No memory tagging |

**Not relevant to this workload but present:** `FEAT_AES`, `FEAT_SHA1`,
`FEAT_SHA256`, `FEAT_SHA3`, `FEAT_SHA512`, `FEAT_PMULL`, `FEAT_CRC32`
(crypto and checksum); `FEAT_PAuth`, `FEAT_PAuth2`, `FEAT_PACIMP`,
`FEAT_BTI`, `FEAT_FPAC` (control-flow integrity); `FEAT_LSE`, `FEAT_LSE2`,
`FEAT_LRCPC`, `FEAT_LRCPC2`, `armv8_1_atomics` (atomics and release
consistency, relevant to threading not arithmetic); `FEAT_DPB`,
`FEAT_DPB2` (cache maintenance); `FEAT_ECV`, `FEAT_WFxT`, `FEAT_DIT`,
`FEAT_SB`, `FEAT_CSV2`, `FEAT_CSV3`, `FEAT_FlagM`, `FEAT_FlagM2`,
`FEAT_JSCVT`, `FEAT_FRINTTS`, `FEAT_AFP`.

Cache geometry: L1i 128 KB, L1d 64 KB, L2 4 MB as reported, line size
**128 bytes**, page size 16 KB. [M]

### 6.2 Measured instruction throughput on M4

P-core and E-core, GOPS or GFLOPS. [P] Jena SME microbenchmarks.

| instruction | in -> out | P-core | E-core |
|---|---|---:|---:|
| **`SMOPA` (SME)** | i8 -> i32 | **4020** | 716 |
| `SMOPA` (SME) | i16 -> i32 | 2010 | 358 |
| **`BFMOPA` (SME)** | bf16 -> f32 | **2011** | 357 |
| **`FMOPA` (SME)** | f16 -> f32 | **2010** | 358 |
| `FMOPA` (SME) | f32 -> f32 | 2008 | 357 |
| `AMX FMA` | f32 | 2006 | 357 |
| `FMOPA` (SME) | f64 -> f64 | 503 | 89 |
| `FMLA` (SME2 streaming) | f32 | 502 | 179 |
| **`FMLA` (NEON)** | f32 | **107** | 46 |

Three conclusions.

1. **SME is about 19x NEON** on matrix work, 2008 against 107 GFLOPS.
2. **SME and AMX are the same block**, 2008 against 2006. SME is the
   documented ISA for hardware we already had.
3. **On M4 only int8 gets a datatype speedup.** bf16, fp16 and fp32 all
   land at about 2010. The unit is fp32-centric. Do not expect a bf16 win
   over fp32; do expect a 2x int8 win.

The unit is **per-cluster, not per-core**: 2008 GFLOPS is the P-cluster
total, not a per-core figure multiplied by 12.

### 6.3 What we actually run on the CPU today

MLX implements affine 4-bit group-64 transposed quantized matmul on the
CPU (`_qmm_t_simd` in `Vendor/mlx-swift/.../backend/cpu/quantized.cpp`),
and a column-split GPU/CPU helper was built and proven correct: each half
is bit-exact against a standalone run on its own device. [M]

But `_qmm_t_simd` is a float SIMD accumulation loop. That is NEON, about
0.107 TFLOPS. **The SME unit is idle.** [I]

Also measured: CPU and GPU quantized kernels are numerically different.
Global relative error against a GPU-only result reaches 0.169 on the
gate/up shape and 0.268 on the down shape. [M] A CPU assist is not a
drop-in for token-identical output.

### 6.4 Where the CPU can win

- **Below the accelerator dispatch floor.** A 64x256x256 matmul completes
  in about 0.026 ms on the CPU, under the ANE's 0.23 ms floor and under a
  GPU launch. [P] Our MTP draft head at seq=1 is 0.10 GFLOP; Metal does it
  in 0.18 ms. [M] SME compute time would be about 0.05 ms. The GEMV is
  weight-bound rather than compute-bound, so treat this as a **candidate
  requiring measurement**, not a claim.
- **As a third stream in a partition.** 2.0 TFLOPS bf16 against Metal's
  6.2-6.9 is roughly a 24% throughput addition if overlapped.
- **Never** for wide standalone elementwise work; the GPU has the bandwidth.

---

## 7. What we are leaving on the table

Ordered by size, largest first.

| # | gap | size | confidence |
|---|---|---|---|
| 1 | ~~GPU 4-bit issue-slot dilution~~ **CLOSED** | 13.4 of 14.4 at the same shape; ~1.05x remains | [M] see section 0 |
| 2 | CPU SME unused; quantized CPU path is NEON | 0.107 -> 2.01 TFLOPS, ~19x on that path | [P] rates, [M] code read |
| 3 | ~~ANE W8A8 / fused / retiled~~ **CLOSED** | ANE never beats correctly measured Metal at these shapes | [M] see section 0 |
| 4 | Three engines never overlapped | up to ~1.3x on top of the above | [I] |

Item 1 outranks the rest: it needs no new engine, no device crossing, no
Core ML, no static shapes, and introduces no second numerical path.

---

## 8. Open items

- **Crossing cost between Metal and Core ML was never measured.** An
  earlier probe measured a host round trip at 0.90 ms with a copy and
  0.0007 ms to wrap host memory, but it never wrapped a live MLX Metal
  buffer. [M] A partition that saves 4 ms and spends 8 ms copying is not
  a partition.
- **ANE accuracy at K=17408 is unmeasured.** Cosine 0.999996 was measured
  on an attention block only.
- **CPU SME throughput on our actual shapes is unmeasured.** The 2010
  GFLOPS figure is a microbenchmark, not our GEMM.

---

## Sources

Apple, *Deploying Transformers on the Apple Neural Engine* (2022) and
*Deploying Attention-Based Vision Transformers to Apple Neural Engine* (2024).
coremltools guides: Compute Units, Typed Execution, Flexible Input Shapes,
Quantisation Overview; issue 2510 (ANE refuses per-block quantisation).
Bryngelson, *Apple Neural Engine: Architecture, Programming, and
Performance*, arXiv:2606.22283.
Kumaresan, *Orion: Characterizing and Programming Apple's Neural Engine*,
arXiv:2603.06728.
*Rigel: Reverse-Engineering the Metal 4.1 Tensor Compute Path on the Apple
M4 Max GPU*, arXiv:2606.12765.
*NPUMoE: Efficient Mixture-of-Experts LLM Inference with Apple Silicon
NPUs*, arXiv:2604.18788.
*Hello SME! Generating Fast Matrix Multiplication Kernels Using the
Scalable Matrix Extension*, arXiv:2409.18779, and the Jena microbenchmark
tables at https://scalable.uni-jena.de/opt/sme/micro.html
Arm, *Scalable Matrix Extension* parts 1 and 2.
Our own: E1, E2, E3 reports; tflops-gap report; cpu-assist report;
`sysctl hw.optional` on the host.

---

## 12. Probe results, 2026-08-29 (Swift harness, one point per process)

### 12.1 The position effect did not reproduce

`GPUPositionEffectTests` crossed six preludes with a compute-bound
measurement. **No prelude degraded the GEMM.** [M]

| prelude | GEMM TFLOPS |
|---|---:|
| none (control) | 14.72 |
| cpu (200 ms, no MLX op) | 14.76 |
| alloc (128 MiB, materialised, dropped) | 14.73 |
| tinygpu (one-element dispatch) | 14.74 |
| bandwidth (128 MiB read+write) | 14.74 |
| biggpu (full 4096^3 bf16 GEMM) | 14.70 |
| **biggpu, then `Memory.clearCache()`** | **10.81** |

The documented 14.7 -> 6.4 collapse is **not** caused by prior GPU work,
prior allocation, prior dispatch, or prior CPU load. The only arm that
degraded is the one that calls `Memory.clearCache()`, and that is the helper
`PrefillMatmulCostTests.quiesce()` runs between blocks precisely to make
in-process tables comparable. It costs **1.36x** on the measurement that
follows. [M]

This explains part of the collapse, not all of it: 1.36x against an observed
2.3x. The remainder is unexplained and stays open. What does not depend on
the explanation: the one-per-process numbers in section 0 were taken with the
controlled harness and stand on their own.

### 12.2 Short kernels run faster after a prelude, not slower

The bandwidth-bound measurement moves the opposite way. [M]

| prelude | bandwidth GB/s |
|---|---:|
| none (control) | 245.6 |
| alloc | 257.4 |
| bandwidth | 298.9 |
| biggpu | **342.0** |

A 1 ms kernel on an idle GPU does not ramp the clock; a prelude leaves the
GPU ramped. **1.39x for free on short kernels from a warm GPU.** [I] This is
the mirror image of 12.1 and it means "quiesce before timing" actively
mis-measures short kernels in both directions: it removes the ramp and pays
the cache-clear penalty.

### 12.3 D2 — the weight-residency cliff is inverted

Fixed M=1024, K=5120, sweeping N. q4 weight bytes = N x 2560. [M]

| N | weight MB | TFLOPS |
|---:|---:|---:|
| 1024 | 2.6 | 2.63 |
| 2048 | 5.2 | 7.24 |
| 4096 | 10 | 9.36 |
| 8192 | 21 | 13.29 |
| **17408** | **45** | **13.53** |
| 34816 | 89 | 9.27 |

The kernel source comment records "a 50 MB gate_up set does [exceed cache],
a 5.9 MB set does not, which is why the two shapes measure 5.0 and 10.2
TFLOPS." Measured one-per-process, the ordering is **reversed**: the 45 MB
set is the fastest point on the curve and the 5.2 MB set runs at half its
rate. Small weight sets are slow because they do not fill the machine, not
fast because they fit in cache. The 89 MB drop is real and unexplained.

### 12.4 D3 — weight traffic is not the limiter

BM=32 against BM=64 via `MLX_QMM_BM`, at a cache-resident and a
DRAM-resident weight set. [M]

| weight set | BM=32 | BM=64 | ratio |
|---|---:|---:|---:|
| 5.2 MB (resident) | 6.32 | 7.25 | **1.15x** |
| 45 MB (DRAM) | 12.50 | 13.50 | **1.08x** |

BM=64 halves the number of full weight passes. If traffic bound the kernel,
the larger set would benefit more. It benefits **less**. Traffic is not the
limiter, which agrees with 12.3 and closes the last open explanation for the
quantized GEMM being slow. It is not slow.

### 12.5 Consequences

- **The GPU quantized GEMM is finished work.** 93-95% of bf16 at the same
  shape, tile already at its best value, traffic not binding.
- **`Memory.clearCache()` is a performance hazard**, not a neutral hygiene
  call. The one call on the model path
  (`Qwen36MTPBlockSession.wireResidentWeightsIfEnabled`) runs once at setup
  behind a `wiredTicketRetainer == nil` guard, so the hot path is clean. [M]
- **E3's Metal column is understated by about 2x**, now shown directly rather
  than inferred: E3 read 6.87 TFLOPS for gate/up at seq=1024; the controlled
  harness reads 13.53 at that exact shape.
- **The forward-pass gap is not a position artifact.** The forward runs at
  about 6.11 TFLOPS while its dominant kernel does 13.5 in isolation, and no
  prelude in 12.1 reproduces that degradation. The gap therefore belongs to
  the non-GEMM composition of the forward: attention, the gated-delta
  recurrence, norms, activations, RoPE and per-op dispatch. Measuring that
  composition directly is the next step, and it replaces the whole engine-
  migration programme.

---

## 13. Prefill attribution on the real tower, 2026-08-29

`QwenPhaseBreakdownTests`, real Qwen 3.8 checkpoint plus MTP head, 1024-token
chunk. Per-layer numbers come from the unfused synced seam, so their SHARES
are the signal; the separately measured fused total is the truth. [M]

Caveat: the host was not quiet (a wedged `spotlightknowledged` held ~95% of a
core). The fused total nonetheless lands within 2% of the independently
derived 8.2 s/chunk cost model, so the attribution is usable.

| phase | depth 0 | depth ~9k |
|---|---:|---:|
| embed | 16.6 ms | 13.0 ms |
| 48 gated-delta blocks | 5626.4 ms (75.5%), mean **117.22** | 5596.4 ms (67.9%), mean 116.59 |
| 16 full-attention blocks | 1807.1 ms (24.3%), mean **112.94** | 2632.3 ms (31.9%), mean **164.52** |
| synced total | 7450.1 ms | 8241.8 ms |
| **fused truth** | **8052.2 ms** | **8238.6 ms** |

### 13.1 Roughly half the chunk is not GEMM

Pricing each layer's GEMMs at the measured 13.45 TFLOPS (section 0):

| layer | GEMM GFLOP | GEMM ms at 13.45 | measured ms | non-GEMM |
|---|---:|---:|---:|---:|
| gated-delta | 784.8 | 58.4 | 117.2 | **58.9 (50.2%)** |
| full-attention (depth 0) | 788.1 | 58.6 | 112.9 | **54.3 (48.1%)** |

Chunk GEMM total 50.3 TFLOP, which independently reproduces the cost model's
50.07. At 13.45 TFLOPS that is **3.74 s against an 8.05 s chunk**, so
**4.31 s (53%) of cold prefill is not matrix work**.

The largest single bucket is the gated-delta layers' non-GEMM half:
48 x 58.9 ms = **2.83 s, 35% of the whole chunk**.

### 13.2 The depth sweep separates the two costs

Gated-delta blocks are flat in depth (5626 ms at 0, 5596 ms at 9k), so their
cost is per-token work: recurrence scan, depthwise conv1d, norms, gating.
Full-attention blocks grow 112.9 -> 164.5 ms per layer, which is the
quadratic KV term. These need different fixes: operator fusion for the first,
attention work for the second.

### 13.3 Consequence for the engine question

The GPU's matrix units are idle for about half of prefill because half of
prefill is not matrix work. Adding a slower engine to absorb GEMM slices
attacks the 47% that already runs near ceiling, not the 53% that does not.
Fusion of the low-intensity operators into the GEMM epilogues that produce
their inputs is the larger and safer target, and it stays entirely on the GPU
in Metal.

### 13.4 Decode, with a caveat

The verify-width sweep shows host graph-build dominating GPU eval at small M
(M=1: 35.7 ms build against 7.5 ms eval). **`swift test` builds debug by
default**, so `build_ms` is debug-speed host code and does not represent the
release serve path. `eval_ms` is real GPU time. Do not act on the build
column without re-measuring under `-c release`.

---

## 14. CORRECTION to section 13: the forward is ~85% GEMM, not ~47%

`GemmChainCostTests`, quiet host, one process. The seven gated-delta
projections, 785.0 GFLOP total, with NO norms, activations or recurrence. [M]

| measurement | ms | TFLOPS | vs solitary |
|---|---:|---:|---:|
| one big GEMM alone in a process (section 0) | - | **13.5** | 1.00 |
| seven projections, sequential, one process | 88.79 | **8.84** | 1.53x slower |
| same seven, chained in one eval | 106.57 | **7.36** | 1.83x slower |

### 14.1 Re-pricing

Section 13 priced the chunk's GEMMs at 13.5 TFLOPS, a rate only a solitary
kernel reaches, and concluded 53% of prefill was non-GEMM. At the chained
rate of 7.36 TFLOPS the same 50.3 TFLOP costs **6.83 s against an 8.05 s
chunk**, so non-GEMM is **~1.22 s, about 15%**.

That reconciles with the operator measurements, which section 13 could not:
scan 307 ms + conv1d 36 ms + norms/gating/dispatch ~800 ms = ~1.2 s.

### 14.2 Why section 12.1 missed this

The position-effect probe ran ONE prelude then ONE measurement of the same
kernel at the same shape, and correctly found no degradation. The
degradation appears when **many differently-shaped GEMMs run in sequence**,
which is what a forward pass is. "No prelude poisons a GEMM" was
over-generalised into "the position effect is not mechanical". It is
mechanical; section 12.1 simply did not reproduce the conditions.

### 14.3 The target

Operator fusion addresses ~15% of prefill and is no longer the lead item.
The lead item is the **1.83x between a solitary GEMM and a chained one**,
which applies to ~85% of prefill. It separates into:

- **1.53x** from several DIFFERENT shapes in one process. Mechanism unknown.
  Candidates: per-shape JIT kernel variants and pipeline-state switching,
  allocator churn across differently-shaped intermediates.
- **1.20x** from dependency chaining on top of that: each GEMM waits on its
  predecessor, so there is no cross-GEMM overlap.

The 1.20x is expected and partly irreducible. The 1.53x is unexplained and
is the larger term.

### 14.4 CPU column split: dead as implemented

`ColumnSplitSpeedTests`, M=1024, N=17408, K=5120. [M]

| CPU fraction | CPU columns | wall ms |
|---:|---:|---:|
| 0 | 0 | 13.4 |
| 0.01 | 174 | 54,429 |
| 0.02 | 348 | 110,010 |

Linear at 0.313 s per output column, about 0.033 GFLOPS: roughly 3,000x
below single-core NEON and 60,000x below SME. MLX's `_qmm_t_simd` is not a
slow-but-usable path. A CPU slice would need an SME quantized GEMM written
from scratch, and even at SME's 2.0 TFLOPS against 13.5 the prize is ~13% of
columns. Sweep stopped after three points; the line is straight.

### 14.5 Operator costs, for the record

Per 1024-token chunk, all 48 gated-delta layers. [M]

| operator | ms/chunk |
|---|---:|
| gated-delta recurrence | ~307 |
| depthwise conv1d | ~36 |

The recurrence was the standing suspect for the flat prefill curve. It is
12% of the non-GEMM bucket and about 4% of the chunk. Exonerated.

---

## 15. CORRECTION: both the CPU and the ANE were closed on bad implementations

Sections 0 and 14 closed the CPU and the ANE. Both closures measured a poor
implementation and attributed the result to the hardware. Both are withdrawn.

### 15.1 CPU: the matrix unit was never used

| path | GFLOPS | note |
|---|---:|---|
| MLX `_qmm_t_simd` (what section 14.4 measured) | **0.033** | hand-written scalar float SIMD loop |
| Accelerate `cblas_sgemm` -> AMX/SME | **2482** | M=1024, N=2176, K=5120 |

A **75,000x** gap. The CPU matrix unit delivers **2.48 TFLOPS**, about 18% of
Metal's 13.5, so a CPU column slice can absorb ~15% of columns. [M]

Blocker is now dequantisation, not compute: a scalar 4-bit -> fp32 expansion
costs 669 ms against 9.19 ms of GEMM. Weights never change, so the fix is to
cache the dequantised slice (~2 GB in bf16 for a 15% slice) or vectorise it.

### 15.2 ANE: reachable from pure Swift, no Python

The claim that authoring a Core ML model requires `coremltools` is wrong.
`MLModelAsset(specification:)` (macOS 13+) accepts raw `.mlmodel` protobuf
bytes in memory. The Core ML `.proto` files are proto3 and the subset needed
for a one-layer `innerProduct` model encodes in ~30 lines of Swift.

`tools/ane-native/main.swift` does the whole chain: hand-encoded protobuf ->
`MLModelAsset` -> `MLComputePlan` (which proves placement per layer) ->
`MLModel.prediction`. [M]

Two gotchas that cost a segfault each:

- **FLOAT16 top-level input/output crashes** inside `prediction`. Use FLOAT32
  IO with fp16 weights; it still places on the ANE.
- **`arrayInputShapeMapping` defaults to RANK5_ARRAY_MAPPING.** Set field 5 to
  `EXACT_ARRAY_MAPPING` (1).

`MLComputePlan` is now the placement oracle. It reports the chosen device
BEFORE timing, which is stronger evidence than power rails and would have
caught the `MLTensor` failure below immediately.

### 15.3 MLTensor does not reach the ANE

`MLTensor` + `withMLTensorComputePolicy(.cpuAndNeuralEngine)` compiles and
runs, but places this work on the CPU: `ane_power` flat 0.0 across the run,
and timing identical to `cpuOnly` (2.556 vs 2.553 TFLOPS, both on the
Accelerate/SME signature). The compute policy is a hint and Core ML declined
it. Per-op `MLTensor` dispatch is not an ANE path for this shape. [M]

An earlier version of that benchmark used `MLTensor(repeating:)` and measured
nothing at all -- constant tensors are folded or CPU-handled. The tell was
that all three "engines" agreed within 27% with byte-identical copy times.
**Agreement between things that should differ is a defect signal.**

### 15.4 The ANE output-channel limit is exactly 16384

M=1024, K=5120, sweeping output channels. Placement from `MLComputePlan`. [M]

| N | device | ms | TFLOPS |
|---:|---|---:|---:|
| 4096 | ANE | 10.99 | 4.29 |
| 8192 | ANE | 18.52 | 4.77 |
| 8704 | ANE | 19.51 | 4.68 |
| 12288 | ANE | 26.30 | 4.96 |
| **16384** | **ANE** | **34.17** | **5.03** |
| **16385** | **CPU** | **204.17** | 0.84 |
| 17408 | CPU | 210.92 | 0.87 |

**2^14 output channels is a hard limit**, and the fallback is SILENT: no
error, no warning, a 6x slower model that still returns correct results. Any
ANE integration must assert placement rather than assume it.

`mlp.gate_up` is [17408, 5120], just past the limit. It splits cleanly into
2 x 8704, both on the ANE at 19.51 ms each.

### 15.5 int8 does not help on this format, and does not move the limit

| dtype | N=12288 | N=17408 | weights at 17408 |
|---|---:|---:|---:|
| fp16 | 26.30 ms, ANE | 210.92 ms, CPU | 170 MB |
| weight-only int8 | 26.28 ms, ANE | 211.85 ms, CPU | **85 MB** |

Identical times at every N, and halving the bytes does **not** move the
cliff -- confirming 15.4 that the limit is a shape bound, not a memory bound.

The NeuralNetwork format appears to dequantise weights at model LOAD, so the
runtime tensor is fp16 either way and only the serialised model shrinks.
E1/E2/E3's 1.6x int8 gain came via `ios18.constexpr_blockwise_shift_scale`,
which is an **ML Program** feature that dequantises at RUNTIME. To get that
benefit natively, author an ML Program (MIL) rather than a NeuralNetwork;
`MLModelAsset(specification:blobMapping:)` exists for exactly that, taking
weight blobs in memory.

`int8DynamicQuantize` (field 22) is NOT the answer: Core ML expands it to
`dynamic_quantize -> inner_product -> dynamic_dequantize` and places the
inner product on **CPU**. The W8A8 double-int8 MAC mode is not reachable this
way. [M]

### 15.6 Three engines, all measured on this machine

| engine | TFLOPS | constraint |
|---|---:|---|
| Metal GPU, 4-bit, isolated | 13.5 | - |
| Metal GPU, chained in a forward | 7.36 | what prefill actually sees |
| ANE, fp16 | 4.96 - 5.03 | N <= 16384; needs its own fp16 weight copy |
| CPU, Accelerate/SME | 2.48 | dequant must be cached; unit is per-CLUSTER |

Perfect overlap gives 20.9 TFLOPS, **1.55x**. Balanced shares are Metal 65%,
ANE 23%, CPU 12%.

**Overlap remains undemonstrated.** Every figure above is measured alone. The
column-split experiment that would show concurrency had its CPU leg crippled
by the scalar kernel and has not been rerun against SME.

---

## 16. Keeping ANE weights quantized at runtime: built, and it does not pay

Question: can we avoid dequantizing to fp16 for the ANE? Yes, and it is built
in pure Swift. It saves memory and costs a little speed.

### 16.1 What was built

`tools/ane-mlprogram/` authors an **ML Program** (MIL) by hand:

```
const w_data   : int8  [N, K]
const w_scale  : fp16  [N, K/64]
const w_offset : int8  [N, K/64]
w = ios18.constexpr_blockwise_shift_scale(data, scale, offset)   -> fp16
y = ios18.matmul(x, w, transpose_y=true)
```

`MLComputePlan` confirms `ios18.matmul -> MLNeuralEngineComputeDevice`. [M]
`proto.swift` is an 84-line protobuf writer; no coremltools, no Python, no
external blob files (weights ride as `immediateValue`, avoiding the
undocumented blob format).

This matters beyond int8: the op's type domain is
`int4, uint4, int8, uint8, fp16, fp32` and block size is implied by
`data.shape / scale.shape`. Our affine group-64 maps directly with
`offset = -bias/scale`, so **uint4 weights are expressible** -- the ANE could
hold our real checkpoint format rather than a dequantized fp16 copy.

### 16.2 Runtime dequantization is 3-5% SLOWER, not 1.6x faster

| N | NeuralNetwork fp16 (load-time dequant) | ML Program int8 (runtime dequant) |
|---:|---:|---:|
| 8192 | **18.52 ms** | 19.44 ms |
| 12288 | **26.30 ms** | 26.94 ms |
| 16384 | 34.17 ms | 34.60 ms |

E1/E2/E3's 1.6x int8 gain did **not** reproduce. [M]

Explanation, consistent with our own curve: ANE throughput RISES with N
(1.31 -> 4.97 TFLOPS), which is a dispatch- and compute-limited signature,
not a bandwidth-limited one. Halving weight bytes only helps when waiting on
weight traffic. E1/E2 measured whole attention and GDN blocks -- many ops,
far more weight streaming per unit of compute -- which is the plausible home
of their 1.6x. A single matmul is not that regime.

**So the benefit is memory, not speed:** int8 halves resident weight bytes and
uint4 would quarter them. For a partition that was going to cost ~+7 GB in
fp16, that is worth having; it is not a throughput win.

### 16.3 The 16384 limit is ANE-level, not op-level

| op | N=16384 | N=17408 |
|---|---|---|
| NeuralNetwork `innerProduct` | ANE, 34.17 ms | CPU, 210.92 ms |
| ML Program `ios18.matmul` | ANE, 34.60 ms | CPU, 214.31 ms |

Identical boundary under two different ops and two different model formats.
2^14 output channels is a property of the engine. The fallback stays silent
and costs ~6x. **Assert placement with `MLComputePlan`; never assume it.**

### 16.4 Still unvalidated

Numerics have not been checked against a reference. The ML Program runs and
places correctly; that it computes the right answer is untested.

---

## 17. The double-int8 lane exists in hardware and Core ML never emits it

This closes the W8A8 question with a documented mechanism rather than an
inference. Source: Bryngelson, arXiv:2606.22283, sections 7.3, 9.1 and 20.2.

### 17.1 The hardware mode is real

> "GetNumOutputChannelsPerCycle returns eight in the int8 fast path and four
> in the default fp16 path ... one core produces up to eight output channels
> per cycle in double-int8 mode and four output channels per cycle in fp16."

> "The multiply array has a double-int8 mode that packs two int8 products into
> one multiply-accumulate, so int8 arithmetic runs at about twice the fp16
> rate on the same array."

### 17.2 The Core ML frontend does not reach it

> "**The frontend does not reach that doubled lane.** The int8 compile flag
> quantizes the weights and leaves the multiply-accumulate in fp16, so it
> halves the streamed weight bytes and **never emits the eight-channel int8
> compute path**. The flag thus changes weight bandwidth, not compute rate."

> "The int8 path reaches about 1.5 times faster **only where the weight is
> large enough to stream from main memory**, near a 4096-by-4096 weight at a
> batch of 256 or more."

So Core ML int8 is always the compressed-weight path. It can reduce weight
bandwidth; it can never reduce compute.

### 17.3 Our measurements are exactly what that predicts

At N=8192, K=5120 the fp16 weight set is 80 MB moved in 18.52 ms:

```
80 MB / 0.01852 s = 4.3 GB/s   against the ANE's ~85 GB/s ceiling  = ~5%
```

Nowhere near bandwidth-bound, so halving the weight bytes buys nothing. [M]
Measured int8-vs-fp16 was within 5% at every N. Predicted, not anomalous.

### 17.4 All three routes tried, all consistent

| route | result | mechanism |
|---|---|---|
| NeuralNetwork + quantization params | int8 == fp16 | weights expanded at LOAD |
| ML Program + `constexpr_blockwise_shift_scale` | int8 3-5% slower | op's `DstT` is fp16/fp32 only, so it dequantizes before the matmul |
| `int8DynamicQuantize` (field 22) | placed on **CPU** | expands to dynamic_quantize/dequantize, not ANE-supported |

The doubled lane is unreachable through Core ML by any of them. The only
access is the direct ANE compiler, which the same paper calls "undocumented,
unsupported, and version-fragile". Not a basis for a serve path.

### 17.5 What survives

int8 and uint4 remain worth using on the ANE for **memory**, not speed: half
and a quarter of the resident weight bytes respectively, at a 3-5% latency
cost. The ANE's ceiling for our workload stays **~4.97 TFLOPS**.

### 17.6 Why int8 compute is unreachable: the IR has no int8 arithmetic op

Section 17.2 quoted the paper's behavioural claim. The MIL op type domains
show the structural reason, which is stronger: it cannot be worked around by
a different graph shape or a newer OS.

| op | activation domain | weight domain |
|---|---|---|
| `matmul` | fp16, fp32, int32 | same |
| `linear` | fp16, fp32, int32 | same |
| `conv` | fp16, fp32 | same |
| `conv_quantized` | **fp16, fp32** | uint8 |
| `constexpr_blockwise_shift_scale` | output `DstT`: **fp16, fp32** | int4/uint4/int8/uint8 in |
| `quantize` / `dequantize` | produce/consume int8 | run on **CPU** (measured) |

**No MIL op performs int8 x int8 arithmetic.** Every arithmetic op declares a
floating-point activation domain. `conv_quantized`, the op built specifically
for quantized weights, is explicit: `T: (fp32, fp16)`, `U: (uint8,)` -- the
weight is quantized, the activation is not, and the weight dequantizes at the
multiplier input.

The doubled lane needs BOTH operands int8. The IR cannot express that, so no
Core ML graph can reach it. The nearest integer arithmetic is `int32` on
matmul, and widening int8 to int32 discards the packing that makes the lane
fast.

Four constructions were built and measured, all consistent: [M]

| construction | placement | why |
|---|---|---|
| constexpr int8 weights -> matmul | ANE, fp16 MAC | constexpr `DstT` is fp16/fp32 |
| quantize/dequantize on activation | all CPU | q/dq not ANE-eligible, drag matmul off |
| dequantize on both operands | all CPU | same |
| `int8DynamicQuantize` (NeuralNetwork) | CPU | expands to dynamic_quantize/dequantize |

Conclusion: the ANE ceiling for this workload is **~4.97 TFLOPS** and the 2x
doubled lane is not accessible from any supported API.

### 17.7 WITHDRAWN: 17.6's structural claim was too strong

17.6 concluded "no Core ML graph can reach the int8 lane" from the MIL op type
domains. That inference does not hold, and online sources contradict it.

**Apple documents an int8-int8 compute path on our exact hardware**
(coremltools, *Quantization Performance*): "8-bit activation plus weight
quantization can lead to considerable latency benefits on the Neural Engine by
leveraging the faster int8-int8 compute path supported in newer hardware
(A17 Pro, M4)", with a measured ResNet50 example at **1.38 ms -> 0.77 ms**
(1.8x) on iPhone 15 Pro.

The error in 17.6: float type domains describe the FRONTEND ops. W8A8 is
expressed as `quantize -> op -> dequantize` pairs, and the Core ML **compiler**
is what lowers that pattern to an int8 kernel. The IR does not need an int8
arithmetic op for the backend to run one.

**But the mechanism is genuinely disputed, not merely unmeasured.** coremltools
issue #2432: a developer built W8A8 with `LinearQuantizer` for iOS 18 and found
Core ML "dequantizing inputs and weights to fp16, executing convolution and
ReLU in fp16, requantizing outputs back to int8" -- the same thing our probes
show. They asked whether the documented speedup is actually int8 compute or
just reduced activation data movement between layers. **The issue is open with
no Apple response.**

So the accurate state is:

- MEASURED (ours): our hand-built `quantize`/`dequantize` pattern around
  `matmul` places every op on CPU, on M4.
- PUBLISHED (Apple): W8A8 gives 1.8x on a **convolution** model on A17 Pro,
  attributed to an int8-int8 path.
- PUBLISHED (issue 2432, unrefuted): the emitted graph computes in fp16 anyway.
- UNTESTED: the `quantize`/`dequantize` pattern around a **conv** (1x1) rather
  than a matmul, on M4.

The untested cell is the one that matters. Every int8 fast-path reference --
the paper's measurements, Apple's guidance to express linear layers as 1x1
convolutions, and Apple's own W8A8 example -- is convolution-shaped. Our probes
were all matmul/innerProduct. That is the next experiment, and until it runs,
the ANE's int8 ceiling is **open, not closed**.

---

## 18. Conv versus matmul, and fusion depth

`tools/ane-conv/` builds a 1x1 convolution stack in (B, C, 1, S) layout,
optionally W8A8. C=5120, S=1024, fp16, placement from `MLComputePlan`. [M]

| layers | placement | ms | TFLOPS |
|---:|---|---:|---:|
| 1 | ANE=1 | 13.82 | 3.89 |
| 2 | ANE=2 | 23.08 | 4.65 |
| 4 | ANE=4 | 42.18 | **5.09** |

### 18.1 The "1x1 conv is 3x faster than matmul" claim is FALSE here

A 1x1 conv at C=5120 reaches 3.89 TFLOPS against 4.4-5.0 for matmul at
comparable widths. Conv is at best equal and here slightly slower.

That claim came from one reverse-engineering blog (maderix), was never
corroborated by the primary source, and was cited repeatedly in earlier
sections of this document -- including as a reason E2 might have been
mis-built. **It should have been carried as single-source folklore, not as a
working assumption.** Withdraw it wherever it appears above.

Apple's guidance to express linear layers as 1x1 convs still stands on its own
terms (layout and L2 residency), but it does not buy a faster datapath here.

### 18.2 Fusion depth is real, and the marginal rate is the number to use

```
layer 1:         13.82 ms
marginal layer:   ~9.4 ms  ->  53.7 GFLOP / 0.0094 s = 5.65 TFLOPS
fixed overhead:   ~4.4 ms per inference (dispatch + host IO copy)
```

A deep fused graph approaches **~5.7 TFLOPS**. Single-op probes understate the
ANE by about 30% because they pay the fixed cost once per op. Every ANE number
in sections 15-17 is a single-op number and is therefore a floor.

### 18.3 W8A8: not reproducible by hand-authoring

Five constructions, all placing the compute op on CPU: [M]

| construction | placement |
|---|---|
| NeuralNetwork `int8DynamicQuantize` | CPU |
| ML Program `constexpr` int8 -> matmul | ANE, but fp16 MAC |
| quantize/dequantize on activation + matmul | CPU |
| `dequantize` on both operands + matmul | CPU |
| quantize/dequantize + **1x1 conv** | CPU |

Verdict: **not reproducible here**, NOT "impossible". Apple publishes a
ResNet50 W8A8 result (1.38 -> 0.77 ms on iPhone 15 Pro) and attributes it to an
int8-int8 path on A17 Pro / M4. Without coremltools there is no reference W8A8
model to diff against, so the failure cannot be localised to the pattern, the
scales, the opset, or the OS. Unresolvable from this machine.

### 18.4 Refined three-engine picture

| engine | TFLOPS | share of a balanced split |
|---|---:|---:|
| Metal GPU, 4-bit | 13.5 | 62% |
| ANE, fp16, fused, N <= 16384 | **5.7** | 26% |
| CPU, Accelerate/SME | 2.5 | 12% |
| **combined** | **21.7** | **1.61x** |

Overlap remains undemonstrated. Every figure is measured with one engine alone.

---

## 19. The ANE ceiling is ~6.8 TFLOPS, and W8A8 stays unreproducible

### 19.1 W8A8: seven constructions, all CPU

The research corrected a real error first: coremltools' activation-quantization
pass declares its supported ops as

```python
SUPPORTED_UNARY_OP_TYPES = ["conv", "avg_pool", "max_pool", "linear"]
SUPPORTED_BINARY_OP_TYPES = ["add"]
```

**`matmul` is not among them**, so constructions 2-4 in section 18.3 were
structurally incapable of working and are uninformative. The pass also inserts
a PREFIX pair only (`quantize -> dequantize -> op`), not the before-and-after
wrapping used earlier, and the documented mode is `linear_symmetric`, where
`zero_point` is None and omitted.

Rebuilt to that exact recipe -- conv, prefix-only, symmetric, stacked -- and it
still places every op on CPU, at 1 and at 4 layers. Timing rules out hidden
fusion: at C=2048, S=1024, 4 layers, fp16 runs 4.54 ms (7.56 TFLOPS, ANE=4)
against W8A8 at 51.63 ms (0.67 TFLOPS, CPU=16), an **11x** gap. [M]
`MLComputePlan` was accurate; there is no secret int8 path being taken.

**Verdict: not reproducible here. NOT "impossible."** Apple publishes a
ResNet50 W8A8 result and attributes it to an int8-int8 path on A17 Pro / M4.
Without coremltools there is no reference model to diff against, so the gap
cannot be localised to the encoding, a calibration artifact, the opset, or
something the runtime accepts only from Apple's own toolchain.

### 19.2 Fused conv width sweep: an interior optimum

4 layers, S=1024, fp16, all ANE-placed. [M]

| C | weights/layer | ms | TFLOPS |
|---:|---:|---:|---:|
| 1024 | 2.0 MB | 1.49 | 5.76 |
| 2048 | 8.0 MB | 5.08 | 6.76 |
| **3072** | 18 MB | 11.35 | **6.81** |
| 4096 | 32 MB | 21.64 | 6.35 |
| 5120 | 50 MB | 42.21 | 5.09 |

Peak near C=2048-3072, falling off both ways: dispatch overhead below, working
set above. Run-to-run variance is about 11% (C=2048 read 7.56 and 6.76 on two
runs), so treat these as +/-10%.

### 19.3 Two regimes, opposite slopes

- ANE **matmul**, single op: throughput RISES with width (1.31 -> 4.97 TFLOPS
  as N grows to 16384). Climbing out of dispatch overhead.
- ANE **fused conv**: throughput FALLS with width past ~3072. Descending into a
  working-set wall.

Every ANE figure in sections 15-18 came from the first regime at its widest,
which is the worst corner of this surface. **The ANE ceiling is ~6.8 TFLOPS,
not 4.97.** The engine prefers narrow, deep, fused graphs.

Practical consequence: routing `gate_up` [17408, 5120] to the ANE should split
it into ~8 chunks of 2048 output channels (21 MB of weights each, inside the
good band) rather than 2 chunks of 8704 to clear the 16384 limit. The limit is
a correctness constraint; the width curve is the performance one, and it binds
much earlier.

### 19.4 Refined three-engine picture

| engine | TFLOPS | share |
|---|---:|---:|
| Metal GPU, 4-bit | 13.5 | 59% |
| ANE, fp16, fused, C ~ 2048-3072 | **6.8** | 30% |
| CPU, Accelerate/SME | 2.5 | 11% |
| **combined** | **22.8** | **1.69x** |

Overlap still undemonstrated; every figure is one engine measured alone.

---

## 20. Overlap is REAL: measured, not inferred

Every engine figure in sections 0-19 was measured with that engine running
alone, so "1.69x if overlapped" was arithmetic. This measures it.

`tools/overlap/` runs both engines in one process for a fixed wall-clock
window and counts completed iterations, so the metric is sustained THROUGHPUT
rather than best-of-N (a best-case sample hides contention by construction).
The CPU leg runs on its own `userInitiated` thread so Core ML's async
machinery cannot starve it and make serialization look like parallelism.

ANE = 4-layer 1x1 conv stack at C=3072 (the section 19.2 peak).
CPU = Accelerate `cblas_sgemm`, M=1024, N=2176, K=5120. 8-second windows. [M]

| mode | ANE TFLOPS | CPU TFLOPS | total |
|---|---:|---:|---:|
| solo ANE | 6.726 | - | 6.73 |
| solo CPU | - | 2.393 | 2.39 |
| **both** | **6.881** | **1.902** | **8.78** |

**96.3% of the arithmetic sum** (8.78 measured against 9.12 ideal).

### 20.1 The contention is asymmetric, and that identifies its cause

The ANE loses nothing (6.73 -> 6.88, a rise inside noise). The CPU loses about
20% (2.39 -> 1.90).

Memory-bandwidth contention would degrade BOTH engines. Degrading only the
engine whose cycles are being borrowed points instead at **CPU cycles spent
driving Core ML** -- marshalling inputs, dispatching, collecting results.
[I, from the asymmetry]

That predicts the tax does not compound much when Metal joins, since Metal's
driving cost is also CPU-side and small. Being tested.

### 20.2 What this does and does not license

Licensed: the ANE and the CPU matrix unit genuinely run at the same time on
this hardware, and their throughputs add at ~96% efficiency. A partition that
routes work to both is sound in principle.

Not licensed yet: Metal was not in this test (MLX lives inside the package;
this binary is standalone). Nor does it show that a REAL layer decomposes
cleanly -- these are independent synthetic workloads, not slices of one GEMM
with a join at the end.

### 20.3 Three-engine isolation: Metal+ANE works, the CPU is the saboteur

Metal measured inside a 45-second load window, against a solo baseline taken
minutes earlier on the same quiet host. [M]

| configuration | Metal | ANE | CPU | combined | vs Metal alone |
|---|---:|---:|---:|---:|---:|
| Metal solo | 13.65 | - | - | 13.65 | 1.00 |
| **Metal + ANE** | **13.63** | **6.76** | - | **20.39** | **1.49x** |
| Metal + CPU | 8.56 | - | 2.32 | 10.89 | 0.80x |
| Metal + ANE + CPU | 2.76 | 6.86 | 1.91 | 11.52 | 0.84x |

**Metal and the ANE overlap essentially perfectly.** Neither degrades: Metal
13.65 -> 13.63, ANE 6.73 -> 6.76. Combined **20.39 TFLOPS, a measured 1.49x**.
This is the partition result, and it is an observation rather than arithmetic.

**The CPU leg is net negative.** It costs Metal 37% to contribute 2.3, and with
the ANE also dispatching the starvation compounds and Metal collapses to 2.76.

Cause is the harness, not the hardware: **Accelerate's `cblas_sgemm` is
internally multithreaded and takes the whole P-cluster**, starving MLX's
host-side graph construction. MLX builds its graph on the host before `eval()`
blocks on the GPU, so a saturated CPU inflates what looks like GPU time.
Whether a thread cap rescues it is under test (`VECLIB_MAXIMUM_THREADS`).

### 20.4 Standing recommendation

Partition **Metal + ANE**. Measured 1.49x, both engines at full rate, no
crossing cost in this test. Leave the CPU out unless a bounded thread budget
demonstrably stops it starving the GPU host path.

Caveat unchanged: these are independent synthetic workloads, not slices of one
projection with a join. A real partition adds scatter/gather and a
synchronisation point that this test does not model.

### 20.5 The thread-cap fix FAILED, and 20.3's explanation was wrong

| `VECLIB_MAXIMUM_THREADS` | Metal TFLOPS | CPU TFLOPS |
|---|---:|---:|
| unset | 8.564 | 2.321 |
| 4 | 8.553 | 2.316 |
| 2 | 8.568 | 2.216 |

Metal stays pinned near 8.56 at every setting. **Capping Accelerate's threads
does not rescue the CPU leg**, so 20.3's "sgemm saturates the P-cluster and
starves MLX's host graph build" is not the mechanism. Withdrawn. The true
cause of the 37% Metal loss is unidentified: either the env var is not
honoured, or the cost is memory-system or scheduler level rather than core
occupancy.

Side finding, and it corroborates section 6.2 from a second direction: **CPU
throughput is flat in thread count** (2.22 / 2.32 / 2.32 at 2 / 4 / unlimited).
More threads add no matrix throughput because SME is a **per-cluster** unit,
not per-core. Published microbenchmarks said so; this measures it. [M]

### 20.6 Final standing recommendation

**Partition Metal + ANE. Measured 20.39 TFLOPS, 1.49x, both engines at full
rate. Leave the CPU out.**

The CPU is not merely marginal, it is net negative in combination (0.80x with
Metal, 0.84x in the three-engine case), and the obvious fix does not work.
Revisit only if the 37% interaction is identified and eliminated.

---

## 21. The REAL partition is SLOWER than Metal alone (0.958x)

Section 20 measured 1.49x with independent synthetic workloads and flagged that
a real partition adds conversions, a scatter, a join and a sync point.
`ANEMetalPartitionTests` does the real thing: one affine 4-bit group-64
`gate_up` projection (M=1024, K=5120, N=17408) split by output columns, ANE
prediction dispatched to a queue while Metal runs `quantizedMM` on the
remainder, joined on a semaphore. [M]

| | ms | TFLOPS |
|---|---:|---:|
| Metal only | 13.39 | 13.63 |
| Metal + ANE, 33% split (aneN=5696) | **13.97** | 13.06 |
| **speedup** | **0.958x** | |

**The synthetic 1.49x did not survive.**

### 21.1 Why: the width curve does not transfer

Section 19.2's sweep used **square** C x C convs, so reducing C shrank both
dimensions. In a real projection **K=5120 is fixed**, so an ANE slice of Cout
columns always carries `Cout * 5120 * 2` bytes. At Cout=5696 that is **58 MB**,
inside the band where 19.2 measured 5.09 TFLOPS -- not the 6.76 peak the split
fraction was computed from.

Splitting OUTPUT columns cannot reduce the working set below `Cout * K * 2`.
Only splitting K would, and that requires partial-sum accumulation across
engines. **Applying the square-conv curve to a rectangular projection was an
error.**

At 5.09 TFLOPS the ANE needs ~11.7 ms for its share while Metal needs ~9.0 ms
for the remaining 11712 columns, so the **ANE becomes the critical path**. The
residual ~2 ms is most likely per-call `MLMultiArray` copies (~10 MB in,
~12 MB out), which the synthetic test never paid.

### 21.2 Setup cost is its own problem

**5.81 s** to dequantize the slice, bake it into a Core ML model, and compile.
Across 64 layers and several projections each that is tens of minutes, so any
integration needs compiled-model caching on disk, not just a fast steady state.

### 21.3 Status

The 1.49x in section 20 stands as a statement about the ENGINES: they do run
concurrently at full rate. It does not survive as a statement about a
PARTITIONED PROJECTION at our shapes. Whether any split fraction wins is under
test; the arithmetic suggests a shallow optimum near 25% worth roughly 1.3x
before copy overhead, which the measured ~2 ms tax would erase.

### 21.4 CORRECTION: the optimum is 22%, not 33% -- and only ONE projection wins

The 0.958x in 21 was a badly chosen fraction, not a failed partition. Sweeping
it on the real `gate_up` projection: [M]

| ANE cols | fraction | split ms | speedup |
|---:|---:|---:|---:|
| 1344 | 7.7% | 12.42 | 1.077 |
| 2560 | 14.7% | 11.74 | 1.141 |
| **3776** | **21.7%** | **11.02** | **1.221** |
| 5184 | 29.8% | 13.45 | 0.997 |
| 5696 | 32.7% | 13.97 | 0.958 |

The cliff between 3776 and 5184 is the working set crossing ~40 MB, matching
the 19.2 knee. Mechanism consistent; my fraction was wrong.

### 21.5 Numerics pass

At the optimum, with the join included: [M]

```
speedup      1.225
gpuHalfExact 0.00e+00   Metal columns match the reference EXACTLY
globalRel    0.00498    0.5% divergence from Metal-only
```

`gpuHalfExact = 0` is the structural check: the partition computes the right
thing. 0.5% is **34-54x better than the CPU column split** (0.169-0.268),
because the ANE computes fp16 from the SAME dequantized 4-bit values and fp16
carries 10 mantissa bits against bf16's 8. Only accumulation order differs.
It is not zero, so emitted tokens will occasionally differ at near-ties.

### 21.6 Only `gate_up` wins. `down` is catastrophic.

| projection | shape | speedup | setup |
|---|---|---:|---:|
| `gate_up` | K=5120, N=17408 | **1.225** | 5.0 s |
| `qkv` | K=5120, N=10240 | 0.976 | 5.0 s |
| `down` | **K=17408**, N=5120 | **0.330** | 17.0 s |

`down` degrades further with more ANE columns (0.330 / 0.257 / 0.199 at
768 / 1088 / 1536). Its K=17408 makes the ANE input 35.6 MB per call and gives
the conv 17408 input channels; the ANE slice runs near **1 TFLOPS**. The ANE is
bad at deep-input convolutions, and no fraction rescues it.

`qkv` is neutral: a smaller projection, so the ~2 ms fixed copy tax consumes
the gain.

**Rule: partition only projections that are WIDE (large N) and SHALLOW
(moderate K). Route everything else entirely to Metal.**

### 21.7 End-to-end cold prefill: ~1.08x

```
gate_up share of chunk GEMM   46.5%   (64 layers x 365 GFLOP of 50,280)
GEMM time factor  0.465/1.225 + 0.535 = 0.916  ->  GEMM speedup 1.09x
chunk  6.83 s GEMM -> 6.26 s,  + 1.22 s non-GEMM
```

**8.05 s -> 7.48 s per 1024-token chunk, about 1.08x.** [I, from measured parts]

Costs to weigh against 8%:
- ~5 s Core ML compile per `gate_up`, x64 layers = ~5 min startup without
  on-disk compiled-model caching.
- ANE weight copy for 22% of `gate_up`: ~3 GB as uint4 via
  `constexpr_blockwise_shift_scale`, ~12 GB as fp16.
- 0.5% numerical divergence; occasional near-tie token differences.
- A second numerical path to maintain, and per-projection tuned fractions.

The estimate chain that produced this: 1.49x (synthetic, independent
workloads) -> 1.22x (one real projection) -> 1.08x (whole prefill, after only
one projection shape qualifies). Each step was a measurement correcting the
previous extrapolation.

---

## 22. uint4 weights and disk-cached compilation, measured

Charging Core ML compilation to inference was invalid -- it is a build-time
artifact step like the weight transform. Both corrections were built:
weights kept UINT4 via `constexpr_blockwise_shift_scale`, and the model
compiled to `.mlmodelc` on disk and reused. `ANEMetalPartitionTests`. [M]

| config | speedup | globalRel | startup |
|---|---:|---:|---|
| fp16 weights, in-memory `MLModelAsset` | 1.221 / 1.225 | **0.00498** | ~5 s total |
| uint4 weights, disk-cached `.mlmodelc` | **1.253 / 1.301** | 0.02604 | 5.7 s compile + **16-22 s load** |

### 22.1 uint4 is slightly FASTER

1.25-1.30x against fp16's 1.22x, consistent with less weight traffic reaching
the multiplier. The compression path works and costs nothing on throughput.

### 22.2 But 5x worse numerically, and fp16 offset did not fix it

MIL's form is `scale * (data - offset)` with no additive-bias variant, so an
MLX affine weight (`scale*q + bias`) must be expressed with
`offset = -bias/scale`. That division and re-multiplication does not
round-trip: `scale * (q + bias/scale) != scale*q + bias` unless `bias/scale`
is exactly representable. The fp16 model bakes `scale*q + bias` and rounds
once.

**Untried fix:** carry scale and offset in **fp32** (`DstT` and `OffsetT` both
allow it). That should recover most of the gap.

### 22.3 Pre-compiling to disk is WORSE, and compression does not survive it

```
.mlmodel   10 MB    uint4 weights intact
.mlmodelc 288 MB    Core ML expanded them at compile time
load      16-22 s   against ~5 s to build the asset in memory
```

Compilation was never the bottleneck; loading a 288 MB compiled model is. This
is the "decompressed at model load time" branch the op's own documentation
warns about, now observed.

**Retraction:** section 21.7's "~3 GB as uint4 vs ~12 GB as fp16" was wrong.
Both are expanded in the loaded model, so the resident cost is the larger
figure either way. The uint4 saving is real on disk and in the artifact, not
in memory.

### 22.4 End-to-end, unchanged

At uint4's 1.25x on `gate_up`:

```
GEMM time factor  0.465/1.25 + 0.535 = 0.907  ->  GEMM speedup 1.10x
chunk  6.83 s GEMM -> 6.19 s,  + 1.22 s non-GEMM  =  7.41 s
```

**8.05 s -> 7.41 s per 1024-token chunk, ~1.09x.** Within noise of the 1.08x
computed from the fp16 path. Neither correction moved the end-to-end number.

Best configuration on the evidence: **fp16 weights, in-memory asset, 22%
split, `gate_up` only** -- 5x better numerics, 3-4x faster startup, and a
speedup within 3% of uint4.

---

## 23. CORRECTION: the ANE ceiling is >=12.5 TFLOPS. Depth, not width, was the constraint.

Sections 15-22 quoted an ANE ceiling of 4.97 (single op) then 6.8 (4-layer
fused). Both were **dispatch-limited**. [M]

| C | MB/layer | layers | ms | TFLOPS |
|---:|---:|---:|---:|---:|
| 1024 | 2.00 | 4 | 1.66 | 5.17 |
| 1024 | 2.00 | 16 | 3.96 | 8.67 |
| **1024** | **2.00** | **32** | **5.51** | **12.46** |
| 768 | 1.12 | 32 | 3.72 | 10.39 |
| 2048 | 8.00 | 16 | 11.74 | 11.71 |

Marginal rate rises with depth:

```
4 -> 16 layers:  25.8 GFLOP in 2.30 ms = 11.2 TFLOPS marginal
16 -> 32 layers: 34.4 GFLOP in 1.55 ms = 22.1 TFLOPS marginal
```

**12.46 TFLOPS is within 8% of Metal's 13.5.**

### 23.1 What this invalidates

Nearly every ANE measurement in this document. They used 1-op or 4-layer
graphs, so the 0.23 ms fixed dispatch cost was a large fraction of each. In
particular **section 19.2's "width optimum at C=2048-3072" was an artifact**:
at 4 layers, wider C amortised the fixed cost better, so a depth deficit read
as a width preference. At 32 layers the narrow C=1024 case (2 MB/layer, right
at the paper's on-chip working-set threshold) is the FASTEST.

That also vindicates the paper's 2 MB figure, which section 19 appeared to
contradict: the engine is compute-bound when the operand stays on chip, and
the earlier sweep was too shallow to show it.

### 23.2 What this changes architecturally

The partition should NOT split one projection between engines. It should give
the ANE a **deep contiguous run of the network** -- many fused layers in one
Core ML model -- which is how ANEMLL structures its conversions.

At Metal 13.5 and ANE >=12.5, two engines of comparable throughput running
different chunks concurrently (section 20.3 measured that overlap at full rate
for both) is worth close to **1.9x**, not the 1.08x that the shallow numbers
supported.

All prefill estimates in sections 21-22 are superseded and must be recomputed
against a deep-graph ANE rate.

---

## 24. The double-int8 lane is unreachable. Closed, with mechanism.

A dedicated probe (Opus 5 subagent, ~45 min, C probes under `tools/ane-direct/`)
attacked the doubled lane below Core ML. Clean negative, two independent locks,
each sufficient on its own.

### 24.1 Lock 1 — the compiler cannot express the precondition

The doubled rate is gated by `ZinDoubleMacMode::CanUseDoubleMacModeBasedOnFormats`,
true only when the activation and the kernel are the **same one-byte numeric
class** (PAPER 13585-13589). The shipping compiler admits **fp16-only
activations** (PAPER 5825, 1643-1645). int8 is accepted for *weights* only, and
there it is the compression path that dequantizes **at the multiply port**
(PAPER 7143-7144, 3082-3087, 862). int8 x int8 can never be formed.

### 24.2 Lock 2 — a hand-authored program will not load

Programs you build yourself are rejected at load with `0xe00002e2` /
`kIOReturnNotPermitted`: corecrypto signature check plus a trustcache vnode
check in the kernel driver (PAPER 2959-2986). Only Apple's daemon mints a
loadable program token, and that daemon is the compiler from Lock 1.

### 24.3 New relative to the published paper

On THIS M4 Max the direct IOKit client **opens with no entitlement** — the
paper's M1 account says it requires `com.apple.ane.iokit-user-access` held by
two system binaries (PAPER 2100-2103). MEASURED: `H11ANEIn` opens at connection
types 1 and 4, vending `H11ANEInDirectPathClient`, from an ordinary unsigned
process; all nine selectors dispatch and match the paper's table exactly.

It buys nothing. Both locks sit above the kernel ABI.

### 24.4 Verdict

The 8-channel lane is real in silicon and worth ~2x. Reaching it on a stock
M4 Max needs (a) an Apple-private program-signing key and (b) a compiler that
emits an int8-compute descriptor. Neither exists outside Apple. **The blocker
is cryptographic and a closed compiler, not missing engineering effort.**
Do not pursue further.

### 24.5 The maderix 1.88x refuted; only one compiler exists

A second probe (Opus 5 subagent, ~1h50m, `tools/ane-maderix/`) tested the one
door §24 left shut: the private `_ANEInMemoryModelDescriptor` that maderix
claims is a "reverse-engineered in-memory compiler". [M]

- It IS reachable unentitled (no signature error) -- but `compileWithQoS:`
  calls **`ANECCompile()`**, the same ANECompiler.framework the public
  `MLModelAsset` uses. maderix's "not public Core ML" is a mischaracterization:
  it is a private invocation of Apple's stock compiler with the same
  fp16-only-activation frontend.
- int8 activations: compiler refuses, stack falls to CPU (ANE=0/CPU=64,
  22x slower than the fp16 ANE run). The doubled-MAC precondition is never
  formed.
- int8 weights (the path that stays on ANE): no speedup. 0.717 vs 0.746 ms
  on-chip (int8 4% slower); 5.27 vs 7.04 ms at DRAM-scale weight (int8 34%
  slower).

**All three routes to the ANE -- public MLModelAsset, private
`_ANEInMemoryModelDescriptor`, direct IOKit -- funnel through one
`ANECCompile()` frontend, and it is fp16-activation-only.** The doubled lane
is closed on three independent grounds. Do not pursue.

Caveat: `macmon` reads ane_power ~0 even under confirmed 11.9-TFLOPS ANE load
on this box, so the perf/watt cross-check is instrument-unavailable; placement
(MLComputePlan) and timing are unambiguous regardless.

### 24.6 Best ANE config on real gate_up geometry

Tiling BOTH operands on-chip (weights via column tiles, activations via
sequence chunks). The 2 MB working-set threshold is hard: [M]

| config | live activations | eff TFLOPS | full gate_up |
|---|---|---:|---:|
| untiled single op | 10.5 MB | 4.9 | 37.0 ms |
| 136 tiles x 1024 cols, S=1024 | 10.5 MB | 4.6 | 40.1 ms |
| 136 tiles, S=256 | 2.5 MB | 3.2 | 57.0 ms |
| **128 tiles x 136 cols, S=128** | **1.25 MB** | **9.3** | **19.6 ms** |

S=256 (2.5 MB, just past the threshold) COLLAPSES the gain -- the 2 MB budget
is on the LIVE set and the larger operand dominates. S=128 is the operating
point.

Independent tiles top out ~9-10 eff TFLOPS, short of the 15 a CHAINED synthetic
graph reached (§23), because chained layers reuse activations in place while
tiles re-read the input and pay the fixed cost per sequence chunk. The ANE
prefers sequential depth over parallel width; one projection is the wrong shape
however finely sliced.

Partition arithmetic: ANE 19.6 ms, Metal 13.4 ms, balance f=0.41 ->
7.96 ms -> **~1.68x on gate_up**, up from 1.22x untiled.

## 25. Gated-delta (linear attention) places 100% on the ANE — but the S=512 layer ceiling is ~2.4 TFLOPS, same as attention

Fable agent `ane-gated-delta`, 2026-08-30, ~2.5 h. Swift only (hand-authored
MIL, `MLModelAsset` + `MLComputePlan` + `MLModel.prediction`), no Python. Code:
`tools/ane-gated-delta/`, binary `gdprobe` (modes `ops..ops8|chunk|chunk2|layer`).

**The science question is answered YES.** The gated-delta recurrence
(`S_t = g_t S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T`) was believed
ANE-ineligible because the sequential scan fails ANECCompile (§15). The
chunkwise-parallel WY form (Yang et al., NeurIPS 2024 / Gated DeltaNet ICLR
2025) is all matmuls plus elementwise, and it PLACES:

- [M] recurrence alone: **ANE=456 / CPU=0** at real scale (48 value heads x 8
  chunks, L=64, D=128), inter-chunk state carry included.
- [M] full real-geometry `linear_attention` layer as one MIL program:
  **ANE=593 / CPU=0**, spec 767 MB (under the 2 GB protobuf limit; no proxy
  width needed).
- [M] numerics validated vs a Swift Float sequential scan: `o maxAbsErr=3.2e-4`
  (signal max 0.115) at normalized-k operating points. fp16 through the
  triangular-inverse squaring chain holds <=1e-3 where the real model runs
  (q/k L2-normalized); re-verify against real weights before any production
  claim.

**The economics say NO to a replacement lane.** [M] Full-layer TFLOPS:

| layer | S=128 | S=512 |
|---|---:|---:|
| gated-delta (chunkwise) | 6.89 | **2.48** |
| full-attention (`ane-layer`) | 7.58 | 2.36 |

At the ranked-relevant S=512 BOTH layer types collapse to ~2.4 TFLOPS, far
below Metal's 13.5. The recurrence is NOT the cost: it adds only +5.5 ms of
161 ms at S=512 (3.4%). The bottleneck is the projection/MLP conv stack whose
activations (e.g. 17408 x 512 x 2 B = 17.8 MB) blow the ~2 MB on-chip budget —
the same working-set wall §24 found. At matched S, gated-delta is marginally
*better* than attention, not worse.

**This corrects a stale figure in this doc.** The "~7.8 TFLOPS real layer"
number quoted from the §19-era work was an **S=128** measurement. It is not a
full-width ceiling; at S=512 the real-layer ceiling for either type is ~2.4.
The 15 TFLOPS pure-conv-chain figure (§23) also does not survive on a real
layer at S=512.

**The one structural win is streamability, not per-call speed.** [I, DERIVED
from measured per-call latency] The gated-delta state is a fixed
`[48,128,128]` tensor already threaded as a program input/output, so a long
sequence runs as `ceil(S/128)` calls of the efficient S=128 layer with O(1)
carried state: S=512 as 4 x S=128 ≈ 58 ms vs 161 ms monolithic, **~2.8x** for
the same work, and the gap grows with S. Full attention cannot do this — KV
context grows and SDPA is quadratic over the whole prefix. Caveat: inter-call
scheduling gaps are not in the 58 ms figure.

**Placement idioms (MEASURED, the workaround set).** Everything needed exists
in MIL; the blockers are placement, and whole-graph all-or-nothing on small
graphs (one bad construct throws the ENTIRE program to CPU):

| construct | placement | fix |
|---|---|---|
| `cumsum` (ios16.cumsum) | CPU-only | lower-triangular ones **matmul**, tril mask as a runtime INPUT |
| elementwise with a **const tensor** operand (any rank) | CPU, poisons graph | feed all tril/eye masks as model **inputs** (~few hundred KB) |
| matmul same var both sides, no transpose (M@M) | CPU | `matmul(M, M ⊙ trilS)` (exact for strictly-lower M) or use transpose_y |
| `identity` | CPU | remove; never insert |
| scalar-const elementwise; input-tensor elementwise incl. broadcast; matmul incl. transpose_y and vector operands; slice/concat/reshape/transpose; exp/silu in context; depthwise causal conv k=4 | ANE | — |

**Bottom line.** [I] The ANE is not a replacement lane for gated-delta layers
(6.9 best-case streamed vs 13.5 Metal). It is at most a **parallel offload
lane**: 48 of 64 layers are linear-attention, overlap with Metal is real
(§20, 1.49x), and an ANE lane running gated-delta layers concurrently with
Metal *could* add throughput — but the streaming + state-marshaling + copy
costs of that pipeline are unmeasured and out of scope here. The "ANE cannot
do the scan" blocker is resolved; the "ANE is slower per-FLOP than Metal at
S=512" wall (§24) is not.

## 26. CORRECTION to §25: under-2MB sequence tiling recovers gated-delta to 6.5-9.3 TFLOPS — the S=512 collapse was a monolithic-layout artifact

Driver-run sweep on the Fable agent's `tools/ane-gated-delta/` harness (mode
`tiled`, `layer2.swift`), 2026-08-30, M4 Max, host quiet. [M] all rows.

§25 concluded gated-delta "loses at S=512" (2.48 TFLOPS). That was the
MONOLITHIC layer, which materializes the full `17408 x 512 = 17.8 MB` SwiGLU
activation. Every op except the recurrence is position-wise, so it tiles along
the sequence axis for free. Tiling the 512-token layer into `S/St` tiles with
the `[48,128,128]` recurrence state threaded across them (and the SwiGLU
intermediate split into K feature blocks with an accumulated down projection)
holds the live set small and recovers most of the throughput:

| St | K | live MB | placement | TFLOPS | vs monolithic |
|---:|---:|---:|---|---:|---:|
| 32 | 1-8 | 1.57 | ANE, CPU=0 | 3.6 | 1.5x (16 tiles, overhead-bound) |
| 64 | 1 | 2.23 | ANE, CPU=0 | 5.55 | |
| **64** | **2** | **1.57** | **ANE, CPU=0** | **6.5** | **2.6x, strictly <2 MB** |
| 64 | 4/8 | 1.57 | ANE, CPU=0 | 6.6 | |
| 128 | 1 | 4.46 | ANE, CPU=0 | 6.71 | |
| 128 | 2 | 2.68 | ANE, CPU=0 | 8.27 | |
| **128** | **4** | **2.68** | **ANE, CPU=0** | **9.3** | **3.8x, matches §24 ceiling** |
| 128 | 8 | 2.68 | ANE, CPU=0 | 9.12 | |

(St=48, 96 skipped: 512 not divisible.)

**Numeric validation [M], `gdprobe ycmp` vs the monolithic layer dump (same
weights, same seed):**
- St=64, K=2: **maxAbsErr = 0.00000, rms 0 — BIT-IDENTICAL**. The state carry
  across tiles and the causal-conv 3-col overlap are exact.
- St=128, K=4: maxAbsErr = 3e-5 (fp16 accumulation-order noise from the 4-block
  MLP split), rms ~0 — numerically equivalent, below the model's own
  block-vs-sequential divergence.

Readings:
- **The 1.57 MB recurrence state (`48*128*128` fp16) is the hard floor.** St=64
  is the largest tile keeping every other activation under it; K=2 is needed to
  put the MLP block below that floor. St=128's qkv projection
  (`10240*128*2B = 2.62 MB`) is what forces the fast config to 2.68 MB — K
  cannot shrink qkv, only the MLP.
- **The 2 MB threshold is soft.** 2.68 MB still runs 9.3; collapse comes at
  4.46 MB (6.7) and 17.8 MB (2.48). St=32 (16 tiles) falls to 3.6 —
  per-tile fixed overhead dominates. Optimum is in the middle (St=64-128).
- **Recipe:** St=64/K=2 for a bit-identical layer strictly under 2 MB
  (6.5 TFLOPS); St=128/K=4 for max speed (9.3 TFLOPS, 2.68 MB).

**Revised verdict vs §25.** Gated-delta on the ANE is 6.5-9.3 TFLOPS, not 2.48.
Head-to-head it still trails Metal's 13.5 (9.3 is 69% of Metal). But this
reopens the parallel-offload lane §25 left as the only survivor: 48 of 64
layers are gated-delta, overlap is real and non-degrading (§20, 1.49x), so an
ANE lane at 9.3 running concurrently with Metal at 13.5 is now worth costing
out end to end. The open cost is the partition itself — state marshaling,
cross-lane copies, scheduling — which §21 measured going NET NEGATIVE (0.958x)
on a bad single-projection partition. A layer-granularity depth split (whole
gated-delta layers to ANE, the rest to Metal) is a different, coarser partition
than §21 tested and is the next thing to measure.

## 27. The gated-delta ANE upper bound is 9.3 TFLOPS (St=128/K=4), boxed by an op-count wall and a score-matrix working-set wall

Full St x K sweep, driver-run, 2026-08-30, M4 Max. [M] all rows. St in
{16,32,64,128,256,512} (divisors of 512) x K in {1,2,4,8,16}.

Best-K per tile size:

| St | tiles | TFLOPS | live MB | placement |
|---:|---:|---:|---:|---|
| 16 | 32 | 0.62 | 1.57 | **ANE=0, CPU fallback** |
| 32 | 16 | 3.62 | 1.57 | ANE |
| 64 | 8 | 6.62 | 1.57 | ANE |
| **128** | **4** | **9.32** | **2.68** | **ANE (PEAK)** |
| 256 | 2 | 4.62 | 6.29 | ANE |
| 512 | 1 | 4.86 | 25.17 | ANE |

**Peak = 9.32 TFLOPS at St=128, K=4** (2.68 MB, 510 ops, 4 tiles). Global
optimum across the whole space; nothing beats it. It EQUALS the §24 pure
tiled-gate_up ceiling (9.3) — a real full layer cannot exceed a single tiled
projection because its non-conv ops (recurrence matmuls, head
reshapes/transposes) are less ANE-efficient than conv, so the §23 pure-conv
chain figure (15) is unreachable for a real layer.

The peak is boxed by two independent walls:

1. **Op-count / graph-size wall (LEFT, new).** St=16 = 32 tiles = ~3,100+ ops
   throws the ENTIRE graph to CPU (ANE=0), 0.6 TFLOPS — despite a 1.57 MB live
   set. Not a memory failure. The cliff is between 16 tiles (St=32, ~1,630 ops,
   places) and 32 tiles (St=16, fails): a hard ANE graph-size ceiling around
   **2,000-3,000 ops**. (St=256/K=16 also tipped to CPU — whole-graph fallback
   is op-MIX sensitive, not purely op-count.)
2. **Working-set wall (RIGHT).** Past St=128 the binding activation flips from
   the qkv projection (`10240*St*2B`) to the **DeltaNet score matrix
   `48*St^2*2B`**: 1.57 MB (St=128) -> 6.29 MB (St=256) -> 25.17 MB (St=512).
   Throughput falls to ~4.6-4.9 and plateaus (score-matrix bound, so K-tiling
   the MLP barely helps past St=128).

Revised bounds (correcting the inherited "2 MB threshold"):
- **Peak throughput: 9.3 TFLOPS**, at St=128/K=4.
- **Working-set bound for near-peak: ~2.7 MB** (not 2 MB); halved by ~6 MB.
- **Op-count bound: ~2-3k ops / ~16-24 tiles** before whole-graph CPU fallback.
- Smaller-than-peak live set does NOT help: St=64 at 1.57 MB gives only 6.6.
  4 tiles is the amortization sweet spot between too-few-to-pipeline and
  too-many-to-place. The peak is tile-count limited, not memory limited.

Operating point for any ANE offload lane: **St=128, K=4, 9.3 TFLOPS, 2.68 MB,
bit-identical-to-2e-5 numerics (§26).** Still 69% of Metal's 13.5 head-to-head;
the offload-lane question (concurrent ANE+Metal depth split) is unchanged from
§26 and remains the deciding end-to-end measurement.

## 28. The ANE+Metal offload lane breaks even at best: the engines contend on unified-memory bandwidth

Two-engine harness (Tests/MLXFastTests/Model/ANEMetalPipelineTests.swift), real
tiled gated-delta layer on ANE concurrent with an MLX GEMM on Metal,
2026-08-30, M4 Max. [M] measured, [I] inferred.

**Contention is real and large** (two independent measurements agree):

| measurement | ANE sustain | Metal sustain | combined TF | vs sum-of-solos |
|---|---|---|---|---|
| Stage 1 overlap (St=128) | 0.83 | **0.41** | 11.64 | 0.67 |
| footprint row (St=128) | ~0.93 | **0.45** | 12.93 | — |

When the real gated-delta layer runs on the ANE concurrently with a Metal GEMM,
**Metal collapses to 41-45% of its solo rate**. Overlap ratio 0.67, not the ~1.0
that §20's synthetic-workload 1.49x implied. §20 was memory-light; the real
layer is not, so the clean-overlap number does NOT hold for the workload that
matters. [M]

**Corrected for real Metal, the lane breaks even.** The harness's Metal proxy
runs at 6.93 TFLOPS (matched-FLOP GEMM shape), well below real Metal's 13.5, so
its combined figures are optimistic. With real Metal: concurrent combined ~=
ANE(0.83 x 9.3 = 7.7) + Metal(0.45 x 13.5 = 6.1) = **~13.8 ~= Metal-alone
13.5**. The contention tax on Metal (the faster engine) roughly cancels the
ANE's added throughput. [I]

**Why the footprint lever (shrink the ANE working set) does not fix it.** The
contended resource is unified-memory BANDWIDTH, dominated by the layer's ~175 MB
of 4-bit weight reads per forward -- NOT the on-chip working set (§24-27). The
St knob moves the wrong resource: shrinking St reduces on-chip SRAM footprint
(which Metal does not contend for) while INCREASING weight re-reads (St=64 =
8 chunks re-read weights vs St=128 = 4). Structurally this holds Metal's sustain
flat or worse. The St=64 empirical point could not be captured -- repeated
distinct-model ANE AOT compiles (Espresso E5, 25 GB peak) wedged the CoreML XPC
daemon -- but the mechanism predicts no recovery. [I]

**The only remaining ANE bandwidth lever** is fewer/larger forwards (St=512 =
weights read once, not 4-8x), but that hits the 25 MB score-matrix spill (§27,
St=512 -> 2.48 TFLOPS solo). Its own experiment; not obviously net positive.

**Verdict.** The science is settled and positive (gated-delta places on the ANE,
9.3 TFLOPS, bit-identical, §26/27). The engineering payoff for cold prefill is
not there: single-engine the ANE (9.3) is slower than Metal (13.5), and
concurrent the two engines contend on memory bandwidth so the offload lane
breaks even with Metal-alone. Shelve the ANE lane for prefill; the GPU
quantized-GEMM and CPU-SME paths remain the better-founded bets. Harness note:
the two-engine test carried three real bugs found and fixed during measurement
(semaphore-grid deadlock, per-cell alloc memory wedge, defer-only report loss);
its repeated-distinct-model load path remains fragile against the ANE compiler.

## 29. CORRECTION to §28: the ANE+Metal contention is NOT bandwidth — arithmetic intensity rules it out; power/thermal or dispatch is the likely cause

§28 attributed Metal's 41-45% collapse under concurrent ANE load to
unified-memory weight bandwidth. That is arithmetically wrong. [I]

Arithmetic intensity of the gated-delta layer: 409 GFLOP per 512 tokens against
~200 MB of 4-bit weights (383M params x 0.5 B + group-64 scales), read ~4x
across St=128 chunks -> ~500 FLOP/byte. To sustain 9.3 TFLOPS the ANE needs
9.3e12 / 500 = ~18 GB/s of weight bandwidth. Metal at 13.5 TFLOPS on
similar-intensity work needs ~27 GB/s. Combined ~45 GB/s against the M4 Max's
~546 GB/s bus = **~8% utilization**. Bandwidth is not the contended resource; it
cannot explain a ~50% Metal collapse. [I]

Corrected likely causes of Metal -> 45% under concurrent ANE:
1. **Power/thermal DVFS** (leading hypothesis): GPU (~30-50 W) + ANE together
   exceed the sustained package power budget; the GPU clocks down. A ~50%
   throughput drop is consistent with combined-load throttling. If this is the
   cause, the offload lane is dead for a FUNDAMENTAL reason (two big engines
   cannot both run flat-out within one power budget) — not fixable by
   scheduling, prefetch, or footprint.
2. **CPU-side dispatch contention**: MLX eval and CoreML prediction both need
   the CPU to submit/synchronize GPU/ANE work; they may serialize host-side.
   This WOULD be a harness artifact, potentially fixable with async dispatch.

Not yet distinguished (would need a power-telemetry-instrumented run: macmon
package power during solo-GPU vs concurrent, and a dispatch-isolated variant).
The §28 practical verdict (offload breaks even, shelve for prefill) stands, but
the mechanism is power/dispatch, not bandwidth. NOTE the cross-implication: if
the cause is power, any CPU-SME concurrent lane (§ priority-2 work) faces the
same package-power wall, reinforcing §20's net-negative CPU finding.

SRAM addendum: neither engine's on-chip memory can hold a ~200 MB layer (ANE
working SRAM is a few MB; GPU threadgroup memory tens of KB; the shared SLC tens
of MB). "Pre-load a layer into SRAM while the other engine works" is therefore
infeasible AND unnecessary — weight streaming needs only ~18 GB/s. ANE
activations already fit on-chip (<2 MB tiling, §26); weights never can, and do
not need to.

## 30. Dispatch contention RULED OUT: pure CPU load does not reproduce Metal's ANE-induced collapse

Discriminator test (Tests/MLXFastTests/Model/ANEMetalPipelineTests.swift,
`metalDispatchSensitivity`), 2026-08-30. Metal 4-bit GEMM throughput vs N pure
CPU spinner threads (arithmetic only; no accelerator, no memory traffic, no
XPC). [M]

| CPU spinners | Metal sustain vs solo |
|---:|---:|
| 0 | 1.00 |
| 1 | 0.78 |
| 2 | 0.85 |
| 4 | 0.88 |
| 6 | 0.98 |
| 8 | 1.01 |

Pure CPU load does NOT reproduce the ANE's 0.45 Metal collapse (§28). Even 8
pegged cores leave Metal at ~1.0; the 1-spinner 0.78 dip is a core-placement
artifact that recovers as spinners spread. Robust to host contamination: any
background load would only lower these, and they stayed high.

**Conclusion: the ANE->Metal collapse is ANE-specific hardware co-execution,
NOT generic CPU-dispatch starvation.** The "free the CPU / async submission"
fix is therefore dead. By elimination, the mechanism is NOT bandwidth (§29, 8%
utilized), NOT CPU dispatch (this test), NOT thermal (idle, cool machine). The
remaining candidates, all ANE-specific and none software-schedulable:
1. memory-fabric/controller ARBITRATION (interleaved request streams add
   latency to GPU accesses even at low bandwidth);
2. power-delivery DVFS (GPU clocks down when the ANE unit draws current, even
   cool);
3. kernel/IOKit driver-path contention (CoreML ANE submit vs MLX Metal submit
   both enter the kernel driver -- the one candidate the pure-CPU spinner does
   not exercise; would need a separate test, and is a long shot for a lane that
   only breaks even at full overlap).

Not distinguished further (would need macmon package-power + GPU-clock telemetry
during solo-GPU vs ANE+GPU, via the wedge-prone concurrent harness). Not worth
it: the ANE offload lane is break-even at best regardless of which of these it
is, so the §28 verdict (shelve for prefill) stands with the mechanism now
bounded to hardware-level ANE/GPU co-execution.

## 31. Chained-GEMM dilution decomposed: the dominant term is shape-SWITCHING, not per-shape ceiling or dependency

Agent gpu-gemm-dilution, 2026-08-30, M4 Max. [M] except where noted. Reused
existing GemmChainCostTests / SameShapeChainTests / PrefillMatmulCostTests.

Reconfirmed §14: one GEMM alone 13.5 TFLOPS; 7 gated-delta projections isolated
(separate eval, no dependency) 8.84; chained (one eval, dependent) 7.36.
Isolated/solo = 1.53x; chained/isolated = 1.20x.

Decomposition of the 1.53x isolated dilution:
- **Dependency chaining alone**: ~1.23x. SameShapeChainTests, identical 5120^2
  shape, 7 deep, fresh weights: 13.5 -> 10.99. Matches the doc's prior 1.2x.
- **Per-shape ceiling**: NOT the cause. Pricing each of the 7 real projection
  shapes at its OWN honest solo rate (qkv 13.5, z 13.0, ba 1.64, out 13.3,
  gate/up 13.4, down 13.5) predicts 59.1 ms for the chunk; measured isolated is
  88.79 ms. The `ba` shape (96 output cols) is genuinely slow solo (1.64
  TFLOPS) but is 1 of 785 GFLOP (0.6 ms) -- negligible.
- **Residual: 1.50x (88.79 / 59.1)** unexplained by ceiling or dependency. It
  correlates with ALTERNATING between different GEMM shapes in one process ->
  leading candidates **Metal pipeline-state switching or allocator churn** per
  shape change. Not yet separated (needs an alternating-vs-grouped 2-shape
  chain on a quiet host). [I]

**Actionable direction** (the real prefill lever, ~85% of prefill is GEMM):
reduce the per-shape-switch cost -- group same-shape ops, cache/warm Metal
pipeline states across shapes, reuse allocations across the projection sequence.
No fix prototyped yet.

**Measurement-environment caveat**: this run could not get clean absolute
numbers. spotlightknowledged (~99% CPU), OrbStack, and concurrent agents made
host-quiet-gate.sh refuse; re-measuring one fixed shape gave 8.5 vs 11.8 vs the
doc's clean 13.4 (~1.4x host-noise spread). The RATIOS above are from
same-session back-to-back runs and clean doc numbers; absolute TFLOPS on a noisy
host are not reliable. Clean prototyping needs a quiesced machine.

## 32. UNIFIED FINDING: concurrent memory-touching work on ANY second engine degrades the GPU ~30-55% — memory-fabric arbitration, not bandwidth, dispatch, or thermal

Agent cpu-sme-lane (2026-08-30) + §30 spinner test + §28 ANE test, together
resolve the multi-engine contention mechanism. [M] / [I].

Three results reconcile into one mechanism:

| second-engine workload | touches unified memory? | GPU throughput |
|---|---|---|
| pure-compute CPU spinners (§30) | NO | ~1.0 (unaffected) |
| CPU SME2 / cblas_sgemm matmul (cpu-sme-lane) | YES | 0.52-0.73 (lost 27-48%) |
| ANE gated-delta layer (§28) | YES | 0.45 (lost 55%) |

The discriminator is **memory traffic, not CPU cycles and not bandwidth
volume**. Pure compute on a second engine is free; any concurrent
memory-TOUCHING work degrades the GPU 30-55%, even though aggregate bandwidth
is only ~8% utilized (§29). Signature of **memory-fabric / controller
ARBITRATION**: a second engine's memory requests interleave with the GPU's and
add latency to its stream, independent of bandwidth headroom. Hardware property
of the unified-memory M-series; not fixable by software scheduling, QoS, or
footprint. [I]

cpu-sme-lane specifics [M]: SME2 raw ceiling ~1.4 TFLOPS/thread, ~2.7 at 2
threads (weak); cblas_sgemm 2.0-2.7 TFLOPS EXCEEDS raw SME2, so the §-prior 2.48
was SGEMM/NEON-bound, not SME2. Every concurrent config net-NEGATIVE vs
GPU-alone (best case -1.6 TFLOPS); QoS lowering did nothing; only cutting GPU
dispatch frequency shrank (never flipped) the loss. Extends §20.

**Strategic consequence.** Multi-engine offload is closed for GPU-bound prefill:
the GPU is the fast engine and any concurrent memory-touching helper (ANE or
CPU) slows it more than it adds. The remaining lever is SINGLE-ENGINE GPU
optimization, which does not fight the fabric -- above all the shape-switching
dilution recovery (§31, ~1.5x recoverable on the GPU alone). That is where
cold-prefill effort should go.

One lever untested (both agents flagged, out of budget): CPU doing the ~15%
NON-GEMM prefill work (gated-delta scan, conv1d, norms) instead of GEMM. It
still touches memory, so §32 predicts it is also degraded by fabric contention
when overlapped -- likely net-negative or marginal, but not measured.

## 33. DIRECT confirmation: pure memcpy (no driver, no dispatch, no shared path) crushes Metal — driver-path and coding-artifact ruled out

Discriminator `metalMemoryContention` (ANEMetalPipelineTests.swift),
2026-08-30. memcpy-only threads (64 MB buffers >> LLC -> DRAM traffic; no
arithmetic, no IOKit, no XPC, no accelerator, no shared submission path with
Metal). A/B vs §30's pure-compute spinners on the same host (Spotlight noise
common-mode; sustain ratios robust). [M]

| threads | memory-streaming sustain | compute-spinner sustain (§30) |
|---:|---:|---:|
| 0 | 1.00 | 1.00 |
| 1 | **0.40** | 0.78 |
| 2 | **0.45** | 0.85 |
| 4 | **0.56** | 0.88 |
| 6 | **0.50** | 0.98 |

One memcpy thread drops Metal to 0.40; pure compute barely moves it. Memory
degradation (0.40-0.56) matches ANE (0.45, §28) and CPU-SME (0.52-0.73, §32).

**Rules out, by direct measurement:**
- **kernel/IOKit driver-path contention**: memcpy makes zero driver/IOKit/XPC
  calls yet degrades the GPU as hard as the ANE. A driver-path cause cannot be
  produced by a workload with no driver path.
- **synchronous/async coding artifact**: memcpy shares no queue, lock,
  submission path, or dependency with the Metal loop, and has trivial one-shot
  dispatch. A slowdown caused by a thread sharing nothing but DRAM cannot be an
  artifact of how the harness dispatches, and no async restructuring removes it.

Nuance [I]: one memcpy thread pushes ~50-100 GB/s, far more than the ANE
layer's ~18 GB/s (§29). So memcpy proves memory ACCESS is sufficient to degrade
the GPU and cleanly rules out driver/dispatch/coding, but does NOT prove the
ANE's damage is raw bandwidth volume. The ANE hurting Metal as much as memcpy
despite far lower bandwidth points at fabric ARBITRATION/latency (interleaved
request streams add latency to the GPU independent of volume). Either way:
memory-fabric hardware, not software-addressable. This closes the mechanism
question for §28/§30/§32.

## 34. INCONCLUSIVE (host noise): cache-resident vs DRAM contention could not be measured; measurement quality is now the binding constraint

Attempted the "keep data on-chip to dodge fabric contention" test
(`metalCacheVsDram`): one memcpy thread, working set swept 256 KB -> 256 MB,
Metal sustain each. Result NOT trustworthy. [host-noise-invalid]

This run: 256KB=0.74, 2MB=0.76, 8MB=0.84, 32MB=0.86, 128MB=0.93, 256MB=0.96
(monotonic, small hurts MORE). But the 128 MB row copies a 64 MB half-buffer --
the identical op to §33's 64 MB memcpy, which measured 0.40. Same operation,
0.93 vs 0.40 across runs = 2.3x disagreement => host-noise-dominated
(spotlightknowledged ~99% CPU + background apps). No cache-vs-DRAM conclusion
can be drawn. Do NOT cite this run's monotonic trend as a finding.

What still holds: the QUALITATIVE fabric result (§30 compute spinners ~0.9 vs
§33 memory streaming lower) survives because it is a same-run A/B. The
fine-grained size/cache dependence does not.

Architectural blocker independent of the measurement: even if cache-resident
traffic proved contention-free, a cross-engine on-chip partition has no
workload -- a layer's ~200 MB weights cannot be cache-resident, and the only
weight-free ops (attention scores) are a thin FLOP slice. There is no shared
programmable on-chip memory between ANE and GPU anyway (private threadgroup
mem / ANE SRAM; the shared SLC is an unpinnable hardware cache; no cross-engine
tensor-stream API).

**Binding constraint going forward: measurement quality.** Three consecutive
timing efforts (GPU dilution absolute numbers, this sweep) were degraded by
host noise; host-quiet-gate.sh correctly refused. Any further timing work --
including the §31 GPU shape-switching dilution fix -- needs a quiesced machine
(Spotlight indexing off/finished, OrbStack closed, gate passing). The
single-engine GPU-fusion direction (fuse intermediates into registers/
threadgroup memory to cut the GPU's own DRAM traffic) remains the productive
lever and does not require a second engine or fabric sharing.

## 35. CORRECTION: the ANE->GPU slowdown is REAL (0.61, not 0.45) and genuinely ANE-caused — GPU-fallback artifact ruled out by power; milder magnitude may make offload NET-POSITIVE

Prompted by skepticism that 8% bandwidth cannot cause 45% loss (correct instinct)
plus a Fable literature pass (ane-gpu-contention-research: no documented
mechanism for 45% at 8% BW; mlx-vlm #1943 hybrid ANE+GPU prefill measured net
+32%, not a collapse; prime suspect = Core ML GPU-fallback making a 2nd Metal
client). Resolved with two clean tests. [M]

**(a) Noise-robust paired toggle** (`aneMetalToggle`): Metal runs continuously;
ANE toggles on/off in 8 adjacent paired cycles so slow host noise cancels in the
ratio. Mean Metal on/off ratio = **0.61** (min 0.57, max 0.70, tight). The
slowdown is REAL and robust to host noise -- but MILDER than the single
before/after 0.45 (§28), which was noise-inflated.

**(b) ANE-only power check** (macmon, no Metal): during the St=128 predict loop,
ane_power ~4.24 W while gpu_power ~0.003 W (max 0.05), gpu_usage <1.4%. The
workload is genuinely on the ANE; the GPU is idle. **Core ML GPU-fallback ruled
out** -- the 0.61 is not GPU-vs-GPU time-slicing, it is real ANE<->GPU
co-execution contention. Mechanism candidates unchanged (fabric arbitration /
SLC / power), still not software-fixable, but the effect is real.

**Economics REVISED.** §28 used 0.45 and concluded break-even ("dead"). With the
corrected 0.61:
- combined ~= ANE(0.83 x 9.3 = 7.7) + Metal(0.61 x 13.5 = 8.2) = **15.9 TF vs
  Metal-alone 13.5 = ~1.18x, NET POSITIVE** [I, arithmetic].
- Consistent with mlx-vlm's measured +32% hybrid precedent.

CAVEATS (why this is not yet a green light): the 0.83 ANE sustain is from the
noisy Stage 1; "combined" is arithmetic not a measured combined-throughput run;
and the every-4th-layer attention pattern limits free partitioning (pipeline
bubbles, §-pipeline). The DIRECTION has shifted from "dead" to "plausibly
net-positive," but confirming needs a clean MEASURED combined-throughput test on
a QUIESCED host, plus honest pipeline-structure accounting. §28/§32's "shelve"
verdict is DOWNGRADED to "re-measure on a quiet machine before deciding."

## 36. mlx-vlm/omlx hybrid ANE+GPU prefill: concurrent channel-split, +32-36% on our exact architecture — our 0.61 was a saturated-duty-cycle worst case

Fable study (mlxvlm-hybrid-study) of mlx-vlm #1943 + jundot/omlx #2756. [sources
in report; evidence-graded there].

- **Concurrent, not serial** [confirmed]: per-GEMM output-CHANNEL split -- 40-53%
  of MLP gate/up + GDN projection channels on the ANE, concurrent with Metal
  computing the channel suffix. Our concurrent-offload framing is the right
  model. omlx #2756 merged 2026-08-17.
- **Numbers**: +32.57% M1 Max (Qwen3.8-27B-4bit GDN text tower, 2048-tok
  prefill); +35.6% M3 Ultra. Architecturally an EXACT match to our target.
- **Duty cycle is the key correction** [inferred]: their GPU-under-contention
  efficiency is ~0.80 vs our measured 0.61, because the ANE duty cycle is only
  ~38.8% (bursty per-layer), not saturated. Our toggle/§35 (0.61) and the
  saturated hybrid test run the ANE at 100% duty -- the WORST case. Real
  per-layer offload contends the GPU only ~40% of the time, so effective
  slowdown is milder and the economics are better than §35's 1.18x.
- **Mechanism**: a PRIVATE ANE "procedure bank" API (INT8 requant), NOT Core ML.
  That is what enables fine-grained concurrent channel-split at low duty. Our
  Core ML (MLModelAsset) path is coarser; whether it can match this is open.

Transfer to our LOCAL-SERVE goal (user does not care about the ranked track):
- Ranked-track blockers (token-fidelity cosine 0.9985; INT8 quant-envelope
  violation) do NOT apply to local serve -- irrelevant to this user's goal.
- Real hurdle: the private ANE API vs our Core ML path.
- One M5 datapoint 6x slower/2x memory, but M1 Max + M3 Ultra both won; M4 Max
  is untested. 

Consequence for our measurement: re-measure contention at ~40% ANE duty cycle,
not saturation; the saturated hybrid number is a lower bound on the offload win.

## 37. Measured hybrid offload: NET-POSITIVE ~1.31x (noise-robust), confirming §35/§36 -- raw ratio noise-inflated, corrected via sustains

hybridCombinedThroughput test, paired GPU-solo vs GPU+ANE concurrent, 6 cycles.
Two runs (bf16 then 4-bit GPU workload). [M] sustains; [I] corrected ratio.

Raw mean ratio came out 2.04 (bf16) / 2.06 (4-bit) -- both INFLATED because the
GPU-solo baseline measured only ~7.1 TF instead of the clean ~13.5 (§14/§31).
Cause is persistent host noise: spotlightknowledged, then corespotlightd (~232%
CPU), plus TGOnDeviceInferenceProviderService (Telegram on-device inference,
uses ANE/GPU -- directly contaminates). The additive combined metric is
noise-sensitive: noise depresses the GPU (both phases) but not the ANE
contribution, inflating combined/solo.

Noise-ROBUST extraction (paired sustains cancel host noise; combine with clean
solo rates from prior clean runs):
- GPU sustain under concurrent ANE = 0.62 [M] (matches toggle 0.61, §35).
- ANE sustain under concurrent GPU ~= 1.0 [M] -- the ANE is NOT slowed by the
  GPU; the GPU is the sole victim (asymmetric).
- combined = GPU(0.62 x 13.5) + ANE(1.0 x 9.3) = 8.4 + 9.3 = 17.7 TF vs
  GPU-alone 13.5 => **ratio ~= 1.31, NET POSITIVE** [I].

This is the measured confirmation of §35's 1.18x arithmetic and consistent with
mlx-vlm's measured +32% on this architecture (§36). Saturated-ANE worst case;
the ~40% duty-cycle correction (§36) pushes it higher. 

Caveat: a fully clean absolute ratio still needs a quiescent host (the raw ratio
cannot be trusted while corespotlightd/Telegram-inference run). But the
DIRECTION is robust across every measurement: offloading gated-delta work to the
ANE concurrently with GPU GEMMs beats GPU-alone. The §28/§32 "shelve" verdict is
OVERTURNED -- the offload lane is net-positive; the earlier "dead" rested on a
noise-inflated 0.45 slowdown and a break-even arithmetic.

Higher-upside lever now under study: the omlx "procedure bank" private ANE API
that runs INT8 (the doubled-MAC lane, §24) -- if replicable within security
constraints, the ANE side runs up to 2x faster, materially improving the ~1.31x.

## 38. "Procedure bank" verdict: NOT an int8 unlock (fp16 ANECCompile), but a legitimate replicable fp16-concurrency path; omlx +32-36% cross-validates our ~1.31x

Agent procedure-bank-study read the actual omlx source + native dylib installed
at /Applications/oMLX.app/Contents/Resources/omlx/ (real MIL templates and
selectors from the binary, not web guesswork). [strong evidence].

1. **Signature = legitimate (GO)**: the procedure bank calls Apple's own
   compiler via `_ANEInMemoryModelDescriptor` + `compileWithQoS:` (= ANECCompile).
   Apple mints the signed token; we supply MIL text + weight bytes. NO SIP-off,
   entitlement, forging, or trustcache edit. Same door §24.5's maderix probe
   already opened unentitled on this M4 Max (no 0xe00002e2 -- that error only hit
   hand-forged programs skipping the compiler). It IS ANECCompile, not a bypass.
2. **int8 doubled lane STAYS CLOSED**: embedded MIL is
   `func procedureNNN<ios18>(tensor<fp16,[1,K,1,S]> x)` -- fp16 activation --
   with int8 WEIGHTS via constexpr_blockwise_shift_scale -> dequant-to-fp16 (the
   §17/§24 compression path). No int8xint8, no doubled-mac mode in the binary.
   The "procedure bank = int8 unlock" hypothesis is REFUTED.
3. **omlx's +32-36% is pure fp16 GPU-parallel-ANE CONCURRENCY** engineering:
   per-GEMM output-channel split, IOSurface zero-copy handoff, dual-ANE-instance
   pinning, 256-procedure banking. Not a faster ANE datapath.

**Cross-validation**: omlx's measured +32-36% = 1.32-1.36x matches our
independent noise-robust ~1.31x (§37) -- two different methods, same magnitude.
Strong evidence the fp16 offload is genuinely net-positive at ~1.3x.

Replicability: GO within constraints (fp16, legitimate compiler path). Minimal
proof-of-life probe ~1 ObjC file / ~1 eng-day (tools/procedure-bank-study/).
Caveats: (i) confirm ANE placement via IOReport/powermetrics (macmon ANE power
unreliable per §24.5, though our §-power test did read 4.2 W cleanly); (ii)
M4-Max net speedup UNTESTED -- both omlx winning datapoints are M1/M3, not M4;
(iii) needs a quiesced host to measure. The real work is the concurrency
engineering (channel split + zero-copy + dual-ANE pinning), not a datapath trick.

## 39. Shape-switching REFUTED as the dilution cause: grouping same-shape GEMMs is free (ratio 0.974) -- the §31 fix does not work

Clean paired experiment (ShapeSwitchTests.shapeSwitchCost): 16 GEMMs (8 qkv-shape
[10240,5120] + 8 gate-shape [17408,5120], 4-bit, S=512) run ALTERNATING
(s1,s2,... 16 switches) vs GROUPED (s1x8 then s2x8, 1 switch). Same
FLOP/shapes/chain-depth; only order differs. [M].

Mean alt/grp ratio = **0.974** (min 0.803, max 1.050) -- grouping is if anything
marginally SLOWER. Ordering/shape-switching is FREE in the dependent-chain
regime. The §31 "1.5x residual = pipeline-state switching + allocator churn"
hypothesis is REFUTED (it was a noise artifact; that agent could not get clean
numbers). Grouping same-shape projections / caching pipeline states would
recover NOTHING -- do not build it.

The dilution is still real (this chained 16-GEMM sequence runs ~7.7 TF vs 13.5
single), but the cause is dependency serialization + weight-cache residency
(many distinct weight matrices thrashing cache), NOT switching. Those are
largely inherent to a sequential multi-projection forward and much harder to
recover than a reordering.

Strategic consequence: the GPU-dilution lever's cheap win is GONE. The validated
fp16 ANE offload (~1.31x, §37/38, cross-checked by omlx +32-36%) is now the more
promising cold-prefill lever -- it adds a second engine's throughput rather than
trying to un-dilute an inherently serial GPU chain. Recommendation flips to:
pursue the ANE offload probe.

---

## §40. ANE offload built and measured: makeInput fix + honest A/B verdict (2026-08-31)

The ANE offload was built two ways (Coarse whole-projection, ChannelSplit
per-GEMM output-channel split) and measured against an all-GPU reference MLP
on the idle M4. This section records the decisive diagnostic and the final
verdict. It supersedes the "pursue the ANE offload probe" recommendation at the
end of §39: the probe was pursued, and the outcome is negative for this
architecture.

### 40.1 The 100x slowdown was one line: asData() on a strided view

The first A/B measured the offload 100-500x SLOWER than all-GPU. Per-call
isolation (S=512, K=5120) found the cost was entirely `makeInput`, flat at
~2.4s regardless of ANE output size, while `predict` (the ANE compute) ran
correctly at 2.9-19ms (~5 TF) and `readOutput` at 0.3-10ms. A step breakdown of
`mlxToMultiArray_1C1S` located the whole 2.4s in `xT.asData().data`
(transpose+eval 0.13ms, alloc 0.02ms, memcpy 0.11ms).

Root cause, read from `MLXArray+Bytes.swift` `copy(from:toContiguous:)`: the
transpose `[S,K] -> [K,S]` is a lazy view with strides `[1,K]`, which match no
contiguous layout, so `contiguousToDimension()` reports "nothing contiguous"
and `asData()` copies one fp16 element at a time over K*S elements. That
per-element scalar copy is the 2.4s (8.5s at K=17408).

Fix: `contiguous(xT)` before `asData()` forces one GPU kernel to write a
row-contiguous buffer, after which `asData()` takes the whole-buffer memcpy
fast path. Measured: `makeInput` 2442ms -> ~0.7ms at K=5120, 8346ms -> ~1.4ms
at K=17408 (~3500x). Same bytes. Committed; ANEGemm arbitrary-S and split-path
equivalence tests stay green.

### 40.2 The GPU already reaches 13.5 TF; single-op probes underfill

A parallel question (why local probes read ~7.2 TF at N=17408 vs the doc's
13.53) resolved as a measurement artifact: isolated single-op `quantizedMM`
eval-each reads 4.93 TF, chained 7.8 TF, but the real all-GPU MLP (3 chained
projections + silu) runs at 0.0205s for S=512 = 13.4 TF (12.6 TF at S=1024).
The GPU reaches full throughput in the real chained forward; single-op probes
do not fill it. No GPU problem exists.

### 40.3 Post-fix A/B: every offload variant still loses

With `makeInput` fixed, the A/B was re-run (synthetic weights, idle machine,
1-ULP correctness on all variants). Ratios are ref/variant; below 1.0 means the
offload is slower:

| variant | S=512 | S=1024 |
|---|---|---|
| CoarseOffloadMLP | 0.59 | 0.58 |
| ChannelSplitMLP f=0.25 | 0.50 | 0.52 |
| ChannelSplitMLP f=0.375 | 0.32 | 0.39 |
| ChannelSplitMLP f=0.50 | 0.40 | 0.33 |

Every variant loses to all-GPU by 1.7-3x. The fix removed the 100x pathology
but did not flip the outcome.

Why, precisely: `makeInput` (~1ms) and `readOutput` (~2.5-10ms) run on the
caller thread; only `predict` overlaps the GPU. Each split projection pays a
non-overlappable `makeInput + readOutput` tax (~3.5ms) on top of
`max(predict, gpuSuffix)`, times three projections is ~10ms of pure overhead.
The all-GPU reference pipelines all three projections in one `eval` with no
barriers (~21ms total). The offload inserts three CPU-roundtrip barriers: ANE
fp16 output -> CPU bytes -> re-upload to MLX -> concat. Coarse loses for a
second, structural reason: ANE at ~5 TF is slower than GPU at ~13.4 TF, so the
whole `up` on the ANE (19ms) exceeds the whole `up` on the GPU (7ms).

The earlier "channel-split at f~0.27 gives 1.38x if makeInput is free" estimate
was wrong. It modeled only the predict-vs-gpuSuffix overlap and ignored the
caller-thread conversion barriers and the CPU roundtrip. The measurement
supersedes that model.

### 40.4 What a win would actually require

A winning ANE offload needs omlx's architecture, not Core ML
`MLModel.prediction` plus an MLX roundtrip:

- IOSurface zero-copy handoff, so the ANE result lands on a GPU-shared buffer
  with no `makeInput`/`readOutput` conversion and no CPU roundtrip (removes the
  ~3.5ms/projection non-overlappable tax and the barrier).
- A private ANE procedure-bank dispatch path with far lower per-call latency
  than `MLModel.prediction`.

Both are a substantially larger build than the Core ML path measured here, and
omlx's winning datapoints are M1/M3 — the M4 ANE is untested for this pattern.

### 40.5 Verdict

The `contiguous(xT)` fix is a real, committed improvement (it removes a 3500x
input-conversion pathology and is useful for any future ANE work). It does not
make this offload architecture beat an all-GPU MLP on the M4. Recommendation:
do not ship the Core ML offload path as a prefill accelerator; a genuine
attempt requires the IOSurface + procedure-bank rebuild, which is a separate,
larger decision.

---

## §41. Feasibility spike: IOSurface zero-copy + private ANE procedure-bank on M4 (2026-08-31)

Question: can the two things omlx has and our Core ML offload lacks — IOSurface
zero-copy handoff and a private procedure-bank dispatch (not
`MLModel.prediction`) — be built on this M4, unentitled? Ran the existing
`tools/ane-direct` and `tools/ane-maderix` probes. Answer: yes, all gates clear.

### 41.1 Reachability results (measured, this box, unentitled)

- IOKit ANE user client: `H11ANEIn` (service `H11ANE`) `IOServiceOpen` returns
  `OPENED` for connection types 1 and 4; `H1xANELoadBalancer` (ANEDriverRoot)
  type 1 also opens. `AppleT6041ANEHAL` is denied (kIOReturnNotPermitted) but
  is not the client ANEServices drives. The raw kernel path is open unentitled.
- Private frameworks: `ANEServices`, `ANECompiler`, `AppleNeuralEngine` all
  `dlopen` OK; `ANECCompile` resolves.
- In-memory compile path (`_ANEInMemoryModelDescriptor` /
  `_ANEInMemoryModel` / `_ANEClient`): all classes present and instantiable.
  `modelWithMILText:weights:optionsPlist:` builds a real descriptor with a
  valid `hexStringIdentifier`; `inMemoryModelWithDescriptor:` builds a model;
  `compileWithQoS:options:error:` invokes `ANECCompile()` and fails with
  `InvalidCompilationParam` — NOT an entitlement error. The probe fed a
  deliberately empty MIL; the entitlement gate is already passed, and we
  already have a valid MIL (`buildConvMatmul`, which Core ML compiles and runs
  today). `_ANEInMemoryModel` exposes `loadWithQoS:options:error:` (residency),
  `evaluateWithQoS:options:request:error:` (dispatch), `unloadWithQoS:`,
  `queueDepth`, `programHandle`, `intermediateBufferHandle`, `sharedConnection`
  — the procedure-bank primitives.
- Zero-copy output: MLX-swift natively supports
  `MLXArray(rawPointer:shape:dtype:finalizer:)`, "transfers ownership of a raw
  pointer compatible with an MTLBuffer", and its own doc example wraps an
  `IOSurface.baseAddress`. On unified memory one IOSurface aliases an ANE
  buffer and an MLXArray with no copy.

### 41.2 What is left to build (not present anywhere in tree today)

- Drive `loadWithQoS` + `evaluateWithQoS:request:` to completion with the real
  `buildConvMatmul` MIL, replacing the Core ML `MLModel.prediction` call in
  `ANEGemm`. Requires reverse-engineering the `_ANERequest` input/output
  buffer-binding ABI (dump `_ANERequest` — only Descriptor and InMemoryModel
  were dumped so far). This is the one real unknown-effort chunk.
- Allocate input/output IOSurfaces, bind to `_ANERequest`, alias both to MLX.
  Input side needs one GPU-side blit MLX-buffer -> input IOSurface (sub-ms,
  contiguous), or arranging MLX to compute into an IOSurface-backed buffer.
- Wire the new ANE leg into `ChannelSplitMLP` in place of the Core ML path.

### 41.3 Risk and payoff ceiling

- Private API: breaks on macOS updates, unsupported, not submittable to any
  sanctioned track. Acceptable ONLY for the local serve fork (own machine, not
  ranked). omlx's winning data is M1/M3; M4 ANE untested for this pattern.
- Payoff is bounded by ANE throughput, not by the handoff. ANE ~5 TF vs GPU
  ~13.4 TF caps a zero-copy channel-split at ~1.4x on the MLP projections at
  f~0.27, and only on the MLP (attention and MoE routing are unaffected), and
  only in prefill. Realistic end-to-end cold-prefill gain is roughly 1.1-1.25x
  for a substantial private-API build. The handoff fix removes the barrier tax
  that made the Core ML path LOSE; it does not raise the ceiling above the
  overlap math.

### 41.4 Verdict

Buildable on this M4, unentitled, with one genuine reverse-engineering chunk
(`_ANERequest` buffer binding). It is an architectural build (a new private ANE
dispatch runtime), local-fork-only, with a modest (~1.1-1.25x prefill) ceiling.
Worth it only if that prefill gain matters enough to justify a private-API
subsystem that macOS updates can break.

---

## §42. Private-ANE dispatch: compile is open, load+execute is entitlement-walled (2026-08-31)

Goal: replace the Core ML ANE leg with a private in-memory compile + IOSurface
zero-copy + procedure-bank dispatch, to remove the conversion barriers that
made the Core ML channel-split offload lose (§40). Built and probed the whole
path unentitled on this M5/macOS 26.5.2. Result: the compile half is reachable,
the load+execute half is not.

### 42.1 What is reachable unentitled

- `ANECCompile` (ANECompiler.framework) emits a valid `.hwx` (magic
  `0xBEEFFACE`, TargetArchitecture `h13`) from an ordinary unentitled process.
  The earlier `_ANEInMemoryModel.compileWithQoS` `InvalidCompilationParam` was a
  wrong INPUT FORMAT, not an entitlement wall: the correct input is a 2-stage
  Espresso IR dump (`espresso_create_context`/`create_plan`/`plan_add_network`/
  `plan_build`/`dump_ir` → `net.plist`) fed to `ANECCompile`, and the practical
  source of that IR is a Core ML `.mlmodelc`. Signature/keys from tinygrad
  PR #240 and `freedomtan/coreml_to_ane_hwx`, still valid.

### 42.2 What is walled

Load and execute of a program on the ANE require a client an unentitled process
cannot get:

- `H11ANEIn` has two user-client flavors. **DirectPath** (open types 1/4) opens
  unentitled but is attach-only — no `ProgramCreate` (its sel 3 is
  `OutputSetEnqueue`); `DeviceOpen` returns `kIOReturnNotFound` without a
  broker-minted program handle. **UserClient** (open type 0) owns
  `ProgramCreate`, and its `IOServiceOpen` fails `kIOReturnUnsupported`
  (`0xe00002c7`) unentitled.
- Second wall even past UserClient open: a self-compiled, non-`aned`-signed
  `.hwx` fails load with `kIOReturnNotPermitted` (`0xe00002e2`) — signature /
  trustcache. The ANE only runs programs the trusted `aned` broker mints and
  loads.

The `aned` XPC broker path (`_ANEClient` load/evaluate) is the other route and
is itself entitlement-gated (fails silently returning nil without the private
`com.apple.aned` entitlements).

### 42.3 Consequence

Running our own compiled matmul on the ANE from an unentitled process is not
possible: compile is open, but nothing we can reach will load the result. The
Core ML `ANEGemm` path works only because Core ML IS the entitled broker — it
compiles, signs, loads, and dispatches through `aned` on our behalf. We cannot
replicate the load+execute half; we can only call it through Core ML.

The private procedure-bank half of the plan (low-latency dispatch bypassing
`MLModel.prediction`) is therefore dead for an unentitled process. The IOSurface
zero-copy half is still achievable, but only INSIDE the Core ML path: Core ML
accepts IOSurface-backed inputs/outputs (`CVPixelBuffer`), so the MLMultiArray
conversion barriers (§40) can be removed while keeping Core ML as the entitled
load/execute broker. That is a lower ceiling than omlx (it still pays the
`MLModel.prediction` per-call dispatch and cannot bank procedures), and its
value against an all-GPU MLP is unproven.

### 42.4 Verdict

Two of three gates cleared (compile reachable; dispatch surface partially
reachable), one gate hard-blocked (unentitled program load). The private-ANE
dispatch subsystem is not buildable on this machine without ANE entitlements we
cannot self-grant. The only ANE access available to us is Core ML, and the only
remaining optimization inside it is IOSurface-backed zero-copy I/O — a smaller
lever than the full private path, to be pursued only if the Core ml + IOSurface
number beats all-GPU, which §40's economics leave in doubt.

---

## §43. Correction: the private ANE path IS viable unentitled — oMLX proves it (2026-08-31)

§42 concluded "load+execute is entitlement-walled / route dead." That
conclusion was wrong. It generalized from a probe of the DIRECT `H11ANEIn`
IOKit UserClient, which is indeed walled — but that is not the door the working
implementations use.

oMLX (`github.com/jundot/omlx`, open source) implements exactly this — ANE/GPU
split prefill for Qwen 3.5/3.6/3.8, including the M5/NAX family — and runs
UNENTITLED: its `oMLX.entitlements` sets only `app-sandbox=false`, with no ANE
entitlement. Its mechanism (`omlx/custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm`):

- Private classes `_ANEInMemoryModelDescriptor` / `_ANEInMemoryModel` /
  `_ANERequest` / `_ANEIOSurfaceObject` — the §41 in-memory path, NOT direct
  IOKit and NOT the aned XPC `_ANEClient`.
- `compileWithQoS:options:error:` (QoS 21) then `loadWithQoS:options:error:`
  (QoS 21). These succeed unentitled.
- The program is MIL TEXT, e.g. `func procedure000<ios18>(tensor<fp16,[1,C,1,S]> x) { ... }`,
  passed via `initWithNetworkText:weights:optionsPlist:isMILModel:`. §42's Task-2
  `InvalidCompilationParam` came from feeding a BINARY MIL proto instead of this
  MIL text — a format bug, exactly as §42.1 suspected, now pinned.
- Dispatch: `_ANEIOSurfaceObject objectWithIOSurface:` wraps input/output
  IOSurfaces; `_ANERequest initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:`
  submits them. Zero-copy to the GPU via the private `MTLDevice
  newBufferWithIOSurface:` — the same IOSurface is an ANE buffer and an MTLBuffer.
- ANE instance pinning via execution-options keys `kANEFProcedureVariantHint`
  and `kANEFAneInstanceHint` (1..4). A persistent compile cache lives at
  `~/Library/Caches/omlx/ane/v1/<os-build>/<identifier>`.
- Offloaded weights are requantized to per-output-channel INT8 (approximate,
  not bit-exact). Reported gain: up to ~35% faster 16k-token prefill.

Corrected verdict: the private ANE offload (compile + load + execute + IOSurface
zero-copy + procedure banking) is achievable on this M5 unentitled, via the
`_ANEInMemoryModel` Obj-C path. It is not a from-scratch reverse-engineering
task — oMLX is a complete, current, open-source reference for our exact model
family. The remaining work is porting its proven pattern (or using oMLX
directly), not proving feasibility.
