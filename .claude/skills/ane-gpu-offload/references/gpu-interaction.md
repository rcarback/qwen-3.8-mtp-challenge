# GPU / ANE interaction: when to split, and the numbers behind it

The question is never "can the ANE run this" but "does moving it beat leaving
it on the GPU, after the handoff and after the two engines contend for the
same DRAM." This file is the measured decision table.

## The engines, by bandwidth

| engine | DRAM read, measured here (M4 Max) | independent figure |
| --- | --- | --- |
| GPU, streaming GEMV | 330 to 440 GB/s isolated; 160 to 212 in the concurrency harness | chip peak 546 GB/s |
| ANE | 107 to 110 GB/s | 85 GB/s M1 roofline (paper); 78 GB/s M3 Max fit (field guide) |
| CPU, four threads | 121 to 129 GB/s | P-cluster capped at 224 by the fabric, 243 with E-cores, on M1 Max (AnandTech) |

The ANE is the narrowest engine. Anything bandwidth-bound belongs on the
GPU; the architecture paper says so directly for autoregressive decode.

## Concurrency is close to zero-sum

Four-second windows, GB/s achieved by each engine (`MemoryHeadroomTests`,
two samples):

| engines | GPU | ANE | CPU | sum |
| --- | --- | --- | --- | --- |
| GPU alone | 189 to 212 | | | 189 to 212 |
| ANE alone | | 107 to 110 | | 110 |
| GPU + ANE | 153 | 64 to 66 | | 217 to 220 |
| GPU + CPU | 122 to 138 | | 25 to 27 | 149 to 163 |
| all three | 90 to 108 | 20 to 31 | 20 to 25 | 146 to 149 |

Two engines never beat the GPU alone in this harness. One caveat the skill
keeps: the GPU chain in that harness reads well below the isolated 330 to 440,
so it is partly host-bound and the ANE's staging thread may be stealing
encoder time rather than DRAM. It shows the two engines do not add; it does
not prove the fabric has no headroom. Even with perfect sharing the ANE adds
at most about 110 GB/s on work that is independent of the GPU's, and at
decode there is no such work.

## Decode: the ANE loses, do not revisit without new silicon

| shape (m x k x n) | ANE ms | GPU ms, isolated | ANE / GPU |
| --- | --- | --- | --- |
| 1 x 2560 x 2560 | 0.41 | 0.28 | 1.5 |
| 1 x 2560 x 10240 | 0.80 | 0.33 | 2.4 |
| 1 x 2560 x 248320 (`lm_head`) | 10.9 | 1.65 | 6.6 |
| 8 x 2560 x 10240 | 1.05 | 0.62 | 1.7 |

The ANE floor is 0.30 ms per program including staging; in-graph the GPU is
5 to 10x faster because its launches pipeline and the ANE's do not. At S=1
the split projection is 7.5 to 8.3 ms against a 0.56 to 1.1 ms GPU MLP; times
48 layers that is about 450 ms per token against an 81 ms GPU step. The
routed experts, which dominate decode bytes, cannot run on the ANE at all
(no dynamic gather), and computing all 512 and masking to 10 would stream
about 60 GB per token. Dead, every way.

## Prefill: the ANE compute wins, the handoff decides

At S=128 the balance tips. `in_proj_qkv [10240x2560]`, warm, median of 50:

| arm | ms |
| --- | --- |
| ANE compute (multifunction `.mlpackage`, surface-backed I/O) | 0.632 |
| GPU 4-bit group-64 (production) | 0.684 |
| GPU dense bf16 | 0.839 |

0.92x on compute, with a zero-copy output (0.005 ms) and an input handoff
whose measured 0.22 ms is a standalone launch floor that a fused forward does
not pay separately (see `zero-copy.md`). The earlier single A/B on the direct
lane read about 1.13 to 1.30x on the serial prefill leg, on a bf16 tree; the
split variants that fragment the fused graph lost (shared expert on ANE -2.8
and -3.3 percent, channel split -3.6, grouped experts -6.8 and -7.4,
micro-batched lane -25.2). The lesson from those six: a split that adds a
barrier inside the fused MLP loses to the barrier, whatever the ANE compute
saves. Offload whole projections between graph boundaries, never a slice
inside one.

`MLX_ANE_MIN_SEQ=128` is the measured balance point: 0.94 to 1.05x at S=128,
losing below it.

## `r` re-measured (2026-09-06, zero-copy, production GPU arms)

GPU time over ANE time, fp16 ANE weights, by S=16/128/256/512/1024: expert
`gate_up` 1.59/2.90/2.18/1.14/1.08; expert `down` 2.31/2.36/1.84/1.79/0.77;
MoE `in_proj_qkv` 0.82/1.56/1.01/0.73/0.93; dense MLP `gate` half
0.94/0.99/0.41/0.47/0.70; dense MLP `down` half 0.31/0.61/0.16/0.17/0.23.
Three things this settles. The ANE beats an ISOLATED GPU matmul on expert
shapes at prefill widths, but production runs the batched gather at 0.31
microseconds per token-expert pair against 1.5 for the ANE's best rate, so
the expert partition is dead on the right denominator. The ANE's efficiency
on large dense shapes peaks at S=128 (10.9 TF/s) and halves above it, and 2D
spatial layouts do not recover it, so tile the ANE leg at 128 rows. The
down projection with its 8704-deep input is the ANE's weak spot (1.9 TF/s);
give the ANE gate and up and keep every down on the GPU.

## The bounded ceiling, stated honestly

The dense projections the ANE can take are a minority of prefill FLOPs; the
routed experts dominate and stay on the GPU. A perfect ANE/GPU split at
prefill widths has a measured ceiling of 1.16x to 1.49x
(`r` measurement, bead 5ae), against a dense bf16 GPU arm that is itself
slower than the quantized production matmul. So the whole ANE prefill lever is
worth tens of percent on prefill at best, and prefill is the smaller half of
a long-context turn once the checkpoint cache is warm.

## The ANE under load: a close result is a win

The tables above are measured on a cool, quiet box with the GPU otherwise
idle. That is one of three conditions the box actually runs in, and the
least common one for an interactive machine. Under UI load, another GPU
client, or a throttled GPU, the GPU-only path slows and the ANE path does
not, because the engines are independent. So an ANE arm that is within
about 10 percent, or within 10 ms per call, of the GPU when cool is a
competitive primary path under load, and the preferred one when the GPU is
shared. Measure every candidate lane in three conditions: cool and idle
(the tables here), loaded (a synthetic GPU load standing in for the UI, the
`loaded` arms), and hot (no cool gap between prompts). The 2026-09-06
dense re-run is the pattern: int8 at fraction 0.5 read +5.7 percent hot and
+16.6 percent with a 45-second cool gap per prompt, and every ANE arm's
paired ratio moved with the engine's temperature, not the GPU's. Pipelining
ANE work under the GPU's is better still; but a lane that only matches the
GPU when cool is not a null result, it is the resilient path.

## The decision rule

1. Is the op at S >= 128 and a whole projection between graph boundaries?
   If not, stay on the GPU.
2. Does the weight fit the program budget as a per-op-type, per-dtype bank?
   If not, bank it (`program-limit.md`) before measuring.
3. Is the I/O zero-copy (`zero-copy.md`)? If not, the measurement is an
   artifact.
4. Measure the whole forward, counterbalanced, cool box, one process, both
   arms on the same weight tree. Do not compare an ANE arm on a bf16 tree to a
   GPU arm on a q4 tree; that measures the trees. Cool is not enough: the box
   must also be quiet. On 2026-09-06 a Time Machine pass and the Photos
   media analyser (`mediaanalysisd`, 170 percent CPU for ninety minutes)
   churned the page cache the 95 GB n-gram table lives in, and every prompt
   of a serve arm took minutes of wall time outside the model's own timers
   while those timers reported normal rates. Gate on `tmutil status` and on
   the analysers' CPU (`quiet_gate` in the close-out environment), pause the
   analysers with SIGSTOP for the timed phases and resume them after, and
   record the wall time of every request beside the model's timers so hidden
   time is visible.
5. Keep the GPU baseline honest: the production quantized matmul, not dense
   bf16.
6. Judge a close result under load, not only when cool: a lane within 10
   percent of the GPU on a cool box is the primary path when the GPU is
   shared with the UI or throttled, and the loaded arm is the one that
   decides it.

## What would change the verdict

- A silicon generation with a wider ANE path or a documented int8 compute
  path on the engine (A17 Pro / M4 class advertises int8 weight-and-activation
  compute; unmeasured here for this workload). See `chip-support.md`.
- A workload where the ANE's work is independent of the GPU's for long
  stretches, so the fixed dispatch floor amortizes and the two do not contend:
  batch prefill of many prompts, or a dense model without a routed MoE.
- True zero-copy activations produced into surface-backed buffers, removing
  the last input handoff.
