# ANE and GPU split compute, re-evaluated (2026-09-06)

This record re-examines every closed bead that rested on a claim about the
Apple Neural Engine (ANE), under the facts established this week: the ANE
consumes int8 and int4 weight storage, the output handoff is zero-copy, and
the multifunction package removes the 126-program wall. It covers both towers, the dense `Qwen3.8-27B`
and the `Qwen3.8-Flash-Next` mixture of experts (MoE). Every number is from this machine (M4 Max,
128 GiB, macOS 26.5.2) unless a source is named.

## The facts that changed, and the one that did not

| fact | status | evidence |
| --- | --- | --- |
| The ANE runs int8 per-channel weights | confirmed | `ANEQuantizedWeightProbeTests`, `ANEComputePlanProbeTests` |
| The ANE runs int4 palette weights, per-tensor and grouped | confirmed | same, plus the compute plan for the grouped LUT |
| The ANE runs the MLX group-affine forms (q4 g32, q4 g64, q8 g32) | **refuted** | Core ML places the conv on the GPU for every blockwise variant |
| Output from the ANE is zero-copy | confirmed | `outputBackings` into an IOSurface, 0.005 ms |
| The 126-program wall is per program, not per function | confirmed | multifunction `.mlpackage`, every function dispatches |
| ANE and GPU bandwidth add | not changed | GPU alone 189 to 212 GB/s, both engines 217 to 220 |
| The per-dispatch floor at one row | not changed | about 0.30 ms per program, whatever the weight form |

The refuted row matters most. The blockwise `constexpr_blockwise_shift_scale`
op compiles through Core ML, and the conv that consumes it is then
`supported=cpu/gpu` in the compute plan, for int8 group 32, int4 group 32,
int4 group 64, with an offset, and without one. The ANE can never hold the
GPU's quantized tensor byte for byte. An ANE-resident weight is always a
second, re-quantized copy. That fixes the memory cost of every split design
and it means every ANE arm carries its own quantization error on top of the
GPU's.

## The weight forms, measured

Core ML compute plan and timing, one `[1280 x 2560]` projection at S=256,
surface-backed input and output, median of 30 predictions. Error is against
the fp32 product of the form's own effective weight, so it measures the
dispatch, not the quantizer.

| form | op | ANE | ms | max rel error |
| --- | --- | --- | --- | --- |
| fp16 | `const` | yes | 0.293 | 0.0003 |
| int8 per-channel affine | `constexpr_affine_dequantize` | yes | 0.271 | 0.0005 |
| int8 blockwise g32 | `constexpr_blockwise_shift_scale` | no, GPU | 0.370 | |
| int4 blockwise g32 | same | no, GPU | 0.376 | |
| int4 blockwise g64 | same | no, GPU | 0.358 | |
| int4 blockwise g32, no offset | same | no, GPU | 0.359 | |
| int4 palette, one LUT | `constexpr_lut_to_dense` | yes | 0.268 | 0.0003 |
| int4 palette, one LUT per 64 rows | same, grouped | yes | 0.267 | 0.0004 |
| int4 palette, int8 LUT plus per-channel blockwise scale | both ops | yes | 0.774 | 0.0004 |

Two readings. At this shape the palette reads a quarter of the bytes and
gains 9 percent, so the conv is compute-bound at S=256 and the palette's
bandwidth win is not the lever here. The chained per-channel blockwise scale
stays on the ANE and costs 2.9x. A per-channel scale belongs on the conv
output as a `mul`, which is free.

The same plan run over one program of miscellaneous ops at `[1, 2560, 1,
256]` names the ANE as the preferred device for every op probed: silu, gelu,
sigmoid, tanh, exp, erf, clip, sqrt, rsqrt, abs, softmax, layer_norm,
reduce_mean, reduce_max, mul, add, reshape, transpose, concat,
slice_by_index, matmul, linear, and scaled_dot_product_attention. Eligibility
is not precision. The silu table error stands.

## The fused SwiGLU-down program in three forms

The dense tower's ANE lane runs the whole multilayer perceptron (MLP) prefix
as one program. `ANEFusedFormProbeTests` builds each form through the
production in-memory path with random weights at hidden 1024, inner 512,
S=256, and scores it against the fp32 SwiGLU of the original weights.

| form | build ms | max rel error | mean rel error | ms per call |
| --- | --- | --- | --- | --- |
| fp16 | 108 | 0.0012 | 0.0013 | 0.594 |
| int8 per-channel | 219 | 0.0138 | 0.0135 | 0.914 |
| int4 palette, per-row root-mean-square scale, fitted codebook | 419 | 0.2377 | 0.1691 | 0.861 |

The int8 error is the per-channel quantizer compounded through three
projections. The int4 error is what a 16-level codebook does to Gaussian
weights, and the GPU's own group-64 affine quantizer is the reference point:

| quantizer of a Gaussian weight | error per weight |
| --- | --- |
| 16-level fitted codebook (the ANE int4 form) | about 10 percent |
| the same, compounded through gate, up and down | 17 percent measured |
| q4 group-64 affine (the GPU's form) | about 9 percent |

An int4 ANE arm is a second quantization of a tensor the GPU already
quantized once, at about the same distance from bf16 but in a different
direction. Whether that changes tokens is the end-to-end question below.
Both compressed forms are slower than fp16 per call at this small shape,
because the in-memory path does not fold the dequantize away.

## Closed beads, re-examined

Every closed bead whose verdict rested on an ANE claim, and what the new
facts do to it. Beads about the MTP head, the n-gram table's encoding,
expert merging, KV quantization, and the launch-cost family are listed once
below as unaffected, with the one cross-reference each carries.

| bead | prior verdict | what changes | verdict now |
| --- | --- | --- | --- |
| `1ar`, ANE vs GPU at one row | ANE loses 5 to 10x in-graph, with a 0.30 ms floor per program | Nothing. A weight form cannot lower a dispatch floor, and at one row the byte term is already the smaller term. | Closed, unchanged. The one design the arithmetic leaves open is the whole-dense-layer program, filed as bead `i6v` at P4. |
| `bhz`, expert split at decode geometry | Both engines pay dispatch overhead on a 3.3 MFLOP problem | Nothing, and the compute plan adds that a decode-time ANE expert share would be a second int4 copy with its own error. | Closed, unchanged. |
| `yvk`, the `r` measurement | `r` 0.16 to 0.49 at S of 128 and above, dense fp16 both sides | The ANE arm paid a Core ML input and output copy per call (6.5 MB at S=1024) and held fp16 weights. The GPU arm was dense fp16 while production is 4-bit. | Superseded by bead `xi7`, which re-measures with zero-copy I/O, int4 and fp16 ANE forms, and production-quantized GPU arms. |
| `5ae`, int4 ANE expert partition | Not worth building, on the `r` ceiling and the fragmentation losses and the program count | The count wall is gone. The ceiling rests on the superseded `r`. The fragmentation losses bound the join, not the return. The memory cost is now known from the compute plan. It is a second int4 palette copy at 0.5 byte per parameter, 12 GB at a 0.2 share. | Reopened as a decision, bead `48v`, blocked on `xi7`. |
| `e2l`, dense-lane bank re-measure (open) | Pending | The zero-copy measurement answered the primitive: 0.632 ms ANE against 0.684 ms quantized GPU at S=128, an 8 percent compute win inside a window that holds only the projection. The int8 form now builds from the q8 tree. | Measured end to end by bead `wok`, and closed with it. |
| `v0c.13`, micro-batched pipelined prefill | Minus 25 percent at 700 tokens, the restructuring cost, not the crossing | Nothing at 700 tokens. At 4k to 8k tokens a 1024-token micro-batch has pipeline depth to recover the restructuring cost. | Closed, unchanged. Long-context variant filed as bead `60w` at P3. |
| `2i7`, N-split and CPU column assist | Closed by the compute-engine map | Nothing. | Closed, unchanged. |
| `p84`, multifunction to zero-copy bridge | Solved | Nothing. | Closed. |
| the dense tower's fp16 split (doc section, no bead) | Fraction 0.3125 fp16, plus 5.7 percent net; 10 of 64 layers refused at two buckets | int8 halves and int4 quarters the ANE program bytes, and the Core ML path banks all 64 layers with per-row codebooks. | Measured by bead `avk`. The bank is bead `8xv`. |
| `3y8`, `mw7`, `voh`, the q8 dense tree | q8 everywhere; mixed policies cost perplexity | The ANE int8 per-channel form re-quantizes q8 g32 once more, at about 0.8 percent per projection. | Closed, unchanged. The cost is measured in bead `wok`. |
| `jrv`, `fio`, `2x7`, expert group-64 | Do not adopt; 9 percent of decisions change | Nothing. An ANE expert copy is a separate representation whatever the GPU's group is. | Closed, unchanged. |
| `o0w`, `abl`, `t79`, `d7f`, `wjl`, `2o8`, `6ct`, `8ol`, `3pq`, `4x6`, the n-gram table | int4 adopted; readahead two layers early | The ANE has no gather, so it plays no part. The int4 table's 25 GB page footprint is what leaves room for an ANE expert copy beside the 78 GB q8 tree. | Closed, unchanged. The dependency is recorded in bead `48v`. |
| `0ja`, `ccj`, `muj`, `0e7`, `fnd`, `6us`, `8fg`, `583`, `msy`, `qtp`, `a8w`, the MoE kernel and pool family | Various | Nothing. All are GPU-side or quality questions. | Closed, unchanged. |
| `738`, `2u9`, `2t6`, `xgy`, `lk9`, `ptc`, `afu`, `bog`, `4cr`, `azl`, the launch-cost family | 33 ms of a 57 ms step is launches; whole-step compile is blocked by the host n-gram gather | The whole-dense-layer ANE program is the one launch consolidation that does not need the compile. | Closed, unchanged. Bead `i6v` cross-references it. |
| `njq`, KV int4 | 0.2 percent of the step | Nothing. | Closed, unchanged. |
| `do6`, `6qv`, `xt9`, `506`, `1fc`, `367`, `cul`, `cmf`, `a3f`, the speculative family | Various | Nothing. The head runs at one row, where the ANE loses. | Closed, unchanged. |

## How the end-to-end arms are measured

Both towers use the same three instruments, one model-holding process at a
time, GPU control first and last so drift is visible.

- **Speed.** One resident `serve` per arm answers the six real prompts
  (biology, cooking, geology, law, logistics, music; 603 to 665 tokens each)
  once, cold, at draft depth 0. The row records prompt tokens, seed prefill
  seconds, prefill tokens per second and decode tokens per second. The first
  prompt of an ANE arm pays that bucket's program compile and is reported but
  excluded from the mean. Script: `serve-sweep.sh`.
- **Agreement.** The greedy 96-token completion of every arm is diffed
  against the GPU control's: the first differing character and the identical
  prefix fraction. A divergence says the arms differ, not which is better.
  Script: `agree.py`.
- **Perplexity.** `DensePerplexityTests` scores the mean negative
  log-likelihood of the true next token over 512 teacher-forced positions per
  prompt, one process per arm, with the ANE lane engaged by the 512-token
  forward. This is the quality number; an ANE form that raises it is worse
  whatever the agreement column says.
- **Per-layer error.** One `MLX_ANE_VERIFY=1` run per form logs the max-abs
  error of each offloaded MLP against the all-GPU expression it replaced.

## Dense tower speed

Prefill tokens per second, six prompts, one resident serve per arm, draft
depth 0. The first prompt of every ANE arm pays the bucket-1024 program
compile (18 to 26 tokens per second) and is excluded from the mean. The
control repeated at the end of the sweep (`gpu2`) drifted 0.8 percent below
the first control, which bounds run-to-run noise.

| arm | cooking | geology | law | logistics | music | mean | vs control | decode tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpu (control) | 122.1 | 135.5 | 120.4 | 119.5 | 121.4 | 123.8 | | 12.37 |
| fp16, fraction 0.3125 | 127.9 | 140.4 | 128.9 | 126.1 | 125.5 | 129.8 | +4.8 percent | 12.69 |
| int8, fraction 0.3125 | 131.0 | 143.6 | 131.5 | 130.0 | 129.3 | 133.1 | +7.5 percent | 12.59 |
| int4, fraction 0.3125 | 131.0 | 143.6 | 127.4 | 130.4 | 130.2 | 132.5 | +7.1 percent | 12.44 |
| int8, fraction 0.5 | 136.0 | 144.2 | 137.2 | 132.4 | 104.2 | 130.8 | +5.7 percent | 12.40 |
| int4, fraction 0.5 | 142.3 | 137.8 | 107.5 | 111.1 | 79.7 | 115.7 | -6.5 percent | 12.82 |
| gpu2 (control repeat) | 119.9 | 132.4 | 119.0 | 121.0 | 121.8 | 122.8 | -0.8 percent | 12.23 |

Paired per prompt against the control, every fraction-0.3125 arm wins five
of five prompts:

| arm | paired ratio range |
| --- | --- |
| fp16 | 1.034 to 1.071 |
| int8 | 1.060 to 1.092 |
| int4 | 1.058 to 1.091 |

The compressed forms add about 2.5 points over fp16. That is the weight
bytes of the ANE leg. At bucket 1024 the fp16 prefix is 167 MB per layer,
while the conv is mostly compute-bound, so a smaller weight buys a little
and not a lot. Decode is untouched, as expected of a prefill-only lane.

The fraction-0.5 arms tell a different story. Both open faster than any
0.3125 arm on the cool early prompts (int4 at 1.165, int8 at 1.114 and
1.140) and then fall prompt by prompt, int4 to 0.657 on the sixth. The
cool-down gate runs only between arms, so an arm that puts more work on the
ANE heats it through the six prompts. Read this as thermal throttling under
the heavier ANE share, and as evidence that the balance point sits above
0.3125 when the engine is cool. A per-prompt cool gap is needed to measure
it, and `serve-sweep.sh` now has one (`PROMPT_GAP`).

Programs: every ANE arm built 117 programs and refused 11 at the second
bucket, the 126-count wall; the Core ML bank (bead `8xv`) is what lifts it.

## Dense tower quality

Two of the three quality instruments are in. The per-layer verify run logs
the max-abs error of each offloaded MLP against the all-GPU expression it
replaced. The worst layer is layer 9 in every form.

| form | worst-layer max-abs error | reference max | relative |
| --- | --- | --- | --- |
| fp16 | 0.031 | 9.2 | 0.3 percent |
| int8 | 0.063 | 9.2 | 0.7 percent |
| int4 | 0.078 | 2.7 (bucket 512) | 2.9 percent |

Greedy 96-token completions against the GPU control, six prompts:

| arm | identical | first divergence, characters, on the others |
| --- | --- | --- |
| fp16, 0.3125 | 2 of 6 | 2, 154, 235, 387 |
| int8, 0.3125 | 3 of 6 | 2, 2, 337 |
| int4, 0.3125 | 2 of 6 | 2, 74, 229, 2 |
| int8, 0.5 | 2 of 6 | 2, 74, 235, 399 |
| gpu2 (control repeat) | 6 of 6 | |

A divergence at character 2 is a first-token flip on a near tie, and every
diverging completion stays on topic and well formed. The control repeat
reproduces itself exactly, so the divergences belong to the ANE arms and not
to run-to-run noise. Which arm is closer to the model is the perplexity
column, reported below as those arms finish.

### Perplexity

Teacher-forced next-token perplexity over 3072 positions (six prompts, 512
each), one process per arm, every arm with all 64 programs built. An earlier
pass of these arms ran with the disk full, so most programs failed to stage
and those layers silently ran on the GPU; those numbers were discarded, and
the harness now reports its program build counts beside every result.

| arm | perplexity | vs control |
| --- | --- | --- |
| gpu (control) | 5.566 | |
| fp16, fraction 0.3125 | 5.565 | 0.0 percent |
| int8, fraction 0.3125 | 5.563 | 0.0 percent |
| int4, fraction 0.3125 | 5.772 | +3.7 percent |
| int8, fraction 0.5 | 5.571 | +0.1 percent |
| int4, fraction 0.5 | 5.923 | +6.4 percent |

int8 is lossless at this resolution, at a third of the MLP and at half of
it. The int4 form with one codebook per tensor costs 3.7 percent of
perplexity at a third and 6.4 percent at half, in proportion to its share,
which is the second quantization compounding with the GPU's first. The
per-row codebooks of the Core ML bank are the int4 that has a chance of
closing that gap, and they are measured below when the bank arms run.

**Dense tower verdict so far.** int8 at fraction 0.3125 is the best point
measured: prefill 7.5 percent over the GPU control on every prompt, decode
untouched, perplexity unchanged, and completions that differ from the GPU
only at near-tie tokens. It replaces the fp16 lane, which paid 2.5 points of
speed for the same quality.
