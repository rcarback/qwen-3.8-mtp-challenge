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

## MoE tower, the fused lanes had never run on the q8 tree

The first MoE perplexity arms came back identical to the control to three
decimals. They were the control. The shared-expert lane and both split lanes
still carried the guard that refuses a `QuantizedLinear`, which was added on
2026-09-05 after the packed q8 weight crashed the fp16 builder, and only the
plain micro-batch lane had been given the dequantize path this morning. On
the q8 tree, the tree every MoE measurement now uses, the fused lanes
reached the GPU path every time and built zero programs. The three call sites now dequantize through MLX like the
plain lane does, and every MoE arm below reports its program build count.

The MoE control itself: teacher-forced perplexity 4.327 over 3072 positions,
131 seconds for the arm including the 78 GB load, swap peaking at 5 GB.

## Cleanup sweep (2026-09-06)

The disk reached 155 MB free during the perplexity arms. A read-only audit
(four area agents, then one skeptic per dead claim, 36 agents in all)
classified every large consumer. The owner then chose what to remove.

| removed | size | why |
| --- | --- | --- |
| Flash-Next bf16 source download | 388 GB | re-downloadable; no transform planned |
| `Qwen3.8-27B` bf16 base | 111 GB | lineage only; the bf16-prefix ANE arm is superseded by int8 |
| Flash-Next `weights/` model shards (bf16 dense) | 87 GB | superseded by the q8 tree; `ngram`, `ngram-int4` and the tokenizer stay, so the q8 tree's links resolve |
| n-gram int8 and nvfp4 tables | 83 GB | the encoding decision kept int4 |
| Qwen3.5 source checkpoints, DFlash2 drafter trees, stray head downloads | 17 GB | unreferenced or concluded |
| app, Xcode and SwiftPM caches, simulator devices | 14 GB | rebuilt on demand |
| session temp (compiled probe models, ANE staging directories, the pipeline cache, activation captures, a venv, logs) | 19 GB | regenerable |

Two things the sweep found that matter beyond disk. First, the fused MoE
lanes had never engaged on the q8 tree (the section above). Second, macOS
had been holding about 700 GB of purgeable snapshot space, which it released
on its own once the volume filled, so the free-space reading during the
incident understated what the box had. `tools/ane-probes/cleanup-run.sh`
now runs after every phase of the measurement queue.

### MoE shared-expert lane, perplexity

The shared expert as one fused ANE program per layer, weights re-quantized
from the q8 tree, 3072 teacher-forced positions:

| arm | perplexity | vs control |
| --- | --- | --- |
| gpu (control) | 4.327 | |
| shared expert, fp16 | 4.338 | +0.25 percent |
| shared expert, int8 | 4.336 | +0.2 percent |
| shared expert, int4 | 4.760 | +10.0 percent |

The fp16 and int8 forms sit at the fp16-compute floor. The int4 form is not
usable here: one codebook per tensor over a `[640 x 2560]` projection that
every token passes through costs 10 percent of perplexity, against 3.7
percent for a third of the dense tower's MLP. Any int4 on the shared expert
needs the per-row codebooks of the bank form.

### MoE split lanes, perplexity

The gated-delta `in_proj_qkv` and the attention `q_proj` with the first
0.3125 of their output channels on the ANE and the rest on the GPU, the two
partials concatenated, 3072 positions:

| arm | perplexity | vs control |
| --- | --- | --- |
| gpu (control) | 4.327 | |
| split, int8 | 4.315 | -0.3 percent, within noise |
| split, int4 | 4.551 | +5.2 percent |

int8 is lossless on both MoE lanes. The per-tensor int4 costs 5 percent on
the split projections and 10 percent on the shared expert. On this tower
the int4 question is settled against the per-tensor codebook; only the
per-row form remains a candidate.


### MoE tower speed, the fused lanes on the q8 tree

Six prompts, resident serve, draft depth 0, the box gated on temperature
and on background quiescence, every lane arm building its programs (48 to
49 shared-expert programs per bucket, 36 `in_proj` plus 13 `q_proj` splits
per bucket before the count wall). Wall time per request is now recorded
beside the model's timers; a first pass of this sweep was discarded because
a backup and the photo analyser had made each request take minutes.

| arm | prefill tok/s, prompts 2 to 6 | vs control | paired range | decode tok/s | wall per request |
| --- | --- | --- | --- | --- | --- |
| gpu (control) | 274.2 | | | 18.31 | 7.7 s |
| shared expert, fp16 | 267.0 | -2.6 percent | 0.969 to 0.978 | 18.47 | 8.2 s |
| shared expert, int8 | 267.9 | -2.3 percent | 0.972 to 0.984 | 18.34 | 8.3 s |
| shared expert, int4 | 265.9 | -3.0 percent | 0.963 to 0.975 | 18.25 | 8.7 s |
| split projections, int8 | 271.8 | -0.9 percent | 0.979 to 0.999 | 18.10 | 8.1 s |
| split projections, int4 | 270.1 | -1.5 percent | 0.978 to 0.992 | 18.11 | 8.4 s |
| gpu2 (control repeat) | 270.7 | -1.3 percent | 0.983 to 0.990 | 18.18 | 7.9 s |

The shared-expert lanes lose 2 to 3 percent in every form. The fp16 lane
measured the same loss on 2026-09-03, because the window holds one expert
beside ten and the two barriers per layer cost more than the overlap
returns. The split lanes at int8 sit inside the control's own drift, so on
this tower the compressed forms turn the dense-projection lane from a small
loss into parity, and no further. Decode is untouched everywhere.

**MoE verdict.** The fused dense lanes do not pay on this tower in any
weight form, because the ANE has no piece of work large enough between
graph boundaries. int8 is lossless in quality and free in speed; that is
the ceiling of this design, not a reason to ship it. Bead `wok` closes on
this table, and the open bead `e2l` with it.

## An external reference point, mlx-serve

While this record was being written, an X post reported `ddalcu/mlx-serve`
pull request 363 on an M5 Max: 8-bit dense with 4-bit experts, the KV cache
at 8 bits, a cold context ladder to 512K, about 1140 tokens per second of
prefill and 47.9 of generation there. The pull request's own table and the
project's release notes put numbers on our hardware class beside ours, on
the same weight configuration:

| runtime, M4 Max | prefill tok/s | serial decode tok/s | with speculation |
| --- | --- | --- | --- |
| this tree, q8 dense and q4 experts, 650-token prompts | 274 | 18.3 | 13.8 at depth 2 |
| mlx-serve v26.8.11, 4-bit pack, short context | | 60 | 78 |
| mlx-serve v26.9.1, 32k prompt | 699 | 69 | |
| mlx-serve v26.9.1, 256k prompt | 551 | | |

That is a 3.5x decode gap and a 2 to 5x prefill gap against an open
runtime on the same silicon, and it dwarfs every ANE lever measured this
week. The same release notes report a Neural Engine prefill offload
that makes a 16k-token prompt 19 to 35 percent faster, which is bead `60w`'s
premise measured by someone else. Bead `qex` reproduces their
protocol on this box and runs ours on the same ladder. It then reads their
source to attribute the gap across the decode step, the gated-delta kernel,
the MoE dispatch, the n-gram table path, the 8-bit KV cache, and the ANE
offload.


## Summary tables

Each table walks the same four stages, from the tree before this fork's
GPU work to GPU plus ANE together. Speed is tokens per
second on the six real prompts (603 to 665 tokens) unless a row says
otherwise; earlier rows come from the 2026-09-03 throughput matrix on a
732-token prompt and are marked. Fidelity is teacher-forced perplexity over
3072 positions and the count of greedy 96-token completions identical to
the GPU control. A cell that says pending is a queued arm, not an estimate.

### Dense tower, `Qwen3.8-27B`, q4 group-64

| stage | prefill tok/s | decode tok/s | perplexity | identical completions |
| --- | --- | --- | --- | --- |
| before GPU work, depth 0, 732-token prompt (2026-09-03) | 135.5 | 13.03 | | golden, exact |
| after GPU work, depth-2 drafting, 732-token prompt (2026-09-03) | 130.5 | 18.65 | | golden, exact |
| after GPU work, depth 0, control for the rows below | 123.8 | 12.37 | 5.566 | 6 of 6 (repeat) |
| ANE fp16, fraction 0.3125 | 129.8 (+4.8 percent) | 12.69 | 5.565 | 2 of 6 |
| ANE int8, fraction 0.3125 | 133.1 (+7.5 percent) | 12.59 | 5.563 | 3 of 6 |
| ANE int4, fraction 0.3125 | 132.5 (+7.1 percent) | 12.44 | 5.772 | 2 of 6 |
| ANE int8, fraction 0.5 | 130.8, throttles late | 12.40 | 5.571 | 2 of 6 |
| ANE int4, fraction 0.5 | 115.7, throttles late | 12.82 | 5.923 | |
| ANE bank, per-row int4, all 64 layers | pending | pending | pending | pending |
| GPU plus ANE: int8 0.3125 with depth-2 drafting | pending | pending | | |

The dense tower's per-operation split at prefill: the MLP is three of its
four dense projections by weight bytes, and the ANE prefix holds 0.3125 of
its intermediate channels, so the lane moves about a quarter of the MLP's
bytes and about a fifth of the layer's. The +7.5 percent is that share's
overlap with the GPU's remainder. Decode is untouched by every ANE row
because the lane arms only at 128 tokens and above.

### MoE tower, Qwen3.8-Flash-Next

| stage | prefill tok/s | decode tok/s | perplexity | identical completions |
| --- | --- | --- | --- | --- |
| before GPU work: bf16 dense, q4 experts, 732-token prompt (2026-09-03) | 307.3 | 16.50 | | |
| before GPU work with the fp16 ANE micro-batch lane (2026-09-03) | 274.3 (-11 percent) | 16.48 | | 3 of 4 |
| after GPU work: q8 dense, compiled blocks, n-gram readahead; depth 0, control | 274.2 | 18.31 | 4.327 | 6 of 6 (repeat) |
| after GPU work, depth-2 drafting, 732-token prompt (2026-09-03) | 357.4 | 13.77 | | |
| ANE split projections, int8 | 271.8 (-0.9 percent) | 18.10 | 4.315 | 1 of 6 |
| ANE split projections, int4 | 270.1 (-1.5 percent) | 18.11 | 4.551 | 0 of 6 |
| ANE shared expert, fp16 | 267.0 (-2.6 percent) | 18.47 | 4.338 | 0 of 6 |
| ANE shared expert, int8 | 267.9 (-2.3 percent) | 18.34 | 4.336 | 3 of 6 |
| ANE shared expert, int4 | 265.9 (-3.0 percent) | 18.25 | 4.760 | 0 of 6 |
| GPU plus ANE: split int8 with depth-2 drafting | pending | pending | | |
| reference, mlx-serve on an M4 Max: 4-bit pack, short context | | 60 to 69 | | |
| reference, mlx-serve on an M4 Max: 32k prompt | 699 | | | |

On the MoE the routing amplifies small activation differences, so the
completion-agreement column is a weaker instrument than on the dense tower:
the int8 arms are lossless in perplexity and still diverge on most prompts.

### Per-operation breakdown, where it is measured

| operation | GPU | ANE | note |
| --- | --- | --- | --- |
| MoE decode step, pipelined | 61.2 ms per token | not applicable | 2026-09-05, one row |
| MoE decode, full-attention layer, serialised | 2.40 ms x 12 layers | | the graph amortises about half of the serialised total |
| MoE decode, gated-delta layer, serialised | 2.52 ms x 36 layers | | 75 percent of layer time by count |
| MoE `in_proj_qkv` `[10240, 2560]` at S=128 | 0.684 ms, q4 group-64 | 0.632 ms, fp16, zero-copy output | 0.92x, the compute win the lanes cannot bank |
| dense fused MLP prefix, 0.3125 of the channels, bucket 1024 | | 167 MB fp16, 83 int8, 42 int4 per layer | compute-bound at this bucket, so the form buys about 2.5 points |
| expert `gate_up` `[1280, 2560]` and `down` `[2560, 640]` | production q4 group-32 gather | fp16 and int4 palette | the `r` table below, bead `xi7` |
| whole gated-delta layer at S=1 as one ANE program | 1.27 ms per layer on the GPU (pipelined) | pending | bead `i6v` |

## `r`, re-measured with zero-copy I/O and production-quantized GPU arms

Bead `xi7`. Median of 40 calls, ANE through Core ML with surface-backed
input and output, GPU through `quantizedMatmul` at each projection's
production form. `r` is GPU time over ANE time, so above 1 the ANE is faster.

| shape | S=16 | S=128 | S=256 | S=512 | S=1024 |
| --- | --- | --- | --- | --- | --- |
| expert `gate_up` `[1280, 2560]`, GPU q4 g32 | 1.59 | 2.90 | 2.18 | 1.14 | 1.08 |
| expert `down` `[2560, 640]`, GPU q4 g32 | 2.31 | 2.36 | 1.84 | 1.79 | 0.77 |
| MoE `in_proj_qkv` `[10240, 2560]`, GPU q8 g32 | 0.82 | 1.56 | 1.01 | 0.73 | 0.93 |
| dense MLP `gate` half `[8704, 5120]`, GPU q4 g64 | 0.94 | 0.99 | 0.41 | 0.47 | 0.70 |
| dense MLP `down` half `[5120, 8704]`, GPU q4 g64 | 0.31 | 0.61 | 0.16 | 0.17 | 0.23 |

ANE throughput behind those ratios, fp16 weights: the dense `gate` conv runs
10.9 TF/s at S=128 and 4.7 to 5.5 TF/s at S of 256 and above; the expert
`gate_up` runs 4.3 TF/s at S=128 and 7.3 at S=1024; `in_proj_qkv` 10.7 TF/s
at S=128 and 11.5 at S=256. Numerics: max relative error 0.0004 to 0.0005
against the fp32 product on every fp16 row.

Four readings.

1. **The expert partition is dead on the right denominator.** Against an
   isolated GPU matmul the ANE wins 1.6 to 2.9x at the widths a routed
   expert sees in prefill, so the old `r` was wrong in the other direction.
   Production, though, runs the batched gather: 70,000 token-expert
   pairs per layer in 22 ms, 0.31 microseconds per pair. The ANE's best
   expert rate, 7.3 TF/s at S=1024, is 1.5 microseconds per pair, and a
   grouped program at capacity 64 does not reach that. The GPU's batched
   gather beats a perfectly batched ANE expert lane by about five times.
   Bead `48v` closes on this arithmetic.
2. **The ANE likes S=128.** On the large dense shapes its efficiency halves
   above 128 rows, and 2D spatial layouts do not recover it: `[1,K,4,128]`
   equals `[1,K,1,512]` and `[1,K,64,8]` is four times worse. A dense-lane
   program at S=128 dispatched eight times could beat one at S=1024. That
   is measured through Core ML at the top level; the in-memory fused
   program is timed at real shape below before anything is redesigned.
3. **The down projection is the ANE's weak spot.** Its 8704-deep input runs
   at 1.9 TF/s where the gate runs 5.5 at the same S. In the fused prefix the
   down is a third of the FLOPs and more than half of the ANE leg's time, so
   a split that gives the ANE gate and up only, and the GPU every down, is
   the next dense-lane design to measure.
4. **Never scale a constexpr weight with a runtime op.** The int4 rows of this
   run built the palette weight as `mul(lut_to_dense(...), scale)`; Core ML
   rebuilt the dense weight on every call at 0.1 to 3.8 seconds per conv. The
   production form applies the per-channel scale to the conv output and ran
   at full speed in the sweep. The int4 `r` rows are void and are rerun in
   the output-side form.

One caveat on the dense rows: the sweep's own +7.5 percent at bucket 1024
implies an ANE leg well under the GPU's 27 ms MLP, while these Core ML
timings would put it near 50 ms. The two paths differ (Core ML `prediction`
against the direct in-memory dispatch), so the direct path is timed at the
real fused shape next, and the dense `r` rows are treated as the Core ML
path's numbers until then.

## The whole-layer decode program, probed

Bead `i6v` asked whether one ANE dispatch per whole dense layer could beat
the GPU's per-layer launch cost at decode. The probe is one gated-delta
layer of Qwen3.8-Flash-Next at S=1 as a single Core ML program with int4
palette weights: input norm, `in_proj_qkv`, `in_proj_z`, the `a` and `b`
gates, the depthwise conv, the 48-head gated-delta state update against a
constant state, the gated output norm, `out_proj`, the residual, the MLP
norm, and the shared expert, 59 ops in all.

| | ms per call |
| --- | --- |
| the whole layer on the ANE, Core ML path | 0.860 |
| the same program CPU-only | 3.169 |
| the GPU's gated-delta layer today, pipelined | about 1.27 |
| the GPU's gated-delta layer, serialised | 2.52 |
| mlx-serve's whole S=1 forward on an M4 Max, per layer | 0.33 |

The compute plan places all 59 ops on the ANE. At 0.86 ms the program is
three times the per-dispatch floor and it replaces only the dense part of
the layer. The routed experts stay on the GPU at about 0.35 ms, and the two
crossings cost about 0.2 ms each in the fused lanes. That sums to about 1.5
ms against the GPU's 1.27, before the recurrent state's fp16 round trip and
the full-attention layers' KV residency are solved. The design does not
pay, and bead `i6v` closes. The line that matters is the last one. An open
GPU runtime does the whole layer in 0.33 ms, a quarter of ours, so the
launch-cost problem this design tried to route around has a GPU answer.

## mlx-serve, the source half of bead `qex`

A workflow of 97 read-only agents read mlx-serve per subsystem, compared
each with our path, proposed levers, and put every lever through a skeptic.
Six levers survived, summing to about 1.7 ms of our 61 ms step; 74 were
refuted, most of them the ports of mlx-serve's own fused kernels, on the
ground that our compile pass measured only 2 percent for the elementwise
glue those kernels also absorb. The digest, with the ranked table, the
narrative and every refutation, is `docs/perf/mlx-serve-attribution-2026-09-06.md`.

The reading's mechanism claim stands whatever the lever count: their layer
is a handful of hand-written Metal kernels (hyper-connection read and
write, gated-delta prework and norm-gate, fused expert gate-up and
down-reduce) and ours is 60 to 100 MLX operations, so the 45 ms difference
is dispatch structure, not bytes. The skeptics could not size those kernels
from the source, and mlx-serve ships each one behind a kill switch, so the
empirical half of the bead attributes them on this box by switching them
off one at a time in their own binary.
