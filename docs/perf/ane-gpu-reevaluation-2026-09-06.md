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
  excluded from the mean. Script: `tools/ane-probes/serve-sweep.sh`.
- **Agreement.** The greedy 96-token completion of every arm is diffed
  against the GPU control's: the first differing character and the identical
  prefix fraction. A divergence says the arms differ, not which is better.
  Script: `tools/ane-probes/agree.py`.
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
it, and `tools/ane-probes/serve-sweep.sh` now has one (`PROMPT_GAP`).

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
| ANE int4, fraction 0.5 | 115.7 hot, throttles late. 125.4 with a cool gap on a UI-active box (+16.9 percent over its 105.8 control, three clean prompts). Clean rerun 119.6 (+1.8 over a 117.5 control, +7.8 over the 110.9 repeat) | 12.82 | 5.923 | |
| ANE int4, fraction 0.625, cool gap on a UI-active box | 123.1 (+16.3 percent over 105.8, +11.0 over the 110.9 repeat) | 11.7 | pending | |
| ANE bank, per-row int4, all 64 layers, Core ML path | 63.3 (-49 percent) | 11.87 | 5.722 (S512 package, 64 bank functions used, no fallback) | 0 of 6 |
| ANE bank, 64-row int4 codebooks, Core ML path (placed on the GPU by Core ML) | 67.4 (-36 percent over 105.8, UI-active box) | 12.1 | 6.031 (S512 package, 64 bank functions used, no fallback) | |
| ANE bank, per-row int4, as two 32-layer parts that Core ML places on the ANE | 47.8 (-56 percent over a 108.0 control with Spotlight paused; the first control, under Spotlight, ran at 69.5) | 11.4 | 5.722 (same codebooks as the per-row row) | 0 of 6 |
| GPU plus ANE: int8 0.3125 with depth-2 drafting, cool gap | 140.4 (+15.5 percent over the depth-2 control's 121.5) | 22.40 (control 21.94) | | |
| under a 40 percent duty GPU load: control, then control repeat | 105.6, then 93.3 | 11.29, then 10.68 | | |
| under the same load: ANE int8, fraction 0.5 | 106.1 (0.5 percent above the first control, 13.7 above the repeat) | 10.12 | | |
| under the same load: ANE int4, fraction 0.3125 | 99.6 (5.6 percent below the first control, 6.8 above the repeat) | 9.27 | | |
| under the same load, later in the evening: control, then control repeat | 92.2, then 74.6 | 10.4, then 9.2 | | |
| under the same load: ANE int8, fraction 0.625 | 116.8 (26.7 percent above the first control, 56.5 above the repeat) | 10.2 | | |
| under the same load: ANE int8, fraction 0.75 | 96.2 (4.3 percent above the first control, 28.9 above the repeat) | 10.2 | | |

A caution on the "identical completions" column, found while filling it
for the later arms: the GPU control's own greedy completion is not one
completion. Across the evening's control runs there are three families,
each internally identical (gpu3 through gpu9 and gpu4-loaded in one; gpu,
gpu-loaded and gpu2-loaded in another; gpu3-loaded alone), and any two
families differ from the first or second token ("The text provided" against
"This is a fascinating"). That is a first-token near-tie resolved
differently by processes that run the same weights and the same kernels,
and the cause is not found. So an arm's 0 of 6 against a control from
another family says nothing about the lane; the perplexity column is the
fidelity instrument on this tower, and the agreement column is read only
against a control of the same family.

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
| GPU plus ANE: split int8 with depth-2 drafting | not run: the q8 tree's depth-2 serve hung at load for 18 minutes (the 2026-09-03 depth-2 MoE lost 17 percent on the bf16 tree in any case) | | | |
| under a 40 percent duty GPU load: control, then control repeat | 144.0, then 115.8 | 13.46, then 13.58 | | |
| under the same load: ANE split projections, int8 | 142.5 (1.1 percent below the first control, 23.0 above the repeat) | 13.74 | | |
| under the same load: ANE shared expert, int8 | 131.5 (8.7 percent below the first control, 13.6 above the repeat) | 13.85 | | |
| reference, mlx-serve release notes, M4 Max, 4-bit pack, short context | | 60 to 69 | | |
| reference, mlx-serve release notes, M4 Max, 32k prompt | 699 | | | |
| mlx-serve at HEAD on this box (2026-09-07, mixed 4/8 pack, kv8, serial with MTP and PLD off), short / 4k / 32k / 128k | 309 / 224 / 219 / 253 | 22.3 / 19.7 / 15.5 / 17.5 | | |
| mlx-serve at HEAD on this box, its MTP on, same rungs | 267 / 297 / 296 / 281 | 36.3 / 33.1 / 31.1 / 33.4 | | |
| ours on this box the same night (q8 tree, bf16 KV, depth 0), short / 4.8k / 38k / 150k | 83 / 258 / 149 / timed out | 13.3 / 14.4 / 10.5 / timed out | | |

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
from the source. mlx-serve ships each one behind a kill switch, and the
empirical half of the bead attributes them on this box by switching them
off one at a time in their own binary.

## Dense tower re-run with a cool gap per prompt

The first dense sweep gated the box only between arms, and the
fraction-0.5 arms throttled through their six prompts. This re-run waits 45
seconds before every prompt, control included, and measures the bank arm
beside the two forms.

| arm | prefill tok/s, prompts 2 to 6 | vs control | paired range | decode |
| --- | --- | --- | --- | --- |
| gpu3 (control) | 124.8 | | | 12.58 |
| bank, per-row int4, Core ML path, 64 layers | 63.3 | -49.3 percent | 0.38 to 0.57 | 11.87 |
| int4, fraction 0.3125 | 143.3 | +14.8 percent | 1.11 to 1.19 | 12.56 |
| int8, fraction 0.5 | 145.5 | +16.6 percent | 1.12 to 1.21 | 12.61 |
| gpu4 (control repeat) | 128.2 | +2.7 percent | 1.00 to 1.05 | 12.79 |

Two results. With the engines cool, the ANE lane is worth twice what the
hot sweep said (int4 at 0.3125: +7.1 hot, +14.8 cool) and the balance point
is above 0.3125 (int8 at 0.5: +5.7 hot with a collapse on the last prompt,
+16.6 cool on every prompt). The lane's gain tracks the ANE's temperature,
not the GPU's, and the control moved only 2.7 percent. Second, the Core ML
bank path loses half the prefill: it engaged on all 64 layers at bucket 1024
and its per-call cost on the big fused program is far above the direct
path's, the same fact the `r` table's dense rows showed. The bank's
per-row-codebook quality is not read from its perplexity arm after all (see
the correction under "The bank's quality"); its speed needs the direct path,
which cannot bank, or a per-row LUT in the in-memory program.

The reading the owner asked to be kept in view is that these lanes are
measured against an idle and cool GPU, which is the least common state of
an interactive machine. Under UI load the GPU path slows and the ANE path does not, so a
lane at parity here is the resilient primary path there. The loaded-box arms
that make that concrete are queued.

### The bank's quality

Correction, 22:35. The perplexity arm reported 5.565 with the bank
configured, and that number is the fp16 lane's, not the bank's. The harness
feeds 512-token windows, the bank holds only an S1024 package, and the lane
logged "bank unavailable for bucket 512" on every layer and built the fp16
direct program instead ("source=dequant4"). The same happened to the 64-row
bank's arm. queue14 generated a
bucket-512 package for each bank and reran the arm with the count of bank
functions used printed beside the number. The 64-row bank, with all 64
functions used and no fallback: perplexity 6.031, against 5.566 for the
control, 5.565 for the fp16 lane and 5.772 for the per-tensor int4 palette
on the direct path. One 16-entry codebook shared by 64 rows is worse than
one palette with a per-row scale, because the rows inside a block differ in
scale and the block's codebook covers the largest of them. The per-row
bank, same protocol: 5.722. One codebook per row lands between the
per-tensor palette (5.772) and int8 (5.563). It is the best int4 form
measured, and it is not lossless. The int4 fidelity ladder on this tower
now reads per-tensor palette 5.772, per-row codebook 5.722, 64-row codebook
6.031, int8 5.563, control 5.566. The speed question is separate:
the Core ML dispatch path is the wrong carrier for either bank (the next
section, and the 64-row plan below), so the per-row and grouped codebooks
matter only if the in-memory program accepts them, which is the next probe.

### Why the bank was slow

The bank package's own compute plan, read after the arms: every op of its
fused program, the three convs and the SiLU chain, is `preferred=gpu`. The
per-row palette (`lut` shaped `[5440, 1, 1, 1, 16, 1]`) falls off the ANE
where the per-64-row palette of the compute-plan probe stayed on it. The
bank arm ran its ANE prefix on the GPU through Core ML, beside MLX, and
lost half the prefill for it. The 49 percent loss says nothing about the
Core ML dispatch path or the per-row quantizer; it says the ANE compiler has
a limit on palette groups that a per-row codebook exceeds. The bank is
regenerated with one codebook per 64 rows, plan-checked before it runs, and
its perplexity is measured again against the per-row 5.565.


## The fused prefix at real shape on the direct path

`ANEFusedRealShapeProbeTests`: the dense tower's fused MLP prefix (hidden
5120, F=5440, the shipped fraction) through the production in-memory
dispatch, random weights, median of 12 calls including the input staging
and the output read.

| form | S=128 | S=256 | S=512 | S=1024 | ms per row at 1024 |
| --- | --- | --- | --- | --- | --- |
| fp16 | 3.25 ms | 10.03 | 17.98 | 33.89 | 0.033 |
| int8 | 2.33 | 5.77 | 10.20 | 18.92 | 0.018 |
| int4 | 2.59 | 4.55 | 7.88 | 14.96 | 0.015 |

At S=1024 the fp16 program runs 5.0 TF/s and the int4 program 11.4. The
compressed forms are 1.8 to 2.3 times faster than fp16 at the bucket the
lane uses. DRAM bandwidth is not the reason, since 167 MB costs 1.6 ms. The
engine re-streams its weights per spatial tile, and the per-row cost shows
it. For fp16 that cost rises from 0.025 ms at S=128 to 0.033 at S=1024. For
int4 it falls from 0.020 to 0.015. The palette's bandwidth win is absent at
one small conv and real on the fused program at real shape. That is why the
cool sweep put int4 at +14.8 percent and int8 at fraction 0.5 at +16.6
percent. fp16 reached 4.8 percent. The Core ML path's fp16 conv times
agree with these within 5 percent once scaled by output width. The
earlier suspicion that Core ML itself was slow is withdrawn. The bank was
slow because its program ran on the GPU.

The balance point follows. At int4 the whole MLP would cost the ANE about
48 ms at S=1024 against the GPU's about 45. The split that finishes both
legs together is near fraction 0.5, and 0.625 is the first fraction past
it. Both are queued cool.

One accounting gap stays open. With the fp16 leg at 34 ms per layer at
bucket 1024, the fp16 lane should lose to the GPU's 27 ms MLP. It measured
+4.8 percent. The serve's prefill may run the lane at a smaller bucket than
the probe assumes, although the program logs say bucket 1024 for the
613-token chunk. The discrepancy is recorded rather than explained.

### `r` for int4, rerun in the output-side form

The int4 palette with the per-channel scale on the conv output, Core ML
path, GPU at production quantization. `r` is GPU time over ANE time.

| shape | S=16 | S=128 | S=256 | S=512 | S=1024 |
| --- | --- | --- | --- | --- | --- |
| expert `gate_up` | 1.67 | 3.00 | 3.29 | 0.98 | 0.88 |
| expert `down` | 2.01 | 2.25 | 2.64 | 2.19 | 0.87 |
| MoE `in_proj_qkv` | 0.77 | 1.09 | 0.99 | 0.93 | 0.80 |
| dense MLP `gate` half | 0.66 | 1.15 | 0.99 | 1.07 | 1.03 |
| dense MLP `down` half | 0.57 | 0.64 | 0.57 | 0.54 | 0.99 |

The dense gate at S=1024 runs 13.0 TF/s on the ANE against 5.5 in fp16, and
the down 7.4 against 1.9. With palette weights the ANE matches the GPU's
production q4 matmul on both dense shapes at the bucket the lane uses,
where fp16 sat at 0.2 to 0.7. That is the same weight re-streaming effect
seen on the direct path, and it is the number behind the balance point at
fraction 0.5. The output-side scale costs nothing measurable. The void rows of the first
run had the scale on the weight and were 100 to 3000 times slower on the
same packages.

## The loaded box, dense tower

Bead `7yp`. A synthetic GPU client (`tools/ane-probes/gpu_load`, the
`fma_burn` kernel at 40 percent duty on a 50 ms period) runs for the whole
arm, control included, gaps included, so the GPU never cools inside an arm.
Six prompts, 45 seconds between prompts, one arm per process, the four arms
in the order shown. The mean is over prompts 2 to 6 because the ANE arms
build their programs on prompt 1. Cool numbers are the gapped re-run above.

| arm | prefill tok/s, prompts 2 to 6 | vs first control | paired range | last prompt | decode | cool prefill |
| --- | --- | --- | --- | --- | --- | --- |
| gpu-loaded (control) | 105.6 | | | 70.0 | 11.29 | 124.8 |
| int8, fraction 0.5 | 106.1 | +0.5 percent | 0.70 to 1.52 | 106.5 | 10.12 | 145.5 |
| int4, fraction 0.3125 | 99.6 | -5.6 percent | 0.75 to 1.57 | 109.8 | 9.27 | 143.3 |
| gpu2-loaded (control repeat) | 93.3 | -11.6 percent | 0.84 to 0.95 | 64.4 | 10.68 | 128.2 |

Three readings. First, the load costs the GPU control 16 to 27 percent of its
cool prefill and 10 to 15 percent of its decode, and the control moves 12
percent between its two runs, which is wider than the gap between any arm
and either control. The per-prompt pairs spread from 0.70 to 1.57 for the
same reason. A verdict finer than "at the control's level" needs more
repeats. Second, the two ANE arms sit at that level, 0.5 percent above and 5.6 below
on the first control and 13.7 and 6.8 percent above the repeat, and both hold the
last prompt at 106 to 110 tok/s where both controls fell to 64 to 70. That
last column is the resilience the owner's rule describes: the arm that
shares the GPU degrades through the arm, the arm with an ANE share does
not. Third, the cool gain of +15 to +17 percent shrinks to parity because the
fraction is tuned for a cool GPU. The ANE's share of the split runs at its
cool speed under load while the GPU's share slows, so the ANE finishes
first and waits. The higher fraction held up better (int8 at 0.5 over int4
at 0.3125), which points the same way. A load-aware fraction, raised when the
GPU is shared, is the follow-up arm. The decode column is GPU-only in every
arm, because the lane arms at 128 tokens and above, so its spread is the
box's thermal state and the arm order, not the lane. The load process was
stopped before it printed its busy share, so the achieved duty is the
requested 40 percent by construction, not measured.

### Higher fractions under the same load

The follow-up arm the first loaded table asked for: int8 at fractions 0.625
and 0.75 under the same 40 percent duty load, a control on each side, on
the UI-active box (23:00). Means over prompts 2 to 6.

| arm | prefill tok/s | vs gpu3-loaded | vs gpu4-loaded | paired range vs gpu3-loaded | last prompt | decode |
| --- | --- | --- | --- | --- | --- | --- |
| gpu3-loaded (control) | 92.2 | | | | 71.6 | 10.0 to 10.8 |
| int8, fraction 0.625 | 116.8 | +26.7 percent | +56.5 | 1.07 to 1.53 | 109.7 | 9.5 to 10.7 |
| int8, fraction 0.75 | 96.2 | +4.3 percent | +28.9 | 0.92 to 1.37 | 97.9 | 8.3 to 11.9 |
| gpu4-loaded (control repeat) | 74.6 | -19.0 percent | | 0.62 to 1.06 | 76.1 | 7.6 to 10.0 |

With the controls at 92 and 75, fraction 0.625 is above both on every
prompt, where 0.5 sat at parity under the same load earlier in the
evening. Fraction 0.75 falls back toward parity: past the balance point
the ANE's share becomes the long pole, since its time does not shrink
under load while the GPU's share does. The balance under this load is
near 0.625, against 0.3125 to 0.5 on a quiet cool box, which is the
load-aware fraction the first table asked for, now with a number.

One property of every serve arm in this record needs stating. A prompt of
600 to 665 tokens builds two programs per layer, bucket 512 and bucket
1024, so 128 programs against the 126-program limit per process
(`references/chip-support.md`). The last two layers' bucket-1024 programs
fail with `0x50004` and those layers run their GPU path at that bucket; the
int8 arms at 0.625 and 0.75 lost four. Every arm carries the same
shortfall, so the comparisons stand and the ANE numbers are a little
conservative. A single-bucket layout, or the procedure bank, removes it.

### The ANE-side time under the same load

The fused prefix program at real shape (hidden 5120, F=5440, gate, up, silu,
mul, down as one program, the real-shape probe), timed on the ANE alone,
cool, then with the same 40 percent duty GPU load running (its busy share
measured at 41 percent), then cool again. Milliseconds per call.

| form, rows | cool | loaded | cool repeat |
| --- | --- | --- | --- |
| int8, S=512 | 9.82 | 9.33 | 9.82 |
| int8, S=1024 | 18.46 | 18.32 | 18.44 |
| int4, S=512 | 7.83 | 7.48 | 7.87 |
| int4, S=1024 | 14.77 | 14.55 | 14.75 |

The ANE's time does not move when the GPU is loaded. The loaded column is
1 to 5 percent faster, inside the run-to-run spread of a warm chip. What
the loaded tables lose belongs to the GPU's side of the split and to the
host's crossings, so a larger ANE share holds up better under load, and a
fixed fraction tuned cool under-uses the ANE there.

### The loaded box, MoE tower

Same load, same protocol, the q8 tree, arms in the order shown. Cool
numbers are the fused-lane sweep above.

| arm | prefill tok/s, prompts 2 to 6 | vs first control | paired range | last prompt | decode | cool prefill |
| --- | --- | --- | --- | --- | --- | --- |
| gpu-loaded (control) | 144.0 | | | 135.1 | 13.46 | 274.2 |
| split projections, int8 | 142.5 | -1.1 percent | 0.80 to 1.16 | 107.8 | 13.74 | 271.8 |
| shared expert, int8 | 131.5 | -8.7 percent | 0.57 to 1.09 | 126.5 | 13.85 | 267.9 |
| gpu2-loaded (control repeat) | 115.8 | -19.6 percent | 0.51 to 1.27 | 171.4 | 13.58 | 270.7 |

Against the control repeat the split lane is 23.0 percent above and the
shared lane 13.6 above. Two readings. First, the MoE tower loses far more to
the load than the dense tower. It gives up 47 to 58 percent of its cool
prefill and 25 percent of its decode, where the dense tower gave up 16 to 27
and 10 to 15. The likely cause, which this run does not measure, is dispatch
count. The MoE prefill is hundreds of short kernels for the expert gathers
and the gated-delta state, and each one waits behind a load dispatch, where
the dense tower's big matmuls hold the GPU for longer per dispatch. Second, the lanes hold their cool standing
under load, at the control's level and inside a control envelope 20 percent
wide. Split int8 sits 0.9 percent below the control when cool and 1.1 below
the first loaded control. Shared int8 sits 2.3 below when cool and 8.7 below
loaded. The
per-prompt pairs spread 0.51 to 1.27 for the controls alone, so no MoE lane
can be ranked against the control under this load without more repeats,
and the last-prompt column carries no pattern here (the control repeat
ended on its best prompt). What the MoE lanes offload is a small share of
the layer (two projections, or the shared expert), so the GPU's share sets
the pace under load as it does when cool; the resilience the dense tower
showed needs a larger ANE share than these lanes carry.

## Cool dense arms on a UI-active box

The queue that carried the int4 balance-point arms and the 64-row bank ran
while the owner was using the machine (mail clients and a browser at 100
to 160 percent CPU, WindowServer at 20 to 40 percent), and Time Machine's
hourly backup copied a fresh 92 GB download during the fraction-0.5 arm.
The gates timed out without reaching the quiet state, and these arms are
paired against their own controls. Those controls sit at the
synthetic-load level (105.8 and 110.9 against 124.8 and 128.2 when quiet). Six prompts, 45 s gap,
means over prompts 2 to 6 with any prompt whose request wall exceeded 60 s
excluded as stalled.

| arm | prefill tok/s, clean prompts | vs gpu5 | vs gpu6 | paired range vs gpu5 | decode | stalled prompts |
| --- | --- | --- | --- | --- | --- | --- |
| gpu5 (control) | 105.8 | | | | 9.9 to 11.4 | none |
| int4, fraction 0.5 | 125.4 | +16.9 percent | +14.1 | 0.95 to 1.40 | 10.7 to 12.5 clean | cooking, geology, law: walls 242 to 312 s |
| int4, fraction 0.625 | 123.1 | +16.3 percent | +11.0 | 1.01 to 1.41 | 11.2 to 11.9 | none |
| bank, 64-row codebooks, Core ML path | 67.4 | -36.3 percent | -39.2 | 0.57 to 0.71 | 11.6 to 12.7 | none |
| gpu6 (control repeat) | 110.9 | +4.8 percent | | 1.01 to 1.16 | 10.8 to 12.0 | none |
| gpu7 (control, 23:23) | 117.5 | | | | 9.3 to 12.3 | none |
| int4, fraction 0.5, clean rerun (23:33) | 119.6 | +1.8 percent vs gpu7, +7.8 vs gpu6 | | 0.70 to 1.30 vs gpu7 | 9.3 to 12.1 | none |

Three readings. The int4 form at fractions 0.5 and 0.625 holds the full
+16 percent on a box whose GPU is shared with the owner's session, where
the fixed fractions 0.3125 and 0.5 under the synthetic load sat at parity.
This is the same direction as the loaded table: the more of the MLP the ANE
carries, the less the arm depends on the GPU's share. The clean rerun of
the fraction-0.5 arm, paired with a fresh control forty minutes later, came
in at +1.8 percent (+7.8 against the earlier repeat), with four of its six
decodes at 9.3 to 9.8 where the control decoded at 12.2, so the owner's
session was busier during that arm than during its control. The stalled
run's three clean prompts (+16.9) and the rerun (+1.8 to +7.8) bracket the
answer for 0.5 on a UI-active box. Fraction 0.625 (+11 to +16) is the safer
setting there, and under the synthetic load it is the clear one. The 64-row bank confirms its compute plan. Core ML
placed all 8 ops of its layer program on the GPU, and the arm ran at the
same 63 to 67 tok/s the per-row bank did. The stalls are the third reading.
Three requests in one arm took 240 to 310 s against 12 to 15 for the
others, with the model's own prefill or decode timer absorbing the stall
(0.3 tok/s decode, 2.1 tok/s prefill), which is a backup churning the page
cache under memory-mapped weights. The request wall column caught it; the
pack, the scratchpad and the Hub cache are excluded from Time Machine now
and the rule is in the skill.

## Grouped codebooks at real shape, one conv at a time

The 64-row bank's layer program is placed on the GPU with its convs
reported `supported=cpu/gpu`, and the question was whether Core ML rejects
the grouped palette itself at real shape. It does not. A sweep of
single-conv packages (`tools/ane-probes/gen_lut_group_probe.py`, iOS18,
one 16-entry codebook per block of rows) at the bank's own shapes:

| shape | rows per codebook (groups) | placement |
| --- | --- | --- |
| gate `[5440 x 5120]` | per tensor; 2720 (2); 1360 (4); 680 (8); 320 (17); 160 (34); 64 (85) | ANE, all seven |
| down `[5120 x 5440]` | per tensor; 2560 (2); 1280 (4); 640 (8); 320 (16); 64 (80) | ANE, all six |
| probe `[1280 x 2560]` | 64 (20) | ANE |

A conv with 80 to 85 codebooks at real shape is ANE-eligible on its own. What the bank's program adds is the fused SwiGLU (two convs, the
SiLU spelled as `x / (1 + exp(-x))` with a `real_div`, a `mul`, the down
conv) and the 64-function package. The next probe built one
layer as its own package and plan-checked it, then packages with more
functions:

| package | functions | size | placement of layer 0's 8 ops |
| --- | --- | --- | --- |
| one layer, 64-row codebooks, the SiLU chain | 1 | 42 MB | ANE, 8 of 8 |
| one layer, per-row codebooks, the SiLU chain | 1 | 42 MB | ANE, 8 of 8 |
| layers 0 to 1, 64-row | 2 | 85 MB | ANE, 8 of 8 |
| layers 0 to 3 | 4 | 159 MB | ANE, 8 of 8 |
| layers 0 to 7 | 8 | 319 MB | ANE, 8 of 8 |
| layers 0 to 15 | 16 | 638 MB | ANE, 8 of 8 |
| layers 0 to 31 | 32 | 1.2 GB | ANE, 8 of 8 |
| layers 0 to 63, the bank as built | 64 | 2.7 GB | GPU, 8 of 8 |

Neither the codebooks nor the fused program is the cause. A multifunction
package of 32 layers is placed on the ANE and the 64-layer package is not,
so the boundary sits between 32 and 64 functions, or between 1.2 and 2.7
GB of one package. The fix is to ship a bank as parts by layer range
(`S<bucket>.p<k>.mlpackage`, each with its metadata), which the loader
now does. The per-row bank generated as two 32-layer parts, plan-checked
and measured end to end between two controls, is queued (queue16), and
its result decides whether the Core ML path competes with the direct path
once its programs run on the ANE.

It does not. The per-row bank as two 32-layer parts, 63 of 64 bank
functions used at bucket 1024 (the 64th hit the 126-program limit beside
the 64 direct bucket-512 programs), one build failure, on the ANE by its
compute plan:

| arm | prefill tok/s, clean prompts | decode | note |
| --- | --- | --- | --- |
| gpu8 (control) | 69.5 | 7.9 to 9.4 | Spotlight's store at 130 percent CPU through the arm |
| bank parts, per-row, fraction 0.3125 | 47.8 | 10.4 to 12.4 | 0.64 to 0.78 of gpu8, 0.40 to 0.46 of gpu9; first prompt 4.0 tok/s while the parts loaded |
| gpu9 (control repeat, Spotlight workers paused) | 108.0 | 10.6 to 11.5 | |

With its programs on the ANE the bank runs at 44 to 56 percent below the
control, the same band as the GPU-placed banks (63 to 67 tok/s against 105
to 125). The device was never the cost. The Core ML dispatch of a
multifunction model, one `MLModel` per function with the runtime's own
input and output handling per call, is what the direct path's in-memory
programs and IOSurface handoff avoid. Verdict for the bank: dead as a
carrier at any placement. What survives it is the per-row codebook's
fidelity (5.722), which reaches the direct path only if the in-memory
compiler accepts a grouped LUT; that probe is the open item.

## The mlx-serve ladder, both runtimes on this box

Bead `qex`, the empirical half. mlx-serve at HEAD (`862bddf`, PR #363
merged, built from source with the pinned Zig nightly and the pinned MLX
submodules) on ddalcu's mixed 4/8-bit pack (8-bit dense, 4-bit experts, a
merged 4-bit n-gram table, 107 GB on disk), `--kv-quant 8`, against our
serve on the q8 tree (q8 dense, q4 experts, bf16 KV) at depth 0. The same
`llmprobe --bench-only --rungs 4k,32k,128k --runs 1` ladder on each, one
server at a time, minutes apart, on the UI-active box (our dense controls
decoded at 10 to 12 that evening against 12.6 quiet, and our MoE control at
18.3 quiet earlier the same day). mlx-serve's numbers are llmprobe's from
its report; ours are our server's own timers, because llmprobe reads no
throughput from our stream (its report carries only our TTFT). The two
tokenizers differ: the same rung text is 4,244 tokens to mlx-serve and
4,837 to us.

| runtime, arm | short prompt decode | 4k: prefill, decode | 32k: prefill, decode | 128k: prefill, decode | resident |
| --- | --- | --- | --- | --- | --- |
| mlx-serve, serial (MTP and PLD off) | 22.3 | 224 tok/s, 19.7 | 219, 15.5 | 253, 17.5 (TTFT 518 s) | 5.6 GB RSS after, pack mapped lazily |
| mlx-serve, its MTP (`--mtp`, adaptive depth up to 3) | 36.3 | 297, 33.1 | 296, 31.1 | 281, 33.4 (TTFT 467 s) | 31.6 GB RSS after |
| ours, depth 0 | 13.3 | 258 (4.8k tokens), 13.7 to 15.1 | 149 (38k tokens), 9.6 to 11.4 | failed: the worker timed out on a 150k-token prefill | worker 19 GB RSS plus the mapped trees |

Four readings. First, the 3.5x decode gap the bead opened with is two
factors and a different machine. On this box mlx-serve's serial decode is
1.5 to 1.7 times ours (22.3 against 13.3 short, 19.7 against 14.4 at 4k,
15.5 against 10.5 at 32k), and its MTP adds 1.6 to 1.9 on top (36.3, 33.1,
31.1, 33.4). The 60 to 69 tok/s of the release notes is the 4-bit pack on a
quiet box with prompt lookup on; their serial number on the mixed pack
here is 22. Second, prefill is at parity at 4k (224 against 258 on our
longer token count) and diverges with context: 219 against 149 at 32k, and
253 at 128k where ours does not finish. Their QSA arms and the split-K
kernels from PR #363 are what holds prefill flat to 128k. Third, they
reuse prefixes and we do not: their second 32k request prefilled 8k new
tokens over 24k cached, ours prefilled all 38k again (258 s twice). Fourth,
their MTP works on the MoE at every rung and ours hangs at load (bug
`9o0`), so the 1.6 to 1.9 they take from speculation is a lever we hold
and cannot pull today.

The 128k failure is ours to fix before any long-context claim: the serve
parent gives the worker a fixed request timeout
(`RuntimeWorkerOptions.defaultRequestTimeoutSeconds`) and a 150k-token
prefill at 100 to 150 tok/s runs past it, so the parent kills the worker
mid-prefill. Filed as a bug. What transfers, in the order the numbers rank it:

- the decode step (their 22 against our 13 on the same pack class is the
  step, not the pack)
- prefix reuse across requests
- the long-context prefill arms
- MTP on the MoE

Each is its own bead with an A/B against 13.3 and 258 on this box, or
against 18.3 and 274 quiet.
