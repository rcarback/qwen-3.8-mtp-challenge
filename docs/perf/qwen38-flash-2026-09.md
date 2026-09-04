# Qwen3.8-Flash-Next port record, September 2026

Local perf fork only. This page records what the port verified and measured.
Every number below names the command that produced it.

## Identity

| Item | Value |
|---|---|
| Source | `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Source size | 131 safetensors shards, 360,023,349,944 bytes with tokenizer files (`tools/qwen38-flash/manifest.json`) |
| Runtime model type | `qwen4_exp_text` (flattened `text_config`), registered next to `qwen4_exp` |
| Runtime tree | dense tensors bf16 byte copies, routed experts affine 4-bit group 32, n-gram table bf16 in `ngram/` (memory-mapped) |
| Reference | mlx-lm PR 1788 (`tools/qwen38-flash/qwen4_exp.py`), llama.cpp PR 27742 and 27836 |

## Status

Now:

- Model code, transform, generate verb, serve wiring and the native MTP head
  are in the tree and run on the real checkpoint. 32 unit tests pass, the
  mlx-lm reference parity case included
  (`MLXFAST_RUN_QWEN4EXP_PY_PARITY=1 swift test --force-resolved-versions --filter Qwen4Exp`).
- The download and the transform are done. The runtime tree matches the
  design: routed experts 4-bit, dense tensors bf16, n-gram table mapped.
- The port agrees with the mlx-lm reference to under 1e-2 max logit delta.
  See "Reference parity".
- Serve answers at draft depth 0 and at depth 2, where the native head accepts
  0.889 of its drafts and decodes 1.16 times faster with identical output.
  See "Native MTP head at depth 2".
- The checkpoint uses two norm conventions and the port splits them correctly.
  See "Norm conventions".
- The n-gram hash constants come from the checkpoint tensors. The mlx-lm
  reference recomputes different multipliers from its seed formula, so the
  runtime never recomputes them.
- The ANE dense lane and its micro-batched, layer-major prefill are in the tree
  (`MLX_ANE_DIRECT=1`, `MLX_QWEN4EXP_ANE_MICROBATCH`, default 256). The
  pipeline structure is verified against the plain forward with GPU
  projections; the ANE program itself is verified only by the opt-in runtime
  test.

- The ANE dense lane is measured and stays off. It costs 48 one-off program
  compiles and it is not token-identical. See "ANE dense lane A/B".

Next:

- Accept rate and decode across the varied-prose set rather than one prompt.
- If the lane is ever wanted, measure it inside a persistent server where the
  48 compiles are paid once, and decide whether an fp16 argmax flip is
  acceptable there.

## Measurements

### Download and transform

| Step | Result |
|---|---|
| Download | 144 files, `verify OK`, 74 min 29 s wall (`tools/qwen38-flash/download.sh`) |
| Download transport | `huggingface_hub[hf_xet]` under Python 3.12, about 58 MB/s |
| Source on disk | 335 GiB |
| Transform | 2 min 52 s wall (`qwen4exp-transform`) |
| Runtime tree | 81.16 GiB of model shards plus 95 GiB n-gram table |

The Homebrew `hf` client runs on Python 3.14, which has no `hf_xet` wheel. It
falls back to an HTTP bridge that measured about 27 MB/s. Routing `hf` through
`uvx --python 3.12 --from 'huggingface_hub[hf_xet]'` about doubled the rate.

Transform output by category, read from the shard headers:

| Category | Bytes |
|---|---|
| Routed experts | 71.78 GiB |
| Attention | 5.13 GiB |
| Embedding and head | 2.38 GiB |
| Other | 1.30 GiB |
| Dense MLP | 0.57 GiB |
| Total | 81.16 GiB in 1540 tensors |

Dtypes are `U32` 57.42 GiB (4-bit expert payload), `BF16` 23.74 GiB (dense
tensors plus expert scales and biases), and `I64` for the n-gram constants.

### Weight load memory

The first real-model run was killed by the kernel with `SIGKILL` about
2.5 minutes in, before it printed anything. A sampler recorded the collapse.

| Elapsed | Process RSS | Free RAM | Compressed |
|---|---|---|---|
| 1 s | 0.00 GB | 64.92 GB | 0.69 GB |
| 11 s | 0.17 GB | 0.06 GB | 0.51 GB |
| 23 s | 17.50 GB | 0.05 GB | 0.51 GB |
| 29 s | 49.24 GB | 0.01 GB | 9.57 GB |
| 39 s | 7.87 GB | 0.01 GB | 77.97 GB |
| 50 s | 5.63 GB | 0.07 GB | 99.54 GB |

Two defects produced this, and the first is the one that mattered.

**The loader read the n-gram table as model weights.** `loadWeights` collected
shards with `FileManager.enumerator(at:includingPropertiesForKeys:)`, which
recurses into subdirectories by default. The runtime tree keeps its n-gram
table in `<weights>/ngram/`, as 128 shards the model maps itself and never
loads as parameters. The enumerator swept them in:

| Enumerated | Shards | Bytes |
|---|---|---|
| Top level, the model | 20 | 81.16 GiB |
| Nested, the n-gram table | 128 | 95.37 GiB |
| Total the loader queued | 148 | 176.53 GiB |

176 GiB does not fit in 128 GiB, so no amount of load pacing could have saved
the run. The enumerator now passes `.skipsSubdirectoryDescendants` and
`.skipsHiddenFiles`. Checkpoints keep their shards beside `config.json`, and
the ranked tree is flat (3 shards at the top level, none nested), so the ranked
path cannot observe the change. Note that `loadArrays(directory:)` still
recurses; it is only reached for separately pinned weight trees, which are
flat, so it was left alone.

**Loading 81 GiB in parallel still doubles the footprint.** `loadWeights` was
tuned for a 21.6 GB tree. It calls `F_RDADVISE` on every shard at once, then
evaluates all shards concurrently. The advisory read fills the unified buffer
cache while the concurrent evaluations allocate the same bytes again as Metal
buffers. In the trace above, free memory reached zero at 11 seconds while
process RSS was still 0.17 GB, which is the advisory read alone.

`loadWeights` now measures the shard bytes against physical memory. A tree
whose bytes, doubled, plus 8 GiB of headroom exceed physical memory loads one
shard at a time with a single-shard read-ahead and clears the MLX allocator
cache after each shard. Smaller trees keep the parallel path: the 21.6 GiB
ranked tree needs 51 GiB against 128 GiB, so it never streams. The 81.16 GiB
tree does stream. `Qwen4ExpGenerate` also caps the MLX allocator cache at
4 GiB so freed buffers return to the operating system.

### Reference parity

`MLXFAST_RUN_QWEN4EXP_PY_PARITY=1 swift test --force-resolved-versions --filter Qwen4Exp`
passes, 32 of 32. The parity case compares the Swift port against the mlx-lm
reference (`tools/qwen38-flash/qwen4_exp.py`) on the tiny synthetic tree, at
`max |logit delta| < 1e-2`.

Getting there needed three harness corrections. None was a defect in the port.

| Correction | Max logit delta |
|---|---|
| Reference could not be imported at all | no result |
| Reference in float32 and unquantized, Swift in bf16 and 4-bit | 2.837607 |
| Both quantized, both float32, reference scales in float32 | 0.080376 |
| Both quantized, both float32, reference scales rounded to bf16 | under 1e-2, passes |

The three corrections were:

1. The reference file is a package module and imports its siblings relatively,
   so it cannot be imported as a standalone file. `parity.py` now loads it under
   the name `mlx_lm.models.qwen4_exp`, which makes those imports resolve.
2. The reference sizes its n-gram table from `ngram_vocab_size_base`, which
   defaults to 20,000,000 and produced a 40,000,064-row table against the tiny
   tree's 16 rows. The tiny source now sets that base to 8 and the divisor to 2,
   which yields the first four primes after 7, sizes [11, 13, 17, 19], and 30
   rows per shard. The tiny source also stores the multipliers the reference
   derives from the config seed, because the reference rebuilds them and ignores
   what the checkpoint holds.
3. The comparison has to be like for like. `parity.py` now quantizes the routed
   experts to 4-bit affine at group size 32, matching the transform, and rounds
   its scales and biases to bf16, matching what the transform stores. The Swift
   side upcasts its bf16 tensors to float32, matching the reference's compute
   precision.

The last row is the informative one: once both sides hold the same values in
the same precision, the port and the reference agree. A wrong norm convention,
a wrong hash, or a wrong residual path would show as a delta of order 1, as the
2.84 row shows. This also confirms the norm split recorded below.

### Real-model smoke

`qwen4exp-generate` against the transformed tree, M4 Max 128 GiB, one run each.

| Run | Prompt tokens | Prefill | Decode tokens | Decode tokens/s | MLX peak |
|---|---|---|---|---|---|
| Chat template, 160 max | 74 | 9.30 s | 160 | 12.29 | 87.49 GB |
| Raw, 48 max | 4 | 10.71 s | 39 | 5.70 | 87.31 GB |

Load stages for the raw run, from `BENCH_VERBOSE=1`:

| Stage | Time |
|---|---|
| rdadvise | 1.6 ms |
| read shards (streamed) | 14,862.7 ms |
| sanitize | 1.4 ms |
| quantize wire | 60.5 ms |
| update params | 14.8 ms |
| bf16 convert | 8.3 ms |
| eval | 4.4 ms |

Streaming reads the 81.16 GiB tree at about 5.5 GB/s, so the streaming branch
costs nothing measurable against the parallel one. Peak process RSS was
51.27 GB and the compressor peaked at 42.30 GB, then drained to 1.18 GB.

Both continuations are coherent. The raw prompt "The quick brown fox" returns
" jumps over the lazy dog." and then repeats that sentence, which is the
expected greedy behaviour with no chat template. The chat prompt returns
reasoning text about separate chaining and open addressing. Prefill of about
10 s is dominated by first-forward kernel compilation, not by the prompt: the
4-token prompt and the 74-token prompt take the same time.

Decode differs between the two runs (12.29 against 5.70 tokens per second)
because the n-gram table pages differ in residency, not because of the prompt
length. These are single cold runs and are directional only.

### Serve

`mlxfast-swift serve` answers on the qwen4_exp tree at draft depth 0, the serial
control. Command and result:

```text
DARKBLOOM_QWEN_GEOMETRY_UNPINNED=1 MLXFAST_NO_SANDBOX=1 .build/release/mlxfast-swift serve \
  --weights <weights> --mtp-head none --mtp-depth 0 --port 8080
```

| Measure | Value |
|---|---|
| Prompt tokens | 23 |
| Seed prefill | 0.32 s |
| Completion tokens | 49 in 50 rounds |
| Decode | 15.85 tokens/s |
| End to end | 14.3 tokens/s |
| Accept rate | not applicable, depth 0 offers no drafts |

The answer is correct: it names Rayleigh scattering and the wavelength
dependence. `--mtp-head none` under `DARKBLOOM_QWEN_GEOMETRY_UNPINNED=1` is the
existing headless escape; this tower carries its MTP head inside the checkpoint
rather than as a separately pinned artifact.

Four gates stood between the tree and a served answer. Each was a Qwen 3.8
assumption in a shared path, and each fix branches on the family so the ranked
path keeps its exact behaviour.

| Gate | Where | Fix |
|---|---|---|
| Exact config key set | `QwenRuntimeWorker.validateRuntimeWorkerPinnedConfigurationSchema` | Branch on `model_type`, with a second pinned key list for qwen4_exp |
| Pinned architecture values | `QwenRuntimeWorker.validateRuntimeWorkerPinnedConfigurationData` | Branch on `model_type`, with a qwen4_exp value gate honouring the same geometry escape |
| Qwen35 eager-loader contract | `QwenRuntimeWorker.runPreflightWorker` | Skip the `Qwen35WeightLoader` checks for a family that does not use that loader |
| Backbone family allow list | `Qwen36MTPHeadAttachment.backboneLayout` | Accept the `qwen4_exp` prefix |

A fifth failure was a trap rather than a gate. `Qwen36MTPBlockSession`
warms the recurrent replay kernel behind `precondition(replayRecurrentPrefix(...))`
at three call sites. This tower keeps no replay tape, so it returns false by
contract and the session repairs generically, but the assertion took the process
down with a precondition failure before it could. The protocol gained
`publishesRecurrentReplayTape`, which defaults to true and is false only for
qwen4_exp; the three warms skip the assertion when there is no replay kernel to
compile. The trims around them still run, since it is the cache row counts that
select the next width's dispatch shapes.

The ranked tree still passes `preflight`, and
`swift test --force-resolved-versions --filter "Qwen36|Qwen35|QwenMTP|Qwen4Exp"`
passes 32 of 32, which covers every file this work touched. The unfiltered
suite was not run to completion: it was still executing after 40 minutes with
no output, and it was stopped rather than waited out.

### Native MTP head at depth 2

The head is native to this checkpoint: 38 `mtp.*` tensors, 1672.7 MiB, carried
in the same tree as the backbone. `--mtp-head none` disables only the external
head the ranked track merges; the session reads `model.hasMTPHead`, so the
native head stays live.

Same prompt as the depth-0 run above, same session shape.

| Measure | Depth 0 | Depth 2 |
|---|---|---|
| Rounds | 50 | 18 |
| Tokens per round | 0.98 | 2.72 |
| Accept rate | not applicable | 0.889 (32 of 36) |
| Effective draft depth | 0 | 2.00 |
| Decode | 15.85 tokens/s | 18.32 tokens/s |
| End to end | 14.3 tokens/s | 16.8 tokens/s |
| Seed prefill | 0.32 s | 0.24 s |

The completion is character-for-character identical to the depth-0 completion,
which is the correctness signal that matters here: the target verifies every
drafted token, so the head can change speed and nothing else.

The 1.16 times decode speedup in this table does NOT generalise. This prompt is
23 tokens, where fixed per-round cost dominates. At a realistic 732 tokens
depth 2 is a 17 percent LOSS on this tower; see "Throughput matrix".

Accept rate of 0.889 is high against the roughly 0.75 the DFlash track reports
on prose. Two differences explain it. This head ships with the checkpoint
rather than being trained separately, and it conditions on the wide
hyper-connection residual the trunk already computes. Depth 2 admits at most 3
tokens per round, so 2.72 tokens per round is close to the ceiling that depth
allows.

One prompt is one prompt. Treat these as directional until they are measured
across the varied-prose set.

### ANE dense lane A/B

`MLX_ANE_DIRECT=1` puts the gated-delta in-projection and the attention
`q_proj` on the ANE in fp16 during prefill. The lane arms only when a prefill
holds at least two full micro-batches, so at the default micro-batch of 256 it
needs 512 or more tokens. A first pass at 336 tokens never armed it; that run is
discarded.

Four prompts, greedy, 32 decode tokens, one cold run per arm.

| Prompt | Tokens | GPU prefill | ANE prefill | Overhead | Tokens identical |
|---|---|---|---|---|---|
| 1 | 720 | 6.98 s | 21.79 s | +14.81 s | yes |
| 1 long | 2112 | 9.56 s | 24.33 s | +14.76 s | yes |
| 2 | 900 | 9.39 s | 21.94 s | +12.55 s | yes |
| 3 | 810 | 8.15 s | 21.97 s | +13.82 s | **no** |

**The cost is compilation.** Both prompt-1 runs build exactly 48 ANE programs,
one per layer, and the overhead is 14.81 s and 14.76 s against a prompt three
times longer, so it is not per-token and not per-micro-batch.

These are cold single-shot processes, so the compile lands inside the measured
prefill. The resident-server measurement in "Throughput matrix" separates the
two: the compile is a one-time startup cost, and post-load ANE prefill is still
11 to 25 percent SLOWER than the GPU. Do not read dispatch parity out of the
subtraction above; the direct measurement contradicts it.

**The lane is not token-identical.** Prompt 3 diverges at character 67, after
"...so that a":

| Arm | Continuation |
|---|---|
| GPU | `\n\nThe gated delta network keeps a recurr` |
| ANE | ` linear attention layer can summarise an` |

That is a near-tie argmax flip, the expected consequence of computing those
projections in fp16 where the GPU uses bf16. One divergence in four runs across
three prompts is enough: the lane changes output, so it cannot be used where
token identity is required.

Both findings point the same way, and the lane already ships off by default.
Leave `MLX_ANE_DIRECT` unset. It would become interesting only in a persistent
server, and only where an fp16-induced argmax flip is acceptable.

### Throughput matrix, both towers, post-load

Every number below comes from a resident `serve` process after its startup
warm, so kernel and ANE compilation is already paid. M4 Max, 128 GiB, one run
per cell, `max_tokens` 32. "Warm" repeats the previous prompt so the session
resumes from its recorded resume point; "cold" is a prompt the process has not
seen. Prefill rate is `prompt_tokens / seed_prefill_seconds`.

| Session | Startup | Phase | Prompt | Prefill | Prefill tok/s | Decode tok/s |
|---|---|---|---|---|---|---|
| MoE depth 0 | 34 s | cold A | 732 | 2.382 s | 307.3 | 16.50 |
| MoE depth 0 | | warm A | 732 | 0.504 s | 1453.3 | 2.38 |
| MoE depth 0 | | cold B | 1032 | 3.144 s | 328.3 | 15.75 |
| MoE depth 2 | 32 s | cold A | 732 | 2.048 s | 357.4 | 13.77 |
| MoE depth 2 | | warm A | 732 | 0.427 s | 1714.7 | 3.61 |
| MoE depth 2 | | cold B | 1032 | 3.141 s | 328.5 | 10.21 |
| MoE depth 0, ANE | 45 s | cold A | 732 | 2.669 s | 274.3 | 16.48 |
| MoE depth 0, ANE | | warm A | 732 | 0.502 s | 1457.0 | 2.32 |
| MoE depth 0, ANE | | cold B | 1032 | 4.164 s | 247.9 | 15.65 |
| Dense depth 0 | 21 s | cold A | 732 | 5.402 s | 135.5 | 13.03 |
| Dense depth 0 | | warm A | 732 | 0.162 s | 4522.2 | 12.38 |
| Dense depth 0 | | cold B | 1032 | 7.313 s | 141.1 | 12.47 |
| Dense depth 2 | 19 s | cold A | 732 | 5.608 s | 130.5 | 18.65 |
| Dense depth 2 | | warm A | 732 | 0.146 s | 5019.7 | 20.46 |
| Dense depth 2 | | cold B | 1032 | 7.309 s | 141.2 | 18.29 |

Drafting statistics for the depth-2 sessions:

| Session | Phase | Accept | Tokens/round |
|---|---|---|---|
| MoE depth 2 | cold A | 0.769 | 2.46 |
| MoE depth 2 | cold B | 0.633 | 2.13 |
| MoE depth 2 | warm A | 0.808 | 2.46 |
| Dense depth 2 | cold A | 0.548 | 2.00 |
| Dense depth 2 | cold B | 0.562 | 2.00 |
| Dense depth 2 | warm A | 0.643 | 2.29 |

Four things this says.

**The MoE prefills about 2.3 times faster.** 307 against 135 tokens per second
on the same prompt. It activates 10 of 512 experts per token where the dense
tower reads every weight, and prefill is weight-bandwidth bound.

**Drafting helps the dense tower and hurts the MoE.** Dense depth 2 decodes
18.65 against 13.03 at depth 0, a clear win. MoE depth 2 decodes 13.77 against
16.50, a 17 percent loss, and this is *despite* the MoE head drafting better:
it accepts 0.769 of its drafts against the dense head's 0.548, and lands 2.46
tokens per round against 2.00. The head is not the problem. A verify row is.
For a dense tower the extra rows in a verify are nearly free, because the same
weights are read whichever rows are present. For an MoE each extra row can
route to a different expert set, so verifying three rows can touch up to three
times the expert weights. Speculation does not amortize on this architecture
the way it does on a dense one.

This corrects an earlier claim on this page that depth 2 gives the MoE a 1.16
times speedup. That was measured on a 23-token prompt, where fixed per-round
cost dominates. On a realistic 732-token prompt depth 2 is a net loss for the
MoE.

**The MoE's warm-cache decode collapses.** On a resumed prompt MoE decode falls
to 2.38 tokens per second from 16.50, about seven times slower, while the dense
tower holds at 12.38 and its prefill drops to 0.162 s as expected. The collapse
reproduces across three independent MoE sessions (2.38, 2.32, 3.61) and never
appears on the dense tower. The likely mechanism, stated as a hypothesis rather
than a measurement: this tower publishes no recurrent replay tape, so
`replayRecurrentPrefix` declines and the session repairs its 36 gated-delta
layers generically. The attention KV resumes cheaply, which is why prefill
stays fast, and the recurrent rebuild lands inside the decode window. The dense
tower publishes a tape and pays nothing. For multi-turn serving this is the
largest single optimization target on this model.

**The ANE compile is a startup cost. The lane as built is not worth it, but
the ANE is not the reason** -- see "Where the ANE lane's cost actually is",
which isolates the micro-batching harness as the regression.
Startup is 45 s with the lane against 34 and 32 s without, so the 48 programs
compile once during the serve warm, not once per prompt. The earlier one-shot
figure of 14.8 s was that same cost measured inside a process that answered one
prompt and exited. But amortizing it does not rescue the lane: post-load ANE
prefill is 274.3 and 247.9 tokens per second against the GPU's 307.3 and 328.3,
so the lane is 11 to 25 percent slower per token in steady state. Decode and
warm prefill are untouched, as expected for a prefill-only lane. This also
corrects the earlier inference on this page that ANE dispatch reaches GPU
parity; that came from subtracting a compile constant from a cold run, and the
direct post-load measurement disagrees with it.

### Where the ANE lane's cost actually is

Three arms, same tower, same prompts, resident serve, prefill tokens/s:

| Arm | Structure | Projections | cold A | cold B |
|---|---|---|---|---|
| `moe_plain` | one fused lazy graph | GPU | **305.1** | **303.7** |
| `moe_mb_gpu` | micro-batched, 144 evals | GPU | 252.6 | 235.4 |
| `moe_mb_ane` | micro-batched, 144 evals | ANE | 267.5 | 233.8 |

The regression is the micro-batching harness, not the ANE. Micro-batching alone
costs 17 to 22 percent. Adding the ANE on top of that same structure *recovers*
6 percent on prompt A and is a wash on prompt B. Measured against a fair
baseline the ANE contributes throughput; it is the harness around it that
loses more than the lane gains.

Three costs in `forwardMicroBatched` are on the serial path, and only the
middle one is overlapped:

1. `makeInput(...)` materializes a lazy array into an IOSurface with a bf16 to
   fp16 conversion, on the calling thread, before `ConcurrentEngines.run`.
2. `predict` on the ANE, overlapped against the GPU's `finish`. This part works.
3. `readOutput(...).asType(...)` copies back with an fp16 to bf16 conversion,
   after the concurrent section.

Plus one `eval(o)` per layer per micro-batch: 48 x 3 = 144 hard barriers for a
732-token prefill, against a baseline that builds one graph and evaluates once.

### ANE program shape stability

The dense tower and the MoE tower cache ANE programs the same way, `[Int:
Program]` keyed on a sequence length, but they key on different lengths, and it
decides everything.

| Tower | Key | Consequence |
|---|---|---|
| Dense | full prompt length, prewarmed only at `S=512` | every new prompt length recompiles all 64 layers |
| MoE | fixed 256-token micro-batch | compiles once, reused at every prompt length |

Measured on the dense tower with the ANE lane on, at unprewarmed lengths:

| Request | Prompt | Prefill | Prefill tok/s |
|---|---|---|---|
| A, first | 732 | 31.83 s | 23.0 |
| A, repeat | 732 | 0.178 s | 4105.7 |
| B (1032, different length) | 1032 | 34.57 s | 29.9 |

The repeat is a prompt-cache hit, not evidence of program reuse. The two cold
requests at different lengths each pay about 34 s, which is the 64-layer
compile, not compute. A dense ANE measurement is therefore only meaningful at a
prewarmed shape; at an arbitrary prompt length the lane is compile-bound.

Fixed-tile keying is the property to keep. The MoE lane already has it.

### Expert distribution across both engines: what bounds it

Offloading `in_proj` and `q_proj` is the wrong unit of work for an MoE. Those
are small dense projections; the expert GEMMs are the load, and all 512 experts
per layer currently run on the GPU while the ANE handles a sliver. The
architecturally interesting split is by expert, so both engines carry expert
work at once.

What bounds it is memory, not ANE speed. Expert parameters across 48 layers:

| Representation | Size |
|---|---|
| 4-bit affine with scales, as shipped | 71.78 GiB (77.1 GB) |
| fp16, which the ANE weight const requires today | 241.6 GB |
| int8 | 120.8 GB |
| int4 | about 60.4 GB |

Partitioning experts, so each lives on exactly one engine, costs
`f * 241.6 + (1 - f) * 77.1` GB at fp16:

| ANE share `f` | Total expert bytes | Fits in 128 GB with about 12 GB of KV and activations |
|---|---|---|
| 0.10 | 93.6 GB | yes |
| 0.15 | 101.7 GB | tight |
| 0.20 | 110.0 GB | no |

So fp16 caps the ANE share near 0.12. If the ANE runs expert GEMMs at rate `r`
relative to the GPU, the balanced split is `f* = r / (1 + r)` and the ceiling is
`1 + r`. Even a slow ANE at `r = 0.5` wants `f* = 0.33` for a 1.5x prefill
ceiling, and fp16 memory will not reach it: at `f = 0.12` the ceiling is 1.14x.

An int8 ANE weight const moves `f*` into range (`f = 0.35` costs 92.4 GB). An
int4 const would make a balanced 50/50 split *smaller* than the tree is today
(30.2 + 38.6 = 68.8 GB against 77.1 GB). `ANEMILBuilder` already carries the
dtype codes (`int8 = 21`, `int4 = 25`, `uint4 = 35`) and a const path for
quantize and dequantize scales, but the builder emits an fp16 weight const only.

The measurement that gates the whole design is `r`: ANE against GPU throughput
on expert-shaped GEMMs, `[640, 2560]` and `[2560, 640]`, at the 256-token tile.
Nothing above should be built before `r` is known, because `r` sets both the
optimal split and whether the quantized weight path is worth implementing.

### Norm conventions

The checkpoint stores two norm conventions, and the stored weights identify
which is which. A weight clustered near 1.0 is a direct scale. A weight
clustered near 0.0 is zero-centred, which means the runtime applies `1 + w`.

| Tensor | Mean | Std | Range | Convention |
|---|---|---|---|---|
| `layers.0.linear_attn.norm.weight` | +0.9668 | 0.0326 | +0.875 to +1.023 | direct |
| `layers.0.attn_hyper_connection.hc_norm.weight` | -0.0635 | 0.4729 | -5.938 to +6.625 | zero-centred |
| `layers.0.mlp_hyper_connection.hc_norm.weight` | -0.1095 | 1.0008 | -2.188 to +3.625 | zero-centred |
| `mtp.pre_fc_norm_hidden.weight` | -0.3284 | 0.3153 | -1.133 to +0.371 | zero-centred |
| `mtp.pre_fc_norm_embedding.weight` | -0.7642 | 0.0673 | -0.879 to -0.275 | zero-centred |

The port already splits these two cases. `Qwen4ExpRMSNormGated` holds the
direct weights and initialises to ones. `Qwen4ExpRMSNorm` holds the
zero-centred weights and initialises to zeros. The measured means match the
initialisers, so the split is correct.

The two `mtp.pre_fc_norm_*` tensors are not centred on zero, but they stay on
the zero-centred side. Under `1 + w` their scales are 0.672 and 0.236, which
attenuates each stream before the concatenation and the `fc` projection. Every
element of `pre_fc_norm_embedding` is negative, so `1 + w` stays in
[0.121, 0.725] and never changes sign. Under the direct convention the same
tensor would flip the sign of the whole embedding stream. Numerical parity
against the reference remains the deciding check.

## ANE against GPU on expert GEMMs

`ANEGemmBench` (`Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEGemmBench.swift`)
builds one fp16 ANE 1x1-conv program per shape through
`Qwen4ExpANEProjection` and times it against the GPU kernel an ANE offload
would actually replace: `MLX.quantizedMM` against a 4-bit affine, group-32
quantized weight (matching this tower's routed-expert conversion --
`Sources/MLXFastModel/Qwen4ExpTransform.swift`'s `expertGroupSize` default;
group 64 is the separate main-backbone dense-tensor conversion, a different
set of tensors) with bf16 activations, not a dense fp32 matmul. No model and
no routing are involved;
the sweep measures the two engines only. The worker verb `qwen4exp-ane-bench`
(`Sources/MLXFastHarness/Qwen4ExpANEBench.swift`, dispatched from
`Sources/MLXFastRuntimeWorkerCLI/main.swift`) sweeps the token dimension `m`
at `[8, 16, 32, 64, 128, 256, 512]` against the two expert shapes for this
tower: gate/up at `[640, 2560]` and down at `[2560, 640]`, 5 iterations per
shape, minimum time kept per shape per run. Every sample also passes a
per-shape correctness gate (the ANE output's mean relative error against a
dense fp32 reference must be under 2e-2, the same tolerance
`Qwen4ExpANEDenseLaneTests.testProjectionMatchesGPUWithinFP16` uses); a shape
that fails is dropped and logged rather than reported at an unvalidated
speed. All 140 samples across the 10 runs behind this section passed that
gate, at a relative error around 0.0003 -- roughly 70x inside tolerance.

With 512 experts at top-10 routing for this tower, a 256-token tile sends an
average of `256 * 10 / 512 = 5` tokens to each expert, so the bucket sizes
below span the range from under- to over-provisioned relative to that
average.

### Revision note

A first pass at this section (superseded, not reproduced here) measured the
ANE against a dense float32 GPU matmul and reported `r` in the 0.78-1.20
range at a naive per-call timing, then, in a first fix attempt, an amortised
timing that turned out to rest on an unsafe measurement pattern. Both
readings are retracted. The GPU comparator was wrong (a real offload
displaces the quantized gather-GEMM the model actually runs, not a dense
fp32 matmul), and a later attempt to fix that by amortising per-call
overhead -- reusing one staged ANE program across many `predict` calls,
materialising the ANE's read on every repeat of a tight loop, batching many
independent GPU calls into one `eval`, and running the ANE and GPU timed
loops as separate sequential blocks -- was found, by bisection, to
reproducibly corrupt process memory under this project's own required
verification command (`MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test`, a debug
build). None of those patterns survived into the version below. The
corruption's actual source looks like a pre-existing issue in the private-API
`ANEDirectDispatch`/`ANEInMemoryModel` bridging this file calls into (outside
this task's edited files) -- its symptom was sensitive to unrelated
allocation-size changes elsewhere in the same function, which is the
signature of heap corruption surfacing at an unrelated site, not of a bug in
whichever line happened to be varied at the time. The measurement below does
not attempt to amortise ANE per-call overhead; it reports one dispatch per
timed repeat on both engines, interleaved, which was the configuration
verified stable across 6 consecutive debug-build test runs and 10
five-shape-sweep release-build runs with zero crashes or dropped shapes.

### Measured rate, 10 runs

The sweep ran 10 times back to back on an idle machine (no other
model-holding process resident, confirmed by `ps aux` before each run),
across two sessions of 5 runs each. All 10 runs returned all 14 samples with
no correctness-gate drops. Per-shape rate is still noisy at this scale (the
GPU quantized kernel is fast enough that both legs are still partly
dispatch-bound at the smallest buckets -- achieved throughput rises from
roughly 60-75 GFLOP/s at `m=8` to 1.4-2.9 TFLOP/s at `m=512` on both engines,
visible in the sweep's per-shape stderr log, so the smaller buckets are not
yet at whatever this hardware's ceiling is), but combining across bucket size
and 2x-weighting gate/up against down (below) averages enough of that out to
show a clear, monotonic-with-a-late-dip trend rather than the flat, noisy
scatter the retracted first pass showed.

Combined rate per bucket (one full expert forward is 2x gate/up + 1x down),
mean and standard deviation across the 10 runs, at the corrected group-32
quantization (a prior version of this table, retracted, used group 64 --
see the fix-round-2 note in the report; the correction moves every `r` up by
roughly 0.04-0.11, no bucket crosses the `0.15` stop threshold or moves `f*`
across the 0.12 fp16-memory line discussed earlier in this document):

| m | r (mean) | r (std) | r (min) | r (max) | f* = r/(1+r) | ceiling = 1+r |
|---|---|---|---|---|---|---|
| 8 | 0.9193 | 0.1062 | 0.7344 | 1.0993 | 0.4790 | 1.9193 |
| 16 | 0.9311 | 0.0720 | 0.8551 | 1.0837 | 0.4822 | 1.9311 |
| 32 | 0.9922 | 0.0745 | 0.8950 | 1.1376 | 0.4981 | 1.9922 |
| 64 | 1.1251 | 0.0787 | 0.9864 | 1.2201 | 0.5294 | 2.1251 |
| 128 | 1.2348 | 0.0700 | 1.1468 | 1.3384 | 0.5525 | 2.2348 |
| 256 | 1.2884 | 0.0388 | 1.2150 | 1.3440 | 0.5630 | 2.2884 |
| 512 | 1.1427 | 0.0936 | 1.0401 | 1.2868 | 0.5333 | 2.1427 |

`r` rises from the smallest bucket to a peak around `m = 256` and eases back
slightly at `m = 512`; every bucket's mean is comfortably above the `0.15`
stop threshold (the lowest single-run reading across all 140 samples was
`0.73`, still well clear), so the sweep does not hit the stop condition (`r <
0.15` at every bucket size). No single bucket stands out as uniquely "best" --
the means across `m = 64` through `512` sit within about 0.15 of each other,
inside one bucket's own run-to-run standard deviation -- so this section does
not pick one point estimate; the whole table, and the range `f* ≈ 0.48-0.56`
/ ceiling `≈ 1.92-2.29` it implies, is the result.

### This is the production-representative number, not a pessimistic floor

`ANEDirectDispatch.Prepared` being a documented one-shot handoff object is not
only a constraint on how this benchmark had to be written -- it is a
constraint on the ANE offload itself, if Task 5 builds one. A real per-call
ANE expert dispatch inside decode or prefill would build a `Prepared` from
`Qwen4ExpANEProjection.callAsFunction` exactly the same way this benchmark's
timed loop does: stage, predict, read, once, every call, because that is the
only supported way to drive this API today. There is no cheaper "warm,
reused-surface" path available to production that this benchmark failed to
exercise. So the `r` table above is not a worst case awaiting a smarter
integration to beat -- it is the number a straightforward integration would
actually get. (Whether a *future* change to `ANEDirectDispatch` itself could
add safe reuse is a separate question this task did not investigate; nothing
here rules that out, but nothing here supports assuming it either.)

One residual, known bias: `aneSeconds` in the table above does not include
materialising the ANE's output into a usable MLX buffer (see
`ANEGemmSample.aneSeconds`'s doc comment in `ANEGemmBench.swift` for why --
doing that on every repeat was one of the patterns found unsafe). `gpuSeconds`
does include full materialisation. That asymmetry biases every `r` in this
table upward (toward making the ANE look faster than a fully realised
comparison would show), by an amount this sweep does not measure. The
direction of the bias is known; its size is not.

Full per-run JSON and stderr (per-shape relative error and achieved GFLOP/s)
for this corrected 10-run set are recorded in the fix-round-2 section of
`.superpowers/sdd/2026-09-03-ane-expert-distribution-plan/task-1-report.md`.

## Micro-batch harness: staging overlap and barrier count

Task 2 targeted the two defects "Where the ANE lane's cost actually is" named.
`makeInput` staged the next micro-batch's input on the calling thread, before
the concurrent section, so the copy and its bf16 to fp16 conversion were never
overlapped. And `eval(o)` inside the GPU closure forced one hard barrier per
`(layer, micro-batch)` pair, 144 for a 732-token prefill at 3 micro-batches,
against a baseline that builds one graph and evaluates once.

### Correction, round 1

The paragraph above still describes the two defects correctly. This paragraph
originally described a fix that moved `makeInput` (staging) inside the `ane`
closure alongside `predict`, plus an `eval(mixes.map { $0.x })` hoist that was
supposed to keep that closure MLX-free. Code review found that claim false and
the code unsafe, not merely mis-described. `makeInput` calls
`ANEDirectDispatch.prepare`, which builds its own transpose, contiguous, and
cast graph from its input and calls MLX `eval` on that graph itself
(`ANEDirectDispatch.swift:142-144`), on a background queue. The
`eval(mixes.map { $0.x })` hoist materializes `mixes[next].x`, not the fresh
graph `prepare` builds from it, so it does not stop `prepare` from driving MLX
evaluation off the calling thread, concurrently with the `gpu` closure's own
MLX graph construction. `ConcurrentEngines.swift`, `Qwen4ExpANEDenseLane.swift`,
and `ANEDirectDispatch.swift` all document `makeInput`/`prepare` as
caller-thread-only for exactly this reason, and the commit message's claim
that "the ane closure only copies resident bytes and never drives MLX
evaluation" was therefore wrong, along with the same sentence originally here.

The staging move is reverted. `makeInput` runs on the calling thread again,
before `ConcurrentEngines.run`, matching the pre-Task-2 code. Only `predict`,
which `Qwen4ExpANEDenseLane.swift` marks `BACKGROUND-SAFE`, runs inside the
`ane` closure. `ANEStagedBox` and the `eval(mixes.map { $0.x })` hoist are
both removed. Neither serves a purpose once staging is back on the calling
thread. The barrier consolidation (removing `eval(o)` from the `gpu` closure,
replacing it with one `eval(hs)` per layer, once per layer instead of once per
micro-batch) is unaffected by this correction and is the whole of this task's
change against the pre-Task-2 baseline.

`testMicroBatchedPrefillMatchesPlainForwardAfterRestructure`
(`Tests/MLXFastTests/Model/Qwen4ExpANEDenseLaneTests.swift`) is an equivalence
guard, not a red-green driver: there is no public hook to count `eval` calls,
so the test checks the property that matters, that the restructured loop still
matches the plain forward path. It passes on both the unmodified and the fixed
code, as expected. `swift test --force-resolved-versions --filter Qwen4Exp`
passes 33 of 33 with the fix in place (32 pre-existing plus this one).

### Benchmark result: inconclusive at the CLI's measurement scale

The numbers in this subsection were measured against the round-0 code (the
retracted staging-overlap version), before the round-1 correction above. They
are reproduced unchanged because the CLI methodology problem they document
applies equally to the corrected code: both versions carry the same
per-process compile cost, and neither this task nor the round-1 correction
re-ran this benchmark against the corrected barrier-only version. A
resident-serve measurement against the corrected code is pending.

The plan brief's prescribed command
(`qwen4exp-generate --raw 1`, a fresh process per run, `MLX_QWEN4EXP_FORCE_MICROBATCH=256`,
a 520-token prompt) does not show the fix at this scale. Three runs before the
fix and three after, on an otherwise idle machine, all land in the 40-55
tokens per second range, against the 252.6 and 235.4 tokens per second the
"Where the ANE lane's cost actually is" table recorded for the equivalent
`moe_mb_gpu` arm.

The reason is a measurement mismatch, not a regression. `moe_mb_gpu` was
measured from a resident `serve` process, after its startup warm, so kernel
compilation was already paid before the timed request. `qwen4exp-generate` is
a fresh process every run: `prefill_seconds` starts after weight load but
still includes each op family's first-call Metal kernel compile, which this
build's JIT-compiled kernel families pay fresh every process launch. For a
520-token prompt at a true 250 to 300 tokens per second, compute alone is
about 1.7 to 2.1 seconds. The measured 9.4 to 13.1 seconds implies roughly 7
to 11 seconds of fixed, per-process compile cost that swamps a few hundred
milliseconds of harness overhead.

A temporary, uncommitted warmup pass (call `model()` once on the same prompt
and cache shape, discard the result, then time a second call) confirmed the
mechanism: warm prefill dropped from about 10.6 seconds to 2.3 to 3.2 seconds,
in the 165 to 222 tokens per second range, on both the fixed and the
unmodified code. But at that range, three to four runs per side were not
enough to separate the two: pre-fix samples averaged about 195 tokens per
second (n=3), post-fix about 183 (n=4), with the per-run spread (165 to 222)
larger than the gap between the two averages. This warmup diagnostic was not
committed; it is not part of the editable surface this task's brief named, and
a reliable read needs either many more samples or the doc's own resident
`serve` methodology (as used for the `moe_plain` and `moe_mb_gpu` rows above),
neither of which this task ran.

What is established: the barrier consolidation (144 to 48 hard syncs for a
732-token, 3-micro-batch prefill on the `useANE: false` arm this task
targets) is implemented and verified equivalent to the plain forward path.
The staging-overlap half of the original fix is retracted; see "Correction,
round 1" above. What is not established, from this task's measurements: a
numeric prefill speedup on this machine, at this session's noise floor, for
either the round-0 or the round-1 code. Revisit with a resident-server
measurement, matching the "Where the ANE lane's cost actually is" table's own
method, before drawing a numeric conclusion.

## Dense lane: program-key bucketing and the padding it costs

Commit `00d4e270`. Two changes, one of which pays for the other.

### The compile problem, and the bucket that fixes it

`ANESplitMLPCache` keyed fixed-shape ANE programs on the exact prompt length,
and the runtime prewarms only `S=512`. Any other length therefore rebuilt all
64 layers. The rows at lines 468 to 470 above record what that costs: 31.83
seconds at 732 tokens and 34.57 seconds at 1032, which is compile time and not
compute.

The key now rounds up to a power of two from a floor of 128, so five programs
cover every prompt length this project measures.

### The padding that bucketing costs

A fixed-shape program compiled at the bucket needs its input padded up to the
bucket. The first spelling padded at the caller and passed the padded array
into `ANEFusedSplitMLP`, whose GPU suffix ran on the same rows. Both engines
computed the padding.

| Prompt tokens | Bucket | Both engines pad | ANE leg only |
|---|---|---|---|
| 732 | 1024 | 28.5 percent | 11.1 percent |
| 1032 | 2048 | 49.6 percent | 23.5 percent |

The column that matters is the second one. Only the ANE program is
fixed-shape. `gpuPartial` is a `quantizedMM` chain and accepts any row count,
and at the deployed `MLX_ANE_FRACTION` of 0.3125 the ANE holds `F` = 5440 of
17408 intermediate channels while the GPU holds the other 11968, which is
68.75 percent. Most of the discarded work was therefore on the side that never
needed padding.

The padding moved into `ANEFusedSplitMLP.padForANE`, scoped to the ANE leg.
The class slices the padded rows off the ANE output before the two partials
add.

### What the tests measure

`ANEFusedSplitMLPTests.paddingIsScopedToTheANELeg`, at `S=100` into a program
compiled at 128, hidden 5120, inter 17408, `aneFraction` 0.125.

| Quantity | Measured |
|---|---|
| max abs against the all-GPU reference | 0.015625 |
| mean abs against the all-GPU reference | 0.0010453 |
| max abs when the padding rows change | 0.0 |

The first figure is the same 0.015625 the full-length tests record, so the
short-input path adds no error of its own. The third is the load-bearing one.
Zero padding rows are exact only if the fused ANE program has no cross-token
mixing, and that had been an assumption. Running the same real rows under
padding values 50 times their magnitude moves them by exactly zero, which
measures the assumption instead of restating it. It would not hold for an
attention program, where padded rows enter the softmax.

When the caller's row count already equals the compiled one, the pad and the
slice are both identities, so every existing caller and test sees a bit-exact
no-op. All 8 tests in `ANEFusedSplitMLPTests` and `ANESplitMLPCacheTests`
pass.

### What is left

Fixed-tile keying, one program per layer dispatched `ceil(S / tile)` times at
a tile of 128 or 256 rows, caps the residual waste near 5 to 12 percent and
removes the per-length recompile entirely. It also multiplies every
per-dispatch fixed cost by the dispatch count, so it should land after the
staging work that reduces those costs. Not built.

No timing is reported here. The change removes work that was provably
discarded, and the count of removed rows is exact, but no prefill measurement
was taken on a quiet machine.

## Dense lane measured after bucketing (2026-09-03)

M4 Max, 128 GiB, quiet machine, resident `serve` on the dense tower
(`./weights`, hidden 5120, intermediate 17408), `--mtp-depth 0`, greedy,
`max_tokens` 8. Prefill rate is `prompt_tokens / seed_prefill_seconds` read
from the response. Arms differ only by `MLX_ANE_DIRECT`.

### One compile now covers every length in a bucket

Four distinct prompt lengths, sent cold in order to one process. 561, 754 and
966 tokens all bucket to 1024; 1136 buckets to 2048.

| Prompt tokens | GPU arm | ANE arm |
|---|---|---|
| 561 | 3.954 s | 33.648 s |
| 754 | 5.568 s | 5.345 s |
| 966 | 7.098 s | 6.612 s |
| 1136 | 8.497 s | 7.922 s |

Only the first prompt compiles. The serve log shows two program sets ever
built, 64 at bucket 512 from the startup prewarm and 53 at bucket 1024, and no
further build after the first request. Before bucketing each new length
rebuilt all 64 layers, which the rows above at lines 466 to 470 record as
31.83 s and 34.57 s for two different lengths. A second run in reversed arm
order reproduced this within 1 percent.

Note what the 1136-token prompt shows: it never triggered a bucket-2048 build.
Prefill is chunked, so the sequence length the cache sees is not the prompt
length. The first build logs `S=554` for a 561-token prompt.

### Steady state, paired, seven prompts

One process per arm, eight distinct prompts of increasing length sent cold.
The first is the ANE arm's compile and is excluded. The remaining seven are
paired by token count across the two arms.

| Prompt tokens | GPU tok/s | ANE tok/s | Ratio |
|---|---|---|---|
| 675 | 121.3 | 126.9 | 1.046 |
| 692 | 128.3 | 136.6 | 1.065 |
| 723 | 119.8 | 128.8 | 1.075 |
| 735 | 120.4 | 130.5 | 1.084 |
| 765 | 129.0 | 137.8 | 1.068 |
| 782 | 113.5 | 130.6 | 1.151 |
| 809 | 126.8 | 138.8 | 1.095 |

Mean ratio 1.083, median 1.075, seven wins from seven pairs. The ANE lane is
about 8 percent faster than the GPU path on this tower in steady state.

This corrects the reading at lines 414 to 422 above, which concluded the lane
is 11 to 25 percent slower per token. That measurement was on the MoE tower and
still stands there. It does not describe the dense tower, where no valid
post-compile comparison existed until now.

### The compile still has to be earned back

| Quantity | Value |
|---|---|
| Mean prefill, GPU | 6.042 s |
| Mean prefill, ANE | 5.571 s |
| Saving per prompt | 0.471 s |
| One-time compile above the GPU cost | 28.4 s |
| Break-even | about 60 prompts |

So the lane is worth arming in a persistent server and is a clear loss in a
process that answers a handful of prompts and exits. That is the same
conclusion the earlier one-shot measurements reached, now with the steady-state
half measured instead of inferred.

### Ten of sixty-four layers do not load

The serve log records 10 build failures per bucket-1024 set:

```text
BUILD FAILED S=554 bucket=1024 hidden=5120 inter=17408
error=load("createProgramInstanceForModel:...: Program load failure (0x50004)")
```

So 54 layers run on the ANE and 10 fall back to the GPU, and the 8 percent
above is what a lane at 84 percent strength delivers. The failure appears after
about 54 programs are resident, which points at an ANE program-memory limit
rather than a defect in any one layer: the prefix weights are
3 x 5440 x 5120 fp16 values per layer, about 167 MB, so 54 layers hold roughly
9 GB.

That is the direct link to the int8 finding recorded in
`docs/perf/ane-unified-activation-plan.md`. int8 weights through
`constexpr_blockwise_shift_scale` compile and place the conv on the ANE, and
they halve exactly the bytes that appear to be exhausting this limit. Whether
that admits the remaining 10 layers is unmeasured and is the next thing to try.

## Is a uniform fp16 split worth building? No, and here is the number

The split is not uniform in precision. The ANE prefix runs fp16, dequantized
from the 4-bit weights, while `gpuPartial` runs native 4-bit `quantizedMM` at
group 64. That confounds two variables whenever the lane is compared against
the GPU: the engine changes and the weight precision changes together.

`MLX_ANE_FP16_GPU=1` separates them. It computes the SAME fp16 prefix on the
GPU instead of the ANE, keeping the identical 4-bit suffix, so a three-arm
comparison isolates each variable. One serve per arm, five prompts, first
dropped because it carries the ANE compile.

| Arm | Prefix | Suffix | Mean tok/s |
|---|---|---|---|
| baseline | GPU 4-bit | GPU 4-bit | 126.7 |
| ablation | GPU fp16 | GPU 4-bit | 122.0 |
| lane | ANE fp16 | GPU 4-bit | 133.9 |

| Effect | Isolated by | Result |
|---|---|---|
| Precision, engine held at GPU | 4-bit against fp16 | **-3.7 percent** |
| Engine, precision held at fp16 | GPU against ANE | **+9.8 percent** |
| Net against the all-GPU baseline | | +5.7 percent |

Paired per prompt, the fp16-on-GPU arm loses on four of four (0.941, 0.950,
0.970, 0.991) and the ANE arm wins on four of four (1.077, 1.040, 1.055,
1.056).

Two conclusions follow, and the first one settles the question.

**fp16 is not what helps. A second engine is.** Moving the prefix from 4-bit to
fp16 while staying on the GPU makes it slower, because prefill is
weight-bandwidth bound and fp16 is four times the bytes of 4-bit. The lane wins
in spite of its precision, not because of it.

**So a uniform fp16 split would be worse, not better.** The measured penalty is
3.7 percent for putting 31.25 percent of the intermediate channels in fp16. The
GPU suffix holds the other 68.75 percent, so applying the same change there
moves considerably more weight bytes. Nothing in this data supports building
it, and the fraction sweep agrees independently: 0.0 gives 125.5, 0.3125 gives
130.8, 0.5 gives 98.8, 0.75 gives 68.5, and 1.0 did not finish four prompts in
ten minutes.

The same bandwidth argument applies to int8 on the GPU side, which is twice the
bytes of 4-bit rather than four times, so it should lose by less and still
lose. int8 on the ANE side is the opposite case and remains worth trying: it
halves the 167 MB of fp16 prefix weights each ANE program holds, which is what
appears to exhaust the program-load limit at 54 layers.

### The deployed fraction is defensible

| Fraction | Programs built | Failed | Steady-state tok/s |
|---|---|---|---|
| 0.0 | 0 | 0 | 125.5 |
| 0.125 | 117 | 10 | 120.1 |
| 0.3125 | 117 | 10 | 130.8 |
| 0.5 | 78 | 0 | 98.8 |
| 0.75 | 77 | 0 | 68.5 |
| 1.0 | -- | -- | did not finish |

`MLX_ANE_FRACTION` 0.3125 was previously carried without an end-to-end
measurement behind it. It is the best of the six sampled, and the curve falls
away sharply above it.

## MoE tower, re-measured by resident serve (2026-09-03)

The note at the end of "Micro-batch harness" asked for a resident-server
measurement before drawing a numeric conclusion. This is it. Same machine, same
six prompts as the dense measurements above, one serve per arm, 81 GiB tree,
`--mtp-depth 0`.

| Prompt tokens | plain | micro-batched GPU | micro-batched ANE |
|---|---|---|---|
| 653 | 270.4 | 206.1 | 196.5 |
| 675 | 287.7 | 214.5 | 215.9 |
| 692 | 295.1 | 216.0 | 217.3 |
| 723 | 298.9 | 215.0 | 217.6 |
| 735 | 301.6 | 228.4 | 231.7 |
| 765 | 308.3 | 234.8 | 238.2 |
| mean | **293.7** | **219.1** | **219.5** |

| Effect | Result |
|---|---|
| Micro-batching against plain | **-25.4 percent** |
| ANE on top of micro-batching | **+0.2 percent** |
| The lane, net against plain | **-25.2 percent** |

The `plain` arm reproduces the earlier `moe_plain` row of 305.1 within the
difference the shorter prompts explain. The micro-batching cost reproduces at
the high end of the earlier 17 to 22 percent range.

What is new is the middle row. Adding the ANE on top of micro-batching returns
0.2 percent, which is inside the run-to-run spread, on six of six prompts. The
earlier table put that recovery at about 6 percent. Either figure is far too
small to pay for the 25 percent the restructuring costs.

### Why the two towers disagree

The dense lane wins 5.7 percent and the MoE lane loses 25.2 percent, on the
same machine, with the same second engine. The difference is not the ANE. It is
what each lane has to do to the forward pass to reach it.

The dense lane splits the MLP by intermediate channel. Both halves run inside
one forward, the graph keeps its shape, and the ANE prefix overlaps the GPU
suffix. Nothing is restructured, so the only cost is the dispatch, and the
engine gain survives.

The MoE lane splits by micro-batch. Reaching the ANE at all requires cutting
the prefill into 256-token pieces, which multiplies the per-layer dispatch and
barrier count and forfeits the single fused lazy graph. That costs 25.4 percent
before the ANE contributes anything, and the ANE then contributes nothing
measurable.

This is the direct answer to the idea that a mixture-of-experts model should
suit two engines particularly well, because experts can be spread across them.
The hardware argument is sound and the measurement still says no: the
mechanism that distributes the work costs more than the second engine returns.
An MoE ANE lane becomes interesting only if experts can be reached without
micro-batching the prefill.

### The cost is the restructuring, not the round trips

A natural reading of the MoE lane's slowness is that data round-trips between
the engines: staged out of MLX, into an IOSurface, through the ANE, and back.
The three arms above settle it, because one of them contains no ANE at all.

| Arm | Micro-batched | Data crosses engines | tok/s |
|---|---|---|---|
| plain | no | no | 293.7 |
| micro-batched GPU | yes | **no** | 219.1 |
| micro-batched ANE | yes | yes | 219.5 |

In the middle arm `useANE` is false, so `program` is nil, `aneServes` is false
for every micro-batch, and every projection goes through
`layer.gpuProjection`. There is no `makeInput`, no IOSurface, no host copy, and
nothing leaves the GPU. That arm still pays the entire 25.4 percent. Adding the
round trips on top of it costs a further 0.2 percent.

So the round trips are real and they are not the reason the lane is slow.
Cutting one fused lazy graph into `n` segments multiplies the per-layer
dispatch and barrier count, and that is charged whether or not any data ever
moves between engines.

This confirms rather than contradicts `docs/perf/ane-zero-transfer-spec.md`,
which predicted the same thing from static analysis: it found staging was not
the term that mattered, priced the micro-batching entrance fee at 499 to 986
milliseconds against a lane ceiling of 2.7 to 8.1 percent of prefill, and
cancelled the transfer-reduction tasks on that basis. These are the end-to-end
numbers behind that prediction.

The practical consequence is that transfer-reduction work cannot rescue this
lane. Only reaching the experts without segmenting the prefill can.

## MoE tower, fused single-split lanes (2026-09-03)

The prior section closed on one path forward: reach the MoE experts without
segmenting the prefill. This section measures that path. The fused lane keeps
each piece of work as one independent unit with one join per layer, and does
not micro-batch. Two features were measured together and separately:

- Feature A, split: an ANE prefix over a channel slice of the linear-attention
  `in_proj_qkv` and attention `q_proj` projections, joined once per layer with
  a GPU suffix over the remaining channels.
- Feature B, shared: the whole shared expert MLP (`mlp.shared_expert`) run
  entirely on the ANE, with no split.

### Env knobs

| Variable | Effect |
|---|---|
| `MLX_ANE_DIRECT` | enables the ANE-direct path |
| `MLX_QWEN4EXP_ANE_MODE` | selects `split`, `shared`, or `both` |
| `MLX_QWEN4EXP_ANE_LOG` | turns on the `[qwen4exp-ane]` build/failure log |
| `MLX_QWEN4EXP_ANE_SPLIT_FRACTION` | fraction of intermediate channels sent to the ANE prefix in split mode (not swept this run) |

### Program budget from the spec

The design spec for this lane set a budget of 4096 MB and 256 programs for
`Qwen4ExpANEFused`, and projected in its section 5.7 that the `both` mode's
1.65 GB across 96 programs sits well below the point where ANE program loads
start failing.

### Measurement table

Five arms ran on one machine, one serve process at a time, six fixed prompts
per arm, `--mtp-depth 0`. Figures are copied verbatim from the measurement
report.

Per-prompt tok/s (prompt tokens = prompt_tokens divided by seed-prefill
seconds):

| tokens | plain | split | shared | both | plain2 |
|---:|---:|---:|---:|---:|---:|
| 653 | 260.1 | 109.3 | 104.1 | 63.0 | 181.2 |
| 675 | 258.1 | 185.3 | 253.6 | 262.2 | 268.5 |
| 692 | 257.8 | 182.3 | 265.9 | 270.0 | 270.7 |
| 723 | 250.6 | 181.7 | 265.3 | 265.6 | 272.1 |
| 735 | 266.4 | 184.0 | 272.5 | 272.3 | 276.7 |
| 765 | 270.1 | 189.2 | 281.0 | 269.3 | 282.7 |

Mean across all six prompts:

| arm | mean tok/s | vs plain |
|---|---:|---:|
| plain (control) | 260.52 | -- |
| split | 171.97 | -34.0 percent |
| shared | 240.40 | -7.7 percent |
| both | 233.73 | -10.3 percent |
| plain2 (control repeat) | 258.65 | -0.7 percent |

Mean across prompts 2 to 6, excluding the first-prefill-after-load cost that
every arm pays regardless of mode:

| arm | mean tok/s (2-6) | vs plain (2-6) |
|---|---:|---:|
| plain (control) | 260.60 | -- |
| split | 184.50 | -29.2 percent |
| shared | 267.66 | +2.7 percent |
| both | 267.88 | +2.8 percent |
| plain2 (control repeat) | 274.14 | +5.2 percent |

The measurement report treats the prompts 2-6 view as the more trustworthy
comparison, since the first prompt in every arm after a fresh process load
pays a large, mode-independent tax that swamps any per-mode effect if left in
the mean. The plain2 repeat itself measures +5.2 percent over the original
plain on prompts 2-6, using byte-identical code, which sets a floor for what
counts as run-to-run drift rather than a code effect.

### Build and failure counts

The `both` arm is the only one with a surviving build log (188 tagged lines).
The `split` and `shared` arms wrote no lines to the served log at all,
including two lines that fire unconditionally regardless of the logging flag,
so their build and failure counts could not be read from this run. The
measurement report attributes this to stdio buffering combined with an
ungraceful process kill, not to the lanes being inert (see the token-head
differences below).

From the `both` arm's log:

| Event | Count |
|---|---:|
| split `in_proj_qkv` built | 47 |
| split `q_proj` built | 16 |
| split `in_proj_qkv` BUILD FAILED (bucket=1024) | 23 |
| split `q_proj` BUILD FAILED (bucket=1024) | 8 |
| shared built (49 at bucket=512, 14 at bucket=1024) | 63 |
| shared BUILD FAILED (bucket=1024) | 31 |

At bucket 1024, 31 of 45 shared-expert attempts and 31 of 54 split attempts
failed to build (`Program load failure (0x50004)`) and fell back to GPU. This
reproduces the known ANE program-memory limit reported elsewhere in this
document, at a smaller byte count than the design spec's own projection: the
spec argued the `both` mode's 1.65 GB across 96 programs sits well below the
observed failure point, and on this run it does not.

### Token-head differences

Every arm that touches the ANE (split, shared, both) diverges from the
plain/plain2 control on at least one of the six prompts, and never on the same
prompt for all three ANE-touching arms:

| tokens | plain / plain2 | split | shared | both |
|---:|---|---|---|---|
| 653 | same across all arms | same | same | same |
| 675 | baseline text | same | diverges (word swap) | same |
| 692 | baseline text | diverges | diverges (same alternate as split) | same |
| 723 | same across all arms | same | same | same |
| 735 | same across all arms | same | same | diverges |
| 765 | same across all arms | same | same | same |

Plain and plain2 agree exactly on all six prompts, as expected for
byte-identical greedy decoding. The divergences are consistent with fp16
representation differences in the ANE lanes, the same effect the design spec
anticipates. Because a silently inert lane cannot change model output, these
divergences are evidence that split and shared genuinely executed ANE compute
on this run, even where the confirming build log lines were lost.

### Reading against the other two lanes

| Lane | Net effect vs its control |
|---|---:|
| MoE, micro-batch (prior section) | -25.2 percent |
| Dense, single-split fp16-on-ANE (earlier section) | +5.7 percent |
| MoE, fused split (this section, prompts 2-6) | -29.2 percent |
| MoE, fused shared (this section, prompts 2-6) | +2.7 percent |
| MoE, fused both (this section, prompts 2-6) | +2.8 percent |

Shared and both, on the prompts 2-6 view, land close to the plain2 drift floor
of +5.2 percent, so neither shows a result distinguishable from run-to-run
noise on this single pass. Split is the one clear signal in this table: a
sustained loss close in size to the micro-batch lane's loss, even though split
does not micro-batch and keeps one graph per layer. This suggests the earlier
reading that micro-batching itself is the tax needs a caveat: a single-split,
single-join structure can also lose, at least for the channel-prefix
projections split targets here.

Shared, which moves a whole expert MLP to the ANE with no split, does not show
the same loss and sits near parity with plain. That is closer in shape to the
dense tower's win, where the split also stays inside one graph per layer, but
the shared result here is too close to the drift floor to call a win on this
one pass.

The `both` arm's bucket-1024 build failure rate means roughly half its ANE
program load attempts fell back to GPU, so its throughput number is a blend of
ANE and GPU compute and should not be read as a clean measurement of the fused
design at full strength.

Missing log evidence for split and shared, no thermal cooldown gate between
arms, and a single pass per arm mean none of these MoE fused-lane numbers
should be treated as final. A rerun with the stdio-buffering fix from the
measurement report, and a proper cooldown gate between arms, is needed before
drawing a conclusion about split or shared on this workload.
