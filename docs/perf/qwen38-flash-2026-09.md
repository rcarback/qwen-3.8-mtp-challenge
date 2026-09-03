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

**The ANE compile is a startup cost, and the lane is still not worth it.**
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
