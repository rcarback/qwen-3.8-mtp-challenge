# CARBACK MLXFast Optimization Work Plan

**Goal**

Reduce inference latency and memory use across supported MLX models while preserving each workload's required behavior.

**Status**

Proposed experiments, 2026-09-07. This file is a plan, not a claim of new performance results.

**Execution**

Work through the applicable tasks sequentially with `superpowers:executing-plans`. Start with measurement and correctness. Do not run simultaneous model-holding processes. Implementation, commits, and ranked submission are separate actions from writing this plan.

**Architecture**

Keep the native MLX path as the reference. Reuse proven mechanisms through explicit geometry and capability checks. Introduce a shared abstraction only after two model adapters need the same behavior.

**Stack**

Swift, vendored MLX/MLXLMCommon, Metal, and optional local ANE support. Python tools remain offline research or local serving tools.

**Basis**

The user requested this upstream review and work plan. Evidence includes the current source tree and [ANE reevaluation](docs/perf/ane-gpu-reevaluation-2026-09-06.md), [closed-bead audit](docs/perf/closed-bead-review-2026-09-06.md), and [cleanup review](docs/perf/upstream-cleanup-2026-09-07.md).

## 1. Review boundary

The fetched upstream is `https://github.com/Layr-Labs/qwen-3.8-mtp-challenge`, remote `origin`.
Its main tip and this branch's merge base are `0863b06ac16e26e48fc06e97444095b00feb66d4`.
The reviewed local HEAD is `5b1ffea135cbb849ee08e88e5bf8d178aa3ab161`.
Upstream already includes accepted submissions, so this is not a pristine MLX or Laguna baseline.

The committed delta contains 343 files, 70,231 inserted lines, and 474 deleted lines.
The appendix inventories every changed file, including tests, scripts, skills, and records.
The review also includes the uncommitted cleanup of rejected offload paths and the corrected fusion tests.
It does not treat the other worker's unfinished clean-branch measurements as verified results.

Coverage means the entire change inventory is assigned a disposition and work area.
It does not mean every source line has passed a formal correctness audit.
Historical measurements motivate experiments and do not substitute for fresh paired controls.

## 2. What applies to any model

A reusable optimization process applies broadly. Individual kernels and cache policies do not.

| Model capability | Relevant work | Check before using it |
| --- | --- | --- |
| Dense linear layers | Weight precision, packing, GEMV/GEMM selection, fused epilogues | Actual shapes, strides, dtype, quantization metadata, tied weights |
| Full or sliding-window attention | Fused attention, KV precision, cache layout | Head dimensions, GQA ratio, mask type, sinks, positional encoding, window bounds |
| Recurrent or hybrid layers | State snapshots, prefill chunking, speculative repair | Every recurrent state must restore or replay correctly |
| Routed experts | Gather batching, gate/up fusion, routing and load balance | Expert count, top-k, skew, activation, quantization, reduction order |
| Native MTP or compatible draft model | Speculation and proposal-only compression | Hidden-state convention, tokenizer mapping, verifier and rollback support |
| Large lookup or embedding tables | Page layout, selective reads, prefetch | Hash semantics, encoding, cache state, real pages touched |
| Repeated requests | Prefix reuse and disk checkpoints | Exact prefix, model/tokenizer identity, cache policy, state compatibility |
| Large prefill MLP and compatible ANE | Static split across engines | Supported weight form, actual device placement, quality, shared-memory contention |

Start with one small model from each relevant architecture family that fits the machine.
Then validate on the current dense Qwen and MoE towers where appropriate.
Do not require ANE, MTP, or Qwen-specific cache methods from a model that lacks them.

## 3. Contract and measurement rules

Separate three kinds of change before running an experiment:

1. **Behavior-preserving execution:** kernel scheduling and exact layout changes, including valid state reuse. Use the reference computation and required token gates.
2. **Approximate representation:** target-weight/KV quantization, rotation, pruning, or merging. Require an explicit workload quality budget. These are not automatically legal ranked changes.
3. **Changed input or product behavior:** compaction, truncation, altered sampling, or reduced context. Evaluate task outcomes separately from inference speed.

`benchmark.json` and the current trusted track contract determine ranked scope and scoring.
Local serving code, new vendored files, and arbitrary target-weight transforms may not ship.
Do not port local overrides into trusted gates or infer eligibility from an old document.
The ranked decode window charges seed prefill, but ordinary prefill tok/s is not the ranked score.

For each accepted experiment, retain one receipt with:

- Git revision and dirty patch identity, model and tokenizer hashes, weight format and head identity.
- Hardware, software versions, power and thermal state, achieved competing load, and arm order.
- Actual prompt/output token counts, exact tokenized inputs when comparable, cache state, and seed.
- Requested path and executed path, including fallback reason, shape, device, and program count.
- Cold load/compile time, warm prefill, TTFT, decode time, request wall time, p50/p95, and peak memory.
- Logit/NLL checks and required output checks, with per-prompt results and failure receipts.

Use alternating paired arms with at least three valid pairs per selected workload initially.
Add repeats only when uncertainty prevents a decision. Report the distribution and control drift.
Exclude invalid samples only under rules fixed before the run, and retain their reasons.
A stalled or timed-out request remains a product result even when excluded from steady-state timing.

Default promotion rule: required correctness passes, the latency difference exceeds measured noise,
and memory or tail latency does not violate the workload budget.
If no effect can be distinguished, retain the simpler reference path.

## 4. Findings across the upstream delta

| Change family | What the fork provides | Disposition |
| --- | --- | --- |
| CLI, worker protocol, and geometry validation | Local model support, chat/serve, headless operation, traces, configurable limits | Keep capabilities. Audit mirrored validation and local/trusted boundaries. |
| Qwen model and target adapters | Dense, MoE, recurrent state, norm conventions, external proposal heads | Keep model-specific semantics. Generalize capabilities only after a second adapter proves the interface. |
| Loader and transforms | Bounded shard loading, q8 dense tree, codecs, separate head loading | Keep memory safeguards. Replace assumptions about every checkpoint's layout with explicit discovery rules. |
| Dense kernels and stream plumbing | Explicit-stream fix, packed prework, wider QMV dispatch, tile probes | Keep verified fixes. Revalidate geometry-specific choices per model and device. |
| Attention and KV | Head-dimension-256 support, quantized fused decode, rotation and width bounds | Keep guarded paths. Check packing support, causal alignment, rollback, and long-context behavior. |
| MoE | Gate/up fusion, router prototype, scalar fused expert prototype, pool experiments | Keep measured exact fusion. Retire rejected execution paths before designing another scheduler. |
| Speculation | Native MTP, DFlash2, lookup proposals, acceptance policy, head cost estimates | Keep proposal diversity optional. Reprice actual work and repair depth-zero recovery. |
| Session state | RAM snapshots, learned boundaries, disk cache, prefix matching | Keep valid reuse. Fix accounting and prove long-horizon restore behavior. |
| ANE | Direct fused split, quantized MIL forms, bridges, probes, multiple MoE lanes | Keep useful direct probes. The bank and early split designs were removed locally. |
| Serving and compaction | HTTP API, ordered tool JSON, state retention, native and proxy compaction | Keep API behavior. Avoid two compaction owners and distinguish changed prompts from faster inference. |
| Test and research tooling | Synthetic kernels, parity fixtures, quality runs, timing scripts | Keep reproducible evidence. Remove false-positive tests and unsafe cleanup assumptions. |
| Documents, raw results, and skill files | Historical decisions and machine-specific recipes | Retain provenance. Mark superseded claims and avoid treating recipes as universal defaults. |

## 5. Work order

| Order | Task | Applicability | Existing beads |
| --- | --- | --- | --- |
| Now | A. Make measurements truthful | All | `faq`, `3dh`, `mi6`, `dyv` |
| Now | B. Repair resource and state accounting | Serving and recurrent models | `urw`, `c44`, `6tb` |
| Now | C. Establish the current byte and time budget | All | `dl3` |
| Next | D. Remove work before changing kernels | All | `qex` |
| Next | E. Test selective weight compression | Models with suitable dense weights | `mc6` |
| Next | F. Tune GEMV/GEMM by real geometry | Dense and expert projections | `qex` |
| Next | G. Reduce attention traffic and intermediate storage | Attention models | `asj` |
| Next | H. Tune prefill chunks within memory limits | All, with state-specific checks | `8it`, `39j` |
| Next | I. Recover the value of speculation | Models with a verified proposal path | `9o0`, `nnr`, `668`, `w75` |
| Next | J. Improve routed-expert execution | MoE only | `5jk`, `qex` |
| Next | K. Reduce lookup-table I/O | Models with large lookup tables | `0e5`, `9ad`, `zyf` |
| Next | L. Make prefix reuse reliable and effective | Repeated-request serving | `6tb`, `c44` |
| Next | M. Share dense work between ANE and GPU | Compatible hardware and non-MoE kernels | `dl3`, `faq`, `4r9` |
| Later | N. Bound compaction and simplify serving ownership | Local agent workloads | New task after ownership review |
| Later | O. Generalize only validated mechanisms | At least two model families | New task after cross-model receipts |

Tasks D through M depend on A and C, but do not all depend on each other.
B blocks adoption of deeper checkpoint reuse. H and G share memory-budget evidence.
Fix MoE startup before model-holding MTP experiments. Model-free policy tests can proceed first.
Do not create duplicate beads for the entries already tracked.

## A. Make measurements truthful

**Files**

`tools/ane-probes/serve-sweep.sh`, `tools/ane-probes/agree.py`,
`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift`,
`Tests/MLXFastTests/Model/DensePerplexityTests.swift`, and the existing phase/dispatch tests.

- [ ] Add an invalid-arm fixture: request a missing bank/bucket or unsupported kernel shape and verify the receipt identifies fallback.
- [ ] Count actual fused dispatch after its enable flag and all guards pass. `QuantizedDispatchStats.record` currently receives shape support without `fusedQuantizedEnabled`.
- [ ] Replace silent early returns in selected runtime tests with explicit skip traits. Require enabled runs before claiming hardware coverage.
- [ ] Verify the benchmark server is the process started by the arm. Readiness must fail on child exit or port conflict.
- [ ] Keep character-level completion agreement descriptive. Capture token IDs and logit gaps before calling a difference a near tie.
- [ ] Separate host construction, command encoding, GPU execution, and waiting with a bounded trace. Compare instrumented and normal latency.

**Deliverable**

One successful receipt and deliberate negative controls for fallback and disabled execution.
**Stop**

Do not rank kernels until these controls distinguish their paths.
The local cleanup already fixed `dyv`: explicit fusion overrides now select different test arms.

## B. Repair resource and state accounting

**Files**

`Sources/MLXFastHarness/QwenRuntimeMTPWorker.swift`,
`Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift`,
`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`, `QwenSessionCacheStore.swift`, and `QwenPrefillDiskCache.swift`.

- [ ] For `urw`, open and release a trace sink repeatedly, then verify owned descriptors close while stderr remains usable.
- [ ] Change ownership at the existing `FileHandle` construction sites instead of adding a logging layer.
- [ ] For `c44`, evict an earlier checkpoint while a later checkpoint retains shared KV storage. Compare charged bytes with live allocations.
- [ ] Test append, capacity growth, branching, disk restoration, and recurrent state independently.
- [ ] Use conservative full charges where backing-storage identity is unavailable. Measure lost cache capacity before adding an ownership ledger.

**Deliverable**

Bounded descriptor and cache lifetimes under repeated requests.
**Stop**

Do not increase retention depth while the byte ledger can undercount.

## C. Establish the current byte and time budget

**Files**

`Tests/MLXFastTests/Model/DecodeBandwidthTests.swift`, `Qwen36MTPHeadCostTests.swift`,
`Sources/MLXFastModel/Qwen36MTPHeadCost.swift`, and existing phase probes.

- [ ] Inventory loaded tensors by family, precision, group size, payload bytes, metadata bytes, and derived copies.
- [ ] Replace the old bf16-dense byte baseline when measuring the current q8 MoE tree.
- [ ] Record KV and recurrent bytes separately from weights, activations, disk mappings, and allocator caches.
- [ ] Measure decode, short prefill, long prefill, and verify widths independently.
- [ ] Report logical-byte estimates separately from measured memory counters. Do not call logical bytes divided by wall time DRAM bandwidth.

Use these bounds only as hypotheses:

```text
serial-work estimate = max(arithmetic / effective_compute, traffic / effective_bandwidth)
                       + unhidden host and dispatch costs
serial speedup ceiling = 1 / ((1 - affected_fraction) + affected_fraction / local_speedup)
split completion       = preparation + max(ANE_leg, GPU_leg) + join
```

Overlapping terms cannot be added without a timeline. Repeated reads and cache hits change the traffic term.
A direct ANE prefix measured 33.89 ms fp16, 18.92 ms int8, and 14.96 ms int4 at S=1024.
That is 2.27 times faster versus fp16 and 1.26 versus int8, not a full-request result.
The dense GPU reference already uses q4. Moving part of it to int4 ANE does not quarter its total weight payload.

**Deliverable**

A current per-model ledger identifying the largest measured cost.
**Stop**

Do not tune a small diagnostic term while a larger unmeasured term remains.

## D. Remove work before changing kernels

**Files**

`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`,
`Models/Qwen4Exp/Qwen4ExpNorms.swift`, `Qwen4ExpAttention.swift`, and `MLXLMCommon/Load.swift`.

- [ ] Trace repeated casts, scalar construction, dequantization, position tables, copies, and synchronization in the current forward.
- [ ] Remove one redundant operation or hoist one immutable value, then compare exact outputs and full-forward latency.
- [ ] Key reusable positional values by all relevant geometry, offsets, dtype, device, and encoding parameters. Test rollback to an earlier offset.
- [ ] Keep query and sparse-indexer position conventions separate where their positions differ.
- [ ] Inspect the loader's global attachment state before enabling concurrent model loads. Preserve scoped restoration on exceptions.

**Deliverable**

A small exact patch with fewer executed operations or bytes.
**Stop**

Reject hoists that grow unbounded caches or need model-specific guesses.

## E. Test selective weight compression

**Files**

`MLXLMCommon/Load.swift`, `Sources/MLXFastModel/Qwen4ExpTransform.swift`,
`Tests/MLXFastTests/Model/DensePerplexityTests.swift`, and transform tests.

- [ ] Use original higher-precision weights when available. Record lineage if requantizing an existing quantized tree.
- [ ] Confirm the current load-time hook actually changes tensors: it skips modules already conforming to `Quantized`.
- [ ] Keep sensitive families at q8 and test q4 on one eligible family at a time. Do not infer this result from blanket q4.
- [ ] Measure payload plus scale/bias overhead, not bit width alone. Check tied embeddings and output projections before altering either.
- [ ] Evaluate held-out NLL, relevant task outcomes, free generation, and full-request latency. Recheck any combined policy.

**Deliverable**

One supported family policy or a documented quality/performance rejection.
**Stop**

No universal one-percent quality allowance. Use the workload's declared gate.
Target-weight changes stay outside a frozen-target ranked track unless its contract explicitly permits them.

## F. Tune GEMV/GEMM by real geometry

**Files**

`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp`,
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`, and existing QMV/tile tests.

- [ ] Capture actual M/N/K, strides, dtype, quantization and selected kernel for decode, prefill, and verification.
- [ ] Benchmark the existing native kernel before the custom Qwen QMV path at each relevant width.
- [ ] Try one tile or split-K choice at a time. Include alignment tails and device threadgroup-memory limits.
- [ ] Check that kernel-cache names include every compile-time specialization. Never use the deliberate collision probe in serving.
- [ ] If changes reach a JIT kernel family, update its runtime-effective generated C++ twin and its source counterpart together.
- [ ] Test another Apple Silicon generation before describing a tile as general. Keep `_nax` dispatch differences explicit.

**Deliverable**

A small shape predicate backed by kernel and full-forward receipts.
**Stop**

No hard-coded Qwen widths in a general helper without an unsupported-shape fallback.

## G. Reduce attention traffic and intermediate storage

**Files**

`MLXLMCommon/AttentionUtils.swift`, `FusedQuantizedSDPA.swift`, `KVCache.swift`,
`Models/Qwen35KVRotation.swift`, `Models/Qwen4Exp/Qwen4ExpAttention.swift`,
and the Metal `scaled_dot_product_attention.cpp` dispatcher.

- [ ] Compare fused native attention with quantized fused attention using identical logical inputs. Include the decomposed path as a control.
- [ ] Test support predicates for packing, bit widths, group alignment, head dimensions, GQA, masks, sinks, and noncontiguous inputs.
- [ ] Audit `FusedQuantizedSDPA.isSupported`: arithmetic divisibility alone does not prove a bit width is supported by its unpacking code.
- [ ] Check bottom-right causal alignment at multiple query rows, unequal key/query lengths, and speculative rejection boundaries.
- [ ] Test KV8 before KV4 at real cache thresholds and 16k/32k contexts where supported. Record both latency and retained bytes.
- [ ] Validate sparse-attention/indexer state and rotation basis through trim, copy, restart, and restore.

**Deliverable**

Lower memory traffic without materializing the attention score matrix where the kernel supports it.
**Stop**

A smaller KV representation that selects a much slower decomposed kernel is not a throughput win.

## H. Tune prefill chunks within memory limits

**Files**

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`, both runtime option definitions,
`Tests/MLXFastTests/Model/QwenPrefillChunkCapTests.swift`, and `QwenPrefillChunkSweepTests.swift`.

- [ ] Record actual forward lengths. A long prompt alone does not prove the pipeline saw a long forward.
- [ ] Compare existing caps 1024, 2048, and 4096 before adding a policy. Try 8192 only if measured headroom permits it.
- [ ] Cover short prompts, 8k, and 32k first. Attempt longer contexts only within the model and memory limits.
- [ ] Verify recurrent state and causal positions across chunk boundaries, with prompt length just below and above each cap.
- [ ] Resolve `39j` with a bounded request-timeout or progress design that still detects a hung worker.

**Deliverable**

A static cap for each tested workload class, with peak memory and cold TTFT.
**Stop**

Do not revive the failed micro-batch restructuring merely because a larger ordinary chunk helps.

## I. Recover the value of speculation

**Files**

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`, `Qwen36MTPTarget.swift`,
`Qwen38DFlash2Head.swift`, `Qwen36MTPHeadAttachment.swift`,
`Qwen4ExpMTPTargetConformance.swift`, and `Sources/MLXFastCore/NGramPromptLookup.swift`.

- [ ] Resolve the q8 MoE head startup hang before measuring native-head speed.
- [ ] Measure serial, forced depth 1, forced depth 2, and current adaptive scheduling with actual accepted tokens per second.
- [ ] Replace transplanted static draft/verify ratios with measured costs for the current model before changing policy.
- [ ] Reproduce hard-to-easy recovery after depth zero using a model-free acceptance trace. Try bounded depth-1 probes or confidence decay.
- [ ] Price repair, head backlog, warmup, and unused final drafts. Include every proposed and rejected row in accounting.
- [ ] Test prompt lookup on grounded editing/retrieval and open prose separately. Keep target verification and stop handling identical.
- [ ] Evaluate cheaper first-position proposals before training a larger head. Keep target weights frozen and tokenizer mapping explicit.
- [ ] Verify full-attention and recurrent rollback separately. An absent replay tape must select generic repair without partial mutation.

**Deliverable**

A measured policy or proposal improvement over both serial and the previous policy on held-out workloads.
**Stop**

Do not target acceptance percentage alone. Do not expose unsupported MTP methods as no-op capabilities for arbitrary models.

## J. Improve routed-expert execution

**Files**

`MLXLMCommon/SwitchLayers.swift`, `FusedRouterSelectKernel.swift`,
`FusedRoutedMoEKernel.swift`, `MoEWorkQueue.swift`, and `Models/Qwen4Exp/Qwen4ExpMoE.swift`.

- [ ] Preserve exact gate/up fusion and test an explicitly unfused control. Verify original expert stacks are released without hiding live copies.
- [ ] Measure router fusion inside a forward at actual token-row geometry, with ties and output-weight mapping checked.
- [ ] Compare routed gather-GEMM against alternatives using the whole batched gather as the denominator.
- [ ] Before rewriting the scalar fused prototype, try native gather plus small activation/reduction fusion. Record intermediate traffic saved.
- [ ] If a tiled prototype is warranted, test skewed routing, hot experts, top-k and tail blocks with actual bf16/fp16 support.
- [ ] Keep expert merging/pruning as separate approximate-model work under `6t8`, not an execution-only optimization.

**Deliverable**

Fewer reads or launches with unchanged routing and reduction semantics.
**Stop**

Dense masking that skips no arithmetic was removed. A 48-times-slower scalar prototype is not a production foundation.

## K. Reduce lookup-table I/O

**Files**

`Models/Qwen4Exp/Qwen4ExpNGram.swift`,
`Sources/MLXFastTransform/NGramTableQuantize.swift`, and n-gram gather/codec tests.

- [ ] Count every page intersecting each record, including records crossing a page boundary. Deduplicate pages across records.
- [ ] Separate logical bytes, pages touched, page faults, physical reads, and warm-cache reads.
- [ ] Re-establish cold residual gather time on the current codec before changing layout.
- [ ] Pilot a small bijective co-occurrence layout on held-out requests before rewriting a multi-gigabyte table.
- [ ] Test bounded prefetch independently. It may hide latency without reducing bytes.
- [ ] Complete codec quality checks with NLL and gap severity. Next-token disagreement is not router disagreement or proof of quality loss.

**Deliverable**

Fewer physical reads or a repeated cold-request latency improvement with bounded memory.
**Stop**

Warm decode's tiny gather time does not justify a large layout project.

## L. Make prefix reuse reliable and effective

**Files**

`QwenSessionCacheStore.swift`, `QwenPrefillDiskCache.swift`,
`Sources/MLXFastTrustedHarness/ServePrefixDecision.swift`, `QwenRuntimeServe.swift`, and worker resume code.

- [ ] Run A/B/C requests sharing a non-stride-aligned prefix, then repeat C after a worker restart.
- [ ] Confirm the exact restored token boundary and every recurrent layer's state, not merely a cache-hit counter.
- [ ] Test branches, edits, cancellation, eviction, corrupt metadata, and tokenizer/weight/cache-policy changes.
- [ ] Measure prefix lookup overhead and retained memory alongside saved prefill time.
- [ ] Simplify overlapping reuse mechanisms only after tracing their consumers. Extension-only reuse and snapshot restore have different guarantees.

**Deliverable**

Lower repeated-request TTFT with correct state after RAM and disk restore.
**Stop**

No reuse based solely on a short-prefix hash or equal token counts.

## M. Share dense work between ANE and GPU

**Files**

`Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/`,
`Models/Qwen4Exp/Qwen4ExpANE*.swift`, and existing direct-prefix probes.

- [ ] Keep the direct dense split as an opt-in local path. Verify actual ANE placement for every new form and package shape.
- [ ] Compare the same split fraction in fp16, int8, and int4, including input staging and output consumption.
- [ ] Use static fractions first. Repeat loaded-box results with achieved load recorded before considering adaptation.
- [ ] Check program capacity and fallback coverage in both quality and speed runs.
- [ ] Reconsider grouped int4 codebooks only against a concrete bandwidth/memory objective and a declared quality budget.
- [ ] Remove additional failed MoE lane integrations only after checking current callers and preserving useful standalone probes.

Dense, non-MoE work is the primary candidate. A model containing MoE layers may still
have substantial dense projections, but each candidate needs its own measured share of layer time.
Do not use the failed expert-partition result to dismiss dense ANE offload.

| Candidate | ANE work | GPU work | Smallest useful experiment |
| --- | --- | --- | --- |
| Fused MLP channel split | Gate, activation, up, and down for one intermediate-channel prefix | Complementary channels and final combination | Reuse `ANEFusedSplitMLP` and compare static fractions at actual sequence buckets. |
| Gate/up-only split | A prefix of the gate and up projections, optionally fused activation | Remaining gate/up work and the complete down projection | Start at one layer. Include transfer of the wide intermediate and any assembly cost. |
| Token-row split | Complete dense operation for one set of prefill rows | The same operation for the other rows | Use token-independent linear/MLP work first. Rejoin rows before attention or recurrence. |
| Dense projection split | A share of a large supported linear projection | Complementary output channels | Compare against the production quantized GPU kernel, including concatenation and padding. |
| Fused dense subgraph | Several adjacent supported operators in one program | Other independent work whose inputs are ready | Price saved dispatches against boundary transfers, state handling, and lost GPU scheduling overlap. |

For gate/up-only offload, the current direct-prefix data makes down-projection cost worth checking.
It does not prove that returning a wider intermediate is cheaper than returning the completed MLP output.
Token-row splitting must preserve row order. Splitting attention or recurrent updates requires a separate dependency proof.
Whole-layer decode offload remains low priority because one-row dispatch and state crossings can dominate.

- [ ] Establish standalone ANE, standalone GPU, and concurrent timings using identical inputs and production weight forms.
- [ ] Measure token-row sizes 128, 256, 512, and 1024 where supported. Compare smaller calls against one larger call with all dispatch costs charged.
- [ ] Reuse direct int8/int4 programs and IOSurface buffers before adding a new transport or package format.
- [ ] Count additional weights, activation staging, padding, and copies. Zero-copy output still consumes shared-memory bandwidth.
- [ ] Trace dependencies to confirm useful overlap. A second thread alone does not prove the engines execute concurrently.
- [ ] Measure cold and warm full-model prefill, then repeat under a controlled GPU load. Record ANE/GPU contention and wall time.
- [ ] Preserve the exact activation and normalization conventions. Test numerical error through the complete subgraph, not just each projection.

Prefer a static split with a measured benefit before introducing a load-dependent policy.
The small combined-bandwidth gain reported in the old probe is a workload observation, not a universal ceiling.
A dense compute-bound subgraph may benefit from the second engine without adding much external-memory bandwidth.

**Deliverable**

Full-request gain with bounded additional memory and acceptable quality.
**Stop**

An isolated projection win is insufficient evidence for a new offload runtime or scheduler.

## N. Bound compaction and simplify serving ownership

**Files**

`Sources/MLXFastTrustedHarness/QwenRuntimeServe.swift`, `CompactionStore.swift`,
`OpenAIPromptRendering.swift`, `OrderedJSON.swift`, and `tools/serve-compactor/`.

- [ ] Trace native compaction and proxy compaction on the same tool conversation. Select one owner per deployment.
- [ ] Check settled-tool detection, expand retrieval, tool-call pairing, streaming, cancellation, and restart behavior.
- [ ] Account for disk retention as well as the in-memory budget. The proxy documents unbounded disk entries.
- [ ] Compare task outcomes and total wall time, including expansion rounds, against the original prompt.
- [ ] Preserve ordered tool argument JSON and model-specific templates. Avoid making tokenizer differences look like runtime speedups.

**Deliverable**

One explicit compaction path with bounded retention and a disable switch.
**Stop**

Compaction changes model input and cannot count as behavior-preserving inference acceleration.

## O. Generalize only validated mechanisms

**Files**

Existing model adapters, `MLXLMCommon` helpers, model factory registration, and their tests.

- [ ] Apply one successful exact mechanism to a second architecture family using its real geometry.
- [ ] Write capability fixtures that cover absent heads and replay tapes. Include differing masks and hidden-state conventions.
- [ ] Test integer boundary inputs in configurable limits. Compare before arithmetic: `depth + 1` can overflow when parsing a maximum integer.
- [ ] Replace the global geometry-unpin escape hatch with explicit local model validation if expanding supported families.
- [ ] Extract only the behavior shared by the two adapters. Keep model-specific norm, cache, routing, and tokenizer rules local.
- [ ] Validate default-off behavior and reference fallback on unsupported geometry.

**Deliverable**

A second-model receipt and a smaller shared helper, if extraction reduces duplication.
**Stop**

Do not build a universal execution engine, autotuner, plugin registry, or benchmark farm before two models need it.

## 6. Validation commands and delivery

Use the existing test target and frozen dependency graph:

```bash
swift test --force-resolved-versions --filter 'ANESplitMLPCacheTests|QwenDepthAndWidthBoundTests'
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions --filter 'ANEFusedSplitMLPTests|SwitchGLUFusedGateUpTests'
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions --filter 'FusedQuantizedSDPA|QwenSessionCacheStore|Qwen4ExpNGram'
swift build -c release --scratch-path .build-worker --force-resolved-versions
```

Select the tests relevant to the change. Inspect skips and environment gates before treating a pass as coverage.
Hardware tests use synthetic inputs unless a test explicitly loads an artifact. Check memory needs first.
Rebuild the Metal library with `tools/build-mlx-metallib.sh` after changing AOT sources.
Use the appropriate model's local benchmark after capturing a fresh unchanged baseline.
For the ranked Qwen track, use `benchmark-qwen-mtp.sh`, not the inherited DFlash wrapper.
Use Yukon for participant operations and the current contract for submission eligibility.

Each completed task produces a small patch, focused tests, a receipt, and a bead update.
Record rejected experiments too. Delete the losing runtime path when no supported use remains.
Do not commit or submit merely because a task in this plan has finished.

## 7. Complete upstream file inventory

The table below maps every committed upstream-delta file to a work area.
Git status A means added. Git status M means modified.
“Removed locally” identifies the subsequent uncommitted cleanup, not an upstream deletion.
Tests and records remain evidence for their area, not independently proven optimizations.

| Status | File | Review disposition / task |
| --- | --- | --- |
| A | [.claude/skills/ane-gpu-offload/SKILL.md](.claude/skills/ane-gpu-offload/SKILL.md) | M / O: local ANE guidance; revalidate hardware assumptions |
| A | [.claude/skills/ane-gpu-offload/references/chip-support.md](.claude/skills/ane-gpu-offload/references/chip-support.md) | M / O: local ANE guidance; revalidate hardware assumptions |
| A | [.claude/skills/ane-gpu-offload/references/fp16.md](.claude/skills/ane-gpu-offload/references/fp16.md) | M / O: local ANE guidance; revalidate hardware assumptions |
| A | [.claude/skills/ane-gpu-offload/references/gpu-interaction.md](.claude/skills/ane-gpu-offload/references/gpu-interaction.md) | M / O: local ANE guidance; revalidate hardware assumptions |
| A | [.claude/skills/ane-gpu-offload/references/int4.md](.claude/skills/ane-gpu-offload/references/int4.md) | M / O: local ANE guidance; revalidate hardware assumptions |
| A | [.claude/skills/ane-gpu-offload/references/int8.md](.claude/skills/ane-gpu-offload/references/int8.md) | M / O: local ANE guidance; revalidate hardware assumptions |
| A | [.claude/skills/ane-gpu-offload/references/operations.md](.claude/skills/ane-gpu-offload/references/operations.md) | M / O: local ANE guidance; revalidate hardware assumptions |
| A | [.claude/skills/ane-gpu-offload/references/program-limit.md](.claude/skills/ane-gpu-offload/references/program-limit.md) | M / O: local ANE guidance; revalidate hardware assumptions |
| A | [.claude/skills/ane-gpu-offload/references/sources.md](.claude/skills/ane-gpu-offload/references/sources.md) | M / O: local ANE guidance; revalidate hardware assumptions |
| A | [.claude/skills/ane-gpu-offload/references/zero-copy.md](.claude/skills/ane-gpu-offload/references/zero-copy.md) | M / O: local ANE guidance; revalidate hardware assumptions |
| M | [.gitignore](.gitignore) | A: artifact hygiene |
| M | [Sources/MLXFastCLI/main.swift](Sources/MLXFastCLI/main.swift) | A / O: protocol and local/trusted bounds |
| M | [Sources/MLXFastCore/Constants.swift](Sources/MLXFastCore/Constants.swift) | A / O: protocol and local/trusted bounds |
| A | [Sources/MLXFastCore/NGramPromptLookup.swift](Sources/MLXFastCore/NGramPromptLookup.swift) | I: proposal, verification, and stop semantics |
| A | [Sources/MLXFastHarness/Qwen4ExpANEBench.swift](Sources/MLXFastHarness/Qwen4ExpANEBench.swift) | M: ANE compute sharing and probes |
| A | [Sources/MLXFastHarness/Qwen4ExpGenerate.swift](Sources/MLXFastHarness/Qwen4ExpGenerate.swift) | O: model-specific adapter and reference parity |
| M | [Sources/MLXFastHarness/QwenRuntime.swift](Sources/MLXFastHarness/QwenRuntime.swift) | A / O: protocol and local/trusted bounds |
| M | [Sources/MLXFastHarness/QwenRuntimeMTPWorker.swift](Sources/MLXFastHarness/QwenRuntimeMTPWorker.swift) | I: proposal, verification, and stop semantics |
| M | [Sources/MLXFastHarness/QwenRuntimeWorker.swift](Sources/MLXFastHarness/QwenRuntimeWorker.swift) | A / O: protocol and local/trusted bounds |
| A | [Sources/MLXFastModel/ANEReexport.swift](Sources/MLXFastModel/ANEReexport.swift) | M: ANE compute sharing and probes |
| M | [Sources/MLXFastModel/Qwen35Config.swift](Sources/MLXFastModel/Qwen35Config.swift) | D / F / O: dense model and geometry guards |
| M | [Sources/MLXFastModel/Qwen35RuntimeWeights.swift](Sources/MLXFastModel/Qwen35RuntimeWeights.swift) | D / F / O: dense model and geometry guards |
| M | [Sources/MLXFastModel/Qwen36MTPBlockSession.swift](Sources/MLXFastModel/Qwen36MTPBlockSession.swift) | B / L: state ownership and prefix reuse |
| M | [Sources/MLXFastModel/Qwen36MTPHeadAttachment.swift](Sources/MLXFastModel/Qwen36MTPHeadAttachment.swift) | I: proposal, verification, and stop semantics |
| A | [Sources/MLXFastModel/Qwen36MTPHeadCost.swift](Sources/MLXFastModel/Qwen36MTPHeadCost.swift) | C: current byte and time budget |
| M | [Sources/MLXFastModel/Qwen36MTPTarget.swift](Sources/MLXFastModel/Qwen36MTPTarget.swift) | I: proposal, verification, and stop semantics |
| A | [Sources/MLXFastModel/Qwen38DFlash2Head.swift](Sources/MLXFastModel/Qwen38DFlash2Head.swift) | I: proposal, verification, and stop semantics |
| A | [Sources/MLXFastModel/Qwen4ExpMTPTargetConformance.swift](Sources/MLXFastModel/Qwen4ExpMTPTargetConformance.swift) | I: proposal, verification, and stop semantics |
| A | [Sources/MLXFastModel/Qwen4ExpTransform.swift](Sources/MLXFastModel/Qwen4ExpTransform.swift) | E / O: weight representation and loading contracts |
| A | [Sources/MLXFastModel/QwenAgreementReport.swift](Sources/MLXFastModel/QwenAgreementReport.swift) | A / C: measurement and attribution |
| A | [Sources/MLXFastModel/QwenPrefillDiskCache.swift](Sources/MLXFastModel/QwenPrefillDiskCache.swift) | B / L: state ownership and prefix reuse |
| A | [Sources/MLXFastModel/QwenSessionCacheStore.swift](Sources/MLXFastModel/QwenSessionCacheStore.swift) | B / L: state ownership and prefix reuse |
| M | [Sources/MLXFastRuntimeWorkerCLI/main.swift](Sources/MLXFastRuntimeWorkerCLI/main.swift) | A / O: protocol and local/trusted bounds |
| A | [Sources/MLXFastTransform/NGramTableQuantize.swift](Sources/MLXFastTransform/NGramTableQuantize.swift) | K: table layout and codec quality |
| M | [Sources/MLXFastTransform/Qwen35CheckpointValidation.swift](Sources/MLXFastTransform/Qwen35CheckpointValidation.swift) | E / O: weight representation and loading contracts |
| M | [Sources/MLXFastTransform/Transform.swift](Sources/MLXFastTransform/Transform.swift) | E / O: weight representation and loading contracts |
| A | [Sources/MLXFastTrustedHarness/CompactionStore.swift](Sources/MLXFastTrustedHarness/CompactionStore.swift) | N: API and compaction behavior |
| A | [Sources/MLXFastTrustedHarness/MinimalHTTPServer.swift](Sources/MLXFastTrustedHarness/MinimalHTTPServer.swift) | N: API and compaction behavior |
| A | [Sources/MLXFastTrustedHarness/OpenAIPromptRendering.swift](Sources/MLXFastTrustedHarness/OpenAIPromptRendering.swift) | N: API and compaction behavior |
| A | [Sources/MLXFastTrustedHarness/OpenAIWireTypes.swift](Sources/MLXFastTrustedHarness/OpenAIWireTypes.swift) | N: API and compaction behavior |
| A | [Sources/MLXFastTrustedHarness/OrderedJSON.swift](Sources/MLXFastTrustedHarness/OrderedJSON.swift) | N: API and compaction behavior |
| M | [Sources/MLXFastTrustedHarness/QwenRuntime.swift](Sources/MLXFastTrustedHarness/QwenRuntime.swift) | A / O: protocol and local/trusted bounds |
| A | [Sources/MLXFastTrustedHarness/QwenRuntimeChat.swift](Sources/MLXFastTrustedHarness/QwenRuntimeChat.swift) | L / N: request lifecycle and input semantics |
| M | [Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift](Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift) | I: proposal, verification, and stop semantics |
| A | [Sources/MLXFastTrustedHarness/QwenRuntimeServe.swift](Sources/MLXFastTrustedHarness/QwenRuntimeServe.swift) | L / N: request lifecycle and input semantics |
| M | [Sources/MLXFastTrustedHarness/QwenRuntimeWorker.swift](Sources/MLXFastTrustedHarness/QwenRuntimeWorker.swift) | A / O: protocol and local/trusted bounds |
| A | [Sources/MLXFastTrustedHarness/ServePrefixDecision.swift](Sources/MLXFastTrustedHarness/ServePrefixDecision.swift) | B / L: state ownership and prefix reuse |
| A | [Tests/MLXFastTests/Model/ANEBF16WeightSourceTests.swift](Tests/MLXFastTests/Model/ANEBF16WeightSourceTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEChannelSplitPoCTests.swift](Tests/MLXFastTests/Model/ANEChannelSplitPoCTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEComputePlanProbeTests.swift](Tests/MLXFastTests/Model/ANEComputePlanProbeTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEDirectDispatchTests.swift](Tests/MLXFastTests/Model/ANEDirectDispatchTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEErrorAttributionTests.swift](Tests/MLXFastTests/Model/ANEErrorAttributionTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEExpertGemmRatioTests.swift](Tests/MLXFastTests/Model/ANEExpertGemmRatioTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEFusedMLPTests.swift](Tests/MLXFastTests/Model/ANEFusedMLPTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEFusedSplitMLPTests.swift](Tests/MLXFastTests/Model/ANEFusedSplitMLPTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEFusedSplitSpeedTests.swift](Tests/MLXFastTests/Model/ANEFusedSplitSpeedTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEGemmBenchTests.swift](Tests/MLXFastTests/Model/ANEGemmBenchTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEGemmTests.swift](Tests/MLXFastTests/Model/ANEGemmTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEInMemoryModelTests.swift](Tests/MLXFastTests/Model/ANEInMemoryModelTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEMetalPartitionTests.swift](Tests/MLXFastTests/Model/ANEMetalPartitionTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEMetalPipelineTests.swift](Tests/MLXFastTests/Model/ANEMetalPipelineTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEMultiFunctionProbeTests.swift](Tests/MLXFastTests/Model/ANEMultiFunctionProbeTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEProcedureBankProbeTests.swift](Tests/MLXFastTests/Model/ANEProcedureBankProbeTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEProgramCountLimitTests.swift](Tests/MLXFastTests/Model/ANEProgramCountLimitTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEQuantizedWeightProbeTests.swift](Tests/MLXFastTests/Model/ANEQuantizedWeightProbeTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANERealActivationCaptureTests.swift](Tests/MLXFastTests/Model/ANERealActivationCaptureTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANERealActivationProbeTests.swift](Tests/MLXFastTests/Model/ANERealActivationProbeTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANERuntimeBridgeTests.swift](Tests/MLXFastTests/Model/ANERuntimeBridgeTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANESplitMLPCacheTests.swift](Tests/MLXFastTests/Model/ANESplitMLPCacheTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANESplitRatioProbeTests.swift](Tests/MLXFastTests/Model/ANESplitRatioProbeTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEStackSpeedTests.swift](Tests/MLXFastTests/Model/ANEStackSpeedTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEWeightPrepTests.swift](Tests/MLXFastTests/Model/ANEWeightPrepTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/ANEZeroCopyReadTests.swift](Tests/MLXFastTests/Model/ANEZeroCopyReadTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/CPUColumnSplitQuantizedMMTests.swift](Tests/MLXFastTests/Model/CPUColumnSplitQuantizedMMTests.swift) | M: heterogeneous compute; require full-path evidence |
| A | `Tests/MLXFastTests/Model/ChannelSplitMLPTests.swift` | Removed locally. Retain historical evidence. |
| A | [Tests/MLXFastTests/Model/CleanGPUBaselineTests.swift](Tests/MLXFastTests/Model/CleanGPUBaselineTests.swift) | A / O: supporting probe, contract, or test |
| A | `Tests/MLXFastTests/Model/CoarseOffloadMLPTests.swift` | Removed locally. Retain historical evidence. |
| A | [Tests/MLXFastTests/Model/ColumnSplitSpeedTests.swift](Tests/MLXFastTests/Model/ColumnSplitSpeedTests.swift) | M: heterogeneous compute; require full-path evidence |
| A | [Tests/MLXFastTests/Model/ConcurrentEnginesTests.swift](Tests/MLXFastTests/Model/ConcurrentEnginesTests.swift) | M: heterogeneous compute; require full-path evidence |
| A | [Tests/MLXFastTests/Model/DecodeBandwidthTests.swift](Tests/MLXFastTests/Model/DecodeBandwidthTests.swift) | C: current byte and time budget |
| A | [Tests/MLXFastTests/Model/DecodeBreakdownTests.swift](Tests/MLXFastTests/Model/DecodeBreakdownTests.swift) | A / C: measurement and attribution |
| A | [Tests/MLXFastTests/Model/DecodeCompileDivergenceTests.swift](Tests/MLXFastTests/Model/DecodeCompileDivergenceTests.swift) | A / C: measurement and attribution |
| A | [Tests/MLXFastTests/Model/DecodeCompileGainTests.swift](Tests/MLXFastTests/Model/DecodeCompileGainTests.swift) | A / C: measurement and attribution |
| A | [Tests/MLXFastTests/Model/DecodeCompileRealTests.swift](Tests/MLXFastTests/Model/DecodeCompileRealTests.swift) | A / C: measurement and attribution |
| A | [Tests/MLXFastTests/Model/DecodeLayerCostTests.swift](Tests/MLXFastTests/Model/DecodeLayerCostTests.swift) | A / C: measurement and attribution |
| A | [Tests/MLXFastTests/Model/DecodeStepCostTests.swift](Tests/MLXFastTests/Model/DecodeStepCostTests.swift) | A / C: measurement and attribution |
| A | [Tests/MLXFastTests/Model/DecodeSyncCensusTests.swift](Tests/MLXFastTests/Model/DecodeSyncCensusTests.swift) | A / C: measurement and attribution |
| A | [Tests/MLXFastTests/Model/DensePerplexityTests.swift](Tests/MLXFastTests/Model/DensePerplexityTests.swift) | E / O: weight representation and loading contracts |
| A | [Tests/MLXFastTests/Model/ExpertGroupSizeTests.swift](Tests/MLXFastTests/Model/ExpertGroupSizeTests.swift) | J: routed experts; separate exact and approximate work |
| A | [Tests/MLXFastTests/Model/ExpertSimilarityTests.swift](Tests/MLXFastTests/Model/ExpertSimilarityTests.swift) | J: routed experts; separate exact and approximate work |
| A | [Tests/MLXFastTests/Model/FusedQuantizedSDPASpeedTests.swift](Tests/MLXFastTests/Model/FusedQuantizedSDPASpeedTests.swift) | G: attention and cache traffic |
| A | [Tests/MLXFastTests/Model/FusedQuantizedSDPATests.swift](Tests/MLXFastTests/Model/FusedQuantizedSDPATests.swift) | G: attention and cache traffic |
| A | [Tests/MLXFastTests/Model/FusedRoutedMoETests.swift](Tests/MLXFastTests/Model/FusedRoutedMoETests.swift) | J: routed experts; separate exact and approximate work |
| A | [Tests/MLXFastTests/Model/FusedRouterSelectTests.swift](Tests/MLXFastTests/Model/FusedRouterSelectTests.swift) | J: routed experts; separate exact and approximate work |
| A | [Tests/MLXFastTests/Model/GPUPositionEffectTests.swift](Tests/MLXFastTests/Model/GPUPositionEffectTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/GatedDeltaScanCostTests.swift](Tests/MLXFastTests/Model/GatedDeltaScanCostTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/GemmChainCostTests.swift](Tests/MLXFastTests/Model/GemmChainCostTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/KVQuantizationPolicyTests.swift](Tests/MLXFastTests/Model/KVQuantizationPolicyTests.swift) | G: attention and cache traffic |
| A | [Tests/MLXFastTests/Model/KVSnapshotAliasingTests.swift](Tests/MLXFastTests/Model/KVSnapshotAliasingTests.swift) | B / L: state ownership and prefix reuse |
| A | [Tests/MLXFastTests/Model/MemoryHeadroomTests.swift](Tests/MLXFastTests/Model/MemoryHeadroomTests.swift) | C: current byte and time budget |
| A | [Tests/MLXFastTests/Model/MoEGatherTileTests.swift](Tests/MLXFastTests/Model/MoEGatherTileTests.swift) | J: routed experts; separate exact and approximate work |
| A | [Tests/MLXFastTests/Model/MoEWorkQueueTests.swift](Tests/MLXFastTests/Model/MoEWorkQueueTests.swift) | J: routed experts; separate exact and approximate work |
| A | [Tests/MLXFastTests/Model/NGramRotationTests.swift](Tests/MLXFastTests/Model/NGramRotationTests.swift) | K: table layout and codec quality |
| A | `Tests/MLXFastTests/Model/OffloadComparisonTests.swift` | Removed locally. Retain historical evidence. |
| A | [Tests/MLXFastTests/Model/OffloadDiagnosticTests.swift](Tests/MLXFastTests/Model/OffloadDiagnosticTests.swift) | M: heterogeneous compute; require full-path evidence |
| A | [Tests/MLXFastTests/Model/PrefillMatmulCostTests.swift](Tests/MLXFastTests/Model/PrefillMatmulCostTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/QmvVariantSweepTests.swift](Tests/MLXFastTests/Model/QmvVariantSweepTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/QuantizedAttentionCostTests.swift](Tests/MLXFastTests/Model/QuantizedAttentionCostTests.swift) | G: attention and cache traffic |
| A | [Tests/MLXFastTests/Model/QuantizedMatmulBatchTests.swift](Tests/MLXFastTests/Model/QuantizedMatmulBatchTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/QuantizedMatmulTileBenchTests.swift](Tests/MLXFastTests/Model/QuantizedMatmulTileBenchTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/QuantizedPackingProbeTests.swift](Tests/MLXFastTests/Model/QuantizedPackingProbeTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/Qwen35ANESplitOffloadTests.swift](Tests/MLXFastTests/Model/Qwen35ANESplitOffloadTests.swift) | M: ANE compute sharing and probes |
| M | [Tests/MLXFastTests/Model/Qwen35ArtifactContractTests.swift](Tests/MLXFastTests/Model/Qwen35ArtifactContractTests.swift) | D / F / O: dense model and geometry guards |
| A | [Tests/MLXFastTests/Model/Qwen35KVRotationTests.swift](Tests/MLXFastTests/Model/Qwen35KVRotationTests.swift) | G: attention and cache traffic |
| A | [Tests/MLXFastTests/Model/Qwen35RotatedAttentionTests.swift](Tests/MLXFastTests/Model/Qwen35RotatedAttentionTests.swift) | G: attention and cache traffic |
| A | [Tests/MLXFastTests/Model/Qwen36MTPHeadCostTests.swift](Tests/MLXFastTests/Model/Qwen36MTPHeadCostTests.swift) | C: current byte and time budget |
| A | [Tests/MLXFastTests/Model/Qwen36MTPHeadPrimingCapTests.swift](Tests/MLXFastTests/Model/Qwen36MTPHeadPrimingCapTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Model/Qwen36MTPHeadStepBenchTests.swift](Tests/MLXFastTests/Model/Qwen36MTPHeadStepBenchTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Model/Qwen38DFlash2ParityTests.swift](Tests/MLXFastTests/Model/Qwen38DFlash2ParityTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Model/Qwen38DFlash2ProfileTests.swift](Tests/MLXFastTests/Model/Qwen38DFlash2ProfileTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Model/Qwen4ExpANEDenseLaneTests.swift](Tests/MLXFastTests/Model/Qwen4ExpANEDenseLaneTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/Qwen4ExpANEExpertLaneTests.swift](Tests/MLXFastTests/Model/Qwen4ExpANEExpertLaneTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/Qwen4ExpANEFusedModeTests.swift](Tests/MLXFastTests/Model/Qwen4ExpANEFusedModeTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/Qwen4ExpANEProgramBudgetTests.swift](Tests/MLXFastTests/Model/Qwen4ExpANEProgramBudgetTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/Qwen4ExpANESharedExpertTests.swift](Tests/MLXFastTests/Model/Qwen4ExpANESharedExpertTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/Qwen4ExpANESplitProjectionTests.swift](Tests/MLXFastTests/Model/Qwen4ExpANESplitProjectionTests.swift) | M: ANE compute sharing and probes |
| A | [Tests/MLXFastTests/Model/Qwen4ExpAttentionTests.swift](Tests/MLXFastTests/Model/Qwen4ExpAttentionTests.swift) | G: attention and cache traffic |
| A | [Tests/MLXFastTests/Model/Qwen4ExpConfigurationTests.swift](Tests/MLXFastTests/Model/Qwen4ExpConfigurationTests.swift) | O: model-specific adapter and reference parity |
| A | [Tests/MLXFastTests/Model/Qwen4ExpGatedDeltaNetTests.swift](Tests/MLXFastTests/Model/Qwen4ExpGatedDeltaNetTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/Qwen4ExpMTPTests.swift](Tests/MLXFastTests/Model/Qwen4ExpMTPTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Model/Qwen4ExpMoETests.swift](Tests/MLXFastTests/Model/Qwen4ExpMoETests.swift) | J: routed experts; separate exact and approximate work |
| A | [Tests/MLXFastTests/Model/Qwen4ExpModelTests.swift](Tests/MLXFastTests/Model/Qwen4ExpModelTests.swift) | O: model-specific adapter and reference parity |
| A | [Tests/MLXFastTests/Model/Qwen4ExpNGramEncodingDivergenceTests.swift](Tests/MLXFastTests/Model/Qwen4ExpNGramEncodingDivergenceTests.swift) | K: table layout and codec quality |
| A | [Tests/MLXFastTests/Model/Qwen4ExpNGramGatherTests.swift](Tests/MLXFastTests/Model/Qwen4ExpNGramGatherTests.swift) | K: table layout and codec quality |
| A | [Tests/MLXFastTests/Model/Qwen4ExpNGramQuantizeTests.swift](Tests/MLXFastTests/Model/Qwen4ExpNGramQuantizeTests.swift) | K: table layout and codec quality |
| A | [Tests/MLXFastTests/Model/Qwen4ExpNGramTests.swift](Tests/MLXFastTests/Model/Qwen4ExpNGramTests.swift) | K: table layout and codec quality |
| A | [Tests/MLXFastTests/Model/Qwen4ExpNormsTests.swift](Tests/MLXFastTests/Model/Qwen4ExpNormsTests.swift) | O: model-specific adapter and reference parity |
| A | [Tests/MLXFastTests/Model/Qwen4ExpParityTests.swift](Tests/MLXFastTests/Model/Qwen4ExpParityTests.swift) | O: model-specific adapter and reference parity |
| A | [Tests/MLXFastTests/Model/Qwen4ExpTransformDenseTests.swift](Tests/MLXFastTests/Model/Qwen4ExpTransformDenseTests.swift) | E / O: weight representation and loading contracts |
| A | [Tests/MLXFastTests/Model/Qwen4ExpTransformGroup64Tests.swift](Tests/MLXFastTests/Model/Qwen4ExpTransformGroup64Tests.swift) | E / O: weight representation and loading contracts |
| A | [Tests/MLXFastTests/Model/Qwen4ExpTransformTests.swift](Tests/MLXFastTests/Model/Qwen4ExpTransformTests.swift) | E / O: weight representation and loading contracts |
| A | [Tests/MLXFastTests/Model/QwenAgreementReportTests.swift](Tests/MLXFastTests/Model/QwenAgreementReportTests.swift) | A / C: measurement and attribution |
| A | [Tests/MLXFastTests/Model/QwenDepthAndWidthBoundTests.swift](Tests/MLXFastTests/Model/QwenDepthAndWidthBoundTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Model/QwenEvalBarrierTests.swift](Tests/MLXFastTests/Model/QwenEvalBarrierTests.swift) | A / O: supporting probe, contract, or test |
| A | [Tests/MLXFastTests/Model/QwenForwardStreamTests.swift](Tests/MLXFastTests/Model/QwenForwardStreamTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/QwenLookupEquivalenceTests.swift](Tests/MLXFastTests/Model/QwenLookupEquivalenceTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Model/QwenLookupWidthCostTests.swift](Tests/MLXFastTests/Model/QwenLookupWidthCostTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Model/QwenMTPSnapshotInvariantTests.swift](Tests/MLXFastTests/Model/QwenMTPSnapshotInvariantTests.swift) | B / L: state ownership and prefix reuse |
| A | [Tests/MLXFastTests/Model/QwenNumericAgreementHarness.swift](Tests/MLXFastTests/Model/QwenNumericAgreementHarness.swift) | A / C: measurement and attribution |
| A | [Tests/MLXFastTests/Model/QwenPhaseBreakdownTests.swift](Tests/MLXFastTests/Model/QwenPhaseBreakdownTests.swift) | A / C: measurement and attribution |
| A | [Tests/MLXFastTests/Model/QwenPrefillChunkCapTests.swift](Tests/MLXFastTests/Model/QwenPrefillChunkCapTests.swift) | H: prefill geometry and memory |
| A | [Tests/MLXFastTests/Model/QwenPrefillChunkSweepTests.swift](Tests/MLXFastTests/Model/QwenPrefillChunkSweepTests.swift) | H: prefill geometry and memory |
| A | [Tests/MLXFastTests/Model/QwenPrefillDiskCacheTests.swift](Tests/MLXFastTests/Model/QwenPrefillDiskCacheTests.swift) | B / L: state ownership and prefix reuse |
| A | [Tests/MLXFastTests/Model/QwenPrefillScalingTests.swift](Tests/MLXFastTests/Model/QwenPrefillScalingTests.swift) | H: prefill geometry and memory |
| A | [Tests/MLXFastTests/Model/QwenRoundBudget.swift](Tests/MLXFastTests/Model/QwenRoundBudget.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Model/QwenRoundBudgetTests.swift](Tests/MLXFastTests/Model/QwenRoundBudgetTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Model/QwenSessionCacheStoreConcurrencyProbe.swift](Tests/MLXFastTests/Model/QwenSessionCacheStoreConcurrencyProbe.swift) | B / L: state ownership and prefix reuse |
| A | [Tests/MLXFastTests/Model/QwenSessionCacheStoreTests.swift](Tests/MLXFastTests/Model/QwenSessionCacheStoreTests.swift) | B / L: state ownership and prefix reuse |
| A | [Tests/MLXFastTests/Model/QwenSessionResumeTests.swift](Tests/MLXFastTests/Model/QwenSessionResumeTests.swift) | B / L: state ownership and prefix reuse |
| A | [Tests/MLXFastTests/Model/QwenVerifyDepthCapTests.swift](Tests/MLXFastTests/Model/QwenVerifyDepthCapTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Model/QwenWidthWallProbeTests.swift](Tests/MLXFastTests/Model/QwenWidthWallProbeTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Model/RMSNormScaleHoistTests.swift](Tests/MLXFastTests/Model/RMSNormScaleHoistTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/RotationScaleUniformityTests.swift](Tests/MLXFastTests/Model/RotationScaleUniformityTests.swift) | G: attention and cache traffic |
| A | [Tests/MLXFastTests/Model/RouterCostTests.swift](Tests/MLXFastTests/Model/RouterCostTests.swift) | J: routed experts; separate exact and approximate work |
| A | [Tests/MLXFastTests/Model/SMEGemmCostTests.swift](Tests/MLXFastTests/Model/SMEGemmCostTests.swift) | M: heterogeneous compute; require full-path evidence |
| A | [Tests/MLXFastTests/Model/SameShapeChainTests.swift](Tests/MLXFastTests/Model/SameShapeChainTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/ShapeSwitchTests.swift](Tests/MLXFastTests/Model/ShapeSwitchTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/StreamOrDeviceFactoryTests.swift](Tests/MLXFastTests/Model/StreamOrDeviceFactoryTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/Model/SwitchGLUFusedGateUpTests.swift](Tests/MLXFastTests/Model/SwitchGLUFusedGateUpTests.swift) | J: routed experts; separate exact and approximate work |
| A | [Tests/MLXFastTests/Model/WideVerifyLadderBandTests.swift](Tests/MLXFastTests/Model/WideVerifyLadderBandTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Model/WideVerifyMatmulTileTests.swift](Tests/MLXFastTests/Model/WideVerifyMatmulTileTests.swift) | D / F: operation reuse and kernel dispatch |
| A | [Tests/MLXFastTests/NGramPromptLookupTests.swift](Tests/MLXFastTests/NGramPromptLookupTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/QwenLookupCorpusTests.swift](Tests/MLXFastTests/QwenLookupCorpusTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/QwenLookupDraftDecisionTests.swift](Tests/MLXFastTests/QwenLookupDraftDecisionTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/QwenLookupSessionSurfaceTests.swift](Tests/MLXFastTests/QwenLookupSessionSurfaceTests.swift) | B / L: state ownership and prefix reuse |
| A | [Tests/MLXFastTests/QwenMTPHostQoSPolicyTests.swift](Tests/MLXFastTests/QwenMTPHostQoSPolicyTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/QwenMTPRoundTraceSummaryTests.swift](Tests/MLXFastTests/QwenMTPRoundTraceSummaryTests.swift) | I: proposal, verification, and stop semantics |
| M | [Tests/MLXFastTests/QwenMTPVerbTests.swift](Tests/MLXFastTests/QwenMTPVerbTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Serve/CompactionRetrievalTests.swift](Tests/MLXFastTests/Serve/CompactionRetrievalTests.swift) | N: API and compaction behavior |
| A | [Tests/MLXFastTests/Serve/MinimalHTTPServerTests.swift](Tests/MLXFastTests/Serve/MinimalHTTPServerTests.swift) | N: API and compaction behavior |
| A | [Tests/MLXFastTests/Serve/OpenAIPromptRenderingTests.swift](Tests/MLXFastTests/Serve/OpenAIPromptRenderingTests.swift) | N: API and compaction behavior |
| A | [Tests/MLXFastTests/Serve/OrderedJSONTests.swift](Tests/MLXFastTests/Serve/OrderedJSONTests.swift) | N: API and compaction behavior |
| A | [Tests/MLXFastTests/Serve/PrefixReuseTests.swift](Tests/MLXFastTests/Serve/PrefixReuseTests.swift) | B / L: state ownership and prefix reuse |
| A | [Tests/MLXFastTests/Serve/QwenIncrementalDecodeTests.swift](Tests/MLXFastTests/Serve/QwenIncrementalDecodeTests.swift) | A / C: measurement and attribution |
| A | [Tests/MLXFastTests/Serve/QwenMTPSamplingTests.swift](Tests/MLXFastTests/Serve/QwenMTPSamplingTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Serve/QwenServeRoundTailTests.swift](Tests/MLXFastTests/Serve/QwenServeRoundTailTests.swift) | I: proposal, verification, and stop semantics |
| A | [Tests/MLXFastTests/Serve/QwenServeRoundTextTests.swift](Tests/MLXFastTests/Serve/QwenServeRoundTextTests.swift) | I: proposal, verification, and stop semantics |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEActivationCapture.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEActivationCapture.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEBF16WeightSource.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEBF16WeightSource.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEDirectDispatch.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEDirectDispatch.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEFusedMLP.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEFusedMLP.swift) | M: ANE compute sharing and probes |
| A | `Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEFusedMLPBank.swift` | Removed locally. Retain historical evidence. |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEFusedSplitMLP.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEFusedSplitMLP.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEGemm.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEGemm.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEGemmBench.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEGemmBench.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEGroupedExpertMLP.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEGroupedExpertMLP.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEInMemoryModel.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEInMemoryModel.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEMILBuilder.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEMILBuilder.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANERuntimeBridge.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANERuntimeBridge.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEWeightPrep.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEWeightPrep.swift) | M: ANE compute sharing and probes |
| A | `Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ChannelSplitMLP.swift` | Removed locally. Retain historical evidence. |
| A | `Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/CoarseOffloadMLP.swift` | Removed locally. Retain historical evidence. |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ConcurrentEngines.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ConcurrentEngines.swift) | M: heterogeneous compute; require full-path evidence |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/Qwen35ANESplitOffload.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/Qwen35ANESplitOffload.swift) | M: ANE compute sharing and probes |
| M | [Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift) | A / O: supporting probe, contract, or test |
| M | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift) | D / F / O: dense model and geometry guards |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35KVRotation.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35KVRotation.swift) | G: attention and cache traffic |
| M | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift) | I: proposal, verification, and stop semantics |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpANEDenseLane.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpANEDenseLane.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpANEExpertLane.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpANEExpertLane.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpANEFusedMode.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpANEFusedMode.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpANESharedExpertLane.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpANESharedExpertLane.swift) | M: ANE compute sharing and probes |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpAttention.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpAttention.swift) | G: attention and cache traffic |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpConfiguration.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpConfiguration.swift) | O: model-specific adapter and reference parity |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpGatedDeltaNet.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpGatedDeltaNet.swift) | D / F: operation reuse and kernel dispatch |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpMTP.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpMTP.swift) | I: proposal, verification, and stop semantics |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpMoE.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpMoE.swift) | J: routed experts; separate exact and approximate work |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpModel.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpModel.swift) | O: model-specific adapter and reference parity |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpNGram.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpNGram.swift) | K: table layout and codec quality |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpNorms.swift](Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp/Qwen4ExpNorms.swift) | O: model-specific adapter and reference parity |
| M | [Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift](Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift) | G: attention and cache traffic |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLMCommon/FusedQuantizedSDPA.swift](Vendor/mlx-swift-lm/Libraries/MLXLMCommon/FusedQuantizedSDPA.swift) | G: attention and cache traffic |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLMCommon/FusedRoutedMoEKernel.swift](Vendor/mlx-swift-lm/Libraries/MLXLMCommon/FusedRoutedMoEKernel.swift) | J: routed experts; separate exact and approximate work |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLMCommon/FusedRouterSelectKernel.swift](Vendor/mlx-swift-lm/Libraries/MLXLMCommon/FusedRouterSelectKernel.swift) | J: routed experts; separate exact and approximate work |
| M | [Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift](Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift) | G: attention and cache traffic |
| M | [Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift](Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift) | E / O: weight representation and loading contracts |
| A | [Vendor/mlx-swift-lm/Libraries/MLXLMCommon/MoEWorkQueue.swift](Vendor/mlx-swift-lm/Libraries/MLXLMCommon/MoEWorkQueue.swift) | J: routed experts; separate exact and approximate work |
| M | [Vendor/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift](Vendor/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift) | J: routed experts; separate exact and approximate work |
| M | [Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp](Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp) | D / F: operation reuse and kernel dispatch |
| M | [Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp](Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp) | G: attention and cache traffic |
| M | [Vendor/mlx-swift/Source/MLX/Stream.swift](Vendor/mlx-swift/Source/MLX/Stream.swift) | D / F: operation reuse and kernel dispatch |
| A | [docs/compute-engine-map.md](docs/compute-engine-map.md) | A: historical evidence; use later corrections |
| A | [docs/dflash2-head-port-plan.md](docs/dflash2-head-port-plan.md) | A: historical evidence; use later corrections |
| A | [docs/perf/accept-runs/fixed-2026-09-02/golden-dequant-README.json](docs/perf/accept-runs/fixed-2026-09-02/golden-dequant-README.json) | A: historical evidence; use later corrections |
| A | [docs/perf/accept-runs/fixed-2026-09-02/golden-gpu-README.json](docs/perf/accept-runs/fixed-2026-09-02/golden-gpu-README.json) | A: historical evidence; use later corrections |
| A | [docs/perf/accept-runs/fixed-2026-09-02/mtp-accept-20260902-182411.csv](docs/perf/accept-runs/fixed-2026-09-02/mtp-accept-20260902-182411.csv) | A: historical evidence; use later corrections |
| A | [docs/perf/accept-runs/fixed-2026-09-02/score-gpu-README-d2.json](docs/perf/accept-runs/fixed-2026-09-02/score-gpu-README-d2.json) | A: historical evidence; use later corrections |
| A | [docs/perf/accept-runs/fixed-2026-09-02/score-gpu-README-d8.json](docs/perf/accept-runs/fixed-2026-09-02/score-gpu-README-d8.json) | A: historical evidence; use later corrections |
| A | [docs/perf/accept-runs/mtp-accept-20260901-225614.csv](docs/perf/accept-runs/mtp-accept-20260901-225614.csv) | A: historical evidence; use later corrections |
| A | [docs/perf/ane-gpu-reevaluation-2026-09-06.md](docs/perf/ane-gpu-reevaluation-2026-09-06.md) | A: historical evidence; use later corrections |
| A | [docs/perf/ane-hybrid-collapse-2026-09.md](docs/perf/ane-hybrid-collapse-2026-09.md) | A: historical evidence; use later corrections |
| A | [docs/perf/ane-unified-activation-plan.md](docs/perf/ane-unified-activation-plan.md) | A: historical evidence; use later corrections |
| A | [docs/perf/ane-unified-activation-spec.md](docs/perf/ane-unified-activation-spec.md) | A: historical evidence; use later corrections |
| A | [docs/perf/ane-zero-transfer-spec.md](docs/perf/ane-zero-transfer-spec.md) | A: historical evidence; use later corrections |
| A | [docs/perf/bead-close-out-2026-09-02.md](docs/perf/bead-close-out-2026-09-02.md) | A: historical evidence; use later corrections |
| A | [docs/perf/closed-bead-review-2026-09-06.md](docs/perf/closed-bead-review-2026-09-06.md) | A: historical evidence; use later corrections |
| A | [docs/perf/decode-methods-2026-09.md](docs/perf/decode-methods-2026-09.md) | A: historical evidence; use later corrections |
| A | [docs/perf/decode-profile-m4max-2026-09.md](docs/perf/decode-profile-m4max-2026-09.md) | A: historical evidence; use later corrections |
| A | [docs/perf/decode-round-trace-m4max-2026-09.md](docs/perf/decode-round-trace-m4max-2026-09.md) | A: historical evidence; use later corrections |
| A | [docs/perf/mlx-serve-attribution-2026-09-06.md](docs/perf/mlx-serve-attribution-2026-09-06.md) | A: historical evidence; use later corrections |
| A | [docs/perf/mtp-accept-matrix-2026-09.md](docs/perf/mtp-accept-matrix-2026-09.md) | A: historical evidence; use later corrections |
| A | [docs/perf/qwen38-flash-2026-09.md](docs/perf/qwen38-flash-2026-09.md) | A: historical evidence; use later corrections |
| A | [docs/perf/raw-phase-off.txt](docs/perf/raw-phase-off.txt) | A: historical evidence; use later corrections |
| A | [docs/perf/raw-phase-replica.txt](docs/perf/raw-phase-replica.txt) | A: historical evidence; use later corrections |
| A | [docs/perf/raw-phase-sumTable.txt](docs/perf/raw-phase-sumTable.txt) | A: historical evidence; use later corrections |
| A | [docs/perf/raw-round-trace-2026-09-02-arm-A.log](docs/perf/raw-round-trace-2026-09-02-arm-A.log) | A: historical evidence; use later corrections |
| A | [docs/perf/raw-round-trace-2026-09-02-arm-B.log](docs/perf/raw-round-trace-2026-09-02-arm-B.log) | A: historical evidence; use later corrections |
| A | [docs/perf/raw-round-trace-2026-09-02-arm-C.log](docs/perf/raw-round-trace-2026-09-02-arm-C.log) | A: historical evidence; use later corrections |
| A | [docs/perf/raw-round-trace-2026-09-02-arm-D.log](docs/perf/raw-round-trace-2026-09-02-arm-D.log) | A: historical evidence; use later corrections |
| A | [docs/perf/raw-round-trace-2026-09-02-arm-E.log](docs/perf/raw-round-trace-2026-09-02-arm-E.log) | A: historical evidence; use later corrections |
| A | [docs/perf/raw-round-trace-2026-09-02-arm-F.log](docs/perf/raw-round-trace-2026-09-02-arm-F.log) | A: historical evidence; use later corrections |
| A | [docs/perf/raw-round-trace-2026-09-02-qos.log](docs/perf/raw-round-trace-2026-09-02-qos.log) | A: historical evidence; use later corrections |
| A | [docs/perf/raw-round-trace-2026-09-02-qos.macmon.jsonl](docs/perf/raw-round-trace-2026-09-02-qos.macmon.jsonl) | A: historical evidence; use later corrections |
| A | [docs/perf/raw-round-trace-2026-09-02.log](docs/perf/raw-round-trace-2026-09-02.log) | A: historical evidence; use later corrections |
| A | [docs/perf/raw-round-trace-2026-09-02.macmon.jsonl](docs/perf/raw-round-trace-2026-09-02.macmon.jsonl) | A: historical evidence; use later corrections |
| A | [docs/qwen-prefill-research-plan.md](docs/qwen-prefill-research-plan.md) | A: historical evidence; use later corrections |
| M | [setup-dflash.sh](setup-dflash.sh) | I: proposal, verification, and stop semantics |
| A | [tools/ane-conv/main.swift](tools/ane-conv/main.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-conv/proto.swift](tools/ane-conv/proto.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-direct/probe_clients.c](tools/ane-direct/probe_clients.c) | M: ANE compute sharing and probes |
| A | [tools/ane-direct/probe_dlopen.c](tools/ane-direct/probe_dlopen.c) | M: ANE compute sharing and probes |
| A | [tools/ane-direct/probe_iokit.c](tools/ane-direct/probe_iokit.c) | M: ANE compute sharing and probes |
| A | [tools/ane-direct/probe_sel.c](tools/ane-direct/probe_sel.c) | M: ANE compute sharing and probes |
| A | [tools/ane-fraction-sweep.sh](tools/ane-fraction-sweep.sh) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/chunk.swift](tools/ane-gated-delta/chunk.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/chunk2.swift](tools/ane-gated-delta/chunk2.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/harness.swift](tools/ane-gated-delta/harness.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/layer.swift](tools/ane-gated-delta/layer.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/layer2.swift](tools/ane-gated-delta/layer2.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/main.swift](tools/ane-gated-delta/main.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/probe2.swift](tools/ane-gated-delta/probe2.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/probe3.swift](tools/ane-gated-delta/probe3.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/probe4.swift](tools/ane-gated-delta/probe4.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/probe5.swift](tools/ane-gated-delta/probe5.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/probe6.swift](tools/ane-gated-delta/probe6.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/probe7.swift](tools/ane-gated-delta/probe7.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/probe8.swift](tools/ane-gated-delta/probe8.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/probe_ops.swift](tools/ane-gated-delta/probe_ops.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/proto.swift](tools/ane-gated-delta/proto.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-gated-delta/savechunk.swift](tools/ane-gated-delta/savechunk.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-hwx/compile_probe.m](tools/ane-hwx/compile_probe.m) | M: ANE compute sharing and probes |
| A | [tools/ane-layer/layer.swift](tools/ane-layer/layer.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-layer/main.swift](tools/ane-layer/main.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-layer/proto.swift](tools/ane-layer/proto.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-load/load_probe.m](tools/ane-load/load_probe.m) | M: ANE compute sharing and probes |
| A | [tools/ane-maderix/dump_ane.m](tools/ane-maderix/dump_ane.m) | M: ANE compute sharing and probes |
| A | [tools/ane-maderix/inmem_probe.m](tools/ane-maderix/inmem_probe.m) | M: ANE compute sharing and probes |
| A | [tools/ane-maderix/main.swift](tools/ane-maderix/main.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-maderix/proto.swift](tools/ane-maderix/proto.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-mlprogram/main.swift](tools/ane-mlprogram/main.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-mlprogram/proto.swift](tools/ane-mlprogram/proto.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-native/main.swift](tools/ane-native/main.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-probes/README.md](tools/ane-probes/README.md) | M: ANE compute sharing and probes |
| A | [tools/ane-probes/agree.py](tools/ane-probes/agree.py) | M: ANE compute sharing and probes |
| A | [tools/ane-probes/cleanup-run.sh](tools/ane-probes/cleanup-run.sh) | M: ANE compute sharing and probes |
| A | [tools/ane-probes/gates.sh](tools/ane-probes/gates.sh) | M: ANE compute sharing and probes |
| A | [tools/ane-probes/gen_fused_bank.py](tools/ane-probes/gen_fused_bank.py) | M: ANE compute sharing and probes |
| A | [tools/ane-probes/gen_layer_probe.py](tools/ane-probes/gen_layer_probe.py) | M: ANE compute sharing and probes |
| A | [tools/ane-probes/gen_lut_group_probe.py](tools/ane-probes/gen_lut_group_probe.py) | M: ANE compute sharing and probes |
| A | [tools/ane-probes/gen_probe_pkgs.py](tools/ane-probes/gen_probe_pkgs.py) | M: ANE compute sharing and probes |
| A | [tools/ane-probes/gen_r_pkgs.py](tools/ane-probes/gen_r_pkgs.py) | M: ANE compute sharing and probes |
| A | [tools/ane-probes/gpu_load.swift](tools/ane-probes/gpu_load.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-probes/serve-sweep.sh](tools/ane-probes/serve-sweep.sh) | M: ANE compute sharing and probes |
| A | [tools/ane-swift/main.swift](tools/ane-swift/main.swift) | M: ANE compute sharing and probes |
| A | [tools/ane-swift/run-with-power.sh](tools/ane-swift/run-with-power.sh) | M: ANE compute sharing and probes |
| A | [tools/column-split-sweep.sh](tools/column-split-sweep.sh) | M: heterogeneous compute; require full-path evidence |
| A | [tools/cpu-sme-lane/gpu_burn.swift](tools/cpu-sme-lane/gpu_burn.swift) | M: heterogeneous compute; require full-path evidence |
| A | [tools/cpu-sme-lane/gpu_matmul.metal](tools/cpu-sme-lane/gpu_matmul.metal) | M: heterogeneous compute; require full-path evidence |
| A | [tools/cpu-sme-lane/sme2_kernel.s](tools/cpu-sme-lane/sme2_kernel.s) | M: heterogeneous compute; require full-path evidence |
| A | [tools/cpu-sme-lane/sme_peak.c](tools/cpu-sme-lane/sme_peak.c) | M: heterogeneous compute; require full-path evidence |
| A | [tools/dflash2/README.md](tools/dflash2/README.md) | I: proposal, verification, and stop semantics |
| A | [tools/dflash2/dump_dflash2_fixture.py](tools/dflash2/dump_dflash2_fixture.py) | I: proposal, verification, and stop semantics |
| A | [tools/dflash2/quantize_dflash2.py](tools/dflash2/quantize_dflash2.py) | I: proposal, verification, and stop semantics |
| A | [tools/gemm-point-sweep.sh](tools/gemm-point-sweep.sh) | D / F: operation reuse and kernel dispatch |
| A | [tools/host-quiet-gate.sh](tools/host-quiet-gate.sh) | A / O: supporting probe, contract, or test |
| A | [tools/int8-probe/main.swift](tools/int8-probe/main.swift) | A / O: protocol and local/trusted bounds |
| A | [tools/int8-probe/proto.swift](tools/int8-probe/proto.swift) | A / O: supporting probe, contract, or test |
| A | [tools/kvquant-ab-speed.sh](tools/kvquant-ab-speed.sh) | G: attention and cache traffic |
| A | [tools/kvquant-fidelity.sh](tools/kvquant-fidelity.sh) | G: attention and cache traffic |
| A | [tools/mtp-accept-summary.py](tools/mtp-accept-summary.py) | I: proposal, verification, and stop semantics |
| A | [tools/mtp-accept-sweep.sh](tools/mtp-accept-sweep.sh) | I: proposal, verification, and stop semantics |
| A | [tools/mtp-round-trace-summary.sh](tools/mtp-round-trace-summary.sh) | I: proposal, verification, and stop semantics |
| A | [tools/overlap/main.swift](tools/overlap/main.swift) | M: ANE compute sharing and probes |
| A | [tools/overlap/proto.swift](tools/overlap/proto.swift) | M: ANE compute sharing and probes |
| A | [tools/position-effect-sweep.sh](tools/position-effect-sweep.sh) | D / F: operation reuse and kernel dispatch |
| A | [tools/qwen38-flash/download.sh](tools/qwen38-flash/download.sh) | O: model-specific adapter and reference parity |
| A | [tools/qwen38-flash/manifest.json](tools/qwen38-flash/manifest.json) | O: model-specific adapter and reference parity |
| A | [tools/qwen38-flash/parity.py](tools/qwen38-flash/parity.py) | O: model-specific adapter and reference parity |
| A | [tools/qwen38-flash/qwen4_exp.py](tools/qwen38-flash/qwen4_exp.py) | O: model-specific adapter and reference parity |
| A | [tools/serve-compactor/.gitignore](tools/serve-compactor/.gitignore) | N: API and compaction behavior |
| A | [tools/serve-compactor/README.md](tools/serve-compactor/README.md) | N: API and compaction behavior |
| A | [tools/serve-compactor/compactor.py](tools/serve-compactor/compactor.py) | N: API and compaction behavior |
| A | [tools/serve-compactor/test_compactor.py](tools/serve-compactor/test_compactor.py) | N: API and compaction behavior |

Inventory check: 343 committed upstream-delta paths, each listed once.
