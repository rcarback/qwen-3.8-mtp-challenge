---
name: mlxfast
description: Use when improving MLX inference speed or memory on Apple Silicon, or reviewing performance claims involving Metal kernels, quantization, KV caches, speculation, or ANE/GPU sharing.
---

# mlxfast

Optimize the requested workload while preserving its required model behavior.
Prefer deleting work, then simplifying, then tuning. A smaller model, a faster
kernel, and a faster request are different results. Name the result you measured.

## Order of work and completion checks

Use this sequence for optimization work. For a review, inspect existing evidence
against these checks. A plan can specify a missing measurement without claiming
that the check has passed.

| Step | Work | Completion check |
| --- | --- | --- |
| 1. Boundary | Identify model, hardware, workload, baseline, objective, and permitted changes | State what may ship and the scoring/correctness contract. Distinguish local serving from ranked submissions |
| 2. Instruments | Read [measurement](references/measurement.md) for experiments and performance reviews. Verify both arms | Evidence distinguishes execution from eligibility and reports fallback. Add diagnostics where existing evidence is insufficient |
| 3. Budget | Identify limiting time, bytes, or capacity at relevant phases and shapes | Name the measured cost and opportunity bound, or the missing measurement. Use [budget](references/budget.md) |
| 4. Gate | Classify each component using the table below | Name required fidelity checks and any permitted quality budget before tuning |
| 5. Remove work | Delete redundant operations or simplify data flow before replacing kernels | Verify required behavior and the claimed reduction in work, memory, or complexity. Speed claims also need timing |
| 6. Targeted work | Select applicable mechanisms using the capability map. Compare production kernels at real geometry | Validate supported shapes and fallback boundaries. Match evidence to the claimed scope |
| 7. Decision | Apply the evidence rules and review checklist | Record accepted, rejected, inconclusive, or blocked, with reasons and the fate of experimental code |

In this repository, the current manifest overrides inherited track prose.

## Change and correctness

| Change | Required check |
| --- | --- |
| Execution, layout, fusion, scheduling, exact state reuse | Required numerical and token-fidelity contract, plus state and boundary tests. Floating-point reassociation does not guarantee bit equality. |
| Target weight or state approximation | Deployment permission and a declared quality budget. Assess held-out NLL and behavior as well as reconstruction error. |
| Proposal-only draft change | Correct target verification, sampling, and rollback. Measure acceptance and total cost without treating proposal error as target quality loss. |
| Changed input or product behavior | Task outcomes and total request cost against the original input. Label prompt reduction separately from inference throughput. |

Exact equality remains mandatory wherever the contract requires it. A permissive
quality budget cannot replace a frozen-target correctness gate. Changing kernels
or engines does not by itself change the target weight representation.

## Evidence and decision

- Label evidence as an estimate, isolated measurement, full-forward measurement,
  or deployment result. A microbenchmark supports a kernel claim, not a request speedup.
- Verify the executed path after enable flags and support guards. Eligibility,
  successful compilation, and preferred device placement alone do not prove execution.
- Use repeated paired timings with controlled machine state. Separate cold startup,
  warm operation, and intentional contention. Match token counts and cache state.
- Keep a deterministic shape-gated win with a tested fallback for losing shapes.
  Remove rejected experimental code with no supported use. Inconclusive timing is
  not a speedup. Simpler or smaller code can still justify itself.
- Use the answer contract below. Keep field summaries short and link detailed evidence.

## Answer contract

For optimization plans, experiment reports, and performance reviews, cover these
nine fields in this order. A compact table or labeled sentences are sufficient.
Detailed measurements can live in a linked record, but summarize each field here.

| Field | Required content |
| --- | --- |
| Boundary | Objective, permitted changes, target hardware/software, and applicable contract |
| Budget | Largest relevant measured cost and opportunity bound. Distinguish estimates from observations |
| Kind and gate | Classification per changed component and required correctness or quality checks |
| Arms | Control/treatment, pair count and order, workload, cache/thermal/load conditions, and observed variability |
| Geometry | Actual shapes, query widths, context lengths, precision, and selected kernel or engine relevant to the claim |
| Engagement | Evidence that treatment executed and control did not, including fallback checks where applicable |
| Confounds | Other changed variables, timing boundaries/build mode, and unresolved attribution such as prefetch versus byte reduction |
| Decision | Accept, reject, inconclusive, or blocked. State resource limits, supported regimes, and what code stays or leaves |
| Receipt | Link to evidence with revision/artifact identity, reproduction command, timing, and required correctness results |

For plans, mark unrun fields **planned** or **not measured** and specify how to
obtain the evidence. In reviews, missing evidence is a finding. Use **not applicable**
with a reason for fields outside the task. Missing timings block a speed claim,
not a justified simplification. A follow-up about one field can answer that field
without repeating the full report. User-requested formats take precedence.

## Capability map and reference routing

Choose by the model's actual capabilities. Read only the references needed for
the task. Detailed adapter checks remain in [generalizing](references/generalizing.md).

| Capability or task | Candidate work and prerequisite | Reference |
| --- | --- | --- |
| Benchmark or speed review | Comparable arms, actual execution, and the applicable fidelity contract | [Measurement](references/measurement.md) |
| Unknown bottleneck | Current time/byte inventory and workload-specific opportunity bound | [Budget](references/budget.md) |
| Repeated forward work | Reuse values within valid input, state, and lifetime boundaries | [Remove work](references/remove-work.md) |
| Dense linear layers | GEMV/GEMM and packing. Check geometry, dtype, strides, quantization, and chip dispatch | [Kernels](references/kernels.md) |
| Explicit full/windowed attention KV | Fusion, precision, and layout. Check heads, masks, sinks, positions, and windows | [Attention and KV](references/attention-kv.md) |
| Chunkable prefill, including hybrid layers | Bound memory and preserve causal positions and recurrent state across chunks | [Prefill chunking](references/prefill-chunking.md) |
| Eligible weight families | Quantization with supported kernels, tied-consumer consistency, and a permitted quality budget | [Weight compression](references/weight-compression.md) |
| Compatible proposal source | Check target verification, rollback, and tokenizer mapping. Hidden-state drafters also need matching hidden-state conventions | [Speculation](references/speculation.md) |
| Routed experts | Gather/fusion with matching top-k, routing ties, reduction, quantization, and skew | [Routed experts](references/routed-experts.md) |
| Large indexed tables | Page layout and prefetch. Check encoding, index semantics, and cache residency | [Lookup tables](references/lookup-tables.md) |
| Repeated token prefixes | Reuse with complete state identity, valid restore boundaries, and bounded retention | [Prefix reuse](references/prefix-reuse.md) |
| Dense subgraph and available ANE, including dense work inside MoE | Supported representation, actual placement, independent work, and measured join cost | [Engine offload](references/engine-offload.md) |
| Authorized prompt reduction | Compaction with explicit ownership, recovery policy, and task-quality checks | [Serving compaction](references/serving-compaction.md) |
| Another model needs the mechanism | Adapter/capability checks and a separate correctness and performance result | [Generalizing](references/generalizing.md) |
| Historical claim needs checking | Locate the source and distinguish its measured scope from inference | [Pitfalls](references/pitfalls.md) |

A model without a draft source, ANE, or recurrent replay capability must decline
that path or use a valid fallback. Do not invent support to fit the workflow.

## Review checklist

Before recommending acceptance of a performance claim, check:

- [ ] The objective, baseline, delivery boundary, and timed window match the claim.
- [ ] Geometry and dispatched kernels/engines match the target workload and hardware.
- [ ] Control and treatment demonstrably differ after all flags and support guards.
- [ ] Repeated comparable arms report order, variability, cache state, and machine conditions.
- [ ] Required fidelity checks pass, with approximation and proposal-only gates kept distinct.
- [ ] Byte reduction, overlap, startup, and other changed variables have separate attribution.
- [ ] End-to-end evidence supports an end-to-end claim, and resource/tail-latency budgets hold.
- [ ] The decision handles unsupported or losing shapes and identifies retained or deleted code.
- [ ] The answer covers all nine fields and links reproducible evidence or states what is missing.

An unchecked item is a review finding. It blocks only the claim that depends on it.

References contain dated observations from this fork. Recheck source symbols and
current contracts before applying them elsewhere. Paths in backticks beginning
`Sources/`, `Vendor/`, `Tests/`, `tools/`, or `docs/` are repository-root relative.
Missing historical evidence is an evidence gap, not permission to invent a result.
