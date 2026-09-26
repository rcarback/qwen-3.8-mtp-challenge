# Historical pitfalls and evidence

These observations come from this fork's August–September 2026 work, mainly on an
M4 Max with 128 GB. They explain questions to ask, not universal hardware rules.
Source comments, estimates, isolated probes, and request measurements have different
strength. Check the cited record before reusing a number.

## Failure patterns

| Finding | Evidence | Lesson |
| --- | --- | --- |
| Program identity collision | `docs/perf/ane-hybrid-collapse-2026-09.md`: per-layer checks missed whole-model repetition when layer programs shared identity | Test distinct layer weights and whole-model health after program-loading changes. |
| Silent ANE fallback | `docs/perf/ane-gpu-reevaluation-2026-09-06.md`: disk-full and quantized-module guards invalidated early arms | Record completed treatment work and partial fallback, not only program creation. |
| Two fused controls | Bead `dyv`, `docs/perf/upstream-cleanup-2026-09-07.md` | Assert the effective control state before its first forward. |
| Eligibility mistaken for dispatch | `AttentionUtils.swift` counts support before the feature-enable guard | Place execution evidence after all dispatch guards. |
| Sweep order reversed a winner | Commit `b2833214`: ascending and descending chunk sweeps disagreed | Counterbalance order and treat small unresolved differences as inconclusive. |
| Prefetch confounded codec speed | `081a2259`, `5749df08`: batched and early reads removed most apparent codec benefit | Compare codecs with the same prefetch policy. |
| Reconstruction metric missed quality loss | `802160cb`: group-64 experts changed median 43.5 decisions per 512 positions | Use held-out model quality and the required fidelity gate. |
| Naive baseline favored custom GLU | `8f305436`, `cf2ca3f9`: about 110 ms per-expert loop versus 22 ms gather, custom fused arms much slower | Compare with the production batched kernel. |
| Real widths lost, unused width won | `3e4fed91`: router tail 1.0% slower at one row and 12.9% slower at 7000 | Use actual shapes. Keep shape-gated wins only when the workload reaches them. |
| Target selected another kernel | `backend/metal/quantized.cpp`: `_nax` can bypass plain row-tile dispatch | M4 tuning does not establish M5 performance. |
| Host/eval timing was overinterpreted | `docs/perf/decode-round-trace-m4max-2026-09.md` | Eval wall time is not a pure GPU counter. Instrumentation and build mode change overlap. |
| Padding outweighed staging | `docs/perf/ane-unified-activation-plan.md`: bucket padding discarded substantial dense MLP work | Price padded rows and joins before tuning small transfer costs. |
| Replay prediction failed live | `2fdee6bf`: predicted 12.8x concurrency cost versus 1.00x live | Validate the predictor's baseline and confirm serving claims on the live path. |
| INT4 local speed versus quality | `docs/perf/ane-gpu-reevaluation-2026-09-06.md`: prefix 2.27x versus fp16, target PPL increased | Separate leg speed, baseline precision, request gain, and quality. |
| Flags preserved abandoned designs | `docs/perf/upstream-cleanup-2026-09-07.md`: cleanup removed 1,201 lines | Retain experiment evidence instead of unused runtime machinery. |

## Corrected assumptions

- RAM residency removes checkpoint disk reads from the hot path, not memory traffic.
- ANE/GPU concurrency does not guarantee additive bandwidth. It also does not
  prove overlap can never help another shape or fused dense subgraph.
- Observed ANE dispatch floors and program-count limits depend on the tested path.
- Lower draft acceptance can be worthwhile when proposal generation is cheaper.
- Equal timing does not prove equal dispatch, and a warm cache can have zero misses.
- Kernel or engine rounding differences are not automatically target requantization.
- Attention is not universally the largest traffic term. Count both K and V.
- Generated prompts are valid tests when representative. Repetitive self-continuations
  are insufficient evidence of general speculation performance.
- Background interference warrants attribution, not automatic changes to backup,
  indexing, or unrelated running processes.

## Source map

Use repository-root paths for the main records:

- `docs/perf/ane-gpu-reevaluation-2026-09-06.md`: dense/MoE ANE results, bandwidth,
  device placement, quality, and failed bank/micro-batch experiments.
- `docs/perf/qwen38-flash-2026-09.md`: MoE experiments and later corrections.
- `docs/perf/closed-bead-review-2026-09-06.md`: reviewed experiments and follow-ups.
- `docs/perf/upstream-cleanup-2026-09-07.md`: deleted code and control-test repair.
- `CARBACK-MLXFAST.md`: broader upstream review and work plan.

Commit IDs identify historical experiments, not necessarily current code. Bead
status can change. If an identifier or record is absent in another checkout,
label it unavailable and use current source and measurements. Do not depend on
private `.plans/` audit files or stale line numbers to justify shipped behavior.
