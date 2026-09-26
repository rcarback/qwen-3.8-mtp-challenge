# Tune prefill chunks within memory limits

Chunking trades fewer large forwards against smaller working sets. It affects
activation memory, matmul efficiency, kernel selection, and repeated weight reads.
A materialized attention score matrix is one possible cost, not a requirement.
Fused attention avoids that full matrix but still has other working memory.
Recurrent layers can also change kernel behavior with chunk width.

## Choose the experiment

1. Record actual dispatched chunk widths. Prompt length alone does not prove a
   large forward reached the model.
2. Identify peak allocations on the current attention and MLP paths. Use a
   width-times-cache-depth score bound only where scores really materialize.
3. Try a few static caps within the memory budget before an adaptive policy.
4. Compare whole-request prefill, TTFT, peak memory, and any timeout behavior.
   Use appended-chunk probes to explain the result, not as the final result.
5. Counterbalance cap order. At small effects, reverse or randomize the sweep.
6. Preserve causal offsets and recurrent state across boundaries. Compare the
   required numerical/token-fidelity behavior with the reference chunk schedule.

Correctness cases include exact cap multiples, one token below/above a cap,
nonempty cache, rollback, and short final chunks. Numerical equality is not
implied by equivalent real-number arithmetic: chunking can change kernel dispatch.

Measure representative short and long prompts, plus deployment limits relevant
to the proposed change. Report OOMs and timeouts. Do not allocate an impossible
unchunked reference solely to satisfy a comparison recipe.

## Predictors and guardrails

A chunk-by-depth grid predicts whole-prompt cost only inside its measured domain.
Mark extrapolation explicitly. Price wide chunks at their actual width instead of
silently substituting the grid edge. Include loop and state-handling overhead.

In this repository, `resolveChunkCap` clamps the tested cap override to 512–16384.
A 512-token seed remains one chunk for all those caps. That cap change leaves this seed schedule unchanged. Choose a floor for another
model from its own memory budget and workload.

## Historical results

- `b2833214`: ascending and descending dense sweeps reversed the apparent winner.
  Their averaged differences were about 1%, with direction unresolved.
- Bead `8it`: cap 4096 versus 1024 showed a directional 5–17% MoE prefill gain
  at 7,526 tokens. It needs repeated paired validation.
- `docs/perf/ane-gpu-reevaluation-2026-09-06.md`: splitting the MoE forward into
  four micro-batches lost about 10% before adding ANE work.

Different towers need not produce the same optimum. Keep the existing cap when
no measured benefit clears uncertainty. A lower memory peak can justify a cap
without a speed win. Test static choices before automating selection.
Use [measurement](measurement.md) for timing and correctness.
