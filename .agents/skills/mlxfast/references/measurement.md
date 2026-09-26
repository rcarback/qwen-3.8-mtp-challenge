# Measure the claimed improvement

Choose the measurement scope before running. An isolated kernel benchmark can
establish a kernel improvement. Promotion to a model-speed claim needs a real
forward, and a serving claim needs request wall time and the intended workload.

## Comparable arms

1. Record the revision, dirty changes, model and tokenizer identity, weight format,
   toolchain, hardware, and relevant settings. Confirm the baseline remains current.
2. Fix the timed window, inputs, output length, sampling, cache state, and warmup.
   Use a release build for deployment timing. Debug probes remain useful diagnostics.
3. Compare the production baseline with one changed variable. Use separate processes
   when initialization, compiled pipeline caches, or process-wide flags can contaminate arms.
   Otherwise an isolated, resettable in-process comparison is valid.
4. Alternate or counterbalance arms. Start with at least three valid pairs for a
   speed claim. Repeat further when uncertainty prevents a decision. Three pairs do
   not establish a reliable p95 or statistical significance by themselves.
5. Predeclare sample exclusions. Preserve rejected samples and reasons. Timeouts and
   startup failures remain product results even when excluded from steady-state timing.
6. Report paired differences and variability. For small effects, reverse order or
   randomize the sweep. Use the deployment's specified aggregation for ranked scores.
   A ratio of means, mean of ratios, and median of ratios are different statistics.

Representative held-out inputs may be authored, collected, or generated. Match
real lengths, domains, and routing. A target's repetitive greedy self-continuation
can flatter speculation and is insufficient as the sole workload.

## Machine state and timing boundaries

Use the same thermal and power policy for both arms. Follow an existing ranked
thermal gate unchanged. Example: one deployment's ranked runner gates every timed
leg at 40C. Record missing telemetry and its limitation.

For isolated latency, wait for unrelated load to subside. For contention studies,
reproduce the intended competing workload and measure its achieved load too.
Separate those results from quiescent measurements. Run one large model residency
at a time on this repository's local benchmark machine.

Do not suspend unrelated processes, change backup/indexing settings, or delete
unrelated artifacts as an implicit benchmarking step. Inspect contention and disk
space first. Clean only artifacts owned by this experiment within the user's scope.

Measure request wall time alongside model timings. A large gap can include queueing,
tokenization, compilation, paging, or blocking. Identify the cause before calling
it contamination. Report startup/compile and warm operation separately, charging
both whenever the user's objective includes both.

MLX builds lazy graphs. Include evaluation and completion of timed outputs, and
exclude previously queued work consistently. Re-evaluate changed inputs so a repeat
measures execution rather than already-evaluated arrays. See
[MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

Host build time and `eval` wall time are useful boundaries, but `eval` can include
encoding, compilation, execution, and waiting. They are not automatically CPU/GPU
counters. Forced per-layer barriers change overlap. Use an uninstrumented total
and a timeline before adding phase times or deriving a host-only speedup ceiling.

## Prove the arms differ

Reuse existing executed-path logs or tracing before adding counters. Diagnostics
must have negligible overhead when disabled. Compare diagnostic and uninstrumented
runs if logging can perturb the result.

Record the decision after feature flags and support guards, including fallback
reason. A control must disable the treatment explicitly when the process default
can enable it. Force an unsupported case when fallback is part of the design.
Equal timings alone do not prove that both arms ran the same path.

For ANE, distinguish program creation, preferred placement, actual dispatch, and
completed execution. Count successful treatment work and partial fallbacks. A
successful test that returned early before the model ran is not runtime coverage.
For server tests, verify readiness belongs to the launched process and revision.

## Match correctness to the change

- **Execution changes:** test tensor tolerances, state invariants, and the exact
  token-fidelity rule the deployment specifies. Reassociation or another engine
  can change rounding. Do not silently weaken a strict gate to accommodate it.
- **Target approximations:** declare the permitted quality loss before tuning.
  Measure held-out NLL/perplexity and task or generation behavior. Include token
  disagreements and reference logit gaps when diagnosing divergences.
- **Proposal-only changes:** test the verifier and rollback. Greedy acceptance and
  distribution-preserving stochastic sampling require different algorithms.
  Draft-head NLL is diagnostic, not the target's quality gate.
- **Changed inputs:** assess task outcomes and total cost, including retrieval or
  retries caused by the change.

Repeated n-grams and distinct-token counts can detect generation collapse, but
they are smoke checks, not substitutes for task quality. A small perplexity
difference on a few prompts does not prove unchanged behavior. Use the same
candidate weight identity for quality and speed, even if those checks run in
separate sessions.

## Experiment record and stopping rule

Record enough to reproduce the claim:

- Revision, artifact identities, hardware/software, settings, and reproduction command.
- Workload, token counts, cache state, timed boundary, pair order, and exclusions.
- Executed path, relevant shapes, fallback count, and diagnostic overhead.
- Paired timings and variability, startup cost if relevant, and peak memory.
- Required correctness results, quality results when applicable, and unresolved gaps.
- Decision: accepted, rejected, inconclusive, or blocked by a prerequisite.

Promote a speed claim when correctness passes and repeated differences clear the
observed uncertainty within resource and latency budgets. Keep a tested shape
predicate where only some shapes win. Delete abandoned experimental paths without
supported use, retaining their evidence. A simplification or memory saving may be
worth keeping without a speed gain. State that justification explicitly.
