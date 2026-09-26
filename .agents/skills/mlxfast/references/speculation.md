# Reduce total decode cost through speculation

A proposal source is useful when verified tokens cost less than serial
decode. Acceptance alone is not the objective. A cheaper head with lower
acceptance can win, and a larger head with higher acceptance can lose.

## Correctness first

For greedy decoding, verify proposals against the target and accept the valid
prefix under the deployment's fidelity contract. Restore all rejected KV and
recurrent state before continuing. Batched target execution may round differently
from serial execution, so use the actual contract instead of assuming bit equality.

For stochastic decoding, checking target argmax is insufficient. Preserve the
target distribution with the correct acceptance and rejection-sampling algorithm,
including proposal probabilities and logits processors. Same-seed token equality
is not generally a distributional correctness test. See the
[speculative decoding paper](https://arxiv.org/abs/2211.17192).

Changing only proposals does not change the target distribution when verification
is correct. Head quantization, substitution, and lookup drafting need
verifier, rollback, acceptance, and cost checks. Draft NLL is optional diagnostic
evidence. If unverified proposals can reach the user, classify the resulting
behavior change separately. A shared quantized head must not mutate target weights.

## Price a round

```text
cost_per_emitted_token = total_charged_round_time / total_emitted_tokens
```

Charge drafting, rejected rows, target verification, repair/replay, sampling,
synchronization, and policy probes. Record offered and accepted tokens separately
from target bonus tokens. Compare against both direct serial decode and the
runtime's depth-zero policy: depth zero can retain wrapper overhead.

Use representative held-out prompts at deployment context and output length.
Copy-heavy text can favor lookup drafting. Repetitive target-generated continuations
can overstate acceptance. Neither workload should stand in for general prose.

## Small experiments

1. Test the shipped policy against depth zero and a small set of fixed depths.
2. Measure draft and verify costs on this architecture before fitting policy costs.
3. Trace adaptive recovery after rejected rounds drive depth to zero. Charge any
   recovery probe. A policy with no new acceptance observations can stay stuck.
4. Try cheaper compatible proposal sources before training more machinery.
5. Optimize batched verification and state repair only after proving the required
   hidden-state convention, cache semantics, and tokenizer compatibility.

External drafters can propose through tokens without matching target hidden states.
Native or hidden-state drafters have stronger architecture-specific requirements.
A model without a native head may still support lookup or external proposals.
Check whether those sources are allowed in the current deployment.

## Tests and decision

Cover accept-none, accept-some, accept-all, end-of-sequence, maximum depth, and
recovery from zero. For stateful or serving changes, add cancellation and resume.
Check full-attention and recurrent rollback separately. Record the actual proposal
source and widths. A policy flag is not proof that drafting engaged.

Report per-prompt cost and acceptance, then the required aggregate. Speculation
can help a prefill-heavy request only through its decode share. Use that share to
bound the request gain instead of declaring the entire technique inapplicable.

Historical examples:

- `60b36856`: model-specific round pricing moved a MoE result from 15.03 to
  21.03 tok/s against 17.88 serial. The earlier flat price did not transfer.
- Bead `nnr`: an adaptive recovery gap produced 17.71 tok/s versus 20.4 at forced
  depth one. This is evidence for testing recovery, not a universal depth optimum.
- `docs/perf/bead-close-out-2026-09-02.md`: lookup drafting helped copying but
  regressed prose. Eligibility still depends on the current track rules.
- `docs/dflash2-head-port-plan.md`: a substituted head lost 15.6% in the full
  comparison. One-offer acceptance was insufficient to predict it.

Keep serial fallback where speculation loses. Remove an unhelpful proposal source
when no supported workload justifies it. Use [measurement](measurement.md) for
paired timing. A correctness failure blocks every speed claim.
