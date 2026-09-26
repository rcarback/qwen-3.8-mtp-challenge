# Generalize mechanisms across models

Share the smallest capability surface that actual consumers need. Reusing code
reduces maintenance work. It does not transfer a speedup or correctness result
from the first model to the second.

## Capability checks

| Mechanism | Model-specific contract |
| --- | --- |
| Dense linear optimization | Shapes, dtype, packing, quantization metadata, tied consumers |
| Attention/cache optimization | Head geometry, mask, positions, sinks, windows, cache classes |
| Hybrid or recurrent execution | State ownership, update order, snapshot and replay semantics |
| Expert optimization | Routing, top-k, activation, reduction order, load distribution |
| Hidden-state drafting | Hidden-state/norm convention, target head, verifier and rollback |
| Token-only proposals | Tokenizer mapping, proposal probabilities, target verification |
| Prefix reuse | Complete state identity and valid restore boundaries |
| ANE offload | Actual operator support, weight form, placement, memory and join cost |

Do not require a native head, ANE, or replay tape from a model without one.
Unsupported capabilities should decline explicitly to a tested reference path.
Use conservative defaults for newly introduced capabilities. Empty outputs or a
no-op method must not masquerade as implemented support.

## Extraction sequence

1. Keep model arithmetic in the model adapter. Extract only operations the shared
   caller actually uses. A protocol is useful when it removes real duplication.
2. Implement the second consumer against its own forward, state, and tokenizer.
   Avoid transplanting constants for verify cost, geometry, or norm placement.
3. Preserve validators for existing models. Add architecture-specific checks instead
   of weakening the first model's required contract.
4. Test missing capabilities, fallback, state boundaries, and configuration parsing.
   Validate ranges before arithmetic such as `depth + 1`, including `Int.max`.
5. Compare the shared route with each model's existing route, then apply that
   model's required fidelity gate. Measure performance separately per model.

A second model should exercise the property being generalized. It need not be
from an unrelated family for every helper. Use a structurally different model
when claiming architecture-independent support. Small models make useful tests,
but they do not establish large-model bandwidth behavior.

## Repository examples

`Sources/MLXFastModel/Qwen36MTPTarget.swift` defines a native-MTP target seam.
`Sources/MLXFastModel/Qwen4ExpMTPTargetConformance.swift` supplies a different tower's
forward and state conventions, including declined recurrent replay support.
Audit callers of `publishesRecurrentReplayTape` before relying on that declaration.

`docs/perf/qwen38-flash-2026-09.md` reported 1.16x speculation on a 23-token prompt,
then a 17% regression at 732 tokens. The real-window result shows why a short
smoke test cannot promote the generalized mechanism's speed claim.

`QwenDepthAndWidthBoundTests.swift` covers parts of the configuration surface.
Inspect all accepted parsers and arithmetic separately. Existing depth tests do
not prove another width parser cannot overflow.

## Decision

Keep straightforward shared wiring when real consumers need it and behavior is
correct. Do not build a plugin registry or autotuner before the measured need.
Missing tests usually call for targeted coverage, not deleting a valid interface.
If only one consumer remains and the abstraction adds work, simplify it.
Use [measurement](measurement.md) for per-model evidence and report unsupported
capabilities instead of describing every model as accelerated.
