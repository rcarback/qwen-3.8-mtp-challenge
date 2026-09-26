# Improve routed-expert execution

Profile the actual routing, gather GEMM, activation, and reduction chain. MoE
cost can include weight traffic, arithmetic, launch overhead, sorting, and padding.
Do not assume it is solely a dispatch problem.

## Candidate sequence

1. Compare against the production batched gather path with real expert count,
   top-k, quantization, and routing skew. A per-expert loop is diagnostic only.
2. Try exact gate/up concatenation where quantization groups remain independent
   along the concatenation axis. Validate scales, biases, layout, and reduction.
3. Prepare reusable packed weights once when appropriate. Release redundant stacks
   only after confirming fallback, serialization, and other widths can use the new layout.
   Report startup preparation when the objective charges it.
4. Reduce duplicate routing or sorting work before replacing the full GLU kernel.
5. Evaluate batching by expert, tile occupancy, vectorized dequantization, and
   reuse across rows. Charge sorting, gather indexes, unsorting, and reduction.
6. Tune the router tail only after measuring its full-forward share.

Expert merging and changes to top-k alter target computation and need the
approximate gate. Activation pruning has the same requirement, and zeroing values
saves no arithmetic when the next dense kernel still multiplies every element.

## Validation

Apply [measurement](measurement.md). Match selected experts, weights, and tie
behavior under the required contract. Test one hot expert, idle experts, skewed
loads, duplicate selections where permitted, and quantization tails.

Make the unfused control explicit. `SwitchGLU.forceFuseGateUp` can override a
process default, so inspect the effective state before its first forward.
A gate/up equivalence test previously compared two fused arms (bead `dyv`).

A different accumulation order can change rounding without changing representation.
Test numerical and token fidelity. Do not automatically invent a target quality
budget to excuse a failed strict execution gate.

Cover affected decode, verify, and prefill widths. Retain a candidate behind a
small support predicate when real shapes win and others use a verified fallback.
Do not retain a special kernel solely because it wins at an unreachable width.

## Historical evidence

Local M4 Max experiments on the Qwen MoE tower:

| Experiment | Result and limitation |
| --- | --- |
| Production gather versus naive loop, `8f305436` | About 22 ms versus 110 ms. The loop was an unsuitable optimization baseline. |
| Gate/up fusion, `b183eed3` | About 0.6 ms or 1% per decode step, counterbalanced. |
| Router tail, `3e4fed91` | 1.0% slower at one row, 12.9% slower at 7000 rows, faster at an unused 512-row width. |
| Scalar fused GLU, `cf2ca3f9` and `FusedRoutedMoEKernel.swift` | Hundreds of milliseconds versus about 22 ms for gather GEMM. Threadgroup tuning did not close the gap. |

These results reject specific designs, not all future fusions. Measure a current
launch/traffic budget before using an old 1–2 ms estimate as a ceiling. Try a new
kernel only when its design addresses the measured cost.

Dense projections and shared experts inside a MoE model can still be ANE candidates.
Use [engine offload](engine-offload.md), including the remaining GPU work and join.
