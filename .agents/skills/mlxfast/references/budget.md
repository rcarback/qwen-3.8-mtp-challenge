# Establish the time and byte budget

Start from the current checkpoint and actual forward. A prior model's inventory
is a cross-check, not the current model's measured budget.

## Build the ledger

Separate phases relevant to the goal: prefill, serial decode, supported verify
widths, startup, and serving overhead. Record actual forward widths, context,
batch size, layer geometry, and selected kernels. Choose representative regimes
and boundary cases. Do not sweep irrelevant model families or unreachable widths.

Inventory these separately:

| Quantity | Include |
| --- | --- |
| Artifact bytes | Payload, scales, biases, codebooks, padding, indexes |
| Resident bytes | Original and derived weights, compiled programs, allocator cache |
| Working state | K/V, recurrent snapshots, activations, temporary materializations |
| Logical traffic | Weights selected per step, reads/writes, replay and packing passes |
| Storage traffic | Faulted pages and reads for out-of-core tables or checkpoints |
| Measured traffic | Hardware counters, with the counter's scope and limitations |

RAM-resident weights still consume memory bandwidth. A diagnostic reporting zero
disk-streaming bytes does not mean zero DRAM traffic. For MoE, distinguish total
resident experts from selected experts and reuse across a multi-row batch.

For grouped weight formats, an approximate storage model is:

```text
weight_bytes = parameter_count * bits / 8
             + group_count * (scale_bytes + bias_bytes)
             + codebook_bytes + padding_bytes
```

Count actual packing and tensor shapes when available. Compression ratios against
fp16 do not apply unchanged when production already uses q4 or q8.

## Bound the opportunity

```text
idealized device time >= max(FLOPs / attainable_compute, bytes / attainable_bandwidth)
Amdahl speedup = 1 / ((1 - p) + p / s)
```

Here `p` is the affected fraction of baseline time and `s` its isolated speedup.
The Amdahl model assumes unaffected work and overlap stay unchanged. It is a
planning bound, not an observation. If 20% of a request becomes 2.4x faster,
the ideal request gain is about 1.13x.

Logical bytes divided by time estimate effective bandwidth. They do not measure
actual DRAM traffic: cache reuse, repeated reads, packing, and decompression can
change it. Compare byte predictions with timing and available counters.

Use [measurement](measurement.md) for lazy evaluation and host/eval attribution.
Neither a flat prefill curve nor a small quantization gain isolates a bottleneck.
One could reflect dispatch, arithmetic, decompression, padding, or memory traffic.

## Useful first experiments

- Remove a redundant read, copy, cast, materialized intermediate, or synchronization.
- Reduce target weight/KV precision only within the permitted quality contract.
- Improve weight reuse across rows or experts without replaying the whole forward.
- Separate memory capacity improvements from bandwidth and latency improvements.
- Measure independent engine overlap after charging preparation and join costs.

Choose by expected absolute gain and effort. A small term can justify a cheap
simplification. No universal 1% or 5% threshold marks work below it as useless.

## Repository evidence

`Tests/MLXFastTests/Model/DecodeBandwidthTests.swift` uses synthetic geometry and
logical byte estimates. Recompute its inventory for the deployed checkpoint.
`docs/perf/decode-round-trace-m4max-2026-09.md` records host/eval experiments.
`docs/perf/ane-gpu-reevaluation-2026-09-06.md` contains the dense split byte derivation.
For the wider backlog, use `CARBACK-MLXFAST.md` and current bead status.
