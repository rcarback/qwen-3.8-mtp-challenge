# Decode step profile, M4 Max 128 GB, 2026-09

Instrument: `QwenPhaseBreakdownTests` (`[decode width 1 @ depth 512]`),
default arm `sumTable`. Seam shares are synced and unfused; they scale to the
fused step. Idle box, GPU at 44.0C, no thermal gate (untimed instrument).

| quantity | value | source |
|---|---|---|
| fused width-1 step (best of 5) | 39.655 ms | Step 4 |
| serve step, mtp-timed serial p50 (2026-09-01) | 81 ms | ledger |
| seam total, all 64 layers | 100.3 ms | Step 4 table |
| linear layers (48), seam sum | 76.6 ms | Step 4 table |
| full-attention layers (16), seam sum | 23.4 ms | Step 4 table |
| embed | 0.3 ms | Step 4 table |

Derived buckets, scaled to the fused step (`scale = fused / seam_total`):
they use `mlp_ms_per_layer` = 1.008 ms (Step 6, isolated probe, best of
15). That figure is not decomposable into per-sync overhead and MLP
compute -- see below.

`scale = 39.655 / 100.3 = 0.3954`

| bucket | formula | ms | share |
|---|---|---|---|
| MLP, 64 layers | `64 * mlp_ms_per_layer` | 64.512 | 162.7% |
| gated-delta mixer, 48 layers | `(linear_sum - 48 * mlp) * scale` | 11.156 | 28.1% |
| full attention mixer, 16 layers | `(fa_sum - 16 * mlp) * scale` | 2.875 | 7.3% |
| lm_head | width-sweep table 3, `lm_head` row at M=1 | 2.9 | 7.3% |
| host and sync | `fused - (sum of the four rows above)` | -41.788 | -105.4% |

Arm comparison (fused width-1 step): sumTable 39.655 ms, replica 50.049 ms,
off 39.893 ms.

Exit criterion: the four GPU buckets plus host sum to the fused step by
construction; the check is that `host and sync` is between 0 and 15% of
the fused step and no bucket is negative. Result under the plan's
multiplicative formula: FAIL (formula defect -- the `MLP, 64 layers` bucket
alone exceeds the fused step). See the per-sync split below, which is not
an independent check either.

The `MLP, 64 layers` bucket alone (64.512 ms) exceeds the entire fused
width-1 step (39.655 ms). This forces `host and sync` sharply negative
(-41.788 ms, -105.4%). Per the brief's Step 8 reading, the per-layer MLP
figure from Step 6 does not transfer into the model. That figure comes from
an isolated micro-benchmark (`ANE-SPEED-ISO ... leg=baseline`, measured at
`MLXFAST_SEQ_LEN=1`, `MLXFAST_ANE_FRACTION=0.3125`, best of 15, 1.008 ms).
It runs a different dispatch and dtype path than the fused decode step
measured here. Its output was not saved to a committed
`docs/perf/raw-phase-*.txt` file. This report does not extend into Task 2.
No `task-2-brief.md` exists in this plan directory as of this run.

## Per-sync overhead split (not an independent check)

The plan's multiplicative scale is the wrong model for a seam that syncs
after every layer. Each `eval` call costs an additive constant, not a
multiplicative factor. The seam total (100.3 ms) minus the fused step
(39.655 ms), divided across all 64 syncs, gives that constant.

| bucket | formula | ms | share of fused step |
|---|---|---|---|
| per-sync overhead c | `(seam_total - fused) / 64` | 0.947 | not applicable |
| gated-delta layers (48), corrected | `linear_sum - 48 * c` | 31.116 | 78.5% |
| full attention layers (16), corrected | `fa_sum - 16 * c` | 8.239 | 20.8% |
| embed, corrected | `max(embed - c, 0)` | 0.000 | 0.0% |
| sum of corrected buckets | | 39.355 | 99.2% |
| residual against the fused step | `fused - sum` | 0.300 | 0.8% |

This split is an identity, not a closure test.
`linear_sum - 48c + fa_sum - 16c = seam_total - 64c = fused` holds by
construction for any value of `c`, so the table above cannot fail. The
residual (0.300 ms) equals the embed value exactly. That happens because
`embed - c` is negative (0.3 - 0.947 < 0) and clipped to zero in the row
above it. The residual is that clip, not independent evidence that the
split is correct. No split of `c` between the gated-delta and
full-attention buckets is tested by this table. The "gated-delta layers
(48)" row is a layer-kind bucket -- mixer, MLP, and norms together -- not a
mixer-only bucket.

The only independent cross-check available for `c` comes from the width-16
verify seam: `c = (267.1 - 189.4) / 64 = 1.21 ms`
(`docs/perf/raw-phase-sumTable.txt:37-41`). That is the same order of
magnitude as the 0.947 ms figure above. This supports the per-sync
reading: each `eval` boundary costs roughly one to two milliseconds,
regardless of layer kind. It does not prove the layer-kind split in the
table above.

The isolated MLP probe
(`Tests/MLXFastTests/Model/ANEFusedSplitSpeedTests.swift:35-36, 48-62,
65-74`) reads a real three-matrix group-64 4-bit MLP at 5120x17408. That
read moves about 134 MB of weights plus about 17 MB of scales and biases,
about 151 MB total. At the M4 Max peak of 546 GB/s, that read takes at
least about 0.27 ms. So the probe's per-sync overhead is at most about
0.73 ms (1.008 ms minus about 0.27 ms), not the 0.947 ms constant above.
The constant does not transfer to the probe. Consequence: per-layer MLP
compute is bounded below by about 0.27 ms. So 64 layers of MLP take at
least about 17 ms, at least about 43% of the fused forward (39.655 ms).

The split between the gated-delta mixer and the MLP inside the
linear-layer bucket is unmeasured by this instrument. At
31.1 ms / 48 = 0.65 ms per linear layer, with MLP bounded below by about
0.27 ms of that. The gated-delta mixer is bounded above by about 0.38 ms
per layer -- about 18 ms total, at most about 46%. That is against an MLP
of at least about 17 ms, at least about 43%. The two are indistinguishable
from this data.

## Serve step versus model forward

| quantity | ms | source |
|---|---|---|
| fused width-1 forward (this doc) | 39.655 | Step 4 |
| lm_head + argmax, 1 row | 2.9 | Step 4, table 3 |
| forward + lm_head | 42.6 | sum |
| serve decode step, mtp-timed depth 0, worker block p50 (2026-09-01) | 81 | `.superpowers/sdd/2026-08-31-ane-iosurface-procedure-bank/progress.md`, "worker block p50 0.081s" entry |
| outside the model forward | `81 - 42.6` = about 38 | difference |
| share of the serve step outside the forward | about 47% | |

This about 47% share is the `host and sync` bucket of the scope's decision
rule, measured at the serve boundary instead of inside the forward.

### Conditions that differ

The comparison above is not paired: the in-process reference and the serve
step were measured under different conditions.

| condition | in-process reference | serve step |
|---|---|---|
| (a) statistic | best of 5 (minimum) | p50 over about 128 rounds |
| (b) build | debug build, MLX host core unoptimized (`docs/perf/raw-phase-*.txt:6` says "Building for debugging..."; `Vendor/mlx-swift/Package.swift:215-223` carries no optimization flag) | release build |
| (c) process | in-process test | sandboxed child process |
| (d) host work between steps | none, between the five repeated calls | one JSON protocol round trip per round |
| (e) GPU temperature | 44.0C, untimed, no thermal gate | 40C thermal gate |
| (f) timer span | wraps model forward (graph build and eval) only | wraps the full parent-counted round |

Condition (a) biases the gap upward by perhaps 5-10% (the reference is a
minimum, the serve step a median). Condition (b) biases the gap downward
(the reference ran unoptimized). Neither is quantified beyond this
estimate.

The headline this comparison omits: 15.1 GB of quantized weight bytes
(`docs/qwen3.6-weight-contract.md:38-40`) read in 39.655 ms is about
381 GB/s. That is about 70% of the M4 Max peak of 546 GB/s -- the scope's
own realistic target. In-process, the forward is already near that
target; the serve path loses about 2x against it (81 ms versus 42.6 ms).

## Ruling

Rule applied: 1. Selected lever: scope section C, host synchronization at
the serve boundary. Bucket share: about 47% (`host and sync`, about 38 ms
of the 81 ms parent-counted round). Rule 1 fires at any share over 20%.
Expected gain if the bucket reaches 70% of peak bandwidth: not applicable
to a host bucket. Bound instead: up to about 38 ms per token if the
outside-forward time were zero. A realistic target is the parent-counted
step within 10% of forward + lm_head (about 47 ms). That is about 34 ms
per token saved, 1.7x on serial decode.
MTP schedule finding carried forward: pending the accept matrix. The
single-prompt observation -- bf16 hybrid accept 0.60, dequant 0.12,
all-GPU 0.53 at depth 8 -- stands as n=1.

Follow-on plan: `.plans/2026-09-02-decode-host-sync-plan.md` (local, not
committed). Its first task splits the about 38 ms into measured buckets.
The static reading of the depth-0 round
(`Qwen36MTPBlockSession.swift:2569-2644`) shows one target forward, one GPU
top-2, one `eval`, and two host reads. The protocol path is one JSON line
each way. Neither explains about 38 ms on its own, so the bucket table
decides which lever the later tasks take.

The follow-on plan's four tasks are:

- instrument the depth-0 round on parent, worker and session
- host-thread QoS
- command-buffer and ladder sweep
- serial-step prefetch across the protocol gap

Corrected stop route: if the gap does not reproduce, run the headroom
plan's Task 2 (the sub-layer mixer/MLP seam). Do this before choosing
between scope sections A and B.
