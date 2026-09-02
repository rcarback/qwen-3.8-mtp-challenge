# Decode step profile, M4 Max 128 GB, 2026-09

Instrument: `QwenPhaseBreakdownTests` (`[decode width 1 @ depth 512]`),
default arm `sumTable`. Seam shares are synced and unfused; they scale to the
fused step. Idle box, GPU under 44C, no thermal gate (untimed instrument).

| quantity | value | source |
|---|---|---|
| fused width-1 step (best of 5) | 39.655 ms | Step 4 |
| serve step, mtp-timed serial p50 (2026-09-01) | 81 ms | ledger |
| seam total, all 64 layers | 100.3 ms | Step 4 table |
| linear layers (48), seam sum | 76.6 ms | Step 4 table |
| full-attention layers (16), seam sum | 23.4 ms | Step 4 table |
| embed | 0.3 ms | Step 4 table |
| mlp per layer at S=1 (GPU, best of 15) | 1.008 ms | Step 6 |

Derived buckets, scaled to the fused step (`scale = fused / seam_total`):

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
multiplicative formula: FAIL (formula defect, see the additive model
below). Result under the additive model: PASS.

The `MLP, 64 layers` bucket alone (64.512 ms) exceeds the entire fused
width-1 step (39.655 ms). This forces `host and sync` sharply negative
(-41.788 ms, -105.4%). Per the brief's Step 8 reading, the per-layer MLP
figure from Step 6 does not transfer into the model. That figure comes from
an isolated micro-benchmark (`ANE-SPEED-ISO ... leg=baseline`, measured at
`MLXFAST_SEQ_LEN=1`, `MLXFAST_ANE_FRACTION=0.3125`). It runs a different
dispatch and dtype path than the fused decode step measured here. This
report does not extend into Task 2. No `task-2-brief.md` exists in this
plan directory as of this run.

## Additive per-sync overhead model

The plan's multiplicative scale is the wrong model for a seam that syncs
after every layer. Each `eval` call costs an additive constant, not a
multiplicative factor. The seam total (100.3 ms) minus the fused step
(39.655 ms), divided across all 64 syncs, gives that constant.

| bucket | formula | ms | share of fused step |
|---|---|---|---|
| per-sync overhead c | `(seam_total - fused) / 64` | 0.947 | n/a |
| gated-delta layers (48), corrected | `linear_sum - 48 * c` | 31.116 | 78.5% |
| full attention layers (16), corrected | `fa_sum - 16 * c` | 8.239 | 20.8% |
| embed, corrected | `max(embed - c, 0)` | 0.000 | 0.0% |
| sum of corrected buckets | | 39.355 | 99.2% |
| residual vs fused step | `fused - sum` | 0.300 | 0.8% |

The residual (0.300 ms, 0.8% of the fused step) is within 5% of the fused
step.

Under this model, the isolated MLP probe (1.008 ms) is mostly the
0.947 ms per-sync constant, plus about 0.06 ms of MLP work. This explains
why `64 * mlp` exceeded the fused step. The probe measured one dispatch
boundary, not real per-layer compute across 64 layers.

## Serve step versus model forward

| quantity | ms | source |
|---|---|---|
| fused width-1 forward (this doc) | 39.655 | Step 4 |
| lm_head + argmax, 1 row | 2.9 | Step 4, table 3 |
| forward + lm_head | 42.6 | sum |
| serve decode step, mtp-timed depth 0, worker block p50 (2026-09-01) | 81.3 | `.superpowers/sdd/2026-08-31-ane-iosurface-procedure-bank/progress.md`, "WHOLE-SYSTEM TABLE" entry |
| outside the model forward | `81.3 - 42.6` = 38.7 | difference |
| share of the serve step outside the forward | 47.6% | |

This 47.6% share is the `host and sync` bucket of the scope's decision
rule, measured at the serve boundary instead of inside the forward.

## Ruling

Rule applied: 1. Selected lever: scope section C, host synchronization at
the serve boundary. Bucket share: 47.6% (`host and sync`, 38.7 ms of the
81.3 ms parent-counted round). Rule 1 fires at any share over 20%.
Expected gain if the bucket reaches 70% of peak bandwidth: not applicable to
a host bucket. Bound instead: up to 38.7 ms per token if the outside-forward
time were zero; a realistic target is the parent-counted step within 10% of
forward + lm_head (about 47 ms), i.e. ~34 ms per token, 1.7x on serial
decode.
MTP schedule finding carried forward: pending the accept matrix; the
single-prompt observation (bf16 hybrid accept 0.60 vs dequant 0.12 vs all-GPU
0.53 at depth 8) stands as n=1.

Follow-on plan: `.plans/2026-09-02-decode-host-sync-plan.md` (local, not
committed). Its first task splits the 38.7 ms into measured buckets. The
static reading of the depth-0 round (`Qwen36MTPBlockSession.swift:2569-2644`)
shows one target forward, one GPU top-2, one `eval`, and two host reads. The
protocol path is one JSON line each way. Neither explains 38.7 ms on its own,
so the bucket table decides which lever the later tasks take.
