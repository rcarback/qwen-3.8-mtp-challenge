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
the fused step and no bucket is negative. Result: FAIL.

The `MLP, 64 layers` bucket alone (64.512 ms) exceeds the entire fused
width-1 step (39.655 ms). This forces `host and sync` sharply negative
(-41.788 ms, -105.4%). Per the brief's Step 8 reading, the per-layer MLP
figure from Step 6 does not transfer into the model. That figure comes from
an isolated micro-benchmark (`ANE-SPEED-ISO ... leg=baseline`, measured at
`MLXFAST_SEQ_LEN=1`, `MLXFAST_ANE_FRACTION=0.3125`). It runs a different
dispatch and dtype path than the fused decode step measured here. This
report does not extend into Task 2. No `task-2-brief.md` exists in this
plan directory as of this run.
