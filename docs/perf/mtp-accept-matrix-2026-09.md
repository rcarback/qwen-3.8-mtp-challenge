# MTP accept-rate matrix across weight configurations

This document reports a multi-prompt sweep of the MTP draft accept rate under
three ANE weight-source configurations. Only the all-GPU rows are valid. The
dequant and bf16 hybrid rows are invalid: see "Hybrid rows" below.

## Run conditions

The sweep ran behind the 40C thermal gate. Each leg decoded 128 tokens at
draft depth 8 or draft depth 2. It used four prompts: `README.md`,
`docs/qwen-prefill-research-plan.md`, `docs/qwen-mtp-go-live-runbook.md`, and
`docs/private-benchmark-security.md`. The `gpu` config set no flags. The
`dequant` config set `MLXFAST_NO_SANDBOX=1` and `MLX_ANE_DIRECT=1`. The
`bf16` config set the same two flags plus `MLX_ANE_BF16_WEIGHTS`. The full
sweep CSV is `docs/perf/accept-runs/mtp-accept-20260901-225614.csv`.

## All-GPU result (valid)

Median rows, from
`python3 tools/mtp-accept-summary.py docs/perf/accept-runs/mtp-accept-20260901-225614.csv`:

| config | depth | n | accept rate | mean draft len | speedup | matched |
|---|---|---|---|---|---|---|
| gpu | 8 | 4 | 0.515 | 2.66 | 1.364 | True |
| gpu | 2 | 4 | 0.584 | 1.97 | 1.369 | True |

Per-leg rows, one per prompt:

| config | prompt | depth | accept rate | mean draft len | speedup | matched |
|---|---|---|---|---|---|---|
| gpu | README | 8 | 0.540 | 2.49 | 1.363 | True |
| gpu | README | 2 | 0.619 | 1.95 | 1.385 | True |
| gpu | qwen-prefill-research-plan | 8 | 0.490 | 2.83 | 1.365 | True |
| gpu | qwen-prefill-research-plan | 2 | 0.549 | 1.97 | 1.353 | True |
| gpu | qwen-mtp-go-live-runbook | 8 | 0.662 | 3.10 | 1.559 | True |
| gpu | qwen-mtp-go-live-runbook | 2 | 0.762 | 1.98 | 1.533 | True |
| gpu | private-benchmark-security | 8 | 0.455 | 2.34 | 1.295 | True |
| gpu | private-benchmark-security | 2 | 0.541 | 1.97 | 1.345 | True |

**Question 3 (which depth has the higher median speedup):** the two depths
are close. Depth 2 gives a median speedup of about 1.369, and depth 8 gives
about 1.364. The two values are within about 0.005 of each other, so neither
depth shows a clear speedup advantage for the all-GPU config.

## Hybrid rows (invalid, generation collapse)

The dequant and bf16 rows are identical to three decimals on every prompt,
and both sit near an accept rate of about 0.95. That similarity is not a
finding about the bf16 weight source. The goldens the wrapper generated for
these two configs are degenerate. On all four prompts, the model repeats one
token starting from an early step. The measured accept rate reflects the MTP
head predicting a stuck token, not speculation on real text. Do not read the
hybrid median rows as results.

The table below gives the distinct-token ratio for each prompt's golden
(unique tokens divided by total tokens). A healthy golden, like the all-GPU
ones above, has a high ratio. Both hybrid configs produced the same ratio per
prompt.

| prompt | gpu ratio | hybrid ratio (dequant and bf16) |
|---|---|---|
| README | 0.41 | 0.01 |
| qwen-prefill-research-plan | 0.48 | 0.01 |
| qwen-mtp-go-live-runbook | 0.56 | 0.01 |
| private-benchmark-security | 0.55 | 0.05 |

**Question 1 (does bf16 exceed dequant on the median prompt):** this
question cannot be answered from this run. The bf16 and dequant goldens are
both degenerate, so their accept rate measures a stuck token stream rather
than draft speculation.

**Question 2 (does either hybrid differ from all-GPU by more than the spread
across prompts):** this question also cannot be answered from this run.
No healthy hybrid measurement exists in this data set to compare against the
all-GPU spread.

## Diagnosis (controller, 2026-09-02)

Follow-up 64-step goldens on the `README` prompt, using the same CLI and
flags as the sweep, narrowed the cause. These diagnostic goldens are scratch
runs and are not committed.

| configuration | distinct ratio | note |
|---|---|---|
| all-GPU, 4-bit | 0.41 | normal |
| GPU-fp16 ablation (`MLX_ANE_FP16_GPU=1`, same fp16 weights, no ANE) | 0.70 | normal |
| ANE, dequant fraction 0.3125 | 0.02 | token 96597 repeated from step 0 |
| ANE, bf16 source | 0.03 | token 91914 repeated from step 1 |
| ANE, dequant fraction 0.0625 | 0.16 | degraded |

The fp16 numeric representation is not the cause, because the GPU-fp16
ablation uses the same fp16 weights without ANE offload and stays healthy.
The ANE-specific error on real activations compounds across 64 layers into a
collapse that single-layer synthetic tests at 1 ULP and self-consistent
goldens cannot detect.
