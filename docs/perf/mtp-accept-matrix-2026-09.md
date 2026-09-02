# MTP accept-rate matrix across weight configurations

This document reports a multi-prompt sweep of the MTP draft accept rate under
three ANE weight-source configurations. The sweep ran behind the 40C thermal
gate. Each leg decoded 128 tokens at draft depth 8 or draft depth 2. It used
four prompt files: `README.md`, `docs/qwen-prefill-research-plan.md`,
`docs/qwen-mtp-go-live-runbook.md`, and `docs/private-benchmark-security.md`.
The `gpu` config set no ANE flags. The `dequant` config set
`MLXFAST_NO_SANDBOX=1` and `MLX_ANE_DIRECT=1`, and sourced ANE fp16 weights by
dequantizing the 4-bit weights. The `bf16` config added
`MLX_ANE_BF16_WEIGHTS` pointing at the original bf16 checkpoint, so the ANE
weights came from the bf16 source instead of a dequantized 4-bit source.

## Median summary by config and depth

| config | depth | n | accept rate | mean draft len | serial s/tok | mtp s/tok | speedup | matched |
|---|---|---|---|---|---|---|---|---|
| bf16 | 2 | 4 | 0.960 | 1.99 | 0.1042 | 0.0607 | 1.720 | True |
| bf16 | 8 | 4 | 0.943 | 5.47 | 0.1044 | 0.0510 | 2.051 | True |
| dequant | 2 | 4 | 0.960 | 1.99 | 0.1045 | 0.0607 | 1.708 | True |
| dequant | 8 | 4 | 0.943 | 5.47 | 0.1046 | 0.0511 | 2.052 | True |
| gpu | 2 | 4 | 0.584 | 1.97 | 0.1081 | 0.0780 | 1.369 | True |
| gpu | 8 | 4 | 0.515 | 2.66 | 0.1076 | 0.0783 | 1.364 | True |

Every row has n equal to 4, one measurement per prompt file, and matched
equal to True. All 24 legs passed token fidelity.

## Per-leg detail

| config | prompt | depth | accept rate | mean draft len | speedup | matched |
|---|---|---|---|---|---|---|
| gpu | README | 8 | 0.540 | 2.49 | 1.363 | True |
| gpu | README | 2 | 0.619 | 1.95 | 1.385 | True |
| dequant | README | 8 | 0.947 | 5.38 | 2.066 | True |
| dequant | README | 2 | 0.966 | 1.98 | 1.708 | True |
| bf16 | README | 8 | 0.947 | 5.38 | 2.069 | True |
| bf16 | README | 2 | 0.966 | 1.98 | 1.744 | True |
| gpu | qwen-prefill-research-plan | 8 | 0.490 | 2.83 | 1.365 | True |
| gpu | qwen-prefill-research-plan | 2 | 0.549 | 1.97 | 1.353 | True |
| dequant | qwen-prefill-research-plan | 8 | 0.939 | 5.75 | 2.047 | True |
| dequant | qwen-prefill-research-plan | 2 | 0.955 | 2.00 | 1.709 | True |
| bf16 | qwen-prefill-research-plan | 8 | 0.939 | 5.75 | 2.041 | True |
| bf16 | qwen-prefill-research-plan | 2 | 0.955 | 2.00 | 1.704 | True |
| gpu | qwen-mtp-go-live-runbook | 8 | 0.662 | 3.10 | 1.559 | True |
| gpu | qwen-mtp-go-live-runbook | 2 | 0.762 | 1.98 | 1.533 | True |
| dequant | qwen-mtp-go-live-runbook | 8 | 0.973 | 5.55 | 2.056 | True |
| dequant | qwen-mtp-go-live-runbook | 2 | 0.988 | 2.00 | 1.732 | True |
| bf16 | qwen-mtp-go-live-runbook | 8 | 0.973 | 5.55 | 2.060 | True |
| bf16 | qwen-mtp-go-live-runbook | 2 | 0.988 | 2.00 | 1.736 | True |
| gpu | private-benchmark-security | 8 | 0.455 | 2.34 | 1.295 | True |
| gpu | private-benchmark-security | 2 | 0.541 | 1.97 | 1.345 | True |
| dequant | private-benchmark-security | 8 | 0.881 | 4.92 | 1.935 | True |
| dequant | private-benchmark-security | 2 | 0.912 | 1.98 | 1.668 | True |
| bf16 | private-benchmark-security | 8 | 0.881 | 4.92 | 1.944 | True |
| bf16 | private-benchmark-security | 2 | 0.912 | 1.98 | 1.661 | True |

## Findings

1. The bf16 hybrid's accept rate does not exceed the dequant hybrid's on the
   median prompt. Both configs report the same median accept rate, about
   0.96 at depth 2 and about 0.94 at depth 8.
2. Both hybrid configs differ from the all-GPU config by more than the
   spread across prompts within a config. The hybrid-to-GPU gap in median
   accept rate is about 0.38 at depth 2 and about 0.43 at depth 8. The
   largest within-config spread across the four prompts is about 0.22 for
   the all-GPU config and about 0.09 for either hybrid config. Prompt-to-
   prompt noise does not explain the gap between hybrid and all-GPU.
3. Depth 8 gives the higher median `mtp_decode_speedup` for both hybrid
   configs, about 2.05 compared with about 1.71 to 1.72 at depth 2. Depth 2
   gives a slightly higher median speedup for the all-GPU config, about
   1.369 compared with about 1.364 at depth 8.
