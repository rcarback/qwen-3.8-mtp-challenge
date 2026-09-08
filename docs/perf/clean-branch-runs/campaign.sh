#!/usr/bin/env bash
# Three counterbalanced rounds at the ranked window over the eight goldens.
#  R1: base, clean                       (alternating order per prompt)
#  R2: base repeat, clean with the 8-bit proposal head disabled (MLX_QWEN_MTP_HEAD_QUANT=0)
#  R3: clean repeat
set -uo pipefail
SP=/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/clean
BASE=/Users/carback1/Code/llms/qwen-base.noindex
CLEAN=/Users/carback1/Code/llms/qwen-clean.noindex
echo "== R1 $(date +%T)"; "$SP/measure.sh" "$SP/results/r1" base="$BASE" clean="$CLEAN"
echo "== R2 $(date +%T)"; "$SP/measure.sh" "$SP/results/r2" base2="$BASE" cleanhq0="$CLEAN@MLX_QWEN_MTP_HEAD_QUANT=0"
echo "== R3 $(date +%T)"; "$SP/measure.sh" "$SP/results/r3" clean2="$CLEAN"
echo "== campaign done $(date +%T)"
