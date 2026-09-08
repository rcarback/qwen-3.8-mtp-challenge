#!/usr/bin/env bash
# R4: the clean tree with the proposal head at 4 bits (MLX_QWEN_MTP_HEAD_QUANT=4),
# over every prompt that survived the seed check. Runs after campaign-swap.sh.
set -uo pipefail
SP=/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/clean
CLEAN=/Users/carback1/Code/llms/qwen-clean.noindex
while ! grep -q '^== swap done' "$SP/campaign-swap.log" 2>/dev/null; do sleep 60; done
while pgrep -f 'mlxfast-swift|mlxfast-runtime-worker' >/dev/null; do sleep 30; done
survivors="$(grep '^== screened prompts:' "$SP/campaign-swap.log" | sed 's/^== screened prompts://')"
export PROMPTS="public cooking geology music readme runbook${survivors}"
echo "== R4 prompts: $PROMPTS"
echo "== R4 $(date +%T)"; "$SP/measure.sh" "$SP/results/r4" cleanhq4="$CLEAN@MLX_QWEN_MTP_HEAD_QUANT=4"
echo "== r4 done $(date +%T)"
