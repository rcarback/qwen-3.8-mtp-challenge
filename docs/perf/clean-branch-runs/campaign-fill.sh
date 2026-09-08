#!/usr/bin/env bash
# Fill-in for the arms voided in R1 and R2 by the stale goldens path in measure.sh
# (env.sh clobbered SP). The public arms of those rounds are valid and are kept.
# Waits for campaign.sh to finish, parks the voided measure logs, then reruns
# base/clean (R1) and base2/cleanhq0 (R2) over the seven golden prompts.
set -uo pipefail
SP=/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/clean
BASE=/Users/carback1/Code/llms/qwen-base.noindex
CLEAN=/Users/carback1/Code/llms/qwen-clean.noindex
while ! grep -q '^== campaign done' "$SP/campaign.log" 2>/dev/null; do sleep 30; done
for r in r1 r2; do mv "$SP/results/$r/measure.log" "$SP/results/$r/measure-void-stalepath.log"; done
export PROMPTS="cooking geology music readme prefill-plan runbook security"
echo "== R1 fill $(date +%T)"; "$SP/measure.sh" "$SP/results/r1" base="$BASE" clean="$CLEAN"
echo "== R2 fill $(date +%T)"; "$SP/measure.sh" "$SP/results/r2" base2="$BASE" cleanhq0="$CLEAN@MLX_QWEN_MTP_HEAD_QUANT=0"
echo "== fill done $(date +%T)"
