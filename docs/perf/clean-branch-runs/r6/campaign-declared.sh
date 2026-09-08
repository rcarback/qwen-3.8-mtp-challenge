#!/usr/bin/env bash
# Three-way comparison with the ranked candidate's declared head (amal-david q2-q4 rerank,
# tree digest verified against mtp-head.manifest.json): base + declared head, clean +
# declared head, clean + pinned head at the 4-bit default. Same prompts as rounds 1 to 5.
set -uo pipefail
SP=/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/clean
BASE=/Users/carback1/Code/llms/qwen-base.noindex
CLEAN=/Users/carback1/Code/llms/qwen-clean.noindex
DH=/Users/carback1/.cache/mlxfast/declared-head-ae628274
while pgrep -f 'mlxfast-swift|mlxfast-runtime-worker|swift-build' >/dev/null; do sleep 15; done
export PROMPTS="public cooking geology music readme runbook dyeing"
echo "== R6 declared-head comparison $(date +%T)"
"$SP/measure.sh" "$SP/results/r6" basedh="$BASE@MLXFAST_QWEN_MTP_HEAD_DIR=$DH" cleandh="$CLEAN@MLXFAST_QWEN_MTP_HEAD_DIR=$DH" clean4pin="$CLEAN"
echo "== r6 done $(date +%T)"
