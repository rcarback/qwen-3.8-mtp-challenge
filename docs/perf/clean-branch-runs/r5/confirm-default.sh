#!/usr/bin/env bash
# Confirm the new 4-bit default: one arm per prompt with NO head environment variable,
# on geology (high acceptance) and cooking (low acceptance). Expect geology near 0.0419
# and cooking near 0.0806 s/token with all tokens matched. Waits for the rebuild.
set -uo pipefail
SP=/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/clean
CLEAN=/Users/carback1/Code/llms/qwen-clean.noindex
while ! grep -q '^== build done' "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/2283deea-2e48-4912-9762-af060a417604/tasks/bnhqc4zky.output" 2>/dev/null; do sleep 15; done
grep -q '^== build done rc=0' "/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/2283deea-2e48-4912-9762-af060a417604/tasks/bnhqc4zky.output" || { echo "build failed, not measuring"; exit 1; }
while pgrep -f 'mlxfast-swift|mlxfast-runtime-worker|swift-build' >/dev/null; do sleep 15; done
export PROMPTS="geology cooking"
echo "== R5 default confirm $(date +%T)"; "$SP/measure.sh" "$SP/results/r5" cleandef="$CLEAN"
echo "== r5 done $(date +%T)"
