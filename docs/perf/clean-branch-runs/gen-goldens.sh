#!/usr/bin/env bash
# Generate 512-step goldens with the BASE tree (origin/main tip) for seven prose prompts.
# Waits for base-build.sh to finish. One model-holding process at a time.
set -uo pipefail
SP=/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/clean
WT=/Users/carback1/Code/llms/qwen-clean.noindex
while ! grep -q "^== base-build done" "$SP/base-build.log" 2>/dev/null; do sleep 30; done
cd "$WT" || exit 1
[ -x .build/release/mlxfast-swift ] || { echo "no mlxfast-swift"; exit 1; }
[ -f weights/config.json ] || { echo "no transformed weights"; exit 1; }
for p in cooking geology music readme prefill-plan runbook security; do
  out="$SP/goldens/golden-base-$p.json"
  echo "== golden $p $(date +%T)"
  .build/release/mlxfast-swift generate-golden --prompt-file "$SP/prompts/$p.txt" --weights weights --output "$out" --name "base_$p" --steps 512 > "$SP/goldens/gen-$p.log" 2>&1
  echo "rc=$? $(jq -c '.cases[0] | {name, n_prompt: (.prompt_tokens|length), n_expected: (.expected_tokens|length)}' "$out" 2>/dev/null)"
done
echo "== goldens done $(date +%T)"
