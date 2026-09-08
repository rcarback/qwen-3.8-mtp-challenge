#!/usr/bin/env bash
# Replace the two goldens voided by a first-position seed near-tie on BOTH trees
# (prefill-plan, security). Generates goldens for sailing and dyeing on the base
# tree, screens each with a short depth-0 seed check, then runs the same five arms
# the campaign ran (R1 base/clean, R2 base2/cleanhq0, R3 clean2) on the survivors.
set -uo pipefail
SP=/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/clean
BASE=/Users/carback1/Code/llms/qwen-base.noindex
CLEAN=/Users/carback1/Code/llms/qwen-clean.noindex
HEAD=/Users/carback1/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head
D="$SP/results/swap-screen"; mkdir -p "$D"
while ! grep -q '^== seed-diag done' "$SP/seed-diag.log" 2>/dev/null; do sleep 60; done
while pgrep -f 'mlxfast-swift|mlxfast-runtime-worker' >/dev/null; do sleep 30; done
keep=""
for p in sailing dyeing; do
  out="$SP/goldens/golden-base-$p.json"
  echo "== golden $p $(date +%T)"
  ( cd "$BASE" && .build/release/mlxfast-swift generate-golden --prompt-file "$SP/prompts/$p.txt" --weights weights --output "$out" --name "base_$p" --steps 512 ) > "$SP/goldens/gen-$p.log" 2>&1
  echo "   rc=$? $(jq -c '.cases[0] | {name, n_prompt: (.prompt_tokens|length), n_expected: (.expected_tokens|length)}' "$out" 2>/dev/null)"
  jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' "$out" > "$D/plan-$p.json"
  ok=1
  for tree in base clean; do
    T=/Users/carback1/Code/llms/qwen-$tree.noindex
    ( cd "$T" && .build/release/mlxfast-swift mtp-verify --weights weights --mtp-head "$HEAD" --emitted "$D/plan-$p.json" --generate 17 --mtp-depth 8 --output "$D/rows-$tree-$p.json" --plan-output "$D/gen-$tree-$p.json" ) > "$D/verify-$tree-$p.log" 2>&1
    ( cd "$T" && .build/release/mlxfast-swift mtp-timed --weights weights --mtp-head "$HEAD" --golden "$D/rows-$tree-$p.json" --tokens 16 --mtp-depth 0 ) > "$D/timed0-$tree-$p.json" 2> "$D/timed0-$tree-$p.err"
    rc=$?
    echo "   screen $tree depth0 rc=$rc $(grep -o 'seed prefill token [0-9]* disagreed with the reference.s [0-9]*' "$D/timed0-$tree-$p.err" | head -1)"
    [ "$rc" = 0 ] || ok=0
  done
  [ "$ok" = 1 ] && keep="$keep $p" || echo "   SKIP $p: seed check failed"
done
echo "== screened prompts:$keep"
[ -n "$keep" ] || { echo "== swap done (nothing to run) $(date +%T)"; exit 0; }
for r in r1 r2 r3; do [ -f "$SP/results/$r/measure.log" ] && mv "$SP/results/$r/measure.log" "$SP/results/$r/measure-before-swap.log"; done
export PROMPTS="${keep# }"
echo "== R1 swap $(date +%T)"; "$SP/measure.sh" "$SP/results/r1" base="$BASE" clean="$CLEAN"
echo "== R2 swap $(date +%T)"; "$SP/measure.sh" "$SP/results/r2" base2="$BASE" cleanhq0="$CLEAN@MLX_QWEN_MTP_HEAD_QUANT=0"
echo "== R3 swap $(date +%T)"; "$SP/measure.sh" "$SP/results/r3" clean2="$CLEAN"
echo "== swap done $(date +%T)"
