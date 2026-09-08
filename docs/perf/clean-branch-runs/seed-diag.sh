#!/usr/bin/env bash
# Seed-token diagnostic for the two goldens whose depth-0 serial control failed at
# step 0 on the clean tree (prefill-plan: 2695 vs 7793; security: 27370 vs 2531).
# For each tree and prompt: keep the verify-generate rows (row 0 carries the top-2
# tokens and logits of the seed position), then run mtp-timed at depth 0 and 8
# against those rows and record the outcome. Runs after campaign-fill.sh is done.
set -uo pipefail
SP=/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/clean
D="$SP/results/seed-diag"; mkdir -p "$D"
HEAD=/Users/carback1/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head
while ! grep -q '^== fill done' "$SP/campaign-fill.log" 2>/dev/null; do sleep 60; done
while pgrep -f 'mlxfast-swift|mlxfast-runtime-worker' >/dev/null; do sleep 30; done
for tree in base clean; do
  T=/Users/carback1/Code/llms/qwen-$tree.noindex
  for prompt in prefill-plan security; do
    tag="$tree-$prompt"
    jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' "$SP/goldens/golden-base-$prompt.json" > "$D/plan-$prompt.json"
    echo "== verify $tag $(date +%T)"
    ( cd "$T" && .build/release/mlxfast-swift mtp-verify --weights weights --mtp-head "$HEAD" \
        --emitted "$D/plan-$prompt.json" --generate 17 --mtp-depth 8 \
        --output "$D/rows-$tag.json" --plan-output "$D/gen-$tag.json" ) > "$D/verify-$tag.log" 2>&1
    echo "   verify rc=$? $(grep -o 'reference_seed_token=[0-9]*' "$D/verify-$tag.log" | head -1)"
    for depth in 0 8; do
      echo "== timed depth $depth $tag $(date +%T)"
      ( cd "$T" && .build/release/mlxfast-swift mtp-timed --weights weights --mtp-head "$HEAD" \
          --golden "$D/rows-$tag.json" --tokens 16 --mtp-depth "$depth" ) > "$D/timed$depth-$tag.json" 2> "$D/timed$depth-$tag.err"
      echo "   timed$depth rc=$? $(grep -o 'seed prefill token [0-9]* disagreed with the reference.s [0-9]*' "$D/timed$depth-$tag.err" | head -1)"
    done
    echo "   row0: $(jq -c '(.rows // .reference_rows // [])[0] | {top2_tokens, top2_logits, token, emitted_token, logits: (.reference_emitted_token_logits // .logits)}' "$D/rows-$tag.json" 2>/dev/null | cut -c1-300)"
  done
done
echo "== seed-diag done $(date +%T)"
