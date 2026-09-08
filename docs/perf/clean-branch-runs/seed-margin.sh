#!/usr/bin/env bash
# Top-5 logits at step 0 (the seed position) on the golden path, base tree, for the
# three prompts whose timed-verb seed token flipped on both trees and one that passed.
# Runs after campaign-r4.sh; one model-holding process at a time.
set -uo pipefail
SP=/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/clean
BASE=/Users/carback1/Code/llms/qwen-base.noindex
D="$SP/results/seed-margin"; mkdir -p "$D"
while ! grep -q '^== r4 done' "$SP/campaign-r4.log" 2>/dev/null; do sleep 60; done
while pgrep -f 'mlxfast-swift|mlxfast-runtime-worker' >/dev/null; do sleep 30; done
for p in prefill-plan security sailing cooking; do
  echo "== trace $p step 0 $(date +%T)"
  ( cd "$BASE" && .build/release/mlxfast-swift correctness-trace --weights weights --golden "$SP/goldens/golden-base-$p.json" --step 0 --top-k 5 ) > "$D/trace-$p.json" 2> "$D/trace-$p.err"
  echo "   rc=$? $(python3 -c '
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception as e:
    print("unparsable:", e); sys.exit()
def find(o, depth=0):
    if isinstance(o, dict):
        keys=[k for k in o if "top" in k.lower() or "logit" in k.lower() or "margin" in k.lower() or "expected" in k.lower()]
        if keys: print({k:o[k] for k in keys})
        for v in o.values(): find(v, depth+1)
    elif isinstance(o, list) and depth < 3:
        for v in o[:3]: find(v, depth+1)
find(d)' "$D/trace-$p.json" | head -4 | cut -c1-400)"
done
echo "== seed-margin done $(date +%T)"
