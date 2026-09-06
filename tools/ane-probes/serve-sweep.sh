#!/bin/zsh
# One serve arm: start one serve session, send every prompt once (cold), and
# append one TSV row per prompt.
#
# usage: serve-sweep.sh OUT.tsv ARM DEPTH [ENV=VAL ...]
#
# Row: arm, prompt, prompt_tokens, prefill_s, prefill_tok_s, decode_tok_s,
# rounds, accepted, rejected, eff_depth, completion_tokens, request_wall_s.
# The raw response (resp-ARM-PROMPT.json) and the completion text
# (out-ARM-PROMPT.txt) land beside OUT.tsv for agree.py. Environment:
#   WEIGHTS      model tree (default: the Flash-Next q8 tree)
#   HEAD         MTP head directory (default: none, the headless backbone)
#   PROMPTS_DIR  directory of *.txt prompts (default: prompts/ beside OUT.tsv)
#   MAXTOK       completion length (default 128)
#   PROMPT_GAP   seconds to sleep before every prompt, so the engines cool
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; NAME=$2; DEPTH=$3; shift 3
OUTDIR="$(cd "$(dirname "$OUT")" && pwd)"
cd "$REPO" || exit 1
W=${WEIGHTS:-$HOME/.cache/mlxfast/qwen3.8-flash-next/weights-q8}
PORT=${PORT:-8091}
MAXTOK=${MAXTOK:-128}
for kv in "$@"; do export "$kv"; done
export DARKBLOOM_QWEN_GEOMETRY_UNPINNED=1 MLXFAST_NO_SANDBOX=1
.build/release/mlxfast-swift serve --weights "$W" --mtp-head "${HEAD:-none}" --mtp-depth "$DEPTH" --port "$PORT" > "$OUTDIR/serve-$NAME.log" 2>&1 &
SPID=$!
for i in $(seq 1 240); do
  curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/v1/models" && break
  sleep 2
done
sleep 1
for f in "${PROMPTS_DIR:-$OUTDIR/prompts}"/*.txt; do
  p=$(basename "$f" .txt)
  [ "${PROMPT_GAP:-0}" -gt 0 ] && sleep "$PROMPT_GAP"
  python3 - "$f" "$MAXTOK" > "$OUTDIR/body-$p.json" <<'PY'
import json,sys
t=open(sys.argv[1]).read()
print(json.dumps({"model":"q","messages":[{"role":"user","content":t}],"max_tokens":int(sys.argv[2])}))
PY
  t_req0=$(date +%s.%N)
  r=$(curl -s -m 900 -H 'Content-Type: application/json' -d "@$OUTDIR/body-$p.json" "http://127.0.0.1:$PORT/v1/chat/completions")
  wall=$(python3 -c "import time; print(round(time.time()-$t_req0, 2))")
  printf '%s' "$r" > "$OUTDIR/resp-$NAME-$p.json"
  printf '%s' "$r" | python3 -c "
import json,sys
d=json.loads(sys.stdin.read(), strict=False); u=d['usage']; m=u['mtp']
open('$OUTDIR/out-$NAME-$p.txt','w').write(d['choices'][0]['message']['content'])
pt=u['prompt_tokens']; ps=m['seed_prefill_seconds']
print('\t'.join(str(x) for x in ['$NAME','$p',pt,round(ps,3),round(pt/ps,1),round(m['decode_tokens_per_second'],2),m['rounds'],m['accepted_drafts'],m['rejected_drafts'],m.get('effective_draft_depth'),u['completion_tokens'],'$wall']))
" >> "$OUT" 2>> "$OUTDIR/serve-$NAME.log" || echo "$NAME	$p	ERROR" >> "$OUT"
done
kill $SPID; wait $SPID 2>/dev/null
