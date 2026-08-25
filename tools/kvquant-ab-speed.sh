#!/usr/bin/env bash
# Paired A/B decode-rate measurement for the fused quantized attention kernel.
#
# The fidelity harness runs every configuration once, in sequence, over the best
# part of an hour. That cannot resolve a ten percent effect: measured drift
# across its legs was as large as the largest effect it was trying to see.
#
# This runs the two configurations alternately inside a round, so drift lands on
# both legs, and flips which one goes first between rounds, so any advantage
# from running second cancels. The reported figure is the median of the
# per-round paired ratios, never a ratio between legs from different rounds.
#
# Usage: tools/kvquant-ab-speed.sh <prompt-file> [max-tokens] [rounds]
set -euo pipefail

PROMPT_FILE="${1:?usage: kvquant-ab-speed.sh <prompt-file> [max-tokens] [rounds]}"
MAX_TOKENS="${2:-512}"
ROUNDS="${3:-3}"
PORT=8098
BINARY=".build/release/mlxfast-swift"
WORKER=".build-worker/release/mlxfast-runtime-worker"
HEAD="${MLXFAST_MTP_HEAD:-$HOME/.cache/mlxfast/declared-q2q4-rerank}"
WORK="${MLXFAST_KVQUANT_WORK:-.local/kvquant-ab}"
SETTLE="${MLXFAST_AB_SETTLE:-45}"
mkdir -p "$WORK"

if [[ ! -x "$BINARY" || ! -x "$WORKER" ]]; then
  echo "build both binaries first" >&2
  exit 1
fi
STALE="$(find Sources Vendor/mlx-swift-lm/Libraries -name '*.swift' \
  -newer "$WORKER" -print -quit 2>/dev/null)"
if [[ -n "$STALE" ]]; then
  echo "$WORKER is older than $STALE; rebuild the worker" >&2
  exit 1
fi

# One leg: start serve with the fused kernel forced on or off, send the prompt,
# take the decode rate the worker reports, stop.
run_leg() {
  local fused="$1" tag="$2"
  env DARKBLOOM_KV_QUANT_BITS=4 DARKBLOOM_KV_QUANT_GROUP=64 \
    DARKBLOOM_KV_QUANT_MIN_OFFSET=0 DARKBLOOM_KV_QUANT_ROTATE=1 \
    DARKBLOOM_KV_FUSED_SDPA="$fused" \
    "$BINARY" serve --weights weights --mtp-head "$HEAD" --mtp-depth 2 \
    --max-tokens "$MAX_TOKENS" --port "$PORT" \
    >"$WORK/$tag.serve.log" 2>&1 &
  local pid=$!

  local ready=0
  for _ in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "serve failed to start for $tag; see $WORK/$tag.serve.log" >&2
    kill "$pid" 2>/dev/null || true
    return 1
  fi

  jq -n --rawfile p "$PROMPT_FILE" --argjson n "$MAX_TOKENS" \
    '{model:"qwen", temperature:0, max_tokens:$n,
      messages:[{role:"user", content:$p}]}' >"$WORK/$tag.request.json"
  local code status=0
  code=$(curl -sS -o "$WORK/$tag.response.json" -w '%{http_code}' \
    -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' \
    --data-binary "@$WORK/$tag.request.json" 2>/dev/null) || status=$?
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  if [[ "$status" -ne 0 || "$code" != "200" ]]; then
    echo "$tag: request failed (curl=$status http=$code)" >&2
    return 1
  fi
  # grep exits non-zero when the log carries no rate line, which under set -e
  # would kill this script silently. Absorb it and let the caller see an empty
  # rate instead.
  { grep -o '[0-9.]* tok/s decode' "$WORK/$tag.serve.log" || true; } |
    tail -1 | awk '{print $1}'
}

for round in $(seq 1 "$ROUNDS"); do
  # Odd rounds run the decomposed path first, even rounds run it second, so a
  # systematic advantage to whichever goes first cancels across a pair of rounds.
  if ((round % 2 == 1)); then
    order=("0:slow" "1:fused")
  else
    order=("1:fused" "0:slow")
  fi
  # macOS ships bash 3.2, which has no associative arrays.
  rate_slow=0
  rate_fused=0
  for entry in "${order[@]}"; do
    IFS=: read -r fused name <<<"$entry"
    sleep "$SETTLE"
    r=$(run_leg "$fused" "r${round}-${name}")
    if [[ "$name" == "slow" ]]; then rate_slow="${r:-0}"; else rate_fused="${r:-0}"; fi
    jq -nc --argjson round "$round" --arg leg "$name" \
      --arg rate "${r:-unknown}" '{round:$round, leg:$leg, decode_tok_s:$rate}'
  done
  jq -nc --argjson round "$round" \
    --arg slow "$rate_slow" --arg fused "$rate_fused" \
    '{round:$round, slow:($slow|tonumber), fused:($fused|tonumber),
      ratio:(if ($slow|tonumber) > 0 then (($fused|tonumber)/($slow|tonumber)) else null end)}'
done
