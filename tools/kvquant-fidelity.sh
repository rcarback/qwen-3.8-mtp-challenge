#!/usr/bin/env bash
# Compare emitted token streams across KV cache configurations on one long
# prompt. Greedy decoding makes the length of the identical prefix a direct
# fidelity measure against the bfloat16 reference.
#
# Usage: tools/kvquant-fidelity.sh <prompt-file> [max-tokens]
set -euo pipefail

PROMPT_FILE="${1:?usage: kvquant-fidelity.sh <prompt-file> [max-tokens]}"
MAX_TOKENS="${2:-256}"
PORT=8099
BINARY=".build/release/mlxfast-swift"
HEAD="${MLXFAST_MTP_HEAD:-$HOME/.cache/mlxfast/declared-q2q4-rerank}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [[ ! -x "$BINARY" ]]; then
  echo "build first: swift build -c release --force-resolved-versions" >&2
  exit 1
fi

# The reference leg must run first so later legs have something to diff.
# An empty bits field disables the policy entirely, which is the bf16 path.
CONFIGS=(
  "bf16::"
  "q8-rot:8:1"
  "q4-rot:4:1"
  "q4-plain:4:0"
  "q3-rot:3:1"
  "q3-plain:3:0"
  "q2-rot:2:1"
)

run_config() {
  local label="$1" bits="$2" rotate="$3" out="$4"
  local env_args=()
  if [[ -n "$bits" ]]; then
    env_args+=("DARKBLOOM_KV_QUANT_BITS=$bits")
    env_args+=("DARKBLOOM_KV_QUANT_GROUP=64")
    # Quantize from the first token so short test prompts still exercise it.
    env_args+=("DARKBLOOM_KV_QUANT_MIN_OFFSET=0")
    env_args+=("DARKBLOOM_KV_QUANT_ROTATE=$rotate")
  fi

  env ${env_args[@]+"${env_args[@]}"} "$BINARY" serve \
    --weights weights --mtp-head "$HEAD" --mtp-depth 2 \
    --max-tokens "$MAX_TOKENS" --port "$PORT" \
    >"$WORK/$label.serve.log" 2>&1 &
  local pid=$!

  # Model load is minutes, not seconds. Poll rather than sleeping blind.
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
    echo "serve failed to start for $label; see $WORK/$label.serve.log" >&2
    kill "$pid" 2>/dev/null || true
    return 1
  fi

  jq -n --rawfile p "$PROMPT_FILE" --argjson n "$MAX_TOKENS" \
    '{model:"qwen", temperature:0, max_tokens:$n,
      messages:[{role:"user", content:$p}]}' |
    curl -fsS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'content-type: application/json' --data-binary @- |
    jq -r '.choices[0].message.content' >"$out"

  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  # One model residency at a time: do not start the next leg until this
  # process has actually released its 14 GiB.
  sleep 5
}

for entry in "${CONFIGS[@]}"; do
  IFS=: read -r label bits rotate <<<"$entry"
  run_config "$label" "$bits" "$rotate" "$WORK/$label.txt"

  # Identical prefix in characters. Token-level would be better, but the
  # completions endpoint returns text and character prefix length is a
  # monotone proxy that needs no tokenizer.
  if [[ "$label" == "bf16" ]]; then
    prefix=$(wc -c <"$WORK/bf16.txt" | tr -d ' ')
  else
    prefix=$(cmp -l "$WORK/bf16.txt" "$WORK/$label.txt" 2>/dev/null |
      head -1 | awk '{print $1-1}')
    if [[ -z "$prefix" ]]; then
      prefix=$(wc -c <"$WORK/bf16.txt" | tr -d ' ')
    fi
  fi

  rate=$(grep -o '[0-9.]* tok/s decode' "$WORK/$label.serve.log" |
    tail -1 | awk '{print $1}')
  jq -nc --arg label "$label" --arg bits "${bits:-16}" \
    --arg rotate "${rotate:-n/a}" --argjson prefix "${prefix:-0}" \
    --arg rate "${rate:-unknown}" \
    --argjson total "$(wc -c <"$WORK/bf16.txt" | tr -d ' ')" \
    '{label:$label, bits:$bits, rotate:$rotate,
      identical_prefix_chars:$prefix, reference_chars:$total,
      decode_tok_s:$rate}'
done
