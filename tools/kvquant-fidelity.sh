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
# serve runs the model inside the sandboxed worker: the trusted binary links
# no model code, so this is the binary that carries every edit under test.
WORKER=".build-worker/release/mlxfast-runtime-worker"
HEAD="${MLXFAST_MTP_HEAD:-$HOME/.cache/mlxfast/declared-q2q4-rerank}"
WORK="${MLXFAST_KVQUANT_WORK:-.local/kvquant-work}"
mkdir -p "$WORK"

if [[ ! -x "$BINARY" ]]; then
  echo "build first: swift build -c release --force-resolved-versions" >&2
  exit 1
fi

# A stale binary is the worst failure this script can have: every leg still
# runs, still emits plausible numbers, and silently measures code that is not
# in the working tree. Refuse rather than measure the wrong build.
if [[ ! -x "$WORKER" ]]; then
  echo "build the worker first:" >&2
  echo "  swift build -c release --scratch-path .build-worker \\" >&2
  echo "    --product mlxfast-runtime-worker --force-resolved-versions" >&2
  exit 1
fi
STALE="$(find Sources Vendor/mlx-swift-lm/Libraries -name '*.swift' \
  -newer "$WORKER" -print -quit 2>/dev/null)"
if [[ -n "$STALE" ]]; then
  echo "$WORKER is older than $STALE" >&2
  echo "rebuild the worker:" >&2
  echo "  swift build -c release --scratch-path .build-worker \\" >&2
  echo "    --product mlxfast-runtime-worker --force-resolved-versions" >&2
  exit 1
fi

# The reference leg must run first so later legs have something to diff.
# An empty bits field disables the policy entirely, which is the bf16 path.
CONFIGS=(
  "bf16:::"
  "q8-plain-slow:8:0:0"
  "q8-rot-slow:8:1:0"
  "q8-rot-fused:8:1:1"
  "q4-plain-slow:4:0:0"
  "q4-rot-slow:4:1:0"
  "q4-rot-fused:4:1:1"
  "q4-plain-fused:4:0:1"
)

run_config() {
  local label="$1" bits="$2" rotate="$3" fused="$4" out="$5"
  local env_args=()
  if [[ -n "$bits" ]]; then
    env_args+=("DARKBLOOM_KV_QUANT_BITS=$bits")
    env_args+=("DARKBLOOM_KV_QUANT_GROUP=64")
    # Quantize from the first token so short test prompts still exercise it.
    env_args+=("DARKBLOOM_KV_QUANT_MIN_OFFSET=0")
    env_args+=("DARKBLOOM_KV_QUANT_ROTATE=$rotate")
    # 0 forces the decomposed path so the fused kernel can be A/B compared.
    env_args+=("DARKBLOOM_KV_FUSED_SDPA=$fused")
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

  local status=0
  jq -n --rawfile p "$PROMPT_FILE" --argjson n "$MAX_TOKENS" \
    '{model:"qwen", temperature:0, max_tokens:$n,
      messages:[{role:"user", content:$p}]}' \
    >"$WORK/$label.request.json"
  # No -f: a 4xx/5xx body is the diagnosis, and -f discards it. Check the
  # status code explicitly instead.
  local code
  code=$(curl -sS -o "$WORK/$label.response.json" -w '%{http_code}' \
    -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' \
    --data-binary "@$WORK/$label.request.json" 2>"$WORK/$label.curl.log") || status=$?
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 5
  if [[ "$status" -ne 0 || "$code" != "200" ]]; then
    echo "$label: request failed (curl=$status http=$code)" >&2
    echo "  response: $WORK/$label.response.json" >&2
    echo "  serve log: $WORK/$label.serve.log" >&2
    return 1
  fi
  jq -r '.choices[0].message.content' <"$WORK/$label.response.json" >"$out"

}

for entry in "${CONFIGS[@]}"; do
  IFS=: read -r label bits rotate fused <<<"$entry"
  if ! run_config "$label" "$bits" "$rotate" "$fused" "$WORK/$label.txt"; then
    jq -nc --arg label "$label" --arg bits "${bits:-16}" \
      --arg rotate "${rotate:-n/a}" --arg fused "${fused:-n/a}" \
      '{label:$label, bits:$bits, rotate:$rotate, fused:$fused,
        error:"leg failed"}'
    if [[ "$label" == "bf16" ]]; then
      echo "reference leg failed; the remaining legs have nothing to diff" >&2
      exit 1
    fi
    continue
  fi

  # Identical prefix in characters. Token-level would be better, but the
  # completions endpoint returns text and character prefix length is a
  # monotone proxy that needs no tokenizer.
  if [[ "$label" == "bf16" ]]; then
    prefix=$(wc -c <"$WORK/bf16.txt" | tr -d ' ')
  else
    # cmp -l exits 1 whenever the files differ, which is the expected result
    # for every quantized leg. Under set -e and pipefail an assignment from a
    # failing substitution kills the script with no message, so absorb the
    # status here: a difference is this script's measurement, not its error.
    prefix=$({ cmp -l "$WORK/bf16.txt" "$WORK/$label.txt" 2>/dev/null || true; } |
      head -1 | awk '{print $1-1}')
    if [[ -z "$prefix" ]]; then
      prefix=$(wc -c <"$WORK/bf16.txt" | tr -d ' ')
    fi
  fi

  # Divergence from bf16 answers "how much did quantization cost". It does not
  # answer "did the fused kernel change anything", because two legs that
  # diverge from bf16 at the same character are not thereby the same stream.
  # Compare a fused leg against the decomposed leg it is an A/B of, directly.
  partner=""
  case "$label" in
  *-fused) partner="${label%-fused}-slow" ;;
  esac
  paired=""
  if [[ -n "$partner" && -f "$WORK/$partner.txt" ]]; then
    paired=$({ cmp -l "$WORK/$partner.txt" "$WORK/$label.txt" 2>/dev/null || true; } |
      head -1 | awk '{print $1-1}')
    if [[ -z "$paired" ]]; then
      paired=$(wc -c <"$WORK/$label.txt" | tr -d ' ')
    fi
  fi

  # grep exits 1 when the log has no rate line, which is not fatal either.
  rate=$({ grep -o '[0-9.]* tok/s decode' "$WORK/$label.serve.log" || true; } |
    tail -1 | awk '{print $1}')
  jq -nc --arg label "$label" --arg bits "${bits:-16}" \
    --arg rotate "${rotate:-n/a}" --arg fused "${fused:-n/a}" \
    --arg partner "${partner:-n/a}" --arg paired "${paired:-n/a}" \
    --argjson prefix "${prefix:-0}" \
    --arg rate "${rate:-unknown}" \
    --argjson total "$(wc -c <"$WORK/bf16.txt" | tr -d ' ')" \
    '{label:$label, bits:$bits, rotate:$rotate, fused:$fused,
      identical_prefix_chars:$prefix, reference_chars:$total,
      ab_partner:$partner, paired_prefix_chars:$paired,
      decode_tok_s:$rate}'
done
