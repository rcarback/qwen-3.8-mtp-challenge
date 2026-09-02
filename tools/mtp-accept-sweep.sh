#!/usr/bin/env bash
# Multi-prompt MTP accept-rate matrix over weight configurations.
#
# For each prompt file and each config, generate a self-consistent golden
# with that config's flags, then run benchmark-qwen-mtp.sh --local-iterate
# against it at each depth. The wrapper removes its run directory on exit,
# so the score payload is the record; its metrics carry the MTP leg.
#
# Usage: tools/mtp-accept-sweep.sh [--dry-run] OUT_DIR PROMPT_FILE...
# Env:   MLXFAST_ACCEPT_CONFIGS  space-separated subset of: gpu dequant bf16 (default all)
#        MLXFAST_ACCEPT_DEPTHS   space-separated depths (default "8 2")
#        MLXFAST_ACCEPT_STEPS    golden steps (default 200)
#        MLX_ANE_BF16_WEIGHTS    required when bf16 is in the config set
set -euo pipefail

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=1; shift; fi
if [[ $# -lt 2 ]]; then
  echo "usage: $0 [--dry-run] OUT_DIR PROMPT_FILE..." >&2
  exit 2
fi
out_dir="$1"; shift
prompts=("$@")
configs="${MLXFAST_ACCEPT_CONFIGS:-gpu dequant bf16}"
depths="${MLXFAST_ACCEPT_DEPTHS:-8 2}"
steps="${MLXFAST_ACCEPT_STEPS:-200}"
swift_bin="${MLXFAST_SWIFT_BIN:-.build-worker/release/mlxfast-swift}"
stamp="$(date +%Y%m%d-%H%M%S)"
csv="${out_dir}/mtp-accept-${stamp}.csv"

config_env() {
  # Prints the env assignments for a config, one per line.
  case "$1" in
    gpu) ;;
    dequant) echo "MLXFAST_NO_SANDBOX=1"; echo "MLX_ANE_DIRECT=1" ;;
    bf16)
      [[ -n "${MLX_ANE_BF16_WEIGHTS:-}" ]] || { echo "bf16 needs MLX_ANE_BF16_WEIGHTS" >&2; exit 2; }
      echo "MLXFAST_NO_SANDBOX=1"; echo "MLX_ANE_DIRECT=1"
      echo "MLX_ANE_BF16_WEIGHTS=${MLX_ANE_BF16_WEIGHTS}" ;;
    *) echo "unknown config: $1" >&2; exit 2 ;;
  esac
}

run() {
  if [[ "$dry_run" == "1" ]]; then printf '%q ' "$@"; echo; else "$@"; fi
}

mkdir -p "$out_dir"
echo "config,prompt,depth,accepted_draft_rate,effective_mean_draft_len,serial_seconds_per_token,mtp_seconds_per_token,mtp_decode_speedup,all_tokens_matched" > "$csv"

for prompt in "${prompts[@]}"; do
  pname="$(basename "${prompt%.*}")"
  for cfg in $configs; do
    # Portable read loop: macOS ships bash 3.2, which lacks mapfile (bash 4+).
    cfg_env=()
    while IFS= read -r line; do cfg_env+=("$line"); done < <(config_env "$cfg")
    golden="${out_dir}/golden-${cfg}-${pname}.json"
    # Golden generation is model-holding: one at a time, in-process worker.
    run env "${cfg_env[@]+"${cfg_env[@]}"}" "$swift_bin" generate-golden \
      --prompt-file "$prompt" --weights weights --output "$golden" \
      --name "${cfg}_${pname}" --steps "$steps"
    for depth in $depths; do
      score="${out_dir}/score-${cfg}-${pname}-d${depth}.json"
      run env "${cfg_env[@]+"${cfg_env[@]}"}" \
        MLXFAST_SWIFT_BIN="$swift_bin" \
        MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS=128 \
        MLXFAST_QWEN_MTP_DEPTH="$depth" \
        MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE="$golden" \
        MLXFAST_QWEN_MTP_LOCAL_WORK_DIR="${out_dir}/work-${cfg}-${pname}-d${depth}" \
        MLXFAST_SCORE_PATH="$score" \
        ./benchmark-qwen-mtp.sh --local-iterate
      if [[ "$dry_run" == "0" ]]; then
        python3 - "$score" "$cfg" "$pname" "$depth" >> "$csv" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["metrics"]
print(",".join([sys.argv[2], sys.argv[3], sys.argv[4]] + [str(m[k]) for k in (
    "accepted_draft_rate", "effective_mean_draft_len", "serial_seconds_per_token",
    "mtp_seconds_per_token", "mtp_decode_speedup", "all_tokens_matched")]))
PY
      fi
    done
  done
done
echo "wrote $csv"
