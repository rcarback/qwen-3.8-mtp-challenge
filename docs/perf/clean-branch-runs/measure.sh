#!/usr/bin/env bash
# Counterbalanced arms of ./benchmark-qwen-mtp.sh --local-iterate at the ranked window
# (512 decode tokens, offered depth 8) over base-generated goldens.
# Usage: measure.sh RESULTS_DIR ARMSPEC...   where ARMSPEC = name=/path/to/tree[@VAR=val[,VAR=val]]
# Env: PROMPTS (space list, default all seven + public), ORDER (ab or ba per prompt alternates by default)
set -uo pipefail
SP=/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad
source "$SP/close-out/env.sh"
GOLDENS=/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/clean/goldens  # env.sh redefines SP, so the goldens path must not derive from it
RES="$1"; shift
mkdir -p "$RES"
export COOL_CEILING=240 QUIET_CEILING=240
bg_pids() { pgrep -x mediaanalysisd; pgrep -x photoanalysisd; }
for p in $(bg_pids); do kill -STOP "$p" 2>/dev/null; done
trap 'for p in $(bg_pids); do kill -CONT "$p" 2>/dev/null; done; pkill -CONT -x mdworker_shared; echo "analysers resumed"' EXIT
( while ! grep -q "^== measure done" "$RES/measure.log" 2>/dev/null; do pkill -STOP -x mdworker_shared; sleep 15; done ) &
PROMPTS="${PROMPTS:-public cooking geology music readme prefill-plan runbook security}"
arms=("$@")
i=0
for prompt in $PROMPTS; do
  if [ "$prompt" = public ]; then golden=correctness_prompts/public_longcopy_gate_english_512_256.json; else golden="$GOLDENS/golden-base-$prompt.json"; fi
  # alternate arm order per prompt: even prompts forward, odd prompts reversed
  if (( i % 2 == 0 )); then order=("${arms[@]}"); else order=(); for (( k=${#arms[@]}-1; k>=0; k-- )); do order+=("${arms[$k]}"); done; fi
  for spec in "${order[@]}"; do
    name="${spec%%=*}"; rest="${spec#*=}"; tree="${rest%%@*}"; extra=""
    if [[ "$rest" == *@* ]]; then extra="${rest#*@}"; extra="${extra//,/ }"; fi
    out="$RES/score-$name-$prompt.json"
    echo "== $name $prompt $(date +%T)" | tee -a "$RES/measure.log"
    quiet_gate
    ( cd "$tree" && env $extra MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS=512 MLXFAST_QWEN_MTP_DEPTH=8 \
        MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE="$golden" MLXFAST_SCORE_PATH="$out" \
        ./benchmark-qwen-mtp.sh --local-iterate ) > "$RES/run-$name-$prompt.log" 2>&1
    rc=$?
    line=$(jq -c '{score, serial: .metrics.serial_seconds_per_token, mtp: .metrics.mtp_seconds_per_token, eff: .metrics.effective_mean_draft_len, acc: .metrics.accepted_draft_rate, matched: .metrics.all_tokens_matched, tripwire: .metrics.public_drift_tripwire_passed}' "$out" 2>/dev/null)
    echo "   rc=$rc ${line:-no payload; $(grep -m1 -i 'FAILED\|error' "$RES/run-$name-$prompt.log" | cut -c1-160)}" | tee -a "$RES/measure.log"
  done
  i=$((i+1))
done
echo "== measure done $(date +%T)" | tee -a "$RES/measure.log"
