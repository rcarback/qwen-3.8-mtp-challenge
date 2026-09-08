#!/usr/bin/env bash
# Rerun every r6 arm the local cool gate voided, with the local gate off. Evidence for
# the switch: on this box the gate passes only while the GPU sensor reports a power-gated
# 1.6C, so every gated arm to date was effectively ungated, and those arms repeated to a
# few tenths of a percent between rounds. Waits for campaign-declared.sh.
set -uo pipefail
SP=/private/tmp/claude-501/-Users-carback1-Code-llms-qwen-3-8-mtp-challenge/3bff0647-7d6b-423f-9179-7e750d574fa5/scratchpad/clean
BASE=/Users/carback1/Code/llms/qwen-base.noindex
CLEAN=/Users/carback1/Code/llms/qwen-clean.noindex
DH=/Users/carback1/.cache/mlxfast/declared-head-ae628274
while ! grep -q '^== r6 done' "$SP/campaign-declared.log" 2>/dev/null; do sleep 60; done
while pgrep -f 'mlxfast-swift|mlxfast-runtime-worker|swift-build' >/dev/null; do sleep 15; done
export MLXFAST_LOCAL_COOL_GATE=0
voided="$(awk '/^== / {name=$2; prompt=$3} /rc=1/ {print name" "prompt}' "$SP/results/r6/measure.log" | sort -u)"
echo "== voided arms:"; echo "$voided" | sed 's/^/   /'
mv "$SP/results/r6/measure.log" "$SP/results/r6/measure-before-fill.log"
while read -r name prompt; do
  [ -n "$name" ] || continue
  case "$name" in
    basedh)    spec="basedh=$BASE@MLXFAST_QWEN_MTP_HEAD_DIR=$DH" ;;
    cleandh)   spec="cleandh=$CLEAN@MLXFAST_QWEN_MTP_HEAD_DIR=$DH" ;;
    clean4pin) spec="clean4pin=$CLEAN" ;;
    *) echo "   unknown arm $name, skipped"; continue ;;
  esac
  echo "== rerun $name $prompt $(date +%T)"
  PROMPTS="$prompt" "$SP/measure.sh" "$SP/results/r6" "$spec"
done <<< "$voided"
echo "== r6fill done $(date +%T)"
