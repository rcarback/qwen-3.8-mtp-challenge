#!/bin/bash
# Measures one (prelude, measurement) pair per process.
#
# The whole point of this probe is that position inside a process changes GPU
# timings by ~2.3x, so anything measured after something else in the same
# process is not comparable. One point per process is the only control that
# works; see GPUPositionEffectTests for the observation this isolates.
#
# Usage: tools/position-effect-sweep.sh <points-file> <out.tsv>
# Each line: <prelude> <measure> <clear:0|1>
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

POINTS=${1:?points file}
OUT=${2:?output tsv}

printf "prelude\tmeasure\tcache\tms\tmetric\tunit\n" >"$OUT"

while read -r prelude measure clear; do
  case "$prelude" in '' | '#'*) continue ;; esac
  quiet=0
  for _ in $(seq 1 60); do
    if ./tools/host-quiet-gate.sh >/dev/null 2>&1; then quiet=1; break; fi
    sleep 15
  done
  [ "$quiet" = "1" ] || { echo "host never went quiet before $prelude/$measure" >&2; exit 1; }
  line=$(MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
    MLXFAST_POS_PRELUDE="$prelude" MLXFAST_POS_MEASURE="$measure" \
    MLXFAST_POS_CLEARCACHE="$clear" \
    swift test --force-resolved-versions --filter positionEffectPoint </dev/null 2>&1 |
    grep '^POSPOINT' | head -1)
  if [ -z "$line" ]; then echo "no measurement for $prelude/$measure" >&2; exit 1; fi
  printf "%s\n" "${line#POSPOINT	}" >>"$OUT"
  echo "$prelude/$measure/$clear -> ${line#POSPOINT	}"
done <"$POINTS"

echo "done"
