#!/bin/bash
# One CPU column fraction per process. See ColumnSplitSpeedTests for why.
# Usage: tools/column-split-sweep.sh <fractions-file> <out.tsv>
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
POINTS=${1:?fractions file}
OUT=${2:?output tsv}
printf "fraction\tM\tN\tK\tcpuCols\tgpuCols\tms\ttflops\n" >"$OUT"
while read -r f; do
  case "$f" in '' | '#'*) continue ;; esac
  quiet=0
  for _ in $(seq 1 60); do
    if ./tools/host-quiet-gate.sh >/dev/null 2>&1; then quiet=1; break; fi
    sleep 15
  done
  [ "$quiet" = "1" ] || echo "WARN: host not quiet before fraction $f" >&2
  line=$(MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_SPLIT_FRACTION="$f" \
    swift test --force-resolved-versions --filter columnSplitSpeedPoint </dev/null 2>&1 |
    grep '^SPLITPOINT' | head -1)
  if [ -z "$line" ]; then echo "no measurement for fraction $f" >&2; exit 1; fi
  printf "%s\n" "${line#SPLITPOINT	}" >>"$OUT"
  echo "f=$f -> ${line#SPLITPOINT	}"
done <"$POINTS"
echo "done"
