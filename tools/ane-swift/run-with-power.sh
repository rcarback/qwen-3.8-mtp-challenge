#!/bin/bash
# Runs anebench with macmon sampling CONCURRENTLY for the whole run, then
# reports the peak of each power rail alongside the timing line.
#
# An earlier version invoked macmon sequentially before/after the benchmark.
# Those windows did not overlap the work, and it produced the tell-tale
# nonsense of a 1.6 W ANE reading during a GPU-mode run. Placement claims are
# only as good as the window they were sampled in.
#
# Sample files are left in TMPDIR on purpose; they are the raw evidence for
# any placement claim made from this script.
set -uo pipefail
MODE=${1:?mode}
TMP=$(mktemp -t macmon_$MODE)
macmon pipe -s 0 -i 200 >"$TMP" 2>/dev/null &
MM=$!
sleep 1
ANE_MODE="$MODE" ./anebench >/tmp/bench_$MODE.txt 2>&1
BRC=$?
sleep 1
kill $MM 2>/dev/null; wait $MM 2>/dev/null
ane=$(grep -oE '"ane_power":[0-9.]*' "$TMP" | cut -d: -f2 | sort -rn | head -1)
gpu=$(grep -oE '"gpu_power":[0-9.]*' "$TMP" | cut -d: -f2 | sort -rn | head -1)
n=$(grep -c ane_power "$TMP")
printf "mode=%-4s samples=%-4s peakANE=%-8s peakGPU=%-8s | %s\n" \
  "$MODE" "$n" "${ane:-NA}" "${gpu:-NA}" \
  "$(grep ANEPOINT /tmp/bench_$MODE.txt | sed 's/^ANEPOINT\t//')"
echo "  raw samples: $TMP"
exit $BRC
