#!/bin/bash
# Refuses to proceed while the host is too noisy to time on.
#
# The screensaver is the specific hazard this exists for. Its idle timer is a
# separate clock from display sleep, so `caffeinate -d` does not suppress it,
# and `loginwindow` respawns it within seconds of a `pkill`, so a watchdog does
# not either. The only durable fix is idleTime 0, which this script verifies
# rather than attempts: writing it needs the user's own shell.
set -uo pipefail

fail() {
  echo "host-quiet-gate: $1" >&2
  exit 1
}

idle=$(defaults -currentHost read com.apple.screensaver idleTime 2>/dev/null || echo unset)
if [ "$idle" != "0" ]; then
  echo "host-quiet-gate: screensaver idleTime is '$idle', must be 0" >&2
  echo "  run this in your own shell, then retry:" >&2
  echo "    defaults -currentHost write com.apple.screensaver idleTime 0" >&2
  exit 1
fi

if pgrep -x legacyScreenSaver >/dev/null 2>&1; then
  fail "legacyScreenSaver is running"
fi

busy=$(ps -Ao pcpu=,comm= -r | head -5 |
  awk '$1 > 25.0 && $2 !~ /mlxfast|swift|clang|ld$/ { print $2 " " $1 }')
if [ -n "$busy" ]; then
  echo "host-quiet-gate: unrelated processes over 25% CPU:" >&2
  echo "$busy" >&2
  exit 1
fi

if command -v macmon >/dev/null 2>&1; then
  read -r p t < <(macmon pipe -s 1 -i 1 2>/dev/null | head -1 |
    jq -r '"\(.gpu_power) \(.temp.gpu_temp_avg)"')
  ok=$(echo "$p < 2.0 && $t < 53.0" | bc -l 2>/dev/null || echo 0)
  [ "$ok" = "1" ] || fail "gpu not idle: ${p}W ${t}C"
  echo "host-quiet-gate: ok (gpu ${p}W ${t}C)"
else
  echo "host-quiet-gate: ok (macmon absent, gpu not checked)"
fi
