#!/bin/bash
# Refuses to proceed while the host is too noisy to time on.
#
# The screensaver is the specific hazard this exists for. Its idle timer is a
# separate clock from display sleep, so `caffeinate -d` does not suppress it,
# and `loginwindow` respawns it within seconds of a `pkill`, so a watchdog does
# not either. The only durable fix is idleTime 0, which this script verifies
# rather than attempts: writing it needs the user's own shell.
#
# THE BINDING SIGNAL IS THE GPU, NOT THE CPU (revised 2026-08-26). The first
# version vetoed on any unrelated process above 25% CPU, and that veto fired on
# `WindowServer` and the terminal emulator while the GPU sat at its 0.35 W idle
# floor -- including on the terminal this harness prints into, which makes the
# gate refuse partly because of its own output. Compositing a desktop is not
# contention for a prefill that pulls the GPU to full power. The screensaver
# was never dangerous because it used a CPU; it was dangerous because it drove
# GPU power from 0.3 W to 9.2 W. So the GPU is sampled over a window, which
# catches a PERIODIC spike that a single reading would miss, and the CPU check
# is kept only for a genuine compute hog outside the windowing stack.
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

# Compute hogs only. WindowServer, the terminal emulator, and the browser
# render process are excluded BY NAME because they are windowing-stack load
# whose GPU cost the sampled window below measures directly and far better.
busy=$(ps -Ao pcpu=,comm= -r | head -12 |
  awk '$1 > 40.0 && $2 !~ /mlxfast|swift|clang|ld$|WindowServer|ghostty|Terminal|iTerm|Renderer|plugin-container|claude/ { print $2 " " $1 }')
if [ -n "$busy" ]; then
  echo "host-quiet-gate: unrelated compute over 40% CPU:" >&2
  echo "$busy" >&2
  exit 1
fi

if ! command -v macmon >/dev/null 2>&1; then
  fail "macmon is absent, so the GPU cannot be checked; install it with 'brew install macmon'"
fi

# Sample across a window rather than once. A single reading cannot distinguish
# a quiet machine from one between spikes, which is exactly the shape of the
# five-minute screensaver cycle that spoiled two earlier sweeps.
peak_power=0
peak_temp=0
while read -r p t; do
  [ -z "$p" ] && continue
  over=$(echo "$p > $peak_power" | bc -l 2>/dev/null || echo 0)
  [ "$over" = "1" ] && peak_power=$p
  over=$(echo "$t > $peak_temp" | bc -l 2>/dev/null || echo 0)
  [ "$over" = "1" ] && peak_temp=$t
done < <(macmon pipe -s 8 -i 1000 2>/dev/null |
  jq -r '"\(.gpu_power) \(.temp.gpu_temp_avg)"')

[ "$peak_power" = "0" ] && fail "macmon produced no samples"

ok=$(echo "$peak_power < 2.0 && $peak_temp < 53.0" | bc -l 2>/dev/null || echo 0)
[ "$ok" = "1" ] || fail "gpu not idle over 8s: peak ${peak_power}W ${peak_temp}C"
printf "host-quiet-gate: ok (gpu peak %.2fW %.1fC over 8s)\n" "$peak_power" "$peak_temp"
