#!/usr/bin/env bash
# E2: M1 crossing, M2 gated-delta projections, M3 int8 distribution.
# Converts first (not timed), then measures one at a time.
set -euo pipefail
cd "$(dirname "$0")"
PY="$(dirname "$0")/../venv/bin/python"
export PYTHONUNBUFFERED=1

echo "=== E2 convert crossing (M1) ==="
"$PY" convert_crossing.py

echo "=== E2 convert GDN projections (M2) ==="
"$PY" convert_gdn.py

echo "=== E2 M1 crossing measure ==="
"$PY" measure_crossing.py

echo "=== E2 M3 int8 distribution ==="
"$PY" m3_int8_dist.py --n 32

echo "=== E2 M2 GDN projection measure ==="
"$PY" measure_gdn.py

echo "=== E2 done ==="
ls -la artifacts/
