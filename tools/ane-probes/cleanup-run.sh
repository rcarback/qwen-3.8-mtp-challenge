#!/usr/bin/env bash
# Remove what an ANE probe, sweep or serve run leaves behind. Idempotent; run
# it at the end of every run and before any timed arm. It deletes only
# generated, regenerable data:
#   - compiled Core ML models the probes leave in the user temp dir
#     (*.mlmodelc, mf-*, bank_*), each 85 MB to 812 MB
#   - the in-memory ANE staging directories a crashed arm leaves behind
#     (<hex>_<hex>_<hex>, up to 167 MB each; a clean load removes its own)
#   - the ANE Metal pipeline cache in the user temp dir (6.7 GB seen)
#   - any package set or generator scratch dir passed as an argument
#     (for example the r probe's packages once the r run has finished, or a
#     bank generator's tmp-S<bucket> directory)
# It never touches model trees, the repository, or anything outside the user
# temp dir and the paths given on the command line.
#
# usage: cleanup-run.sh [PATH ...]
set -euo pipefail

temp_dir="$(getconf DARWIN_USER_TEMP_DIR)"
before="$(df -k / | tail -1 | awk '{print $4}')"

python3 - "$temp_dir" "$@" <<'PY'
import os, re, shutil, sys

temp_dir = sys.argv[1]
extra = sys.argv[2:]
freed = 0

def size_of(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total

def remove(path, why):
    global freed
    if not os.path.lexists(path):
        return
    size = size_of(path)
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except OSError as exc:
        print(f"skip {path}: {exc}")
        return
    freed += size
    print(f"removed {path} ({size / 1e9:.2f} GB, {why})")

staging = re.compile(r"^[0-9A-Fa-f]+_[0-9A-Fa-f]+_[0-9A-Fa-f]+$")
for name in os.listdir(temp_dir):
    path = os.path.join(temp_dir, name)
    if name.endswith(".mlmodelc") or name.startswith("mf-") or name.startswith("bank_"):
        remove(path, "compiled probe model")
    elif staging.match(name):
        remove(path, "ANE staging directory")
    elif name == "ane-metal-pipeline-cache":
        remove(path, "ANE Metal pipeline cache")

for path in extra:
    remove(os.path.abspath(path), "given on the command line")

print(f"freed {freed / 1e9:.2f} GB")
PY

after="$(df -k / | tail -1 | awk '{print $4}')"
echo "free: $((before / 1048576)) GiB -> $((after / 1048576)) GiB"
