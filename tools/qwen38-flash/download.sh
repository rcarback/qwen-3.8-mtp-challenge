#!/usr/bin/env bash
# Download Qwen/Qwen3.8-Flash-Next at the pinned revision and verify sizes and sha256.
# Modes: manifest (write manifest.json from the HF API), download (default), verify.
set -euo pipefail
REPO="Qwen/Qwen3.8-Flash-Next"
REVISION="de4b8e4d43b917e7706784d8bb445c9af86a3540"
HERE="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$HERE/manifest.json"
DEST="${QWEN38_FLASH_SOURCE:-$HOME/.cache/mlxfast/qwen3.8-flash-next/source}"
MODE="${1:-download}"

need() { command -v "$1" >/dev/null 2>&1 || {
  echo "missing tool: $1" >&2
  exit 2
}; }
need curl
need python3
need shasum

write_manifest() {
  curl -sfL "https://huggingface.co/api/models/${REPO}/revision/${REVISION}?blobs=true" |
    python3 -c '
import json,sys
repo, revision = sys.argv[1], sys.argv[2]
d=json.load(sys.stdin)
rows=[{"path":s["rfilename"],"size":s.get("size"),"sha256":(s.get("lfs") or {}).get("sha256")} for s in d["siblings"]]
rows=[r for r in rows if not r["path"].startswith(".git")]
json.dump({"repo":repo,"revision":revision,"files":sorted(rows,key=lambda r:r["path"])},sys.stdout,indent=1)
' "$REPO" "$REVISION" >"$MANIFEST"
  echo "wrote $MANIFEST ($(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['files']))" "$MANIFEST") files)"
}

verify() {
  local bad=0
  while IFS=$'\t' read -r path size sha; do
    local f="$DEST/$path"
    if [ ! -f "$f" ]; then
      echo "MISSING $path"
      bad=1
      continue
    fi
    local have
    have=$(stat -f %z "$f")
    if [ "$size" != "None" ] && [ "$have" != "$size" ]; then
      echo "SIZE $path have=$have want=$size"
      bad=1
      continue
    fi
    if [ "$sha" != "None" ]; then
      local got
      got=$(shasum -a 256 "$f" | cut -d' ' -f1)
      if [ "$got" != "$sha" ]; then
        echo "SHA $path"
        bad=1
      fi
    fi
  done < <(python3 -c '
import json,sys
for r in json.load(open(sys.argv[1]))["files"]:
    print(r["path"], r["size"], r["sha256"], sep="\t")
' "$MANIFEST")
  if [ "$bad" = 0 ]; then
    echo "verify OK: $DEST"
  else
    echo "verify FAILED" >&2
    exit 1
  fi
}

download() {
  [ -f "$MANIFEST" ] || write_manifest
  mkdir -p "$DEST"
  local free_kb
  free_kb=$(df -k "$DEST" | tail -1 | awk '{print $4}')
  local need_kb
  need_kb=$(python3 -c '
import json,sys;print(sum((r["size"] or 0) for r in json.load(open(sys.argv[1]))["files"])//1024 + 1048576)' "$MANIFEST")
  if [ "$free_kb" -lt "$need_kb" ]; then
    echo "need $((need_kb / 1048576)) GiB free at $DEST, have $((free_kb / 1048576)) GiB" >&2
    exit 3
  fi
  if command -v hf >/dev/null 2>&1; then
    hf download "$REPO" --revision "$REVISION" --local-dir "$DEST"
  else
    while IFS= read -r path; do
      mkdir -p "$DEST/$(dirname "$path")"
      curl -fL --retry 5 --retry-delay 10 -C - -o "$DEST/$path" \
        "https://huggingface.co/${REPO}/resolve/${REVISION}/${path}"
    done < <(python3 -c 'import json,sys;[print(r["path"]) for r in json.load(open(sys.argv[1]))["files"]]' "$MANIFEST")
  fi
  verify
}

case "$MODE" in
manifest) write_manifest ;;
download) download ;;
verify) verify ;;
*)
  echo "usage: $0 [manifest|download|verify]" >&2
  exit 2
  ;;
esac
