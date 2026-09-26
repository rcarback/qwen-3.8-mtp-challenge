#!/usr/bin/env python3
"""Print the characterisation matrix from measure JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from catalog import ARTIFACTS, build_cells, engine_watts, tflops

core = json.loads((ARTIFACTS / "measure_coreml.json").read_text())
metal = json.loads((ARTIFACTS / "measure_metal.json").read_text())

rows = list(core["results"]) + list(metal["results"])
by = {}
for r in rows:
    if r.get("error") or "ms" not in r:
        continue
    by[(r["tag"], r["backend"], r.get("precision"))] = r


def fmt(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def place(r):
    p = (r or {}).get("placement") or {}
    c = p.get("counts_compute") or {}
    return ",".join(f"{k}:{v}" for k, v in sorted(c.items())) or "—"


print("| cell | gflop | ANE fp16 ms | ANE TFLOPS | ANE W | ANE place | GPU fp16 ms | GPU TFLOPS | GPU W | Metal ms | Metal TFLOPS | Metal W | winner |")
print("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|")

cells = build_cells()
for cell in cells:
    ane = by.get((cell.tag, "ane", "fp16"))
    gpu = by.get((cell.tag, "gpu", "fp16"))
    met = by.get((cell.tag, "metal", "affine-4-group-64")) or by.get(
        (cell.tag, "metal", "fp16")
    )
    ane8 = by.get((cell.tag, "ane", "int8"))

    def pack(r, backend):
        if not r:
            return "—", "—", "—"
        ms = r.get("ms")
        tf = r.get("tflops")
        w = r.get("engine_watts")
        if backend == "ane":
            pl = (r.get("placement") or {}).get("counts_compute") or {}
            if pl.get("CPU") and not pl.get("ANE"):
                w = ((r.get("power") or {}).get("cpu_power") or {}).get("mean")
        return fmt(ms), fmt(tf), fmt(w)

    a_ms, a_tf, a_w = pack(ane, "ane")
    g_ms, g_tf, g_w = pack(gpu, "gpu")
    m_ms, m_tf, m_w = pack(met, "metal")

    cands = []
    if ane and ane.get("ms"):
        pl = (ane.get("placement") or {}).get("counts_compute") or {}
        label = "ANE" if pl.get("ANE") else "ANE-as-CPU"
        cands.append((ane["ms"], label, ane["ms"]))
    if gpu and gpu.get("ms"):
        cands.append((gpu["ms"], "GPU-CoreML", gpu["ms"]))
    if met and met.get("ms"):
        cands.append((met["ms"], "Metal-mlx", met["ms"]))
    if not cands:
        winner = "—"
    else:
        best = min(cands, key=lambda t: t[0])
        others = [c for c in cands if c[1] != best[1]]
        if others:
            ratio = others[0][0] / best[0] if best[0] else None
            # vs next-best
            nxt = min(others, key=lambda t: t[0])
            ratio = nxt[0] / best[0]
            winner = f"{best[1]} {ratio:.2f}x vs {nxt[1]}"
        else:
            winner = best[1]
    print(
        f"| {cell.tag} | {cell.gflop:.2f} | {a_ms} | {a_tf} | {a_w} | {place(ane)} | {g_ms} | {g_tf} | {g_w} | {m_ms} | {m_tf} | {m_w} | {winner} |"
    )

print("\n## int8 ANE vs fp16 ANE (weighted cells)")
print("| cell | fp16 ms | int8 ms | int8/fp16 | int8 TFLOPS |")
print("|---|---:|---:|---:|---:|")
for cell in cells:
    a = by.get((cell.tag, "ane", "fp16"))
    b = by.get((cell.tag, "ane", "int8"))
    if not a or not b:
        continue
    ratio = b["ms"] / a["ms"] if a.get("ms") else None
    print(
        f"| {cell.tag} | {fmt(a.get('ms'))} | {fmt(b.get('ms'))} | {fmt(ratio)} | {fmt(b.get('tflops'))} |"
    )
