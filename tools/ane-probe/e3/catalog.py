#!/usr/bin/env python3
"""Cell catalog, geometry, FLOP accounting, and shared paths for E3."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
E2_COMMON = HERE.parent / "e2" / "common.py"
CONFIG_PATH = Path(
    "/Users/carback1/Code/llms/qwen-3.8-mtp-challenge/weights/config.json"
)

SEQS = (128, 512, 1024, 2048)
SMALL_SEQS = (1, 2, 8)
KV_LENS = (4096, 16384)
COREML_PRECISIONS = ("fp16", "int8")
SEED = 42


def _load_e2_common():
    spec = importlib.util.spec_from_file_location("e2_common", E2_COMMON)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {E2_COMMON}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e2_common"] = mod
    spec.loader.exec_module(mod)
    return mod


e2 = _load_e2_common()

import coremltools as ct  # noqa: E402

UNITS = {
    "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
    "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
    "all": ct.ComputeUnit.ALL,
}

inspect_placement = e2.inspect_placement
start_macmon = e2.start_macmon
stop_macmon = e2.stop_macmon
summarize_power = e2.summarize_power


def load_geometry() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text())
    hidden = int(cfg["hidden_size"])
    n_q = int(cfg["num_attention_heads"])
    n_kv = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    layer_types = list(cfg["layer_types"])
    return {
        "hidden_size": hidden,
        "intermediate_size": int(cfg["intermediate_size"]),
        "num_attention_heads": n_q,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "q_out": 2 * n_q * head_dim,
        "kv_out": n_kv * head_dim,
        "o_in": n_q * head_dim,
        "mtp_fc_in": 2 * hidden,
        "rms_norm_eps": float(cfg.get("rms_norm_eps", 1e-6)),
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "n_linear_attn": sum(1 for t in layer_types if t == "linear_attention"),
        "n_full_attn": sum(1 for t in layer_types if t == "full_attention"),
        "quantization": cfg.get("quantization"),
        "model_type": str(cfg["model_type"]),
        "vocab_size": int(cfg["vocab_size"]),
    }


@dataclass(frozen=True)
class Cell:
    op: str
    seq: int
    extra: str
    in_features: int
    out_features: int
    kv: int
    n_q: int
    n_kv: int
    head_dim: int
    width: int
    flops: int
    has_weights: bool
    notes: str

    @property
    def tag(self) -> str:
        if self.kv:
            return f"{self.op}_s{self.seq}_kv{self.kv}"
        return f"{self.op}_s{self.seq}"

    @property
    def gflop(self) -> float:
        return self.flops / 1e9


def build_cells(geo: dict | None = None) -> list[Cell]:
    geo = geo or load_geometry()
    hidden = geo["hidden_size"]
    inter = geo["intermediate_size"]
    n_q = geo["num_attention_heads"]
    n_kv = geo["num_key_value_heads"]
    head_dim = geo["head_dim"]
    q_out = geo["q_out"]
    kv_out = geo["kv_out"]
    o_in = geo["o_in"]
    cells: list[Cell] = []

    def gemm(op: str, seq: int, inn: int, out: int, notes: str) -> Cell:
        return Cell(
            op=op,
            seq=seq,
            extra="",
            in_features=inn,
            out_features=out,
            kv=0,
            n_q=0,
            n_kv=0,
            head_dim=0,
            width=0,
            flops=2 * seq * out * inn,
            has_weights=True,
            notes=notes,
        )

    for seq in SEQS:
        cells.append(
            gemm(
                "large_gemm_up",
                seq,
                hidden,
                inter,
                "MLP gate/up [17408,5120] at prefill seq",
            )
        )
        cells.append(
            gemm(
                "large_gemm_down",
                seq,
                inter,
                hidden,
                "MLP down [5120,17408] at prefill seq",
            )
        )
        # Four projections, FLOPs summed.
        attn_flops = 2 * seq * hidden * (q_out + 2 * kv_out + o_in)
        cells.append(
            Cell(
                op="attn_proj",
                seq=seq,
                extra="",
                in_features=hidden,
                out_features=q_out,
                kv=0,
                n_q=n_q,
                n_kv=n_kv,
                head_dim=head_dim,
                width=0,
                flops=attn_flops,
                has_weights=True,
                notes="Q(fused-gate)+K+V+O, O takes a separate [S,6144] input",
            )
        )
        cells.append(
            Cell(
                op="elem_swiglu",
                seq=seq,
                extra="",
                in_features=inter,
                out_features=inter,
                kv=0,
                n_q=0,
                n_kv=0,
                head_dim=0,
                width=inter,
                flops=seq * inter * 9,
                has_weights=True,
                notes="RMSNorm+SiLU+mul at [1,S,17408]; FLOPs are order-of-magnitude",
            )
        )
        for kv in KV_LENS:
            cells.append(
                Cell(
                    op="attn_scores",
                    seq=seq,
                    extra=f"kv{kv}",
                    in_features=head_dim,
                    out_features=n_q,
                    kv=kv,
                    n_q=n_q,
                    n_kv=n_kv,
                    head_dim=head_dim,
                    width=0,
                    flops=4 * n_q * seq * kv * head_dim,
                    has_weights=False,
                    notes="QK^T + AV, GQA expand 4->24, no softmax, KV cached",
                )
            )

    for seq in SMALL_SEQS:
        cells.append(
            gemm(
                "mtp_fc",
                seq,
                2 * hidden,
                hidden,
                "MTP head fc [5120,10240] at draft seq",
            )
        )
        cells.append(
            gemm(
                "mtp_mlp_up",
                seq,
                hidden,
                inter,
                "Same MLP up GEMM as large_gemm_up, draft/decode seq (GEMV-ish)",
            )
        )
    return cells


def package_path(cell: Cell, precision: str) -> Path:
    return ARTIFACTS / f"{cell.tag}_{precision}.mlpackage"


def tflops(flops: int, ms: float) -> float | None:
    if not ms or ms <= 0:
        return None
    return (flops / 1e12) / (ms / 1e3)


def perf_per_watt(flops: int, ms: float, watts: float) -> float | None:
    t = tflops(flops, ms)
    if t is None or not watts or watts <= 0:
        return None
    return t / watts


def engine_watts(power: dict, backend: str) -> float | None:
    if not power:
        return None
    key = {
        "ane": "ane_power",
        "gpu": "gpu_power",
        "metal": "gpu_power",
        "cpu": "cpu_power",
    }.get(backend, "all_power")
    stats = power.get(key) or {}
    mean = stats.get("mean")
    return float(mean) if mean is not None else None


if __name__ == "__main__":
    geo = load_geometry()
    cells = build_cells(geo)
    print(json.dumps({"geo": geo, "n_cells": len(cells), "ops": sorted({c.op for c in cells})}, indent=2))
    for c in cells:
        print(f"{c.tag:32} gflop={c.gflop:8.2f} weights={c.has_weights} {c.notes}")
