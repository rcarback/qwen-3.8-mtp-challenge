# DFlash2 head research scripts

Local research tooling for the DFlash2 drafter port. Nothing here ships: `tools/`
is outside the `benchmark.json` editable surface, so `yukon submit` does not
package it.

Both scripts import the upstream MLX reference module `dflash/model_mlx.py` from
`z-lab/Qwen3.8-27B-DFlash2`. Point `DFLASH2_REFERENCE_PATH` at the directory that
contains the `dflash` package.

## Dump a parity fixture

Runs the reference drafter on deterministic synthetic inputs and writes every
input, every per-layer activation, and the reference proposal to one safetensors
file. `Tests/MLXFastTests/Model/Qwen38DFlash2ParityTests.swift` reads it.

```bash
uv run --with "mlx==0.32.0" --with mlx-lm python tools/dflash2/dump_dflash2_fixture.py \
    ~/.cache/mlxfast/dflash2-27b fixture.safetensors 5 8        # bfloat16
uv run --with "mlx==0.32.0" --with mlx-lm python tools/dflash2/dump_dflash2_fixture.py \
    ~/.cache/mlxfast/dflash2-27b fixture-4bit.safetensors 5 8 4,64
```

The target model is never loaded. The embedding table and the vocabulary
projection are the only things the drafter borrows from it, and the fixture
saves both as arrays, so the 15 GB backbone buys nothing here.

Compare a head against the fixture dumped at the SAME precision. A bfloat16 head
against a 4-bit fixture measures quantization error, not the port.

## Quantize the head

```bash
uv run --with "mlx==0.32.0" --with mlx-lm python tools/dflash2/quantize_dflash2.py \
    ~/.cache/mlxfast/dflash2-27b ~/.cache/mlxfast/dflash2-27b-4bit 4 64
```

| Precision | Size |
|---|---|
| bfloat16 | 3.58 GiB |
| affine 8-bit group-64 | 1.90 GiB |
| affine 4-bit group-64 | 1.01 GiB |

The manifest cap is 2 GiB, so 8-bit and 4-bit both fit and bfloat16 does not.
