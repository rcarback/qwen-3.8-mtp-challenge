---
name: ane-gpu-offload
description: Use when building or diagnosing dense ANE offload on Apple Silicon, including MIL weight programs, IOSurface handoff, device placement, program loading, and ANE/GPU sharing.
---

# ANE / GPU offload

Use this skill for the MIL, weight-blob, IOSurface, and runtime mechanics of
ANE offload. Use [mlxfast](../mlxfast/SKILL.md) for the optimization workflow,
answer contract, and review checks. Local experiments do not change the ranked
editable surface or correctness contract.

## Current decision guidance

Read [engine offload](../mlxfast/references/engine-offload.md) before selecting a
representation or split. It incorporates the later dense and MoE re-evaluation.
Read [measurement](../mlxfast/references/measurement.md) for benchmark preparation
and evidence. Preserve their distinction between estimates, isolated results,
and full-request gains.

The reference files below retain historical probes and recipes. Their absolute
claims about fp16-only execution, universal decode losses, mandatory banks, and
shared GPU/ANE quantized tensors are superseded by the guidance here. Their
instructions to pause unrelated processes or change backup settings do not apply
without separate user authorization.

## Checks before building

| Question | Required distinction |
| --- | --- |
| Can this representation run? | Successful MIL compilation, preferred device, and completed ANE execution are separate evidence |
| Are the effective weights unchanged? | Matching bit width does not imply matching quantization groups, scales, codebooks, or values |
| Does this reduce traffic? | Compare against the deployed GPU format and count metadata, padding, retained copies, and join traffic |
| Can the engines overlap? | Use ready independent work and concurrent measurements. Shared memory can limit the benefit |
| Does program count bind? | Measure the current runtime's live program budget before adding a bank |
| Does the model remain correct? | Validate distinct layer identities, state/buffer lifetime, and the deployment's fidelity gate |

## Weight forms and engine behavior

The tested fp16, per-channel int8, and int4 LUT programs are different weight
representations. The tested MLX-like blockwise q4/q8 Core ML convolution forms
placed on GPU. A second ANE representation can require requantization and extra
resident memory. Do not assume byte-identical GPU/ANE weights from a shared bit width.

Treat the observed fp16 weight-only path separately from supported W8A8 paths.
A custom MIL probe failure does not refute other compiler paths. Verify current
operator, hardware, and OS support for the exact program. The current
[engine guide](../mlxfast/references/engine-offload.md) links Apple's support documentation.

The roughly 126-program limit was observed on the local M4 Max runtime. It is a
measured constraint for that setup, not a universal constant. Multifunction banks
can change program count, but the tested dense bank lost throughput even after
ANE placement returned. Start with a small static split before bank infrastructure.

Single-row programs lost with about 0.30 ms dispatch overhead in local probes.
That result discourages those programs, while larger fused subgraphs remain
separate experiments. Dense MLPs and projections inside MoE models can still be
candidates. The measured dense int8 split justifies evaluation at its tested
prefill shapes, not an unconditional offload policy.

Zero-copy IOSurface sharing removes a staging copy. Producer writes, consumer
reads, synchronization, and buffer lifetime still affect the request.

## Mechanism references

Read only the recipe needed, and check its dated conclusions against the current
guidance above. Preserve useful format details without inheriting obsolete policies.

| Mechanism | Reference |
| --- | --- |
| Probe provenance and reverse-engineering sources | [Sources](references/sources.md) |
| fp16 MIL and weight blobs | [fp16](references/fp16.md) |
| Per-channel affine int8 | [int8](references/int8.md) |
| INT4 palette and tested blockwise rejection | [int4](references/int4.md) |
| IOSurface input/output and lifetime | [Zero-copy](references/zero-copy.md) |
| Program-count probes and multifunction mechanics | [Program limits](references/program-limit.md) |
| Historical bandwidth and concurrency probes | [GPU interaction](references/gpu-interaction.md) |
| Tested MIL operations | [Operations](references/operations.md) |
| Dated chip capability probes | [Chip support](references/chip-support.md) |

Use official documentation for supported API contracts, current source for the
code path, and hardware probes for observed execution. A compute-plan preference
alone cannot prove actual dispatch. Reverse-engineering results also need a
stated hardware/runtime scope.

Direct Swift builds and tests in this repository require
`--force-resolved-versions`. Select relevant runtime tests and verify that their
environment gates actually execute them. Do not disable sandboxing as a default.

Before a run, inspect disk headroom and competing load. Clean only this experiment's
owned artifacts within the user's authorized scope. Broad temporary-directory
cleanup can remove another worker's programs or caches. Record completed dispatches
and partial fallback instead of labeling every successfully loaded program a win.
