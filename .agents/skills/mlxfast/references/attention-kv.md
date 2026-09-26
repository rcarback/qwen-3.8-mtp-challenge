# Reduce attention traffic and KV bytes

Attention traffic depends on context, KV heads, layer count, cache precision, and
kernel reuse. Weight reads can dominate short-context decode. Inventory both
before deciding that attention is the bottleneck.

## Count K and V separately

For batch size B and cached length T_l at each attention layer l:

```text
KV_payload_bytes = B * sum_l[T_l * Hkv_l * (Dk_l * bk_l + Dv_l * bv_l) / 8]
```

Bit widths are `bk` and `bv`. For equal head dimensions D and fp16 K/V, this is
`4 * B * sum_l(T_l * Hkv_l * D)` bytes. The factor four includes both K and V.
Add scales, biases, padding, and any unquantized tail. Sliding-window layers use
their retained window length. Recurrent state belongs in a separate ledger.

This estimates storage. One logical full-cache read per decode step is a traffic
model, not a hardware counter. Query-head reuse, cache residency, repeated reads,
and temporary writes change actual traffic. Multiply by context before comparing
per-position storage with a whole-step time budget.

## Separate the experiments

| Experiment | Comparison and gate |
| --- | --- |
| Fuse attention without a score matrix | Same cache representation and semantics. Test numerical and required token fidelity. |
| Reduce KV precision | Fixed kernel policy where possible. Measure quality, metadata, quantize/dequantize work, and total latency. |
| Rotate before quantization | Isolate rotation from precision and dispatch. Account for query transforms, cache writes, and value/output transforms if used. |
| Trim or ring-buffer a sliding window | Preserve position/mask semantics. Measure retained bytes and copies. |
| Chunk attention query rows | Preserve global causal offsets. Compare attention-only chunks with the extra weight passes of whole-forward chunks. |

An orthogonal rotation preserves dot products in exact arithmetic when applied
consistently to Q and K. Quantization and finite-precision transforms introduce
error. It is not free at read time merely because an extra kernel is fused away.

## Dispatch contract

Check supported bit widths explicitly, as well as packing divisibility, group
size, head dimensions, GQA divisibility, strides, dtype, and batch support.
Check masks, sinks, bias metadata, context bounds, and query-row range at the
call site or predicate that owns each condition. Unsupported cases need a valid
reference path, not a reinterpretation of the requested attention.

Test boundaries and malformed configurations. Cover supported query widths for
correctness and representative widths for performance. Add contexts long enough
to exercise the claimed memory benefit. Test quantized cache update, rotation,
rollback, and resume together when the deployment combines them.

## Current code cautions

`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/FusedQuantizedSDPA.swift` accepts query
rows 1–8, affine mode with biases, and no sinks. Its `isSupported` tests packing
divisibility but does not explicitly enumerate valid bit widths. Treat it as an
code to audit, not a complete model for a new predicate.

`AttentionUtils.swift` records `supportedShape` before the `fusedQuantizedEnabled`
guard. Those statistics can count eligible calls while the kernel is disabled.
Do not describe an eligibility counter as actual execution evidence. Inspect and
instrument the dispatch after all guards when proving engagement.

The reported 9–12x cost compares decomposed quantized attention with fused
bfloat16 attention at tested decode shapes. Measure the new fused quantized kernel
separately before assigning it a speedup. Measure all three
paths at the same geometry. `FusedQuantizedSDPASpeedTests.swift` is an env-gated
probe, and a test assertion alone is not a numeric performance record.

Bead `asj` tracks long-context KV evaluation. A tiny KV share at 1k context does
not settle behavior at 16k or 32k. A slower compact cache may still serve a
memory-capacity objective. State that tradeoff instead of calling it a speedup.
See [measurement](measurement.md) for gates and [budget](budget.md) for traffic.
