# Reduce lookup-table I/O

Use this path for large indexed or memory-mapped tables where page faults or
scattered reads matter. Separate disk traffic, resident memory reads, and logical
record size. Smaller records need not make a warm gather faster.

## Minimal experiments

1. Measure actual rows, record widths, gather time, and cache state in the forward.
   Use a small fixture for correctness, and the real working set for I/O claims.
2. Consolidate split fields when that reduces distinct pages touched. Preserve the
   row mapping, codec metadata, and exact decoded values for a layout-only change.
3. Batch page hints across nearby ranges. Measure the hint overhead and actual
   faults. A hint is not a guarantee of page residency.
4. Move readahead earlier only when offsets are known and unrelated compute can
   overlap it. Keep mapped storage alive until background work completes.
5. Test compression against equally prefetched controls. Separate codec, layout,
   and overlap effects before combining them.
6. Pilot a bijective row remap on held-out access patterns before rewriting the table.

For a nonempty record interval `[offset, offset + length)` and page size P:

```text
first_page = floor(offset / P)
last_page  = floor((offset + length - 1) / P)
```

Count every page in the interval, then take the union across records. Check
bounds and integer overflow before calculating offsets. Distinct pages touched
are not the same as faults or disk reads. Already-resident pages do not fault.

## Quality and comparisons

Layout consolidation, bijective remapping, and read hints use the execution gate.
Lossy codecs need the target approximation gate. Test records crossing pages,
segment boundaries, truncated files, and encode/decode compatibility as applicable.

Compare cold/first-touch and warm behavior separately. State how cache state was
established, without clearing unrelated system caches. Measure full-forward cost,
on-disk size, memory pressure, and peak temporary space for a transform.

Source hook: `Qwen4ExpNGramTable.Stats` in `Qwen4ExpNGram.swift`, gated by
`MLX_QWEN4EXP_NGRAM_STATS`. Check current counters before treating them as page-fault
measurements. Bead `0e5` tracks interval coverage, and `9ad` tracks a layout pilot.

## Historical evidence

Local M4 Max MoE table experiments:

- `2675de37`: contiguous int8 records reduced gather from about 2395 to 879 ms.
- `081a2259`: batched hints changed bf16 gather from 1007 to 870 ms. INT4
  stayed around 852–861 ms, removing most of the apparent codec advantage.
- `5749df08`: early readahead reduced gather from 852 to 448 ms and full forward
  from 9866 to 9228 ms. That is latency hiding, not a measured byte reduction.
- `docs/perf/qwen38-flash-2026-09.md`: warm single-row gather was 0.024 ms in
  a 59.84 ms step. The same table had substantial first-touch cost.

Use [measurement](measurement.md). Keep compression for disk or capacity savings
when the stated goal requires them and quality passes. Retire added sorting or
prefetch machinery when it adds overhead without useful benefit in served regimes.
A single old null result does not rule out another access pattern.
