# The 126-program limit and the program design around it

## The limit

A process can hold about 126 loaded ANE programs. The 127th `loadWithQoS`
fails with `Program load failure (0x50004)`. It is a count, not a byte budget:
8 KB and 26 MB programs both fail at the 127th
(`ANEProgramCountLimitTests`). Three independent groups hit the same wall:
Orion measured about 119 compilations per process and identified it as
compiler-side internal state; the field guide sees the same `0x50004` as
runtime program-load behavior; this repository measured 126 loads. It is
per-process software state, not chip memory.

One fixed shape is one program. A dense lane at (layer, projection, sequence
bucket) is about 48 programs per bucket on this model (12 attention `q_proj`
plus 36 gated-delta `in_proj`), so one or two buckets fit and a third does not.
Anything that wants many buckets, many dtypes, or the experts, exceeds the
limit immediately.

## What relieves it, measured

| mechanism | count relief | dispatch |
| --- | --- | --- |
| Procedure bank: many `func`s in one bare in-memory MIL program | Yes. 512 functions loaded in 64 programs, no failure (`ANEProcedureBankProbeTests`). The loader counts programs. | **Only `main` runs.** `procedureIndex >= 1` returns `Program Inference error` under every naming scheme; the in-memory descriptor registers one procedure. |
| Multifunction `.mlpackage`: functions declared in the model description, compiled with `MLModel.compileModel(at:)`, loaded per function with `MLModelConfiguration.functionName` | Yes, same packing. | **Every function runs**, each with its own weights at the fp16 floor (`ANEMultiFunctionProbeTests`). |
| Orion's `exec()` restart or unload-and-reload with new weights | Resets the counter | Process-level; heavy. |
| Persistent compiled-program cache across processes (oMLX AOT cache) | Reuses compiled programs; does not raise the per-process count. | |

So the workaround is the multifunction `.mlpackage`. The in-memory
`MLModelAsset(specification:)` blob rejects a multifunction description
("does not support the multi-function description") at both CoreML8 and
CoreML9 opsets, and a top-level input beside the functions list is a hard
error, so the description must be functions-only and it must go through a
`.mlpackage` on disk. The raw `_ANEModel`/`_ANEClient` loader exposes
`procedureInfoForProcedureIndex:` but reads Espresso `model.espresso.net` and
cannot consume the ML Program `model.mil` that `compileModel` writes, so it is
not the route.

## The program design

Do not build one program per (layer, projection, bucket). Build **one
program per op-type, per dtype, per bucket, packing every layer of that type
as a procedure**:

```text
program  attn_qproj_int8_S512    functions: layer3, layer7, ..., layer47   (12)
program  gdn_inproj_int8_S512    functions: layer0, layer1, ..., layer46   (36)
program  attn_qproj_int8_S1024   ...
program  gdn_inproj_int8_S1024   ...
```

Four buckets of two op-types is eight programs, against 192 without packing.
Each program's weight blob is the concatenation of its layers' weights
(`buildMultiWeightBlob`, one chunk per layer, chunk type 1 for fp16/int8, 8
for packed int4), and each function's `BLOBFILE(offset=...)` points at its
layer's chunk header. The per-program byte ceiling is unmeasured above 26 MB
per program; a 36-layer int8 `in_proj` bank is 36 x 26 MB = 0.94 GB in one
program, so measure that ceiling before assuming a full-tower bank.

Split by dtype as well, because a program's constexpr weight op is one form:
an fp16 bank, an int8 bank and an int4 bank are three programs, not one with
mixed functions.

## Building a bank

`buildMultiFunctionConvSpec(procs:opset:)` emits the multifunction `Model`
proto: a `functions` list of `FunctionDescription`s (field 20) with
`defaultFunctionName` (field 21) and no top-level I/O, paired with the MIL
program's `functions` map, one entry per proc. Today it inlines fp16 weight
consts; the bank form needs each proc's weight as a BLOBFILE chunk of a shared
`weight.bin` and the int8 or int4 constexpr op from the dtype sub-skills.
`ANEMultiFunctionProbeTests.writeMLPackage` shows the minimal `.mlpackage`
(a `Manifest.json` plus `Data/com.apple.CoreML/model.mlmodel`); add
`weights/weight.bin` beside the model for BLOBFILE weights.

## Dispatching a bank

Load each function once with `MLModelConfiguration.functionName`, keep the
`MLModel` per (program, function) resident, and drive it with the zero-copy
pattern in `zero-copy.md`. The load happens at model init, outside any timed
window; a multifunction load per function is the cost of one compile shared
across functions plus a per-function load.

## Two things banking does not change

- It does not reduce dispatch count. One projection is still one dispatch;
  the fixed per-dispatch floor is paid per call. Banking is a load-count
  fix, not a latency fix.
- Time-separated functions cannot share one dispatch. Packing all 36 layers'
  `in_proj` into one program and dispatching them together would feed layers
  1 through 35 inputs that do not exist yet. Each layer dispatches its own
  function when its input is ready; a data flag cannot skip the others,
  because the ANE has no data-dependent control flow (a `select` picks an
  output, every compiled op still runs).
