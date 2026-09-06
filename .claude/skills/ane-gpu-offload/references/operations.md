# ANE operations: what runs, what is accepted, what is rejected

The Core ML MIL op set (168 core ops at the iOS18 opset) is a superset of
what the ANE executes. The independent analyses are explicit that "the ANE
compiler accepts a subset of MIL operations, and several operations that are
valid in CoreML's MIL specification are silently rejected or produce
incorrect results on the ANE." So this reference gives each op a status:

- **confirmed** — dispatched and checked numerically in this repository.
- **rejected** — the ANE compiler or engine refused it here.
- **source** — behavior documented by an independent analysis, not yet
  probed here.
- **unprobed** — a valid MIL op with no ANE evidence either way; assume
  nothing.

The engine is a static fp16 dataflow machine: no data-dependent control flow,
no dynamic gather, fp16 compute with a wide accumulator. Every design choice
below follows from that.

## Matrix multiply and convolution

| op | status | usage |
| --- | --- | --- |
| `conv` (1x1, groups 1) | **confirmed** | The workhorse. `x[1,IN,1,S] @ w[OUT,IN,1,1]` is `x @ w.T` per position. This is how a linear layer is expressed for the ANE (oMLX, maderix, this repo). Accumulation is a wide fp32-class sum of fp16 products: a 16k-term sum of ones is bit-exact; the error is fp16 products and storage, not accumulation. |
| `conv` with `weight` from a constexpr op | **confirmed (int8)** | Weight dequantized on-chip; see the dtype sub-skills. |
| `matmul`, `linear`, `einsum` | unprobed | Every working ANE stack here and in the sources spells the projection as a 1x1 conv. Prefer conv. |
| `conv_transpose`, `conv_quantized` | unprobed | |
| `scaled_dot_product_attention` | unprobed | The fused attention op exists in MIL. Attention's dynamic KV length does not fit fixed-shape programs; expect to keep attention on the GPU. |

## Weight compression (constexpr ops)

| op | status | usage |
| --- | --- | --- |
| `constexpr_affine_dequantize` | **confirmed** | int8 (or uint8) data, scalar or vector scale, scalar zero-point, `axis`. Params inline in the `[...]` attribute bracket. 0.8 percent relative error at int8. `int8.md`. |
| `constexpr_blockwise_shift_scale` | **rejected (in-memory path)** | int4/uint4/int8/uint8 data with a per-block scale (group along an axis) and optional offset. iOS18 op. coremltools emits it and Core ML compiles it; the in-memory ANE compiler returns `InvalidMILProgram`. Open: try the `.mlpackage` compile path. `int4.md`. |
| `constexpr_lut_to_dense` | source | Palettized weights: `uint8` packed indices plus a LUT of 2, 4, 16, 64 or 256 entries (1, 2, 4, 6, 8 bits). The int4 palette form is what the architecture paper measured at about 2.37x fp16 bandwidth. Not yet probed here. |
| `constexpr_lut_to_sparse`, `constexpr_sparse_to_dense`, `constexpr_sparse_blockwise_shift_scale` | source | Structured sparsity. The paper measured a weight with at least half zeros at 1.55 to 1.64x faster at 0.43x the bytes (a one-bit keep-mask plus packed fp16 nonzeros). Not probed here. |
| `constexpr_cast` | unprobed | |
| `quantize`, `dequantize` (runtime, on activations) | unprobed | Activation int8 compute is advertised for A17 Pro / M4 class; unmeasured here. |

## Activations and elementwise

| op | status | usage |
| --- | --- | --- |
| `silu` | **confirmed, imprecise** | Lowers to the engine's lookup-table sigmoid. Measured error on real activations 10 to 30x the fp16 floor; it caused the hybrid's generation collapse. Do not use for a gate. |
| `sigmoid`, `sigmoid` then `mul` | **confirmed, imprecise** | Same LUT path as `silu`. |
| `exp`, `real_div`, `add`, `mul` (the `.expDiv` SiLU: `x / (1 + exp(-x))`) | **confirmed, accurate** | Lands at the conv rounding floor. The production default for SiLU. |
| `tanh` (the `.tanhForm` SiLU: `x * 0.5 * (1 + tanh(x/2))`) | **confirmed, accurate** | Also at the floor. |
| `mul` (SwiGLU gate times up), `add`, `sub`, `real_div` | **confirmed** | Used inside the fused SwiGLU-down program. |
| `identity` | **confirmed** | The `.none` activation diagnostic. |
| `gelu`, `relu`, `relu6`, `leaky_relu`, `prelu`, `elu`, `softplus`, `clip`, `erf`, `sqrt`, `rsqrt`, `log`, `abs`, `neg`, `pow`, `maximum`, `minimum`, `floor`, `ceil`, `round`, `sin`, `cos` | unprobed | Any op that lowers to a table (sigmoid-like, erf, possibly gelu) should be assumed imprecise until measured against the fp16 floor, per the `silu` result. Spell it from `exp`/`tanh` where precision matters. |
| `softmax` | unprobed | |

## Normalization and reduction

| op | status |
| --- | --- |
| `layer_norm`, `batch_norm`, `instance_norm`, `l2_norm`, `local_response_norm` | unprobed |
| `reduce_sum`, `reduce_mean`, `reduce_max`, `reduce_min`, `reduce_argmax`, `reduce_argmin`, `reduce_prod`, `reduce_log_sum`, `reduce_log_sum_exp`, `reduce_l1_norm`, `reduce_l2_norm`, `reduce_sum_square` | unprobed |
| `avg_pool`, `max_pool`, `l2_pool` | unprobed |

## Shape, movement and indexing

| op | status | usage |
| --- | --- | --- |
| `reshape`, `transpose`, `concat`, `split`, `squeeze`, `expand_dims`, `tile`, `pad`, `stack`, `slice_by_index`, `slice_by_size` | unprobed | Static-shape reshuffles; expected to lower. Keep the activation layout `[1, C, 1, S]` (channels, then sequence) so no transpose is needed inside the program. |
| `gather`, `gather_along_axis`, `gather_nd`, `scatter*` | **not dispatchable in the needed form** | The ANE has no dynamic (data-dependent) gather. This is why the routed experts cannot run on it: a program is one fixed weight set, and selecting 10 of 512 experts by a runtime index is exactly the gather it lacks. Static (compile-time) index patterns may lower; runtime ones do not. |
| `band_part`, `reverse`, `sliding_windows` | unprobed | |

## Control flow and state

| op | status | usage |
| --- | --- | --- |
| `cond`, `while_loop` | **not usable** | The engine is a static dataflow machine. A `select` chooses an already-computed output; it cannot skip the other branch's work. "Dispatch everything and flag which runs" therefore computes everything. |
| `select`, comparison ops (`equal`, `greater`, `less`, ...), `logical_*` | unprobed as ANE ops | Valid MIL; on the ANE they are masks over computed results, never short-circuits. |
| `read_state`, `list_*`, `make_list`, `gru`, `lstm`, `rnn` | unprobed | Stateful and recurrent ops; the gated-delta recurrence is done on the GPU here. |
| `random_*`, `topk`, `argsort`, `non_maximum_suppression`, `non_zero` | unprobed | Data-dependent output shapes or randomness; unlikely to lower. |

## Data types

| type | status |
| --- | --- |
| fp16 activations and weights | **confirmed** — the native compute type. |
| fp32 program | **rejected** — `ANECCompile FAILED: CompilationFailure`. The ANE is fp16-native. An fp16-in, fp32-out cast compiles but the cast is after the conv; it does not change accumulation. |
| int8 weights (affine) | **confirmed** — dequantized to fp16 on-chip. |
| int4 weights (blockwise, palette) | source / open — see `int4.md`. |
| int8 activations (A17 Pro / M4 class int8-int8 compute) | source, unmeasured here. |
| fp4 / MXFP4 / NVFP4 | **not native** — these are Metal 4.1 and MLX GPU formats. NVFP4's 16-value E2M1 codebook can be re-expressed as a grouped 4-bit palette (`constexpr_lut_to_dense`), which is representation, not a native mode. |

## Program structure

| feature | status | usage |
| --- | --- | --- |
| One `func main<ios18>` per program | **confirmed** | The form every working program here uses. |
| Many functions in one bare in-memory program | **confirmed loads, `main`-only dispatch** | `program-limit.md`. |
| Multifunction `.mlpackage` (functions declared in the model description) | **confirmed, every function dispatches** | `program-limit.md`. |
| `BLOBFILE` weight reference | **confirmed** | oMLX chunk format; `fp16.md`, `int4.md` for the chunk type field. |
| Inline weight const | **confirmed** for small tensors | Bounded by the 2 GB protobuf message limit for the binary path; use `BLOBFILE` for real weights. |
| Fused multi-op program (gate conv, up conv, activation, mul, down conv) | **confirmed** | oMLX's `fp16_swiglu_down_mil`: the whole MLP as one dispatch, no per-projection barrier. ane-infer reports 3.6 TFLOPS for an eight-op fused FFN. Intermediates stay in IOSurfaces. |
