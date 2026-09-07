# Local fork cleanup, 2026-09-07

The cleanup removes rejected experiments while retaining useful inference and serving paths.
Comparison base: fetched `origin/main`, `0863b06ac16e26e48fc06e97444095b00feb66d4`.
This is the configured Layr-Labs Qwen-MTP upstream, including accepted submissions.
It is not the original Laguna-only source tree.

The source-delta inventory covers `Sources/` and both vendored packages.
The detailed deletion review follows the offload call graph and its measured results.
This is not a claim that every added source line has been proved correct.

| area | decision | evidence |
| --- | --- | --- |
| Core ML dense MLP bank | Delete loader, initializer option, cache selection, and duplicate dispatch branch. | The second pass remained slower after ANE placement was restored. Missing buckets also silently selected another weight form. |
| Coarse and per-projection channel split MLPs | Delete both classes and their three dedicated test suites. | Only tests called them. The recorded speed ratios were 0.32–0.59, and the fused split supersedes them. |
| Activation sparsity mask | Delete the option and restore direct activation-to-down projection. | The down GEMM still visits every channel. The experiment added a mask pass and measured a 0.8 percent slowdown. |
| Direct fused ANE split | Keep fp16, int8, int4, padding, and diagnostic controls. | Int8 improved local prefill, and int4 reduced direct-prefix latency. Neither result proves ranked fidelity. |
| GPU kernels, stream fix, and load safeguards | Keep. | Explicit-stream dispatch previously ignored its argument. Loader changes exclude the n-gram tree and limit peak loading memory. |
| Serving, session cache, MoE model, and speculative decoding | Keep with existing follow-ups. | These provide user-visible behavior. Removing them because they are large would lose capability. Cache accounting and MTP startup already have beads. |
| Bandwidth probes | Keep and extend only after a current byte inventory. | Logical bytes per second do not isolate DRAM traffic. Bead `dl3` records the missing comparison. |
| Fused gate/up tests | Fix the control setup. | The process default fused both arms. The existing placeholder assertion reproduced the mistake. |

The old `MLX_ANE_BANK_DIR` and `MLX_MOE_ACT_SPARSITY` experiments are removed.
Historical results and standalone probe tools remain as evidence, not supported serving options.
The bank removal does not reject grouped codebooks or other bandwidth-saving methods.

The gate/up per-instance override now accepts explicit true or false, with nil using the process default.
Tests select an unfused control before its first forward and assert that it retains separate stacks.
Existing callers that force fusion still select true.

Beads filed in this pass:

- `dl3`: measure current weight and activation traffic.
- `faq`: validate measurement receipts before selecting defaults.
- `720`: track this deletion review and its verification.
- `urw`: close owned trace descriptors when trace objects are released.
- `dyv`: repair the fused/unfused test comparison.

The trace-descriptor finding remains separate from this deletion patch.
No model benchmark, ranked submission, or commit was made.

Verification: the full test target built with `--force-resolved-versions`.
The selected run passed 15 tests with `MLXFAST_RUN_MLX_RUNTIME_TESTS=1`:
two cache tests, seven MoE tests, and six ANE split tests.
The S=512 checks executed, including exact GPU-only and concurrent/sequential equality.
These synthetic tests do not establish full-model quality or a performance gain.
The code and test delta removes 1,201 net lines across 11 files.
Whitespace checks and error-level documentation lint pass.
