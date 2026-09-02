// Arithmetic-bound GPU compute kernel used as a GEMM-ish stand-in for the
// starvation test. Each thread does a long dependent FMA chain over
// registers (no significant memory traffic), so the kernel is compute-bound
// and its achieved throughput is a fair proxy for "is the GPU's compute
// pipeline being fed at full rate" while a concurrent CPU lane runs.
#include <metal_stdlib>
using namespace metal;

kernel void fma_burn(device float *out [[buffer(0)]],
                      constant uint &itersPerThread [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    float4 a = float4(1.0001f, 1.0002f, 1.0003f, 1.0004f);
    float4 b = float4(0.9999f, 0.9998f, 0.9997f, 0.9996f);
    float4 acc = float4(0.0f);
    for (uint i = 0; i < itersPerThread; i++) {
        acc = fma(a, b, acc);
        acc = fma(b, a, acc);
        acc = fma(a, acc, b);
        acc = fma(acc, b, a);
    }
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}
