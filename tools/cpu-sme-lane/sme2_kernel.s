// Hand-written SME2 FMOPA peak-throughput kernel for M4 Max.
//
// Not a real GEMM: this issues back-to-back FMOPA outer-product
// accumulates into four independent ZA tiles (za0..za3) using SVL-wide
// operand vectors held in registers. It measures peak FMOPA issue/retire
// throughput of the SME2 MOPA unit, uncontaminated by memory-load
// bandwidth, so it is an upper bound on what any real SME2 GEMM kernel
// (including Accelerate's own) can achieve on this silicon.
//
// Each FMOPA on a full ZA32 tile with SVL=svl elements performs
// 2 * svl * svl FLOPs (svl x svl outer product, multiply + add).
// Four independent tiles per loop iteration -> 4 * 2 * svl^2 FLOPs/iter.
//
// void sme2_peak_loop(uint64_t iters)
.arch armv9-a+sme2
.text
.global _sme2_peak_loop
.p2align 2
_sme2_peak_loop:
    smstart
    zero {za}
    ptrue p0.s
    fmov z0.s, #1.0
    fmov z1.s, #1.0
1:
    fmopa za0.s, p0/m, p0/m, z0.s, z1.s
    fmopa za1.s, p0/m, p0/m, z0.s, z1.s
    fmopa za2.s, p0/m, p0/m, z0.s, z1.s
    fmopa za3.s, p0/m, p0/m, z0.s, z1.s
    subs x0, x0, #1
    b.ne 1b
    smstop
    ret

// uint64_t sme2_query_svl_words(void) -- returns SVL in 32-bit words
.global _sme2_query_svl_words
.p2align 2
_sme2_query_svl_words:
    smstart sm
    cntw x0
    smstop sm
    ret
