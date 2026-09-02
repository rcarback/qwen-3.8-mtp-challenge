// SME2 / Accelerate matmul peak-throughput probe for M4 Max.
//
// Measures:
//   1. cblas_sgemm (fp32) via Accelerate, at prefill-relevant shapes,
//      scaled across thread counts.
//   2. Peak FMOPA issue throughput of the SME2 MOPA unit directly, via
//      the hand-written kernel in sme2_kernel.s (see that file for what
//      it does and does not measure -- it is an upper bound, not a GEMM).
//
// Both are run with 1, 2, 4, 8, 12 threads to see whether throughput
// plateaus after ~1-2 threads (one SME unit per P-cluster, 2 P-clusters
// on M4 Max) or keeps scaling with thread count.
//
// Build:
//   clang -O3 -march=armv9-a+sme2 -DACCELERATE_NEW_LAPACK \
//       -framework Accelerate -lpthread \
//       sme_peak.c sme2_kernel.s -o sme_peak
//
// Usage:
//   ./sme_peak sgemm <M> <K> <N> <threads> <iters>
//   ./sme_peak sme2fmopa <threads> <iters_per_thread>
//   ./sme_peak sme2fmopa_run <threads> <duration_ms> [utility]

#include <Accelerate/Accelerate.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

// M4 Max E-cores do not implement SME2 (fmopa on an E-core-scheduled thread
// takes SIGILL). Default pthread_create threads can land on an E-core under
// default QoS; force QOS_CLASS_USER_INTERACTIVE so the scheduler places
// every worker on a P-core. This matters for both kernels here (SME2 needs
// it to not crash; sgemm needs it for a fair P-core-only comparison).
static int spawn_pcore(pthread_t *tid, void *(*fn)(void *), void *arg) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
    int rc = pthread_create(tid, &attr, fn, arg);
    pthread_attr_destroy(&attr);
    return rc;
}

extern void sme2_peak_loop(uint64_t iters);
extern uint64_t sme2_query_svl_words(void);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ---- Accelerate cblas_sgemm thread worker --------------------------------
typedef struct {
    const float *A, *B;
    float *C;
    int M, K, N, iters;
    double elapsed;
} SgemmArg;

static void *sgemm_worker(void *p) {
    SgemmArg *t = (SgemmArg *)p;
    double t0 = now_sec();
    float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < t->iters; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    t->M, t->N, t->K, alpha, t->A, t->K, t->B, t->N,
                    beta, t->C, t->N);
    }
    t->elapsed = now_sec() - t0;
    return NULL;
}

static void run_sgemm(int M, int K, int N, int threads, int iters) {
    pthread_t *tids = malloc(sizeof(pthread_t) * threads);
    SgemmArg *args = malloc(sizeof(SgemmArg) * threads);
    for (int t = 0; t < threads; t++) {
        float *A = malloc(sizeof(float) * (size_t)M * K);
        float *B = malloc(sizeof(float) * (size_t)K * N);
        float *C = malloc(sizeof(float) * (size_t)M * N);
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (float)(i % 7) * 0.01f;
        for (size_t i = 0; i < (size_t)K * N; i++) B[i] = (float)(i % 5) * 0.01f;
        memset(C, 0, sizeof(float) * (size_t)M * N);
        args[t] = (SgemmArg){A, B, C, M, K, N, iters, 0.0};
    }
    double wall0 = now_sec();
    for (int t = 0; t < threads; t++) spawn_pcore(&tids[t], sgemm_worker, &args[t]);
    for (int t = 0; t < threads; t++) pthread_join(tids[t], NULL);
    double wall = now_sec() - wall0;

    double flops_per_iter = 2.0 * M * K * N;
    double total_flops = flops_per_iter * iters * threads;
    double gflops = total_flops / wall / 1e9;
    printf("kernel=sgemm M=%d K=%d N=%d threads=%d iters=%d wall=%.4fs GFLOPS=%.2f TFLOPS=%.4f\n",
           M, K, N, threads, iters, wall, gflops, gflops / 1000.0);
}

// ---- SME2 FMOPA peak worker ----------------------------------------------
typedef struct {
    uint64_t iters;
    double elapsed;
} FmopaArg;

static void *fmopa_worker(void *p) {
    FmopaArg *t = (FmopaArg *)p;
    double t0 = now_sec();
    sme2_peak_loop(t->iters);
    t->elapsed = now_sec() - t0;
    return NULL;
}

static void run_sme2fmopa(int threads, uint64_t iters) {
    uint64_t svl = sme2_query_svl_words();
    pthread_t *tids = malloc(sizeof(pthread_t) * threads);
    FmopaArg *args = malloc(sizeof(FmopaArg) * threads);
    for (int t = 0; t < threads; t++) args[t] = (FmopaArg){iters, 0.0};

    double wall0 = now_sec();
    for (int t = 0; t < threads; t++) spawn_pcore(&tids[t], fmopa_worker, &args[t]);
    for (int t = 0; t < threads; t++) pthread_join(tids[t], NULL);
    double wall = now_sec() - wall0;

    // 4 tiles/iter * 2 FLOPs (mul+add) * svl * svl per tile.
    double flops_per_iter = 4.0 * 2.0 * (double)svl * (double)svl;
    double total_flops = flops_per_iter * (double)iters * threads;
    double gflops = total_flops / wall / 1e9;
    printf("kernel=sme2fmopa svl_words=%llu threads=%d iters=%llu wall=%.4fs GFLOPS=%.2f TFLOPS=%.4f\n",
           (unsigned long long)svl, threads, (unsigned long long)iters, wall, gflops, gflops / 1000.0);
}

// ---- SME2 FMOPA duration-based worker (for concurrent-with-GPU testing) --
// NOTE: at -O2/-O3 this translation unit was observed to miscompile a
// `double` duration parameter into 0 when a struct field of the same
// benchmark-args type was declared _Atomic (Apple clang 21.0.0 / macOS
// 26.5). Verified at -O0 the parameter arrives correctly; switching the
// shared-state variable to `volatile sig_atomic_t` (plain word-sized flag,
// not a _Atomic struct field) and dropping the unused qos_lowered field
// fixed it at -O2/-O3 too. Kept as plain globals/params, no aggregate
// _Atomic members, to stay clear of whatever triggered the miscompile.
typedef struct {
    uint64_t iters_done;
    double elapsed;
} FmopaRunArg;

static volatile sig_atomic_t g_stop_flag;

static void *fmopa_run_worker(void *p) {
    FmopaRunArg *t = (FmopaRunArg *)p;
    double t0 = now_sec();
    uint64_t batch = 200000;
    uint64_t total = 0;
    while (!g_stop_flag) {
        sme2_peak_loop(batch);
        total += batch;
    }
    t->iters_done = total;
    t->elapsed = now_sec() - t0;
    return NULL;
}

// duration_ms is an integer millisecond count (not a double) -- a `double`
// duration parameter here was observed to arrive as 0 inside this function
// at -O1/-O2/-O3 on this toolchain (Apple clang 21.0.0 / macOS 26.5),
// reproducibly, even though the same value printed correctly at the call
// site; passing plain integer milliseconds sidesteps whatever that was.
static void run_sme2fmopa_duration(int threads, long duration_ms, int use_utility_qos) {
    uint64_t svl = sme2_query_svl_words();
    pthread_t *tids = malloc(sizeof(pthread_t) * threads);
    FmopaRunArg *args = malloc(sizeof(FmopaRunArg) * threads);
    for (int t = 0; t < threads; t++) args[t] = (FmopaRunArg){0, 0.0};
    g_stop_flag = 0;

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_set_qos_class_np(&attr, use_utility_qos ? QOS_CLASS_UTILITY : QOS_CLASS_USER_INTERACTIVE, 0);

    double wall0 = now_sec();
    for (int t = 0; t < threads; t++) pthread_create(&tids[t], &attr, fmopa_run_worker, &args[t]);
    double duration_sec = (double)duration_ms / 1000.0;
    while (now_sec() - wall0 < duration_sec) {
        struct timespec nap = {0, 5 * 1000 * 1000}; // 5ms poll granularity
        nanosleep(&nap, NULL);
    }
    g_stop_flag = 1;
    for (int t = 0; t < threads; t++) pthread_join(tids[t], NULL);
    double wall = now_sec() - wall0;

    uint64_t total_iters = 0;
    for (int t = 0; t < threads; t++) total_iters += args[t].iters_done;
    double flops_per_iter = 4.0 * 2.0 * (double)svl * (double)svl;
    double gflops = flops_per_iter * (double)total_iters / wall / 1e9;
    printf("kernel=sme2fmopa_run threads=%d qos=%s wall=%.4fs GFLOPS=%.2f TFLOPS=%.4f\n",
           threads, use_utility_qos ? "utility" : "user_interactive", wall, gflops, gflops / 1000.0);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s sgemm M K N threads iters\n", argv[0]);
        fprintf(stderr, "       %s sme2fmopa threads iters_per_thread\n", argv[0]);
        fprintf(stderr, "       %s sme2fmopa_run threads duration_ms [utility]\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "sgemm") == 0) {
        if (argc < 7) { fprintf(stderr, "sgemm needs M K N threads iters\n"); return 1; }
        run_sgemm(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
    } else if (strcmp(argv[1], "sme2fmopa") == 0) {
        if (argc < 4) { fprintf(stderr, "sme2fmopa needs threads iters\n"); return 1; }
        run_sme2fmopa(atoi(argv[2]), strtoull(argv[3], NULL, 10));
    } else if (strcmp(argv[1], "sme2fmopa_run") == 0) {
        if (argc < 4) { fprintf(stderr, "sme2fmopa_run needs threads duration_ms [utility]\n"); return 1; }
        int use_utility = (argc >= 5 && strcmp(argv[4], "utility") == 0);
        run_sme2fmopa_duration(atoi(argv[2]), atol(argv[3]), use_utility);
    } else {
        fprintf(stderr, "unknown kernel %s\n", argv[1]);
        return 1;
    }
    return 0;
}
