/*
 * tests/m10/bench_metal.c — CPU-vs-Metal microbench for the M10 Metal
 * backend: decode GEMV (M=1) and prefill-class GEMM at the real Qwen
 * shapes, plus the router scoring matvec (256x2048) and the 248320x2048
 * head. Effective weight-streaming bandwidth included. Informational —
 * NOT a gate.
 *
 * The CPU side is the DISPATCHED hot path (M9b ILP NEON / M9c 8-row
 * groups, pool at APUS_THREADS). Metal is the direct entry points
 * (synchronous dispatch, one command buffer per op).
 */
#define APUS_BF16_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION
#include "bf16.h"
#include "moe.h"
#include "backend_metal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void bench_gemv(size_t O, size_t K, const char *tag, int iters) {
    uint16_t *w = malloc(O * K * 2);
    uint16_t *x = malloc(K * 2);
    uint16_t *y = malloc(O * 2);
    for (size_t i = 0; i < O * K; i++)
        w[i] = (uint16_t)(0x2E00 | (rng_u64() % 1024));
    for (size_t i = 0; i < K; i++)
        x[i] = apus_bf16_bits((float)(i % 7) * 0.25f);
    double bc = 1e9, bm = 1e9;
    volatile uint16_t sink;
    for (int r = 0; r < iters; r++) {
        double t = now_s();
        ApusScratchMark mk = apus_scratch_mark();
        apus_bf16_gemv_hot(w, x, y, O, K);
        apus_scratch_reset(mk);
        sink = y[0];
        double d = now_s() - t;
        if (d < bc) bc = d;
        t = now_s();
        apus_metal_bf16_gemv(w, x, y, O, K);
        sink = y[0];
        d = now_s() - t;
        if (d < bm) bm = d;
    }
    (void)sink;
    double gbs = (double)O * K * 2 / 1e9;
    printf("  gemv %-10s O=%-6zu K=%-5zu cpu %8.3f ms (%5.1f GB/s)  "
           "metal %8.3f ms (%5.1f GB/s)  x%.2f\n", tag, O, K,
           bc * 1e3, gbs / bc, bm * 1e3, gbs / bm, bc / bm);
    free(w); free(x); free(y);
}

static void bench_gemm(size_t M, size_t O, size_t K, const char *tag,
                       int iters) {
    uint16_t *w = malloc(O * K * 2);
    uint16_t *x = malloc(M * K * 2);
    uint16_t *y = malloc(M * O * 2);
    for (size_t i = 0; i < O * K; i++)
        w[i] = (uint16_t)(0x2E00 | (rng_u64() % 1024));
    for (size_t i = 0; i < M * K; i++)
        x[i] = apus_bf16_bits((float)(i % 7) * 0.25f);
    double bc = 1e9, bm = 1e9;
    volatile uint16_t sink;
    for (int r = 0; r < iters; r++) {
        double t = now_s();
        ApusScratchMark mk = apus_scratch_mark();
        apus_bf16_gemm_hot(w, x, y, M, O, K);
        apus_scratch_reset(mk);
        sink = y[0];
        double d = now_s() - t;
        if (d < bc) bc = d;
        t = now_s();
        apus_metal_bf16_gemm(w, x, y, M, O, K);
        sink = y[0];
        d = now_s() - t;
        if (d < bm) bm = d;
    }
    (void)sink;
    double gf = 2.0 * M * O * K / 1e9;
    printf("  gemm %-10s M=%-4zu O=%-6zu K=%-5zu cpu %8.3f ms (%5.1f GF/s) "
           " metal %8.3f ms (%5.1f GF/s)  x%.2f\n", tag, M, O, K,
           bc * 1e3, gf / bc, bm * 1e3, gf / bm, bc / bm);
    free(w); free(x); free(y);
}

static void bench_matvec(size_t O, size_t K, int iters) {
    uint16_t *w = malloc(O * K * 2);
    uint16_t *x = malloc(K * 2);
    float *y = malloc(O * sizeof(float));
    for (size_t i = 0; i < O * K; i++)
        w[i] = (uint16_t)(0x2E00 | (rng_u64() % 1024));
    for (size_t i = 0; i < K; i++)
        x[i] = apus_bf16_bits((float)(i % 7) * 0.25f);
    double bc = 1e9, bm = 1e9;
    volatile float sink;
    for (int r = 0; r < iters; r++) {
        double t = now_s();
        ApusScratchMark mk = apus_scratch_mark();
        apus_moe_matvec_f32_hot(w, x, y, O, K);
        apus_scratch_reset(mk);
        sink = y[0];
        double d = now_s() - t;
        if (d < bc) bc = d;
        t = now_s();
        apus_metal_bf16_matvec_f32(w, x, y, O, K);
        sink = y[0];
        d = now_s() - t;
        if (d < bm) bm = d;
    }
    (void)sink;
    printf("  matvec-f32 O=%-6zu K=%-5zu cpu %8.3f ms   metal %8.3f ms   "
           "x%.2f\n", O, K, bc * 1e3, bm * 1e3, bc / bm);
    free(w); free(x); free(y);
}

int main(void) {
    printf("bench_metal: CPU hot path vs Metal (best of N, threads %d)\n",
           apus_pool_threads());
    char err[256];
    if (apus_metal_enable(err, sizeof err)) {
        printf("  metal unavailable: %s\n", err);
        return 1;
    }
    /* decode GEMV shapes */
    bench_gemv(8192, 2048, "gdn-qkv/wq", 9);
    bench_gemv(2048, 4096, "gdn-out/wo", 9);
    bench_gemv(512, 2048, "wk/wv+sh-gu", 9);
    bench_gemv(1024, 2048, "expert-gu", 9);
    bench_gemv(2048, 512, "expert-d", 9);
    bench_gemv(256, 2048, "router", 9);
    bench_gemv(248320, 2048, "head", 3);
    bench_matvec(256, 2048, 9);   /* fp32-out matvec (kept machinery) */
    /* prefill-class GEMM (the M9 batched phase-A projections) */
    bench_gemm(512, 8192, 2048, "gdn-qkv/wq", 5);
    bench_gemm(512, 2048, 4096, "gdn-out/wo", 5);
    bench_gemm(512, 1024, 2048, "expert-gu", 5);
    bench_gemm(512, 2048, 512, "expert-d", 5);
    printf("  metal: %llu B wrapped, %llu B uploaded, %llu dispatches\n",
           (unsigned long long)apus_metal_bytes_wrapped(),
           (unsigned long long)apus_metal_bytes_uploaded(),
           (unsigned long long)apus_metal_dispatches());
    apus_metal_disable();
    return 0;
}
