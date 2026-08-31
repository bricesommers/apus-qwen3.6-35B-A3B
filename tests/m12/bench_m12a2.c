/*
 * tests/m12/bench_m12a2.c — scalar vs AVX2 timing for the M12 x86 kernels.
 * NOT a gate: prints seconds per call for the scalar anchor and the
 * dispatched (AVX2) paths on the real Qwen GEMV shapes plus a GDN step
 * and a GQA decode row (M12 re-anchor: the GDN/GQA kernels run the
 * DOCUMENTED scalar fallback on x86 — no AVX2 port; their column is the
 * dispatched path, scalar on x86). Single-threaded (APUS_THREADS=1) so
 * the scalar/AVX2 contrast is per-core.
 *
 * NOTE: under tools/docker/test-linux.sh this runs on linux/amd64 under
 * Rosetta translation on Apple Silicon — the numbers are EMULATED
 * (representative of the speedup STRUCTURE, not of native x86-64
 * hardware). Marked accordingly in tests/m12/README.md.
 */
#define APUS_BF16_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION
#define APUS_GDN_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION

#include "bf16.h"
#include "moe.h"
#include "gdn.h"
#include "attn.h"

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
static float rng_float(void) {
    return (float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0
                   - 2.0);
}

static void bench_gemv(size_t O, size_t K, int iters) {
    uint16_t *w = malloc(O * K * 2);
    uint16_t *x = malloc(K * 2);
    uint16_t *y = malloc(O * 2);
    float *xf = malloc(K * sizeof(float));
    for (size_t i = 0; i < O * K; i++)
        w[i] = apus_bf16_bits(rng_float() * 0.5f);
    for (size_t i = 0; i < K; i++) x[i] = apus_bf16_bits(rng_float());
    double t0 = now_s();
    for (int it = 0; it < iters; it++)
        apus_bf16_gemv_scalar(w, x, y, O, K);
    double t1 = now_s();
#if APUS_X86
    for (int it = 0; it < iters; it++)
        apus_bf16_gemv_x86(w, x, xf, y, O, K);
#else
    for (int it = 0; it < iters; it++)
        apus_bf16_gemv_scalar(w, x, y, O, K);
#endif
    double t2 = now_s();
    printf("  gemv O=%-6zu K=%-5zu scalar %8.3f ms  avx2 %8.3f ms  "
           "(%.2fx)\n", O, K,
           (t1 - t0) / iters * 1e3, (t2 - t1) / iters * 1e3,
           (t1 - t0) / (t2 - t1 > 1e-12 ? t2 - t1 : 1e-12));
    free(w); free(x); free(y); free(xf);
}

static void bench_gdn(size_t H, size_t D, int iters) {
    float *q = malloc(H * D * sizeof(float));
    float *k = malloc(H * D * sizeof(float));
    uint16_t *v = malloc(H * D * 2);
    float *g = malloc(H * sizeof(float));
    uint16_t *bt = malloc(H * 2);
    float *S = malloc(H * D * D * sizeof(float));
    float *o = malloc(H * D * sizeof(float));
    for (size_t i = 0; i < H * D; i++) {
        q[i] = rng_float();
        k[i] = rng_float();
        v[i] = apus_bf16_bits(rng_float());
    }
    for (size_t i = 0; i < H; i++) {
        g[i] = -5.0f * (float)((rng_u64() >> 40) / 16777216.0);
        bt[i] = apus_bf16_bits(
            (float)((rng_u64() >> 40) / 16777216.0));
    }
    memset(S, 0, H * D * D * sizeof(float));
    /* M12: the Qwen GDN recurrence step (c/gdn.h) — the documented
     * scalar fallback on x86 (no AVX2 port; the c/x86.h disposition). */
    double t0 = now_s();
    for (int it = 0; it < iters; it++)
        apus_gdn_step(S, q, k, v, g, bt, o, H, D, D);
    double t1 = now_s();
    printf("  gdn step H=%-3zu D=%-4zu dispatched %8.3f ms\n", H, D,
           (t1 - t0) / iters * 1e3);
    free(q); free(k); free(v); free(g); free(bt); free(S); free(o);
}

static void bench_attn(size_t H, size_t Hkv, size_t d, size_t Tk,
                       float scale, int iters) {
    uint16_t *q = malloc(H * d * 2);
    uint16_t *kc = malloc(Tk * Hkv * d * 2);
    uint16_t *vc = malloc(Tk * Hkv * d * 2);
    uint16_t *o = malloc(H * d * 2);
    uint16_t *abuf = malloc(Tk * 2);
    float *ebuf = malloc(Tk * sizeof(float));
    for (size_t i = 0; i < H * d; i++) q[i] = apus_bf16_bits(rng_float());
    for (size_t i = 0; i < Tk * Hkv * d; i++)
        kc[i] = apus_bf16_bits(rng_float());
    for (size_t i = 0; i < Tk * Hkv * d; i++)
        vc[i] = apus_bf16_bits(rng_float());
    double t0 = now_s();
    for (int it = 0; it < iters; it++)
        apus_attn_gqa_decode(q, kc, vc, o, Tk, H, Hkv, d, scale, abuf,
                             ebuf);
    double t1 = now_s();
    printf("  gqa decode H=%-3zu d=%-4zu Tk=%-5zu dispatched %8.3f ms\n",
           H, d, Tk, (t1 - t0) / iters * 1e3);
    free(q); free(kc); free(vc); free(o); free(abuf); free(ebuf);
}

int main(void) {
    printf("bench_m12a2: scalar vs AVX2 (APUS_THREADS=%d)\n",
           apus_pool_threads());
#if APUS_X86
    printf("  avx2 dispatch: %s (APUS_X86_DISABLE %s)\n",
           apus_x86_have_avx2() ? "active" : "INACTIVE",
           getenv("APUS_X86_DISABLE") ? "set" : "unset");
#else
    printf("  APUS_X86 == 0 on this platform — no AVX2 kernels\n");
#endif
    bench_gemv(8192, 2048, 20);      /* GDN qkv / GQA wq */
    bench_gemv(2048, 4096, 20);      /* GDN out / GQA wo */
    bench_gemv(1024, 2048, 20);      /* expert gate_up */
    bench_gemv(2048, 512, 20);       /* expert down */
    bench_gemv(248320, 2048, 2);     /* LM head */
    bench_gdn(32, 128, 50);
    bench_attn(16, 2, 256, 4096, 0.0625f, 20);
    return 0;
}
