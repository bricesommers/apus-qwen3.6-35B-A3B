/*
 * tests/m3/bench_bf16.c — informational GEMV benchmark for c/bf16.h.
 * Reports GB/s (bf16 weight bytes streamed per call) and GFLOP/s (2*O*K
 * MACs) at the real Qwen3.6-35B-A3B shapes, scalar vs NEON vs mt.
 * GEMV (M=1, the decode path); mt uses the default pool size.
 */
#define APUS_BF16_IMPLEMENTATION
#include "bf16.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t rng_state = 0x123456789abcdef0ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

typedef void (*gemv_fn)(const uint16_t *, const uint16_t *, float *,
                        uint16_t *, size_t, size_t);

static void bench_shape(const char *name, gemv_fn fn,
                        const uint16_t *w, const uint16_t *x, float *xf,
                        uint16_t *y, size_t O, size_t K) {
    /* calibrate to ~0.5 s of measured time per entry */
    fn(w, x, xf, y, O, K);  /* warmup */
    double t0 = now_s();
    fn(w, x, xf, y, O, K);
    double t1 = now_s();
    size_t reps = (size_t)(0.5 / (t1 - t0 > 1e-9 ? t1 - t0 : 1e-9));
    if (reps < 3) reps = 3;
    if (reps > 2000) reps = 2000;

    double best = 1e30;
    for (int trial = 0; trial < 3; trial++) {
        double a = now_s();
        for (size_t r = 0; r < reps; r++)
            fn(w, x, xf, y, O, K);
        double b = now_s();
        double per = (b - a) / (double)reps;
        if (per < best) best = per;
    }
    double bytes = 2.0 * (double)O * (double)K;
    double flops = 2.0 * (double)O * (double)K;
    printf("  %-10s O=%6zu K=%4zu  %9.3f us  %7.2f GB/s  %7.2f GFLOP/s\n",
           name, O, K, best * 1e6, bytes / best / 1e9, flops / best / 1e9);
}

/* scalar has a different signature (no xf scratch) — adapt */
static void gemv_scalar_adapter(const uint16_t *w, const uint16_t *x,
                                float *xf, uint16_t *y, size_t O, size_t K) {
    (void)xf;
    apus_bf16_gemv_scalar(w, x, y, O, K);
}

int main(void) {
    static const struct { size_t O, K; const char *what; } shapes[] = {
        {8192, 2048,  "GDN in_proj_qkv / attn q_proj (2048->8192)"},
        {4096, 2048,  "GDN in_proj_z (2048->4096)"},
        {2048, 4096,  "GDN out_proj / attn o_proj (4096->2048)"},
        {512, 2048,   "attn k/v_proj (2048->512)"},
        {1024, 2048,  "expert gate_up (2048->1024)"},
        {2048, 512,   "expert down (512->2048)"},
        {256, 2048,   "router (2048->256)"},
    };
    printf("bench_bf16: GEMV decode-path benchmark (M=1); "
           "pool threads = %d\n", apus_pool_threads());
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        size_t O = shapes[s].O, K = shapes[s].K;
        uint16_t *w = malloc(O * K * 2);
        uint16_t *x = malloc(K * 2);
        uint16_t *y = malloc(O * 2);
        float *xf = malloc(K * sizeof(float));
        for (size_t i = 0; i < O * K; i++)
            w[i] = apus_bf16_bits(
                (float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0 - 2.0));
        for (size_t i = 0; i < K; i++)
            x[i] = apus_bf16_bits(
                (float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0 - 2.0));

        printf("  [%s]\n", shapes[s].what);
        bench_shape("scalar", gemv_scalar_adapter, w, x, xf, y, O, K);
#ifdef __ARM_NEON
        bench_shape("neon", apus_bf16_gemv_neon, w, x, xf, y, O, K);
#endif
        bench_shape("mt", apus_bf16_gemv_mt, w, x, xf, y, O, K);
        free(w); free(x); free(y); free(xf);
    }
    return 0;
}
