/*
 * tests/m9a/test_m9a.c — M9a strictly-bitwise performance pass gates,
 * re-anchored to the Qwen3.6 shapes (M9). The hot-GEMV wiring checks
 * (gemv_hot err/esc vs in-test FP64 truth at every real Qwen shape with
 * emphasis on the 8/4/1-row tail boundaries; the frozen bitwise 8-chain
 * kernel pinned == the scalar anchor at a partial-chain boundary; the
 * logits widen path SIMD == scalar) run on the Qwen projection/head
 * shapes; everything else is pinned by the milestone digests (m3/m4a/
 * m4c/m5/m6a/m6b/m6c/m7a/m8 byte-identical — see tests/m9a/README.md).
 *
 * The FNV digest is diffed across APUS_THREADS=1/4/8 by the Makefile.
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_BF16_IMPLEMENTATION
#define APUS_GDN_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION
#define APUS_LAYER_IMPLEMENTATION
#define APUS_COMPAT_IMPLEMENTATION
#define APUS_CACHE_IMPLEMENTATION
#define APUS_MODEL_IMPLEMENTATION
#include "json.h"
#include "st.h"
#include "bf16.h"
#include "gdn.h"
#include "attn.h"
#include "moe.h"
#include "layer.h"
#include "compat.h"
#include "cache.h"
#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static long checks = 0;

#define CHECK(cond, ...) do { \
    checks++; \
    if (!(cond)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_float(void) {
    return (float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 8.0 - 4.0);
}
static uint16_t rng_bf16_scaled(float s) {
    return apus_bf16_bits(rng_float() * s);
}

static void digest_bytes(uint64_t *h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) {
        *h ^= b[i];
        *h *= 0x100000001B3ull;
    }
}
static uint64_t g_digest = 0xCBF29CE484222325ull;

int main(void) {
    printf("test_m9a: strictly-bitwise perf-pass gates (Qwen M9)\n");
    fprintf(stderr, "  pool threads: %d\n", apus_pool_threads());

    /* M9b re-anchor: gemv_hot now dispatches to the ILP NEON kernel —
     * the approved bounded reorder class (tests/m9b). The gate is the
     * m3 err/esc class vs in-test FP64 truth (bound 1e-4); the 8-chain
     * bitwise kernel (apus_bf16_gemv_neon) stays pinned bitwise below. */
    static const size_t shapes[][2] = {
        {8192, 2048}, {2048, 4096}, {4096, 2048}, {1024, 2048},
        {2048, 512}, {512, 2048}, {256, 2048}, {32, 2048},
        {256, 256}, {64, 256}, {256, 64},
        {8, 33}, {9, 33}, {12, 33}, {13, 33}, {4, 33}, {5, 33},
        {1, 1}, {3, 1}, {17, 100}, {248320, 2048},
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        size_t O = shapes[s][0], K = shapes[s][1];
        uint16_t *w = malloc(O * K * 2);
        uint16_t *x = malloc(K * 2);
        uint16_t *y1 = malloc(O * 2);
        uint16_t *y2 = malloc(O * 2);
        for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
        for (size_t i = 0; i < K; i++) x[i] = rng_bf16_scaled(1.0f);
        ApusScratchMark mk = apus_scratch_mark();
        apus_bf16_gemv_hot(w, x, y2, O, K);
        apus_scratch_reset(mk);
        /* the m3 masked metric vs in-test f64 truth (the reorder
         * class; bound 1e-4 of esc) */
        double worst = 0;
        for (size_t o = 0; o < O; o++) {
            double acc = 0.0, esc = 0.0;
            for (size_t k = 0; k < K; k++) {
                double p = (double)apus_bf16_f32(w[o * K + k])
                         * (double)apus_bf16_f32(x[k]);
                acc += p;
                esc += fabs(p);
            }
            double err = fabs((double)apus_bf16_f32(y2[o]) - acc);
            double outq = err <= 0.004 * fabs(acc) ? 0.0
                          : err / (esc > 1e-30 ? esc : 1e-30);
            if (outq > worst) worst = outq;
        }
        CHECK(worst < 1e-4, "gemv_hot err/esc %.3g >= 1e-4 at O=%zu K=%zu",
              worst, O, K);
        (void)y1;
        digest_bytes(&g_digest, y2, O * 2);
        free(w); free(x); free(y1); free(y2);
    }
    /* the scalar anchor vs the BITWISE SIMD rows (the frozen m3
     * contract kernel — unchanged by M9b; M12a-2: the c/x86.h AVX2
     * kernel on x86, same bitwise contract) at a partial-chain boundary
     * (O=20: 2x8 + 1x4 groups) */
    {
        size_t O = 20, K = 2048;
        uint16_t *w = malloc(O * K * 2);
        uint16_t *x = malloc(K * 2);
        uint16_t *y1 = malloc(O * 2);
        uint16_t *y2 = malloc(O * 2);
        for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
        for (size_t i = 0; i < K; i++) x[i] = rng_bf16_scaled(1.0f);
        apus_bf16_gemv_scalar(w, x, y1, O, K);
#ifdef __ARM_NEON
        float *xf = malloc(K * sizeof(float));
        apus_bf16_gemv_neon(w, x, xf, y2, O, K);
        CHECK(memcmp(y1, y2, O * 2) == 0,
              "gemv_neon(8-chain) != scalar at O=20 K=2048");
        free(xf);
#elif APUS_X86
        if (apus_x86_have_avx2()) {
            float *xf = malloc(K * sizeof(float));
            apus_bf16_gemv_x86(w, x, xf, y2, O, K);
            CHECK(memcmp(y1, y2, O * 2) == 0,
                  "gemv_x86(8-chain) != scalar at O=20 K=2048");
            free(xf);
        } else {
            CHECK(1, "no AVX2 (placeholder)");
        }
#endif
        free(w); free(x); free(y1); free(y2);
    }
    /* logits widen path: SIMD widen == scalar widen (exact) */
    {
        size_t V = 248320;
        uint16_t *lb = malloc(V * 2);
        float *a = malloc(V * sizeof(float));
        float *b = malloc(V * sizeof(float));
        for (size_t i = 0; i < V; i++) lb[i] = rng_bf16_scaled(4.0f);
        for (size_t i = 0; i < V; i++) a[i] = apus_bf16_f32(lb[i]);
#ifdef __ARM_NEON
        apus_bf16_widen_neon(lb, b, V);
        CHECK(memcmp(a, b, V * sizeof(float)) == 0,
              "widen_neon != scalar widen (logits path)");
#elif APUS_X86
        if (apus_x86_have_avx2()) {
            apus_bf16_widen_x86(lb, b, V);
            CHECK(memcmp(a, b, V * sizeof(float)) == 0,
                  "widen_x86 != scalar widen (logits path)");
        } else {
            memcpy(b, a, V * sizeof(float));
            CHECK(1, "no AVX2 (placeholder)");
        }
#else
        memcpy(b, a, V * sizeof(float));
#endif
        digest_bytes(&g_digest, b, V * sizeof(float));
        free(lb); free(a); free(b);
    }
    printf("  digest %016llx\n", (unsigned long long)g_digest);
    printf("test_m9a: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
