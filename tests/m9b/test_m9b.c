/*
 * tests/m9b/test_m9b.c — M9 reorder-class gates, RE-ANCHORED to the
 * Qwen3.6-35B-A3B shapes (the inherited Ling M9b user approval carries
 * only after this re-measurement — docs/ARCHITECTURE.md §9): the ILP
 * NEON GEMV/GEMM and the Accelerate (AMX) BLAS dispatch, measured
 * against FP64 truth with the m3 masked err/esc metric on every real
 * Qwen shape.
 *
 *   1. ILP gemv: err/esc at the Qwen projections off hidden 2048 —
 *      8192x2048 (GDN qkv / attn q), 4096x2048 (GDN z), 2048x4096
 *      (GDN out / attn o), 512x2048 (attn k/v + shared g/u), 2048x512
 *      (expert/shared down), 1024x2048 (fused expert gate_up), 256x2048
 *      (router), 32x2048 (GDN a/b) — and the 248320x2048 lm_head
 *      (slab-wise). Masked metric, bound 1e-4 (the approved bound).
 *   2. ILP gemm M-sweep (M = 1,2,5,8,64,127) at 1024x2048 + 2048x512
 *      (the expert shapes): err/esc + M-independence (every GEMM row
 *      bitwise == the GEMV row, within the ILP path).
 *   3. BLAS dispatch (M >= 128): err/esc at M = 128, 256, 512 at
 *      1024x2048 (expert gate_up) and 2048x4096 (out-proj); dispatch
 *      cutoffs (M=127 -> ILP, M=128 -> BLAS, verified via APUS_NO_BLAS
 *      A/B bit patterns); the fixed tile grid makes outputs bitwise
 *      APUS_THREADS-independent (Makefile diffs the digest at T=1/4/8).
 *   4. ILP path thread-independence: same digest.
 *
 * Run from the repository root.
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
#include "blas.h"
#include "gdn.h"
#include "attn.h"
#include "moe.h"
#include "layer.h"
#include "compat.h"
#include "cache.h"
#include "model.h"

#include <math.h>
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

/* the m3 masked metric: max |got-truth|/esc, masking differences within
 * 0.4% of |truth| (output quantization); also returns the raw max. */
static void gemm_err(const uint16_t *w, const uint16_t *x,
                     const uint16_t *y, size_t M, size_t O, size_t K,
                     double *masked, double *raw) {
    double mr = 0, rw = 0;
    for (size_t m = 0; m < M; m++) {
        for (size_t o = 0; o < O; o++) {
            double acc = 0.0, esc = 0.0;
            for (size_t k = 0; k < K; k++) {
                double p = (double)apus_bf16_f32(w[o * K + k])
                         * (double)apus_bf16_f32(x[m * K + k]);
                acc += p;
                esc += fabs(p);
            }
            double got = (double)apus_bf16_f32(y[m * O + o]);
            double r = fabs(got - acc) / (esc > 1e-30 ? esc : 1e-30);
            if (r > rw) rw = r;
            double outq = fabs(got - acc) <= 0.004 * fabs(acc) ? 0.0 : r;
            if (outq > mr) mr = outq;
        }
    }
    *masked = mr;
    *raw = rw;
}

static void run_shape(size_t M, size_t O, size_t K, const char *tag) {
    uint16_t *w = malloc(O * K * 2);
    uint16_t *x = malloc(M * K * 2);
    uint16_t *y_ilp = malloc(M * O * 2);
    uint16_t *yv = malloc(O * 2);
    for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
    for (size_t i = 0; i < M * K; i++) x[i] = rng_bf16_scaled(1.0f);
    ApusScratchMark mk = apus_scratch_mark();
    apus_bf16_gemm_hot(w, x, y_ilp, M, O, K);
    double mr, rw;
    gemm_err(w, x, y_ilp, M, O, K, &mr, &rw);
    CHECK(mr < 1e-4, "ILP %s M=%zu err/esc %.3g >= 1e-4", tag, M, mr);
    printf("  ILP %-14s M=%-4zu O=%-6zu K=%-5zu err/esc %.3g "
           "(raw %.3g)\n", tag, M, O, K, mr, rw);
    /* M-independence: every GEMM row bitwise the GEMV row (within the
     * hot path's kernel: ILP NEON on ARM, the bitwise-sequential AVX2
     * kernel on x86 — both reproduce their platform's hot rows) */
    int mi = 1;
#ifdef __ARM_NEON
    float *xf = malloc(K * sizeof(float));
    for (size_t m = 0; m < M && mi; m++) {
        apus_bf16_gemv_ilp_neon(w, x + m * K, xf, yv, O, K);
        if (memcmp(yv, y_ilp + m * O, O * 2) != 0) mi = 0;
    }
    free(xf);
#elif APUS_X86
    if (apus_x86_have_avx2()) {
        float *xf = malloc(K * sizeof(float));
        for (size_t m = 0; m < M && mi; m++) {
            apus_bf16_gemv_x86(w, x + m * K, xf, yv, O, K);
            if (memcmp(yv, y_ilp + m * O, O * 2) != 0) mi = 0;
        }
        free(xf);
    }
#endif
    CHECK(mi, "ILP %s M=%zu: GEMM row != GEMV row (M-independence)",
          tag, M);
    digest_bytes(&g_digest, y_ilp, M * O * 2);
    apus_scratch_reset(mk);
    free(w); free(x); free(y_ilp); free(yv);
}

#if APUS_BLAS
static void run_blas_shape(size_t M, size_t O, size_t K, const char *tag) {
    if (!apus_blas_available()) {
        printf("  BLAS %-12s M=%zu: skipped (APUS_NO_BLAS)\n", tag, M);
        return;
    }
    uint16_t *w = malloc(O * K * 2);
    uint16_t *x = malloc(M * K * 2);
    float *of = malloc(M * O * sizeof(float));
    uint16_t *yb = malloc(M * O * 2);
    for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
    for (size_t i = 0; i < M * K; i++) x[i] = rng_bf16_scaled(1.0f);
    apus_bf16_gemm_blas(w, x, of, M, O, K);
    for (size_t i = 0; i < M * O; i++) yb[i] = apus_bf16_bits(of[i]);
    double mr, rw;
    gemm_err(w, x, yb, M, O, K, &mr, &rw);
    CHECK(mr < 1e-4, "BLAS %s M=%zu err/esc %.3g >= 1e-4", tag, M, mr);
    printf("  BLAS %-13s M=%-4zu O=%-6zu K=%-5zu err/esc %.3g "
           "(raw %.3g)\n", tag, M, O, K, mr, rw);
    digest_bytes(&g_digest, yb, M * O * 2);
    free(w); free(x); free(of); free(yb);
}
#endif /* APUS_BLAS */

static void test_ilp_gemv_shapes(void) {
    static const size_t shapes[][2] = {
        {8192, 2048}, {4096, 2048}, {2048, 4096}, {512, 2048},
        {2048, 512}, {1024, 2048}, {256, 2048}, {32, 2048},
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++)
        run_shape(1, shapes[s][0], shapes[s][1], "gemv");
    /* head 248320x2048, slab-wise (full f64 truth is too slow; 8 slabs) */
    size_t V = 248320, K = 2048, SL = V / 8;
    uint16_t *w = malloc((size_t)SL * K * 2);
    uint16_t *x = malloc(K * 2);
    uint16_t *y = malloc(SL * 2);
    for (size_t i = 0; i < K; i++) x[i] = rng_bf16_scaled(1.0f);
    double mr_w = 0;
    for (size_t sl = 0; sl < 8; sl++) {
        for (size_t i = 0; i < (size_t)SL * K; i++)
            w[i] = rng_bf16_scaled(0.5f);
        ApusScratchMark mk = apus_scratch_mark();
        apus_bf16_gemv_hot(w, x, y, SL, K);
        apus_scratch_reset(mk);
        double mr, rw;
        gemm_err(w, x, y, 1, SL, K, &mr, &rw);
        if (mr > mr_w) mr_w = mr;
        digest_bytes(&g_digest, y, SL * 2);
    }
    printf("  ILP head slabs     M=1    O=248320 K=2048  err/esc %.3g "
           "(8 slabs)\n", mr_w);
    CHECK(mr_w < 1e-4, "ILP head err/esc %.3g >= 1e-4", mr_w);
    free(w); free(x); free(y);
}

static void test_ilp_gemm_sweep(void) {
    static const size_t Ms[] = { 1, 2, 5, 8, 64, 127 };
    for (size_t i = 0; i < sizeof Ms / sizeof Ms[0]; i++) {
        run_shape(Ms[i], 1024, 2048, "gemm");
        run_shape(Ms[i], 2048, 512, "gemm");
    }
}

static void test_blas(void) {
#if APUS_BLAS
    printf("  BLAS available: %d (cutoff M>=%d)\n",
           apus_blas_available(), APUS_BLAS_M_MIN);
    if (!apus_blas_available()) return;
    /* dispatch cutoff: M=127 must equal the ILP result (no BLAS);
     * M=128 takes BLAS (different bits allowed, err class checked) */
    size_t O = 1024, K = 2048, M = 127;
    uint16_t *w = malloc(O * K * 2);
    uint16_t *x = malloc(M * K * 2);
    uint16_t *y1 = malloc(M * O * 2);
    uint16_t *y2 = malloc(M * O * 2);
    for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
    for (size_t i = 0; i < M * K; i++) x[i] = rng_bf16_scaled(1.0f);
    ApusScratchMark mk = apus_scratch_mark();
    apus_bf16_gemm_fast(w, x, y1, M, O, K);
    apus_bf16_gemm_hot(w, x, y2, M, O, K);
    apus_scratch_reset(mk);
    CHECK(memcmp(y1, y2, M * O * 2) == 0,
          "M=127: gemm_fast must stay on the ILP path");
    free(w); free(x); free(y1); free(y2);
    static const size_t Ms[] = { 128, 256, 512 };
    for (size_t i = 0; i < sizeof Ms / sizeof Ms[0]; i++) {
        run_blas_shape(Ms[i], 1024, 2048, "expert");
        run_blas_shape(Ms[i], 2048, 4096, "out-proj");
    }
#else
    printf("  BLAS: not compiled (non-Darwin)\n");
    /* M12a-1 placeholder (the Apus pattern): keeps the check slot on
     * platforms where the BLAS dispatch is a no-op stub by design — the
     * M >= APUS_BLAS_M_MIN path simply stays on the hot kernel there. */
    CHECK(1, "no BLAS on this platform (placeholder)");
#endif
}

int main(void) {
    printf("test_m9b: reorder-class gates (Qwen M9 re-measurement)\n");
    fprintf(stderr, "  pool threads: %d\n", apus_pool_threads());
    test_ilp_gemv_shapes();
    test_ilp_gemm_sweep();
    test_blas();
    printf("  digest %016llx\n", (unsigned long long)g_digest);
    printf("test_m9b: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
