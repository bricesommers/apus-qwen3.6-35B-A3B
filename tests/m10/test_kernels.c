/*
 * tests/m10/test_kernels.c — kernel-level gates for the M10 Metal
 * backend (c/backend_metal.mm).
 *
 * THE CONTRACT: every Metal shader replicates the DISPATCHED CPU
 * kernel's rounding sequence EXACTLY (c/backend_metal.h) — the M9b ILP
 * NEON GEMV/GEMM per-row order and the c/moe.h fp32-out matvec anchor.
 * This suite pins BITWISE identity (not a tolerance class) between the
 * Metal ops and the CPU hot paths on every shape, plus the fail-soft
 * behavior (budget exhaustion -> CPU fallback, bitwise) and the
 * hook-table wiring (hooked gemv_hot == unhooked gemv_hot, bitwise).
 *
 *   1. bf16_gemv: Metal == apus_bf16_gemv_hot BITWISE — the real Qwen
 *      shapes off hidden 2048 (8192x2048 GDN qkv / attn wq, 4096x2048
 *      GDN z, 32x2048 GDN b/a, 2048x4096 GDN out / attn wo, 512x2048
 *      attn wk/wv + shared g/u, 1024x2048 expert gate_up, 2048x512
 *      expert down / shared d, 1x2048 shared gate, 256x2048 router
 *      scoring, the 248320x2048 lm_head slab-wise), odd tails
 *      (1x1, 7x3, 33x100, 129x1032, 5x1000), group boundaries
 *      (O=20: 2x8+1x4), IEEE specials fill (no NaN codes — the same
 *      policy as tests/m12: GPU NaN payload selection in mixed NaN
 *      arithmetic is not part of the contract; normative inputs are
 *      finite).
 *   2. bf16_gemm: Metal == apus_bf16_gemm_hot BITWISE at
 *      M = 1,2,3,5,64,300 on Qwen shapes; M-independence (GEMM row 3 ==
 *      GEMV of that row, both Metal).
 *   3. bf16_matvec_f32: Metal == apus_moe_matvec_f32_hot BITWISE (kept
 *      generic machinery — the Qwen router scores via the gemv hook;
 *      pinned here regardless).
 *   4. Fail-soft: APUS_METAL_DENSE_MB=1 — a >1MB weight returns
 *      "unsupported" and the hooked call falls back to the CPU kernel
 *      BITWISE; a small weight still goes to the GPU.
 *   5. Hook wiring: apus_bf16_gemv_hot with hooks filled == _cpu entry
 *      point BITWISE; determinism (repeated calls bitwise).
 *
 * Run from the repository root.
 */
#define APUS_BF16_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION
#include "bf16.h"
#include "moe.h"
#include "backend_metal.h"

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

static uint64_t rng_state = 0xC2B2AE3D27D4EB4Full;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_float(void) {   /* uniform in (-2, 2) */
    return (float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0
                   - 2.0);
}
static uint16_t rng_bf16_scaled(float s) {
    return apus_bf16_bits(rng_float() * s);
}
/* IEEE specials, no NaN codes (see header note). */
static uint16_t special_code(void) {
    static const uint16_t sp[] = {
        0x0000, 0x8000, 0x7F80, 0xFF80, 0x7F7F, 0xFF7F,
        0x0001, 0x8001, 0x007F, 0x0080, 0xBF80, 0x3DCC,
    };
    if (rng_u64() % 4 == 0)
        return sp[rng_u64() % (sizeof sp / sizeof sp[0])];
    return rng_bf16_scaled(1.0f);
}

static uint64_t g_digest = 0xCBF29CE484222325ull;

/* Weight buffers must NOT be freed while the backend is enabled: the
 * zero-copy cache is keyed by CPU pointer, and a freed-then-reused
 * address would alias a stale wrap (the stable-pointer invariant,
 * c/backend_metal.h). Registry-freed at exit, after apus_metal_disable(). */
static void *g_alloc[512];
static int g_nalloc;
static void *keep(void *p) {
    if (g_nalloc < 512) g_alloc[g_nalloc++] = p;
    return p;
}
static void free_all(void) {
    for (int i = 0; i < g_nalloc; i++) free(g_alloc[i]);
    g_nalloc = 0;
}
static void digest_bytes(const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) {
        g_digest ^= b[i];
        g_digest *= 0x100000001B3ull;
    }
}

static void test_gemv_shape(size_t O, size_t K, int specials) {
    uint16_t *w = keep(malloc(O * K * 2));
    uint16_t *x = keep(malloc(K * 2));
    uint16_t *y1 = keep(malloc(O * 2));
    uint16_t *y2 = keep(malloc(O * 2));
    for (size_t i = 0; i < O * K; i++)
        w[i] = specials ? special_code() : rng_bf16_scaled(0.5f);
    for (size_t i = 0; i < K; i++)
        x[i] = specials ? special_code() : rng_bf16_scaled(1.0f);
    ApusScratchMark mk = apus_scratch_mark();
    apus_bf16_gemv_hot_cpu(w, x, y1, O, K);   /* the TRUE CPU dispatch */
    apus_scratch_reset(mk);
    int rc = apus_metal_bf16_gemv(w, x, y2, O, K);
    CHECK(rc == 0, "gemv O=%zu K=%zu: metal unsupported", O, K);
    CHECK(memcmp(y1, y2, O * 2) == 0,
          "gemv O=%zu K=%zu sp=%d: Metal != dispatched CPU", O, K, specials);
    /* determinism */
    memset(y2, 0, O * 2);
    apus_metal_bf16_gemv(w, x, y2, O, K);
    CHECK(memcmp(y1, y2, O * 2) == 0, "gemv O=%zu K=%zu: nondeterministic",
          O, K);
    digest_bytes(y2, O * 2);
}

static void test_head_slab(void) {
    size_t V = 248320, K = 2048, SL = V / 8;
    uint16_t *w = keep(malloc((size_t)SL * K * 2));
    uint16_t *x = keep(malloc(K * 2));
    uint16_t *y1 = keep(malloc(SL * 2));
    uint16_t *y2 = keep(malloc(SL * 2));
    for (size_t i = 0; i < K; i++) x[i] = rng_bf16_scaled(1.0f);
    long bad = 0;
    for (size_t sl = 0; sl < 8; sl++) {
        for (size_t i = 0; i < (size_t)SL * K; i++)
            w[i] = (uint16_t)(rng_u64() & 0x3FFF) | 0x2E00;
        ApusScratchMark mk = apus_scratch_mark();
        apus_bf16_gemv_hot(w, x, y1, SL, K);
        apus_scratch_reset(mk);
        apus_metal_bf16_gemv(w, x, y2, SL, K);
        if (memcmp(y1, y2, SL * 2) != 0) bad++;
        digest_bytes(y2, SL * 2);
    }
    CHECK(bad == 0, "head slabs: %ld/8 Metal != CPU", bad);
    printf("  head 248320x2048: 8 slabs Metal == CPU bitwise\n");
}

static void test_gemm_shape(size_t M, size_t O, size_t K) {
    uint16_t *w = keep(malloc(O * K * 2));
    uint16_t *x = keep(malloc(M * K * 2));
    uint16_t *y1 = keep(malloc(M * O * 2));
    uint16_t *y2 = keep(malloc(M * O * 2));
    for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
    for (size_t i = 0; i < M * K; i++) x[i] = rng_bf16_scaled(1.0f);
    ApusScratchMark mk = apus_scratch_mark();
    apus_bf16_gemm_hot_cpu(w, x, y1, M, O, K);  /* the TRUE CPU dispatch */
    apus_scratch_reset(mk);
    int rc = apus_metal_bf16_gemm(w, x, y2, M, O, K);
    CHECK(rc == 0, "gemm M=%zu O=%zu K=%zu: metal unsupported", M, O, K);
    CHECK(memcmp(y1, y2, M * O * 2) == 0,
          "gemm M=%zu O=%zu K=%zu: Metal != dispatched CPU", M, O, K);
    digest_bytes(y2, M * O * 2);
}

static void test_matvec_shape(size_t O, size_t K) {
    uint16_t *w = keep(malloc(O * K * 2));
    uint16_t *x = keep(malloc(K * 2));
    float *y1 = keep(malloc(O * sizeof(float)));
    float *y2 = keep(malloc(O * sizeof(float)));
    for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
    for (size_t i = 0; i < K; i++) x[i] = rng_bf16_scaled(1.0f);
    ApusScratchMark mk = apus_scratch_mark();
    apus_moe_matvec_f32(w, x, y1, O, K);   /* scalar anchor (== NEON hot,
                                              pinned in tests/m4a/m6c) */
    apus_scratch_reset(mk);
    int rc = apus_metal_bf16_matvec_f32(w, x, y2, O, K);
    CHECK(rc == 0, "matvec O=%zu K=%zu: metal unsupported", O, K);
    CHECK(memcmp(y1, y2, O * sizeof(float)) == 0,
          "matvec O=%zu K=%zu: Metal != dispatched CPU", O, K);
    digest_bytes(y2, O * sizeof(float));
}

int main(void) {
    printf("test_kernels: Metal ops vs dispatched CPU kernels (bitwise)\n");

    /* 4. fail-soft FIRST (budget is read at init) */
    setenv("APUS_METAL_DENSE_MB", "1", 1);
    {
        char err[256];
        CHECK(apus_metal_enable(err, sizeof err) == 0,
              "enable: %s", err);
        size_t O = 8192, K = 2048;      /* 32 MB — over the 1 MB budget */
        uint16_t *w = keep(malloc(O * K * 2));
        uint16_t *x = keep(malloc(K * 2));
        uint16_t *y1 = keep(malloc(O * 2));
        uint16_t *y2 = keep(malloc(O * 2));
        for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
        for (size_t i = 0; i < K; i++) x[i] = rng_bf16_scaled(1.0f);
        int rc = apus_metal_bf16_gemv(w, x, y1, O, K);
        CHECK(rc != 0, "budget: big weight must return unsupported");
        ApusScratchMark mk = apus_scratch_mark();
        apus_bf16_gemv_hot(w, x, y1, O, K);   /* hooked -> CPU fallback */
        apus_scratch_reset(mk);
        mk = apus_scratch_mark();
        apus_bf16_gemv_hot_cpu(w, x, y2, O, K);
        apus_scratch_reset(mk);
        CHECK(memcmp(y1, y2, O * 2) == 0,
              "budget fallback: hooked != pure CPU");
        /* small weight fits the budget -> GPU path, bitwise */
        size_t O2 = 64, K2 = 256;
        uint16_t *w2 = keep(malloc(O2 * K2 * 2));
        uint16_t *x2 = keep(malloc(K2 * 2));
        uint16_t *ya = keep(malloc(O2 * 2));
        uint16_t *yb = keep(malloc(O2 * 2));
        for (size_t i = 0; i < O2 * K2; i++) w2[i] = rng_bf16_scaled(0.5f);
        for (size_t i = 0; i < K2; i++) x2[i] = rng_bf16_scaled(1.0f);
        rc = apus_metal_bf16_gemv(w2, x2, ya, O2, K2);
        CHECK(rc == 0, "budget: small weight must go to GPU");
        mk = apus_scratch_mark();
        apus_bf16_gemv_hot_cpu(w2, x2, yb, O2, K2);
        apus_scratch_reset(mk);
        CHECK(memcmp(ya, yb, O2 * 2) == 0,
              "budget small: Metal != CPU");
        digest_bytes(y1, O * 2);
        apus_metal_disable();
        free_all();      /* cache dropped: safe to free now */
        setenv("APUS_METAL_DENSE_MB", "8192", 1);
        CHECK(apus_metal_enable(err, sizeof err) == 0,
              "re-enable: %s", err);
    }
    CHECK(apus_metal_is_enabled(), "backend not enabled after re-init");

    /* 1. bf16_gemv bitwise battery */
    test_gemv_shape(1, 1, 0);
    test_gemv_shape(7, 3, 0);
    test_gemv_shape(33, 100, 0);
    test_gemv_shape(20, 2048, 0);        /* 2x8 + 1x4 group boundary */
    test_gemv_shape(129, 1032, 0);
    test_gemv_shape(5, 1000, 0);
    test_gemv_shape(64, 4096, 0);
    test_gemv_shape(64, 256, 1);         /* IEEE specials (no NaN) */
    test_gemv_shape(8192, 2048, 0);      /* GDN qkv / attn wq */
    test_gemv_shape(4096, 2048, 0);      /* GDN z */
    test_gemv_shape(32, 2048, 0);        /* GDN b/a */
    test_gemv_shape(2048, 4096, 0);      /* GDN out / attn wo */
    test_gemv_shape(512, 2048, 0);       /* attn wk/wv + shared g/u */
    test_gemv_shape(1024, 2048, 0);      /* expert gate_up */
    test_gemv_shape(2048, 512, 0);       /* expert down / shared d */
    test_gemv_shape(1, 2048, 0);         /* shared-expert gate */
    test_gemv_shape(256, 2048, 0);       /* router scoring (moe_route) */
    test_head_slab();

    /* 2. bf16_gemm bitwise + M-independence */
    {
        size_t O = 1024, K = 2048, M = 5;
        uint16_t *w = keep(malloc(O * K * 2));
        uint16_t *x = keep(malloc(M * K * 2));
        uint16_t *y1 = keep(malloc(M * O * 2));
        uint16_t *y2 = keep(malloc(M * O * 2));
        for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
        for (size_t i = 0; i < M * K; i++) x[i] = rng_bf16_scaled(1.0f);
        apus_metal_bf16_gemm(w, x + 3 * K, y1, 1, O, K);
        apus_metal_bf16_gemm(w, x, y2, M, O, K);
        CHECK(memcmp(y1, y2 + 3 * O, O * 2) == 0,
              "gemm M-independence (Metal): row 3 != M=1 row");
    }
    test_gemm_shape(1, 8192, 2048);
    test_gemm_shape(2, 1024, 2048);
    test_gemm_shape(3, 2048, 512);
    test_gemm_shape(5, 4096, 2048);
    test_gemm_shape(64, 1024, 2048);
    test_gemm_shape(300, 2048, 4096);    /* prefill-class M, odd tail */

    /* 3. fp32-out matvec bitwise (kept generic machinery) */
    test_matvec_shape(256, 2048);
    test_matvec_shape(32, 256);
    test_matvec_shape(7, 3);
    test_matvec_shape(100, 33);
    test_matvec_shape(1, 1);
    test_matvec_shape(64, 4096);

    /* 5. hook wiring: hooked gemv_hot == _cpu entry, bitwise */
    {
        size_t O = 4096, K = 2048;
        uint16_t *w = keep(malloc(O * K * 2));
        uint16_t *x = keep(malloc(K * 2));
        uint16_t *y1 = keep(malloc(O * 2));
        uint16_t *y2 = keep(malloc(O * 2));
        for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
        for (size_t i = 0; i < K; i++) x[i] = rng_bf16_scaled(1.0f);
        ApusScratchMark mk = apus_scratch_mark();
        apus_bf16_gemv_hot(w, x, y1, O, K);        /* hook taken (Metal) */
        apus_scratch_reset(mk);
        mk = apus_scratch_mark();
        apus_bf16_gemv_hot_cpu(w, x, y2, O, K);    /* pure CPU dispatch */
        apus_scratch_reset(mk);
        CHECK(memcmp(y1, y2, O * 2) == 0,
              "hook wiring: hooked gemv_hot != _cpu");
        CHECK(apus_metal_dispatches() > 0,
              "hook wiring: no GPU dispatch happened");
        digest_bytes(y1, O * 2);
    }

    printf("  metal: %llu B wrapped zero-copy, %llu B uploaded, %llu "
           "dispatches\n",
           (unsigned long long)apus_metal_bytes_wrapped(),
           (unsigned long long)apus_metal_bytes_uploaded(),
           (unsigned long long)apus_metal_dispatches());
    printf("digest %016llx\n", (unsigned long long)g_digest);
    printf("test_kernels: %ld checks, %d failures\n", checks, failures);
    apus_metal_disable();
    free_all();          /* cache dropped: safe to free */
    return failures ? 1 : 0;
}
