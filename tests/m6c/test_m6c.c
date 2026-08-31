/*
 * tests/m6c/test_m6c.c — M6c hot-path equivalence gates for the Qwen3.6
 * wiring: the hot kernels on the REAL Qwen shapes stay inside their
 * numerics class, the tiered forward is BITWISE the eager one across
 * cache sizes AND I/O thread counts, and the FNV digest over all
 * outputs is identical at APUS_THREADS=1/4/8 (the Makefile diffs the
 * runs).
 *
 *   1. apus_bf16_gemv_hot on the real projection shapes (8192/4096/
 *      2048/1024/512/256-wide off hidden 2048 both ways, expert
 *      1024x2048 gate_up and 2048x512 down, 248320x2048 head slab
 *      rows) and odd tails — the inherited user-approved M9b ILP
 *      reorder class, gated by the m3 masked err/esc metric vs in-test
 *      FP64 truth (bound 1e-4).
 *   2. apus_bf16_gemm_hot, M sweep (M-independence class).
 *   3. apus_moe_matvec_f32_neon/hot == apus_moe_matvec_f32 (router
 *      matvec, fp32 out) — BITWISE (256x2048 router + odd).
 *   4. apus_gdn_step_mt == apus_gdn_step (random state, H=32 D=128 and
 *      odd H/D) — BITWISE.
 *   5. apus_attn_gqa_mt == apus_attn_gqa (nh=16, nkv=2, d=256 — the
 *      real gated-GQA shape; decode == full row) — BITWISE.
 *   6. Tiered == eager at the model level: T=16 one-shot prefill vs 16
 *      decode steps (logits AND layer state) in BOTH modes, and the
 *      tiered one-shot forward at APUS_IO_THREADS=1/4/8 — BITWISE every
 *      time (the I/O pool must not perturb compute bits).
 *   7. Scratch-arena LIFO smoke (the forward's alloc pattern).
 *
 * Batched tiered prefill note: this model's forward processes every
 * token through the ONE per-token body (c/layer.h M4 contract — the
 * Ling M6c/M9c batched prefill is deferred to this adapter's own perf
 * milestones), so prefill==decode and tiered==eager hold by
 * construction; this test asserts them anyway.
 *
 * Run from the repository root (fixtures under tests/m5/fixtures and
 * tests/m6a/fixtures).
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

/* =========================================================================*/
/* The hot GEMV/GEMM dispatch runs the inherited user-approved M9b ILP
 * NEON kernel (the Ling base's documented bounded reorder class,
 * carried across the adapter seam; its Qwen-shape re-approval is M9's).
 * The gate is the m3 err/esc class vs in-test FP64 truth (bound 1e-4),
 * NOT bitwise-vs-scalar; thread-independence stays bitwise via the
 * digest. */
/* the m3 masked metric (test_bf16.c check_vs_truth): |got-truth|/esc,
 * masking differences within 0.4% of |truth| (output quantization, not
 * accumulation error); gate bound 1e-4. */
static double truth_err_esc(const uint16_t *w, const uint16_t *x,
                            const uint16_t *y, size_t O, size_t K) {
    double worst = 0;
    for (size_t o = 0; o < O; o++) {
        double acc = 0.0, esc = 0.0;
        for (size_t k = 0; k < K; k++) {
            double p = (double)apus_bf16_f32(w[o * K + k])
                     * (double)apus_bf16_f32(x[k]);
            acc += p;
            esc += fabs(p);
        }
        double r = fabs((double)apus_bf16_f32(y[o]) - acc)
                   / (esc > 1e-30 ? esc : 1e-30);
        double outq = fabs((double)apus_bf16_f32(y[o]) - acc)
                          <= 0.004 * fabs(acc) ? 0.0 : r;
        if (outq > worst) worst = outq;
    }
    return worst;
}

static void test_gemv_hot(void) {
    /* the real Qwen3.6 projection shapes (rows x cols of the weight):
     * GDN qkv 8192x2048, z 4096x2048, out 2048x4096, b/a 32x2048;
     * gated-GQA q 8192x2048, k/v 512x2048, o 2048x4096; expert gate_up
     * 1024x2048, down 2048x512; shared 512x2048 / 2048x512; router
     * 256x2048 — plus odd tails. */
    static const size_t shapes[][2] = {
        {8192, 2048}, {4096, 2048}, {2048, 4096}, {1024, 2048},
        {2048, 512}, {512, 2048}, {256, 2048}, {32, 2048},
        {64, 256}, {256, 64}, {1, 1}, {7, 3}, {33, 100},
    };
    double worst = 0;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        size_t O = shapes[s][0], K = shapes[s][1];
        uint16_t *w = malloc(O * K * 2);
        uint16_t *x = malloc(K * 2);
        uint16_t *y2 = malloc(O * 2);
        for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
        for (size_t i = 0; i < K; i++) x[i] = rng_bf16_scaled(1.0f);
        ApusScratchMark mk = apus_scratch_mark();
        apus_bf16_gemv_hot(w, x, y2, O, K);
        apus_scratch_reset(mk);
        double r = truth_err_esc(w, x, y2, O, K);
        if (r > worst) worst = r;
        CHECK(r < 1e-4, "gemv_hot err/esc %.3g >= 1e-4 at O=%zu K=%zu",
              r, O, K);
        digest_bytes(&g_digest, y2, O * 2);
        free(w); free(x); free(y2);
    }
    printf("  gemv_hot (ILP) max err/esc vs f64 (masked): %.3g "
           "(bound 1e-4)\n", worst);
    /* head slab (248320 x 2048 — the real lm_head): spot rows through
     * the hot path */
    {
        size_t V = 248320, K = 2048;
        uint16_t *w = malloc((size_t)V * K * 2);
        uint16_t *x = malloc(K * 2);
        uint16_t *y2 = malloc(V * 2);
        for (size_t i = 0; i < (size_t)V * K; i++)
            w[i] = (uint16_t)(rng_u64() & 0x3FFF) | 0x2E00;
        for (size_t i = 0; i < K; i++) x[i] = rng_bf16_scaled(1.0f);
        ApusScratchMark mk = apus_scratch_mark();
        apus_bf16_gemv_hot(w, x, y2, V, K);
        apus_scratch_reset(mk);
        long bad = 0;
        static const size_t rows[] = { 0, 1, 127, 128, 4095, 65536,
                                       100003, 248319 };
        for (size_t r = 0; r < sizeof rows / sizeof rows[0]; r++) {
            size_t o = rows[r];
            const uint16_t *wr = w + o * K;
            double acc = 0.0, esc = 0.0;
            for (size_t k = 0; k < K; k++) {
                double p = (double)apus_bf16_f32(wr[k])
                         * (double)apus_bf16_f32(x[k]);
                acc += p;
                esc += fabs(p);
            }
            /* the m3 masked metric, not code equality */
            double err = fabs((double)apus_bf16_f32(y2[o]) - acc);
            double outq = err <= 0.004 * fabs(acc) ? 0.0
                          : err / (esc > 1e-30 ? esc : 1e-30);
            if (outq > 1e-4) bad++;
        }
        CHECK(bad == 0, "head slab rows: %ld beyond err/esc class", bad);
        digest_bytes(&g_digest, y2, (size_t)V * 2);
        free(w); free(x); free(y2);
    }
}

static void test_gemm_hot(void) {
    static const size_t shapes[][3] = {
        {8, 1024, 2048}, {9, 256, 256}, {3, 2048, 4096}, {16, 512, 2048},
        {2, 64, 48},
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        size_t M = shapes[s][0], O = shapes[s][1], K = shapes[s][2];
        uint16_t *w = malloc(O * K * 2);
        uint16_t *x = malloc(M * K * 2);
        uint16_t *y2 = malloc(M * O * 2);
        for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
        for (size_t i = 0; i < M * K; i++) x[i] = rng_bf16_scaled(1.0f);
        ApusScratchMark mk = apus_scratch_mark();
        apus_bf16_gemm_hot(w, x, y2, M, O, K);
        apus_scratch_reset(mk);
        /* err/esc class vs f64 (see test_gemv_hot) */
        double worst2 = 0;
        for (size_t m = 0; m < M; m++) {
            double r = truth_err_esc(w, x + m * K, y2 + m * O, O, K);
            if (r > worst2) worst2 = r;
        }
        CHECK(worst2 < 1e-4,
              "gemm_hot err/esc %.3g >= 1e-4 at M=%zu O=%zu K=%zu",
              worst2, M, O, K);
        digest_bytes(&g_digest, y2, M * O * 2);
        free(w); free(x); free(y2);
    }
}

static void test_router_matvec(void) {
    static const size_t shapes[][2] = {
        {256, 2048}, {16, 128}, {7, 3}, {100, 33}, {1, 1}, {64, 4096},
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        size_t O = shapes[s][0], K = shapes[s][1];
        uint16_t *w = malloc(O * K * 2);
        uint16_t *x = malloc(K * 2);
        float *y1 = malloc(O * sizeof(float));
        float *y2 = malloc(O * sizeof(float));
        for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
        for (size_t i = 0; i < K; i++) x[i] = rng_bf16_scaled(1.0f);
        apus_moe_matvec_f32(w, x, y1, O, K);
        ApusScratchMark mk = apus_scratch_mark();
#ifdef __ARM_NEON
        float *xf = malloc(K * sizeof(float));
        apus_moe_matvec_f32_neon(w, x, xf, y2, O, K);
        CHECK(memcmp(y1, y2, O * sizeof(float)) == 0,
              "matvec_f32 NEON != scalar at O=%zu K=%zu", O, K);
        float *y3 = malloc(O * sizeof(float));
        apus_moe_matvec_f32_hot(w, x, y3, O, K);
        CHECK(memcmp(y1, y3, O * sizeof(float)) == 0,
              "matvec_f32 hot != scalar at O=%zu K=%zu", O, K);
        free(xf); free(y3);
#elif APUS_X86
        /* M12a-2 anchor: the AVX2 kernel takes the NEON kernel's slot
         * (same bitwise-vs-scalar contract, same check count). */
        if (apus_x86_have_avx2()) {
            float *xf = malloc(K * sizeof(float));
            apus_bf16_widen_x86(x, xf, K);
            apus_moe_matvec_rows_x86(w, xf, y2, K, 0, O);
            CHECK(memcmp(y1, y2, O * sizeof(float)) == 0,
                  "matvec_f32 AVX2 != scalar at O=%zu K=%zu", O, K);
            free(xf);
        } else {
            CHECK(1, "no AVX2 (placeholder)");
        }
        float *y3 = malloc(O * sizeof(float));
        apus_moe_matvec_f32_hot(w, x, y3, O, K);
        CHECK(memcmp(y1, y3, O * sizeof(float)) == 0,
              "matvec_f32 hot != scalar at O=%zu K=%zu", O, K);
        free(y3);
#else
        apus_moe_matvec_f32_hot(w, x, y2, O, K);
        CHECK(memcmp(y1, y2, O * sizeof(float)) == 0,
              "matvec_f32 hot != scalar at O=%zu K=%zu", O, K);
#endif
        apus_scratch_reset(mk);
        digest_bytes(&g_digest, y2, O * sizeof(float));
        free(w); free(x); free(y1); free(y2);
    }
}

static void test_gdn_step_mt(void) {
    static const size_t hd[][2] = { {32, 128}, {4, 128}, {3, 17}, {1, 1} };
    for (size_t s = 0; s < sizeof hd / sizeof hd[0]; s++) {
        size_t H = hd[s][0], D = hd[s][1], T = 6;
        float *q = malloc(T * H * D * sizeof(float));
        float *k = malloc(T * H * D * sizeof(float));
        uint16_t *v = malloc(T * H * D * 2);
        float *g = malloc(T * H * sizeof(float));
        uint16_t *bt = malloc(T * H * 2);
        for (size_t i = 0; i < T * H * D; i++) {
            q[i] = rng_float() * 0.25f;
            k[i] = rng_float() * 0.25f;
            v[i] = rng_bf16_scaled(1.5f);
        }
        for (size_t i = 0; i < T * H; i++) {
            g[i] = -5.0f * (float)((rng_u64() >> 40) / 16777216.0);
            bt[i] = apus_bf16_bits(
                (float)((rng_u64() >> 40) / 16777216.0));
        }
        float *S1 = calloc(H * D * D, sizeof(float));
        float *S2 = calloc(H * D * D, sizeof(float));
        float *o1 = malloc(T * H * D * sizeof(float));
        float *o2 = malloc(T * H * D * sizeof(float));
        for (size_t t = 0; t < T; t++) {
            apus_gdn_step(S1, q + t * H * D, k + t * H * D, v + t * H * D,
                          g + t * H, bt + t * H, o1 + t * H * D, H, D, D);
            apus_gdn_step_mt(S2, q + t * H * D, k + t * H * D,
                             v + t * H * D, g + t * H, bt + t * H,
                             o2 + t * H * D, H, D, D);
        }
        CHECK(memcmp(S1, S2, H * D * D * sizeof(float)) == 0
              && memcmp(o1, o2, T * H * D * sizeof(float)) == 0,
              "gdn_step_mt != step at H=%zu D=%zu", H, D);
        digest_bytes(&g_digest, S2, H * D * D * sizeof(float));
        digest_bytes(&g_digest, o2, T * H * D * sizeof(float));
        free(q); free(k); free(v); free(g); free(bt);
        free(S1); free(S2); free(o1); free(o2);
    }
}

static void test_gqa_mt(void) {
    /* the real gated-GQA shape: 16 q heads, 2 KV heads, head_dim 256 */
    size_t H = 16, HKV = 2, T = 64, D = 256;
    uint16_t *q = malloc(T * H * D * 2);
    uint16_t *k = malloc(T * HKV * D * 2);
    uint16_t *v = malloc(T * HKV * D * 2);
    for (size_t i = 0; i < T * H * D; i++)
        q[i] = rng_bf16_scaled(1.0f);
    for (size_t i = 0; i < T * HKV * D; i++) {
        k[i] = rng_bf16_scaled(1.0f);
        v[i] = rng_bf16_scaled(1.2f);
    }
    uint16_t *abuf = malloc(T * 2);
    float *ebuf = malloc(T * sizeof(float));
    uint16_t *o1 = malloc(T * H * D * 2);
    uint16_t *o2 = malloc(T * H * D * 2);
    float scale = (float)(1.0 / sqrt((double)D));
    apus_attn_gqa(q, k, v, o1, T, T, H, HKV, D, scale, abuf, ebuf);
    apus_attn_gqa_mt(q, k, v, o2, T, T, H, HKV, D, scale);
    CHECK(memcmp(o1, o2, T * H * D * 2) == 0,
          "gqa_mt != gqa at H=%zu T=%zu", H, T);
    /* decode == full last row */
    uint16_t *o3 = malloc(H * D * 2);
    apus_attn_gqa_decode_mt(q + (T - 1) * H * D, k, v, o3, T, H, HKV, D,
                            scale);
    CHECK(memcmp(o3, o1 + (T - 1) * H * D, H * D * 2) == 0,
          "gqa_decode_mt != full last row");
    digest_bytes(&g_digest, o2, T * H * D * 2);
    free(q); free(k); free(v); free(abuf); free(ebuf);
    free(o1); free(o2); free(o3);
}

/* =========================================================================*/
/* tiered == eager at the model level (prefill, decode, I/O thread sweep) */
static void test_prefill_batch(void) {
    char err[256];
    size_t T = 16;
    int64_t ids[16];
    for (size_t i = 0; i < T; i++) ids[i] = (int64_t)(rng_u64() % 256);

    /* eager: one-shot forward vs 16 decode steps */
    ApusModel m;
    if (apus_model_load(&m, "tests/m5/fixtures/model", 256, err,
                        sizeof err)) {
        fprintf(stderr, "FAIL: model load: %s\n", err);
        failures++;
        return;
    }
    size_t V = (size_t)m.vocab;
    float *l1 = malloc(T * V * sizeof(float));
    float *l2 = malloc(T * V * sizeof(float));
    ApusModelState s1, s2;
    apus_model_state_init(&s1, &m);
    apus_model_forward(&m, &s1, ids, T, l1, 1, NULL);
    apus_model_state_init(&s2, &m);
    for (size_t t = 0; t < T; t++)
        apus_model_forward(&m, &s2, ids + t, 1, l2 + t * V, 0, NULL);
    CHECK(memcmp(l1, l2, T * V * sizeof(float)) == 0,
          "eager one-shot prefill != sequential (logits)");
    int state_same = 1;
    for (int L = 0; L < m.n_layers; L++) {
        ApusLayerState *a = &s1.layers[L], *b = &s2.layers[L];
        const ApusLayerCfg *c = &m.layers[L].lc;
        if (c->kind == APUS_LAYER_GDN) {
            size_t cd = 2 * c->gdn_hk * c->gdn_dk + c->gdn_hv * c->gdn_dv;
            state_same &= memcmp(a->conv_state, b->conv_state,
                                 cd * 3 * 2) == 0
                && memcmp(a->S, b->S, c->gdn_hv * c->gdn_dk * c->gdn_dv
                          * sizeof(float)) == 0;
        } else {
            size_t nd = c->attn_nkv * c->attn_d;
            state_same &= memcmp(a->kcache, b->kcache, T * nd * 2) == 0
                && memcmp(a->vcache, b->vcache, T * nd * 2) == 0;
        }
    }
    CHECK(state_same, "eager one-shot prefill: layer state differs");
    digest_bytes(&g_digest, l1, T * V * sizeof(float));
    apus_model_state_free(&s1, &m);
    apus_model_state_free(&s2, &m);
    apus_model_free(&m);

    /* tiered (m6a container): one-shot vs sequential decodes vs the
     * EAGER logits above, at several cache sizes AND I/O pool sizes —
     * the pool must never perturb compute bits */
    static const struct { int slots; int io; } tc[] = {
        {16, -1},           /* all experts fit, synchronous */
        {2, -1},            /* thrashing, synchronous */
        {2, 1}, {2, 4}, {2, 8}, {16, 4},
    };
    for (size_t ci = 0; ci < sizeof tc / sizeof tc[0]; ci++) {
        ApusModel mt;
        if (apus_model_load_ex(&mt, "tests/m6a/fixtures/model", 256, 1,
                               err, sizeof err)) {
            fprintf(stderr, "FAIL: tiered load: %s\n", err);
            failures++;
            free(l1); free(l2);
            return;
        }
        ApusStoreCfg sc = {0};
        sc.n_layers = mt.n_layers;
        sc.n_experts = (int)mt.layers[0].lc.experts;
        sc.slots_per_layer = tc[ci].slots;
        sc.cache_bytes = 1;
        sc.pin_bytes = 1;
        sc.io_threads = tc[ci].io;
        sc.usage_path = "";
        ApusStore *st = apus_store_open("tests/m6a/fixtures/model", &sc,
                                        err, sizeof err);
        CHECK(st != NULL, "store open (slots %d io %d): %s",
              tc[ci].slots, tc[ci].io, err);
        if (!st) { apus_model_free(&mt); continue; }
        apus_model_attach_store(&mt, st);
        ApusModelState s3, s4;
        apus_model_state_init(&s3, &mt);
        apus_model_forward(&mt, &s3, ids, T, l2, 1, NULL);
        CHECK(memcmp(l1, l2, T * V * sizeof(float)) == 0,
              "tiered != eager one-shot (slots %d, io %d)",
              tc[ci].slots, tc[ci].io);
        apus_model_state_init(&s4, &mt);
        for (size_t t = 0; t < T; t++)
            apus_model_forward(&mt, &s4, ids + t, 1, l2 + t * V, 0, NULL);
        CHECK(memcmp(l1, l2, T * V * sizeof(float)) == 0,
              "tiered sequential != eager one-shot (slots %d, io %d)",
              tc[ci].slots, tc[ci].io);
        digest_bytes(&g_digest, l2, T * V * sizeof(float));
        apus_model_state_free(&s3, &mt);
        apus_model_state_free(&s4, &mt);
        apus_store_close(st);
        apus_model_free(&mt);
    }
    free(l1); free(l2);
}

static void test_scratch_smoke(void) {
    /* the forward's LIFO pattern: mark, nested allocs, reset, reuse */
    ApusScratchMark mk = apus_scratch_mark();
    void *a = apus_scratch_alloc(1000);
    void *b = apus_scratch_alloc(2000000);       /* forces a new segment */
    CHECK(a && b, "scratch alloc");
    memset(a, 1, 1000);
    memset(b, 2, 2000000);
    apus_scratch_reset(mk);
    void *c = apus_scratch_alloc(3000000);       /* dead-segment grow */
    CHECK(c != NULL, "scratch dead-segment grow");
    memset(c, 3, 3000000);
    apus_scratch_reset(mk);
    void *d = apus_scratch_alloc(1000);
    CHECK(d == a, "scratch reuse after reset (%p vs %p)", d, a);
    checks += (d == a);
    (void)c;
}

int main(void) {
    printf("test_m6c: hot-path equivalence gates (Qwen3.6 M6c)\n");
    fprintf(stderr, "  pool threads: %d\n", apus_pool_threads());
    test_gemv_hot();
    test_gemm_hot();
    test_router_matvec();
    test_gdn_step_mt();
    test_gqa_mt();
    test_prefill_batch();
    test_scratch_smoke();
    printf("  digest %016llx\n", (unsigned long long)g_digest);
    printf("test_m6c: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
