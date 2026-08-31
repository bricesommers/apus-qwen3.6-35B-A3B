/*
 * tests/m10/test_model.c — model-level gate for the M10 Metal backend.
 *
 * THE GATE: the Metal-hooked forward is BITWISE identical to the pure
 * CPU forward — every hooked op is bitwise (tests/m10/test_kernels.c),
 * so the whole model must be bitwise, not merely token-identical:
 *
 *   A. EAGER (tests/m5 fixture): apus_model_forward hot at T=64 with
 *      hooks filled == the CPU forward — logits [64,V] BITWISE, and the
 *      per-layer state (conv/S/caches) BITWISE.
 *   B. GREEDY: 24 free-running greedy tokens from the m5 CLI ids, CPU
 *      vs Metal — token stream IDENTICAL (this is the load-bearing
 *      gate; A already proves it by bits).
 *   C. DETERMINISM + CHUNK INVARIANCE with hooks on: repeated prefill
 *      bitwise; prefill(64) == prefill(32) + 32 single decodes bitwise
 *      (Metal through both the batched-GEMM hook and the decode-GEMV
 *      hook).
 *   D. TIERED SLAB SAFETY (tests/m6a fixture, cache_bytes=1 — heavy
 *      eviction, transient slabs): Metal on, T=64 forward == CPU
 *      forward BITWISE. Expert slabs go through the no-hook _cpu
 *      variants (c/cache.h), so the pointer-keyed cache only ever sees
 *      resident weights; this run proves the differentiation end to end
 *      (any stale-buffer aliasing would flip these bits).
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

static uint64_t g_digest = 0xCBF29CE484222325ull;
static void digest_bytes(const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) {
        g_digest ^= b[i];
        g_digest *= 0x100000001B3ull;
    }
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* forward the m5 fixture model: returns 0 on success; logits out [T,V] */
static int fwd_m5(const char *dir, const int64_t *ids, size_t T,
                  float *logits, int tiered_store, char *err,
                  size_t errcap) {
    ApusModel m;
    if (tiered_store) {
        if (apus_model_load_ex(&m, dir, 256, 1, err, errcap)) return -1;
        ApusStoreCfg sc = {0};
        sc.n_layers = m.n_layers;
        sc.n_experts = m.layers[0].lc.experts;         /* the m6a fixture: 16 */
        sc.slots_per_layer = m.layers[0].lc.experts;
        sc.cache_bytes = 1;       /* force eviction: transient slabs */
        sc.pin_bytes = 1;
        sc.io_threads = -1;
        sc.usage_path = "";
        ApusStore *st = apus_store_open(dir, &sc, err, errcap);
        if (!st) { apus_model_free(&m); return -1; }
        apus_model_attach_store(&m, st);
        ApusModelState s;
        apus_model_state_init(&s, &m);
        apus_model_forward(&m, &s, ids, T, logits, 1, NULL);
        /* layer state digest (slab-safety: state must match too) */
        for (int L = 0; L < m.n_layers; L++) {
            ApusLayerState *a = &s.layers[L];
            const ApusLayerCfg *c = &m.layers[L].lc;
            if (c->kind == APUS_LAYER_GDN) {
                size_t cd = 2 * c->gdn_hk * c->gdn_dk
                            + c->gdn_hv * c->gdn_dv;
                digest_bytes(a->conv_state, cd * 3 * 2);
                digest_bytes(a->S, c->gdn_hv * c->gdn_dk * c->gdn_dv
                             * sizeof(float));
            } else {
                size_t nd = c->attn_nkv * c->attn_d;
                digest_bytes(a->kcache, T * nd * 2);
                digest_bytes(a->vcache, T * nd * 2);
            }
        }
        apus_model_state_free(&s, &m);
        apus_store_close(st);
        apus_model_free(&m);
        return 0;
    }
    if (apus_model_load(&m, dir, 256, err, errcap)) return -1;
    ApusModelState s;
    apus_model_state_init(&s, &m);
    apus_model_forward(&m, &s, ids, T, logits, 1, NULL);
    apus_model_state_free(&s, &m);
    apus_model_free(&m);
    return 0;
}

/* greedy free-running decode from the m5 CLI ids, N tokens; fills toks */
static void greedy_run(const char *dir, const int64_t *ids0, size_t n0,
                       int N, int64_t *toks, char *err, size_t errcap) {
    ApusModel m;
    if (apus_model_load(&m, dir, 256, err, errcap)) {
        fprintf(stderr, "FAIL: load: %s\n", err);
        failures++;
        return;
    }
    size_t V = (size_t)m.vocab;
    float *logits = malloc(V * sizeof(float));
    int64_t ids[64];
    memcpy(ids, ids0, n0 * sizeof(int64_t));
    ApusModelState s;
    apus_model_state_init(&s, &m);
    for (int t = 0; t < N; t++) {
        apus_model_forward(&m, &s, ids, n0, logits, 0, NULL);
        size_t best = 0;
        for (size_t v = 1; v < V; v++)
            if (logits[v] > logits[best]) best = v;
        toks[t] = (int64_t)best;
        ids[0] = best;
        n0 = 1;
    }
    apus_model_state_free(&s, &m);
    apus_model_free(&m);
    free(logits);
}

int main(void) {
    printf("test_model: Metal backend model-level gate\n");
    char err[256];
    size_t T = 64;
    int64_t ids[64];
    for (size_t i = 0; i < T; i++) ids[i] = (int64_t)(rng_u64() % 256);
    size_t V = 256;             /* the m5/m6a fixture vocab (config.json) */
    float *l_cpu = malloc(T * V * sizeof(float));
    float *l_mtl = malloc(T * V * sizeof(float));

    /* A. eager: CPU forward, then Metal-hooked forward — BITWISE */
    if (fwd_m5("tests/m5/fixtures/model", ids, T, l_cpu, 0, err,
               sizeof err)) {
        fprintf(stderr, "FAIL: load: %s\n", err);
        failures++;
        return 1;
    }
    uint64_t d_cpu = g_digest;
    CHECK(apus_metal_enable(err, sizeof err) == 0, "enable: %s", err);
    CHECK(apus_metal_is_enabled(), "backend not enabled");
    g_digest = 0xCBF29CE484222325ull;
    if (fwd_m5("tests/m5/fixtures/model", ids, T, l_mtl, 0, err,
               sizeof err)) {
        fprintf(stderr, "FAIL: load: %s\n", err);
        failures++;
        return 1;
    }
    CHECK(memcmp(l_cpu, l_mtl, T * V * sizeof(float)) == 0,
          "A: eager T=64 logits: Metal-hooked != CPU");
    printf("  A eager T=64: logits BITWISE (hooked == CPU)\n");
    digest_bytes(l_mtl, T * V * sizeof(float));
    (void)d_cpu;

    /* B. greedy stream: CPU vs Metal identical. NOTE: each greedy_run
     * reloads the model, so the backend cache is refreshed between runs
     * (the stable-pointer invariant, c/backend_metal.h). */
    {
        static const int64_t ids0[8] = { 3, 1, 4, 1, 5, 9, 2, 6 };
        int64_t tc[24], tm[24];
        apus_metal_disable();
        greedy_run("tests/m5/fixtures/model", ids0, 8, 24, tc, err,
                   sizeof err);
        apus_metal_disable();
        CHECK(apus_metal_enable(err, sizeof err) == 0,
              "re-enable: %s", err);
        greedy_run("tests/m5/fixtures/model", ids0, 8, 24, tm, err,
                   sizeof err);
        CHECK(memcmp(tc, tm, sizeof tc) == 0,
              "B: greedy stream CPU != Metal");
        printf("  B greedy 24: stream IDENTICAL\n");
        digest_bytes(tm, sizeof tm);
    }

    /* C. determinism + chunk invariance with hooks on (fresh cache per
     * reload, same invariant) */
    {
        float *l2 = malloc(T * V * sizeof(float));
        apus_metal_disable();
        CHECK(apus_metal_enable(err, sizeof err) == 0,
              "re-enable: %s", err);
        fwd_m5("tests/m5/fixtures/model", ids, T, l2, 0, err, sizeof err);
        CHECK(memcmp(l_mtl, l2, T * V * sizeof(float)) == 0,
              "C: repeated Metal prefill not deterministic");
        /* chunk: 32 prefill + 32 single decodes, hooks on (fresh cache
         * for this reload too — the stable-pointer invariant) */
        apus_metal_disable();
        CHECK(apus_metal_enable(err, sizeof err) == 0,
              "re-enable: %s", err);
        ApusModel m;
        if (apus_model_load(&m, "tests/m5/fixtures/model", 256, err,
                            sizeof err) == 0) {
            ApusModelState s;
            apus_model_state_init(&s, &m);
            float *lc = malloc(T * V * sizeof(float));
            apus_model_forward(&m, &s, ids, 32, lc, 1, NULL);
            for (size_t t = 32; t < T; t++)
                apus_model_forward(&m, &s, ids + t, 1, lc + t * V, 0, NULL);
            CHECK(memcmp(l_mtl, lc, T * V * sizeof(float)) == 0,
                  "C: Metal chunk invariance (32+1s != 64)");
            printf("  C determinism + chunk invariance: BITWISE\n");
            digest_bytes(lc, T * V * sizeof(float));
            free(lc);
            apus_model_state_free(&s, &m);
            apus_model_free(&m);
        }
        free(l2);
    }

    /* D. tiered slab safety: heavy eviction, Metal on — BITWISE vs CPU */
    {
        apus_metal_disable();
        g_digest = 0xCBF29CE484222325ull;
        if (fwd_m5("tests/m6a/fixtures/model", ids, T, l_cpu, 1, err,
                   sizeof err)) {
            fprintf(stderr, "FAIL: tiered load: %s\n", err);
            failures++;
            return 1;
        }
        uint64_t sd_cpu = g_digest;
        CHECK(apus_metal_enable(err, sizeof err) == 0,
              "re-enable: %s", err);
        g_digest = 0xCBF29CE484222325ull;
        if (fwd_m5("tests/m6a/fixtures/model", ids, T, l_mtl, 1, err,
                   sizeof err)) {
            fprintf(stderr, "FAIL: tiered load: %s\n", err);
            failures++;
            return 1;
        }
        CHECK(memcmp(l_cpu, l_mtl, T * V * sizeof(float)) == 0,
              "D: tiered T=64 logits: Metal != CPU (slab safety)");
        CHECK(g_digest == sd_cpu,
              "D: tiered layer-state digests differ (slab safety)");
        printf("  D tiered slab safety: logits+state BITWISE "
               "(state %016llx)\n", (unsigned long long)sd_cpu);
        digest_bytes(l_mtl, T * V * sizeof(float));
    }

    printf("  metal: %llu B wrapped, %llu B uploaded, %llu dispatches\n",
           (unsigned long long)apus_metal_bytes_wrapped(),
           (unsigned long long)apus_metal_bytes_uploaded(),
           (unsigned long long)apus_metal_dispatches());
    printf("digest %016llx\n", (unsigned long long)g_digest);
    printf("test_model: %ld checks, %d failures\n", checks, failures);
    apus_metal_disable();
    free(l_cpu); free(l_mtl);
    return failures ? 1 : 0;
}
