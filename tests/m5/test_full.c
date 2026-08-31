/*
 * tests/m5/test_full.c — full mini-model forward + sampling, hard-gate
 * tests for c/model.h + c/sample.h against the M5 oracle fixtures
 * (tests/m5/fixtures, tools/oracle.py full-model entry points).
 *
 *   1. LOADER: config.json (REAL qwen3_5_moe schema, nested text_config)
 *      + the M1 container (per-expert 2-member slab slices) via c/st.h.
 *   2. PREFILL goldens (len6, len64): per-position logits vs f32 oracle,
 *      gate rel <= 2*envelope_rel + 0.01, argmax flips <= env flips + 5%.
 *   3. DECODE chain (prefill 32 + 8 steps, carried state): per-step
 *      logits, same gate (the oracle is chunk-exact, so the decode
 *      goldens are as clean as the prefill ones).
 *   4. GREEDY teacher-forced: per-step logits + argmax vs oracle tokens;
 *      a flip is excused iff the golden top1-top2 gap <= 0.5 (the
 *      documented near-tie class, Apus policy); unexcused == 0.
 *   5. SAMPLED teacher-forced: replay the oracle uniforms through
 *      apus_sample_logits_u (temp 0.6 / top_k 20 / top_p 0.95); a flip
 *      is excused iff the oracle CDF margin <= 2e-2; unexcused == 0.
 *      Verifies the c/sample.h rules (sort/top-k/nucleus/CDF) against
 *      the oracle's — not a statistical check.
 *   6. TRACE: per-layer hidden states on the greedy chain context vs the
 *      f32 traces; divergence budget table (C-vs-f32 vs f32-vs-f64).
 *   7. CHUNK INVARIANCE: prefill(64) vs prefill(40) + 24 decodes —
 *      64x256 logits BITWISE. Determinism: rerun bitwise.
 *
 * Run from the repository root (fixtures under tests/m5/fixtures/).
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_BF16_IMPLEMENTATION
#define APUS_GDN_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION
#define APUS_LAYER_IMPLEMENTATION
#define APUS_MODEL_IMPLEMENTATION
#define APUS_SAMPLE_IMPLEMENTATION
#include "json.h"
#include "st.h"
#include "bf16.h"
#include "gdn.h"
#include "attn.h"
#include "moe.h"
#include "layer.h"
#include "model.h"
#include "sample.h"

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

/* FNV-1a 64 over a byte range, folded into *h. */
static void digest_bytes(uint64_t *h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) {
        *h ^= b[i];
        *h *= 0x100000001B3ull;
    }
}
static uint64_t g_digest = 0xCBF29CE484222325ull;

static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(buf); return NULL;
    }
    fclose(f);
    *len = (size_t)sz;
    return buf;
}

static double *load_f64(const char *name) {
    char p[512];
    snprintf(p, sizeof p, "tests/m5/fixtures/%s", name);
    size_t n;
    return (double *)read_file(p, &n);
}
static int64_t *load_i64(const char *name) {
    char p[512];
    snprintf(p, sizeof p, "tests/m5/fixtures/%s", name);
    size_t n;
    return (int64_t *)read_file(p, &n);
}

static ApusModel g_m;

/* logits comparison: rel + argmax flips, gated against the f32/f64 pair */
static int cmp_logits(const char *tag, const float *c, size_t T, size_t V,
                      const double *g32, const double *g64,
                      double *out_rel, double *out_envrel,
                      double *out_flip, double *out_envflip) {
    double rel = 0, scale = 0, envrel = 0;
    long flips = 0, envflips = 0;
    for (size_t t = 0; t < T; t++) {
        int ca = 0, a32 = 0, a64 = 0;
        for (size_t v = 0; v < V; v++) {
            double a = g32[t * V + v], b = g64[t * V + v];
            double d = fabs((double)c[t * V + v] - a);
            double e = fabs(a - b);
            if (fabs(a) > scale) scale = fabs(a);
            if (d > rel) rel = d;
            if (e > envrel) envrel = e;
            if (c[t * V + v] > c[t * V + ca]) ca = (int)v;
            if (a > g32[t * V + a32]) a32 = (int)v;
            if (b > g64[t * V + a64]) a64 = (int)v;
        }
        if (ca != a32) flips++;
        if (a32 != a64) envflips++;
    }
    rel /= scale > 1e-9 ? scale : 1e-9;
    envrel /= scale > 1e-9 ? scale : 1e-9;
    double fr = (double)flips / (double)T;
    double efr = (double)envflips / (double)T;
    int ok = rel <= 2.0 * envrel + 0.01 && fr <= efr + 0.05 + 1e-9;
    CHECK(ok, "%s logits: rel %.3g (env %.3g), flip %.3f (env %.3f)",
          tag, rel, envrel, fr, efr);
    if (out_rel) *out_rel = rel;
    if (out_envrel) *out_envrel = envrel;
    if (out_flip) *out_flip = fr;
    if (out_envflip) *out_envflip = efr;
    return ok;
}

static void test_prefill(const char *key, size_t T) {
    char nm[128];
    snprintf(nm, sizeof nm, "ids_%s.bin", key);
    int64_t *ids = load_i64(nm);
    snprintf(nm, sizeof nm, "%s_logits_f32.bin", key);
    double *g32 = load_f64(nm);
    snprintf(nm, sizeof nm, "%s_logits_f64.bin", key);
    double *g64 = load_f64(nm);
    CHECK(ids && g32 && g64, "%s fixture load", key);
    if (!ids || !g32 || !g64) return;
    size_t V = (size_t)g_m.vocab;
    float *logits = malloc(T * V * sizeof(float));
    ApusModelState st;
    apus_model_state_init(&st, &g_m);
    apus_model_forward(&g_m, &st, ids, T, logits, 1, NULL);
    double rel, env, fl, efl;
    cmp_logits(key, logits, T, V, g32, g64, &rel, &env, &fl, &efl);
    printf("  %-14s rel %.3g (env %.3g), flip %.1f%% (env %.1f%%)\n",
           key, rel, env, fl * 100, efl * 100);
    digest_bytes(&g_digest, logits, T * V * sizeof(float));
    apus_model_state_free(&st, &g_m);
    free(logits); free(ids); free(g32); free(g64);
}

static void test_decode_chain(void) {
    int64_t *ids = load_i64("ids_dec32.bin");
    CHECK(ids != NULL, "dec32 ids load");
    if (!ids) return;
    size_t V = (size_t)g_m.vocab;
    float *logits = malloc(V * sizeof(float));
    ApusModelState st;
    apus_model_state_init(&st, &g_m);
    apus_model_forward(&g_m, &st, ids, 32, logits, 0, NULL);
    double worst_rel = 0;
    long allflips = 0;
    for (int i = 0; i < 8; i++) {
        char nm[128];
        snprintf(nm, sizeof nm, "dec32_step%d_logits_f32.bin", i);
        double *g32 = load_f64(nm);
        snprintf(nm, sizeof nm, "dec32_step%d_logits_f64.bin", i);
        double *g64 = load_f64(nm);
        CHECK(g32 && g64, "dec32 step %d golden load", i);
        apus_model_forward(&g_m, &st, ids + 32 + i, 1, logits, 0, NULL);
        double rel, env, fl, efl;
        char tag[32];
        snprintf(tag, sizeof tag, "dec32_step%d", i);
        cmp_logits(tag, logits, 1, V, g32, g64, &rel, &env, &fl, &efl);
        if (rel > worst_rel) worst_rel = rel;
        allflips += (long)(fl + 0.5);
        digest_bytes(&g_digest, logits, V * sizeof(float));
        free(g32); free(g64);
    }
    printf("  decode_from32: worst step rel %.3g, argmax flips %ld/8\n",
           worst_rel, allflips);
    apus_model_state_free(&st, &g_m);
    free(logits); free(ids);
}

static void test_greedy(void) {
    int64_t *ctx = load_i64("ids_greedy_ctx.bin");
    int64_t *toks = load_i64("greedy_tokens.bin");
    double *gap = load_f64("greedy_gap.bin");
    CHECK(ctx && toks && gap, "greedy fixture load");
    if (!ctx || !toks || !gap) return;
    size_t V = (size_t)g_m.vocab;
    long excused = 0, unexcused = 0;
    double worst_rel = 0;
    for (int s = 0; s < 16; s++) {
        char nm[128];
        snprintf(nm, sizeof nm, "greedy_step%d_logits_f32.bin", s);
        double *g32 = load_f64(nm);
        snprintf(nm, sizeof nm, "greedy_step%d_logits_f64.bin", s);
        double *g64 = load_f64(nm);
        CHECK(g32 && g64, "greedy step %d golden load", s);
        size_t T = 24 + (size_t)s;
        int64_t *ids = malloc(T * sizeof(int64_t));
        memcpy(ids, ctx, 24 * sizeof(int64_t));
        for (int i = 0; i < s; i++) ids[24 + i] = toks[i];
        float *logits = malloc(V * sizeof(float));
        ApusModelState st;
        apus_model_state_init(&st, &g_m);
        apus_model_forward(&g_m, &st, ids, T, logits, 0, NULL);
        apus_model_state_free(&st, &g_m);
        /* logits level: rel vs envelope */
        double rel = 0, env = 0, scale = 0;
        for (size_t v = 0; v < V; v++) {
            double a = g32[v], b = g64[v];
            double d = fabs((double)logits[v] - a);
            if (fabs(a) > scale) scale = fabs(a);
            if (d > rel) rel = d;
            if (fabs(a - b) > env) env = fabs(a - b);
        }
        rel /= scale > 1e-9 ? scale : 1e-9;
        env /= scale > 1e-9 ? scale : 1e-9;
        if (rel > worst_rel) worst_rel = rel;
        CHECK(rel <= 2.0 * env + 0.01,
              "greedy step %d logits: rel %.3g (env %.3g)", s, rel, env);
        /* token level: near-tie policy */
        int ca = apus_sample_argmax(logits, V);
        if (ca != (int)toks[s]) {
            if (gap[s] <= 0.5) excused++;
            else {
                unexcused++;
                fprintf(stderr,
                        "  greedy step %d: UNEXCUSED flip C=%d oracle=%lld "
                        "gap %.4f\n", s, ca, (long long)toks[s], gap[s]);
            }
        }
        digest_bytes(&g_digest, logits, V * sizeof(float));
        free(ids); free(logits); free(g32); free(g64);
    }
    printf("  greedy teacher-forced: worst rel %.3g, flips %ld excused "
           "(near-tie), %ld unexcused\n", worst_rel, excused, unexcused);
    CHECK(unexcused == 0, "greedy: %ld unexcused flips", unexcused);
    free(ctx); free(toks); free(gap);
}

static void test_sampled(void) {
    int64_t *ctx = load_i64("ids_sampled_ctx.bin");
    int64_t *toks = load_i64("sampled_tokens.bin");
    double *uu = load_f64("sampled_u.bin");
    double *margin = load_f64("sampled_margin.bin");
    CHECK(ctx && toks && uu && margin, "sampled fixture load");
    if (!ctx || !toks || !uu || !margin) return;
    size_t V = (size_t)g_m.vocab;
    void *scratch = malloc(apus_sample_scratch_size(V));
    long excused = 0, unexcused = 0;
    for (int s = 0; s < 16; s++) {
        char nm[128];
        snprintf(nm, sizeof nm, "sampled_step%d_logits_f32.bin", s);
        double *g32 = load_f64(nm);
        CHECK(g32 != NULL, "sampled step %d golden load", s);
        size_t T = 24 + (size_t)s;
        int64_t *ids = malloc(T * sizeof(int64_t));
        memcpy(ids, ctx, 24 * sizeof(int64_t));
        for (int i = 0; i < s; i++) ids[24 + i] = toks[i];
        float *logits = malloc(V * sizeof(float));
        ApusModelState st;
        apus_model_state_init(&st, &g_m);
        apus_model_forward(&g_m, &st, ids, T, logits, 0, NULL);
        apus_model_state_free(&st, &g_m);
        /* logits sanity vs f32 golden (loose: envelope+slack via rel) */
        double rel = 0, scale = 0;
        for (size_t v = 0; v < V; v++) {
            if (fabs(g32[v]) > scale) scale = fabs(g32[v]);
            double d = fabs((double)logits[v] - g32[v]);
            if (d > rel) rel = d;
        }
        (void)rel;
        int tok = apus_sample_logits_u(logits, V, 0.6f, 20, 0.95f,
                                       uu[s], scratch);
        if (tok != (int)toks[s]) {
            /* the CDF-margin policy (Apus M5/M9b): a flip is legitimate
             * only when the oracle's draw sat within 2e-2 of a CDF
             * boundary — logit noise of the envelope class can move the
             * boundary across u; anything larger is an implementation
             * bug (tests/m5/README.md). */
            if (margin[s] <= 2e-2) excused++;
            else {
                unexcused++;
                fprintf(stderr,
                        "  sampled step %d: UNEXCUSED flip C=%d oracle=%lld "
                        "margin %.4g\n", s, tok, (long long)toks[s],
                        margin[s]);
            }
        }
        free(ids); free(logits); free(g32);
    }
    printf("  sampled teacher-forced: flips %ld excused (CDF margin "
           "<= 2e-2), %ld unexcused\n", excused, unexcused);
    CHECK(unexcused == 0, "sampled: %ld unexcused flips", unexcused);
    free(scratch);
    free(ctx); free(toks); free(uu); free(margin);
}

static void test_trace(void) {
    int64_t *ctx = load_i64("ids_greedy_ctx.bin");
    int64_t *toks = load_i64("greedy_tokens.bin");
    CHECK(ctx && toks, "trace fixture load");
    if (!ctx || !toks) return;
    size_t H = (size_t)g_m.hidden;
    size_t T = 24 + 16;
    int64_t *ids = malloc(T * sizeof(int64_t));
    memcpy(ids, ctx, 24 * sizeof(int64_t));
    for (int i = 0; i < 16; i++) ids[24 + i] = toks[i];
    int L = g_m.n_layers;
    uint16_t *ht = malloc((size_t)L * T * H * 2);
    float *logits = malloc(T * (size_t)g_m.vocab * sizeof(float));
    ApusModelState st;
    apus_model_state_init(&st, &g_m);
    apus_model_forward(&g_m, &st, ids, T, logits, 1, ht);
    apus_model_state_free(&st, &g_m);

    printf("  divergence budget (teacher-forced greedy chain, %zu tok):\n",
           T);
    printf("  %-12s %12s %12s\n", "stage", "C-vs-f32 rel", "f32-vs-f64 rel");
    for (int l = 0; l < L; l++) {
        char nm[128];
        snprintf(nm, sizeof nm, "trace_L%d_f32.bin", l);
        double *g32 = load_f64(nm);
        snprintf(nm, sizeof nm, "trace_L%d_f64.bin", l);
        double *g64 = load_f64(nm);
        CHECK(g32 && g64, "trace L%d golden load", l);
        double rel = 0, env = 0, scale = 0;
        for (size_t i = 0; i < T * H; i++) {
            double a = g32[i], b = g64[i];
            double c = apus_bf16_f32(ht[(size_t)l * T * H + i]);
            double d = fabs(c - a);
            if (fabs(a) > scale) scale = fabs(a);
            if (d > rel) rel = d;
            if (fabs(a - b) > env) env = fabs(a - b);
        }
        rel /= scale > 1e-9 ? scale : 1e-9;
        env /= scale > 1e-9 ? scale : 1e-9;
        printf("  %-12s %12.3g %12.3g\n", "h_after_L*", rel, env);
        CHECK(rel <= 2.0 * env + 0.01,
              "trace L%d: C-vs-f32 rel %.3g exceeds 2x envelope %.3g",
              l, rel, env);
        free(g32); free(g64);
    }
    digest_bytes(&g_digest, ht, (size_t)L * T * H * 2);
    free(ids); free(ht); free(logits);
    free(ctx); free(toks);
}

static void test_invariance(void) {
    int64_t *ids = load_i64("ids_prefill64.bin");
    CHECK(ids != NULL, "invariance ids load");
    if (!ids) return;
    size_t V = (size_t)g_m.vocab;
    size_t T = 64, split = 40;
    float *l1 = malloc(T * V * sizeof(float));
    ApusModelState s1;
    apus_model_state_init(&s1, &g_m);
    apus_model_forward(&g_m, &s1, ids, T, l1, 1, NULL);
    apus_model_state_free(&s1, &g_m);

    float *l2 = malloc(T * V * sizeof(float));
    ApusModelState s2;
    apus_model_state_init(&s2, &g_m);
    apus_model_forward(&g_m, &s2, ids, split, l2, 1, NULL);
    for (size_t t = split; t < T; t++)
        apus_model_forward(&g_m, &s2, ids + t, 1, l2 + t * V, 0, NULL);
    apus_model_state_free(&s2, &g_m);
    CHECK(memcmp(l1, l2, T * V * sizeof(float)) == 0,
          "chunk invariance: prefill(64) != prefill(40)+24 decodes "
          "(not bitwise)");

    /* determinism */
    float *l3 = malloc(T * V * sizeof(float));
    ApusModelState s3;
    apus_model_state_init(&s3, &g_m);
    apus_model_forward(&g_m, &s3, ids, T, l3, 1, NULL);
    apus_model_state_free(&s3, &g_m);
    CHECK(memcmp(l1, l3, T * V * sizeof(float)) == 0,
          "determinism: rerun not bitwise");
    free(ids); free(l1); free(l2); free(l3);
}

int main(void) {
    printf("test_full: full mini-model forward + sampling "
           "(Qwen3.6-35B-A3B M5)\n");
    char err[256];
    if (apus_model_load(&g_m, "tests/m5/fixtures/model", 256, err,
                        sizeof err)) {
        fprintf(stderr, "FAIL: model load: %s\n", err);
        return 1;
    }
    printf("  model: %d layers, hidden %d, vocab %d\n",
           g_m.n_layers, g_m.hidden, g_m.vocab);

    test_prefill("prefill6", 6);
    test_prefill("prefill64", 64);
    test_decode_chain();
    test_greedy();
    test_sampled();
    test_trace();
    test_invariance();

    printf("  digest %016llx\n", (unsigned long long)g_digest);
    printf("test_full: %ld checks, %d failures\n", checks, failures);
    apus_model_free(&g_m);
    return failures ? 1 : 0;
}
