/*
 * tests/m8/test_mtp.c — MTP speculative decoding hard gates (Qwen M8).
 *
 *   1. mtp_prefill golden: C apus_mtp_forward vs the oracle f32 golden
 *      (envelope + near-tie argmax-flip policy, the m5 rule).
 *   2. mtp_chain golden: 3-step draft chain vs oracle drafts.
 *   3. EQUIVALENCE (the hard gate): spec K=2/3/4 vs non-spec, greedy and
 *      sampled (temp 0.6, top_k 20, top_p 0.95, seed fixed) — 24
 *      emitted tokens BITWISE.
 *   4. ROLLBACK: model-state digest after each spec run == state after
 *      decoding exactly the emitted tokens non-speculatively, BITWISE.
 *   5. Forced draft patterns (draft_override): truth (100% accept incl.
 *      bonus path), garbage (0%), mixed (partial) — streams + digests
 *      still bitwise, stats match the pattern.
 *   6. Tiered spec (store serves the mtp layer-2 slabs through the
 *      mtp.layers.0.* prefix mapping) == eager, bitwise.
 *
 * The FNV digest over everything is diffed across APUS_THREADS=1/4/8 by
 * the Makefile. Run from the repository root.
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
#define APUS_SAMPLE_IMPLEMENTATION
#define APUS_MTP_IMPLEMENTATION
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
#include "sample.h"
#include "mtp.h"

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

static void digest_bytes(uint64_t *h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) {
        *h ^= b[i];
        *h *= 0x100000001B3ull;
    }
}
static uint64_t g_digest = 0xCBF29CE484222325ull;

#define MODEL "tests/m8/fixtures/model"
#define PROMPT_LEN 8
#define STEPS 24
#define VOCAB 256

static const int64_t PROMPT[PROMPT_LEN] = { 3, 1, 4, 1, 5, 9, 2, 6 };

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

static ApusModel g_m;
static ApusMtpW g_mw;

/* ---- goldens --------------------------------------------------------------*/

static void test_mtp_golden(void) {
    size_t n;
    uint16_t *h = (uint16_t *)read_file("tests/m8/fixtures/mtp_pre_h.bin",
                                        &n);
    int64_t *ids = (int64_t *)read_file(
        "tests/m8/fixtures/mtp_pre_ids.bin", &n);
    double *g32 = (double *)read_file(
        "tests/m8/fixtures/mtp_pre_logits_f32.bin", &n);
    double *g64 = (double *)read_file(
        "tests/m8/fixtures/mtp_pre_logits_f64.bin", &n);
    double *gap = (double *)read_file(
        "tests/m8/fixtures/mtp_pre_gap.bin", &n);
    CHECK(h && ids && g32 && g64 && gap, "mtp_pre fixture load");
    if (!h || !ids || !g32 || !g64 || !gap) return;
    size_t T = 12, V = VOCAB, H = 128;
    ApusLayerState mst;
    apus_mtp_state_init(&g_m, &g_mw, &mst);
    float *logits = malloc(T * V * sizeof(float));
    uint16_t *hout = malloc(T * H * 2);
    apus_mtp_forward(&g_m, &g_mw, &mst, NULL, h, ids, T, logits, hout);
    apus_mtp_state_free(&mst);
    double rel = 0, env = 0, scale = 0;
    long flips = 0, unexcused = 0;
    for (size_t t = 0; t < T; t++) {
        int ca = 0, ga = 0;
        for (size_t v = 0; v < V; v++) {
            double a = g32[t * V + v], b = g64[t * V + v];
            double d = fabs((double)logits[t * V + v] - a);
            if (fabs(a) > scale) scale = fabs(a);
            if (d > rel) rel = d;
            if (fabs(a - b) > env) env = fabs(a - b);
            if (logits[t * V + v] > logits[t * V + ca]) ca = (int)v;
            if (a > g32[t * V + ga]) ga = (int)v;
        }
        if (ca != ga) {
            flips++;
            if (gap[t] > 0.5) unexcused++;
        }
    }
    rel /= scale;
    env /= scale;
    printf("  mtp_prefill: rel %.3g (env %.3g), argmax flips %ld/12 "
           "(%ld unexcused)\n", rel, env, flips, unexcused);
    CHECK(rel <= 2.0 * env + 0.01, "mtp_prefill rel outside envelope");
    CHECK(unexcused == 0, "mtp_prefill: %ld unexcused argmax flips",
          unexcused);
    digest_bytes(&g_digest, logits, T * V * sizeof(float));
    free(h); free(ids); free(g32); free(g64); free(gap);
    free(logits); free(hout);

    /* chain golden */
    uint16_t *h0 = (uint16_t *)read_file(
        "tests/m8/fixtures/mtp_chain_h0.bin", &n);
    int64_t *e0 = (int64_t *)read_file(
        "tests/m8/fixtures/mtp_chain_emb0.bin", &n);
    int64_t *gd = (int64_t *)read_file(
        "tests/m8/fixtures/mtp_chain_drafts.bin", &n);
    CHECK(h0 && e0 && gd, "mtp_chain fixture load");
    if (!h0 || !e0 || !gd) return;
    ApusLayerState mst2;
    apus_mtp_state_init(&g_m, &g_mw, &mst2);
    float *lg = malloc(V * sizeof(float));
    uint16_t *ch = malloc(H * 2);
    uint16_t hh[H];
    memcpy(hh, h0, H * 2);
    int64_t eid = e0[0];
    int32_t drafts[3];
    long dflips = 0, dunexc = 0;
    for (int d = 0; d < 3; d++) {
        apus_mtp_forward(&g_m, &g_mw, &mst2, NULL, hh, &eid, 1, lg, ch);
        drafts[d] = apus_sample_argmax(lg, V);
        char nm[128];
        snprintf(nm, sizeof nm,
                 "tests/m8/fixtures/mtp_chain_step%d_logits.bin", d);
        double *gl = (double *)read_file(nm, &n);
        snprintf(nm, sizeof nm,
                 "tests/m8/fixtures/mtp_chain_step%d_gap.bin", d);
        double *gp = (double *)read_file(nm, &n);
        if (drafts[d] != (int32_t)gd[d]) {
            dflips++;
            /* near-tie policy on the oracle's gap */
            double rel2 = 0, sc2 = 0;
            int oa = 0, ca2 = 0;
            for (size_t v = 0; v < V; v++) {
                if (fabs(gl[v]) > sc2) sc2 = fabs(gl[v]);
                double dd = fabs((double)lg[v] - gl[v]);
                if (dd > rel2) rel2 = dd;
                if (gl[v] > gl[oa]) oa = (int)v;
                if (lg[v] > lg[ca2]) ca2 = (int)v;
            }
            if (gp[0] > 0.5 || oa != (int)gd[d]) dunexc++;
            fprintf(stderr, "  chain step %d draft flip C=%d oracle=%lld "
                    "(gap %.3g, C-f32 rel %.3g)\n", d, drafts[d],
                    (long long)gd[d], gp[0], rel2 / sc2);
        }
        digest_bytes(&g_digest, lg, V * sizeof(float));
        free(gl); free(gp);
        memcpy(hh, ch, H * 2);
        eid = drafts[d];
    }
    apus_mtp_state_free(&mst2);
    printf("  mtp_chain: drafts [%d %d %d], flips %ld (%ld unexcused)\n",
           drafts[0], drafts[1], drafts[2], dflips, dunexc);
    CHECK(dunexc == 0, "mtp_chain: %ld unexcused draft flips", dunexc);
    free(h0); free(e0); free(gd); free(lg); free(ch);
}

/* ---- equivalence + rollback ------------------------------------------------*/

/* non-spec decode of the prompt, max_tokens steps; returns stream */
static void run_plain(int max_tokens, float temp, uint64_t seed,
                      int *tokens) {
    ApusModelState st;
    apus_model_state_init(&st, &g_m);
    size_t V = (size_t)g_m.vocab;
    float *lg = malloc(V * sizeof(float));
    void *scratch = malloc(apus_sample_scratch_size(V));
    ApusRng rng;
    apus_rng_seed(&rng, seed);
    apus_model_forward(&g_m, &st, PROMPT, PROMPT_LEN, lg, 0, NULL);
    for (int s = 0; s < max_tokens; s++) {
        int t = apus_sample(lg, V, temp, 20, 0.95f, &rng, scratch);
        tokens[s] = t;
        int64_t id = t;
        apus_model_forward(&g_m, &st, &id, 1, lg, 0, NULL);
    }
    apus_model_state_free(&st, &g_m);
    free(lg); free(scratch);
}

/* model-state digest (main layers: pos, GDN conv+S, GQA cache live
 * rows) */
static uint64_t state_digest(ApusModelState *st) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (int L = 0; L < g_m.n_layers; L++) {
        ApusLayerState *s = &st->layers[L];
        const ApusLayerCfg *c = &g_m.layers[L].lc;
        digest_bytes(&h, &s->pos, sizeof s->pos);
        if (c->kind == APUS_LAYER_GDN) {
            size_t cd = 2 * c->gdn_hk * c->gdn_dk + c->gdn_hv * c->gdn_dv;
            digest_bytes(&h, s->conv_state, cd * 3 * 2);
            digest_bytes(&h, s->S,
                         c->gdn_hv * c->gdn_dk * c->gdn_dv
                         * sizeof(float));
        } else {
            size_t nd = c->attn_nkv * c->attn_d;
            digest_bytes(&h, s->kcache, s->pos * nd * 2);
            digest_bytes(&h, s->vcache, s->pos * nd * 2);
        }
    }
    return h;
}

static const int *g_truth;      /* override ctx: the plain stream */
static int ov_mode;             /* 0 truth, 1 garbage, 2 mixed */
static int g_last_tokens[STEPS + 8];

static int draft_override_tramp(void *ctx, int64_t pos, int depth) {
    (void)ctx;
    int64_t i = pos - PROMPT_LEN;
    if (i < 0 || i >= PROMPT_LEN + STEPS + 8) return 0;
    int t = g_truth[i];
    if (ov_mode == 0) return t;
    if (ov_mode == 1) return (t + 1) % VOCAB;   /* garbage: never matches */
    return depth == 0 ? t : (t + 1) % VOCAB;    /* mixed: only draft-1
                                                   true */
}

/* one spec run; returns state digest via out */
static uint64_t run_spec(int K, float temp, uint64_t seed, int override_mode,
                         ApusSpecStats *ss) {
    ApusModelState st;
    apus_model_state_init(&st, &g_m);
    int tokens[STEPS + 8];
    if (override_mode >= 0)
        ov_mode = override_mode;
    int n = apus_spec_run(&g_m, &g_mw, &st, NULL, PROMPT, PROMPT_LEN,
                          STEPS, temp, 20, 0.95f, seed, K,
                          override_mode >= 0 ? draft_override_tramp : NULL,
                          NULL, tokens, ss);
    (void)n;
    uint64_t d = state_digest(&st);
    digest_bytes(&g_digest, tokens, (size_t)n * sizeof(int));
    apus_model_state_free(&st, &g_m);
    memcpy(g_last_tokens, tokens, (size_t)n * sizeof(int));
    return d;
}

static void test_equivalence(void) {
    for (int sampled = 0; sampled <= 1; sampled++) {
        float temp = sampled ? 0.6f : 0.0f;
        uint64_t seed = 42;
        int plain[STEPS];
        run_plain(STEPS, temp, seed, plain);
        /* plain state digest */
        ApusModelState pst;
        apus_model_state_init(&pst, &g_m);
        size_t V = (size_t)g_m.vocab;
        float *lg = malloc(V * sizeof(float));
        void *scratch = malloc(apus_sample_scratch_size(V));
        ApusRng rng;
        apus_rng_seed(&rng, seed);
        apus_model_forward(&g_m, &pst, PROMPT, PROMPT_LEN, lg, 0, NULL);
        for (int s = 0; s < STEPS; s++) {
            int t = apus_sample(lg, V, temp, 20, 0.95f, &rng, scratch);
            int64_t id = t;
            apus_model_forward(&g_m, &pst, &id, 1, lg, 0, NULL);
        }
        uint64_t plain_digest = state_digest(&pst);
        apus_model_state_free(&pst, &g_m);
        free(lg); free(scratch);

        for (int K = 2; K <= 4; K++) {
            ApusSpecStats ss;
            uint64_t sd = run_spec(K, temp, seed, -1, &ss);
            int same = memcmp(g_last_tokens, plain,
                              STEPS * sizeof(int)) == 0;
            printf("  %s K=%d: stream %s, state digest %s "
                   "(steps %llu, tok/batch %.2f, accept %llu/%llu)\n",
                   sampled ? "sampled" : "greedy ", K,
                   same ? "BITWISE " : "DIFFER  ",
                   sd == plain_digest ? "BITWISE" : "DIFFER ",
                   (unsigned long long)ss.steps,
                   ss.steps ? (double)STEPS / (double)ss.steps : 0.0,
                   (unsigned long long)ss.accepted,
                   (unsigned long long)ss.drafts);
            CHECK(same, "spec K=%d %s stream differs", K,
                  sampled ? "sampled" : "greedy");
            CHECK(sd == plain_digest, "spec K=%d %s state digest differs",
                  K, sampled ? "sampled" : "greedy");
        }
    }
}

static void test_forced(void) {
    int plain[STEPS + 16];
    run_plain(STEPS + 16, 0.0f, 42, plain);
    g_truth = plain;
    static const struct { const char *name; int mode; } pats[] = {
        { "truth", 0 }, { "garbage", 1 }, { "mixed", 2 },
    };
    for (int pi = 0; pi < 3; pi++) {
        ApusSpecStats ss;
        uint64_t sd = run_spec(3, 0.0f, 42, pats[pi].mode, &ss);
        int same = memcmp(g_last_tokens, plain, STEPS * sizeof(int)) == 0;
        CHECK(same, "forced %s: stream differs", pats[pi].name);
        printf("  forced %-7s: stream %s, steps %llu accept %llu/%llu "
               "full %llu refeeds %llu\n", pats[pi].name,
               same ? "BITWISE" : "DIFFER",
               (unsigned long long)ss.steps,
               (unsigned long long)ss.accepted,
               (unsigned long long)ss.drafts,
               (unsigned long long)ss.full_matches,
               (unsigned long long)ss.re_feeds);
        if (pats[pi].mode == 0)
            CHECK(ss.accepted == ss.drafts && ss.full_matches == ss.steps,
                  "forced truth: stats (%llu/%llu acc, %llu full)",
                  (unsigned long long)ss.accepted,
                  (unsigned long long)ss.drafts,
                  (unsigned long long)ss.full_matches);
        if (pats[pi].mode == 1)
            CHECK(ss.accepted == 0, "forced garbage: accepted %llu",
                  (unsigned long long)ss.accepted);
        if (pats[pi].mode == 2)
            CHECK(ss.accepted == ss.steps && ss.drafts == 2 * ss.steps,
                  "forced mixed: accepted %llu of %llu (steps %llu)",
                  (unsigned long long)ss.accepted,
                  (unsigned long long)ss.drafts,
                  (unsigned long long)ss.steps);
        (void)sd;
    }
}

static void test_tiered(void) {
    /* tiered spec (store serves the mtp layer-2 slabs through the
     * mtp.layers.0.* prefix mapping) == eager spec, bitwise */
    char err[256];
    ApusModel mt;
    if (apus_model_load_ex(&mt, MODEL, 256, 1, err, sizeof err)) {
        fprintf(stderr, "FAIL: tiered load: %s\n", err);
        failures++;
        return;
    }
    ApusStoreCfg sc = {0};
    sc.n_layers = mt.n_layers + mt.n_mtp;
    sc.n_main_layers = mt.n_layers;
    sc.n_experts = 16;
    sc.slots_per_layer = 16;
    sc.cache_bytes = 1;
    sc.pin_bytes = 1;
    sc.io_threads = 4;
    sc.usage_path = "";
    ApusStore *st = apus_store_open(MODEL, &sc, err, sizeof err);
    CHECK(st != NULL, "tiered store open: %s", err);
    if (!st) { apus_model_free(&mt); return; }
    apus_model_attach_store(&mt, st);
    ApusMtpW mw;
    if (apus_mtp_load(&mt, &mw, 1, err, sizeof err)) {
        fprintf(stderr, "FAIL: tiered mtp load: %s\n", err);
        failures++;
        apus_store_close(st);
        apus_model_free(&mt);
        return;
    }
    ApusModelState ms;
    apus_model_state_init(&ms, &mt);
    int tokens[STEPS + 8];
    ApusSpecStats ss;
    int n = apus_spec_run(&mt, &mw, &ms, st, PROMPT, PROMPT_LEN, STEPS,
                          0.0f, 20, 0.95f, 42, 2, NULL, NULL, tokens,
                          &ss);
    int plain[STEPS];
    run_plain(STEPS, 0.0f, 42, plain);
    int same = n == STEPS && memcmp(tokens, plain, STEPS * sizeof(int)) == 0;
    printf("  tiered spec K=2: stream %s (accept %llu/%llu)\n",
           same ? "BITWISE" : "DIFFER",
           (unsigned long long)ss.accepted,
           (unsigned long long)ss.drafts);
    CHECK(same, "tiered spec stream differs");
    digest_bytes(&g_digest, tokens, (size_t)n * sizeof(int));
    apus_model_state_free(&ms, &mt);
    apus_mtp_free(&mw);
    apus_store_close(st);
    apus_model_free(&mt);
}

int main(void) {
    printf("test_mtp: MTP speculative decoding gates (Qwen3.6-35B-A3B M8)\n");
    fprintf(stderr, "  pool threads: %d\n", apus_pool_threads());
    char err[256];
    if (apus_model_load(&g_m, MODEL, 256, err, sizeof err)) {
        fprintf(stderr, "FAIL: model load: %s\n", err);
        return 1;
    }
    if (apus_mtp_load(&g_m, &g_mw, 0, err, sizeof err)) {
        fprintf(stderr, "FAIL: mtp load: %s\n", err);
        return 1;
    }
    test_mtp_golden();
    test_equivalence();
    test_forced();
    test_tiered();
    printf("  digest %016llx\n", (unsigned long long)g_digest);
    printf("test_mtp: %ld checks, %d failures\n", checks, failures);
    apus_mtp_free(&g_mw);
    apus_model_free(&g_m);
    return failures ? 1 : 0;
}
