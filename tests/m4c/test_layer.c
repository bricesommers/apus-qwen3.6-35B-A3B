/*
 * tests/m4c/test_layer.c — hard-gate tests for c/layer.h (single-layer
 * forward, Qwen3.6-35B-A3B M4) against the M4b numpy-oracle fixtures
 * (tests/m4b/fixtures).
 *
 * For each layer kind (gdn = Gated DeltaNet + MoE, full = gated GQA
 * full attention + MoE — MoE on every layer):
 *
 *   1. STAGE GOLDENS: prefill_len 7 + 5 decode steps, every named
 *      intermediate compared against the oracle's f32 golden. Gate per
 *      stage: max|C - f32| <= envelope + slack*scale, where envelope =
 *      max|f32 - f64| (the oracle's own dtype divergence, from the
 *      fixture pair) and slack = 0.005 (bf16-valued stages) / 1e-3
 *      (fp32-valued stages) of scale = max|f32 golden| — i.e. C must
 *      land inside the oracle's f32-vs-f64 envelope up to fp32
 *      accumulation-order slack (which includes the ARM gemv hot path's
 *      user-approved M9b ILP reorder class). rtr_idx: EXACT.
 *   2. STATE: after the prefill, the GDN conv state (bf16 codes), S
 *      (fp32), the FULL KV caches (bf16 codes) and pos vs the state
 *      fixtures.
 *   3. CHUNK INVARIANCE (C-side, the load-bearing M5 property): one-shot
 *      T=12 forward vs prefill_len 7 + 5 single-token decodes — outputs
 *      and full state BITWISE identical (structural: one per-token
 *      body).
 *   4. Determinism: two identical runs bitwise identical.
 *
 * Run from the repository root (fixtures under tests/m4b/fixtures/).
 */
#define APUS_BF16_IMPLEMENTATION
#define APUS_GDN_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION
#define APUS_LAYER_IMPLEMENTATION
#include "bf16.h"
#include "gdn.h"
#include "attn.h"
#include "moe.h"
#include "layer.h"

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

/* ---- fixture loading ---- */
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

static double manifest_val(const char *key, int *found) {
    FILE *f = fopen("tests/m4b/fixtures/manifest.txt", "r");
    if (!f) { *found = 0; return 0; }
    char line[256];
    double out = 0;
    *found = 0;
    while (fgets(line, sizeof line, f)) {
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        if (strcmp(line, key) == 0) {
            out = strtod(eq + 1, NULL);
            *found = 1;
            break;
        }
    }
    fclose(f);
    return out;
}

static long man_l(const char *key) {
    int f;
    double v = manifest_val(key, &f);
    if (!f) { fprintf(stderr, "manifest key %s missing\n", key); exit(1); }
    return (long)v;
}

/* measured table: per (kind, stage) max|C-f32|, envelope, scale */
#define MAX_ROWS 512
static struct {
    char kind[8], stage[16];
    double diff, env, scale;
} g_rows[MAX_ROWS];
static int g_nrows = 0;

/* Compare a C stage against the f32 golden inside the f32/f64 envelope.
 * is_f32: 0 = C values are bf16 codes, 1 = fp32. slack per class. */
static void cmp_stage(const char *kind, const char *tag, const char *name,
                      const void *cvals, int is_f32, size_t n,
                      double slack) {
    char p[512];
    snprintf(p, sizeof p, "tests/m4b/fixtures/%s_%s_%s_f32.bin",
             kind, tag, name);
    size_t ln;
    double *g32 = (double *)read_file(p, &ln);
    snprintf(p, sizeof p, "tests/m4b/fixtures/%s_%s_%s_f64.bin",
             kind, tag, name);
    double *g64 = (double *)read_file(p, &ln);
    CHECK(g32 && g64, "%s/%s/%s golden load", kind, tag, name);
    if (!g32 || !g64) { free(g32); free(g64); return; }
    double env = 0, scale = 0, diff = 0;
    for (size_t i = 0; i < n; i++) {
        double a = g32[i], b = g64[i];
        if (fabs(a - b) > env) env = fabs(a - b);
        if (fabs(a) > scale) scale = fabs(a);
        double c = is_f32 ? (double)((const float *)cvals)[i]
                          : (double)apus_bf16_f32(
                                ((const uint16_t *)cvals)[i]);
        if (fabs(c - a) > diff) diff = fabs(c - a);
    }
    double gate = env + slack * (scale > 1e-6 ? scale : 1e-6);
    CHECK(diff <= gate,
          "%s/%s/%s: C-f32 %.4g > envelope %.4g + slack (scale %.4g)",
          kind, tag, name, diff, env, scale);
    if (g_nrows < MAX_ROWS) {
        snprintf(g_rows[g_nrows].kind, 8, "%s", kind);
        snprintf(g_rows[g_nrows].stage, 16, "%s", name);
        g_rows[g_nrows].diff = diff;
        g_rows[g_nrows].env = env;
        g_rows[g_nrows].scale = scale;
        g_nrows++;
    }
    free(g32); free(g64);
}

/* rtr_idx: EXACT */
static void cmp_idx(const char *kind, const char *tag,
                    const int32_t *cvals, size_t n) {
    char p[512];
    snprintf(p, sizeof p, "tests/m4b/fixtures/%s_%s_rtr_idx_f32.bin",
             kind, tag);
    size_t ln;
    int64_t *g = (int64_t *)read_file(p, &ln);
    CHECK(g != NULL, "%s/%s/rtr_idx golden load", kind, tag);
    if (!g) return;
    long bad = 0;
    for (size_t i = 0; i < n; i++)
        if ((int64_t)cvals[i] != g[i]) bad++;
    CHECK(bad == 0, "%s/%s/rtr_idx: %ld/%zu selection mismatches",
          kind, tag, bad, n);
    free(g);
}

/* ---- per-kind driver ---- */

typedef struct {
    ApusLayerCfg cfg;
    ApusLayerW w;
    unsigned char *blobs[64];
    int nblob;
} KindCtx;

static uint16_t *load_u16(KindCtx *kc, const char *kind, const char *name) {
    char p[512];
    snprintf(p, sizeof p, "tests/m4b/fixtures/w_%s_%s.bin", kind, name);
    size_t ln;
    uint16_t *b = (uint16_t *)read_file(p, &ln);
    if (!b) { fprintf(stderr, "missing %s\n", p); exit(1); }
    kc->blobs[kc->nblob++] = (unsigned char *)b;
    return b;
}
static float *load_f32(KindCtx *kc, const char *kind, const char *name) {
    char p[512];
    snprintf(p, sizeof p, "tests/m4b/fixtures/w_%s_%s.bin", kind, name);
    size_t ln;
    float *b = (float *)read_file(p, &ln);
    if (!b) { fprintf(stderr, "missing %s\n", p); exit(1); }
    kc->blobs[kc->nblob++] = (unsigned char *)b;
    return b;
}

static void load_kind(KindCtx *kc, const char *kind, ApusLayerKind k) {
    memset(kc, 0, sizeof *kc);
    ApusLayerCfg *c = &kc->cfg;
    int f;
    c->kind = k;
    c->hidden = (size_t)man_l("hidden");
    c->gdn_hk = (size_t)man_l("gdn_hk");
    c->gdn_hv = (size_t)man_l("gdn_hv");
    c->gdn_dk = (size_t)man_l("gdn_dk");
    c->gdn_dv = (size_t)man_l("gdn_dv");
    c->attn_nh = (size_t)man_l("attn_nh");
    c->attn_nkv = (size_t)man_l("attn_nkv");
    c->attn_d = (size_t)man_l("attn_d");
    c->attn_rot = (size_t)man_l("attn_rot");
    c->rope_theta = manifest_val("rope_theta", &f);
    c->max_seq = (size_t)(man_l("prefill_len") + man_l("decode_len"));
    c->experts = (size_t)man_l("experts");
    c->moe_inter = (size_t)man_l("moe_inter");
    c->shared_inter = (size_t)man_l("shared_inter");
    c->top_k = (size_t)man_l("top_k");
    ApusLayerW *w = &kc->w;
    w->ln1_w = load_u16(kc, kind, "ln1_w");
    w->ln2_w = load_u16(kc, kind, "ln2_w");
    if (k == APUS_LAYER_GDN) {
        w->w_qkv = load_u16(kc, kind, "w_qkv");
        w->w_z = load_u16(kc, kind, "w_z");
        w->w_b = load_u16(kc, kind, "w_b");
        w->w_a = load_u16(kc, kind, "w_a");
        w->conv_w = load_u16(kc, kind, "conv_w");
        w->A_log = load_f32(kc, kind, "A_log");
        w->dt_bias = load_f32(kc, kind, "dt_bias");
        w->onorm_w = load_u16(kc, kind, "onorm_w");
        w->w_out = load_u16(kc, kind, "w_out");
    } else {
        w->wq = load_u16(kc, kind, "wq");
        w->wk = load_u16(kc, kind, "wk");
        w->wv = load_u16(kc, kind, "wv");
        w->wo = load_u16(kc, kind, "wo");
        w->qn_w = load_u16(kc, kind, "qn_w");
        w->kn_w = load_u16(kc, kind, "kn_w");
    }
    w->rtr_w = load_u16(kc, kind, "rtr_w");
    w->exp_gu = load_u16(kc, kind, "exp_gu");
    w->exp_d = load_u16(kc, kind, "exp_d");
    w->sh_g = load_u16(kc, kind, "sh_g");
    w->sh_u = load_u16(kc, kind, "sh_u");
    w->sh_d = load_u16(kc, kind, "sh_d");
    w->sh_gate = load_u16(kc, kind, "sh_gate");
}

/* trace buffer set for a run of T tokens */
typedef struct {
    ApusLayerTrace tr;
    uint16_t *ln1, *attn_out, *res1, *ln2, *out;
    uint16_t *qkv_conv, *beta, *rec_o, *onorm;
    float *gdecay;
    uint16_t *qf, *kf, *attno, *mgate;
    int32_t *rtr_idx;
    uint16_t *rtr_w, *moe_routed, *moe_shared, *moe_out;
} TraceSet;

static void trace_alloc(TraceSet *ts, const ApusLayerCfg *c, size_t T) {
    memset(ts, 0, sizeof *ts);
    size_t H = c->hidden, TK = c->top_k;
    size_t conv_dim = 2 * c->gdn_hk * c->gdn_dk + c->gdn_hv * c->gdn_dv;
    size_t value_dim = c->gdn_hv * c->gdn_dv;
#define A16(f, n) ts->f = malloc((n) * 2); ts->tr.f = ts->f
    A16(ln1, T * H);
    A16(attn_out, T * H);
    A16(res1, T * H);
    A16(ln2, T * H);
    A16(out, T * H);
    if (c->kind == APUS_LAYER_GDN) {
        A16(qkv_conv, T * conv_dim);
        A16(beta, T * c->gdn_hv);
        ts->gdecay = malloc(T * c->gdn_hv * sizeof(float));
        ts->tr.gdecay = ts->gdecay;
        A16(rec_o, T * value_dim);
        A16(onorm, T * value_dim);
    } else {
        A16(qf, T * c->attn_nh * c->attn_d);
        A16(kf, T * c->attn_nkv * c->attn_d);
        A16(attno, T * c->attn_nh * c->attn_d);
        A16(mgate, T * c->attn_nh * c->attn_d);
    }
    ts->rtr_idx = malloc(T * TK * sizeof(int32_t));
    ts->tr.rtr_idx = ts->rtr_idx;
    A16(rtr_w, T * TK);
    A16(moe_routed, T * H);
    A16(moe_shared, T * H);
    A16(moe_out, T * H);
#undef A16
}

static void trace_free(TraceSet *ts) {
    free(ts->ln1); free(ts->attn_out); free(ts->res1); free(ts->ln2);
    free(ts->out); free(ts->qkv_conv); free(ts->beta); free(ts->rec_o);
    free(ts->onorm); free(ts->gdecay); free(ts->qf); free(ts->kf);
    free(ts->attno); free(ts->mgate); free(ts->rtr_idx); free(ts->rtr_w);
    free(ts->moe_routed); free(ts->moe_shared); free(ts->moe_out);
}

static ApusLayerState *state_new(const ApusLayerCfg *c) {
    ApusLayerState *st = calloc(1, sizeof *st);
    if (c->kind == APUS_LAYER_GDN) {
        size_t cd = 2 * c->gdn_hk * c->gdn_dk + c->gdn_hv * c->gdn_dv;
        st->conv_state = calloc(cd * 3, 2);
        st->S = calloc(c->gdn_hv * c->gdn_dk * c->gdn_dv, sizeof(float));
    } else {
        st->kcache = calloc(c->max_seq * c->attn_nkv * c->attn_d, 2);
        st->vcache = calloc(c->max_seq * c->attn_nkv * c->attn_d, 2);
    }
    apus_layer_state_zero(c, st);
    return st;
}

static void state_free(ApusLayerState *st) {
    free(st->conv_state); free(st->S);
    free(st->kcache); free(st->vcache); free(st);
}

static uint16_t *load_input(const char *kind, const char *which, size_t *n) {
    char p[512];
    snprintf(p, sizeof p, "tests/m4b/fixtures/%s_in_%s.bin", kind, which);
    uint16_t *b = (uint16_t *)read_file(p, n);
    if (!b) { fprintf(stderr, "missing %s\n", p); exit(1); }
    return b;
}

/* compare the C state after the prefill vs the state fixtures */
static void cmp_state(const char *kind, const ApusLayerCfg *c,
                      const ApusLayerState *st) {
    size_t T = (size_t)man_l("prefill_len");
    if (c->kind == APUS_LAYER_GDN) {
        size_t cd = 2 * c->gdn_hk * c->gdn_dk + c->gdn_hv * c->gdn_dv;
        char p[512];
        snprintf(p, sizeof p, "tests/m4b/fixtures/%s_state_conv.bin",
                 kind);
        size_t ln;
        uint16_t *g = (uint16_t *)read_file(p, &ln);
        CHECK(g != NULL, "%s state conv load", kind);
        if (g) {
            long bad = 0;
            for (size_t j = 0; j < cd * 3; j++)
                if (st->conv_state[j] != g[j]) bad++;
            /* single-code tie flips are the documented realization
             * class (C sequential fp32 vs the oracle's f64-accumulated
             * ideal; Apus m4c allowed 2% for the same reason) */
            printf("  %s state conv: %ld/%zu code flips\n",
                   kind, bad, cd * 3);
            CHECK(bad * 100 <= (long)(cd * 3),
                  "%s state conv: %ld/%zu code flips (>1%%)",
                  kind, bad, cd * 3);
            free(g);
        }
        snprintf(p, sizeof p, "tests/m4b/fixtures/%s_state_S.bin", kind);
        float *gs = (float *)read_file(p, &ln);
        CHECK(gs != NULL, "%s state S load", kind);
        if (gs) {
            double diff = 0, scale = 0;
            size_t n = c->gdn_hv * c->gdn_dk * c->gdn_dv;
            for (size_t j = 0; j < n; j++) {
                if (fabs((double)gs[j]) > scale) scale = fabs((double)gs[j]);
                double d = fabs((double)st->S[j] - (double)gs[j]);
                if (d > diff) diff = d;
            }
            CHECK(diff <= 1e-3 * (scale > 1e-6 ? scale : 1e-6),
                  "%s state S: diff %.4g scale %.4g", kind, diff, scale);
            printf("  %s state S: max diff %.3g (scale %.3g)\n",
                   kind, diff, scale);
            free(gs);
        }
    } else {
        const char *names[2] = { "kcache", "vcache" };
        const uint16_t *vals[2] = { st->kcache, st->vcache };
        size_t ns = T * c->attn_nkv * c->attn_d;
        for (int i = 0; i < 2; i++) {
            char p[512];
            snprintf(p, sizeof p, "tests/m4b/fixtures/%s_state_%s.bin",
                     kind, names[i]);
            size_t ln;
            uint16_t *g = (uint16_t *)read_file(p, &ln);
            CHECK(g != NULL, "%s state %s load", kind, names[i]);
            if (!g) continue;
            long bad = 0;
            for (size_t j = 0; j < ns; j++)
                if (vals[i][j] != g[j]) bad++;
            /* single-code tie flips: documented realization class */
            printf("  %s state %s: %ld/%zu code flips\n",
                   kind, names[i], bad, ns);
            CHECK(bad * 100 <= (long)ns,
                  "%s state %s: %ld/%zu code flips (>1%%)",
                  kind, names[i], bad, ns);
            free(g);
        }
    }
    CHECK(st->pos == T, "%s state pos: %zu != %zu", kind, st->pos, T);
}

static void run_kind(const char *kind, ApusLayerKind k) {
    KindCtx kc;
    load_kind(&kc, kind, k);
    const ApusLayerCfg *c = &kc.cfg;
    size_t H = c->hidden;
    size_t T = (size_t)man_l("prefill_len");
    size_t TD = (size_t)man_l("decode_len");
    size_t TK = c->top_k;
    size_t conv_dim = 2 * c->gdn_hk * c->gdn_dk + c->gdn_hv * c->gdn_dv;
    size_t value_dim = c->gdn_hv * c->gdn_dv;
    size_t nhd = c->attn_nh * c->attn_d;
    size_t nkvd = c->attn_nkv * c->attn_d;

    size_t np, nd;
    uint16_t *xp = load_input(kind, "prefill", &np);
    uint16_t *xd = load_input(kind, "decode", &nd);
    CHECK(np == T * H * 2 && nd == TD * H * 2, "%s input sizes", kind);

    /* 1. prefill with full trace + per-stage compares */
    ApusLayerState *st = state_new(c);
    TraceSet ts;
    trace_alloc(&ts, c, T);
    uint16_t *out = malloc(T * H * 2);
    apus_layer_forward(c, &kc.w, st, xp, out, T, &ts.tr);

    cmp_stage(kind, "pre", "ln1", ts.ln1, 0, T * H, 0.005);
    cmp_stage(kind, "pre", "attn_out", ts.attn_out, 0, T * H, 0.005);
    cmp_stage(kind, "pre", "res1", ts.res1, 0, T * H, 0.005);
    cmp_stage(kind, "pre", "ln2", ts.ln2, 0, T * H, 0.005);
    cmp_stage(kind, "pre", "out", ts.out, 0, T * H, 0.005);
    if (k == APUS_LAYER_GDN) {
        cmp_stage(kind, "pre", "qkv_conv", ts.qkv_conv, 0,
                  T * conv_dim, 0.005);
        cmp_stage(kind, "pre", "beta", ts.beta, 0, T * c->gdn_hv, 0.005);
        cmp_stage(kind, "pre", "gdecay", ts.gdecay, 1, T * c->gdn_hv,
                  0.001);
        cmp_stage(kind, "pre", "rec_o", ts.rec_o, 0, T * value_dim,
                  0.005);
        cmp_stage(kind, "pre", "onorm", ts.onorm, 0, T * value_dim,
                  0.005);
    } else {
        cmp_stage(kind, "pre", "qf", ts.qf, 0, T * nhd, 0.005);
        cmp_stage(kind, "pre", "kf", ts.kf, 0, T * nkvd, 0.005);
        cmp_stage(kind, "pre", "attno", ts.attno, 0, T * nhd, 0.005);
        cmp_stage(kind, "pre", "mgate", ts.mgate, 0, T * nhd, 0.005);
    }
    cmp_idx(kind, "pre", ts.rtr_idx, T * TK);
    cmp_stage(kind, "pre", "rtr_w", ts.rtr_w, 0, T * TK, 0.005);
    cmp_stage(kind, "pre", "moe_routed", ts.moe_routed, 0, T * H, 0.005);
    cmp_stage(kind, "pre", "moe_shared", ts.moe_shared, 0, T * H, 0.005);
    cmp_stage(kind, "pre", "moe_out", ts.moe_out, 0, T * H, 0.005);
    cmp_state(kind, c, st);
    digest_bytes(&g_digest, out, T * H * 2);

    /* 2. decode chain with carried state, per-step compares */
    for (size_t i = 0; i < TD; i++) {
        char tag[16];
        snprintf(tag, sizeof tag, "dec%zu", i);
        TraceSet ds;
        trace_alloc(&ds, c, 1);
        uint16_t *dout = malloc(H * 2);
        apus_layer_forward(c, &kc.w, st, xd + i * H, dout, 1, &ds.tr);
        cmp_stage(kind, tag, "ln1", ds.ln1, 0, H, 0.005);
        cmp_stage(kind, tag, "attn_out", ds.attn_out, 0, H, 0.005);
        cmp_stage(kind, tag, "res1", ds.res1, 0, H, 0.005);
        cmp_stage(kind, tag, "ln2", ds.ln2, 0, H, 0.005);
        cmp_stage(kind, tag, "out", ds.out, 0, H, 0.005);
        if (k == APUS_LAYER_GDN) {
            cmp_stage(kind, tag, "rec_o", ds.rec_o, 0, value_dim, 0.005);
            cmp_stage(kind, tag, "onorm", ds.onorm, 0, value_dim, 0.005);
        } else {
            cmp_stage(kind, tag, "attno", ds.attno, 0, nhd, 0.005);
        }
        cmp_idx(kind, tag, ds.rtr_idx, TK);
        cmp_stage(kind, tag, "moe_out", ds.moe_out, 0, H, 0.005);
        digest_bytes(&g_digest, dout, H * 2);
        free(dout);
        trace_free(&ds);
    }

    /* 3. chunk invariance: one-shot T+TD vs T + TD decode steps,
     * BITWISE outputs and state */
    uint16_t *xall = malloc((T + TD) * H * 2);
    memcpy(xall, xp, T * H * 2);
    memcpy(xall + T * H, xd, TD * H * 2);
    ApusLayerState *s1 = state_new(c);
    uint16_t *o1 = malloc((T + TD) * H * 2);
    apus_layer_forward(c, &kc.w, s1, xall, o1, T + TD, NULL);
    ApusLayerState *s2 = state_new(c);
    uint16_t *o2 = malloc((T + TD) * H * 2);
    apus_layer_forward(c, &kc.w, s2, xall, o2, T, NULL);
    for (size_t i = 0; i < TD; i++)
        apus_layer_forward(c, &kc.w, s2, xall + (T + i) * H,
                           o2 + (T + i) * H, 1, NULL);
    CHECK(memcmp(o1, o2, (T + TD) * H * 2) == 0,
          "%s: chunk invariance outputs NOT bitwise", kind);
    int state_eq = 1;
    if (k == APUS_LAYER_GDN) {
        state_eq &= memcmp(s1->conv_state, s2->conv_state,
                           conv_dim * 3 * 2) == 0;
        state_eq &= memcmp(s1->S, s2->S,
                           c->gdn_hv * c->gdn_dk * c->gdn_dv
                           * sizeof(float)) == 0;
    } else {
        state_eq &= memcmp(s1->kcache, s2->kcache,
                           (T + TD) * nkvd * 2) == 0;
        state_eq &= memcmp(s1->vcache, s2->vcache,
                           (T + TD) * nkvd * 2) == 0;
    }
    state_eq &= s1->pos == s2->pos;
    CHECK(state_eq, "%s: chunk invariance state NOT bitwise", kind);
    digest_bytes(&g_digest, o1, (T + TD) * H * 2);

    /* 4. determinism: identical rerun bitwise */
    ApusLayerState *s3 = state_new(c);
    uint16_t *o3 = malloc((T + TD) * H * 2);
    apus_layer_forward(c, &kc.w, s3, xall, o3, T + TD, NULL);
    CHECK(memcmp(o1, o3, (T + TD) * H * 2) == 0,
          "%s: rerun NOT bitwise deterministic", kind);

    state_free(st); state_free(s1); state_free(s2); state_free(s3);
    trace_free(&ts);
    free(out); free(o1); free(o2); free(o3); free(xall);
    free(xp); free(xd);
    for (int i = 0; i < kc.nblob; i++) free(kc.blobs[i]);
}

int main(void) {
    printf("test_layer: single-layer forward hard-gate tests "
           "(Qwen3.6-35B-A3B M4c)\n");
    fprintf(stderr, "  pool threads: %d\n", apus_pool_threads());
    int f;
    manifest_val("hidden", &f);
    if (!f) {
        fprintf(stderr, "FAIL: could not load fixture manifest\n");
        return 1;
    }

    run_kind("gdn", APUS_LAYER_GDN);
    run_kind("full", APUS_LAYER_FULL);

    printf("\n  per-stage C-vs-f32 vs envelope (worst rows per stage):\n");
    printf("  %-6s %-12s %12s %12s %12s\n", "kind", "stage",
           "C-f32", "envelope", "scale");
    /* print the worst diff/(env) row per stage name */
    for (int i = 0; i < g_nrows; i++) {
        int worst = 1;
        for (int j = 0; j < g_nrows; j++) {
            if (j == i) continue;
            if (!strcmp(g_rows[i].kind, g_rows[j].kind) &&
                !strcmp(g_rows[i].stage, g_rows[j].stage) &&
                g_rows[j].diff > g_rows[i].diff) { worst = 0; break; }
        }
        if (worst)
            printf("  %-6s %-12s %12.4g %12.4g %12.4g\n",
                   g_rows[i].kind, g_rows[i].stage, g_rows[i].diff,
                   g_rows[i].env, g_rows[i].scale);
    }
    printf("  digest %016llx\n", (unsigned long long)g_digest);
    printf("test_layer: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
