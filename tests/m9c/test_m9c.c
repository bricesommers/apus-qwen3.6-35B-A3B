/*
 * tests/m9c/test_m9c.c — hard gates for the M9 batched-prefill
 * restoration (c/layer.h apus_layer_attn_batch_{gdn,full} +
 * apus_layer_moe_batch + c/cache.h's lockstep store mirrors): the Ling
 * M6c/M9c structure re-expressed on the Qwen kernels.
 *
 * THE CONTRACT: the batched prefill (phase A: projection GEMMs at M=tc
 * through apus_bf16_gemm_hot, ONE batched conv1d over the chunk, ONE
 * gqa_mt over the APUS_PREFILL_ATTN_CHUNK-token chunk; phase B: shared
 * expert at M=T + unique-expert gate_up/down at M=count) is BITWISE
 * identical to the per-token body it replaces — the ILP GEMM's
 * M-independence (tests/m9b), the m4a conv-prefill == step-loop and
 * GQA-decode == full-recompute-row identities. NO gemm_fast/BLAS
 * anywhere in the batched path (the M-independent-bitwise class only),
 * so the identity holds at EVERY T on EVERY platform — there is no
 * M>=128 escape. This suite pins that at T=256 (crossing the chunk
 * boundary... one chunk exactly, plus a 300-token run that crosses it)
 * on synthetic layers of both Qwen kinds:
 *
 *   1. LAYER-LEVEL: apus_layer_forward_hot at T=256/300/64 vs the
 *      sequential apus_layer_forward — EVERY attention trace field
 *      (ln1/attn_out/res1/ln2 + GDN qkv_conv/beta/gdecay/rec_o/onorm +
 *      FULL qf/kf/attno/mgate), the MoE traces, the FFN out, and the
 *      layer state (conv state, S, KV caches, pos) BITWISE.
 *   2. MODEL-LEVEL: eager hot prefill T=64 == sequential decodes
 *      (logits + layer state), mirroring m6c's T=16 gate; tiered (m6a
 *      fixture) same.
 *   3. TIERED ATTENTION stream: the store path's post_attn hook captures
 *      the res1 stream — batched == sequential BITWISE at T=64 AND
 *      T=256 (the phase-B restoration consumes no BLAS class, so the
 *      stream never leaves the pinned realization).
 *
 * The FNV digest is diffed across APUS_THREADS=1/4/8 by the Makefile.
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
static float rng_float(void) {   /* uniform in (-2, 2) */
    return (float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0
                   - 2.0);
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
/* 1. synthetic layer attention gate ----------------------------------------*/

/* mini dims (m4b-flavored, small MoE so phase B runs but stays cheap) */
#define M9C_H    256
#define M9C_HK   2
#define M9C_HV   4
#define M9C_DK   64
#define M9C_DV   64
#define M9C_NH   4
#define M9C_NKV  2
#define M9C_D    64
#define M9C_ROT  16
#define M9C_E    8
#define M9C_I    32
#define M9C_IS   32
#define M9C_TK   2

typedef struct {
    ApusLayerCfg c;
    ApusLayerW w;
    /* backing stores */
    uint16_t *ln1w, *ln2w, *wqkv, *wz, *wb, *wa, *convw, *onw, *wout,
        *wq, *wk, *wv, *wo, *qnw, *knw, *rtrw, *egu, *ed, *sg, *su, *sd,
        *sgate;
    float *alog, *dtb;
    ApusLayerState st;
} SynLayer;

static uint16_t *bf16buf(size_t n, float scale) {
    uint16_t *p = malloc(n * 2);
    for (size_t i = 0; i < n; i++) p[i] = rng_bf16_scaled(scale);
    return p;
}
static uint16_t *normbuf(size_t n) {   /* (1+w) stored values: N(0, 0.1) */
    uint16_t *p = malloc(n * 2);
    for (size_t i = 0; i < n; i++)
        p[i] = apus_bf16_bits(rng_float() * 0.025f);
    return p;
}

static void syn_layer_init(SynLayer *L, ApusLayerKind kind, size_t max_seq) {
    size_t H = M9C_H;
    size_t key_dim = M9C_HK * M9C_DK, value_dim = M9C_HV * M9C_DV;
    size_t conv_dim = 2 * key_dim + value_dim;
    memset(L, 0, sizeof *L);
    L->c.kind = kind;
    L->c.hidden = H;
    L->c.gdn_hk = M9C_HK; L->c.gdn_hv = M9C_HV;
    L->c.gdn_dk = M9C_DK; L->c.gdn_dv = M9C_DV;
    L->c.attn_nh = M9C_NH; L->c.attn_nkv = M9C_NKV;
    L->c.attn_d = M9C_D; L->c.attn_rot = M9C_ROT;
    L->c.rope_theta = 10000.0;
    L->c.max_seq = max_seq;
    L->c.experts = M9C_E; L->c.moe_inter = M9C_I;
    L->c.shared_inter = M9C_IS;
    L->c.top_k = M9C_TK;
    L->w.ln1_w = normbuf(H);
    L->w.ln2_w = normbuf(H);
    if (kind == APUS_LAYER_GDN) {
        L->w.w_qkv = bf16buf(conv_dim * H, 0.05f);
        L->w.w_z = bf16buf(value_dim * H, 0.05f);
        L->w.w_b = bf16buf(M9C_HV * H, 0.05f);
        L->w.w_a = bf16buf(M9C_HV * H, 0.05f);
        L->w.conv_w = bf16buf(conv_dim * 4, 0.3f);
        L->alog = malloc(M9C_HV * sizeof(float));
        for (size_t i = 0; i < M9C_HV; i++)
            L->alog[i] = logf(1.0f + (float)(rng_u64() % 1500) / 100.0f);
        L->w.A_log = L->alog;
        L->dtb = malloc(M9C_HV * sizeof(float));
        for (size_t i = 0; i < M9C_HV; i++)
            L->dtb[i] = 1.0f + rng_float() * 0.125f;
        L->w.dt_bias = L->dtb;
        L->w.onorm_w = normbuf(M9C_DV);
        L->w.w_out = bf16buf(H * value_dim, 0.05f);
        L->st.conv_state = calloc(conv_dim * 3, 2);
        L->st.S = calloc((size_t)M9C_HV * M9C_DK * M9C_DV, sizeof(float));
    } else {
        L->w.wq = bf16buf((size_t)M9C_NH * 2 * M9C_D * H, 0.05f);
        L->w.wk = bf16buf((size_t)M9C_NKV * M9C_D * H, 0.05f);
        L->w.wv = bf16buf((size_t)M9C_NKV * M9C_D * H, 0.05f);
        L->w.wo = bf16buf(H * M9C_NH * M9C_D, 0.05f);
        L->w.qn_w = normbuf(M9C_D);
        L->w.kn_w = normbuf(M9C_D);
        L->st.kcache = malloc(max_seq * M9C_NKV * M9C_D * 2);
        L->st.vcache = malloc(max_seq * M9C_NKV * M9C_D * 2);
    }
    L->w.rtr_w = bf16buf(M9C_E * H, 0.05f);
    L->w.exp_gu = bf16buf((size_t)M9C_E * 2 * M9C_I * H, 0.05f);
    L->w.exp_d = bf16buf((size_t)M9C_E * H * M9C_I, 0.05f);
    L->w.sh_g = bf16buf(M9C_IS * H, 0.05f);
    L->w.sh_u = bf16buf(M9C_IS * H, 0.05f);
    L->w.sh_d = bf16buf(H * M9C_IS, 0.05f);
    L->w.sh_gate = bf16buf(H, 0.05f);
    L->st.pos = 0;
}

/* deep copy of the state (so batched and sequential runs start equal) */
static void syn_state_copy(SynLayer *dst, SynLayer *src, size_t max_seq) {
    if (src->c.kind == APUS_LAYER_GDN) {
        size_t conv_dim = 2 * M9C_HK * M9C_DK + M9C_HV * M9C_DV;
        dst->st.conv_state = malloc(conv_dim * 3 * 2);
        dst->st.S = malloc((size_t)M9C_HV * M9C_DK * M9C_DV
                           * sizeof(float));
        memcpy(dst->st.conv_state, src->st.conv_state, conv_dim * 3 * 2);
        memcpy(dst->st.S, src->st.S,
               (size_t)M9C_HV * M9C_DK * M9C_DV * sizeof(float));
    } else {
        dst->st.kcache = malloc(max_seq * M9C_NKV * M9C_D * 2);
        dst->st.vcache = malloc(max_seq * M9C_NKV * M9C_D * 2);
        memcpy(dst->st.kcache, src->st.kcache,
               max_seq * M9C_NKV * M9C_D * 2);
        memcpy(dst->st.vcache, src->st.vcache,
               max_seq * M9C_NKV * M9C_D * 2);
    }
    dst->st.pos = src->st.pos;
}

static void syn_state_free(SynLayer *L) {
    if (L->c.kind == APUS_LAYER_GDN) {
        free(L->st.conv_state); free(L->st.S);
    } else {
        free(L->st.kcache); free(L->st.vcache);
    }
}

static void syn_layer_free(SynLayer *L) {
    free(L->ln1w); free(L->ln2w); free(L->wqkv); free(L->wz); free(L->wb);
    free(L->wa); free(L->convw); free(L->onw); free(L->wout);
    free(L->wq); free(L->wk); free(L->wv); free(L->wo);
    free(L->qnw); free(L->knw);
    free(L->rtrw); free(L->egu); free(L->ed); free(L->sg); free(L->su);
    free(L->sd); free(L->sgate);
    free(L->alog); free(L->dtb);
    syn_state_free(L);
}

typedef struct {
    uint16_t *ln1, *attn_out, *res1, *ln2, *out;
    uint16_t *qkv_conv, *beta, *rec_o, *onorm;
    float *gdecay;
    uint16_t *qf, *kf, *attno, *mgate;
    int32_t *rtr_idx;
    uint16_t *rtr_w, *moe_routed, *moe_shared, *moe_out;
    ApusLayerTrace tr;
} SynTrace;

static void syn_trace_alloc(SynTrace *t, ApusLayerKind kind, size_t T) {
    size_t H = M9C_H;
    size_t conv_dim = 2 * M9C_HK * M9C_DK + M9C_HV * M9C_DV;
    size_t value_dim = M9C_HV * M9C_DV;
    memset(t, 0, sizeof *t);
    t->ln1 = malloc(T * H * 2);
    t->attn_out = malloc(T * H * 2);
    t->res1 = malloc(T * H * 2);
    t->ln2 = malloc(T * H * 2);
    t->out = malloc(T * H * 2);
    t->tr.ln1 = t->ln1;
    t->tr.attn_out = t->attn_out;
    t->tr.res1 = t->res1;
    t->tr.ln2 = t->ln2;
    t->tr.out = t->out;
    /* phase-B MoE traces (the batched phase B is gated bitwise too) */
    t->rtr_idx = malloc(T * M9C_TK * sizeof(int32_t));
    t->rtr_w = malloc(T * M9C_TK * 2);
    t->moe_routed = malloc(T * H * 2);
    t->moe_shared = malloc(T * H * 2);
    t->moe_out = malloc(T * H * 2);
    t->tr.rtr_idx = t->rtr_idx;
    t->tr.rtr_w = t->rtr_w;
    t->tr.moe_routed = t->moe_routed;
    t->tr.moe_shared = t->moe_shared;
    t->tr.moe_out = t->moe_out;
    if (kind == APUS_LAYER_GDN) {
        t->qkv_conv = malloc(T * conv_dim * 2);
        t->beta = malloc(T * M9C_HV * 2);
        t->gdecay = malloc(T * M9C_HV * sizeof(float));
        t->rec_o = malloc(T * value_dim * 2);
        t->onorm = malloc(T * value_dim * 2);
        t->tr.qkv_conv = t->qkv_conv;
        t->tr.beta = t->beta;
        t->tr.gdecay = t->gdecay;
        t->tr.rec_o = t->rec_o;
        t->tr.onorm = t->onorm;
    } else {
        t->qf = malloc(T * M9C_NH * M9C_D * 2);
        t->kf = malloc(T * M9C_NKV * M9C_D * 2);
        t->attno = malloc(T * M9C_NH * M9C_D * 2);
        t->mgate = malloc(T * M9C_NH * M9C_D * 2);
        t->tr.qf = t->qf;
        t->tr.kf = t->kf;
        t->tr.attno = t->attno;
        t->tr.mgate = t->mgate;
    }
}

static void test_attn_batch_kind(ApusLayerKind kind, size_t T,
                                 const char *name) {
    SynLayer lb, ls;
    syn_layer_init(&lb, kind, T + 8);
    /* identical starting state for the sequential run */
    ls.c = lb.c;
    ls.w = lb.w;
    syn_state_copy(&ls, &lb, T + 8);
    SynTrace tb, ts;
    syn_trace_alloc(&tb, kind, T);
    syn_trace_alloc(&ts, kind, T);
    uint16_t *x = malloc(T * M9C_H * 2);
    uint16_t *ob = malloc(T * M9C_H * 2);
    uint16_t *os = malloc(T * M9C_H * 2);
    for (size_t i = 0; i < T * M9C_H; i++) x[i] = rng_bf16_scaled(1.0f);
    size_t H = M9C_H;
    size_t conv_dim = 2 * M9C_HK * M9C_DK + M9C_HV * M9C_DV;
    size_t value_dim = M9C_HV * M9C_DV;
    apus_layer_forward_hot(&lb.c, &lb.w, &lb.st, x, ob, T, &tb.tr);
    for (size_t t = 0; t < T; t++) {
        /* shift the trace row pointers so each single-token call writes
         * row t (apus_layer_forward always writes row 0 at T=1) */
        ApusLayerTrace trt = ts.tr;
        if (trt.ln1) trt.ln1 += t * M9C_H;
        if (trt.attn_out) trt.attn_out += t * M9C_H;
        if (trt.res1) trt.res1 += t * M9C_H;
        if (trt.ln2) trt.ln2 += t * M9C_H;
        if (trt.qkv_conv) trt.qkv_conv += t * conv_dim;
        if (trt.beta) trt.beta += t * M9C_HV;
        if (trt.gdecay) trt.gdecay += t * M9C_HV;
        if (trt.rec_o) trt.rec_o += t * value_dim;
        if (trt.onorm) trt.onorm += t * value_dim;
        if (trt.qf) trt.qf += t * M9C_NH * M9C_D;
        if (trt.kf) trt.kf += t * M9C_NKV * M9C_D;
        if (trt.attno) trt.attno += t * M9C_NH * M9C_D;
        if (trt.mgate) trt.mgate += t * M9C_NH * M9C_D;
        if (trt.rtr_idx) trt.rtr_idx += t * M9C_TK;
        if (trt.rtr_w) trt.rtr_w += t * M9C_TK;
        if (trt.moe_routed) trt.moe_routed += t * M9C_H;
        if (trt.moe_shared) trt.moe_shared += t * M9C_H;
        if (trt.moe_out) trt.moe_out += t * M9C_H;
        apus_layer_forward(&ls.c, &ls.w, &ls.st, x + t * M9C_H,
                           os + t * M9C_H, 1, &trt);
    }
    CHECK(memcmp(tb.ln1, ts.ln1, T * H * 2) == 0, "%s T=%zu: ln1", name, T);
    CHECK(memcmp(tb.attn_out, ts.attn_out, T * H * 2) == 0,
          "%s T=%zu: attn_out", name, T);
    CHECK(memcmp(tb.res1, ts.res1, T * H * 2) == 0, "%s T=%zu: res1",
          name, T);
    CHECK(memcmp(tb.ln2, ts.ln2, T * H * 2) == 0, "%s T=%zu: ln2", name, T);
    if (kind == APUS_LAYER_GDN) {
        CHECK(memcmp(tb.qkv_conv, ts.qkv_conv, T * conv_dim * 2) == 0,
              "%s T=%zu: qkv_conv", name, T);
        CHECK(memcmp(tb.beta, ts.beta, T * M9C_HV * 2) == 0,
              "%s T=%zu: beta", name, T);
        CHECK(memcmp(tb.gdecay, ts.gdecay,
                     T * M9C_HV * sizeof(float)) == 0,
              "%s T=%zu: gdecay", name, T);
        CHECK(memcmp(tb.rec_o, ts.rec_o, T * value_dim * 2) == 0,
              "%s T=%zu: rec_o", name, T);
        CHECK(memcmp(tb.onorm, ts.onorm, T * value_dim * 2) == 0,
              "%s T=%zu: onorm", name, T);
        int stt = memcmp(lb.st.conv_state, ls.st.conv_state,
                         conv_dim * 3 * 2) == 0
               && memcmp(lb.st.S, ls.st.S,
                         (size_t)M9C_HV * M9C_DK * M9C_DV
                         * sizeof(float)) == 0;
        CHECK(stt, "%s T=%zu: GDN state", name, T);
    } else {
        CHECK(memcmp(tb.qf, ts.qf, T * M9C_NH * M9C_D * 2) == 0,
              "%s T=%zu: qf", name, T);
        CHECK(memcmp(tb.kf, ts.kf, T * M9C_NKV * M9C_D * 2) == 0,
              "%s T=%zu: kf", name, T);
        CHECK(memcmp(tb.attno, ts.attno, T * M9C_NH * M9C_D * 2) == 0,
              "%s T=%zu: attno", name, T);
        CHECK(memcmp(tb.mgate, ts.mgate, T * M9C_NH * M9C_D * 2) == 0,
              "%s T=%zu: mgate", name, T);
        int stt = memcmp(lb.st.kcache, ls.st.kcache,
                         T * M9C_NKV * M9C_D * 2) == 0
               && memcmp(lb.st.vcache, ls.st.vcache,
                         T * M9C_NKV * M9C_D * 2) == 0;
        CHECK(stt, "%s T=%zu: KV caches", name, T);
    }
    CHECK(lb.st.pos == ls.st.pos, "%s T=%zu: pos", name, T);
    CHECK(memcmp(ob, os, T * H * 2) == 0, "%s T=%zu: out", name, T);
    /* phase-B MoE traces (router selection/weights + the batched expert
     * combine path) bitwise too */
    CHECK(memcmp(tb.rtr_idx, ts.rtr_idx, T * M9C_TK * sizeof(int32_t))
          == 0, "%s T=%zu: rtr_idx", name, T);
    CHECK(memcmp(tb.rtr_w, ts.rtr_w, T * M9C_TK * 2) == 0,
          "%s T=%zu: rtr_w", name, T);
    CHECK(memcmp(tb.moe_routed, ts.moe_routed, T * H * 2) == 0,
          "%s T=%zu: moe_routed", name, T);
    CHECK(memcmp(tb.moe_shared, ts.moe_shared, T * H * 2) == 0,
          "%s T=%zu: moe_shared", name, T);
    CHECK(memcmp(tb.moe_out, ts.moe_out, T * H * 2) == 0,
          "%s T=%zu: moe_out", name, T);
    digest_bytes(&g_digest, tb.res1, T * H * 2);
    digest_bytes(&g_digest, tb.attn_out, T * H * 2);
    printf("  attn batch %-5s T=%-4zu: traces+state BITWISE\n", name, T);
    free(x); free(ob); free(os);
    free(tb.ln1); free(tb.attn_out); free(tb.res1); free(tb.ln2);
    free(tb.out); free(tb.qkv_conv); free(tb.beta); free(tb.gdecay);
    free(tb.rec_o); free(tb.onorm);
    free(tb.qf); free(tb.kf); free(tb.attno); free(tb.mgate);
    free(tb.rtr_idx); free(tb.rtr_w); free(tb.moe_routed);
    free(tb.moe_shared); free(tb.moe_out);
    free(ts.ln1); free(ts.attn_out); free(ts.res1); free(ts.ln2);
    free(ts.out); free(ts.qkv_conv); free(ts.beta); free(ts.gdecay);
    free(ts.rec_o); free(ts.onorm);
    free(ts.qf); free(ts.kf); free(ts.attno); free(ts.mgate);
    free(ts.rtr_idx); free(ts.rtr_w); free(ts.moe_routed);
    free(ts.moe_shared); free(ts.moe_out);
    syn_layer_free(&lb);       /* lb owns the weight backing stores */
    syn_state_free(&ls);       /* ls shares lb's weights; own state only */
}

/* =========================================================================*/
/* 2. model-level prefill == sequential (mirrors m6c, T=64) ------------------*/
static void test_model_prefill(void) {
    char err[256];
    size_t T = 64;
    int64_t ids[64];
    for (size_t i = 0; i < T; i++) ids[i] = (int64_t)(rng_u64() % 256);
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
          "eager batched prefill T=64 != sequential (logits)");
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
    CHECK(state_same, "eager batched prefill T=64: layer state differs");
    digest_bytes(&g_digest, l1, T * V * sizeof(float));
    apus_model_state_free(&s1, &m);
    apus_model_state_free(&s2, &m);
    apus_model_free(&m);

    /* tiered T=64 (mirrors m6c) */
    ApusModel mt;
    if (apus_model_load_ex(&mt, "tests/m6a/fixtures/model", 256, 1, err,
                           sizeof err)) {
        fprintf(stderr, "FAIL: tiered load: %s\n", err);
        failures++;
        free(l1); free(l2);
        return;
    }
    ApusStoreCfg sc = {0};
    sc.n_layers = mt.n_layers;
    sc.n_experts = (int)mt.layers[0].lc.experts;
    sc.slots_per_layer = (int)mt.layers[0].lc.experts;
    sc.cache_bytes = 1;
    sc.pin_bytes = 1;
    sc.io_threads = -1;
    sc.usage_path = "";
    ApusStore *stp = apus_store_open("tests/m6a/fixtures/model", &sc, err,
                                     sizeof err);
    CHECK(stp != NULL, "store open: %s", err);
    if (!stp) {
        apus_model_state_free(&s1, &m);
        apus_model_state_free(&s2, &m);
        apus_model_free(&m);
        free(l1); free(l2);
        return;
    }
    apus_model_attach_store(&mt, stp);
    ApusModelState s3, s4;
    apus_model_state_init(&s3, &mt);
    apus_model_forward(&mt, &s3, ids, T, l1, 1, NULL);
    apus_model_state_init(&s4, &mt);
    for (size_t t = 0; t < T; t++)
        apus_model_forward(&mt, &s4, ids + t, 1, l2 + t * V, 0, NULL);
    CHECK(memcmp(l1, l2, T * V * sizeof(float)) == 0,
          "tiered batched prefill T=64 != sequential (logits)");
    if (memcmp(l1, l2, T * V * sizeof(float)) != 0) {
        for (size_t t = 0; t < T; t++)
            for (size_t v = 0; v < V; v++)
                if (l1[t * V + v] != l2[t * V + v]) {
                    fprintf(stderr, "  DIFF tiered T=64: first at t=%zu "
                            "v=%zu: %a vs %a\n", t, v, l1[t * V + v],
                            l2[t * V + v]);
                    t = T;
                    break;
                }
    }
    digest_bytes(&g_digest, l1, T * V * sizeof(float));
    apus_model_state_free(&s3, &mt);
    apus_model_state_free(&s4, &mt);
    apus_store_close(stp);
    apus_model_free(&mt);
    free(l1); free(l2);
}

/* =========================================================================*/
/* 3. tiered attention res1 stream at T=256 (post_attn hook capture) --------*/

typedef struct {
    uint16_t *buf;      /* [n_layers][T][H] captured res1 rows, by pos */
    size_t H;
    size_t T;
    int n;
    int oor;            /* out-of-range hook positions (debug) */
} Res1Cap;

static void cap_post_attn(void *vctx, int layer, const uint16_t *res1,
                          int64_t pos, int s, int t) {
    (void)s; (void)t;
    Res1Cap *c = vctx;
    if (pos < 0 || (size_t)pos >= c->T) {
        fprintf(stderr, "  HOOK OOR: layer=%d pos=%lld s=%d t=%d\n",
                layer, (long long)pos, s, t);
        c->oor++;
        c->n++;
        return;
    }
    memcpy(c->buf + ((size_t)layer * c->T + (size_t)pos) * c->H, res1,
           c->H * 2);
    c->n++;
}

static void test_tiered_attn_stream(size_t T) {
    char err[256];
    int64_t *ids = malloc(T * sizeof(int64_t));
    for (size_t i = 0; i < T; i++) ids[i] = (int64_t)(rng_u64() % 256);
    uint16_t *r1 = NULL, *r2 = NULL;
    float *l = NULL;
    size_t H = 0, V = 0;
    int n_moe = 0, nl = 0;
    for (int run = 0; run < 2; run++) {   /* 0 = batched, 1 = sequential */
        ApusModel mt;
        if (apus_model_load_ex(&mt, "tests/m6a/fixtures/model", 256, 1,
                               err, sizeof err)) {
            fprintf(stderr, "FAIL: tiered load: %s\n", err);
            failures++;
            free(ids);
            return;
        }
        H = (size_t)mt.hidden;
        V = (size_t)mt.vocab;
        nl = mt.n_layers;
        n_moe = nl;   /* every layer is MoE in this model (M4) */
        if (!r1) {
            /* indexed by ABSOLUTE layer (the dense layer never fires —
             * its rows stay zero) */
            r1 = calloc(T * (size_t)nl * H, 2);
            r2 = calloc(T * (size_t)nl * H, 2);
            l = malloc(V * sizeof(float));
        }
        ApusStoreCfg sc = {0};
        sc.n_layers = mt.n_layers;
        sc.n_experts = (int)mt.layers[0].lc.experts;
        sc.slots_per_layer = (int)mt.layers[0].lc.experts;
        sc.cache_bytes = 1;
        sc.pin_bytes = 1;
        sc.io_threads = -1;
        sc.usage_path = "";
        ApusStore *stp = apus_store_open("tests/m6a/fixtures/model", &sc,
                                         err, sizeof err);
        CHECK(stp != NULL, "store open: %s", err);
        if (!stp) { free(ids); free(r1); free(r2); free(l); return; }
        apus_model_attach_store(&mt, stp);
        Res1Cap cap = { run == 0 ? r1 : r2, H, T, 0, 0 };
        ApusStoreFwdHooks hk = { &cap, cap_post_attn, NULL };
        apus_store_fwd_hooks(stp, &hk);
        ApusModelState s;
        apus_model_state_init(&s, &mt);
        if (run == 0) {
            apus_model_forward(&mt, &s, ids, T, l, 0, NULL);
        } else {
            for (size_t t = 0; t < T; t++)
                apus_model_forward(&mt, &s, ids + t, 1, l, 0, NULL);
        }
        CHECK(cap.n == (int)(T * (size_t)n_moe),
              "tiered stream run %d: %d res1 captures != %zu", run, cap.n,
              T * (size_t)n_moe);
        apus_model_state_free(&s, &mt);
        apus_store_close(stp);
        apus_model_free(&mt);
    }
    /* The Qwen M9 restoration runs phase B through gemm_hot ONLY (the
     * M-independent-bitwise class — NO BLAS/gemm_fast anywhere), so the
     * batched path is bitwise the sequential body at EVERY T on every
     * platform: the full res1 stream compares bitwise at T=256 too. */
    if (r1) {
        int same = memcmp(r1, r2, T * (size_t)nl * H * 2) == 0;
        CHECK(same,
              "tiered attention res1 stream T=%zu: batched != sequential",
              T);
        if (!same) {
            for (int L = 0; L < nl; L++)
                for (size_t t = 0; t < T; t++) {
                    size_t cnt = 0, first = 0;
                    for (size_t i = 0; i < H; i++)
                        if (r1[((size_t)L * T + t) * H + i]
                            != r2[((size_t)L * T + t) * H + i]) {
                            if (!cnt) first = i;
                            cnt++;
                        }
                    if (cnt) {
                        fprintf(stderr, "  DIFF stream T=%zu: L=%d t=%zu: "
                                "%zu/%zu ch differ, first i=%zu: "
                                "%04x vs %04x; next:", T, L, t, cnt, H,
                                first,
                                r1[((size_t)L * T + t) * H + first],
                                r2[((size_t)L * T + t) * H + first]);
                        size_t shown = 0;
                        for (size_t i = first + 1; i < H && shown < 3;
                             i++)
                            if (r1[((size_t)L * T + t) * H + i]
                                != r2[((size_t)L * T + t) * H + i]) {
                                fprintf(stderr, " i=%zu %04x vs %04x;", i,
                                        r1[((size_t)L * T + t) * H + i],
                                        r2[((size_t)L * T + t) * H + i]);
                                shown++;
                            }
                        fprintf(stderr, "\n");
                        L = nl;
                        break;
                    }
                }
        }
        printf("  tiered attn stream T=%-4zu: res1 BITWISE (%d MoE "
               "layers)\n", T, n_moe);
        digest_bytes(&g_digest, r1, T * (size_t)nl * H * 2);
    }
    free(r1); free(r2); free(l); free(ids);
}

int main(void) {
    printf("test_m9c: batched-prefill restoration gates (Qwen M9)\n");
    fprintf(stderr, "  pool threads: %d\n", apus_pool_threads());
    test_attn_batch_kind(APUS_LAYER_GDN, 256, "gdn");
    test_attn_batch_kind(APUS_LAYER_FULL, 256, "full");
    test_attn_batch_kind(APUS_LAYER_GDN, 300, "gdn-x");
    test_attn_batch_kind(APUS_LAYER_FULL, 300, "full-x");
    test_attn_batch_kind(APUS_LAYER_GDN, 64, "gdn64");
    test_model_prefill();
    test_tiered_attn_stream(64);
    test_tiered_attn_stream(256);
    printf("  digest %016llx\n", (unsigned long long)g_digest);
    printf("test_m9c: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
