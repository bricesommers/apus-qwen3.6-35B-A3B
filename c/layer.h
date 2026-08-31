/*
 * c/layer.h — ONE Qwen3.6-35B-A3B decoder layer, assembled from the M4
 * kernels (milestone M4c). Replaces the Ling KDA/MLA wiring. C11,
 * libc/libm only; projections via apus_bf16_gemv_hot (decode) /
 * apus_bf16_gemm_hot (M9 batched prefill — the M-independent-bitwise
 * ILP GEMM; the ARM hot dispatch is the user-approved M9b ILP reorder
 * class, absorbed by the m4c envelope). Covers the two layer kinds of
 * docs/M4-CONTRACT.md (layer_types = [linear,linear,linear,full]x10):
 *
 *   gdn  : Gated DeltaNet linear attention + MoE   ("linear_attention",
 *          30 of the real 40 layers)
 *   full : gated GQA full attention + MoE          ("full_attention",
 *          layers 3,7,...,39)
 *
 * MoE on EVERY layer (no dense layers in this model).
 *
 * Wiring (HF:911-954, contract §3-5; pre-norm, plain bf16 residuals):
 *
 *   res1 = rnd(x + mixer(rmsnorm(x, ln1_w)))       (bf16 residual adds)
 *   out  = rnd(res1 + moe(rmsnorm(res1, ln2_w)))
 *
 *   GDN mixer (per token): qkv = rnd(W_qkv.ln1) [q 2*hk*dk | k | v
 *   hv*dv fused] -> ONE fused depthwise causal conv1d k=4 + SiLU
 *   (state, two rounding points); z = rnd(W_z.ln1); b = rnd(W_b.ln1);
 *   a = rnd(W_a.ln1); beta = rnd(sigmoid(b)) (bf16); g =
 *   -exp(A_log)*softplus(a+dt_bias) (fp32); q,k l2norm fp32 over the 16
 *   K heads (q scaled dk^-0.5) then repeat_interleave x2 -> 32 V heads
 *   (bitwise: norming the duplicates gives the same values); delta-rule
 *   step (state S fp32 [hv,dk,dv]); ob = rnd(rec_o); onorm =
 *   RMSNormGated(ob, z, direct w); attn_out = rnd(W_out.on). No RoPE.
 *
 *   FULL mixer (per token, position pos): qg = rnd(W_q.ln1) viewed
 *   [nh, 2d] -> per-head [q(d) | gate(d)]; q,k per-head (1+w) RMSNorm
 *   BEFORE RoPE; partial NeoX RoPE on the first rot dims (theta from
 *   cfg); k,v appended to the cache [pos, nkv, d]; eager GQA attention
 *   vs the cache (fp32 softmax, kv head h/(nh/nkv)); elementwise
 *   sigmoid output gate (bf16); attn_out = rnd(W_o.gated).
 *
 *   MoE FFN (per token): router(ln2) = rnd(ln2.W_gate) -> FP32 softmax
 *   -> top-k (lowest index on ties) -> renormalize -> BF16 weights;
 *   per selected expert gu = rnd(W_gu_e.ln2) (fused [gate|up]),
 *   a = rnd(silu(gate)).up, y_e = rnd(W_d_e.a); routed = rnd(sum
 *   w_e.y_e) (fp32 accum, single round — the oracle's documented
 *   realization class); shared expert silu-MLP (single-round act) on
 *   the SAME ln2 input, gated by rnd(sigmoid(rnd(W_shgate.ln2)));
 *   ffn = rnd(routed + shared).
 *
 * Decode == prefill BY CONSTRUCTION: apus_layer_forward processes every
 * token through the single per-token body (projections are per-row
 * GEMVs, the M4 conv/recurrence/attention kernels are the
 * state-carrying decode forms), so a T-token prefill is BITWISE
 * identical to any chunking into prefill + single-token decodes (the
 * load-bearing M5 property, gated in tests/m4c).
 *
 * apus_layer_forward_hot (T>1) is the M9 RESTORED batched prefill — the
 * Ling M6c/M9c structure re-expressed on the Qwen kernels, STRICTLY
 * inside the M-independent-bitwise class (no new reorder class, NO
 * BLAS/gemm_fast anywhere: every batched linear runs at M=T through
 * apus_bf16_gemm_hot, whose M-independence — every GEMM row bitwise the
 * GEMV row — is gated in tests/m9b):
 *
 *   phase A (chunked at APUS_PREFILL_ATTN_CHUNK): the attention
 *   projections (GDN w_qkv/w_z/w_b/w_a/w_out, FULL wq/wk/wv/wo) batch
 *   at M=tc; the GDN conv runs as ONE apus_gdn_conv1d over the chunk
 *   (contract (a): bitwise the step loop); the FULL attention runs as
 *   ONE apus_attn_gqa_mt over the chunk after the cache appends (each
 *   (t,h) unit is the decode unit's identical body). The sequential
 *   state parts (beta/decay/l2norm/repeat/delta step/onorm, qk-norm/
 *   RoPE/cache append/outgate) stay per token, ascending t.
 *
 *   phase B (full T): router per token (unchanged), the shared expert
 *   linears at M=T, and each UNIQUE routed expert's fused gate_up/down
 *   linears ONCE at M=count over the gathered rows (gather/scatter are
 *   exact copies; the per-token combine keeps the ascending slot
 *   order). The sigmoid shared-gate and per-token residuals are
 *   per-element.
 *
 * Every per-output bit is identical to T sequential per-token bodies —
 * the batched path is gated BITWISE vs the sequential body in
 * tests/m9c (traces AND state at T=256/300 across the chunk boundary,
 * model level eager + tiered at T=64). The Accelerate BLAS dispatch
 * stays a measured-but-unconsumed class (tests/m9b): wiring it into
 * phase A/B would leave the pinned realization at M>=128, breaking the
 * prefill == decode bitwise property there.
 *
 * Verified stage-by-stage against the numpy oracle (tools/oracle.py,
 * fixtures tests/m4b) in tests/m4c — C must land inside the oracle's
 * f32-vs-f64 envelope per stage.
 *
 * Usage: #define APUS_LAYER_IMPLEMENTATION in exactly one TU; that TU
 * must also define APUS_BF16/GDN/ATTN/MOE_IMPLEMENTATION.
 */
#ifndef APUS_LAYER_H
#define APUS_LAYER_H

#include <stddef.h>
#include <stdint.h>

#include "bf16.h"
#include "gdn.h"
#include "attn.h"
#include "moe.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    APUS_LAYER_GDN  = 0,   /* Gated DeltaNet + MoE (linear_attention) */
    APUS_LAYER_FULL = 1,   /* gated GQA + MoE       (full_attention)  */
} ApusLayerKind;

typedef struct {
    ApusLayerKind kind;
    size_t hidden;          /* H (2048) */
    /* GDN attention */
    size_t gdn_hk;          /* K heads (16) */
    size_t gdn_hv;          /* V heads (32); repeat = hv/hk */
    size_t gdn_dk;          /* key head dim (128) */
    size_t gdn_dv;          /* value head dim (128) */
    /* FULL attention */
    size_t attn_nh;         /* q heads (16) */
    size_t attn_nkv;        /* KV heads (2) */
    size_t attn_d;          /* head dim (256) */
    size_t attn_rot;        /* rotary dims (64 = 0.25 * d) */
    double rope_theta;      /* 1e7 */
    size_t max_seq;         /* KV cache capacity (tokens) */
    /* MoE (every layer) */
    size_t experts, moe_inter, shared_inter;
    size_t top_k;           /* 8 */
} ApusLayerCfg;

typedef struct {
    const uint16_t *ln1_w, *ln2_w;              /* [H] (1+w) */
    /* GDN */
    const uint16_t *w_qkv;                      /* [conv_dim, H] */
    const uint16_t *w_z;                        /* [value_dim, H] */
    const uint16_t *w_b, *w_a;                  /* [hv, H] */
    const uint16_t *conv_w;                     /* [conv_dim, 4] */
    const float    *A_log, *dt_bias;            /* [hv] F32 */
    const uint16_t *onorm_w;                    /* [dv] DIRECT weight
                                                 * (shared across heads) */
    const uint16_t *w_out;                      /* [H, value_dim] */
    /* FULL */
    const uint16_t *wq;                         /* [nh*2d, H] ([q|gate]) */
    const uint16_t *wk, *wv;                    /* [nkv*d, H] */
    const uint16_t *wo;                         /* [H, nh*d] */
    const uint16_t *qn_w, *kn_w;                /* [d] (1+w) */
    /* MoE */
    const uint16_t *rtr_w;                      /* [E, H] */
    const uint16_t *exp_gu;                     /* [E, 2I, H] FUSED
                                                 * gate_up */
    const uint16_t *exp_d;                      /* [E, H, I] */
    const uint16_t *sh_g, *sh_u;                /* [Is, H] */
    const uint16_t *sh_d;                       /* [H, Is] */
    const uint16_t *sh_gate;                    /* [1, H] */
} ApusLayerW;

typedef struct {
    /* GDN (conv: last 3 pre-conv inputs per channel; S fp32) */
    uint16_t *conv_state;                       /* [conv_dim*3] */
    float    *S;                                /* [hv*dk*dv] */
    /* FULL */
    uint16_t *kcache;                           /* [max_seq,nkv,d] */
    uint16_t *vcache;                           /* [max_seq,nkv,d] */
    size_t    pos;                              /* tokens processed */
} ApusLayerState;

/* Named-intermediate capture (all optional; NULL = skip). Arrays are
 * [T, ...] row-major, one row per token, filled by apus_layer_forward.
 * dtypes: bf16 codes unless marked f32/i32. Shapes per the Cfg. */
typedef struct {
    uint16_t *ln1, *attn_out, *res1, *ln2, *out;    /* [T,H] */
    /* GDN */
    uint16_t *qkv_conv;                         /* [T,conv_dim] */
    uint16_t *beta;                             /* [T,hv] */
    float    *gdecay;                           /* [T,hv] f32 */
    uint16_t *rec_o;                            /* [T,hv*dv] (post-rnd) */
    uint16_t *onorm;                            /* [T,value_dim] */
    /* FULL */
    uint16_t *qf, *kf;                          /* [T,nh*d] / [T,nkv*d]
                                                 * (post-norm,post-rope) */
    uint16_t *attno;                            /* [T,nh*d] */
    uint16_t *mgate;                            /* [T,nh*d] */
    /* MoE */
    int32_t  *rtr_idx;                          /* [T,top_k] */
    uint16_t *rtr_w;                            /* [T,top_k] */
    uint16_t *moe_routed, *moe_shared, *moe_out;    /* [T,H] */
} ApusLayerTrace;

/* Zero the state (conv state, S, pos; caches need no init). */
void apus_layer_state_zero(const ApusLayerCfg *c, ApusLayerState *st);

/* One layer over T tokens. x, out: [T,H] bf16. Decode is T=1 with
 * carried state; prefill is BITWISE identical to any chunking (see the
 * header contract). tr may be NULL. */
void apus_layer_forward(const ApusLayerCfg *c, const ApusLayerW *w,
                        ApusLayerState *st, const uint16_t *x,
                        uint16_t *out, size_t T, ApusLayerTrace *tr);

/* Hot dispatcher: T=1 (decode) runs the sequential per-token body;
 * T>1 (prefill) runs the M9 batched prefill below — BITWISE identical
 * to the sequential body at every T by construction (see the header
 * contract; gated in tests/m9c across the chunk boundary). */
void apus_layer_forward_hot(const ApusLayerCfg *c, const ApusLayerW *w,
                            ApusLayerState *st, const uint16_t *x,
                            uint16_t *out, size_t T, ApusLayerTrace *tr);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_LAYER_IMPLEMENTATION

#include <math.h>
#include <string.h>

void apus_layer_state_zero(const ApusLayerCfg *c, ApusLayerState *st) {
    if (c->kind == APUS_LAYER_GDN) {
        size_t cd = 2 * c->gdn_hk * c->gdn_dk + c->gdn_hv * c->gdn_dv;
        memset(st->conv_state, 0, cd * 3 * sizeof(uint16_t));
        memset(st->S, 0, c->gdn_hv * c->gdn_dk * c->gdn_dv *
               sizeof(float));
    }
    st->pos = 0;
}

/* bf16 residual add: out[i] = rnd(f32(a[i]) + f32(b[i])) */
static void apus_layer_resadd(const uint16_t *a, const uint16_t *b,
                              uint16_t *out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = apus_bf16_bits(apus_bf16_f32(a[i]) + apus_bf16_f32(b[i]));
}

static void apus_layer_gdn_attn(const ApusLayerCfg *c, const ApusLayerW *w,
                                ApusLayerState *st, const uint16_t *ln1,
                                uint16_t *ao, ApusLayerTrace *tr,
                                size_t t) {
    size_t H = c->hidden;
    size_t hk = c->gdn_hk, hv = c->gdn_hv, dk = c->gdn_dk, dv = c->gdn_dv;
    size_t key_dim = hk * dk, value_dim = hv * dv;
    size_t conv_dim = 2 * key_dim + value_dim;
    float qscale = (float)pow((double)dk, -0.5);
    uint16_t qkv[conv_dim], z[value_dim], b[hv], a[hv], qc[conv_dim];
    uint16_t beta_b[hv], ob[value_dim], on[value_dim];
    float g[hv], reco[value_dim];
    float qn[hk * dk], kn[hk * dk], qr[hv * dk], kr[hv * dk];
    apus_bf16_gemv_hot(w->w_qkv, ln1, qkv, conv_dim, H);
    apus_bf16_gemv_hot(w->w_z, ln1, z, value_dim, H);
    apus_bf16_gemv_hot(w->w_b, ln1, b, hv, H);
    apus_bf16_gemv_hot(w->w_a, ln1, a, hv, H);
    apus_gdn_conv1d_step(qkv, w->conv_w, qc, conv_dim, st->conv_state);
    apus_gdn_beta(b, beta_b, hv);
    apus_gdn_decay(a, w->A_log, w->dt_bias, g, hv);
    /* l2norm over the 16 K heads (fp32 out, q pre-scaled), then
     * repeat_interleave x rep into the 32 V heads — bitwise identical to
     * norming the duplicated heads (the norm is per-head deterministic) */
    apus_gdn_l2norm(qc, qn, hk, dk, qscale);
    apus_gdn_l2norm(qc + key_dim, kn, hk, dk, 1.0f);
    size_t rep = hv / hk;
    for (size_t h = 0; h < hk; h++)
        for (size_t r = 0; r < rep; r++) {
            memcpy(qr + (h * rep + r) * dk, qn + h * dk,
                   dk * sizeof(float));
            memcpy(kr + (h * rep + r) * dk, kn + h * dk,
                   dk * sizeof(float));
        }
    apus_gdn_step_mt(st->S, qr, kr, qc + 2 * key_dim, g, beta_b, reco,
                     hv, dk, dv);
    for (size_t i = 0; i < value_dim; i++)
        ob[i] = apus_bf16_bits(reco[i]);          /* .to(bf16), HF:454 */
    apus_gdn_onorm(ob, z, w->onorm_w, on, hv, dv);
    apus_bf16_gemv_hot(w->w_out, on, ao, H, value_dim);
    if (tr) {
        if (tr->qkv_conv) memcpy(tr->qkv_conv + t * conv_dim, qc,
                                 conv_dim * 2);
        if (tr->beta)  memcpy(tr->beta + t * hv, beta_b, hv * 2);
        if (tr->gdecay) memcpy(tr->gdecay + t * hv, g,
                               hv * sizeof(float));
        if (tr->rec_o) memcpy(tr->rec_o + t * value_dim, ob,
                              value_dim * 2);
        if (tr->onorm) memcpy(tr->onorm + t * value_dim, on,
                              value_dim * 2);
    }
}

static void apus_layer_full_attn(const ApusLayerCfg *c,
                                 const ApusLayerW *w, ApusLayerState *st,
                                 const uint16_t *ln1, uint16_t *ao,
                                 ApusLayerTrace *tr, size_t t) {
    size_t H = c->hidden, nh = c->attn_nh, nkv = c->attn_nkv;
    size_t d = c->attn_d, rot = c->attn_rot, pos = st->pos;
    float scale = (float)(1.0 / sqrt((double)d));
    uint16_t qg[nh * 2 * d], qn[nh * d], k[nkv * d], kn[nkv * d];
    uint16_t v[nkv * d], gate[nh * d], attno[nh * d], og[nh * d];
    apus_bf16_gemv_hot(w->wq, ln1, qg, nh * 2 * d, H);
    apus_bf16_gemv_hot(w->wk, ln1, k, nkv * d, H);
    apus_bf16_gemv_hot(w->wv, ln1, v, nkv * d, H);
    /* per-head [q|gate] split; (1+w) RMSNorm BEFORE RoPE */
    for (size_t h = 0; h < nh; h++) {
        apus_attn_rmsnorm(qg + h * 2 * d, w->qn_w, qn + h * d, d);
        memcpy(gate + h * d, qg + h * 2 * d + d, d * 2);
    }
    for (size_t h = 0; h < nkv; h++)
        apus_attn_rmsnorm(k + h * d, w->kn_w, kn + h * d, d);
    /* partial NeoX RoPE at this token's position */
    for (size_t h = 0; h < nh; h++)
        apus_attn_rope(qn + h * d, qn + h * d, d, rot, (float)pos,
                       c->rope_theta);
    for (size_t h = 0; h < nkv; h++)
        apus_attn_rope(kn + h * d, kn + h * d, d, rot, (float)pos,
                       c->rope_theta);
    /* cache append, then decode attention over rows 0..pos */
    memcpy(st->kcache + pos * nkv * d, kn, nkv * d * 2);
    memcpy(st->vcache + pos * nkv * d, v, nkv * d * 2);
    apus_attn_gqa_decode_mt(qn, st->kcache, st->vcache, attno, pos + 1,
                            nh, nkv, d, scale);
    apus_attn_outgate(attno, gate, og, nh * d);
    apus_bf16_gemv_hot(w->wo, og, ao, H, nh * d);
    if (tr) {
        if (tr->qf) memcpy(tr->qf + t * nh * d, qn, nh * d * 2);
        if (tr->kf) memcpy(tr->kf + t * nkv * d, kn, nkv * d * 2);
        if (tr->attno) memcpy(tr->attno + t * nh * d, attno, nh * d * 2);
        if (tr->mgate) {
            for (size_t i = 0; i < nh * d; i++)
                tr->mgate[t * nh * d + i] = apus_bf16_bits(
                    1.0f / (1.0f + expf(-apus_bf16_f32(gate[i]))));
        }
    }
}

static void apus_layer_moe(const ApusLayerCfg *c, const ApusLayerW *w,
                           const uint16_t *ln2, uint16_t *moe_out,
                           ApusLayerTrace *tr, size_t t) {
    size_t H = c->hidden, E = c->experts;
    size_t I = c->moe_inter, Is = c->shared_inter, TK = c->top_k;
    (void)E;
    int32_t idx[TK];
    uint16_t rw[TK];
    uint16_t gu[2 * I], a[I], ye[TK * H], sg[Is], su[Is], sa[Is], sy[H];
    uint16_t routed[H], sgl[1], sgat[1], shared[H];
    apus_moe_route(ln2, w->rtr_w, idx, rw, E, H, TK);
    for (size_t i = 0; i < TK; i++) {
        size_t e = (size_t)idx[i];
        apus_bf16_gemv_hot(w->exp_gu + e * 2 * I * H, ln2, gu, 2 * I, H);
        apus_moe_silu_act(gu, a, I);
        apus_bf16_gemv_hot(w->exp_d + e * H * I, a, ye + i * H, H, I);
    }
    apus_moe_combine(ye, rw, routed, TK, H);
    apus_bf16_gemv_hot(w->sh_g, ln2, sg, Is, H);
    apus_bf16_gemv_hot(w->sh_u, ln2, su, Is, H);
    apus_moe_silu_mul(sg, su, sa, Is);
    apus_bf16_gemv_hot(w->sh_d, sa, sy, H, Is);
    apus_bf16_gemv_hot(w->sh_gate, ln2, sgl, 1, H);
    sgat[0] = apus_bf16_bits(
        1.0f / (1.0f + expf(-apus_bf16_f32(sgl[0]))));
    for (size_t i = 0; i < H; i++)
        shared[i] = apus_bf16_bits(apus_bf16_f32(sy[i]) *
                                   apus_bf16_f32(sgat[0]));
    apus_layer_resadd(routed, shared, moe_out, H);
    if (tr) {
        if (tr->rtr_idx) memcpy(tr->rtr_idx + t * TK, idx,
                                TK * sizeof(int32_t));
        if (tr->rtr_w) memcpy(tr->rtr_w + t * TK, rw, TK * 2);
        if (tr->moe_routed) memcpy(tr->moe_routed + t * H, routed, H * 2);
        if (tr->moe_shared) memcpy(tr->moe_shared + t * H, shared, H * 2);
        if (tr->moe_out) memcpy(tr->moe_out + t * H, moe_out, H * 2);
    }
}

/* One token through the whole layer; every path funnels here so prefill
 * and decode share the numerics by construction. */
static void apus_layer_token(const ApusLayerCfg *c, const ApusLayerW *w,
                             ApusLayerState *st, const uint16_t *x,
                             uint16_t *out, ApusLayerTrace *tr, size_t t) {
    size_t H = c->hidden;
    uint16_t ln1[H], ao[H], res1[H], ln2[H], moe_out[H];
    apus_attn_rmsnorm(x, w->ln1_w, ln1, H);
    if (c->kind == APUS_LAYER_FULL)
        apus_layer_full_attn(c, w, st, ln1, ao, tr, t);
    else
        apus_layer_gdn_attn(c, w, st, ln1, ao, tr, t);
    st->pos++;
    apus_layer_resadd(x, ao, res1, H);
    apus_attn_rmsnorm(res1, w->ln2_w, ln2, H);
    apus_layer_moe(c, w, ln2, moe_out, tr, t);
    apus_layer_resadd(res1, moe_out, out, H);
    if (tr) {
        if (tr->ln1) memcpy(tr->ln1 + t * H, ln1, H * 2);
        if (tr->attn_out) memcpy(tr->attn_out + t * H, ao, H * 2);
        if (tr->res1) memcpy(tr->res1 + t * H, res1, H * 2);
        if (tr->ln2) memcpy(tr->ln2 + t * H, ln2, H * 2);
        if (tr->out) memcpy(tr->out + t * H, out, H * 2);
    }
}

void apus_layer_forward(const ApusLayerCfg *c, const ApusLayerW *w,
                        ApusLayerState *st, const uint16_t *x,
                        uint16_t *out, size_t T, ApusLayerTrace *tr) {
    for (size_t t = 0; t < T; t++)
        apus_layer_token(c, w, st, x + t * c->hidden,
                         out + t * c->hidden, tr, t);
}

/* =========================================================================*/
/* M9 batched prefill (T>1) — the Ling M6c/M9c structure on the Qwen
 * kernels, STRICTLY inside the M-independent-bitwise class: every batched
 * linear runs at M=T through apus_bf16_gemm_hot (every GEMM row bitwise
 * the GEMV row — tests/m9b's M-independence gate; NO gemm_fast/BLAS, so
 * no M leaves the pinned realization). BITWISE identical to T sequential
 * apus_layer_token calls: gather/scatter are exact copies, the batched
 * conv1d is contract-(a) bitwise the step loop, the chunked gqa_mt runs
 * each (t,h) unit as the decode unit's identical body, the per-token
 * combine keeps the ascending slot order, acts/residuals are per-element.
 * tests/m9c gates traces AND state bitwise at T=256/300 (crossing the
 * chunk boundary) and at the model level (eager + tiered, T=64). */

/* Phase-A chunk: bounds the scratch arena on long prompts. Every
 * per-output bit is chunk-size independent (GEMM M-independence + the
 * state carrying across chunks), so any value is safe; 256 keeps the
 * chunk working set in the tens of MB at the real dims. */
#define APUS_PREFILL_ATTN_CHUNK 256u

/* Batched GDN attention over tc tokens. ln1 [tc,H] -> ao [tc,H].
 * t0 = chunk's first token index (trace rows only). */
static void apus_layer_attn_batch_gdn(const ApusLayerCfg *c,
                                      const ApusLayerW *w,
                                      ApusLayerState *st,
                                      const uint16_t *ln1, uint16_t *ao,
                                      size_t tc, ApusLayerTrace *tr,
                                      size_t t0) {
    size_t H = c->hidden;
    size_t hk = c->gdn_hk, hv = c->gdn_hv, dk = c->gdn_dk, dv = c->gdn_dv;
    size_t key_dim = hk * dk, value_dim = hv * dv;
    size_t conv_dim = 2 * key_dim + value_dim;
    float qscale = (float)pow((double)dk, -0.5);
    ApusScratchMark mk = apus_scratch_mark();
    uint16_t *qkv = apus_scratch_alloc(tc * conv_dim * 2);
    uint16_t *z   = apus_scratch_alloc(tc * value_dim * 2);
    uint16_t *b   = apus_scratch_alloc(tc * hv * 2);
    uint16_t *a   = apus_scratch_alloc(tc * hv * 2);
    uint16_t *qc  = apus_scratch_alloc(tc * conv_dim * 2);
    uint16_t *on  = apus_scratch_alloc(tc * value_dim * 2);
    /* projections at M=tc (BITWISE the per-token gemv_hot rows) */
    apus_bf16_gemm_hot(w->w_qkv, ln1, qkv, tc, conv_dim, H);
    apus_bf16_gemm_hot(w->w_z, ln1, z, tc, value_dim, H);
    apus_bf16_gemm_hot(w->w_b, ln1, b, tc, hv, H);
    apus_bf16_gemm_hot(w->w_a, ln1, a, tc, hv, H);
    /* ONE batched conv over the chunk (contract (a): bitwise the step
     * loop; the state carries) */
    apus_gdn_conv1d(qkv, w->conv_w, qc, conv_dim, tc, st->conv_state);
    /* sequential state parts, per token in ascending t */
    for (size_t t = 0; t < tc; t++) {
        uint16_t beta_b[hv], ob[value_dim];
        float g[hv], reco[value_dim];
        float qn[hk * dk], kn[hk * dk], qr[hv * dk], kr[hv * dk];
        apus_gdn_beta(b + t * hv, beta_b, hv);
        apus_gdn_decay(a + t * hv, w->A_log, w->dt_bias, g, hv);
        apus_gdn_l2norm(qc + t * conv_dim, qn, hk, dk, qscale);
        apus_gdn_l2norm(qc + t * conv_dim + key_dim, kn, hk, dk, 1.0f);
        size_t rep = hv / hk;
        for (size_t h = 0; h < hk; h++)
            for (size_t r = 0; r < rep; r++) {
                memcpy(qr + (h * rep + r) * dk, qn + h * dk,
                       dk * sizeof(float));
                memcpy(kr + (h * rep + r) * dk, kn + h * dk,
                       dk * sizeof(float));
            }
        apus_gdn_step_mt(st->S, qr, kr,
                         qc + t * conv_dim + 2 * key_dim, g, beta_b,
                         reco, hv, dk, dv);
        for (size_t i = 0; i < value_dim; i++)
            ob[i] = apus_bf16_bits(reco[i]);      /* .to(bf16), HF:454 */
        apus_gdn_onorm(ob, z + t * value_dim, w->onorm_w,
                       on + t * value_dim, hv, dv);
        if (tr) {
            if (tr->qkv_conv) memcpy(tr->qkv_conv + (t0 + t) * conv_dim,
                                     qc + t * conv_dim, conv_dim * 2);
            if (tr->beta)  memcpy(tr->beta + (t0 + t) * hv, beta_b,
                                  hv * 2);
            if (tr->gdecay) memcpy(tr->gdecay + (t0 + t) * hv, g,
                                   hv * sizeof(float));
            if (tr->rec_o) memcpy(tr->rec_o + (t0 + t) * value_dim, ob,
                                  value_dim * 2);
            if (tr->onorm) memcpy(tr->onorm + (t0 + t) * value_dim,
                                  on + t * value_dim, value_dim * 2);
        }
    }
    apus_bf16_gemm_hot(w->w_out, on, ao, tc, H, value_dim);
    apus_scratch_reset(mk);
}

/* Batched FULL attention over tc tokens. ln1 [tc,H] -> ao [tc,H].
 * The cache rows are appended BEFORE the single gqa_mt call: query row t
 * attends keys 0..(pos0+t) exactly as the per-token decode path (each
 * (t,h) unit is the decode unit's identical body). */
static void apus_layer_attn_batch_full(const ApusLayerCfg *c,
                                       const ApusLayerW *w,
                                       ApusLayerState *st,
                                       const uint16_t *ln1, uint16_t *ao,
                                       size_t tc, ApusLayerTrace *tr,
                                       size_t t0) {
    size_t H = c->hidden, nh = c->attn_nh, nkv = c->attn_nkv;
    size_t d = c->attn_d, rot = c->attn_rot, pos0 = st->pos;
    float scale = (float)(1.0 / sqrt((double)d));
    ApusScratchMark mk = apus_scratch_mark();
    uint16_t *qg = apus_scratch_alloc(tc * nh * 2 * d * 2);
    uint16_t *kb = apus_scratch_alloc(tc * nkv * d * 2);
    uint16_t *vb = apus_scratch_alloc(tc * nkv * d * 2);
    uint16_t *qn = apus_scratch_alloc(tc * nh * d * 2);
    uint16_t *gate = apus_scratch_alloc(tc * nh * d * 2);
    uint16_t *attno = apus_scratch_alloc(tc * nh * d * 2);
    uint16_t *og = apus_scratch_alloc(tc * nh * d * 2);
    apus_bf16_gemm_hot(w->wq, ln1, qg, tc, nh * 2 * d, H);
    apus_bf16_gemm_hot(w->wk, ln1, kb, tc, nkv * d, H);
    apus_bf16_gemm_hot(w->wv, ln1, vb, tc, nkv * d, H);
    /* per-token per-head (1+w) norms, RoPE at each token's own position,
     * gate split, cache append (all per-row, ascending t) */
    for (size_t t = 0; t < tc; t++) {
        size_t pos = pos0 + t;
        uint16_t kn[nkv * d];
        for (size_t h = 0; h < nh; h++) {
            apus_attn_rmsnorm(qg + (t * nh + h) * 2 * d, w->qn_w,
                              qn + (t * nh + h) * d, d);
            memcpy(gate + (t * nh + h) * d,
                   qg + (t * nh + h) * 2 * d + d, d * 2);
        }
        for (size_t h = 0; h < nkv; h++)
            apus_attn_rmsnorm(kb + (t * nkv + h) * d, w->kn_w,
                              kn + h * d, d);
        for (size_t h = 0; h < nh; h++)
            apus_attn_rope(qn + (t * nh + h) * d, qn + (t * nh + h) * d,
                           d, rot, (float)pos, c->rope_theta);
        for (size_t h = 0; h < nkv; h++)
            apus_attn_rope(kn + h * d, kn + h * d, d, rot, (float)pos,
                           c->rope_theta);
        memcpy(st->kcache + pos * nkv * d, kn, nkv * d * 2);
        memcpy(st->vcache + pos * nkv * d, vb + t * nkv * d, nkv * d * 2);
        if (tr) {
            if (tr->qf) memcpy(tr->qf + (t0 + t) * nh * d,
                               qn + t * nh * d, nh * d * 2);
            if (tr->kf) memcpy(tr->kf + (t0 + t) * nkv * d, kn,
                               nkv * d * 2);
            if (tr->mgate) {
                for (size_t i = 0; i < nh * d; i++)
                    tr->mgate[(t0 + t) * nh * d + i] = apus_bf16_bits(
                        1.0f / (1.0f + expf(-apus_bf16_f32(
                            gate[t * nh * d + i]))));
            }
        }
    }
    apus_attn_gqa_mt(qn, st->kcache, st->vcache, attno, tc, pos0 + tc,
                     nh, nkv, d, scale);
    for (size_t t = 0; t < tc; t++) {
        apus_attn_outgate(attno + t * nh * d, gate + t * nh * d,
                          og + t * nh * d, nh * d);
        if (tr && tr->attno)
            memcpy(tr->attno + (t0 + t) * nh * d, attno + t * nh * d,
                   nh * d * 2);
    }
    apus_bf16_gemm_hot(w->wo, og, ao, tc, H, nh * d);
    apus_scratch_reset(mk);
}

/* Batched MoE FFN over T tokens (phase B). ln2 [T,H] -> moe_out [T,H].
 * Router per token (unchanged); shared expert linears at M=T; each
 * unique routed expert's fused gate_up/down ONCE at M=count over the
 * gathered rows. BITWISE the per-token apus_layer_moe at every row. */
static void apus_layer_moe_batch(const ApusLayerCfg *c,
                                 const ApusLayerW *w, const uint16_t *ln2,
                                 uint16_t *moe_out, size_t T,
                                 ApusLayerTrace *tr) {
    size_t H = c->hidden, E = c->experts;
    size_t I = c->moe_inter, Is = c->shared_inter, TK = c->top_k;
    ApusScratchMark mk = apus_scratch_mark();
    int32_t *idx = apus_scratch_alloc(T * TK * sizeof(int32_t));
    uint16_t *rw = apus_scratch_alloc(T * TK * 2);
    uint16_t *sg = apus_scratch_alloc(T * Is * 2);
    uint16_t *su = apus_scratch_alloc(T * Is * 2);
    uint16_t *sa = apus_scratch_alloc(T * Is * 2);
    uint16_t *sy = apus_scratch_alloc(T * H * 2);
    uint16_t *sgl = apus_scratch_alloc(T * 2);
    uint16_t *routed = apus_scratch_alloc(T * H * 2);
    uint16_t *eo = apus_scratch_alloc(T * TK * H * 2);
    uint16_t *xg = apus_scratch_alloc(T * TK * H * 2);
    uint16_t *gu = apus_scratch_alloc(T * TK * 2 * I * 2);
    uint16_t *a  = apus_scratch_alloc(T * TK * I * 2);
    for (size_t t = 0; t < T; t++)
        apus_moe_route(ln2 + t * H, w->rtr_w, idx + t * TK, rw + t * TK,
                       E, H, TK);
    /* shared expert at M=T (the M-independent GEMM rows) */
    apus_bf16_gemm_hot(w->sh_g, ln2, sg, T, Is, H);
    apus_bf16_gemm_hot(w->sh_u, ln2, su, T, Is, H);
    apus_moe_silu_mul(sg, su, sa, T * Is);
    apus_bf16_gemm_hot(w->sh_d, sa, sy, T, H, Is);
    apus_bf16_gemm_hot(w->sh_gate, ln2, sgl, T, 1, H);
    /* unique experts: each runs gate_up/down ONCE at M=count */
    for (size_t e = 0; e < E; e++) {
        size_t cnt = 0;
        for (size_t t = 0; t < T; t++)
            for (size_t j = 0; j < TK; j++)
                if ((size_t)idx[t * TK + j] == e) {
                    memcpy(xg + cnt * H, ln2 + t * H, H * 2);
                    cnt++;
                }
        if (!cnt) continue;
        apus_bf16_gemm_hot(w->exp_gu + e * 2 * I * H, xg, gu, cnt,
                           2 * I, H);
        for (size_t r = 0; r < cnt; r++)
            apus_moe_silu_act(gu + r * 2 * I, a + r * I, I);
        uint16_t *eb = apus_scratch_alloc(cnt * H * 2);
        apus_bf16_gemm_hot(w->exp_d + e * H * I, a, eb, cnt, H, I);
        size_t r = 0;
        for (size_t t = 0; t < T; t++)
            for (size_t j = 0; j < TK; j++)
                if ((size_t)idx[t * TK + j] == e) {
                    memcpy(eo + (t * TK + j) * H, eb + r * H, H * 2);
                    r++;
                }
    }
    /* per-token combine (ascending slot order) + sigmoid-gated shared +
     * routed/shared residual — the sequential body's exact per-token
     * tail */
    for (size_t t = 0; t < T; t++) {
        uint16_t sgat = apus_bf16_bits(
            1.0f / (1.0f + expf(-apus_bf16_f32(sgl[t]))));
        apus_moe_combine(eo + t * TK * H, rw + t * TK, routed + t * H,
                         TK, H);
        for (size_t i = 0; i < H; i++) {
            uint16_t shared = apus_bf16_bits(
                apus_bf16_f32(sy[t * H + i]) * apus_bf16_f32(sgat));
            moe_out[t * H + i] = apus_bf16_bits(
                apus_bf16_f32(routed[t * H + i])
                + apus_bf16_f32(shared));
            if (tr && tr->moe_shared)
                tr->moe_shared[t * H + i] = shared;
        }
        if (tr) {
            if (tr->rtr_idx) memcpy(tr->rtr_idx + t * TK, idx + t * TK,
                                    TK * sizeof(int32_t));
            if (tr->rtr_w) memcpy(tr->rtr_w + t * TK, rw + t * TK,
                                  TK * 2);
            if (tr->moe_routed) memcpy(tr->moe_routed + t * H,
                                       routed + t * H, H * 2);
            if (tr->moe_out) memcpy(tr->moe_out + t * H,
                                    moe_out + t * H, H * 2);
        }
    }
    apus_scratch_reset(mk);
}

/* Batched prefill: phase A chunked (attention + norms), phase B the
 * batched MoE over the full T, phase C the per-token residual. */
static void apus_layer_prefill(const ApusLayerCfg *c, const ApusLayerW *w,
                               ApusLayerState *st, const uint16_t *x,
                               uint16_t *out, size_t T, ApusLayerTrace *tr) {
    size_t H = c->hidden;
    ApusScratchMark mk = apus_scratch_mark();
    uint16_t *res1 = apus_scratch_alloc(T * H * 2);
    uint16_t *ln2 = apus_scratch_alloc(T * H * 2);
    uint16_t *moe_out = apus_scratch_alloc(T * H * 2);
    for (size_t t0 = 0; t0 < T; t0 += APUS_PREFILL_ATTN_CHUNK) {
        size_t tc = T - t0 < APUS_PREFILL_ATTN_CHUNK
            ? T - t0 : APUS_PREFILL_ATTN_CHUNK;
        ApusScratchMark ckm = apus_scratch_mark();
        uint16_t *ln1 = apus_scratch_alloc(tc * H * 2);
        uint16_t *ao = apus_scratch_alloc(tc * H * 2);
        for (size_t t = 0; t < tc; t++)
            apus_attn_rmsnorm(x + (t0 + t) * H, w->ln1_w, ln1 + t * H, H);
        if (c->kind == APUS_LAYER_FULL)
            apus_layer_attn_batch_full(c, w, st, ln1, ao, tc, tr, t0);
        else
            apus_layer_attn_batch_gdn(c, w, st, ln1, ao, tc, tr, t0);
        st->pos += tc;
        for (size_t t = 0; t < tc; t++) {
            apus_layer_resadd(x + (t0 + t) * H, ao + t * H,
                              res1 + (t0 + t) * H, H);
            apus_attn_rmsnorm(res1 + (t0 + t) * H, w->ln2_w,
                              ln2 + (t0 + t) * H, H);
            if (tr) {
                if (tr->ln1) memcpy(tr->ln1 + (t0 + t) * H, ln1 + t * H,
                                    H * 2);
                if (tr->attn_out) memcpy(tr->attn_out + (t0 + t) * H,
                                         ao + t * H, H * 2);
                if (tr->res1) memcpy(tr->res1 + (t0 + t) * H,
                                     res1 + (t0 + t) * H, H * 2);
                if (tr->ln2) memcpy(tr->ln2 + (t0 + t) * H,
                                    ln2 + (t0 + t) * H, H * 2);
            }
        }
        apus_scratch_reset(ckm);
    }
    apus_layer_moe_batch(c, w, ln2, moe_out, T, tr);
    for (size_t t = 0; t < T; t++) {
        apus_layer_resadd(res1 + t * H, moe_out + t * H, out + t * H, H);
        if (tr && tr->out) memcpy(tr->out + t * H, out + t * H, H * 2);
    }
    apus_scratch_reset(mk);
}

void apus_layer_forward_hot(const ApusLayerCfg *c, const ApusLayerW *w,
                            ApusLayerState *st, const uint16_t *x,
                            uint16_t *out, size_t T, ApusLayerTrace *tr) {
    /* M9: T>1 takes the batched prefill (BITWISE the sequential body —
     * the header contract); T=1 stays on the per-token body. */
    if (T > 1) {
        apus_layer_prefill(c, w, st, x, out, T, tr);
        return;
    }
    apus_layer_forward(c, w, st, x, out, T, tr);
}

#endif /* APUS_LAYER_IMPLEMENTATION */
#endif /* APUS_LAYER_H */
