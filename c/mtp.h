/*
 * c/mtp.h — MTP speculative decoding for Qwen3.6-35B-A3B (M8): the
 * mtp.* block (docs/M4-CONTRACT.md §7) as a draft proposer, plus an
 * exact draft/verify decode loop whose emitted token stream is BITWISE
 * identical to non-speculative decoding for the same seed, greedy and
 * sampled. Machinery adapted from the Ling base's classic-MTP engine
 * (same accept rule, same rollback design).
 * C11, libc + pthreads.
 *
 * The MTP block (HF does NOT implement it — normative: the real
 * checkpoint naming + vLLM qwen3_next_mtp.py). Input pair at position
 * t: h_t = the PRE-final-norm main hidden (apus_model_forward_h) and
 * emb(token_{t+1}); RoPE positions UNSHIFTED (the layer state's own
 * pos):
 *   e  = rmsnorm(emb, pre_fc_norm_embedding)    ((1+w), contract §2)
 *   hh = rmsnorm(h, pre_fc_norm_hidden)         ((1+w); SINGLE norm —
 *        vLLM applies pre_fc_norm_hidden directly to the incoming
 *        hidden, no main-model final norm inside the MTP block)
 *   z  = fc(cat[e, hh])  [2H->H]
 *   residual z -> input_layernorm -> gated GQA (contract §4,
 *   layer_type full_attention) -> +res -> post_attention_layernorm
 *   -> full MoE (contract §5) -> +res -> mtp.norm ((1+w))
 *   -> SHARED lm_head + SHARED embeddings
 *   (mtp_use_dedicated_embeddings=false).
 * The layer body after the fc glue IS a standard FULL decoder layer on
 * input z, so the forward reuses c/layer.h's apus_layer_forward_hot
 * (eager) or c/cache.h's apus_store_layer_forward (tiered; store layer
 * index = the model's n_layers = 40 real) unchanged — numerics
 * identical by construction. Draft chaining (depth-1 approximation,
 * draft quality only — correctness never depends on it): the next
 * pair's h is mtp.norm(block_out), the block's own normed output.
 *
 * The accept rule (the hard invariant): a draft token is accepted iff it
 * equals the main model's OWN pick at that position — argmax for greedy,
 * the main model's own apus_sample draw for sampled. Every emitted token
 * is sampled from the main model's own logits row, consuming exactly one
 * RNG uniform per emitted token in position order; drafts are always
 * argmax and consume no RNG. The verify batch is ONE batched main
 * forward from the carried decode state — bitwise "as if decoded
 * one-by-one" (the M4 per-token-body construction: prefill == decode ==
 * any chunking, gated in tests/m4c). Rollback: partial rejects restore
 * the snapshot and re-feed the accepted prefix in one batched call
 * (bitwise the sequential state by the same property); rejected drafts
 * leave no trace. (Classical rejection sampling is NOT used: it
 * preserves the distribution, not the deterministic per-seed stream;
 * greedy falls out as temp<=0 with zero extra machinery.)
 *
 * The step shape (invariant at step top): main state fed <= q-1; held
 * token x_q (sampled from a valid main row, not yet emitted); drafts
 * z_0..z_{K-2} chained (z_i = candidate for position q+i+1); the MTP
 * CLEAN point contains true pairs through q-1 (the seed pair, whose
 * inputs are always true).
 *   1. Snapshot main + MTP state.
 *   2. Verify batch [x_q, z_0..z_{K-2}] at positions q..q+K-1 -> rows
 *      R[0..K-1] (logits for q+1..q+K) + hiddens H[0..K-1].
 *   3. Walk: emit x_q; for j=1..K-1 sample y from R[j-1]; accept
 *      z_{j-1} iff z_{j-1} == y (emitted at once); stop at the first
 *      mismatch — the sampled replacement becomes the next held token.
 *      Full match: the bonus token from R[K-1] becomes held; main batch
 *      state is kept (it fed exactly the true tokens); the chain state
 *      is clean (every draft emb was accepted).
 *   4. Partial: restore the main snapshot, re-feed batch[0..matched]
 *      (matched+1 tokens) in one batched call. Restore the MTP CLEAN
 *      point and replay the true pairs at positions q..q+matched in one
 *      batched MTP call (pair hiddens H[i] from the verify batch, embs
 *      from the accepted prefix + the replacement) — the last replay row
 *      yields the next seed draft directly. Then chain the remaining
 *      drafts on the MTP block's own hidden (draft-quality only).
 *
 * Snapshot/rollback for the hybrid state: GDN layers carry a full copy
 * of the conv state (last 3 pre-conv inputs per channel) and the fp32
 * recurrent state S; FULL-attention layers (and the MTP layer itself)
 * need only a position rewind — the KV caches are append-only and rows
 * beyond pos are dead, never read. Rejected drafts therefore leave no
 * trace in any state kind.
 *
 * Usage: #define APUS_MTP_IMPLEMENTATION in exactly one TU (needs the
 * model/layer/cache implementations).
 */
#ifndef APUS_MTP_H
#define APUS_MTP_H

#include <stddef.h>
#include <stdint.h>

#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* MTP block weights (zero-copy views into the container + owned eager
 * expert copies when !tiered). */
typedef struct {
    int layer;                    /* store layer index (40 real) */
    ApusLayerCfg lc;              /* full attention + MoE */
    ApusLayerW   lw;
    const uint16_t *enorm_w;      /* [H] pre_fc_norm_embedding ((1+w)) */
    const uint16_t *hnorm_w;      /* [H] pre_fc_norm_hidden   ((1+w)) */
    const uint16_t *fc_w;         /* [H, 2H] */
    const uint16_t *norm_w;       /* [H] mtp.norm             ((1+w)) */
    uint16_t *exp_gu, *exp_d;        /* owned (eager mode only) */
} ApusMtpW;

/* Lazy MTP weight load (only when --spec; config
 * mtp_num_hidden_layers > 0 required). Resolves the mtp.* tensors
 * through the model's ApusStSet — the mtp shard group
 * (apus-qwen-mtp-*) is touched ONLY here, never by a default load.
 * tiered: expert arrays stay NULL and the store (sized n_layers+n_mtp,
 * c/cache.h) serves layer `layer` slabs. */
int  apus_mtp_load(ApusModel *m, ApusMtpW *w, int tiered,
                   char *err, size_t errcap);
void apus_mtp_free(ApusMtpW *w);

/* MTP layer state: an ApusLayerState with gated-GQA KV caches. */
void apus_mtp_state_init(const ApusModel *m, const ApusMtpW *w,
                         ApusLayerState *st);
void apus_mtp_state_free(ApusLayerState *st);

/* One batched MTP step: pairs (h[t] [T,H] bf16, emb_ids[t]) at positions
 * st->pos..+T-1. h is the PRE-final-norm main hidden (or the MTP
 * block's own normed output when draft chaining — same single-norm
 * path either way). logits [T,V] bf16 values widened to f32;
 * hidden_out [T,H] bf16 = mtp.norm(block_out) (the chain input).
 * store may be NULL (eager). */
void apus_mtp_forward(ApusModel *m, const ApusMtpW *w, ApusLayerState *st,
                      ApusStore *store, const uint16_t *h,
                      const int64_t *emb_ids, size_t T,
                      float *logits, uint16_t *hidden_out);

/* --- snapshot / rollback --------------------------------------------------*/

typedef struct {
    int n_layers;
    struct {
        int      kind;
        size_t   pos;
        uint16_t *conv;     /* GDN: conv_dim*3 u16, NULL for FULL */
        float    *S;        /* GDN: hv*dk*dv f32, NULL for FULL */
    } *layers;
    size_t mtp_pos;
    int    model_pos;
} ApusSnap;

void apus_snap_save(const ApusModel *m, const ApusModelState *st,
                    const ApusLayerState *mtp, ApusSnap *sn, int what);
void apus_snap_restore(const ApusModel *m, ApusModelState *st,
                       ApusLayerState *mtp, const ApusSnap *sn, int what);
void apus_snap_free(ApusSnap *sn);

/* --- the draft/verify engine ----------------------------------------------*/

typedef struct {
    uint64_t steps;           /* verify batches */
    uint64_t emitted;         /* tokens emitted */
    uint64_t drafts;          /* draft candidates produced */
    uint64_t accepted;        /* of those, accepted */
    uint64_t full_matches;    /* steps with K-1 accepted + bonus */
    uint64_t re_feeds;        /* partial-reject rollbacks */
} ApusSpecStats;

/* Test hook: when set, drafts come from the hook instead of the MTP
 * argmax (forced acceptance patterns). pos = candidate position,
 * depth = 0..K-2. Return the token id to use. */
typedef int (*ApusDraftOverride)(void *ctx, int64_t pos, int depth);

/* Speculative decode: emits up to max_tokens tokens into out_tokens,
 * BITWISE the non-speculative stream for the same seed (the hard gate).
 * spec_k = verify batch size (2 = one speculative token per batch).
 * Returns the emitted count. */
int apus_spec_run(ApusModel *m, ApusMtpW *mw, ApusModelState *st,
                  ApusStore *store, const int64_t *prompt, size_t n_prompt,
                  int max_tokens, float temp, int top_k, float top_p,
                  uint64_t seed, int spec_k,
                  ApusDraftOverride override, void *ov_ctx,
                  int *out_tokens, ApusSpecStats *stats);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_MTP_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sample.h"

/* --- loader ------------------------------------------------------------------*/

int apus_mtp_load(ApusModel *m, ApusMtpW *w, int tiered,
                  char *err, size_t errcap) {
    memset(w, 0, sizeof *w);
    if (m->n_mtp < 1) {
        snprintf(err, errcap, "mtp: mtp_num_hidden_layers == 0");
        return -1;
    }
    int L = m->n_layers;
    size_t H = (size_t)m->hidden;
    /* dims donor: the first FULL main layer (the MTP layer is a
     * full_attention decoder layer with the same attention/MoE dims —
     * the config carries one attention geometry) */
    const ApusLayerCfg *mc = NULL;
    for (int i = 0; i < m->n_layers; i++)
        if (m->layers[i].lc.kind == APUS_LAYER_FULL) {
            mc = &m->layers[i].lc;
            break;
        }
    if (!mc) {
        snprintf(err, errcap, "mtp: no full_attention main layer");
        return -1;
    }
    w->layer = L;
    ApusLayerCfg *c = &w->lc;
    *c = *mc;
    c->kind = APUS_LAYER_FULL;
    c->max_seq = (size_t)m->max_seq;
    ApusLayerW *lw = &w->lw;
    memset(lw, 0, sizeof *lw);
    char nm[256];
#define MTP_GET16(field, what, ne) do { \
        snprintf(nm, sizeof nm, "%s", what); \
        const ApusStTensor *t = apus_st_set_get(m->set, nm); \
        if (!t || t->dtype != APUS_ST_BF16 \
            || (int64_t)apus_st_nelem(t) != (int64_t)(ne)) { \
            snprintf(err, errcap, "mtp: bad tensor %s", nm); \
            return -1; \
        } \
        field = t->data; \
    } while (0)
    /* real checkpoint names (reference/model.safetensors.index.json;
     * the M1 container keeps them, mtp shard group) */
    int64_t d = (int64_t)c->attn_d;
    MTP_GET16(w->enorm_w, "mtp.pre_fc_norm_embedding.weight", (int64_t)H);
    MTP_GET16(w->hnorm_w, "mtp.pre_fc_norm_hidden.weight", (int64_t)H);
    MTP_GET16(w->fc_w, "mtp.fc.weight", (int64_t)H * 2 * H);
    MTP_GET16(w->norm_w, "mtp.norm.weight", (int64_t)H);
    MTP_GET16(lw->ln1_w, "mtp.layers.0.input_layernorm.weight",
              (int64_t)H);
    MTP_GET16(lw->ln2_w, "mtp.layers.0.post_attention_layernorm.weight",
              (int64_t)H);
    MTP_GET16(lw->wq, "mtp.layers.0.self_attn.q_proj.weight",
              (int64_t)c->attn_nh * 2 * d * H);
    MTP_GET16(lw->wk, "mtp.layers.0.self_attn.k_proj.weight",
              (int64_t)c->attn_nkv * d * H);
    MTP_GET16(lw->wv, "mtp.layers.0.self_attn.v_proj.weight",
              (int64_t)c->attn_nkv * d * H);
    MTP_GET16(lw->wo, "mtp.layers.0.self_attn.o_proj.weight",
              (int64_t)H * c->attn_nh * d);
    MTP_GET16(lw->qn_w, "mtp.layers.0.self_attn.q_norm.weight", d);
    MTP_GET16(lw->kn_w, "mtp.layers.0.self_attn.k_norm.weight", d);
    MTP_GET16(lw->rtr_w, "mtp.layers.0.mlp.gate.weight",
              (int64_t)c->experts * H);
    int64_t E = (int64_t)c->experts, I = (int64_t)c->moe_inter;
    int64_t Is = (int64_t)c->shared_inter;
    MTP_GET16(lw->sh_g, "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
              Is * H);
    MTP_GET16(lw->sh_u, "mtp.layers.0.mlp.shared_expert.up_proj.weight",
              Is * H);
    MTP_GET16(lw->sh_d, "mtp.layers.0.mlp.shared_expert.down_proj.weight",
              (int64_t)H * Is);
    MTP_GET16(lw->sh_gate, "mtp.layers.0.mlp.shared_expert_gate.weight",
              (int64_t)H);
    if (!tiered) {
        w->exp_gu = malloc((size_t)E * 2 * I * H * 2);
        w->exp_d = malloc((size_t)E * H * I * 2);
        if (!w->exp_gu || !w->exp_d) {
            snprintf(err, errcap, "mtp: OOM experts");
            apus_mtp_free(w);
            return -1;
        }
        /* the M1 slab slices: gate_up_proj.weight [2I, H] is ALREADY
         * the fused [gate|up] layout — a plain concat copy per expert,
         * no repack (same rule as the c/model.h loader) */
        for (int64_t e = 0; e < E; e++) {
            const ApusStTensor *t;
            snprintf(nm, sizeof nm,
                     "mtp.layers.0.mlp.experts.%lld.gate_up_proj.weight",
                     (long long)e);
            t = apus_st_set_get(m->set, nm);
            if (!t || t->dtype != APUS_ST_BF16
                || (int64_t)apus_st_nelem(t) != (int64_t)(2 * I * H))
                goto exp_fail;
            memcpy(w->exp_gu + e * 2 * I * H, t->data,
                   (size_t)2 * I * H * 2);
            snprintf(nm, sizeof nm,
                     "mtp.layers.0.mlp.experts.%lld.down_proj.weight",
                     (long long)e);
            t = apus_st_set_get(m->set, nm);
            if (!t || t->dtype != APUS_ST_BF16
                || (int64_t)apus_st_nelem(t) != (int64_t)(H * I))
                goto exp_fail;
            memcpy(w->exp_d + e * H * I, t->data, (size_t)H * I * 2);
        }
        lw->exp_gu = w->exp_gu;
        lw->exp_d = w->exp_d;
    }
#undef MTP_GET16
    return 0;
exp_fail:
    snprintf(err, errcap, "mtp: bad tensor %s", nm);
    apus_mtp_free(w);
    return -1;
}

void apus_mtp_free(ApusMtpW *w) {
    free(w->exp_gu);
    free(w->exp_d);
    w->exp_gu = w->exp_d = NULL;
}

void apus_mtp_state_init(const ApusModel *m, const ApusMtpW *w,
                         ApusLayerState *st) {
    memset(st, 0, sizeof *st);
    size_t nd = w->lc.attn_nkv * w->lc.attn_d;
    st->kcache = calloc((size_t)m->max_seq * nd, 2);
    st->vcache = calloc((size_t)m->max_seq * nd, 2);
    st->pos = 0;
}

void apus_mtp_state_free(ApusLayerState *st) {
    free(st->kcache);
    free(st->vcache);
    memset(st, 0, sizeof *st);
}

/* --- MTP step ------------------------------------------------------------------*/

void apus_mtp_forward(ApusModel *m, const ApusMtpW *w, ApusLayerState *st,
                      ApusStore *store, const uint16_t *h,
                      const int64_t *emb_ids, size_t T,
                      float *logits, uint16_t *hidden_out) {
    size_t H = (size_t)m->hidden, V = (size_t)m->vocab;
    for (size_t t = 0; t < T; t++) {
        uint16_t e[H], hh[H], cat[2 * H], z[H], out[H], yn[H];
        uint16_t lb[V];
        apus_attn_rmsnorm(m->embed + (size_t)emb_ids[t] * H, w->enorm_w,
                          e, H);
        apus_attn_rmsnorm(h + t * H, w->hnorm_w, hh, H);
        memcpy(cat, e, H * 2);
        memcpy(cat + H, hh, H * 2);
        apus_bf16_gemv_hot(w->fc_w, cat, z, H, 2 * H);
        if (store)
            apus_store_layer_forward(store, w->layer, &w->lc, &w->lw, st,
                                     z, out, 1);
        else
            apus_layer_forward_hot(&w->lc, &w->lw, st, z, out, 1, NULL);
        apus_attn_rmsnorm(out, w->norm_w, yn, H);
        apus_bf16_gemv_hot(m->head, yn, lb, V, H);
        for (size_t v = 0; v < V; v++)
            logits[t * V + v] = apus_bf16_f32(lb[v]);
        memcpy(hidden_out + t * H, yn, H * 2);
    }
}

/* --- snapshot ------------------------------------------------------------------*/

#define APUS_SNAP_MAIN 1
#define APUS_SNAP_MTP  2

void apus_snap_save(const ApusModel *m, const ApusModelState *st,
                    const ApusLayerState *mtp, ApusSnap *sn, int what) {
    if (!sn->layers) {
        sn->n_layers = m->n_layers;
        sn->layers = calloc((size_t)m->n_layers, sizeof *sn->layers);
        for (int L = 0; L < m->n_layers; L++) {
            const ApusLayerCfg *c = &m->layers[L].lc;
            if (c->kind == APUS_LAYER_GDN) {
                size_t cd = 2 * c->gdn_hk * c->gdn_dk
                            + c->gdn_hv * c->gdn_dv;
                sn->layers[L].conv = malloc(cd * 3 * 2);
                sn->layers[L].S = malloc(c->gdn_hv * c->gdn_dk
                                         * c->gdn_dv * sizeof(float));
            }
        }
    }
    if (!(what & APUS_SNAP_MAIN)) {
        if (what & APUS_SNAP_MTP) sn->mtp_pos = mtp->pos;
        return;
    }
    for (int L = 0; L < m->n_layers; L++) {
        const ApusLayerCfg *c = &m->layers[L].lc;
        const ApusLayerState *s = &st->layers[L];
        sn->layers[L].kind = (int)c->kind;
        sn->layers[L].pos = s->pos;
        if (c->kind == APUS_LAYER_GDN) {
            size_t cd = 2 * c->gdn_hk * c->gdn_dk
                        + c->gdn_hv * c->gdn_dv;
            memcpy(sn->layers[L].conv, s->conv_state, cd * 3 * 2);
            memcpy(sn->layers[L].S, s->S, c->gdn_hv * c->gdn_dk
                   * c->gdn_dv * sizeof(float));
        }
    }
    sn->mtp_pos = mtp->pos;
    sn->model_pos = st->pos;
}

void apus_snap_restore(const ApusModel *m, ApusModelState *st,
                       ApusLayerState *mtp, const ApusSnap *sn, int what) {
    if (what & APUS_SNAP_MTP) mtp->pos = sn->mtp_pos;
    if (!(what & APUS_SNAP_MAIN)) return;
    for (int L = 0; L < m->n_layers; L++) {
        const ApusLayerCfg *c = &m->layers[L].lc;
        ApusLayerState *s = &st->layers[L];
        s->pos = sn->layers[L].pos;
        if (c->kind == APUS_LAYER_GDN) {
            size_t cd = 2 * c->gdn_hk * c->gdn_dk
                        + c->gdn_hv * c->gdn_dv;
            memcpy(s->conv_state, sn->layers[L].conv, cd * 3 * 2);
            memcpy(s->S, sn->layers[L].S, c->gdn_hv * c->gdn_dk
                   * c->gdn_dv * sizeof(float));
        }
    }
    st->pos = sn->model_pos;
}

void apus_snap_free(ApusSnap *sn) {
    if (!sn->layers) return;
    for (int L = 0; L < sn->n_layers; L++) {
        free(sn->layers[L].conv);
        free(sn->layers[L].S);
    }
    free(sn->layers);
    sn->layers = NULL;
}

/* --- the spec engine ----------------------------------------------------------*/

/* Chain one draft from the chain hidden (the MTP block's own mtp.norm
 * output; the same single-norm forward handles it). */
static void apus_mtp_chain_step(ApusModel *m, ApusMtpW *mw,
                                ApusLayerState *mtp, ApusStore *store,
                                uint16_t *chain_h, int64_t eid,
                                float *lg, ApusDraftOverride override,
                                void *ov_ctx, int64_t cand_pos, int depth,
                                int32_t *out) {
    apus_mtp_forward(m, mw, mtp, store, chain_h, &eid, 1, lg, chain_h);
    *out = override ? override(ov_ctx, cand_pos, depth)
                    : apus_sample_argmax(lg, (size_t)m->vocab);
}

int apus_spec_run(ApusModel *m, ApusMtpW *mw, ApusModelState *st,
                  ApusStore *store, const int64_t *prompt, size_t n_prompt,
                  int max_tokens, float temp, int top_k, float top_p,
                  uint64_t seed, int spec_k,
                  ApusDraftOverride override, void *ov_ctx,
                  int *out_tokens, ApusSpecStats *stats) {
    memset(stats, 0, sizeof *stats);
    size_t H = (size_t)m->hidden, V = (size_t)m->vocab;
    int K = spec_k < 2 ? 2 : spec_k;
    ApusRng rng;
    apus_rng_seed(&rng, seed);
    void *scratch = malloc(apus_sample_scratch_size(V));
    /* main prefill logits: n_prompt rows (logits_all=1); chain steps
     * use one row — so max(n_prompt, K) covers every use */
    float *lg = malloc((n_prompt > (size_t)K ? n_prompt : (size_t)K)
                       * V * sizeof(float));
    /* MTP step logits: up to max(n_prompt, K) rows (the MTP prefill) */
    float *mlog = malloc((n_prompt > (size_t)K ? n_prompt : (size_t)K)
                         * V * sizeof(float));
    float *rows = malloc((size_t)K * V * sizeof(float));
    uint16_t *hid = malloc((size_t)K * H * 2);
    int64_t *batch = malloc((size_t)K * sizeof(int64_t));
    int64_t *ibuf = malloc((n_prompt > (size_t)K ? n_prompt : (size_t)K)
                           * sizeof(int64_t));
    int32_t *drafts = malloc((size_t)(K - 1 > 0 ? K - 1 : 1)
                             * sizeof(int32_t));
    uint16_t *hall = malloc((n_prompt ? n_prompt : 1) * H * 2);
    uint16_t *rout = malloc((n_prompt > (size_t)K ? n_prompt : (size_t)K)
                            * H * 2);
    uint16_t *chain_h = malloc(H * 2);
    ApusSnap snap = {0};

    ApusLayerState mtp;
    apus_mtp_state_init(m, mw, &mtp);

    /* ---- prefill: main forward with all hidden rows, then the batched
     * MTP prefill over the prompt pairs (positions 0..P-2, rolled ids) */
    apus_model_forward_h(m, st, prompt, n_prompt, lg, 1, hall);
    int held = apus_sample(lg + (n_prompt - 1) * V, V, temp, top_k,
                           top_p, &rng, scratch);
    if (n_prompt > 1) {
        for (size_t i = 0; i + 1 < n_prompt; i++)
            ibuf[i] = prompt[i + 1];
        apus_mtp_forward(m, mw, &mtp, store, hall, ibuf, n_prompt - 1,
                         mlog, rout);
    }
    /* ---- initial seed pair (position P-1, true inputs), then the
     * clean MTP snapshot, then the remaining chain steps (dirty) ------*/
    apus_mtp_forward(m, mw, &mtp, store, hall + (n_prompt - 1) * H,
                     (int64_t[]){ (int64_t)held }, 1, mlog, chain_h);
    drafts[0] = override ? override(ov_ctx, (int64_t)n_prompt + 1, 0)
                         : apus_sample_argmax(mlog, V);
    stats->drafts++;
    apus_snap_save(m, st, &mtp, &snap, APUS_SNAP_MTP);
    for (int d = 1; d < K - 1; d++) {
        apus_mtp_chain_step(m, mw, &mtp, store, chain_h, drafts[d - 1],
                            lg, override, ov_ctx,
                            (int64_t)n_prompt + 1 + d, d, &drafts[d]);
        stats->drafts++;
    }

    /* Step-top invariant: main fed through position q-1; held = T_q;
     * drafts[0..K-2] = candidates for T_{q+1}..T_{q+K-1}; the snapshot's
     * MTP part is the CLEAN point (pairs through the seed pair's
     * position q-1), taken BEFORE the chain steps. */
    int n = 0;
    while (n < max_tokens) {
        stats->steps++;
        apus_snap_save(m, st, &mtp, &snap, APUS_SNAP_MAIN);
        /* verify batch [held, z_1..z_{K-1}] at positions q..q+K-1 */
        batch[0] = held;
        for (int j = 1; j < K; j++) batch[j] = drafts[j - 1];
        apus_model_forward_h(m, st, batch, (size_t)K, rows, 1, hid);
        /* walk: emit held; accept draft j iff it equals the main model's
         * own draw from row j-1; one uniform per sampled position */
        out_tokens[n++] = held;
        int matched = 0, mismatch = 0;
        for (int j = 1; j < K && n < max_tokens; j++) {
            int y = apus_sample(rows + (size_t)(j - 1) * V, V, temp,
                                top_k, top_p, &rng, scratch);
            if (y == drafts[j - 1]) {
                out_tokens[n++] = y;
                matched++;
                stats->accepted++;
            } else {
                held = y;
                mismatch = 1;
                break;
            }
        }
        int full = !mismatch && matched == K - 1;
        if (full) {
            held = apus_sample(rows + (size_t)(K - 1) * V, V, temp,
                               top_k, top_p, &rng, scratch);
            stats->full_matches++;
        }
        if (n >= max_tokens) {
            /* exit alignment: the main state must be fed exactly the
             * emitted tokens — on a partial last step the verify batch
             * fed rejected drafts too, so restore + re-feed the emitted
             * prefix (bitwise the sequential state, m4c property) */
            if (!full) {
                apus_snap_restore(m, st, &mtp, &snap, APUS_SNAP_MAIN);
                apus_model_forward_h(m, st, batch, (size_t)(matched + 1),
                                     rows, 1, hid);
            }
            break;
        }

        if (!full) {
            /* partial: restore the MAIN clean point, re-feed the true
             * prefix (matched+1 tokens) in one batched call (bitwise
             * the sequential state by the m4c property) */
            stats->re_feeds++;
            apus_snap_restore(m, st, &mtp, &snap, APUS_SNAP_MAIN);
            apus_model_forward_h(m, st, batch, (size_t)(matched + 1),
                                 rows, 1, hid);
        }
        /* full: the main batch state fed exactly the true tokens — keep. */

        /* MTP: restore the clean point (both cases — the chain state is
         * dirty), then replay the true pairs at positions q..q+matched
         * in one batched call. Pair i<matched: (H[i], z_{i+1}); pair
         * i=matched: (H[matched], held). The last replay pair IS the
         * next step's seed pair. */
        apus_snap_restore(m, st, &mtp, &snap, APUS_SNAP_MTP);
        for (int i = 0; i <= matched; i++)
            ibuf[i] = (i < matched) ? batch[i + 1] : (int64_t)held;
        apus_mtp_forward(m, mw, &mtp, store, hid, ibuf,
                         (size_t)(matched + 1), mlog, rout);
        memcpy(chain_h, rout + (size_t)matched * H, H * 2);
        drafts[0] = override ? override(ov_ctx, st->pos + 1, 0)
                             : apus_sample_argmax(mlog, V);
        stats->drafts++;
        /* the replay ends CLEAN (all true pairs): new clean snapshot,
         * then the remaining chain steps */
        apus_snap_save(m, st, &mtp, &snap, APUS_SNAP_MTP);
        for (int d = 1; d < K - 1; d++) {
            apus_mtp_chain_step(m, mw, &mtp, store, chain_h,
                                drafts[d - 1], lg, override, ov_ctx,
                                st->pos + d + 1, d, &drafts[d]);
            stats->drafts++;
        }
    }
    stats->emitted = (uint64_t)n;

    apus_snap_free(&snap);
    apus_mtp_state_free(&mtp);
    free(scratch);
    free(lg); free(mlog); free(rows); free(hid); free(batch); free(ibuf);
    free(drafts); free(hall); free(rout); free(chain_h);
    return n;
}

#endif /* APUS_MTP_IMPLEMENTATION */
#endif /* APUS_MTP_H */
