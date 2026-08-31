/*
 * c/pilot.h — router-lookahead prefetch ("pilot", M6b) for
 * Qwen3.6-35B-A3B: predict layer L+1's routed experts from layer L's
 * post-attention hidden state (res1, bf16) and warm the M6a expert
 * store before the demand resolve arrives. Port of the Apus c/pilot.h
 * design (SPSC lock-free ring + dedicated pilot thread, drop-newest
 * backpressure, (pos,layer) watermark for stale hints, live recall
 * accounting); DeepSeek-specific mHC/hash-routing parts dropped.
 * C11, libc + pthreads.
 *
 * Prediction math (reuse, never duplicate):
 *   - router input = apus_attn_rmsnorm(res1_L, ln2_w of L+1) — the
 *     post_attention_layernorm (c/attn.h block convention) of the TARGET
 *     layer applied to layer L's post-attention hidden state;
 *   - scores/selection = apus_moe_route (c/moe.h) — the real router's
 *     own code path (bf16 matvec, FP32 softmax, top-k with stable
 *     lowest-index ties, renormalize; NO bias/sigmoid/group stage in
 *     this model). The predicted set is the first pilot_k entries of
 *     the router's top-k output (pilot_k > top_k uses the same
 *     selection machinery at a deeper truncation, apus_moe_route_topn).
 * Depth dL = 1 only. The proxy input differs from the real router input
 * (which also contains layer L+1's attention output), so prediction is
 * approximate — the M6b gates are machinery-exactness (counters ==
 * Python recompute) and BITWISE invariance, not a recall threshold.
 *
 * Delivery: the compute thread (post_attn hook from c/cache.h's tiered
 * forward) pushes (pos, layer, eid) entries onto a bounded SPSC ring; a
 * dedicated pilot thread pops them and calls apus_store_hint (thread-
 * safe, dedup'ed, eviction-guarded). The compute thread NEVER blocks: a
 * full ring drops the NEWEST entry (the ring is FIFO in issue order =
 * time-to-need order, so the oldest queued hint is the most urgent);
 * hints strictly behind the compute thread's (pos, layer) watermark are
 * dropped at the consumer as stale.
 *
 * Recall accounting (decode, s == 1): the router_actual hook compares
 * each MoE layer's actual chosen experts against the pending prediction
 * — actual_hits / actual_experts is measurable live from
 * apus_pilot_stats. Numerics are never touched: the pilot only reads
 * hidden states and issues store hints.
 *
 * Prefill: prefill_last_only (default 1) predicts only for the LAST
 * token of a multi-token forward call (the state that flows into
 * decode); per-token prefill predictions would flood the ring while the
 * I/O pool is busy with demand misses.
 *
 * Usage: #define APUS_PILOT_IMPLEMENTATION in exactly one TU (needs the
 * bf16/attn/moe/cache implementations linked).
 */
#ifndef APUS_PILOT_H
#define APUS_PILOT_H

#include <stddef.h>
#include <stdint.h>

#include "cache.h"      /* ApusStore, ApusStoreFwdHooks, apus_store_hint */

#ifdef __cplusplus
extern "C" {
#endif

/* Read-only view of one MoE layer's router + post-attention-norm
 * weights (owned by the model, not by the pilot). Every layer of this
 * model is MoE; an UNATTACHED layer has no view (rtr_w == NULL). */
typedef struct {
    const uint16_t *rtr_w;      /* [E, H] bf16 */
    const uint16_t *ln2_w;      /* [H] bf16 */
} ApusPilotRouter;

typedef struct {
    ApusStore  *store;          /* hint target; NULL = predict-only */
    int         n_layers;       /* required */
    int         n_experts;      /* required (E) */
    int         top_k;          /* required: router top-k (8) */
    int         hidden;         /* required (H) */
    int         enabled;        /* 0 = attach but never predict */
    int         pilot_k;        /* top-N cap (APUS_PILOT_K, default 8;
                                   capped at top_k). 0 = no lookahead */
    int         prefill_last_only; /* s>1: predict last token only (dflt 1) */
    size_t      ring_entries;   /* ring capacity (rounded up to pow2) */
    const char *dump_path;      /* NULL = off; NDJSON P/A sets (decode) */
} ApusPilotCfg;

typedef struct {
    uint64_t predictions;         /* top-N sets computed */
    uint64_t pred_experts;        /* expert ids predicted (sum of N) */
    uint64_t hints_enqueued;      /* ring pushes accepted */
    uint64_t hints_dropped_full;  /* ring-full drops (drop-newest) */
    uint64_t hints_issued;        /* consumer -> apus_store_hint calls */
    uint64_t hints_dropped_stale; /* consumer-side watermark drops */
    uint64_t actual_experts;      /* routed experts observed (s==1 layers
                                     with a pending prediction) */
    uint64_t actual_hits;         /* of those, present in the prediction */
} ApusPilotStats;

typedef struct ApusPilot ApusPilot;

ApusPilot *apus_pilot_create(const ApusPilotCfg *cfg);
void       apus_pilot_destroy(ApusPilot *p);   /* stop + join + free */

/* Register layer `layer`'s router view (call for every MoE layer). */
void apus_pilot_attach_router(ApusPilot *p, int layer,
                              const ApusPilotRouter *r);

/* The store-hook surface: fill *h with the pilot's post_attn /
 * router_actual callbacks and pass to apus_store_fwd_hooks. */
void apus_pilot_store_hooks(ApusPilot *p, ApusStoreFwdHooks *h);

/* Spawn the consumer thread (idempotent; destroy works without it). */
int  apus_pilot_start(ApusPilot *p);

/* Pure prediction (no ring, no stats, no store): predicted top-n expert
 * ids (router score order) for layer `target` from the post-attention
 * state res1 (bf16 [H]) of one token of the PREVIOUS layer. Returns -1
 * if the target has no router view. */
int  apus_pilot_predict(const ApusPilot *p, int target,
                        const uint16_t *res1, int32_t *idx, int n);

void apus_pilot_stats(ApusPilot *p, ApusPilotStats *out);

/* Test hooks: direct ring access (bypass prediction; the wrappers are
 * the exact producer/consumer paths). */
int  apus_pilot_debug_push(ApusPilot *p, int64_t pos, int layer, int eid);
int  apus_pilot_debug_pop(ApusPilot *p, int64_t *pos, int *layer,
                          int *eid);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_PILOT_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include <stdatomic.h>

#include "attn.h"       /* apus_attn_rmsnorm */
#include "moe.h"        /* apus_moe_route */

#define APUS_PILOT_MAX_K 64    /* stack cap for a prediction set */

struct ApusPilot {
    ApusPilotCfg     cfg;
    ApusPilotRouter *routers;      /* [n_layers] */
    /* SPSC lock-free ring: producer = compute thread, consumer = pilot
     * thread. Entries pack (pos:32 | layer:16 | eid:16) into u64. */
    uint64_t        *ring;
    size_t           ring_mask;
    _Atomic uint64_t r_head;       /* consumer-owned index */
    _Atomic uint64_t r_tail;       /* producer-owned index */
    /* ordering watermark (producer-written, consumer-read) */
    _Atomic uint64_t cur_pos;
    _Atomic int      cur_layer;
    _Atomic int      cur_s;
    /* consumer thread */
    pthread_t        thread;
    int              started;
    pthread_mutex_t  mtx;          /* sleep/wakeup only, never held long */
    pthread_cond_t   cv;
    int              stopping;
    /* pending predictions for recall accounting (compute thread only) */
    int32_t         *pending;      /* [n_layers][pilot_k] */
    uint8_t         *pending_ok;   /* [n_layers] */
    int64_t         *pending_pos;  /* [n_layers] */
    FILE            *dump;
    /* stats (atomics: written by both threads) */
    _Atomic uint64_t st_predictions, st_pred_experts;
    _Atomic uint64_t st_enqueued, st_dropped_full;
    _Atomic uint64_t st_issued, st_dropped_stale;
    _Atomic uint64_t st_actual, st_hits;
};

/* --- ring ------------------------------------------------------------------*/

static uint64_t apus_pilot_pack(int64_t pos, int layer, int eid) {
    return ((uint64_t)(uint32_t)pos << 32)
         | ((uint64_t)(uint16_t)layer << 16) | (uint16_t)eid;
}

/* Producer-side push; drop-newest when full. Returns 1 if enqueued. */
static int apus_pilot_push(ApusPilot *p, int64_t pos, int layer, int eid) {
    uint64_t tail = atomic_load_explicit(&p->r_tail, memory_order_relaxed);
    uint64_t head = atomic_load_explicit(&p->r_head, memory_order_acquire);
    if (tail - head > p->ring_mask) {      /* full (cap = mask+1) */
        atomic_fetch_add(&p->st_dropped_full, 1);
        return 0;
    }
    p->ring[tail & p->ring_mask] = apus_pilot_pack(pos, layer, eid);
    atomic_store_explicit(&p->r_tail, tail + 1, memory_order_release);
    atomic_fetch_add(&p->st_enqueued, 1);
    return 1;
}

/* Consumer-side pop. Returns 1 and fills out-params, 0 when empty. */
static int apus_pilot_pop(ApusPilot *p, int64_t *pos, int *layer,
                          int *eid) {
    uint64_t head = atomic_load_explicit(&p->r_head, memory_order_relaxed);
    uint64_t tail = atomic_load_explicit(&p->r_tail, memory_order_acquire);
    if (head == tail) return 0;
    uint64_t e = p->ring[head & p->ring_mask];
    atomic_store_explicit(&p->r_head, head + 1, memory_order_release);
    *pos = (int64_t)(uint32_t)(e >> 32);
    *layer = (int)(uint16_t)(e >> 16);
    *eid = (int)(uint16_t)e;
    return 1;
}

static void apus_pilot_kick(ApusPilot *p) {
    pthread_mutex_lock(&p->mtx);
    pthread_cond_signal(&p->cv);
    pthread_mutex_unlock(&p->mtx);
}

int apus_pilot_debug_push(ApusPilot *p, int64_t pos, int layer, int eid) {
    int ok = apus_pilot_push(p, pos, layer, eid);
    if (ok) apus_pilot_kick(p);
    return ok;
}

int apus_pilot_debug_pop(ApusPilot *p, int64_t *pos, int *layer,
                         int *eid) {
    return apus_pilot_pop(p, pos, layer, eid);
}

/* --- consumer thread ---------------------------------------------------------*/

/* A hint is worth issuing only if its (pos, layer) is not strictly
 * behind the compute thread's watermark: an earlier token's layer, or an
 * earlier layer of the current token, has already run its MoE. */
static int apus_pilot_stale(ApusPilot *p, int64_t pos, int layer) {
    int64_t cp = (int64_t)atomic_load_explicit(&p->cur_pos,
                                               memory_order_acquire);
    int cl = atomic_load_explicit(&p->cur_layer, memory_order_acquire);
    return pos < cp || (pos == cp && layer < cl);
}

static void *apus_pilot_consumer(void *arg) {
    ApusPilot *p = arg;
    for (;;) {
        pthread_mutex_lock(&p->mtx);
        while (!p->stopping
               && atomic_load_explicit(&p->r_head, memory_order_acquire)
                  == atomic_load_explicit(&p->r_tail, memory_order_acquire))
            pthread_cond_wait(&p->cv, &p->mtx);
        if (p->stopping) {
            pthread_mutex_unlock(&p->mtx);
            return NULL;
        }
        pthread_mutex_unlock(&p->mtx);
        int64_t pos;
        int layer, eid;
        if (!apus_pilot_pop(p, &pos, &layer, &eid)) continue;
        if (apus_pilot_stale(p, pos, layer)) {
            atomic_fetch_add(&p->st_dropped_stale, 1);
            continue;
        }
        if (p->cfg.store) apus_store_hint(p->cfg.store, layer, eid);
        atomic_fetch_add(&p->st_issued, 1);
    }
}

/* --- prediction (shared math) --------------------------------------------------*/

int apus_pilot_predict(const ApusPilot *p, int target,
                       const uint16_t *res1, int32_t *idx, int n) {
    if (!p || target < 0 || target >= p->cfg.n_layers || n <= 0) return -1;
    const ApusPilotRouter *r = &p->routers[target];
    if (!r->rtr_w) return -1;
    int E = p->cfg.n_experts, H = p->cfg.hidden, TK = p->cfg.top_k;
    if (n > APUS_PILOT_MAX_K) n = APUS_PILOT_MAX_K;
    uint16_t x[H];
    /* the real router's own code path: post_attention_layernorm +
     * bf16 router matvec + fp32 softmax selection (c/moe.h
     * apus_moe_route) */
    apus_attn_rmsnorm(res1, r->ln2_w, x, (size_t)H);
    if (n > TK) {
        /* locality measurement / deeper hints: same selection machinery,
         * deeper truncation (topn <= TK would be bitwise route's picks) */
        apus_moe_route_topn(x, r->rtr_w, idx, (size_t)n,
                            (size_t)E, (size_t)H);
        return 0;
    }
    int32_t idx8[TK];
    uint16_t w8[TK];
    apus_moe_route(x, r->rtr_w, idx8, w8, (size_t)E,
                   (size_t)H, (size_t)TK);
    for (int i = 0; i < n; i++) idx[i] = idx8[i];
    return 0;
}

/* --- compute-thread hooks (c/cache.h tiered forward) ----------------------------*/

static void apus_pilot_dump_set(ApusPilot *p, char type, int64_t pos,
                                int layer, const int32_t *eids, int n) {
    fprintf(p->dump, "{\"type\":\"%c\",\"pos\":%lld,\"layer\":%d,\"eids\":[",
            type, (long long)pos, layer);
    for (int j = 0; j < n; j++)
        fprintf(p->dump, "%s%d", j ? "," : "", eids[j]);
    fprintf(p->dump, "]}\n");
}

static void apus_pilot_tr_post_attn(void *ctx, int layer,
                                    const uint16_t *res1, int64_t pos,
                                    int s, int t) {
    ApusPilot *p = ctx;
    atomic_store_explicit(&p->cur_pos, (uint64_t)pos,
                          memory_order_release);
    atomic_store_explicit(&p->cur_layer, layer, memory_order_release);
    atomic_store_explicit(&p->cur_s, s, memory_order_release);
    if (!p->cfg.enabled || p->cfg.pilot_k <= 0) return;
    if (s > 1 && p->cfg.prefill_last_only && t < s - 1) return;
    int target = layer + 1;
    if (target >= p->cfg.n_layers) return;
    if (!p->routers[target].rtr_w) return;      /* unattached target */
    int k = p->cfg.pilot_k;
    int32_t idx[APUS_PILOT_MAX_K];
    if (apus_pilot_predict(p, target, res1, idx, k)) return;
    atomic_fetch_add(&p->st_predictions, 1);
    atomic_fetch_add(&p->st_pred_experts, (uint64_t)k);
    memcpy(p->pending + (size_t)target * k, idx,
           (size_t)k * sizeof(int32_t));
    p->pending_ok[target] = 1;
    p->pending_pos[target] = pos;
    if (p->dump) apus_pilot_dump_set(p, 'P', pos, target, idx, k);
    for (int j = 0; j < k; j++)
        apus_pilot_push(p, pos, target, idx[j]);
    apus_pilot_kick(p);
}

static void apus_pilot_tr_router_actual(void *ctx, int layer,
                                        const int32_t *idx, int n,
                                        int64_t pos, int s) {
    ApusPilot *p = ctx;
    if (s != 1) return;                 /* recall accounting: decode only */
    if (p->dump) {
        apus_pilot_dump_set(p, 'A', pos, layer, idx, n);
        fflush(p->dump);
    }
    int k = p->cfg.pilot_k;
    if (!p->pending_ok[layer] || p->pending_pos[layer] != pos) return;
    const int32_t *pd = p->pending + (size_t)layer * k;
    for (int i = 0; i < n; i++) {
        atomic_fetch_add(&p->st_actual, 1);
        for (int j = 0; j < k; j++)
            if (pd[j] == idx[i]) {
                atomic_fetch_add(&p->st_hits, 1);
                break;
            }
    }
}

void apus_pilot_store_hooks(ApusPilot *p, ApusStoreFwdHooks *h) {
    h->ctx = p;
    h->post_attn = apus_pilot_tr_post_attn;
    h->router_actual = apus_pilot_tr_router_actual;
}

/* --- lifecycle ------------------------------------------------------------------*/

ApusPilot *apus_pilot_create(const ApusPilotCfg *cfg) {
    if (!cfg || cfg->n_layers <= 0 || cfg->n_experts <= 0 || cfg->top_k <= 0
        || cfg->hidden <= 0)
        return NULL;
    ApusPilot *p = calloc(1, sizeof *p);
    p->cfg = *cfg;
    if (p->cfg.pilot_k > APUS_PILOT_MAX_K)
        p->cfg.pilot_k = APUS_PILOT_MAX_K;
    /* prefill_last_only: 0 (unset) = default 1; set <0 to disable */
    if (p->cfg.prefill_last_only == 0) p->cfg.prefill_last_only = 1;
    if (p->cfg.prefill_last_only < 0) p->cfg.prefill_last_only = 0;
    size_t cap = p->cfg.ring_entries ? p->cfg.ring_entries : 4096;
    size_t pow2 = 16;
    while (pow2 < cap) pow2 <<= 1;
    p->ring = malloc(pow2 * sizeof(uint64_t));
    p->ring_mask = pow2 - 1;
    atomic_store(&p->cur_pos, 0);
    atomic_store(&p->cur_layer, -1);
    atomic_store(&p->cur_s, 1);
    pthread_mutex_init(&p->mtx, NULL);
    pthread_cond_init(&p->cv, NULL);
    int nl = p->cfg.n_layers, k = p->cfg.pilot_k ? p->cfg.pilot_k : 1;
    p->routers = calloc((size_t)nl, sizeof *p->routers);
    p->pending = calloc((size_t)nl * k, sizeof(int32_t));
    p->pending_ok = calloc((size_t)nl, 1);
    p->pending_pos = calloc((size_t)nl, sizeof(int64_t));
    if (p->cfg.dump_path) {
        p->dump = fopen(p->cfg.dump_path, "w");
        if (!p->dump)
            fprintf(stderr, "pilot: cannot open dump %s\n",
                    p->cfg.dump_path);
    }
    return p;
}

void apus_pilot_attach_router(ApusPilot *p, int layer,
                              const ApusPilotRouter *r) {
    if (!p || layer < 0 || layer >= p->cfg.n_layers) return;
    p->routers[layer] = *r;
}

int apus_pilot_start(ApusPilot *p) {
    if (!p || p->started) return 0;
    if (pthread_create(&p->thread, NULL, apus_pilot_consumer, p))
        return -1;
    p->started = 1;
    return 0;
}

void apus_pilot_destroy(ApusPilot *p) {
    if (!p) return;
    if (p->started) {
        pthread_mutex_lock(&p->mtx);
        p->stopping = 1;
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mtx);
        pthread_join(p->thread, NULL);
    }
    if (p->dump) fclose(p->dump);
    pthread_mutex_destroy(&p->mtx);
    pthread_cond_destroy(&p->cv);
    free(p->ring);
    free(p->routers);
    free(p->pending);
    free(p->pending_ok);
    free(p->pending_pos);
    free(p);
}

void apus_pilot_stats(ApusPilot *p, ApusPilotStats *out) {
    out->predictions = atomic_load(&p->st_predictions);
    out->pred_experts = atomic_load(&p->st_pred_experts);
    out->hints_enqueued = atomic_load(&p->st_enqueued);
    out->hints_dropped_full = atomic_load(&p->st_dropped_full);
    out->hints_issued = atomic_load(&p->st_issued);
    out->hints_dropped_stale = atomic_load(&p->st_dropped_stale);
    out->actual_experts = atomic_load(&p->st_actual);
    out->actual_hits = atomic_load(&p->st_hits);
}

#endif /* APUS_PILOT_IMPLEMENTATION */
#endif /* APUS_PILOT_H */
