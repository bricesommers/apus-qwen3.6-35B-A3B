/*
 * c/cache.h — expert store (M6a): demand-loading of routed-expert slabs
 * from NVMe through a bounded RAM cache, for the Qwen3.6-35B-A3B
 * apus-qwen container. Port of the Apus c/cache.h design (per-layer LRU
 * with end-of-block promotion, LFRU pins with hysteresis, RSS guard,
 * generation-tagged pthread I/O pool, slab freelist recycling),
 * re-anchored at M6 to the M1 container's FORMAT_VERSION 2 slab: TWO
 * BF16 slices per expert cut from the fused source tensors —
 * gate_up_proj.weight [2I, H] then down_proj.weight [H, I], contiguous
 * in one shard, 6,291,456 B on the real model (I=512, H=2048). C11,
 * libc + pthreads.
 *
 * Addressing: (layer, eid) -> slab record. A cache miss is ONE pread of
 * the whole slab into a 4 KiB-aligned buffer; ApusBf16ExpertW views are
 * zero-copy pointers into it. Slab records come from the apus.index.json
 * expert_slabs records ({layer, expert, shard, offset, nbytes}; layer is
 * an int) when the manifest is present — member offsets then verified
 * against the shard headers — and are otherwise derived from the shard
 * headers alone (the 2 member slices must share one shard and tile a
 * contiguous byte range; the store verifies, it does not assume). Open
 * fails loudly on a split/non-contiguous slab, on a manifest whose
 * format_version != 2 (a v1 3-member-slab container must NOT be
 * silently misread), and on a container with no v2 expert slabs at all.
 * This model has MoE on EVERY layer (no dense layers), so every main
 * layer has records. MTP slab records (converter layer numbering
 * num_hidden_layers + K, tensor names mtp.layers.{K}.mlp.experts.*) are
 * out of range for the default store and skipped; under --spec (M8) the
 * store is sized n_layers + n_mtp with n_main_layers = n_layers, and
 * layers [n_main, n_layers) resolve under the mtp.layers.{L-n_main}
 * tensor prefix through the identical one-pread path.
 *
 * Cache policy (identical to Apus — see its tests/m6a/README.md):
 *   - Per-layer LRU slot arrays. Misses load into a small per-forward
 *     working set (never directly into the LRU), promoted at layer end by
 *     swapping with the coldest slots; hits bump an atomic clock. A
 *     batch-union overflow drops the excess after use — one-shot
 *     streaming experts do not flush the cache.
 *   - Hot-pin store: per-layer pinned slots, never evicted, seeded from
 *     a persistent usage-history file (plain text "layer eid count",
 *     atomically rewritten, merge-with-max). Between-turns REPIN pass
 *     with LFRU score (frequency primary, recency tiebreak) and 25%+4
 *     hysteresis.
 *   - RSS guard: measured RSS (mach task_info) over budget -> LRU
 *     payloads freed in place (slots keep eid/freq identity; pins,
 *     in-flight loads and working-set entries untouched; block
 *     boundaries only).
 *   - Miss overlap: pthread I/O pool, generation-tagged jobs (a
 *     straggler whose slot was recycled drops its payload; stale bytes
 *     can never alias a newer generation). Loads are demand-class
 *     (resolve re-submits, batch-union storms via apus_store_hint_demand,
 *     any LOADING slot a resolve blocks on) or speculative
 *     (apus_store_hint — the pilot surface); workers pop the first
 *     demand-class job before FIFO speculative ones (APUS_STORE_BOOST).
 *     The compute thread never does I/O in pool mode
 *     (io_threads < 0 = synchronous mode for tests). F_NOCACHE streaming
 *     reads keep expert traffic out of the page cache.
 *   - Evicted/dropped slab buffers go on a free list (exact slab_bytes
 *     class) and are reused by the next load; RSS-guard drops really
 *     free().
 *
 * Budgets (env, Apus defaults; all overridable via ApusStoreCfg):
 * APUS_EXPERT_CACHE_MB (4096), APUS_PIN_MB (512), APUS_RSS_GUARD_MB
 * (26624), APUS_IO_THREADS (4), APUS_NOCACHE (1), APUS_USAGE_DECAY (1.0),
 * APUS_BUF_FREE (64), APUS_STORE_BOOST (1). Usage file defaults to
 * <model_dir>/apus.usage.
 *
 * Tiered forward: apus_store_layer_forward mirrors c/layer.h's per-token
 * wiring EXACTLY (same public kernels in the same order — c/layer.h's
 * attention halves are static, so the mirror lives here), with the MoE
 * FFN's expert GEMVs reading store-resolved slab views instead of eager
 * arrays. Same slab bytes + same kernels + same accumulation order =>
 * BITWISE identical to the eager path (the M6a hard gate).
 *
 * Threading contract: resolve/hint/layer_end are called from the compute
 * thread; all entry points are internally locked, so the M6b pilot thread
 * may call apus_store_hint from elsewhere. repin/save_usage/rss_guard are
 * between-forwards only. Compute bits do not depend on I/O timing.
 *
 * Usage: #define APUS_CACHE_IMPLEMENTATION in exactly one TU (also needs
 * the st/json/compat implementations).
 */
#ifndef APUS_CACHE_H
#define APUS_CACHE_H

#include <stddef.h>
#include <stdint.h>

#include "st.h"
#include "layer.h"
#include "blas.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Zero-copy view of one expert's two BF16 slices inside a v2 slab:
 * the FUSED gate_up [2I, H] (the slab slice IS the fused [gate|up]
 * layout — no repack) and down [H, I]. */
typedef struct {
    const uint16_t *gate_up;  /* [2I, H] fused [gate|up] */
    const uint16_t *down;     /* [H, I] */
    int64_t I, H;
} ApusBf16ExpertW;

typedef struct {
    int    n_layers;          /* required: TOTAL model layers (dense
                                 layers simply have no slab records);
                                 under --spec: n_main + n_mtp */
    int    n_main_layers;     /* main-model layers; 0 = n_layers (no
                                 MTP). Layers [n_main, n_layers) resolve
                                 under mtp.layers.{L-n_main} (M8) */
    int    n_experts;         /* required: routed experts per MoE layer */
    size_t cache_bytes;       /* LRU budget; 0 = APUS_EXPERT_CACHE_MB env */
    size_t pin_bytes;         /* pin budget; 0 = APUS_PIN_MB env */
    int    slots_per_layer;   /* explicit LRU slots; 0 = derive */
    int    pins_per_layer;    /* explicit pin slots; 0 = derive */
    size_t rss_budget_bytes;  /* 0 = APUS_RSS_GUARD_MB env */
    int    io_threads;        /* 0 = env/default (4); <0 = synchronous */
    int    nocache;           /* >0 = F_NOCACHE, <0 = cached, 0 = env(1) */
    const char *usage_path;   /* NULL = <model_dir>/apus.usage; "" = off */
    double usage_decay;       /* 0 = APUS_USAGE_DECAY env (default 1.0) */
} ApusStoreCfg;

typedef struct ApusStore ApusStore;

typedef struct {
    uint64_t hits, misses, preads, bytes_read;
    uint64_t evictions, rss_drops, pin_loads, repin_swaps;
    uint64_t hint_loads, demand_loads;
    uint64_t waits, wait_ns, pread_ns;
} ApusStoreStats;

ApusStore *apus_store_open(const char *model_dir, const ApusStoreCfg *cfg,
                           char *err, size_t errcap);
void       apus_store_close(ApusStore *st);

/* Resolve expert (layer, eid): fills *out with zero-copy views into a
 * cache slot, waiting just-in-time if a load is in flight. Returns 0. */
int  apus_store_resolve(ApusStore *st, int layer, int eid,
                        ApusBf16ExpertW *out);

/* Non-blocking prefetch hint (the M6b pilot surface): submits the miss
 * job without waiting. Thread-safe, dedup'ed, and an unconsumed hint
 * never evicts a warm demand-loaded expert. */
void apus_store_hint(ApusStore *st, int layer, int eid);

/* Demand-class variant: identical submission semantics, but the load is
 * served ahead of queued speculative loads (batch-union storms). */
void apus_store_hint_demand(ApusStore *st, int layer, int eid);

/* End-of-block: promote this layer's working set into the LRU, advance
 * the load generation, run the RSS guard. */
void apus_store_layer_end(ApusStore *st, int layer);

/* Between-turns REPIN: LFRU with 25%+4 hysteresis. */
void apus_store_repin(ApusStore *st);

/* Persist usage history atomically (tmp + fsync + rename). */
int  apus_store_save_usage(ApusStore *st);

/* RSS guard (also runs automatically at layer end). */
void apus_store_rss_guard(ApusStore *st);

void     apus_store_stats(const ApusStore *st, ApusStoreStats *out);
size_t   apus_store_slab_bytes(const ApusStore *st);
size_t   apus_store_resident_bytes(ApusStore *st);

/* --- tiered forward + pilot hook surface ------------------------------------*/

/* Hooks fired by the tiered forward (the M6b pilot attaches here).
 * post_attn: after a token's attention half, with res1 (bf16 [H]) and
 *   the absolute position pos; s = batch size of the model forward call,
 *   t = token index within it (for prefill_last_only semantics).
 * router_actual: with the actual chosen expert ids for (layer, pos). */
typedef struct {
    void *ctx;
    void (*post_attn)(void *ctx, int layer, const uint16_t *res1,
                      int64_t pos, int s, int t);
    void (*router_actual)(void *ctx, int layer, const int32_t *idx,
                          int n, int64_t pos, int s);
} ApusStoreFwdHooks;

void apus_store_fwd_hooks(ApusStore *st, const ApusStoreFwdHooks *h);
/* Batch context for the hooks (model.h sets this per token). */
void apus_store_fwd_set_batch(ApusStore *st, int s, int t);

/* Tiered per-token layer forward for model layer `layer` (GDN+MoE /
 * gated-GQA+MoE kinds): BITWISE the eager apus_layer_forward, expert
 * weights resolved from the store. */
void apus_store_layer_forward(ApusStore *st, int layer,
                              const ApusLayerCfg *c,
                              const ApusLayerW *w, ApusLayerState *state,
                              const uint16_t *x, uint16_t *out, size_t T);

/* --- introspection / test hooks -------------------------------------------*/

int  apus_store_debug_layer(ApusStore *st, int layer,
                            int32_t *lru_eids, int n_lru,
                            int32_t *pin_eids, int n_pins);
void apus_store_debug_set_pre_claim(ApusStore *st,
    void (*fn)(ApusStore *st, int layer, int32_t eid, uint64_t gen));
int  apus_store_debug_stale_gen(ApusStore *st, int layer, int32_t eid);
int  apus_store_debug_present(ApusStore *st, int layer, int eid);
int  apus_store_debug_ready(ApusStore *st, int layer, int eid);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_CACHE_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pthread.h>
#include <unistd.h>

#include "compat.h"
#include "json.h"
#include "moe.h"

/* --- slab records ----------------------------------------------------------*/

typedef struct {
    ApusStLazy *lz;
    uint64_t    off;        /* absolute file offset of the slab */
    uint64_t    len;        /* slab bytes */
    uint32_t    rel[2];     /* per-member offset within slab, canonical
                               order gate_up, down */
    int64_t     I, H;
} ApusSlabRec;

/* --- slots -----------------------------------------------------------------*/

enum { APUS_SLOT_EMPTY = 0, APUS_SLOT_LOADING = 1, APUS_SLOT_READY = 2 };

typedef struct {
    int32_t  eid;
    uint8_t *buf;
    uint64_t last;
    uint64_t freq;
    int      state;
    uint64_t gen;
    uint8_t  hot;
} ApusSlot;

typedef struct {
    ApusSlot  *slots;
    int        n_slots;
    ApusSlot  *pins;
    int        n_pins;
    ApusSlot **ws;
    int        ws_n, ws_cap;
    int        has_experts;
} ApusLayerCache;

typedef struct {
    int          layer;
    ApusSlot    *slot;
    ApusSlabRec *rec;
    int          is_pin;
    uint64_t     gen;
} ApusJob;

struct ApusStore {
    int            n_layers, n_main, E, n_moe;
    size_t         slab_bytes;
    ApusSlabRec   *recs;
    ApusLayerCache *lc;
    struct { char *name; ApusStLazy *lz; } *shards;
    int            shards_n, shards_cap;
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    pthread_t      *threads;
    int             n_threads;
    ApusJob        *jobs;
    int             jq_head, jq_n, jq_cap;
    pthread_cond_t  jq_cv;
    int             stopping;
    int             boost;
    uint8_t       **buf_free;
    int             buf_free_n, buf_free_cap;
    uint64_t        clock;
    uint64_t        gen;
    size_t          rss_budget;
    ApusStoreStats  stats;
    char            usage_path[1200];
    int             usage_enabled;
    double          usage_decay;
    void          (*test_pre_claim)(ApusStore *, int, int32_t, uint64_t);
    /* pilot hook surface */
    ApusStoreFwdHooks hooks;
    int             fwd_s, fwd_t;
};

/* --- small utilities --------------------------------------------------------*/

static uint64_t apus_clock_tick(ApusStore *st) { return ++st->clock; }

/* M13 (Apus M15 lesson): apus_aligned_alloc pairs with apus_aligned_free
 * everywhere a slab buffer is released — _aligned_malloc storage aborts
 * on plain free() under Windows; posix_memalign storage being free()-
 * legal hides a mismatched pair on POSIX. */
static void *apus_slab_alloc(size_t n) {
    return apus_aligned_alloc(4096, n);
}

#define APUS_BUF_FREE_MAX 64

static uint8_t *apus_store_buf_get(ApusStore *st, size_t n) {
    if (n == st->slab_bytes) {
        pthread_mutex_lock(&st->mu);
        if (st->buf_free_n > 0) {
            uint8_t *b = st->buf_free[--st->buf_free_n];
            pthread_mutex_unlock(&st->mu);
            return b;
        }
        pthread_mutex_unlock(&st->mu);
    }
    return apus_slab_alloc(n);
}

static void apus_store_buf_put(ApusStore *st, uint8_t *b) {
    if (!b) return;
    if (st->buf_free_n < st->buf_free_cap)
        st->buf_free[st->buf_free_n++] = b;
    else
        apus_aligned_free(b);   /* pairs with apus_slab_alloc (M13) */
}

static ApusSlabRec *apus_store_rec(ApusStore *st, int layer, int eid) {
    return &st->recs[(size_t)layer * st->E + eid];
}

static void apus_slot_views(const ApusSlabRec *rec, const uint8_t *buf,
                            ApusBf16ExpertW *out) {
    out->gate_up = (const uint16_t *)(buf + rec->rel[0]);
    out->down = (const uint16_t *)(buf + rec->rel[1]);
    out->I = rec->I;
    out->H = rec->H;
}

/* --- open: expert name parsing + slab derivation ------------------------------*/

typedef struct {
    int layer, eid, member; /* member: 0=gate_up, 1=down */
    char shard[256];
} ApusExpertTensorRef;

static int apus_ref_cmp(const void *a, const void *b) {
    const ApusExpertTensorRef *x = a, *y = b;
    if (x->layer != y->layer) return x->layer - y->layer;
    if (x->eid != y->eid) return x->eid - y->eid;
    return x->member - y->member;
}

static int apus_parse_expert_name(const char *name, int n_main,
                                  int *layer, int *eid, int *member) {
    /* model.language_model.layers.{L}.mlp.experts.{E}.{gate_up_proj,
     * down_proj}.weight — the v2 synthesized slab-slice names
     * (tools/convert.py _slice_name). Under --spec (M8) the mtp prefix
     * mtp.layers.{K}.mlp.experts.{E}.* is also parsed and maps to layer
     * n_main + K (the converter's slab-record numbering). */
    int L, E;
    char kind[48], chk[320];
    if (sscanf(name,
               "model.language_model.layers.%d.mlp.experts.%d.%47s",
               &L, &E, kind) == 3) {
        if (!strcmp(kind, "gate_up_proj.weight")) *member = 0;
        else if (!strcmp(kind, "down_proj.weight")) *member = 1;
        else return -1;
        snprintf(chk, sizeof chk,
                 "model.language_model.layers.%d.mlp.experts.%d.%s",
                 L, E, kind);
        if (strcmp(chk, name)) return -1;      /* suffix garbage */
        *layer = L;
        *eid = E;
        return 0;
    }
    if (sscanf(name, "mtp.layers.%d.mlp.experts.%d.%47s",
               &L, &E, kind) == 3) {
        if (!strcmp(kind, "gate_up_proj.weight")) *member = 0;
        else if (!strcmp(kind, "down_proj.weight")) *member = 1;
        else return -1;
        snprintf(chk, sizeof chk,
                 "mtp.layers.%d.mlp.experts.%d.%s", L, E, kind);
        if (strcmp(chk, name)) return -1;
        *layer = n_main + L;
        *eid = E;
        return 0;
    }
    return -1;
}

/* Tensor name prefix for a store layer: main layers under
 * model.language_model.layers.{L}, MTP layers ([n_main, n_layers))
 * under mtp.layers.{L - n_main} (M8). */
static void apus_store_layer_prefix(const ApusStore *st, int layer,
                                    char *out, size_t cap) {
    if (layer < st->n_main)
        snprintf(out, cap, "model.language_model.layers.%d", layer);
    else
        snprintf(out, cap, "mtp.layers.%d", layer - st->n_main);
}

static const char * const APUS_SLAB_MEMBER[2] = {
    "gate_up_proj.weight", "down_proj.weight"
};

static ApusStLazy *apus_store_shard(ApusStore *st, const char *dir,
                                    const char *name, int nocache,
                                    char *err, size_t errcap) {
    for (int i = 0; i < st->shards_n; i++)
        if (!strcmp(st->shards[i].name, name)) return st->shards[i].lz;
    if (st->shards_n == st->shards_cap) {
        st->shards_cap = st->shards_cap ? 2 * st->shards_cap : 8;
        st->shards = realloc(st->shards,
                             (size_t)st->shards_cap * sizeof *st->shards);
    }
    char path[1400];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    ApusStLazy *lz = apus_st_lazy_open(path, nocache, err, errcap);
    if (!lz) return NULL;
    st->shards[st->shards_n].name = strdup(name);
    st->shards[st->shards_n].lz = lz;
    st->shards_n++;
    return lz;
}

/* Derive one slab record from the shard headers: the two member
 * slices must live in one shard and tile a contiguous byte range.
 * If off_hint/len_hint >= 0 (apus.index.json expert_slabs record), the
 * derived range must equal it. */
static int apus_store_derive_slab(ApusStore *st, const char *dir,
                                  ApusExpertTensorRef *refs /* [2] */,
                                  int64_t off_hint, int64_t len_hint,
                                  int nocache, char *err, size_t errcap) {
    int layer = refs[0].layer, eid = refs[0].eid;
    ApusSlabRec *rec = apus_store_rec(st, layer, eid);
    memset(rec, 0, sizeof *rec);
    const ApusStLazyTensor *mem[2];
    ApusStLazy *lz = NULL;
    for (int i = 0; i < 2; i++) {
        lz = apus_store_shard(st, dir, refs[i].shard, nocache, err, errcap);
        if (!lz) return -1;
        if (i && lz != apus_store_shard(st, dir, refs[0].shard, nocache,
                                        err, errcap)) {
            snprintf(err, errcap,
                     "store: expert %d.%d split across shards", layer, eid);
            return -1;
        }
        char tname[320], lpre[160];
        apus_store_layer_prefix(st, layer, lpre, sizeof lpre);
        snprintf(tname, sizeof tname,
                 "%s.mlp.experts.%d.%s",
                 lpre, eid, APUS_SLAB_MEMBER[refs[i].member]);
        mem[i] = apus_st_lazy_find(lz, tname);
        if (!mem[i]) {
            snprintf(err, errcap, "store: missing tensor %s", tname);
            return -1;
        }
    }
    /* member order on disk is canonical gate_up/down, but verify by
     * sorted intervals instead of assuming */
    int ord[2] = {0, 1};
    if (mem[1]->file_off < mem[0]->file_off) {
        ord[0] = 1; ord[1] = 0;
    }
    uint64_t base = mem[ord[0]]->file_off, cur = base;
    for (int i = 0; i < 2; i++) {
        if (mem[ord[i]]->file_off != cur) {
            snprintf(err, errcap,
                     "store: expert %d.%d slab not contiguous "
                     "(gap at member %d)", layer, eid, ord[i]);
            return -1;
        }
        cur += mem[ord[i]]->nbytes;
    }
    if (off_hint >= 0 && (base != (uint64_t)off_hint
                          || cur - base != (uint64_t)len_hint)) {
        snprintf(err, errcap,
                 "store: expert %d.%d slab [%llu,%llu) != manifest "
                 "[%lld,%lld)", layer, eid, (unsigned long long)base,
                 (unsigned long long)cur, (long long)off_hint,
                 (long long)(off_hint + len_hint));
        return -1;
    }
    for (int i = 0; i < 2; i++) {
        const ApusStLazyTensor *t = mem[i];
        if (t->dtype != APUS_ST_BF16 || t->ndim != 2) {
            snprintf(err, errcap,
                     "store: expert %d.%d member %d dtype/ndim mismatch",
                     layer, eid, i);
            return -1;
        }
        rec->rel[i] = (uint32_t)(t->file_off - base);
    }
    /* gate_up is [2I, H], down is [H, I] */
    if (mem[0]->shape[0] != 2 * mem[1]->shape[1]
        || mem[0]->shape[1] != mem[1]->shape[0]) {
        snprintf(err, errcap, "store: expert %d.%d dims mismatch",
                 layer, eid);
        return -1;
    }
    rec->lz = lz;
    rec->off = base;
    rec->len = cur - base;
    rec->I = mem[0]->shape[0] / 2;
    rec->H = mem[0]->shape[1];
    if (st->slab_bytes == 0) st->slab_bytes = (size_t)rec->len;
    if (rec->len != st->slab_bytes) {
        snprintf(err, errcap,
                 "store: expert %d.%d slab %llu != slab_bytes %zu",
                 layer, eid, (unsigned long long)rec->len, st->slab_bytes);
        return -1;
    }
    return 0;
}

/* --- usage history -----------------------------------------------------------*/

static int apus_store_load_usage(ApusStore *st) {
    FILE *f = fopen(st->usage_path, "r");
    if (!f) return -1;
    uint64_t *cnt = calloc((size_t)st->n_layers * st->E, sizeof(uint64_t));
    int L, e;
    unsigned long long c;
    while (fscanf(f, "%d %d %llu", &L, &e, &c) == 3)
        if (L >= 0 && L < st->n_layers && e >= 0 && e < st->E)
            cnt[(size_t)L * st->E + e] += c;
    fclose(f);
    for (int l = 0; l < st->n_layers; l++) {
        ApusLayerCache *lc = &st->lc[l];
        for (int p = 0; p < lc->n_pins; p++) {
            int best = -1;
            for (int e2 = 0; e2 < st->E; e2++) {
                if (!cnt[(size_t)l * st->E + e2]) continue;
                int taken = 0;
                for (int q = 0; q < p; q++)
                    if (lc->pins[q].eid == e2) { taken = 1; break; }
                if (taken) continue;
                if (best < 0
                    || cnt[(size_t)l * st->E + e2]
                       > cnt[(size_t)l * st->E + best])
                    best = e2;
            }
            if (best < 0) break;
            lc->pins[p].eid = best;
            lc->pins[p].freq =
                (uint64_t)((double)cnt[(size_t)l * st->E + best]
                           * st->usage_decay);
        }
    }
    free(cnt);
    return 0;
}

int apus_store_save_usage(ApusStore *st) {
    if (!st->usage_enabled) return 0;
    uint64_t *cnt = calloc((size_t)st->n_layers * st->E, sizeof(uint64_t));
    FILE *f = fopen(st->usage_path, "r");
    if (f) {
        int L, e;
        unsigned long long c;
        while (fscanf(f, "%d %d %llu", &L, &e, &c) == 3)
            if (L >= 0 && L < st->n_layers && e >= 0 && e < st->E) {
                c = (unsigned long long)((double)c * st->usage_decay);
                if (c > cnt[(size_t)L * st->E + e])
                    cnt[(size_t)L * st->E + e] = c;
            }
        fclose(f);
    }
    pthread_mutex_lock(&st->mu);
    for (int l = 0; l < st->n_layers; l++) {
        ApusLayerCache *lc = &st->lc[l];
        for (int i = 0; i < lc->n_slots; i++)
            if (lc->slots[i].eid >= 0
                && lc->slots[i].freq
                   > cnt[(size_t)l * st->E + lc->slots[i].eid])
                cnt[(size_t)l * st->E + lc->slots[i].eid] =
                    lc->slots[i].freq;
        for (int i = 0; i < lc->n_pins; i++)
            if (lc->pins[i].eid >= 0
                && lc->pins[i].freq
                   > cnt[(size_t)l * st->E + lc->pins[i].eid])
                cnt[(size_t)l * st->E + lc->pins[i].eid] =
                    lc->pins[i].freq;
    }
    pthread_mutex_unlock(&st->mu);
    char tmp[1300];
    snprintf(tmp, sizeof tmp, "%s.tmp", st->usage_path);
    f = fopen(tmp, "w");
    if (!f) { free(cnt); return -1; }
    for (int l = 0; l < st->n_layers; l++)
        for (int e = 0; e < st->E; e++)
            if (cnt[(size_t)l * st->E + e])
                fprintf(f, "%d %d %llu\n", l, e,
                        (unsigned long long)cnt[(size_t)l * st->E + e]);
    free(cnt);
    fflush(f);
    apus_sys_fsync(f);   /* M13: _commit on Windows */
    if (fclose(f)) { remove(tmp); return -1; }
    /* M13: POSIX rename() atomically replaces an existing destination;
     * Windows rename() fails with EEXIST — the shim keeps the semantics. */
    if (apus_sys_rename(tmp, st->usage_path)) { remove(tmp); return -1; }
    return 0;
}

/* --- I/O pool ------------------------------------------------------------------*/

static void apus_store_job_push(ApusStore *st, ApusJob j) {
    if (st->jq_n == st->jq_cap) {
        /* grow: re-lay-out the ring linearly (a plain realloc keeps the
         * bytes but changes the modulo geometry of wrapped entries) */
        int ncap = st->jq_cap ? 2 * st->jq_cap : 32;
        ApusJob *nj = malloc((size_t)ncap * sizeof *nj);
        for (int i = 0; i < st->jq_n; i++)
            nj[i] = st->jobs[(st->jq_head + i) % st->jq_cap];
        free(st->jobs);
        st->jobs = nj;
        st->jq_cap = ncap;
        st->jq_head = 0;
    }
    int tail = (st->jq_head + st->jq_n) % st->jq_cap;
    st->jobs[tail] = j;
    st->jq_n++;
    pthread_cond_signal(&st->jq_cv);
}

static void apus_store_run_job(ApusStore *st, ApusJob j) {
    uint8_t *buf = apus_store_buf_get(st, (size_t)j.rec->len);
    if (!buf) {
        pthread_mutex_lock(&st->mu);
        j.slot->state = APUS_SLOT_EMPTY;
        j.slot->eid = -1;
        pthread_cond_broadcast(&st->cv);
        pthread_mutex_unlock(&st->mu);
        return;
    }
    struct timespec pt0, pt1;
    clock_gettime(CLOCK_MONOTONIC, &pt0);
    apus_st_lazy_pread(j.rec->lz, j.rec->off, buf, (size_t)j.rec->len);
    clock_gettime(CLOCK_MONOTONIC, &pt1);
    uint64_t pns = (uint64_t)(pt1.tv_sec - pt0.tv_sec) * 1000000000ull
                 + (uint64_t)(pt1.tv_nsec - pt0.tv_nsec);
    apus_fadvise_dontneed(-1, j.rec->off, j.rec->len);
    if (st->test_pre_claim)
        st->test_pre_claim(st, j.layer, j.slot->eid, j.gen);
    pthread_mutex_lock(&st->mu);
    st->stats.preads++;
    st->stats.bytes_read += j.rec->len;
    st->stats.pread_ns += pns;
    if (j.slot->state == APUS_SLOT_LOADING && j.slot->gen == j.gen) {
        j.slot->buf = buf;
        j.slot->state = APUS_SLOT_READY;
        j.slot->hot = 0;
        if (j.is_pin) st->stats.pin_loads++;
    } else {
        apus_store_buf_put(st, buf);
        j.slot->state = APUS_SLOT_EMPTY;
        j.slot->hot = 0;
    }
    pthread_cond_broadcast(&st->cv);
    pthread_mutex_unlock(&st->mu);
}

static void *apus_store_worker(void *arg) {
    ApusStore *st = arg;
    for (;;) {
        pthread_mutex_lock(&st->mu);
        while (!st->jq_n && !st->stopping)
            pthread_cond_wait(&st->jq_cv, &st->mu);
        if (st->stopping && !st->jq_n) {
            pthread_mutex_unlock(&st->mu);
            return NULL;
        }
        /* hot-first pop: demand-class jobs ahead of FIFO speculative */
        int pick = 0;
        if (st->boost)
            for (int i = 0; i < st->jq_n; i++) {
                ApusJob *c = &st->jobs[(st->jq_head + i) % st->jq_cap];
                if (c->slot->hot) { pick = i; break; }
            }
        ApusJob j = st->jobs[(st->jq_head + pick) % st->jq_cap];
        for (int i = pick; i > 0; i--)
            st->jobs[(st->jq_head + i) % st->jq_cap] =
                st->jobs[(st->jq_head + i - 1) % st->jq_cap];
        st->jq_head = (st->jq_head + 1) % st->jq_cap;
        st->jq_n--;
        pthread_mutex_unlock(&st->mu);
        apus_store_run_job(st, j);
    }
}

static void apus_store_submit(ApusStore *st, int layer, ApusSlot *slot,
                              ApusSlabRec *rec, int is_pin) {
    slot->state = APUS_SLOT_LOADING;
    slot->gen = st->gen;
    if (st->n_threads > 0) {
        ApusJob j = { layer, slot, rec, is_pin, st->gen };
        apus_store_job_push(st, j);
    } else {
        ApusJob j = { layer, slot, rec, is_pin, st->gen };
        pthread_mutex_unlock(&st->mu);
        apus_store_run_job(st, j);
        pthread_mutex_lock(&st->mu);
    }
}

static int apus_store_wait_ready(ApusStore *st, int layer, ApusSlot *slot,
                                 ApusSlabRec *rec, int is_pin) {
    struct timespec t0, t1;
    int timed = 0;
    while (slot->state != APUS_SLOT_READY) {
        if (!timed) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            timed = 1;
            if (slot->state == APUS_SLOT_LOADING)
                slot->hot = 1;
        }
        if (slot->state == APUS_SLOT_EMPTY) {
            if (slot->eid < 0) return -1;
            st->stats.demand_loads++;
            slot->hot = 1;
            apus_store_submit(st, layer, slot, rec, is_pin);
        } else {
            pthread_cond_wait(&st->cv, &st->mu);
        }
    }
    if (timed) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        st->stats.waits++;
        st->stats.wait_ns += (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ull
                           + (uint64_t)(t1.tv_nsec - t0.tv_nsec);
    }
    return 0;
}

/* --- resolve / hint ------------------------------------------------------------*/

static int apus_slot_find(const ApusSlot *slots, int n, int32_t eid) {
    for (int i = 0; i < n; i++)
        if (slots[i].eid == eid) return i;
    return -1;
}

static ApusSlot *apus_ws_find(ApusLayerCache *lc, int32_t eid) {
    for (int i = 0; i < lc->ws_n; i++)
        if (lc->ws[i]->eid == eid) return lc->ws[i];
    return NULL;
}

static int apus_store_lookup(ApusStore *st, int layer, int eid,
                             ApusSlot **slot_out, int *kind_out) {
    ApusLayerCache *lc = &st->lc[layer];
    int i = apus_slot_find(lc->pins, lc->n_pins, eid);
    if (i >= 0) { *slot_out = &lc->pins[i]; *kind_out = 1; return 0; }
    ApusSlot *w = apus_ws_find(lc, eid);
    if (w) { *slot_out = w; *kind_out = 2; return 0; }
    i = apus_slot_find(lc->slots, lc->n_slots, eid);
    if (i >= 0 && lc->slots[i].buf) {
        *slot_out = &lc->slots[i];
        *kind_out = 3;
        return 0;
    }
    return -1;
}

int apus_store_resolve(ApusStore *st, int layer, int eid,
                       ApusBf16ExpertW *out) {
    if (!st || layer < 0 || layer >= st->n_layers || eid < 0 || eid >= st->E
        || !st->lc[layer].has_experts)
        return -1;
    ApusSlabRec *rec = apus_store_rec(st, layer, eid);
    pthread_mutex_lock(&st->mu);
    ApusSlot *slot = NULL;
    int kind = 0;
    if (apus_store_lookup(st, layer, eid, &slot, &kind) == 0) {
        /* hit := already cached (pin/LRU resident, or a working-set
         * entry this block already consumed). First consume of a
         * hint-loaded entry is a miss. */
        int resident = !(kind == 2 && slot->freq == 0)
                       && slot->state == APUS_SLOT_READY && slot->buf != NULL;
        if (apus_store_wait_ready(st, layer, slot, rec, kind == 1)) {
            pthread_mutex_unlock(&st->mu);
            return -1;
        }
        slot->freq++;
        slot->last = apus_clock_tick(st);
        if (resident) st->stats.hits++;
        else st->stats.misses++;
    } else {
        ApusLayerCache *lc = &st->lc[layer];
        ApusSlot *w = calloc(1, sizeof *w);
        w->eid = eid;
        w->freq = 1;
        w->hot = 1;
        if (lc->ws_n == lc->ws_cap) {
            lc->ws_cap = lc->ws_cap ? 2 * lc->ws_cap : 8;
            lc->ws = realloc(lc->ws, (size_t)lc->ws_cap * sizeof *lc->ws);
        }
        lc->ws[lc->ws_n++] = w;
        st->stats.demand_loads++;
        apus_store_submit(st, layer, w, rec, 0);
        if (apus_store_wait_ready(st, layer, w, rec, 0)) {
            pthread_mutex_unlock(&st->mu);
            return -1;
        }
        w->last = apus_clock_tick(st);
        st->stats.misses++;
        slot = w;
    }
    apus_slot_views(rec, slot->buf, out);
    pthread_mutex_unlock(&st->mu);
    return 0;
}

static void apus_store_hint_impl(ApusStore *st, int layer, int eid,
                                 int demand) {
    if (!st || layer < 0 || layer >= st->n_layers || eid < 0 || eid >= st->E
        || !st->lc[layer].has_experts)
        return;
    pthread_mutex_lock(&st->mu);
    ApusSlot *slot;
    int kind;
    if (apus_store_lookup(st, layer, eid, &slot, &kind) == 0) {
        if (slot->state == APUS_SLOT_EMPTY && slot->eid >= 0) {
            slot->hot = demand ? 1 : slot->hot;
            st->stats.hint_loads++;
            apus_store_submit(st, layer, slot, apus_store_rec(st, layer, eid),
                              kind == 1);
        } else if (demand && slot->state == APUS_SLOT_LOADING) {
            slot->hot = 1;
        }
        pthread_mutex_unlock(&st->mu);
        return;
    }
    ApusLayerCache *lc = &st->lc[layer];
    ApusSlot *w = calloc(1, sizeof *w);
    w->eid = eid;
    w->freq = 0;
    w->hot = demand ? 1 : 0;
    if (lc->ws_n == lc->ws_cap) {
        lc->ws_cap = lc->ws_cap ? 2 * lc->ws_cap : 8;
        lc->ws = realloc(lc->ws, (size_t)lc->ws_cap * sizeof *lc->ws);
    }
    lc->ws[lc->ws_n++] = w;
    st->stats.hint_loads++;
    apus_store_submit(st, layer, w, apus_store_rec(st, layer, eid), 0);
    pthread_mutex_unlock(&st->mu);
}

void apus_store_hint(ApusStore *st, int layer, int eid) {
    apus_store_hint_impl(st, layer, eid, 0);
}

void apus_store_hint_demand(ApusStore *st, int layer, int eid) {
    apus_store_hint_impl(st, layer, eid, 1);
}

/* --- end-of-block promotion -----------------------------------------------------*/

static ApusSlot *apus_lru_victim(ApusLayerCache *lc) {
    ApusSlot *best = NULL;
    for (int i = 0; i < lc->n_slots; i++) {
        ApusSlot *s = &lc->slots[i];
        if (s->state == APUS_SLOT_LOADING) continue;
        if (s->eid < 0) return s;
        if (!best || s->last < best->last) best = s;
    }
    return best;
}

void apus_store_layer_end(ApusStore *st, int layer) {
    if (!st || layer < 0 || layer >= st->n_layers) return;
    pthread_mutex_lock(&st->mu);
    ApusLayerCache *lc = &st->lc[layer];
    int out = 0;
    for (int i = 0; i < lc->ws_n; i++) {
        ApusSlot *w = lc->ws[i];
        if (w->state != APUS_SLOT_READY) {
            lc->ws[out++] = w;      /* in-flight pilot hint: keep */
            continue;
        }
        ApusSlot *v = apus_lru_victim(lc);
        if (v && v->last <= w->last) {
            if (v->eid >= 0 && v->buf) st->stats.evictions++;
            apus_store_buf_put(st, v->buf);
            v->eid = w->eid;
            v->buf = w->buf;
            v->freq = w->freq;
            v->last = w->last;
            v->state = APUS_SLOT_READY;
            v->hot = 0;
            free(w);
        } else {
            apus_store_buf_put(st, w->buf);
            free(w);
        }
    }
    lc->ws_n = out;
    st->gen++;
    pthread_mutex_unlock(&st->mu);
    apus_store_rss_guard(st);
}

/* --- RSS guard ------------------------------------------------------------------*/

void apus_store_rss_guard(ApusStore *st) {
    uint64_t rss = apus_rss_bytes();
    if (!rss || rss <= st->rss_budget) return;
    uint64_t excess = rss - st->rss_budget;
    pthread_mutex_lock(&st->mu);
    size_t total = 0;
    for (int l = 0; l < st->n_layers; l++)
        total += (size_t)st->lc[l].n_slots;
    ApusSlot **ord = malloc(total * sizeof *ord);
    size_t n = 0;
    for (int l = 0; l < st->n_layers; l++)
        for (int i = 0; i < st->lc[l].n_slots; i++) {
            ApusSlot *s = &st->lc[l].slots[i];
            if (s->state == APUS_SLOT_READY && s->buf) ord[n++] = s;
        }
    for (size_t i = 0; i + 1 < n; i++)
        for (size_t j = i + 1; j < n; j++)
            if (ord[j]->last < ord[i]->last) {
                ApusSlot *t = ord[i]; ord[i] = ord[j]; ord[j] = t;
            }
    uint64_t freed = 0;
    for (size_t i = 0; i < n && freed < excess; i++) {
        apus_aligned_free(ord[i]->buf);   /* slab buffer (M13 pairing) */
        ord[i]->buf = NULL;
        ord[i]->state = APUS_SLOT_EMPTY;
        freed += st->slab_bytes;
        st->stats.rss_drops++;
    }
    free(ord);
    pthread_mutex_unlock(&st->mu);
}

/* --- LFRU REPIN ------------------------------------------------------------------*/

static int apus_lfru_beats(uint64_t c_freq, uint64_t c_last,
                           uint64_t p_freq, uint64_t p_last) {
    uint64_t need = p_freq + p_freq / 4 + 4;      /* 25% + 4 hysteresis */
    if (c_freq >= need) return 1;
    if (c_freq == p_freq && c_last > p_last && c_freq >= need / 2) return 1;
    return 0;
}

void apus_store_repin(ApusStore *st) {
    pthread_mutex_lock(&st->mu);
    for (int l = 0; l < st->n_layers; l++) {
        ApusLayerCache *lc = &st->lc[l];
        if (!lc->n_pins) continue;
        for (int iter = 0; iter < lc->n_pins; iter++) {
            ApusSlot *pin = NULL;
            for (int i = 0; i < lc->n_pins; i++) {
                ApusSlot *p = &lc->pins[i];
                if (p->eid < 0) continue;
                if (!pin || p->freq < pin->freq
                    || (p->freq == pin->freq && p->last < pin->last))
                    pin = p;
            }
            ApusSlot *cand = NULL;
            for (int i = 0; i < lc->n_slots; i++) {
                ApusSlot *s = &lc->slots[i];
                if (s->eid < 0 || s->state != APUS_SLOT_READY || !s->freq)
                    continue;
                if (!cand || s->freq > cand->freq
                    || (s->freq == cand->freq && s->last > cand->last))
                    cand = s;
            }
            if (!pin || !cand
                || !apus_lfru_beats(cand->freq, cand->last,
                                    pin->freq, pin->last))
                break;
            int32_t te = pin->eid;
            uint8_t *tb = pin->buf;
            uint64_t tf = pin->freq, tl = pin->last;
            int tst = pin->state;
            pin->eid = cand->eid; pin->buf = cand->buf;
            pin->freq = cand->freq; pin->last = cand->last;
            pin->state = cand->state;
            cand->eid = te; cand->buf = tb;
            cand->freq = tf; cand->last = tl;
            cand->state = tst;
            st->stats.repin_swaps++;
        }
    }
    pthread_mutex_unlock(&st->mu);
}

/* --- open / close ------------------------------------------------------------------*/

/* Parse apus.index.json expert_slabs into (layer, eid) -> (shard, off,
 * len) hints. Returns the number of records, or -1 when the manifest is
 * absent (header derivation is then the fallback). A PRESENT manifest
 * must be FORMAT_VERSION 2 (2-member fused slabs) — anything else is a
 * hard open error, never a silent misread: fills err and returns -2. */
static int apus_store_manifest_slabs(ApusStore *st, const char *dir,
                                     ApusExpertTensorRef **refs_out,
                                     uint64_t **offs_out,
                                     uint64_t **lens_out, int *n_out,
                                     char *err, size_t errcap) {
    char path[1200], jerr[128];
    snprintf(path, sizeof path, "%s/apus.index.json", dir);
    JVal *idx = json_parse_file(path, jerr, sizeof jerr);
    if (!idx) return -1;
    JVal *fv = json_obj_get(idx, "format_version");
    long fvn = fv && json_type(fv) == J_NUM ? (long)json_num(fv) : -1;
    if (fvn != 2) {
        snprintf(err, errcap,
                 "store: %s format_version %ld != 2 — this build reads "
                 "the v2 2-member fused slab (gate_up+down); a v1 "
                 "3-member-slab container must be reconverted "
                 "(tools/convert.py)", path, fvn);
        json_free(idx);
        return -2;
    }
    JVal *sl = json_obj_get(idx, "expert_slabs");
    if (!sl || json_type(sl) != J_ARR) {
        json_free(idx);
        return -1;
    }
    int n = (int)json_arr_len(sl);
    ApusExpertTensorRef *refs = malloc((size_t)n * 2 * sizeof *refs);
    uint64_t *offs = malloc((size_t)n * sizeof *offs);
    uint64_t *lens = malloc((size_t)n * sizeof *lens);
    int nv = 0;
    for (int i = 0; i < n; i++) {
        JVal *r = json_arr_get(sl, i);
        int layer = (int)json_num(json_obj_get(r, "layer"));
        int eid = (int)json_num(json_obj_get(r, "expert"));
        const char *shard = json_str(json_obj_get(r, "shard"));
        uint64_t off = (uint64_t)json_num(json_obj_get(r, "offset"));
        uint64_t len = (uint64_t)json_num(json_obj_get(r, "nbytes"));
        if (layer < 0 || layer >= st->n_layers || eid < 0 || eid >= st->E)
            continue;      /* out of range: mtp records (M8) etc. */
        for (int m = 0; m < 2; m++) {
            ApusExpertTensorRef *rf = &refs[(size_t)nv * 2 + m];
            rf->layer = layer;
            rf->eid = eid;
            rf->member = m;
            snprintf(rf->shard, sizeof rf->shard, "%s", shard);
        }
        offs[nv] = off;
        lens[nv] = len;
        nv++;
    }
    json_free(idx);
    *refs_out = refs;
    *offs_out = offs;
    *lens_out = lens;
    *n_out = nv;
    return 0;
}

ApusStore *apus_store_open(const char *model_dir, const ApusStoreCfg *cfg,
                           char *err, size_t errcap) {
    ApusStoreCfg c = cfg ? *cfg : (ApusStoreCfg){0};
    if (c.n_layers <= 0 || c.n_experts <= 0) {
        snprintf(err, errcap, "store: n_layers/n_experts required");
        return NULL;
    }
    char dir[1024], path[1200];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json",
             model_dir);
    FILE *f = fopen(path, "rb");
    if (f) {
        fclose(f);
        snprintf(dir, sizeof dir, "%s", model_dir);
    } else {
        snprintf(dir, sizeof dir, "%s/weights", model_dir);
    }
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    char jerr[128];
    JVal *idx = json_parse_file(path, jerr, sizeof jerr);
    if (!idx) { snprintf(err, errcap, "store: %s", jerr); return NULL; }
    JVal *wm = json_obj_get(idx, "weight_map");
    if (!wm || json_type(wm) != J_OBJ) {
        json_free(idx);
        snprintf(err, errcap, "store: no weight_map in %s", path);
        return NULL;
    }

    ApusStore *st = calloc(1, sizeof *st);
    st->n_layers = c.n_layers;
    st->n_main = c.n_main_layers > 0 ? c.n_main_layers : c.n_layers;
    st->E = c.n_experts;
    pthread_mutex_init(&st->mu, NULL);
    pthread_cond_init(&st->cv, NULL);
    pthread_cond_init(&st->jq_cv, NULL);

    /* expert_slabs manifest records (optional; header derivation is the
     * fallback and the verification). A present manifest must be
     * FORMAT_VERSION 2 — a v1 container is rejected, never misread. */
    ApusExpertTensorRef *mrefs = NULL;
    uint64_t *moffs = NULL, *mlens = NULL;
    int n_mani = 0;
    if (apus_store_manifest_slabs(st, dir, &mrefs, &moffs, &mlens, &n_mani,
                                  err, errcap) == -2) {
        json_free(idx);
        apus_store_close(st);
        return NULL;
    }

    /* collect expert tensor refs from the weight_map */
    int need = c.n_layers * c.n_experts * 2;
    ApusExpertTensorRef *refs = malloc((size_t)need * sizeof *refs);
    int *have = calloc((size_t)c.n_layers * c.n_experts, sizeof(int));
    int n_refs = 0;
    for (size_t i = 0; i < json_obj_len(wm); i++) {
        const char *name = json_obj_key(wm, i);
        int L, e, mb;
        if (apus_parse_expert_name(name, st->n_main, &L, &e, &mb))
            continue;
        if (L >= c.n_layers || e >= c.n_experts) continue;
        ApusExpertTensorRef *r = &refs[n_refs++];
        r->layer = L;
        r->eid = e;
        r->member = mb;
        snprintf(r->shard, sizeof r->shard, "%s",
                 json_str(json_obj_val(wm, i)));
        have[(size_t)L * c.n_experts + e]++;
    }
    json_free(idx);
    st->lc = calloc((size_t)c.n_layers, sizeof *st->lc);
    int n_moe = 0;
    for (int l = 0; l < c.n_layers; l++) {
        for (int e = 0; e < c.n_experts; e++)
            if (have[(size_t)l * c.n_experts + e]) {
                st->lc[l].has_experts = 1;
                break;
            }
        if (st->lc[l].has_experts) n_moe++;
    }
    st->n_moe = n_moe;
    if (n_moe == 0) {
        /* no v2 slab slices matched at all — this is a wrong-container
         * error (e.g. a v1 3-member container without a manifest), not
         * a valid 0-MoE model: this model has MoE on every layer. */
        snprintf(err, errcap,
                 "store: no v2 expert slabs (gate_up_proj/down_proj "
                 "slices) found in %s — a v1 3-member container is not "
                 "readable by this build", dir);
        free(refs); free(have);
        free(mrefs); free(moffs); free(mlens);
        apus_store_close(st);
        return NULL;
    }
    for (int i = 0; i < c.n_layers * c.n_experts; i++) {
        int l = i / c.n_experts;
        if (!st->lc[l].has_experts) continue;
        if (have[i] != 2) {
            snprintf(err, errcap,
                     "store: expert %d.%d has %d/2 tensors in weight_map",
                     i / c.n_experts, i % c.n_experts, have[i]);
            free(refs); free(have);
            free(mrefs); free(moffs); free(mlens);
            apus_store_close(st);
            return NULL;
        }
    }
    free(have);

    int nocache = c.nocache > 0 ? 1
                : c.nocache < 0 ? 0
                : apus_env_int("APUS_NOCACHE", 1);

    st->recs = calloc((size_t)c.n_layers * c.n_experts, sizeof *st->recs);
    qsort(refs, (size_t)n_refs, sizeof *refs, apus_ref_cmp);
    /* manifest hints keyed by (layer, eid) */
    for (int i = 0; i < n_refs; i += 2) {
        int64_t oh = -1, lh = -1;
        for (int j = 0; j < n_mani; j++) {
            if (mrefs[(size_t)j * 2].layer == refs[i].layer
                && mrefs[(size_t)j * 2].eid == refs[i].eid) {
                oh = (int64_t)moffs[j];
                lh = (int64_t)mlens[j];
                break;
            }
        }
        if (apus_store_derive_slab(st, dir, &refs[i], oh, lh, nocache,
                                   err, errcap)) {
            free(refs); free(mrefs); free(moffs); free(mlens);
            apus_store_close(st);
            return NULL;
        }
    }
    free(refs);
    free(mrefs); free(moffs); free(mlens);

    /* budgets */
    if (!c.cache_bytes)
        c.cache_bytes = apus_env_mb("APUS_EXPERT_CACHE_MB", 4096) << 20;
    if (!c.pin_bytes)
        c.pin_bytes = apus_env_mb("APUS_PIN_MB", 512) << 20;
    if (!c.rss_budget_bytes)
        c.rss_budget_bytes = apus_env_mb("APUS_RSS_GUARD_MB", 26624) << 20;
    if (!c.io_threads)
        c.io_threads = apus_env_int("APUS_IO_THREADS", 4);
    st->boost = apus_env_int("APUS_STORE_BOOST", 1);
    st->rss_budget = c.rss_budget_bytes;
    if (c.usage_decay == 0.0) {
        const char *d = getenv("APUS_USAGE_DECAY");
        c.usage_decay = d ? atof(d) : 1.0;
        if (c.usage_decay <= 0.0 || c.usage_decay > 1.0)
            c.usage_decay = 1.0;
    }
    st->usage_decay = c.usage_decay;

    int spl = c.slots_per_layer;
    if (spl <= 0) {
        size_t per = c.cache_bytes
                     / ((size_t)(n_moe ? n_moe : 1) * st->slab_bytes);
        spl = per ? (int)per : 1;
    }
    if (spl > c.n_experts) spl = c.n_experts;
    int ppl = c.pins_per_layer;
    if (ppl < 0) ppl = 0;
    if (c.pins_per_layer == 0) {
        size_t per = c.pin_bytes
                     / ((size_t)(n_moe ? n_moe : 1) * st->slab_bytes);
        ppl = (int)per;
    }
    if (ppl > c.n_experts) ppl = c.n_experts;

    for (int l = 0; l < c.n_layers; l++) {
        ApusLayerCache *lc = &st->lc[l];
        lc->n_slots = spl;
        lc->slots = calloc((size_t)spl, sizeof *lc->slots);
        for (int i = 0; i < spl; i++) lc->slots[i].eid = -1;
        lc->n_pins = ppl;
        lc->pins = calloc((size_t)(ppl ? ppl : 1), sizeof *lc->pins);
        for (int i = 0; i < ppl; i++) lc->pins[i].eid = -1;
    }

    if (!c.usage_path) c.usage_path = getenv("APUS_USAGE_PATH");
    if (!c.usage_path) {
        snprintf(st->usage_path, sizeof st->usage_path, "%s/apus.usage",
                 model_dir);
        st->usage_enabled = 1;
    } else if (*c.usage_path) {
        snprintf(st->usage_path, sizeof st->usage_path, "%s", c.usage_path);
        st->usage_enabled = 1;
    }
    if (st->usage_enabled) apus_store_load_usage(st);

    if (c.io_threads > 0) {
        st->n_threads = c.io_threads;
        st->threads = calloc((size_t)st->n_threads, sizeof *st->threads);
        for (int i = 0; i < st->n_threads; i++)
            pthread_create(&st->threads[i], NULL, apus_store_worker, st);
    }
    st->buf_free_cap = apus_env_int("APUS_BUF_FREE", APUS_BUF_FREE_MAX);
    if (st->buf_free_cap < 0) st->buf_free_cap = 0;
    if (st->buf_free_cap > 1024) st->buf_free_cap = 1024;
    st->buf_free = calloc((size_t)st->buf_free_cap, sizeof *st->buf_free);
    if (!st->buf_free) st->buf_free_cap = 0;
    return st;
}

void apus_store_close(ApusStore *st) {
    if (!st) return;
    pthread_mutex_lock(&st->mu);
    st->stopping = 1;
    pthread_cond_broadcast(&st->jq_cv);
    pthread_mutex_unlock(&st->mu);
    for (int i = 0; i < st->n_threads; i++)
        pthread_join(st->threads[i], NULL);
    free(st->threads);
    free(st->jobs);
    if (st->lc) {
        for (int l = 0; l < st->n_layers; l++) {
            ApusLayerCache *lc = &st->lc[l];
            /* slab buffers come from apus_store_buf_get → apus_slab_alloc →
             * apus_aligned_alloc — they MUST go to apus_aligned_free
             * (Windows _aligned_malloc storage aborts on plain free();
             * the M15 heap-corruption bug, invisible on POSIX). */
            for (int i = 0; i < lc->n_slots; i++) apus_aligned_free(lc->slots[i].buf);
            for (int i = 0; i < lc->n_pins; i++) apus_aligned_free(lc->pins[i].buf);
            for (int i = 0; i < lc->ws_n; i++) {
                apus_aligned_free(lc->ws[i]->buf);
                free(lc->ws[i]);
            }
            free(lc->slots);
            free(lc->pins);
            free(lc->ws);
        }
        free(st->lc);
    }
    for (int i = 0; i < st->shards_n; i++) {
        free(st->shards[i].name);
        apus_st_lazy_close(st->shards[i].lz);
    }
    free(st->shards);
    free(st->recs);
    for (int i = 0; i < st->buf_free_n; i++) apus_aligned_free(st->buf_free[i]);
    free(st->buf_free);
    pthread_mutex_destroy(&st->mu);
    pthread_cond_destroy(&st->cv);
    pthread_cond_destroy(&st->jq_cv);
    free(st);
}

/* --- stats / misc ------------------------------------------------------------------*/

void apus_store_stats(const ApusStore *st, ApusStoreStats *out) {
    pthread_mutex_lock(&((ApusStore *)st)->mu);
    *out = st->stats;
    pthread_mutex_unlock(&((ApusStore *)st)->mu);
}

size_t apus_store_slab_bytes(const ApusStore *st) { return st->slab_bytes; }

size_t apus_store_resident_bytes(ApusStore *st) {
    pthread_mutex_lock(&st->mu);
    size_t n = 0;
    for (int l = 0; l < st->n_layers; l++) {
        ApusLayerCache *lc = &st->lc[l];
        for (int i = 0; i < lc->n_slots; i++)
            if (lc->slots[i].buf) n += st->slab_bytes;
        for (int i = 0; i < lc->n_pins; i++)
            if (lc->pins[i].buf) n += st->slab_bytes;
        for (int i = 0; i < lc->ws_n; i++)
            if (lc->ws[i]->buf) n += st->slab_bytes;
    }
    pthread_mutex_unlock(&st->mu);
    return n;
}

/* --- introspection / test hooks -----------------------------------------------------*/

int apus_store_debug_layer(ApusStore *st, int layer,
                           int32_t *lru_eids, int n_lru,
                           int32_t *pin_eids, int n_pins) {
    if (!st || layer < 0 || layer >= st->n_layers) return -1;
    pthread_mutex_lock(&st->mu);
    ApusLayerCache *lc = &st->lc[layer];
    if (lru_eids) {
        int n = n_lru < lc->n_slots ? n_lru : lc->n_slots;
        for (int i = 0; i < n; i++)
            lru_eids[i] = lc->slots[i].buf ? lc->slots[i].eid : -1;
    }
    if (pin_eids) {
        int n = n_pins < lc->n_pins ? n_pins : lc->n_pins;
        for (int i = 0; i < n; i++) pin_eids[i] = lc->pins[i].eid;
    }
    pthread_mutex_unlock(&st->mu);
    return 0;
}

void apus_store_debug_set_pre_claim(ApusStore *st,
    void (*fn)(ApusStore *st, int layer, int32_t eid, uint64_t gen)) {
    st->test_pre_claim = fn;
}

int apus_store_debug_stale_gen(ApusStore *st, int layer, int32_t eid) {
    pthread_mutex_lock(&st->mu);
    ApusLayerCache *lc = &st->lc[layer];
    ApusSlot *w = apus_ws_find(lc, eid);
    if (!w || w->state != APUS_SLOT_LOADING) {
        pthread_mutex_unlock(&st->mu);
        return -1;
    }
    w->gen += 1000;
    pthread_mutex_unlock(&st->mu);
    return 0;
}

int apus_store_debug_present(ApusStore *st, int layer, int eid) {
    if (!st || layer < 0 || layer >= st->n_layers || eid < 0 || eid >= st->E
        || !st->lc[layer].has_experts)
        return 0;
    pthread_mutex_lock(&st->mu);
    ApusSlot *slot;
    int kind;
    int present = apus_store_lookup(st, layer, eid, &slot, &kind) == 0
                  && (slot->buf != NULL || slot->state == APUS_SLOT_LOADING);
    pthread_mutex_unlock(&st->mu);
    return present;
}

int apus_store_debug_ready(ApusStore *st, int layer, int eid) {
    if (!st || layer < 0 || layer >= st->n_layers || eid < 0 || eid >= st->E
        || !st->lc[layer].has_experts)
        return 0;
    pthread_mutex_lock(&st->mu);
    ApusSlot *slot;
    int kind;
    int ready = apus_store_lookup(st, layer, eid, &slot, &kind) == 0
                && slot->state == APUS_SLOT_READY && slot->buf != NULL;
    pthread_mutex_unlock(&st->mu);
    return ready;
}

/* --- tiered forward + pilot hook surface ---------------------------------------------*/

void apus_store_fwd_hooks(ApusStore *st, const ApusStoreFwdHooks *h) {
    if (!st) return;
    pthread_mutex_lock(&st->mu);
    if (h) st->hooks = *h;
    else memset(&st->hooks, 0, sizeof st->hooks);
    pthread_mutex_unlock(&st->mu);
}

void apus_store_fwd_set_batch(ApusStore *st, int s, int t) {
    if (!st) return;
    pthread_mutex_lock(&st->mu);
    st->fwd_s = s;
    st->fwd_t = t;
    pthread_mutex_unlock(&st->mu);
}

/* Attention halves: EXACT mirrors of c/layer.h's static
 * apus_layer_gdn_attn / apus_layer_full_attn (identical public ops in
 * the identical order; c/layer.h cannot be touched and its halves are
 * static). Any drift shows up as a bitwise mismatch in tests/m6a. */
static void apus_store_gdn_attn(const ApusLayerCfg *c,
                                const ApusLayerW *w, ApusLayerState *st,
                                const uint16_t *ln1, uint16_t *ao) {
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
        ob[i] = apus_bf16_bits(reco[i]);
    apus_gdn_onorm(ob, z, w->onorm_w, on, hv, dv);
    apus_bf16_gemv_hot(w->w_out, on, ao, H, value_dim);
}

static void apus_store_full_attn(const ApusLayerCfg *c,
                                 const ApusLayerW *w, ApusLayerState *st,
                                 const uint16_t *ln1, uint16_t *ao) {
    size_t H = c->hidden, nh = c->attn_nh, nkv = c->attn_nkv;
    size_t d = c->attn_d, rot = c->attn_rot, pos = st->pos;
    float scale = (float)(1.0 / sqrt((double)d));
    uint16_t qg[nh * 2 * d], qn[nh * d], k[nkv * d], kn[nkv * d];
    uint16_t v[nkv * d], gate[nh * d], attno[nh * d], og[nh * d];
    apus_bf16_gemv_hot(w->wq, ln1, qg, nh * 2 * d, H);
    apus_bf16_gemv_hot(w->wk, ln1, k, nkv * d, H);
    apus_bf16_gemv_hot(w->wv, ln1, v, nkv * d, H);
    for (size_t h = 0; h < nh; h++) {
        apus_attn_rmsnorm(qg + h * 2 * d, w->qn_w, qn + h * d, d);
        memcpy(gate + h * d, qg + h * 2 * d + d, d * 2);
    }
    for (size_t h = 0; h < nkv; h++)
        apus_attn_rmsnorm(k + h * d, w->kn_w, kn + h * d, d);
    for (size_t h = 0; h < nh; h++)
        apus_attn_rope(qn + h * d, qn + h * d, d, rot, (float)pos,
                       c->rope_theta);
    for (size_t h = 0; h < nkv; h++)
        apus_attn_rope(kn + h * d, kn + h * d, d, rot, (float)pos,
                       c->rope_theta);
    memcpy(st->kcache + pos * nkv * d, kn, nkv * d * 2);
    memcpy(st->vcache + pos * nkv * d, v, nkv * d * 2);
    apus_attn_gqa_decode_mt(qn, st->kcache, st->vcache, attno, pos + 1,
                            nh, nkv, d, scale);
    apus_attn_outgate(attno, gate, og, nh * d);
    apus_bf16_gemv_hot(w->wo, og, ao, H, nh * d);
}

/* Store-backed MoE FFN: mirrors c/layer.h's apus_layer_moe exactly, with
 * expert GEMVs on store-resolved slab views (batch-union demand hints
 * first, then just-in-time resolves — same gemv/act/combine kernels,
 * same order, hence bitwise the eager path). The v2 slab view's gate_up
 * member IS the fused [gate|up] [2I, H] slice — the same bytes the
 * eager path's exp_gu array holds, so the fused GEMV is identical. */
static void apus_store_moe(ApusStore *st, const ApusLayerCfg *c,
                           const ApusLayerW *w, int layer,
                           const uint16_t *ln2, uint16_t *moe_out,
                           int64_t pos) {
    size_t H = c->hidden;
    size_t I = c->moe_inter, Is = c->shared_inter, TK = c->top_k;
    int32_t idx[TK];
    uint16_t rw[TK];
    uint16_t gu[2 * I], a[I], ye[TK * H], sg[Is], su[Is], sa[Is], sy[H];
    uint16_t routed[H], sgl[1], sgat[1], shared[H];
    apus_moe_route(ln2, w->rtr_w, idx, rw, c->experts, H, TK);
    if (st->hooks.router_actual)
        st->hooks.router_actual(st->hooks.ctx, layer, idx, (int)TK, pos,
                                st->fwd_s);
    for (size_t i = 0; i < TK; i++)
        apus_store_hint_demand(st, layer, idx[i]);
    for (size_t i = 0; i < TK; i++) {
        ApusBf16ExpertW ew;
        if (apus_store_resolve(st, layer, idx[i], &ew)) {
            fprintf(stderr, "store: resolve L%d E%d failed\n", layer,
                    idx[i]);
            abort();
        }
        /* M10: the _cpu (no-hook) variants — store slabs are TRANSIENT
         * (LRU), so the Metal backend's pointer-keyed zero-copy cache
         * must never see them (the stable-pointer invariant). Resident
         * weights (router, shared expert, attention) go through the
         * hooked wrappers as usual. */
        apus_bf16_gemv_hot_cpu(ew.gate_up, ln2, gu, 2 * I, H);
        apus_moe_silu_act(gu, a, I);
        apus_bf16_gemv_hot_cpu(ew.down, a, ye + i * H, H, I);
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
    for (size_t i = 0; i < H; i++)
        moe_out[i] = apus_bf16_bits(apus_bf16_f32(routed[i])
                                    + apus_bf16_f32(shared[i]));
    apus_store_layer_end(st, layer);
}

/* --- M9 batched tiered prefill (T>1) -----------------------------------------
 * Mirrors c/layer.h's M9 batched prefill EXACTLY (same phases, same
 * unique-expert grouping, same per-(t,j) values), with expert weights
 * resolved from the store — batch-union demand hints, each unique
 * expert's fused gate_up/down linears ONCE at M=count from its slab
 * views (the no-hook _cpu variants: slabs are TRANSIENT), one layer_end
 * at the end (the Apus block semantics). Same M-independent-bitwise
 * class as the eager path (gemm_hot only, NO BLAS): BITWISE identical
 * to the sequential tiered path and to the eager batched path. */

/* Mirrors of c/layer.h's M9 batched-attention helpers (identical public
 * ops in the identical order; drift shows up as a bitwise mismatch in
 * tests/m6a/m9c). No traces here — the store path uses hooks. */
static void apus_store_attn_batch_gdn(const ApusLayerCfg *c,
                                      const ApusLayerW *w,
                                      ApusLayerState *st,
                                      const uint16_t *ln1, uint16_t *ao,
                                      size_t tc) {
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
    apus_bf16_gemm_hot(w->w_qkv, ln1, qkv, tc, conv_dim, H);
    apus_bf16_gemm_hot(w->w_z, ln1, z, tc, value_dim, H);
    apus_bf16_gemm_hot(w->w_b, ln1, b, tc, hv, H);
    apus_bf16_gemm_hot(w->w_a, ln1, a, tc, hv, H);
    apus_gdn_conv1d(qkv, w->conv_w, qc, conv_dim, tc, st->conv_state);
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
            ob[i] = apus_bf16_bits(reco[i]);
        apus_gdn_onorm(ob, z + t * value_dim, w->onorm_w,
                       on + t * value_dim, hv, dv);
    }
    apus_bf16_gemm_hot(w->w_out, on, ao, tc, H, value_dim);
    apus_scratch_reset(mk);
}

static void apus_store_attn_batch_full(const ApusLayerCfg *c,
                                       const ApusLayerW *w,
                                       ApusLayerState *st,
                                       const uint16_t *ln1, uint16_t *ao,
                                       size_t tc) {
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
    }
    apus_attn_gqa_mt(qn, st->kcache, st->vcache, attno, tc, pos0 + tc,
                     nh, nkv, d, scale);
    for (size_t t = 0; t < tc; t++)
        apus_attn_outgate(attno + t * nh * d, gate + t * nh * d,
                          og + t * nh * d, nh * d);
    apus_bf16_gemm_hot(w->wo, og, ao, tc, H, nh * d);
    apus_scratch_reset(mk);
}

static void apus_store_prefill(ApusStore *st, int layer,
                               const ApusLayerCfg *c, const ApusLayerW *w,
                               ApusLayerState *state, const uint16_t *x,
                               uint16_t *out, size_t T) {
    size_t H = c->hidden, E = c->experts;
    size_t I = c->moe_inter, Is = c->shared_inter, TK = c->top_k;
    int64_t base = (int64_t)state->pos;
    ApusScratchMark mk = apus_scratch_mark();
    uint16_t *res1 = apus_scratch_alloc(T * H * 2);
    uint16_t *ln2 = apus_scratch_alloc(T * H * 2);
    uint16_t *moe_out = apus_scratch_alloc(T * H * 2);
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
    /* phase A: attention + norms, chunked (mirrors c/layer.h's
     * APUS_PREFILL_ATTN_CHUNK batching; the chunk changes no bits) */
    for (size_t t0 = 0; t0 < T; t0 += APUS_PREFILL_ATTN_CHUNK) {
        size_t tc = T - t0 < APUS_PREFILL_ATTN_CHUNK
            ? T - t0 : APUS_PREFILL_ATTN_CHUNK;
        ApusScratchMark ckm = apus_scratch_mark();
        uint16_t *ln1 = apus_scratch_alloc(tc * H * 2);
        uint16_t *ao = apus_scratch_alloc(tc * H * 2);
        for (size_t t = 0; t < tc; t++)
            apus_attn_rmsnorm(x + (t0 + t) * H, w->ln1_w, ln1 + t * H, H);
        if (c->kind == APUS_LAYER_FULL)
            apus_store_attn_batch_full(c, w, state, ln1, ao, tc);
        else
            apus_store_attn_batch_gdn(c, w, state, ln1, ao, tc);
        state->pos += tc;
        for (size_t t = 0; t < tc; t++) {
            for (size_t i = 0; i < H; i++)
                res1[(t0 + t) * H + i] = apus_bf16_bits(
                    apus_bf16_f32(x[(t0 + t) * H + i])
                    + apus_bf16_f32(ao[t * H + i]));
            if (st->hooks.post_attn)
                st->hooks.post_attn(st->hooks.ctx, layer,
                                    res1 + (t0 + t) * H,
                                    base + (int64_t)(t0 + t), st->fwd_s,
                                    (int)(t0 + t));
            apus_attn_rmsnorm(res1 + (t0 + t) * H, w->ln2_w,
                              ln2 + (t0 + t) * H, H);
        }
        apus_scratch_reset(ckm);
    }
    /* phase B: router per token + batch-union hints */
    for (size_t t = 0; t < T; t++) {
        apus_moe_route(ln2 + t * H, w->rtr_w, idx + t * TK, rw + t * TK,
                       E, H, TK);
        if (st->hooks.router_actual)
            st->hooks.router_actual(st->hooks.ctx, layer, idx + t * TK,
                                    (int)TK, base + (int64_t)t,
                                    st->fwd_s);
        for (size_t j = 0; j < TK; j++)
            apus_store_hint_demand(st, layer, idx[t * TK + j]);
    }
    /* shared expert at M=T (resident weights — the hooked variant) */
    apus_bf16_gemm_hot(w->sh_g, ln2, sg, T, Is, H);
    apus_bf16_gemm_hot(w->sh_u, ln2, su, T, Is, H);
    apus_moe_silu_mul(sg, su, sa, T * Is);
    apus_bf16_gemm_hot(w->sh_d, sa, sy, T, H, Is);
    apus_bf16_gemm_hot(w->sh_gate, ln2, sgl, T, 1, H);
    /* unique experts: resolved once, gate_up/down at M=count (the
     * no-hook _cpu variants — slab views are TRANSIENT) */
    for (size_t e = 0; e < E; e++) {
        size_t cnt = 0;
        for (size_t t = 0; t < T; t++)
            for (size_t j = 0; j < TK; j++)
                if ((size_t)idx[t * TK + j] == e) {
                    memcpy(xg + cnt * H, ln2 + t * H, H * 2);
                    cnt++;
                }
        if (!cnt) continue;
        ApusBf16ExpertW ew;
        if (apus_store_resolve(st, layer, (int)e, &ew)) {
            fprintf(stderr, "store: resolve L%d E%zu failed\n", layer, e);
            abort();
        }
        apus_bf16_gemm_hot_cpu(ew.gate_up, xg, gu, cnt, 2 * I, H);
        for (size_t r = 0; r < cnt; r++)
            apus_moe_silu_act(gu + r * 2 * I, a + r * I, I);
        uint16_t *eb = apus_scratch_alloc(cnt * H * 2);
        apus_bf16_gemm_hot_cpu(ew.down, a, eb, cnt, H, I);
        size_t r = 0;
        for (size_t t = 0; t < T; t++)
            for (size_t j = 0; j < TK; j++)
                if ((size_t)idx[t * TK + j] == e) {
                    memcpy(eo + (t * TK + j) * H, eb + r * H, H * 2);
                    r++;
                }
    }
    /* per-token combine + sigmoid-gated shared + residuals (the
     * sequential body's exact per-token tail) */
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
        }
        for (size_t i = 0; i < H; i++)
            out[t * H + i] = apus_bf16_bits(
                apus_bf16_f32(res1[t * H + i])
                + apus_bf16_f32(moe_out[t * H + i]));
    }
    apus_store_layer_end(st, layer);
    apus_scratch_reset(mk);
}

/* Tiered layer forward: T=1 mirrors c/layer.h's per-token body EXACTLY
 * (same kernels, same order), T>1 mirrors the M9 batched prefill below;
 * the MoE FFN's expert linears read store-resolved slab views instead of
 * eager arrays. Same slab bytes + same kernels + same accumulation
 * order => BITWISE identical to the eager path (the M6a hard gate,
 * re-anchored after M4's kernel rewrite). */
void apus_store_layer_forward(ApusStore *st, int layer,
                              const ApusLayerCfg *c,
                              const ApusLayerW *w, ApusLayerState *state,
                              const uint16_t *x, uint16_t *out, size_t T) {
    if (T > 1) {
        apus_store_prefill(st, layer, c, w, state, x, out, T);
        return;
    }
    size_t H = c->hidden;
    for (size_t t = 0; t < T; t++) {
        const uint16_t *xt = x + t * H;
        uint16_t *ot = out + t * H;
        int64_t pos = (int64_t)state->pos;
        uint16_t ln1[H], ao[H], res1[H], ln2[H], moe_out[H];
        apus_attn_rmsnorm(xt, w->ln1_w, ln1, H);
        if (c->kind == APUS_LAYER_FULL)
            apus_store_full_attn(c, w, state, ln1, ao);
        else
            apus_store_gdn_attn(c, w, state, ln1, ao);
        state->pos++;
        for (size_t i = 0; i < H; i++)
            res1[i] = apus_bf16_bits(apus_bf16_f32(xt[i])
                                     + apus_bf16_f32(ao[i]));
        if (st->hooks.post_attn)
            st->hooks.post_attn(st->hooks.ctx, layer, res1, pos,
                                st->fwd_s, st->fwd_t);
        apus_attn_rmsnorm(res1, w->ln2_w, ln2, H);
        apus_store_moe(st, c, w, layer, ln2, moe_out, pos);
        for (size_t i = 0; i < H; i++)
            ot[i] = apus_bf16_bits(apus_bf16_f32(res1[i])
                                   + apus_bf16_f32(moe_out[i]));
    }
}

#endif /* APUS_CACHE_IMPLEMENTATION */
#endif /* APUS_CACHE_H */
