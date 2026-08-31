/*
 * tests/m6a/test_store.c — unit tests for the c/cache.h expert store on
 * the M6a fixture container (2 layers, MoE on BOTH, E=16, slab
 * 24,576 B = the v2 2-member fused slab: gate_up [64,128] then down
 * [128,32] BF16 slices, contiguous; apus.index.json FORMAT_VERSION 2
 * expert_slabs verified against headers; the v1 negative fixture gates
 * the format rejection).
 *
 *   1. Slab derivation + one-pread-per-expert (instrumented) + view
 *      byte-identity vs direct member-tensor reads.
 *   2. FORMAT_VERSION 1 manifest rejected with a clear error (no
 *      silent misread of a 3-member container).
 *   3. Hit/miss accounting, working-set promotion, mid-block overflow
 *      drop, LRU eviction order + recency.
 *   4. Pins: usage-file seeding, first-touch loads, persistence across
 *      reopen, LFRU hysteresis thresholds (need = p + p/4 + 4).
 *   5. RSS guard: 1-byte budget drops LRU payloads in place (identity
 *      kept, reload on demand), pins untouched.
 *   6. Generation-tag straggler (pre-claim hook): stale payload
 *      discarded, re-read, bytes correct.
 *   7. Concurrent-vs-synchronous slab byte identity.
 *   8. Hint eviction guard: unconsumed hints never evict warm experts.
 *   9. Demand boost: a demand-class load overtakes queued speculative
 *      loads in the I/O pool.
 *  10. Usage decay at save (0.5: 100->50, 50->25, 33->16).
 *
 * Run from the repository root (fixtures under tests/m6a/fixtures/).
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
#include "json.h"
#include "st.h"
#include "bf16.h"
#include "gdn.h"
#include "attn.h"
#include "moe.h"
#include "layer.h"
#include "compat.h"
#include "cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

#define MODEL "tests/m6a/fixtures/model"
#define MODEL_V1 "tests/m6a/fixtures/model_v1"
#define USAGE "tests/m6a/bin/test.usage"

#define N_LAYERS 2
#define N_EXPERTS 16
#define SLAB_BYTES 24576
#define GU_BYTES 16384
#define D_BYTES 8192

static ApusStore *open_store(int slots, int pins, int io, size_t rss,
                             const char *usage, double decay) {
    ApusStoreCfg c = {0};
    c.n_layers = N_LAYERS;
    c.n_experts = N_EXPERTS;
    c.slots_per_layer = slots;
    c.pins_per_layer = pins;
    c.cache_bytes = 1;              /* explicit slots win */
    c.pin_bytes = 1;
    c.rss_budget_bytes = rss;
    c.io_threads = io;
    c.usage_path = usage;
    c.usage_decay = decay;
    char err[256];
    ApusStore *st = apus_store_open(MODEL, &c, err, sizeof err);
    if (!st) fprintf(stderr, "store open: %s\n", err);
    return st;
}

/* read one member slice's bytes directly from the shard */
static uint8_t *read_member(int layer, int eid, int member, size_t *n) {
    char err[128], name[320];
    snprintf(name, sizeof name,
             "model.language_model.layers.%d.mlp.experts.%d.%s", layer,
             eid,
             member == 0 ? "gate_up_proj.weight" : "down_proj.weight");
    ApusStLazy *lz = apus_st_lazy_open(MODEL "/apus-qwen-00001.safetensors",
                                       0, err, sizeof err);
    if (!lz) return NULL;
    const ApusStLazyTensor *t = apus_st_lazy_find(lz, name);
    if (!t) { apus_st_lazy_close(lz); return NULL; }
    uint8_t *buf = malloc((size_t)t->nbytes);
    if (apus_st_lazy_pread(lz, t->file_off, buf, (size_t)t->nbytes)) {
        free(buf);
        apus_st_lazy_close(lz);
        return NULL;
    }
    apus_st_lazy_close(lz);
    *n = (size_t)t->nbytes;
    return buf;
}

static void test_open_and_resolve(void) {
    ApusStore *st = open_store(4, 0, -1, 0, "", 0.0);
    CHECK(st != NULL, "store open (sync mode)");
    if (!st) return;
    CHECK(apus_store_slab_bytes(st) == SLAB_BYTES,
          "slab bytes %zu != %d", apus_store_slab_bytes(st), SLAB_BYTES);

    ApusBf16ExpertW ew;
    CHECK(apus_store_resolve(st, 1, 5, &ew) == 0, "resolve L1 E5");
    CHECK(ew.I == 32 && ew.H == 128, "view dims %lldx%lld",
          (long long)ew.I, (long long)ew.H);
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.preads == 1 && ss.misses == 1 && ss.demand_loads == 1,
          "one pread per expert: preads %llu misses %llu demand %llu",
          (unsigned long long)ss.preads, (unsigned long long)ss.misses,
          (unsigned long long)ss.demand_loads);
    /* view bytes == direct member reads (v2: fused gate_up + down) */
    size_t n0, n1;
    uint8_t *gu = read_member(1, 5, 0, &n0);
    uint8_t *d = read_member(1, 5, 1, &n1);
    CHECK(gu && d, "direct member reads");
    CHECK(n0 == GU_BYTES && n1 == D_BYTES, "member sizes %zu/%zu", n0,
          n1);
    CHECK(memcmp(ew.gate_up, gu, n0) == 0 && memcmp(ew.down, d, n1) == 0,
          "slab view byte identity");
    free(gu); free(d);
    /* second resolve: hit, no new pread */
    CHECK(apus_store_resolve(st, 1, 5, &ew) == 0, "re-resolve L1 E5");
    apus_store_stats(st, &ss);
    CHECK(ss.hits == 1 && ss.preads == 1,
          "hit without pread: hits %llu preads %llu",
          (unsigned long long)ss.hits, (unsigned long long)ss.preads);
    CHECK(apus_store_debug_present(st, 1, 5)
          && apus_store_debug_ready(st, 1, 5), "present+ready");
    CHECK(!apus_store_debug_present(st, 1, 6), "E6 not present");
    /* every layer is MoE in this model — but out-of-range addresses
     * still fail loudly */
    CHECK(apus_store_resolve(st, 2, 0, &ew) == -1,
          "out-of-range layer resolve fails");
    CHECK(apus_store_resolve(st, 0, N_EXPERTS, &ew) == -1,
          "out-of-range expert resolve fails");
    apus_store_close(st);
}

static void test_v1_reject(void) {
    /* the v1 negative fixture: a FORMAT_VERSION 1 manifest — the store
     * must refuse loudly, never misread a 3-member container */
    ApusStoreCfg c = {0};
    c.n_layers = N_LAYERS;
    c.n_experts = N_EXPERTS;
    c.usage_path = "";
    char err[512] = {0};
    ApusStore *st = apus_store_open(MODEL_V1, &c, err, sizeof err);
    CHECK(st == NULL, "v1 manifest rejected (open should fail)");
    CHECK(strstr(err, "format_version") != NULL,
          "rejection names the format version: '%s'", err);
}

static void test_promotion_lru(void) {
    /* 2 slots; block of 4 distinct experts: all resolve (working set),
     * promotion keeps the 2 NEWEST, drops 2 (overflow), evictions 2 */
    ApusStore *st = open_store(2, 0, -1, 0, "", 0.0);
    CHECK(st != NULL, "store open");
    if (!st) return;
    ApusBf16ExpertW ew;
    for (int e = 1; e <= 4; e++)
        CHECK(apus_store_resolve(st, 1, e, &ew) == 0, "resolve E%d", e);
    int32_t lru[2];
    apus_store_debug_layer(st, 1, lru, 2, NULL, 0);
    CHECK(lru[0] == -1 && lru[1] == -1, "nothing in LRU before layer_end");
    apus_store_layer_end(st, 1);
    apus_store_debug_layer(st, 1, lru, 2, NULL, 0);
    CHECK(lru[0] == 3 && lru[1] == 4,
          "promotion keeps newest: {%d,%d} != {3,4}", lru[0], lru[1]);
    CHECK(!apus_store_debug_present(st, 1, 1)
          && !apus_store_debug_present(st, 1, 2),
          "overflow experts dropped after use");
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.evictions == 2, "evictions %llu != 2",
          (unsigned long long)ss.evictions);

    /* recency: touch E3 (hit), then a block of E5,E6 evicts E4 then E3
     * (promotion swaps INTO victim slots, so compare contents, not slot
     * order) */
    CHECK(apus_store_resolve(st, 1, 3, &ew) == 0, "re-resolve E3");
    apus_store_resolve(st, 1, 5, &ew);
    apus_store_resolve(st, 1, 6, &ew);
    apus_store_layer_end(st, 1);
    apus_store_debug_layer(st, 1, lru, 2, NULL, 0);
    CHECK((lru[0] == 5 && lru[1] == 6) || (lru[0] == 6 && lru[1] == 5),
          "recency eviction: {%d,%d} != {5,6}", lru[0], lru[1]);
    apus_store_stats(st, &ss);
    CHECK(ss.hits == 1, "recency hit counted (%llu)",
          (unsigned long long)ss.hits);
    apus_store_close(st);
}

static void test_pins(void) {
    /* usage-file seeding */
    FILE *f = fopen(USAGE, "w");
    fprintf(f, "1 5 100\n1 7 50\n");
    fclose(f);
    ApusStore *st = open_store(4, 2, -1, 0, USAGE, 0.0);
    CHECK(st != NULL, "store open (pins)");
    if (!st) return;
    int32_t pins[2];
    apus_store_debug_layer(st, 1, NULL, 0, pins, 2);
    CHECK(pins[0] == 5 && pins[1] == 7,
          "pin seeding order: {%d,%d} != {5,7}", pins[0], pins[1]);
    ApusBf16ExpertW ew;
    CHECK(apus_store_resolve(st, 1, 5, &ew) == 0, "first-touch pin E5");
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.pin_loads == 1 && ss.misses == 1,
          "pin first touch: pin_loads %llu misses %llu",
          (unsigned long long)ss.pin_loads, (unsigned long long)ss.misses);
    CHECK(apus_store_resolve(st, 1, 5, &ew) == 0, "pin E5 again");
    apus_store_stats(st, &ss);
    CHECK(ss.hits == 1, "pin hit (%llu)", (unsigned long long)ss.hits);

    /* persistence across a simulated restart: save, reopen, pins reseed
     * with counts >= seeded (live freq merged with max) */
    CHECK(apus_store_save_usage(st) == 0, "save usage");
    apus_store_close(st);
    st = open_store(4, 2, -1, 0, USAGE, 0.0);
    CHECK(st != NULL, "reopen");
    if (!st) return;
    apus_store_debug_layer(st, 1, NULL, 0, pins, 2);
    CHECK(pins[0] == 5, "pin persists across reopen (got %d)", pins[0]);
    remove(USAGE);
    apus_store_close(st);
}

static ApusStore *make_hyst_store(uint64_t pin_freq) {
    FILE *f = fopen(USAGE, "w");
    fprintf(f, "1 5 %llu\n", (unsigned long long)pin_freq);
    fclose(f);
    return open_store(4, 1, -1, 0, USAGE, 0.0);
}

static void drive_freq(ApusStore *st, int layer, int eid, int n) {
    ApusBf16ExpertW ew;
    for (int i = 0; i < n; i++)
        apus_store_resolve(st, layer, eid, &ew);
    apus_store_layer_end(st, layer);
}

static void test_lfru_hysteresis(void) {
    /* pin freq 10 -> need 10 + 2 + 4 = 16: challenger at 16 swaps */
    ApusStore *st = make_hyst_store(10);
    CHECK(st != NULL, "hyst store A");
    if (!st) return;
    drive_freq(st, 1, 9, 16);
    apus_store_repin(st);
    int32_t pins[1];
    apus_store_debug_layer(st, 1, NULL, 0, pins, 1);
    CHECK(pins[0] == 9, "hysteresis swap at threshold (pin %d != 9)",
          pins[0]);
    apus_store_close(st);
    remove(USAGE);

    /* pin freq 20 -> need 20 + 5 + 4 = 29: 28 does NOT swap, 29 does */
    st = make_hyst_store(20);
    CHECK(st != NULL, "hyst store B");
    if (!st) return;
    drive_freq(st, 1, 10, 28);
    apus_store_repin(st);
    apus_store_debug_layer(st, 1, NULL, 0, pins, 1);
    CHECK(pins[0] == 5, "hysteresis holds below threshold (pin %d != 5)",
          pins[0]);
    ApusBf16ExpertW ew;
    apus_store_resolve(st, 1, 10, &ew);         /* freq 29 */
    apus_store_repin(st);
    apus_store_debug_layer(st, 1, NULL, 0, pins, 1);
    CHECK(pins[0] == 10, "hysteresis swap at 29 (pin %d != 10)", pins[0]);
    apus_store_close(st);
    remove(USAGE);
}

static void test_rss_guard(void) {
    FILE *f = fopen(USAGE, "w");
    fprintf(f, "1 5 100\n");
    fclose(f);
    ApusStore *st = open_store(2, 1, -1, 1, USAGE, 0.0);  /* 1-byte RSS */
    CHECK(st != NULL, "rss store");
    if (!st) return;
    ApusBf16ExpertW ew;
    apus_store_resolve(st, 1, 5, &ew);      /* pin first touch */
    apus_store_resolve(st, 1, 1, &ew);
    apus_store_resolve(st, 1, 2, &ew);
    apus_store_layer_end(st, 1);            /* runs the guard */
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.rss_drops > 0, "rss drops happened (%llu)",
          (unsigned long long)ss.rss_drops);
    int32_t lru[2];
    apus_store_debug_layer(st, 1, lru, 2, NULL, 0);
    CHECK(lru[0] == -1 && lru[1] == -1,
          "LRU payloads dropped in place: {%d,%d}", lru[0], lru[1]);
    CHECK(apus_store_debug_ready(st, 1, 5), "pin payload untouched");
    /* identity kept: resolve reloads on demand */
    CHECK(apus_store_resolve(st, 1, 1, &ew) == 0, "reload after drop");
    apus_store_stats(st, &ss);
    uint64_t misses1 = ss.misses;
    apus_store_layer_end(st, 1);
    apus_store_rss_guard(st);
    apus_store_resolve(st, 1, 2, &ew);      /* dropped earlier: reload */
    apus_store_stats(st, &ss);
    CHECK(ss.misses > misses1, "reload counted as miss");
    apus_store_close(st);
    remove(USAGE);
}

/* generation straggler: stale the in-flight job once via the pre-claim
 * hook; the waiter must re-submit and return correct bytes */
static ApusStore *g_race_store;
static int g_race_armed;
static void pre_claim_hook(ApusStore *st, int layer, int32_t eid,
                           uint64_t gen) {
    (void)gen;
    if (g_race_armed) {
        g_race_armed = 0;
        int rc = apus_store_debug_stale_gen(st, layer, eid);
        CHECK(rc == 0, "stale_gen hook");
    }
}

static void test_straggler(void) {
    ApusStore *st = open_store(4, 0, 2, 0, "", 0.0);   /* pooled I/O */
    CHECK(st != NULL, "race store");
    if (!st) return;
    g_race_store = st;
    g_race_armed = 1;
    apus_store_debug_set_pre_claim(st, pre_claim_hook);
    ApusBf16ExpertW ew;
    CHECK(apus_store_resolve(st, 1, 9, &ew) == 0,
          "resolve survives straggler");
    apus_store_debug_set_pre_claim(st, NULL);
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.preads >= 2, "straggler re-read (preads %llu)",
          (unsigned long long)ss.preads);
    size_t n0;
    uint8_t *gu = read_member(1, 9, 0, &n0);
    CHECK(gu && memcmp(ew.gate_up, gu, n0) == 0,
          "bytes correct after straggler");
    free(gu);
    apus_store_close(st);
}

static void test_concurrent_identity(void) {
    /* same experts through sync and 4-thread pool: bytes identical */
    ApusStore *a = open_store(4, 0, -1, 0, "", 0.0);
    ApusStore *b = open_store(4, 0, 4, 0, "", 0.0);
    CHECK(a && b, "identity stores");
    if (!a || !b) return;
    ApusBf16ExpertW wa, wb;
    int ok = 1;
    for (int e = 0; e < 12 && ok; e++) {
        apus_store_resolve(a, 0, e, &wa);
        apus_store_resolve(b, 0, e, &wb);
        if (memcmp(wa.gate_up, wb.gate_up, GU_BYTES)
            || memcmp(wa.down, wb.down, D_BYTES))
            ok = 0;
    }
    CHECK(ok, "concurrent vs serial slab byte identity (12 experts)");
    apus_store_close(a);
    apus_store_close(b);
}

static void test_hint_eviction_guard(void) {
    ApusStore *st = open_store(2, 0, -1, 0, "", 0.0);
    CHECK(st != NULL, "guard store");
    if (!st) return;
    ApusBf16ExpertW ew;
    apus_store_resolve(st, 1, 1, &ew);
    apus_store_resolve(st, 1, 2, &ew);
    apus_store_layer_end(st, 1);            /* LRU {1,2} warm */
    apus_store_hint(st, 1, 3);
    apus_store_hint(st, 1, 4);
    apus_store_layer_end(st, 1);            /* hints READY, freq 0 */
    int32_t lru[2];
    apus_store_debug_layer(st, 1, lru, 2, NULL, 0);
    CHECK(lru[0] == 1 && lru[1] == 2,
          "hints never evict warm demand-loaded: {%d,%d} != {1,2}",
          lru[0], lru[1]);
    CHECK(!apus_store_debug_present(st, 1, 3),
          "unconsumed hint dropped at promotion");
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.hint_loads == 2, "hint loads counted (%llu)",
          (unsigned long long)ss.hint_loads);
    apus_store_close(st);
}

/* demand boost: completion order recorded by the worker-side pre-claim
 * hook (fires after each pread, in true completion order — timing-
 * independent, unlike polling) */
static int32_t g_comp[64];
static int g_comp_n;
static void comp_hook(ApusStore *st, int layer, int32_t eid,
                      uint64_t gen) {
    (void)st; (void)layer; (void)gen;
    if (g_comp_n < 64) g_comp[g_comp_n++] = eid;
}

static void test_demand_boost(void) {
    /* one worker; queue 6 speculative hints, then one demand hint; the
     * demand load must complete before the still-queued speculative
     * ones */
    ApusStore *st = open_store(8, 0, 1, 0, "", 0.0);
    CHECK(st != NULL, "boost store");
    if (!st) return;
    g_comp_n = 0;
    apus_store_debug_set_pre_claim(st, comp_hook);
    for (int e = 0; e < 6; e++)
        apus_store_hint(st, 1, e);
    apus_store_hint_demand(st, 1, 15);
    /* wait for all 7 to complete */
    ApusStoreStats ss;
    long t0 = 0;
    for (;;) {
        apus_store_stats(st, &ss);
        if (ss.preads >= 7) break;
        struct timespec ts = { 0, 1000000 };
        nanosleep(&ts, NULL);
        if (++t0 > 10000) break;
    }
    apus_store_debug_set_pre_claim(st, NULL);
    CHECK(g_comp_n == 7, "all boost loads completed (%d/7)", g_comp_n);
    int pos15 = -1, pos3 = -1, pos4 = -1, pos5 = -1;
    for (int i = 0; i < g_comp_n; i++) {
        if (g_comp[i] == 15) pos15 = i;
        if (g_comp[i] == 3) pos3 = i;
        if (g_comp[i] == 4) pos4 = i;
        if (g_comp[i] == 5) pos5 = i;
    }
    CHECK(pos15 >= 0 && pos15 < pos3 && pos15 < pos4 && pos15 < pos5,
          "demand overtook queued speculative (completion pos E15=%d "
          "E3=%d E4=%d E5=%d)", pos15, pos3, pos4, pos5);
    apus_store_close(st);
}

static void test_usage_decay(void) {
    FILE *f = fopen(USAGE, "w");
    fprintf(f, "1 1 100\n1 2 50\n1 3 33\n");
    fclose(f);
    ApusStore *st = open_store(4, 0, -1, 0, USAGE, 0.5);
    CHECK(st != NULL, "decay store");
    if (!st) return;
    CHECK(apus_store_save_usage(st) == 0, "decay save");
    apus_store_close(st);
    f = fopen(USAGE, "r");
    int got[4] = {0, 0, 0, 0};
    int L, e;
    unsigned long long c;
    while (fscanf(f, "%d %d %llu", &L, &e, &c) == 3)
        if (L == 1 && e >= 1 && e <= 3) got[e] = (int)c;
    fclose(f);
    CHECK(got[1] == 50 && got[2] == 25 && got[3] == 16,
          "decay 0.5: got %d/%d/%d != 50/25/16",
          got[1], got[2], got[3]);
    remove(USAGE);
}

int main(void) {
    printf("test_store: expert store unit tests (Qwen3.6 v2 slab, M6a)\n");
    test_open_and_resolve();
    test_v1_reject();
    test_promotion_lru();
    test_pins();
    test_lfru_hysteresis();
    test_rss_guard();
    test_straggler();
    test_concurrent_identity();
    test_hint_eviction_guard();
    test_demand_boost();
    test_usage_decay();
    printf("test_store: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
