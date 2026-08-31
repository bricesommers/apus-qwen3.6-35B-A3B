/*
 * tests/m6b/test_pilot.c — unit tests for c/pilot.h (router-lookahead
 * prefetch) on the M6a fixture container.
 *
 *   1. Ring: push/pop FIFO order, drop-newest when full (exact counts),
 *      wraparound.
 *   2. Prediction: apus_pilot_predict == the real code path
 *      (apus_attn_rmsnorm + apus_moe_route / apus_moe_route_topn)
 *      computed directly, bitwise; out-of-range and unattached layers
 *      not predictable.
 *   3. Hooks: post_attn stores the pending prediction + pushes k ring
 *      entries (prefill_last_only honored); router_actual recall
 *      accounting (hits/misses) at matching (pos, layer); decode-only
 *      (s>1 ignored).
 *   4. Consumer thread: issues store hints (issued == enqueued after
 *      drain), stale watermark drops, destroy-with-backlog.
 *   5. Demand/boost interplay is covered in tests/m6a (test_store).
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
#define APUS_MODEL_IMPLEMENTATION
#define APUS_PILOT_IMPLEMENTATION
#include "json.h"
#include "st.h"
#include "bf16.h"
#include "gdn.h"
#include "attn.h"
#include "moe.h"
#include "layer.h"
#include "compat.h"
#include "cache.h"
#include "pilot.h"
#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static ApusPilotCfg base_cfg(void) {
    ApusPilotCfg pc = {0};
    pc.store = NULL;
    pc.n_layers = 2;
    pc.n_experts = 16;
    pc.top_k = 4;
    pc.hidden = 128;
    pc.enabled = 1;
    pc.pilot_k = 8;
    pc.ring_entries = 16;
    return pc;
}

static void test_ring(void) {
    ApusPilotCfg pc = base_cfg();
    ApusPilot *p = apus_pilot_create(&pc);
    CHECK(p != NULL, "pilot create");
    if (!p) return;
    /* fill to capacity (16), then 4 more -> drop-newest */
    for (int i = 0; i < 20; i++)
        apus_pilot_debug_push(p, 100, 1, i);
    ApusPilotStats ps;
    apus_pilot_stats(p, &ps);
    CHECK(ps.hints_enqueued == 16 && ps.hints_dropped_full == 4,
          "ring full: enqueued %llu dropped %llu (want 16/4)",
          (unsigned long long)ps.hints_enqueued,
          (unsigned long long)ps.hints_dropped_full);
    /* FIFO order of the FIRST 16 (newest 4 dropped) */
    int ok = 1;
    for (int i = 0; i < 16; i++) {
        int64_t pos;
        int layer, eid;
        if (!apus_pilot_debug_pop(p, &pos, &layer, &eid)
            || pos != 100 || layer != 1 || eid != i)
            ok = 0;
    }
    CHECK(ok, "ring FIFO order, drop-newest");
    int64_t pos;
    int layer, eid;
    CHECK(apus_pilot_debug_pop(p, &pos, &layer, &eid) == 0,
          "ring empty after drain");
    /* wraparound: push 20 more, drain, order intact */
    for (int i = 0; i < 20; i++)
        apus_pilot_debug_push(p, 200, 2, i + 40);
    apus_pilot_stats(p, &ps);
    CHECK(ps.hints_enqueued == 16 + 16
          && ps.hints_dropped_full == 4 + 4,
          "wraparound counts: enqueued %llu dropped %llu",
          (unsigned long long)ps.hints_enqueued,
          (unsigned long long)ps.hints_dropped_full);
    ok = 1;
    for (int i = 0; i < 16; i++) {
        if (!apus_pilot_debug_pop(p, &pos, &layer, &eid)
            || pos != 200 || layer != 2 || eid != i + 40)
            ok = 0;
    }
    CHECK(ok, "ring wraparound FIFO order");
    apus_pilot_destroy(p);
}

static void test_prediction(void) {
    char err[256];
    ApusModel m;
    if (apus_model_load_ex(&m, MODEL, 256, 1, err, sizeof err)) {
        fprintf(stderr, "FAIL: model load: %s\n", err);
        failures++;
        return;
    }
    ApusPilotCfg pc = base_cfg();
    ApusPilot *p = apus_pilot_create(&pc);
    CHECK(p != NULL, "pilot create");
    if (!p) { apus_model_free(&m); return; }
    for (int L = 0; L < m.n_layers; L++) {
        const ApusLayerW *lw = &m.layers[L].lw;
        ApusPilotRouter r = { lw->rtr_w, lw->ln2_w };
        apus_pilot_attach_router(p, L, &r);
    }
    /* random res1 (bf16) — compare pilot prediction vs the direct
     * rmsnorm+route computation, BITWISE (same public code path) */
    uint16_t res1[128];
    uint64_t rng = 0xC0FFEE;
    for (int i = 0; i < 128; i++) {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        res1[i] = apus_bf16_bits((float)(int)((rng >> 33) % 2000)
                                 / 250.0f - 4.0f);
    }
    /* n <= top_k: prediction == the router's top-4 output, bitwise */
    {
        int32_t pidx[4];
        CHECK(apus_pilot_predict(p, 1, res1, pidx, 4) == 0,
              "predict target 1 (n == top_k)");
        uint16_t x[128];
        const ApusLayerW *lw = &m.layers[1].lw;
        apus_attn_rmsnorm(res1, lw->ln2_w, x, 128);
        int32_t didx[4];
        uint16_t dw[4];
        apus_moe_route(x, lw->rtr_w, didx, dw, 16, 128, 4);
        CHECK(memcmp(pidx, didx, sizeof didx) == 0,
              "predict == real router path (n == top_k)");
    }
    /* n > top_k (pilot_k 8 > top_k 4): the same selection machinery at
     * a deeper truncation == apus_moe_route_topn, bitwise */
    {
        int32_t pidx[8];
        CHECK(apus_pilot_predict(p, 1, res1, pidx, 8) == 0,
              "predict target 1 (n > top_k)");
        uint16_t x[128];
        const ApusLayerW *lw = &m.layers[1].lw;
        apus_attn_rmsnorm(res1, lw->ln2_w, x, 128);
        int32_t didx[8];
        apus_moe_route_topn(x, lw->rtr_w, didx, 8, 16, 128);
        CHECK(memcmp(pidx, didx, sizeof didx) == 0,
              "predict == route_topn path (n > top_k)");
    }
    /* out-of-range target and an unattached layer are not predictable
     * (every layer of this model is MoE — an unattached view is the
     * only "no router" case left) */
    int32_t dummy[8];
    CHECK(apus_pilot_predict(p, 2, res1, dummy, 8) == -1,
          "out-of-range target not predictable");
    ApusPilot *bare = apus_pilot_create(&pc);
    CHECK(bare != NULL, "bare pilot create");
    if (bare) {
        CHECK(apus_pilot_predict(bare, 1, res1, dummy, 8) == -1,
              "unattached layer not predictable");
        apus_pilot_destroy(bare);
    }
    apus_pilot_destroy(p);
    apus_model_free(&m);
}

static void test_hooks(void) {
    char err[256];
    ApusModel m;
    if (apus_model_load_ex(&m, MODEL, 256, 1, err, sizeof err)) {
        fprintf(stderr, "FAIL: model load: %s\n", err);
        failures++;
        return;
    }
    ApusPilotCfg pc = base_cfg();
    ApusPilot *p = apus_pilot_create(&pc);
    for (int L = 0; L < m.n_layers; L++) {
        const ApusLayerW *lw = &m.layers[L].lw;
        ApusPilotRouter r = { lw->rtr_w, lw->ln2_w };
        apus_pilot_attach_router(p, L, &r);
    }
    ApusStoreFwdHooks hooks;
    apus_pilot_store_hooks(p, &hooks);

    uint16_t res1[128];
    for (int i = 0; i < 128; i++) res1[i] = apus_bf16_bits(0.01f * i - 1.0f);

    ApusPilotStats ps;
    /* prefill_last_only: s=4, t=0..2 predict nothing, t=3 predicts */
    hooks.post_attn(hooks.ctx, 0, res1, 10, 4, 0);
    apus_pilot_stats(p, &ps);
    CHECK(ps.predictions == 0, "prefill_last_only skips t<s-1");
    hooks.post_attn(hooks.ctx, 0, res1, 10, 4, 3);
    apus_pilot_stats(p, &ps);
    CHECK(ps.predictions == 1 && ps.hints_enqueued == 8,
          "last-token prediction: %llu predictions, %llu enqueued",
          (unsigned long long)ps.predictions,
          (unsigned long long)ps.hints_enqueued);
    /* decode (s=1): prediction for target 1 at pos 11 */
    hooks.post_attn(hooks.ctx, 0, res1, 11, 1, 0);
    apus_pilot_stats(p, &ps);
    CHECK(ps.predictions == 2, "decode prediction");

    /* recall accounting: the actual set equals the prediction -> 8/8 */
    int32_t idx[8];
    apus_pilot_predict(p, 1, res1, idx, 8);
    hooks.router_actual(hooks.ctx, 1, idx, 8, 11, 1);
    apus_pilot_stats(p, &ps);
    CHECK(ps.actual_experts == 8 && ps.actual_hits == 8,
          "recall full hit: %llu/%llu",
          (unsigned long long)ps.actual_hits,
          (unsigned long long)ps.actual_experts);
    /* actuals at a layer with NO pending prediction: ignored */
    int32_t other[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    hooks.router_actual(hooks.ctx, 0, other, 8, 11, 1);
    apus_pilot_stats(p, &ps);
    CHECK(ps.actual_experts == 8, "no pending -> no accounting");
    /* s>1 actuals are ignored (prefill) */
    hooks.router_actual(hooks.ctx, 1, idx, 8, 12, 4);
    apus_pilot_stats(p, &ps);
    CHECK(ps.actual_experts == 8, "s>1 actuals ignored");
    apus_pilot_destroy(p);
    apus_model_free(&m);
}

static void test_consumer(void) {
    char err[256];
    /* store in synchronous-IO... no: pool mode with 2 threads so hint
     * loads actually overlap */
    ApusStoreCfg sc = {0};
    sc.n_layers = 2;
    sc.n_experts = 16;
    sc.slots_per_layer = 8;
    sc.cache_bytes = 1;
    sc.pin_bytes = 1;
    sc.io_threads = 2;
    sc.usage_path = "";
    ApusStore *st = apus_store_open(MODEL, &sc, err, sizeof err);
    CHECK(st != NULL, "store open: %s", err);
    if (!st) return;
    ApusPilotCfg pc = base_cfg();
    pc.store = st;
    ApusPilot *p = apus_pilot_create(&pc);
    CHECK(p != NULL, "pilot create");
    if (!p) { apus_store_close(st); return; }
    apus_pilot_start(p);
    /* push 12 hints for layer 1 experts 0..11 at pos 5; watermark
     * behind (pos 0) so all issue */
    for (int e = 0; e < 12; e++)
        apus_pilot_debug_push(p, 5, 1, e);
    /* wait for drain */
    ApusPilotStats ps;
    int drained = 0;
    for (int i = 0; i < 5000; i++) {
        apus_pilot_stats(p, &ps);
        if (ps.hints_issued >= 12) { drained = 1; break; }
        struct timespec ts = { 0, 1000000 };
        nanosleep(&ts, NULL);
    }
    CHECK(drained, "consumer drained (issued %llu)",
          (unsigned long long)ps.hints_issued);
    ApusStoreStats ss;
    apus_store_stats(st, &ss);
    CHECK(ss.hint_loads >= 12, "store got hint loads (%llu)",
          (unsigned long long)ss.hint_loads);
    /* stale: move the watermark ahead via the post_attn hook (layer 1's
     * target is out of range, so nothing predicts), then push a hint
     * behind it */
    ApusStoreFwdHooks hooks;
    apus_pilot_store_hooks(p, &hooks);
    uint16_t res1[128] = {0};
    hooks.post_attn(hooks.ctx, 1, res1, 20, 1, 0);
    apus_pilot_debug_push(p, 5, 1, 30);     /* behind watermark */
    int stale_seen = 0;
    for (int i = 0; i < 5000; i++) {
        apus_pilot_stats(p, &ps);
        if (ps.hints_dropped_stale >= 1) {
            stale_seen = 1;
            break;
        }
        struct timespec ts = { 0, 1000000 };
        nanosleep(&ts, NULL);
    }
    CHECK(stale_seen, "stale watermark drop observed (stale %llu)",
          (unsigned long long)ps.hints_dropped_stale);
    /* destroy with backlog */
    for (int e = 0; e < 14; e++)
        apus_pilot_debug_push(p, 99, 1, e);
    apus_pilot_destroy(p);
    apus_store_close(st);
    CHECK(1, "destroy with backlog");
}

int main(void) {
    printf("test_pilot: pilot unit tests (Qwen3.6 fp32-softmax router, M6b)\n");
    test_ring();
    test_prediction();
    test_hooks();
    test_consumer();
    printf("test_pilot: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
