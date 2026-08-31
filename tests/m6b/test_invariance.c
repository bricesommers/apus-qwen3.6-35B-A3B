/*
 * tests/m6b/test_invariance.c — THE hard M6b gate: the pilot changes
 * only WHEN/WHETHER an expert is in RAM, never numerics. Eager vs
 * store-only vs pilot ON at every cache size: BITWISE-identical tokens
 * and logits.
 *
 * Greedy decode: 8-token prompt, 24 steps, per-step logits. The pilot
 * thread issues real loads through the store's I/O pool (APUS_IO_THREADS
 * from env — the Makefile diffs digests at 1/4/8).
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
#define APUS_SAMPLE_IMPLEMENTATION
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
#include "sample.h"

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

#define MODEL "tests/m6a/fixtures/model"
#define USAGE "tests/m6b/bin/pins.usage"

#define PROMPT_LEN 8
#define STEPS 24

static const int64_t PROMPT[PROMPT_LEN] = { 3, 1, 4, 1, 5, 9, 2, 6 };

static void run_greedy(ApusModel *m, int *tokens, float *logits) {
    size_t V = (size_t)m->vocab;
    ApusModelState st;
    apus_model_state_init(&st, m);
    float *lg = malloc(V * sizeof(float));
    apus_model_forward(m, &st, PROMPT, PROMPT_LEN, lg, 0, NULL);
    for (int s = 0; s < STEPS; s++) {
        int tok = apus_sample_argmax(lg, V);
        tokens[s] = tok;
        memcpy(logits + (size_t)s * V, lg, V * sizeof(float));
        int64_t id = tok;
        apus_model_forward(m, &st, &id, 1, lg, 0, NULL);
    }
    apus_model_state_free(&st, m);
    free(lg);
}

static void attach_pilot(ApusModel *m, ApusStore *st, ApusPilot **out,
                         const char *dump) {
    const ApusLayerCfg *mc = &m->layers[0].lc;
    ApusPilotCfg pc = {0};
    pc.store = st;
    pc.n_layers = m->n_layers;
    pc.n_experts = (int)mc->experts;
    pc.top_k = (int)mc->top_k;
    pc.hidden = m->hidden;
    pc.enabled = 1;
    pc.pilot_k = 8;
    pc.ring_entries = 4096;
    pc.dump_path = dump;
    ApusPilot *p = apus_pilot_create(&pc);
    for (int L = 0; L < m->n_layers; L++) {
        const ApusLayerW *lw = &m->layers[L].lw;
        ApusPilotRouter r = { lw->rtr_w, lw->ln2_w };
        apus_pilot_attach_router(p, L, &r);
    }
    ApusStoreFwdHooks hooks;
    apus_pilot_store_hooks(p, &hooks);
    apus_store_fwd_hooks(st, &hooks);
    apus_pilot_start(p);
    *out = p;
}

int main(void) {
    printf("test_invariance: pilot on/off bitwise gate (M6b)\n");
    char err[256];

    ApusModel me;
    if (apus_model_load(&me, MODEL, 256, err, sizeof err)) {
        fprintf(stderr, "FAIL: eager load: %s\n", err);
        return 1;
    }
    size_t V = (size_t)me.vocab;
    int *tok0 = malloc(STEPS * sizeof(int));
    float *lg0 = malloc(STEPS * V * sizeof(float));
    run_greedy(&me, tok0, lg0);
    apus_model_free(&me);

    int io = apus_env_int("APUS_IO_THREADS", 4);
    fprintf(stderr, "  store I/O threads: %d\n", io);

    /* pin seeding file for the pins config */
    FILE *f = fopen(USAGE, "w");
    fprintf(f, "0 0 10\n1 0 10\n");
    fclose(f);

    struct { const char *name; int slots, pins; size_t rss; int pilot;
             const char *usage; } cfgs[] = {
        { "store only, 2 slots", 2, 0, 0, 0, "" },
        { "pilot, 16 slots (all fit)", 16, 0, 0, 1, "" },
        { "pilot, 8 slots", 8, 0, 0, 1, "" },
        { "pilot, 2 slots", 2, 0, 0, 1, "" },
        { "pilot, 2 slots + RSS 1 byte", 2, 0, 1, 1, "" },
        { "pilot, 4 slots + 2 pins", 4, 2, 0, 1, USAGE },
    };
    for (size_t ci = 0; ci < sizeof cfgs / sizeof cfgs[0]; ci++) {
        ApusModel m;
        if (apus_model_load_ex(&m, MODEL, 256, 1, err, sizeof err)) {
            fprintf(stderr, "FAIL: tiered load: %s\n", err);
            return 1;
        }
        ApusStoreCfg sc = {0};
        sc.n_layers = m.n_layers;
        sc.n_experts = 16;
        sc.slots_per_layer = cfgs[ci].slots;
        sc.pins_per_layer = cfgs[ci].pins;
        sc.cache_bytes = 1;
        sc.pin_bytes = 1;
        sc.rss_budget_bytes = cfgs[ci].rss;
        sc.io_threads = io;
        sc.usage_path = cfgs[ci].usage;
        ApusStore *st = apus_store_open(MODEL, &sc, err, sizeof err);
        CHECK(st != NULL, "%s: store open: %s", cfgs[ci].name, err);
        if (!st) { apus_model_free(&m); continue; }
        apus_model_attach_store(&m, st);
        ApusPilot *p = NULL;
        if (cfgs[ci].pilot)
            attach_pilot(&m, st, &p, NULL);
        int *tok = malloc(STEPS * sizeof(int));
        float *lg = malloc(STEPS * V * sizeof(float));
        run_greedy(&m, tok, lg);
        ApusStoreStats ss;
        apus_store_stats(st, &ss);
        int tok_same = memcmp(tok, tok0, STEPS * sizeof(int)) == 0;
        int lg_same = memcmp(lg, lg0, STEPS * V * sizeof(float)) == 0;
        digest_bytes(&g_digest, tok, STEPS * sizeof(int));
        digest_bytes(&g_digest, lg, STEPS * V * sizeof(float));
        printf("  %-30s tokens %s, logits %s "
               "(preads %llu, drops %llu)\n",
               cfgs[ci].name,
               tok_same ? "IDENTICAL" : "DIFFER",
               lg_same ? "BITWISE" : "DIFFER",
               (unsigned long long)ss.preads,
               (unsigned long long)ss.rss_drops);
        CHECK(tok_same, "%s: token stream differs", cfgs[ci].name);
        CHECK(lg_same, "%s: logits differ", cfgs[ci].name);
        if (p) {
            ApusPilotStats ps;
            apus_pilot_stats(p, &ps);
            printf("    pilot: %llu predictions, %llu issued, "
                   "recall %llu/%llu\n",
                   (unsigned long long)ps.predictions,
                   (unsigned long long)ps.hints_issued,
                   (unsigned long long)ps.actual_hits,
                   (unsigned long long)ps.actual_experts);
            apus_pilot_destroy(p);
        }
        free(tok); free(lg);
        apus_store_close(st);
        apus_model_free(&m);
    }
    remove(USAGE);
    free(tok0); free(lg0);
    printf("  digest %016llx\n", (unsigned long long)g_digest);
    printf("test_invariance: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
