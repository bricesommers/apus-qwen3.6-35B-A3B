/*
 * tests/m6a/test_invariance.c — THE hard M6a gate: eager slurp vs the
 * expert store at every cache size produces BITWISE-identical tokens and
 * logits. Memory shortcuts cost speed only, never quality.
 *
 * Greedy decode: 8-token prompt, 24 steps, per-step logits collected.
 * Configurations: eager (apus_layer_forward) vs store at 16 slots/layer
 * (all 16 experts fit), 8, 2, 2 under a 1-byte RSS budget (drops+reloads),
 * 4 + 2 seeded pins, and synchronous I/O mode. The store's I/O pool size
 * comes from APUS_IO_THREADS (the Makefile diffs digests at 1/4/8 —
 * thread timing must not perturb compute bits).
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
#define USAGE "tests/m6a/bin/pins.usage"

#define PROMPT_LEN 8
#define STEPS 24

static const int64_t PROMPT[PROMPT_LEN] = { 3, 1, 4, 1, 5, 9, 2, 6 };

/* greedy chain; collects tokens [STEPS] and logits [STEPS * V] */
static void run_greedy(ApusModel *m, int *tokens, float *logits) {
    size_t V = (size_t)m->vocab;
    ApusModelState st;
    apus_model_state_init(&st, m);
    int64_t ids[1];
    float *lg = malloc(V * sizeof(float));
    apus_model_forward(m, &st, PROMPT, PROMPT_LEN, lg, 0, NULL);
    for (int s = 0; s < STEPS; s++) {
        int tok = apus_sample_argmax(lg, V);
        tokens[s] = tok;
        memcpy(logits + (size_t)s * V, lg, V * sizeof(float));
        ids[0] = tok;
        apus_model_forward(m, &st, ids, 1, lg, 0, NULL);
    }
    apus_model_state_free(&st, m);
    free(lg);
}

static ApusStore *open_store(ApusModel *m, int slots, int pins, int io,
                             size_t rss, const char *usage) {
    ApusStoreCfg c = {0};
    c.n_layers = m->n_layers;
    if (m->n_layers > 0)   /* every layer is MoE in this model (M4) */
        c.n_experts = (int)m->layers[0].lc.experts;
    c.slots_per_layer = slots;
    c.pins_per_layer = pins;
    c.cache_bytes = 1;
    c.pin_bytes = 1;
    c.rss_budget_bytes = rss;
    c.io_threads = io;
    c.usage_path = usage;
    char err[256];
    ApusStore *st = apus_store_open(MODEL, &c, err, sizeof err);
    if (!st) fprintf(stderr, "store open: %s\n", err);
    return st;
}

int main(void) {
    printf("test_invariance: eager-vs-store bitwise gate (M6a)\n");
    char err[256];

    /* eager reference */
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

    struct { const char *name; int slots, pins, io; size_t rss;
             const char *usage; } cfgs[] = {
        { "store 16 slots (all fit)", 16, 0, -2, 0, "" },
        { "store 8 slots", 8, 0, -2, 0, "" },
        { "store 2 slots", 2, 0, -2, 0, "" },
        { "store 2 slots + RSS 1 byte", 2, 0, -2, 1, "" },
        { "store 4 slots + 2 pins", 4, 2, -2, 0, USAGE },
        { "store 2 slots, synchronous", 2, 0, -1, 0, "" },
    };
    /* pin seeding file for the pins config */
    FILE *f = fopen(USAGE, "w");
    fprintf(f, "0 0 10\n1 0 10\n");
    fclose(f);

    for (size_t ci = 0; ci < sizeof cfgs / sizeof cfgs[0]; ci++) {
        ApusModel m;
        if (apus_model_load_ex(&m, MODEL, 256, 1, err, sizeof err)) {
            fprintf(stderr, "FAIL: tiered load: %s\n", err);
            return 1;
        }
        int io_mode = cfgs[ci].io == -2 ? io : cfgs[ci].io;
        ApusStore *st = open_store(&m, cfgs[ci].slots, cfgs[ci].pins,
                                   io_mode, cfgs[ci].rss, cfgs[ci].usage);
        CHECK(st != NULL, "%s: store open", cfgs[ci].name);
        if (!st) { apus_model_free(&m); continue; }
        apus_model_attach_store(&m, st);
        int *tok = malloc(STEPS * sizeof(int));
        float *lg = malloc(STEPS * V * sizeof(float));
        run_greedy(&m, tok, lg);
        ApusStoreStats ss;
        apus_store_stats(st, &ss);
        int tok_same = memcmp(tok, tok0, STEPS * sizeof(int)) == 0;
        int lg_same = memcmp(lg, lg0, STEPS * V * sizeof(float)) == 0;
        digest_bytes(&g_digest, tok, STEPS * sizeof(int));
        digest_bytes(&g_digest, lg, STEPS * V * sizeof(float));
        printf("  %-28s tokens %s, logits %s "
               "(preads %llu, drops %llu, misses %llu, hits %llu)\n",
               cfgs[ci].name,
               tok_same ? "IDENTICAL" : "DIFFER",
               lg_same ? "BITWISE" : "DIFFER",
               (unsigned long long)ss.preads,
               (unsigned long long)ss.rss_drops,
               (unsigned long long)ss.misses,
               (unsigned long long)ss.hits);
        CHECK(tok_same, "%s: token stream differs", cfgs[ci].name);
        CHECK(lg_same, "%s: logits differ", cfgs[ci].name);
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
