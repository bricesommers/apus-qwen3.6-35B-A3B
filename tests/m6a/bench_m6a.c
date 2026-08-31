/*
 * tests/m6a/bench_m6a.c — NVMe slab-read bench + hit-rate/tok/s curve
 * for the c/cache.h expert store on the fixture container (informational
 * only; fixture slabs are 24 KB and OS-cache warm, so decode is
 * compute-bound — the real-model curve needs the real container).
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
#include <time.h>

#define MODEL "tests/m6a/fixtures/model"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static const int64_t PROMPT[8] = { 3, 1, 4, 1, 5, 9, 2, 6 };

int main(void) {
    /* 1. slab pread bench: 64 random (layer, eid) resolves through the
     * store (one pread per unique slab), cached and F_NOCACHE fds */
    char err[128];
    for (int nc = 0; nc < 2; nc++) {
        ApusStoreCfg sc = {0};
        sc.n_layers = 2;
        sc.n_experts = 16;
        sc.slots_per_layer = 16;
        sc.io_threads = -1;
        sc.nocache = nc ? 1 : -1;
        sc.usage_path = "";
        ApusStore *st = apus_store_open(MODEL, &sc, err, sizeof err);
        if (!st) { fprintf(stderr, "%s\n", err); return 1; }
        uint64_t rng = 0x12345;
        double t0 = now_s();
        for (int i = 0; i < 64; i++) {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            int layer = (int)((rng >> 33) % 2);
            int eid = (int)((rng >> 40) % 16);
            ApusBf16ExpertW ew;
            apus_store_resolve(st, layer, eid, &ew);
        }
        ApusStoreStats ss;
        apus_store_stats(st, &ss);
        double dt = now_s() - t0;
        double gb = (double)ss.bytes_read / 1e9;
        printf("pread bench (%s): %llu slabs, %.3f MB in %.3fs = %.2f GB/s\n",
               nc ? "F_NOCACHE" : "cached",
               (unsigned long long)ss.preads, gb * 1000, dt,
               gb / (dt > 0 ? dt : 1e-9));
        apus_store_close(st);
    }

    /* 2. hit-rate / tok/s curve */
    printf("%-14s %10s %8s %8s %10s\n", "slots/layer", "hit rate",
           "preads", "drops", "tok/s");
    static const int slots_tab[] = { 16, 8, 4, 2 };
    for (int c = 0; c < 4; c++) {
        ApusModel m;
        if (apus_model_load_ex(&m, MODEL, 256, 1, err, sizeof err)) {
            fprintf(stderr, "%s\n", err);
            return 1;
        }
        ApusStoreCfg sc = {0};
        sc.n_layers = m.n_layers;
        sc.n_experts = 16;
        sc.slots_per_layer = slots_tab[c];
        sc.cache_bytes = 1;
        sc.pin_bytes = 1;
        sc.io_threads = 4;
        sc.usage_path = "";
        ApusStore *st = apus_store_open(MODEL, &sc, err, sizeof err);
        apus_model_attach_store(&m, st);
        ApusModelState ms;
        apus_model_state_init(&ms, &m);
        size_t V = (size_t)m.vocab;
        float *lg = malloc(V * sizeof(float));
        apus_model_forward(&m, &ms, PROMPT, 8, lg, 0, NULL);
        double t0 = now_s();
        for (int s = 0; s < 24; s++) {
            int tok = apus_sample_argmax(lg, V);
            int64_t id = tok;
            apus_model_forward(&m, &ms, &id, 1, lg, 0, NULL);
        }
        double dt = now_s() - t0;
        ApusStoreStats ss;
        apus_store_stats(st, &ss);
        double hr = (double)ss.hits / (double)(ss.hits + ss.misses);
        printf("%-14d %9.1f%% %8llu %8llu %10.1f\n", slots_tab[c],
               hr * 100, (unsigned long long)ss.preads,
               (unsigned long long)ss.rss_drops, 24.0 / dt);
        free(lg);
        apus_model_state_free(&ms, &m);
        apus_store_close(st);
        apus_model_free(&m);
    }
    return 0;
}
