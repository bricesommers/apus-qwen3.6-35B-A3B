/*
 * tests/m6b/test_recall.c — recall-counter machinery validation: run a
 * piloted greedy decode with the NDJSON P/A dump enabled, then write the
 * live counters. tests/m6b/check_recall.py recomputes the recall from
 * the dump and must match the live C counters EXACTLY.
 *
 * NOTE (documented in tests/m6b/README.md): the fixture model has random
 * weights, so the recall VALUE is random-weight noise (the constant
 * expert_bias dominates selection) — what is validated here is the
 * accounting machinery, not locality.
 *
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

#define MODEL "tests/m6a/fixtures/model"
#define DUMP "tests/m6b/bin/pilot_dump.ndjson"
#define COUNTERS "tests/m6b/bin/pilot_counters.txt"

#define PROMPT_LEN 8
#define STEPS 24

static const int64_t PROMPT[PROMPT_LEN] = { 3, 1, 4, 1, 5, 9, 2, 6 };

int main(void) {
    char err[256];
    ApusModel m;
    if (apus_model_load_ex(&m, MODEL, 256, 1, err, sizeof err)) {
        fprintf(stderr, "FAIL: tiered load: %s\n", err);
        return 1;
    }
    ApusStoreCfg sc = {0};
    sc.n_layers = m.n_layers;
    sc.n_experts = 16;
    sc.slots_per_layer = 8;
    sc.cache_bytes = 1;
    sc.pin_bytes = 1;
    sc.io_threads = 4;
    sc.usage_path = "";
    ApusStore *st = apus_store_open(MODEL, &sc, err, sizeof err);
    if (!st) {
        fprintf(stderr, "FAIL: store open: %s\n", err);
        return 1;
    }
    apus_model_attach_store(&m, st);

    const ApusLayerCfg *mc = &m.layers[0].lc;
    ApusPilotCfg pc = {0};
    pc.store = st;
    pc.n_layers = m.n_layers;
    pc.n_experts = 16;
    pc.top_k = (int)mc->top_k;
    pc.hidden = m.hidden;
    pc.enabled = 1;
    pc.pilot_k = 8;
    pc.ring_entries = 4096;
    pc.dump_path = DUMP;
    ApusPilot *p = apus_pilot_create(&pc);
    if (!p) {
        fprintf(stderr, "FAIL: pilot create\n");
        return 1;
    }
    for (int L = 0; L < m.n_layers; L++) {
        const ApusLayerW *lw = &m.layers[L].lw;
        ApusPilotRouter r = { lw->rtr_w, lw->ln2_w };
        apus_pilot_attach_router(p, L, &r);
    }
    ApusStoreFwdHooks hooks;
    apus_pilot_store_hooks(p, &hooks);
    apus_store_fwd_hooks(st, &hooks);
    apus_pilot_start(p);

    /* greedy decode, 24 steps */
    ApusModelState ms;
    apus_model_state_init(&ms, &m);
    size_t V = (size_t)m.vocab;
    float *lg = malloc(V * sizeof(float));
    apus_model_forward(&m, &ms, PROMPT, PROMPT_LEN, lg, 0, NULL);
    for (int s = 0; s < STEPS; s++) {
        int tok = apus_sample_argmax(lg, V);
        int64_t id = tok;
        apus_model_forward(&m, &ms, &id, 1, lg, 0, NULL);
    }
    apus_model_state_free(&ms, &m);

    ApusPilotStats ps;
    apus_pilot_stats(p, &ps);
    apus_pilot_destroy(p);      /* closes the dump */

    FILE *f = fopen(COUNTERS, "w");
    fprintf(f, "predictions=%llu\n", (unsigned long long)ps.predictions);
    fprintf(f, "pred_experts=%llu\n", (unsigned long long)ps.pred_experts);
    fprintf(f, "actual_experts=%llu\n",
            (unsigned long long)ps.actual_experts);
    fprintf(f, "actual_hits=%llu\n", (unsigned long long)ps.actual_hits);
    fclose(f);
    printf("test_recall: predictions %llu, actual %llu, hits %llu "
           "(recall %.1f%%) — counters written, run check_recall.py\n",
           (unsigned long long)ps.predictions,
           (unsigned long long)ps.actual_experts,
           (unsigned long long)ps.actual_hits,
           ps.actual_experts
               ? 100.0 * (double)ps.actual_hits / (double)ps.actual_experts
               : 0.0);
    free(lg);
    apus_store_close(st);
    apus_model_free(&m);
    return 0;
}
