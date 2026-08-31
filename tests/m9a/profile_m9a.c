/*
 * tests/m9a/profile_m9a.c — fixture-forward profile driver (M9). Loads
 * the m5 Qwen fixture model and loops prefill+decode episodes so the
 * CURRENT forward (gemv/gemm hot dispatch + glue) can be profiled
 * end-to-end — run under sample(1) (the output is committed as
 * tests/m9a/profile_sample.txt) or standalone for a throughput number.
 *
 * The fixture is tiny (2 layers, hidden 128, 16 experts, vocab 512), so
 * the glue share here is an UPPER bound vs the real model — the README
 * pairs this with the real-dim component numbers from
 * tests/m9c/profile_phasea.c. Episodes are capped at 128 tokens (the
 * fixture FULL layer's KV cache is max_seq 256).
 *
 * Informational — NOT a gate. Run from the repository root.
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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    long episodes = argc > 1 ? atol(argv[1]) : 2000;
    char err[256];
    ApusModel m;
    if (apus_model_load(&m, "tests/m5/fixtures/model", 256, err,
                        sizeof err)) {
        fprintf(stderr, "profile_m9a: model load: %s\n", err);
        return 1;
    }
    size_t V = (size_t)m.vocab;
    float *logits = malloc(V * sizeof(float));
    int64_t pre[32], dec = 7;
    for (int i = 0; i < 32; i++) pre[i] = (i * 17 + 3) % 509;
    double t0 = now_s();
    long tokens = 0;
    for (long e = 0; e < episodes; e++) {
        ApusModelState s;
        apus_model_state_init(&s, &m);
        apus_model_forward(&m, &s, pre, 32, logits, 0, NULL);
        for (int t = 0; t < 96; t++) {
            dec = (dec * 5 + 11) % 509;
            apus_model_forward(&m, &s, &dec, 1, logits, 0, NULL);
        }
        tokens += 128;
        apus_model_state_free(&s, &m);
    }
    double dt = now_s() - t0;
    printf("profile_m9a: %ld episodes (%ld tokens, 25%% prefill / 75%% "
           "decode) in %.2f s — %.0f tok/s (fixture, threads %d)\n",
           episodes, tokens, dt, tokens / dt, apus_pool_threads());
    printf("  last-logits checksum %08x (informational)\n",
           (unsigned)(logits[0] * 1000.0f) ^ (unsigned)(logits[V - 1]));
    free(logits);
    apus_model_free(&m);
    return 0;
}
