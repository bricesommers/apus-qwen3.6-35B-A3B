/*
 * c/sample.h — sampling for the apus-ling engine: greedy argmax,
 * temperature + top-k + top-p (nucleus) sampling, and a deterministic
 * RNG. C11, libc only. Ported from Apus (c/sample.h) with top-k added
 * for the Ling defaults.
 *
 * Contract (pinned by tests/m5 against the numpy oracle, tools/oracle.py
 * probs_from_logits/top_kp_draw):
 *   - argmax: lowest index wins exact ties (numpy argmax semantics).
 *   - temperature: z = logits/temp in FP32, max-subtracted softmax, FP32.
 *   - top-k FIRST (HF warper order: TopK then TopP): keep the k
 *     highest-probability tokens (stable on exact ties, lower index
 *     first); top_k <= 0 or >= n disables it.
 *   - top-p: within the kept set, keep token i iff cumsum_before_i <=
 *     top_p (the HF shift rule; always keeps the top token);
 *     renormalize kept mass.
 *   - draw: smallest j with CDF[j] > u, u in [0,1); if rounding leaves no
 *     such j, the last kept token. Given the same probabilities and the
 *     same u, the numpy oracle and this header pick the same token.
 *
 * RNG: splitmix64 — deterministic, explicit seed (CLI --seed), one f64
 * uniform in [0,1) per sampled token. Ling CLI defaults: temperature
 * 0.6, top_p 0.95, top_k 20; temp <= 0 means greedy (--greedy).
 *
 * Usage: #define APUS_SAMPLE_IMPLEMENTATION in exactly one TU.
 */
#ifndef APUS_SAMPLE_H
#define APUS_SAMPLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Greedy: index of the max logit, lowest index on exact ties. */
int apus_sample_argmax(const float *logits, size_t n);

/* Scratch sizing. logits_u needs n floats (probs) + n int32 (order);
 * top_kp needs n int32 only. */
size_t apus_sample_scratch_size(size_t n);    /* n*(sizeof(float)+sizeof(int32_t)) */
size_t apus_top_kp_scratch_size(size_t n);    /* n*sizeof(int32_t) */

/* Top-k + top-p nucleus sample from probabilities with an explicit
 * uniform u in [0,1). probs need not sum exactly to 1 (renormalized on
 * the fly). scratch: apus_top_kp_scratch_size(n) bytes, or NULL for
 * n <= APUS_SAMPLE_STACK_MAX (stack buffer). */
#define APUS_SAMPLE_STACK_MAX 8192
int apus_sample_top_kp_u(const float *probs, size_t n, int top_k,
                         float top_p, double u, void *scratch);

/* Softmax(logits/temp) then top-k/top-p draw with u. temp <= 0 -> argmax
 * (top_k, top_p and u ignored). scratch: apus_sample_scratch_size(n) or
 * NULL for small n. */
int apus_sample_logits_u(const float *logits, size_t n, float temp,
                         int top_k, float top_p, double u, void *scratch);

/* splitmix64 stream. */
typedef struct { uint64_t s; } ApusRng;
void     apus_rng_seed(ApusRng *r, uint64_t seed);
uint64_t apus_rng_next(ApusRng *r);
double   apus_rng_uniform(ApusRng *r);        /* [0,1), 53-bit */

/* Full pipeline with the internal RNG: temp <= 0 -> greedy. */
int apus_sample(const float *logits, size_t n, float temp, int top_k,
                float top_p, ApusRng *rng, void *scratch);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_SAMPLE_IMPLEMENTATION

#include <math.h>
#include <stdlib.h>

int apus_sample_argmax(const float *logits, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;   /* strict >: ties -> lowest */
    return (int)best;
}

size_t apus_sample_scratch_size(size_t n) {
    return n * (sizeof(float) + sizeof(int32_t));
}

size_t apus_top_kp_scratch_size(size_t n) {
    return n * sizeof(int32_t);
}

int apus_sample_top_kp_u(const float *probs, size_t n, int top_k,
                         float top_p, double u, void *scratch) {
    int32_t sbuf[APUS_SAMPLE_STACK_MAX];
    int32_t *ord = scratch ? (int32_t *)scratch : sbuf;
    for (size_t i = 0; i < n; i++) ord[i] = (int32_t)i;
    /* sort ord by prob descending; exact FP32 ties break by lower index
     * (== numpy stable argsort). Shell sort: O(n^1.5) worst, no globals. */
    for (size_t gap = n / 2; gap; gap /= 2)
        for (size_t i = gap; i < n; i++) {
            int32_t v = ord[i];
            size_t j = i;
            while (j >= gap) {
                int32_t w = ord[j - gap];
                if (probs[w] > probs[v] || (probs[w] == probs[v] && w < v))
                    break;
                ord[j] = w;
                j -= gap;
            }
            ord[j] = v;
        }
    /* top-k truncation (before the nucleus, HF warper order) */
    size_t ncand = n;
    if (top_k > 0 && (size_t)top_k < ncand) ncand = (size_t)top_k;
    /* nucleus: keep token i iff cumsum_before_i <= top_p (HF shift rule);
     * always keeps at least the top token */
    double cum = 0.0, kept = 0.0;
    size_t nk = 0;
    for (size_t i = 0; i < ncand; i++) {
        if (cum <= (double)top_p) {
            cum += probs[ord[i]];
            kept = cum;
            nk = i + 1;
        } else {
            break;
        }
    }
    if (nk == 0) { nk = 1; kept = probs[ord[0]]; }
    /* renormalized CDF draw: smallest j with CDF[j] > u */
    double c = 0.0;
    for (size_t i = 0; i < nk; i++) {
        c += (double)probs[ord[i]] / kept;
        if (u < c) return ord[i];
    }
    return ord[nk - 1];
}

int apus_sample_logits_u(const float *logits, size_t n, float temp,
                         int top_k, float top_p, double u, void *scratch) {
    if (temp <= 0.0f) return apus_sample_argmax(logits, n);
    float psbuf[APUS_SAMPLE_STACK_MAX];
    int32_t isbuf[APUS_SAMPLE_STACK_MAX];
    float *p;
    void *oscratch;
    if (scratch) {
        p = (float *)scratch;
        oscratch = (int32_t *)(p + n);
    } else {
        p = psbuf;
        oscratch = isbuf;
    }
    float mx = logits[0];
    for (size_t i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        p[i] = expf((logits[i] - mx) / temp);
        sum += p[i];
    }
    for (size_t i = 0; i < n; i++) p[i] = (float)(p[i] / sum);
    return apus_sample_top_kp_u(p, n, top_k, top_p, u, oscratch);
}

void apus_rng_seed(ApusRng *r, uint64_t seed) { r->s = seed; }

uint64_t apus_rng_next(ApusRng *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

double apus_rng_uniform(ApusRng *r) {
    return (double)(apus_rng_next(r) >> 11) * 0x1.0p-53;
}

int apus_sample(const float *logits, size_t n, float temp, int top_k,
                float top_p, ApusRng *rng, void *scratch) {
    if (temp <= 0.0f) return apus_sample_argmax(logits, n);
    return apus_sample_logits_u(logits, n, temp, top_k, top_p,
                                apus_rng_uniform(rng), scratch);
}

#endif /* APUS_SAMPLE_IMPLEMENTATION */
#endif /* APUS_SAMPLE_H */
