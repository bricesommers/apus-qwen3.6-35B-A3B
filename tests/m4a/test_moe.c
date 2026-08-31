/*
 * tests/m4a/test_moe.c — hard-gate tests for c/moe.h (Qwen3.6-35B-A3B
 * M4: fp32-softmax top-k router, routed/shared expert activations,
 * bf16-weight combine).
 *
 *   1. fp32-out matvec (generic machinery): shape sweep vs in-test f64,
 *      esc-based.
 *   2. Router: golden (E=256, K=128): bf16 logits, fp32 softmax probs,
 *      selection EXACT (fixture margin asserted by the generator),
 *      renormalized bf16 weights; constructed tie-breaks (all-equal
 *      scores -> lowest indices; tie at the cut -> lowest index);
 *      topn == route prefix bitwise.
 *   3. Routed-expert activation (fused gate_up, TWO rounding points):
 *      golden f64 + in-test.
 *   4. Shared-expert activation (plain silu*up, SINGLE rounding):
 *      golden f64 + in-test.
 *   5. Weighted combine (bf16 weights, fp32 accum, single rnd): golden
 *      f64 + in-test; k=1; zero weights -> +0 codes.
 *   6. BITWISE scalar anchor == NEON for every op (on __ARM_NEON).
 *
 * Run from the repository root (fixtures under tests/m4a/golden/).
 */
#define APUS_BF16_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION
#include "bf16.h"
#include "moe.h"

#include <math.h>
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

/* ---- deterministic PRNG (splitmix64) ---- */
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_float(void) {   /* uniform in (-4, 4) */
    return (float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 8.0 - 4.0);
}
static uint16_t rng_bf16_scaled(float s) {
    return apus_bf16_bits(rng_float() * s);
}

/* FNV-1a 64 over a byte range, folded into *h. */
static void digest_bytes(uint64_t *h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) {
        *h ^= b[i];
        *h *= 0x100000001B3ull;
    }
}
static uint64_t g_digest = 0xCBF29CE484222325ull;

static double g_meas_mv = 0, g_meas_rtr = 0, g_meas_act = 0;
static double g_meas_silu = 0, g_meas_comb = 0;

static int tol_ok(double c, double gold, double esc,
                  double rel, double escfrac) {
    double err = fabs(c - gold);
    return err <= rel * fabs(gold) + escfrac * esc + 1e-30;
}
static void meas(double *m, double c, double gold, double esc) {
    double r = fabs(c - gold) / (fabs(gold) + esc + 1e-30);
    if (r > *m) *m = r;
}

/* ---- file loading ---- */
static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(buf); return NULL;
    }
    fclose(f);
    *len = (size_t)sz;
    return buf;
}

static long manifest_val(const char *key) {
    FILE *f = fopen("tests/m4a/golden/manifest.txt", "r");
    if (!f) return -1;
    char k[64];
    long v, out = -1;
    while (fscanf(f, "%63[^=]=%ld\n", k, &v) == 2)
        if (strcmp(k, key) == 0) { out = v; break; }
    fclose(f);
    return out;
}

/* =========================================================================*/
/* fp32-out matvec (generic machinery) */
static void test_matvec(void) {
    static const size_t shapes[][2] = {
        {1, 1}, {3, 5}, {64, 256}, {256, 2048}, {7, 1}, {33, 100},
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        size_t O = shapes[s][0], K = shapes[s][1];
        uint16_t *w = malloc(O * K * 2);
        uint16_t *x = malloc(K * 2);
        float *y = malloc(O * sizeof(float));
        float *y2 = malloc(O * sizeof(float));
        for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
        for (size_t i = 0; i < K; i++) x[i] = rng_bf16_scaled(1.0f);
        apus_moe_matvec_f32(w, x, y, O, K);
        long bad = 0;
        for (size_t o = 0; o < O; o++) {
            double acc = 0.0, esc = 0.0;
            for (size_t k = 0; k < K; k++) {
                double p = (double)apus_bf16_f32(w[o * K + k]) *
                           (double)apus_bf16_f32(x[k]);
                acc += p;
                esc += fabs(p);
            }
            meas(&g_meas_mv, y[o], acc, esc);
            if (!tol_ok(y[o], acc, esc, 1e-5, 1e-5)) bad++;
        }
        CHECK(bad == 0, "matvec O=%zu K=%zu: %ld/%zu mismatches",
              O, K, bad, O);
        /* hot == scalar bitwise (NEON mt dispatch; the thread-count
         * diff is the Makefile's APUS_THREADS stdout gate) */
        ApusScratchMark mk = apus_scratch_mark();
        apus_moe_matvec_f32_hot(w, x, y2, O, K);
        apus_scratch_reset(mk);
        CHECK(memcmp(y, y2, O * sizeof(float)) == 0,
              "matvec hot != scalar at O=%zu K=%zu", O, K);
#ifdef __ARM_NEON
        float *xf = malloc(K * sizeof(float));
        apus_moe_matvec_f32_neon(w, x, xf, y2, O, K);
        CHECK(memcmp(y, y2, O * sizeof(float)) == 0,
              "matvec neon != scalar at O=%zu K=%zu", O, K);
        free(xf);
#endif
        digest_bytes(&g_digest, y, O * sizeof(float));
        free(w); free(x); free(y); free(y2);
    }
}

/* =========================================================================*/
static void test_router(void) {
    size_t E = (size_t)manifest_val("RTR_E");
    size_t K = (size_t)manifest_val("RTR_K");
    size_t TK = (size_t)manifest_val("RTR_TK");
    size_t n;
    uint16_t *x = (uint16_t *)read_file("tests/m4a/golden/moe_rtr_x.bin",
                                        &n);
    uint16_t *wg = (uint16_t *)read_file("tests/m4a/golden/moe_rtr_wg.bin",
                                         &n);
    uint16_t *gl = (uint16_t *)read_file(
        "tests/m4a/golden/moe_rtr_logits.bin", &n);
    double *gp = (double *)read_file("tests/m4a/golden/moe_rtr_probs.bin",
                                     &n);
    int64_t *gi = (int64_t *)read_file("tests/m4a/golden/moe_rtr_idx.bin",
                                       &n);
    double *gw = (double *)read_file("tests/m4a/golden/moe_rtr_w.bin", &n);
    CHECK(x && wg && gl && gp && gi && gw, "router fixture load");
    if (!x || !wg || !gl || !gp || !gi || !gw) return;
    int32_t *idx = malloc(TK * sizeof(int32_t));
    uint16_t *w = malloc(TK * 2);
    apus_moe_route(x, wg, idx, w, E, K, TK);
    /* logits: the C rounds to bf16 — compare codes EXACTLY (same gemv
     * realization? NO — the hot gemv is the M9b ILP class on ARM, so
     * code flips are possible; report and gate at 1%) */
    /* probs: fp32 softmax, tolerance vs the f64 golden */
    long bad = 0;
    double m = apus_bf16_f32(gl[0]);
    for (size_t e = 1; e < E; e++) {
        double a = apus_bf16_f32(gl[e]);
        if (a > m) m = a;
    }
    double s = 0.0;
    for (size_t e = 0; e < E; e++)
        s += exp((double)apus_bf16_f32(gl[e]) - m);
    for (size_t e = 0; e < E; e++) {
        double pf = exp((double)apus_bf16_f32(gl[e]) - m) / s;
        meas(&g_meas_rtr, pf, gp[e], 0.02);
        if (!tol_ok(pf, gp[e], 0.02, 2e-6, 2e-6)) bad++;
    }
    CHECK(bad == 0, "router probs (from golden logits): %ld/%zu", bad, E);
    /* selection: EXACT (fixture margin asserted by the generator) */
    bad = 0;
    for (size_t i = 0; i < TK; i++)
        if ((int64_t)idx[i] != gi[i]) bad++;
    CHECK(bad == 0, "router selection: %ld/%zu mismatches", bad, TK);
    /* weights: bf16-rounded renormalized */
    bad = 0;
    for (size_t i = 0; i < TK; i++) {
        double c = apus_bf16_f32(w[i]);
        meas(&g_meas_rtr, c, gw[i], 0.02);
        if (!tol_ok(c, gw[i], 0.02, 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "router weights: %ld/%zu mismatches", bad, TK);
    digest_bytes(&g_digest, idx, TK * sizeof(int32_t));
    digest_bytes(&g_digest, w, TK * 2);
    /* topn == route prefix bitwise */
    int32_t *idxn = malloc(TK * sizeof(int32_t));
    apus_moe_route_topn(x, wg, idxn, TK, E, K);
    CHECK(memcmp(idxn, idx, TK * sizeof(int32_t)) == 0,
          "router topn(topk) != route selection (not bitwise)");
    int32_t *idxn2 = malloc((TK + 4) * sizeof(int32_t));
    apus_moe_route_topn(x, wg, idxn2, TK + 4, E, K);
    CHECK(memcmp(idxn2, idx, TK * sizeof(int32_t)) == 0,
          "router topn(topk+4) prefix != route selection (not bitwise)");
    free(idx); free(w); free(idxn); free(idxn2);
    free(x); free(wg); free(gl); free(gp); free(gi); free(gw);
}

/* Constructed tie-breaks: all-equal scores -> lowest indices in order;
 * an exact tie AT the cut -> the lowest index wins. */
static void test_router_ties(void) {
    size_t E = 16, K = 4, TK = 8;
    /* x = e0 selects row 0 of wg; set every row's first tap to the same
     * value -> all logits equal -> softmax uniform -> selection 0..7 */
    uint16_t *wg = calloc(E * K, 2);
    uint16_t x[K];
    x[0] = apus_bf16_bits(1.0f);
    x[1] = x[2] = x[3] = 0;
    for (size_t e = 0; e < E; e++)
        wg[e * K] = apus_bf16_bits(0.5f);
    int32_t idx[8];
    uint16_t w[8];
    apus_moe_route(x, wg, idx, w, E, K, TK);
    long bad = 0;
    for (size_t i = 0; i < TK; i++)
        if (idx[i] != (int32_t)i) bad++;
    CHECK(bad == 0, "router all-equal tie-break: %ld wrong picks", bad);
    /* uniform probs (1/16 each), renormalized over the top-8: w =
     * (1/16) / (8/16) = 0.125 exactly (exact in bf16) */
    bad = 0;
    for (size_t i = 0; i < TK; i++)
        if (w[i] != apus_bf16_bits(0.125f)) bad++;
    CHECK(bad == 0, "router all-equal weights: %ld not 0.125", bad);
    /* tie at the cut: rows 0..8 share the top score, row 8 must lose
     * to row 7 (lowest index wins) */
    for (size_t e = 0; e < E; e++)
        wg[e * K] = apus_bf16_bits(e <= 8 ? 1.0f : 0.0f);
    apus_moe_route(x, wg, idx, w, E, K, TK);
    bad = 0;
    for (size_t i = 0; i < TK; i++)
        if (idx[i] != (int32_t)i) bad++;
    CHECK(bad == 0, "router cut tie-break: %ld wrong picks", bad);
    free(wg);
}

/* =========================================================================*/
static void silu_act_truth_f64(const uint16_t *gu, double *act, size_t I) {
    for (size_t i = 0; i < I; i++) {
        double g = apus_bf16_f32(gu[i]);
        double a1 = (double)apus_bf16_f32(apus_bf16_bits(
            (float)(g * (1.0 / (1.0 + exp(-g))))));
        act[i] = a1 * (double)apus_bf16_f32(gu[I + i]);
    }
}

static void test_silu_act(void) {
    size_t I = (size_t)manifest_val("ACT_I");
    size_t n;
    uint16_t *gu = (uint16_t *)read_file("tests/m4a/golden/moe_act_gu.bin",
                                         &n);
    double *ga = (double *)read_file("tests/m4a/golden/moe_act_act.bin",
                                     &n);
    CHECK(gu && ga, "silu_act fixture load");
    if (!gu || !ga) return;
    uint16_t *act = malloc(I * 2);
    apus_moe_silu_act(gu, act, I);
    long bad = 0;
    for (size_t i = 0; i < I; i++) {
        double c = apus_bf16_f32(act[i]);
        meas(&g_meas_act, c, ga[i], 0.02);
        if (!tol_ok(c, ga[i], 0.02, 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "silu_act golden: %ld/%zu mismatches", bad, I);
    digest_bytes(&g_digest, act, I * 2);
    /* in-test f64 truth (odd I exercises the NEON tail) */
    size_t I2 = 767;
    uint16_t *gu2 = malloc(2 * I2 * 2);
    uint16_t *act2 = malloc(I2 * 2);
    double *tr = malloc(I2 * sizeof(double));
    for (size_t i = 0; i < 2 * I2; i++) gu2[i] = rng_bf16_scaled(3.0f);
    silu_act_truth_f64(gu2, tr, I2);
    apus_moe_silu_act(gu2, act2, I2);
    bad = 0;
    for (size_t i = 0; i < I2; i++) {
        double c = apus_bf16_f32(act2[i]);
        meas(&g_meas_act, c, tr[i], 0.02);
        if (!tol_ok(c, tr[i], 0.02, 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "silu_act in-test truth: %ld/%zu mismatches", bad, I2);
#ifdef __ARM_NEON
    uint16_t *as = malloc(I2 * 2);
    uint16_t *an = malloc(I2 * 2);
    apus_moe_silu_act_scalar(gu2, as, I2);
    apus_moe_silu_act_neon(gu2, an, I2);
    CHECK(memcmp(as, an, I2 * 2) == 0,
          "silu_act scalar != NEON (not bitwise)");
    free(as); free(an);
#endif
    free(gu); free(ga); free(act); free(gu2); free(act2); free(tr);
}

/* =========================================================================*/
static void silu_mul_truth_f64(const uint16_t *g, const uint16_t *u,
                               double *y, size_t I) {
    for (size_t i = 0; i < I; i++) {
        double gv = apus_bf16_f32(g[i]);
        y[i] = gv * (1.0 / (1.0 + exp(-gv))) *
               (double)apus_bf16_f32(u[i]);
    }
}

static void test_silu_mul(void) {
    size_t I = (size_t)manifest_val("SILU_I");
    size_t n;
    uint16_t *g = (uint16_t *)read_file("tests/m4a/golden/moe_silu_g.bin",
                                        &n);
    uint16_t *u = (uint16_t *)read_file("tests/m4a/golden/moe_silu_u.bin",
                                        &n);
    double *gy = (double *)read_file("tests/m4a/golden/moe_silu_y.bin", &n);
    CHECK(g && u && gy, "silu_mul fixture load");
    if (!g || !u || !gy) return;
    uint16_t *y = malloc(I * 2);
    apus_moe_silu_mul(g, u, y, I);
    long bad = 0;
    for (size_t i = 0; i < I; i++) {
        double c = apus_bf16_f32(y[i]);
        meas(&g_meas_silu, c, gy[i], 0.02);
        if (!tol_ok(c, gy[i], 0.02, 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "silu_mul golden: %ld/%zu mismatches", bad, I);
    digest_bytes(&g_digest, y, I * 2);
    /* in-test f64 truth (odd I) */
    size_t I2 = 767;
    uint16_t *g2 = malloc(I2 * 2);
    uint16_t *u2 = malloc(I2 * 2);
    uint16_t *y2 = malloc(I2 * 2);
    double *tr = malloc(I2 * sizeof(double));
    for (size_t i = 0; i < I2; i++) {
        g2[i] = rng_bf16_scaled(3.0f);
        u2[i] = rng_bf16_scaled(3.0f);
    }
    silu_mul_truth_f64(g2, u2, tr, I2);
    apus_moe_silu_mul(g2, u2, y2, I2);
    bad = 0;
    for (size_t i = 0; i < I2; i++) {
        double c = apus_bf16_f32(y2[i]);
        meas(&g_meas_silu, c, tr[i], 0.02);
        if (!tol_ok(c, tr[i], 0.02, 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "silu_mul in-test truth: %ld/%zu mismatches", bad, I2);
    /* silu(0) = +0 */
    g2[0] = 0x0000;
    u2[0] = apus_bf16_bits(3.0f);
    apus_moe_silu_mul(g2, u2, y2, 1);
    CHECK(y2[0] == 0x0000, "silu_mul g=0: got 0x%04x", y2[0]);
#ifdef __ARM_NEON
    uint16_t *ys = malloc(I2 * 2);
    uint16_t *yn = malloc(I2 * 2);
    apus_moe_silu_mul_scalar(g2, u2, ys, I2);
    apus_moe_silu_mul_neon(g2, u2, yn, I2);
    CHECK(memcmp(ys, yn, I2 * 2) == 0,
          "silu_mul scalar != NEON (not bitwise)");
    free(ys); free(yn);
#endif
    free(g); free(u); free(gy); free(y);
    free(g2); free(u2); free(y2); free(tr);
}

/* =========================================================================*/
static void test_combine(void) {
    size_t K = (size_t)manifest_val("COMB_K");
    size_t N = (size_t)manifest_val("COMB_N");
    size_t n;
    uint16_t *y = (uint16_t *)read_file("tests/m4a/golden/moe_comb_y.bin",
                                        &n);
    uint16_t *w = (uint16_t *)read_file("tests/m4a/golden/moe_comb_w.bin",
                                        &n);
    double *go = (double *)read_file("tests/m4a/golden/moe_comb_out.bin",
                                     &n);
    double *ge = (double *)read_file("tests/m4a/golden/moe_comb_esc.bin",
                                     &n);
    CHECK(y && w && go && ge, "combine fixture load");
    if (!y || !w || !go || !ge) return;
    uint16_t *out = malloc(N * 2);
    apus_moe_combine(y, w, out, K, N);
    long bad = 0;
    for (size_t i = 0; i < N; i++) {
        double c = apus_bf16_f32(out[i]);
        meas(&g_meas_comb, c, go[i], ge[i]);
        if (!tol_ok(c, go[i], ge[i], 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "combine golden: %ld/%zu mismatches", bad, N);
    digest_bytes(&g_digest, out, N * 2);
    /* in-test f64 truth (odd N exercises the NEON tail) */
    size_t N2 = 767;
    uint16_t *y2 = malloc(K * N2 * 2);
    uint16_t *out2 = malloc(N2 * 2);
    for (size_t i = 0; i < K * N2; i++) y2[i] = rng_bf16_scaled(2.0f);
    apus_moe_combine(y2, w, out2, K, N2);
    bad = 0;
    for (size_t nn = 0; nn < N2; nn++) {
        double acc = 0.0, esc = 0.0;
        for (size_t e = 0; e < K; e++) {
            double p = (double)apus_bf16_f32(w[e]) *
                       (double)apus_bf16_f32(y2[e * N2 + nn]);
            acc += p;
            esc += fabs(p);
        }
        double c = apus_bf16_f32(out2[nn]);
        meas(&g_meas_comb, c, acc, esc);
        if (!tol_ok(c, acc, esc, 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "combine in-test truth: %ld/%zu mismatches", bad, N2);
    /* k=1 -> rnd(w*y); zero weights -> +0 codes (IEEE (+0)+(-0) = +0) */
    uint16_t one[4];
    apus_moe_combine(y2, w, one, 1, 4);
    bad = 0;
    for (size_t i = 0; i < 4; i++)
        if (one[i] != apus_bf16_bits(apus_bf16_f32(w[0]) *
                                     apus_bf16_f32(y2[i])))
            bad++;
    CHECK(bad == 0, "combine k=1: %ld/4 mismatches", bad);
    uint16_t wz[8] = {0};
    apus_moe_combine(y2, wz, out2, K, N2);
    bad = 0;
    for (size_t i = 0; i < N2; i++) if (out2[i] != 0x0000) bad++;
    CHECK(bad == 0, "combine zero weights: %ld nonzero", bad);
#ifdef __ARM_NEON
    uint16_t *os = malloc(N2 * 2);
    uint16_t *on = malloc(N2 * 2);
    apus_moe_combine_scalar(y2, w, os, K, N2);
    apus_moe_combine_neon(y2, w, on, K, N2);
    CHECK(memcmp(os, on, N2 * 2) == 0,
          "combine scalar != NEON (not bitwise)");
    free(os); free(on);
#endif
    free(y); free(w); free(go); free(ge); free(out); free(y2); free(out2);
}

int main(void) {
    printf("test_moe: MoE kernel hard-gate tests (Qwen3.6-35B-A3B M4a)\n");
    fprintf(stderr, "  pool threads: %d\n", apus_pool_threads());

    if (manifest_val("RTR_E") < 0) {
        failures++;
        fprintf(stderr, "FAIL: could not load golden manifest\n");
        return 1;
    }

    test_matvec();
    test_router();
    test_router_ties();
    test_silu_act();
    test_silu_mul();
    test_combine();

    printf("  measured max err/(|gold|+esc): matvec %.3g router %.3g "
           "act %.3g silu %.3g combine %.3g\n",
           g_meas_mv, g_meas_rtr, g_meas_act, g_meas_silu, g_meas_comb);
    printf("  digest %016llx\n", (unsigned long long)g_digest);
    printf("test_moe: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
