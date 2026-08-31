/*
 * tests/m4a/test_gdn.c — hard-gate tests for c/gdn.h (Gated DeltaNet
 * kernels, Qwen3.6-35B-A3B M4).
 *
 *   1. Fused depthwise causal conv1d k=4 + SiLU (TWO rounding points):
 *      golden f64 truth (esc-based tolerance), continuation tokens
 *      (state wrap), BITWISE prefill == decode-stepping, state == last 3
 *      pre-conv inputs (exact codes), in-test f64 truth on random odd
 *      shapes.
 *   2. Decay gate (softplus form, NO lower bound): golden f64, corners
 *      (a+dt = 0 -> -exp(A_log)*log2; x>20 identity branch; exp tail),
 *      in-test f64 truth.
 *   3. Beta (BF16 sigmoid): golden f64 + exact corners (b=0 -> 0.5
 *      code 0x3F00; +20 -> ~1; -20 -> ~0).
 *   4. L2 norm (fp32 out, x*rsqrt, q scale): golden f64 (incl. tiny
 *      row), zero vector, in-test f64 truth.
 *   5. Delta-rule recurrence (fp32 q,k in, per-head scalar g, bf16
 *      beta): golden f64 for o and the final state S (esc-based),
 *      BITWISE apus_gdn_recurrent == stepping apus_gdn_step, decay
 *      extremes, in-test f64 truth.
 *   6. RMSNormGated output norm (DIRECT weight, three rounding points):
 *      golden f64 (incl. tiny-magnitude head), in-test f64 truth.
 *   7. BITWISE scalar anchor == NEON for every op (on __ARM_NEON), and
 *      mt == sequential for the recurrence step.
 *
 * fp32-out ops are gated on err/esc < 1e-5 (one fp32 rounding per op vs
 * f64 truth); bf16-out ops on ~2 bf16 ulp (2^-7 relative) + esc slack.
 * Run from the repository root (fixtures under tests/m4a/golden/).
 */
#define APUS_BF16_IMPLEMENTATION
#define APUS_GDN_IMPLEMENTATION
#include "bf16.h"
#include "gdn.h"

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

static uint32_t f32bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

/* FNV-1a 64 over a byte range, folded into *h. */
static void digest_bytes(uint64_t *h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) {
        *h ^= b[i];
        *h *= 0x100000001B3ull;
    }
}
static uint64_t g_digest = 0xCBF29CE484222325ull;

/* measured maxima (printed for the README) */
static double g_meas_conv = 0, g_meas_decay = 0, g_meas_beta = 0;
static double g_meas_l2 = 0, g_meas_reco = 0, g_meas_recS = 0;
static double g_meas_onorm = 0;

/* tolerance: |c - gold| <= rel*|gold| + escfrac*esc + floor */
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
/* in-test f64 truth: conv (zero state, two rounding points) */
static void conv_truth_f64(const uint16_t *x, const uint16_t *w,
                           size_t C, size_t T, double *out, double *esc) {
    for (size_t t = 0; t < T; t++) {
        for (size_t c = 0; c < C; c++) {
            double acc = 0.0, sc = 0.0;
            for (size_t i = 0; i < 4; i++) {
                long ti = (long)t - 3 + (long)i;
                if (ti < 0) continue;
                double p = (double)apus_bf16_f32(w[c * 4 + i]) *
                           (double)apus_bf16_f32(x[(size_t)ti * C + c]);
                acc += p;
                sc += fabs(p);
            }
            double co = (double)apus_bf16_f32(apus_bf16_bits((float)acc));
            out[t * C + c] = co * (1.0 / (1.0 + exp(-co)));
            esc[t * C + c] = sc;
        }
    }
}

static void test_conv_golden(void) {
    size_t C = (size_t)manifest_val("CONV_C");
    size_t T = (size_t)manifest_val("CONV_T");
    size_t T2 = (size_t)manifest_val("CONV_T2");
    size_t n;
    uint16_t *x = (uint16_t *)read_file("tests/m4a/golden/gdn_conv_x.bin", &n);
    uint16_t *w = (uint16_t *)read_file("tests/m4a/golden/gdn_conv_w.bin", &n);
    uint16_t *x2 = (uint16_t *)read_file("tests/m4a/golden/gdn_conv_x2.bin", &n);
    double *go = (double *)read_file("tests/m4a/golden/gdn_conv_out.bin", &n);
    double *ge = (double *)read_file("tests/m4a/golden/gdn_conv_esc.bin", &n);
    double *go2 = (double *)read_file("tests/m4a/golden/gdn_conv_out2.bin", &n);
    CHECK(x && w && x2 && go && ge && go2, "conv fixture load");
    if (!x || !w || !x2 || !go || !ge || !go2) return;

    uint16_t *out = malloc(T * C * 2);
    uint16_t *state = calloc(C * 3, 2);
    apus_gdn_conv1d(x, w, out, C, T, state);
    long bad = 0;
    for (size_t i = 0; i < T * C; i++) {
        double c = apus_bf16_f32(out[i]);
        meas(&g_meas_conv, c, go[i], ge[i]);
        if (!tol_ok(c, go[i], ge[i], 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "conv golden: %ld/%zu mismatches", bad, T * C);
    digest_bytes(&g_digest, out, T * C * 2);

    /* state after T tokens == last 3 pre-conv inputs (exact codes) */
    long sbad = 0;
    for (size_t c = 0; c < C; c++)
        for (size_t j = 0; j < 3; j++)
            if (state[c * 3 + j] != x[(T - 3 + j) * C + c]) sbad++;
    CHECK(sbad == 0, "conv state codes: %ld mismatches", sbad);

    /* decode continuation vs golden out2, and BITWISE prefill ==
     * decode-stepping over the concatenated sequence */
    uint16_t *out2 = malloc(T2 * C * 2);
    uint16_t *st_dec = malloc(C * 3 * 2);
    memcpy(st_dec, state, C * 3 * 2);
    for (size_t t = 0; t < T2; t++)
        apus_gdn_conv1d_step(x2 + t * C, w, out2 + t * C, C, st_dec);
    bad = 0;
    for (size_t i = 0; i < T2 * C; i++) {
        double c = apus_bf16_f32(out2[i]);
        if (!tol_ok(c, go2[i], 1.0, 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "conv continuation golden: %ld/%zu mismatches",
          bad, T2 * C);

    uint16_t *xall = malloc((T + T2) * C * 2);
    memcpy(xall, x, T * C * 2);
    memcpy(xall + T * C, x2, T2 * C * 2);
    uint16_t *outall = malloc((T + T2) * C * 2);
    uint16_t *st_all = calloc(C * 3, 2);
    apus_gdn_conv1d(xall, w, outall, C, T + T2, st_all);
    CHECK(memcmp(outall, out, T * C * 2) == 0 &&
          memcmp(outall + T * C, out2, T2 * C * 2) == 0,
          "conv prefill != decode-stepping (not bitwise)");
    CHECK(memcmp(st_all, st_dec, C * 3 * 2) == 0,
          "conv final state prefill != decode (not bitwise)");
    digest_bytes(&g_digest, out2, T2 * C * 2);
    digest_bytes(&g_digest, st_all, C * 3 * 2);

    free(x); free(w); free(x2); free(go); free(ge); free(go2);
    free(out); free(out2); free(state); free(st_dec); free(xall);
    free(outall); free(st_all);
}

static void test_conv_intest(void) {
    /* odd channel count, random data, in-test f64 truth; decode == prefill */
    size_t C = 37, T = 9;
    uint16_t *x = malloc(T * C * 2);
    uint16_t *w = malloc(C * 4 * 2);
    for (size_t i = 0; i < T * C; i++) x[i] = rng_bf16_scaled(1.5f);
    for (size_t i = 0; i < C * 4; i++) w[i] = rng_bf16_scaled(0.7f);
    double *tr = malloc(T * C * sizeof(double));
    double *te = malloc(T * C * sizeof(double));
    conv_truth_f64(x, w, C, T, tr, te);
    uint16_t *out = malloc(T * C * 2);
    uint16_t *st = calloc(C * 3, 2);
    apus_gdn_conv1d(x, w, out, C, T, st);
    long bad = 0;
    for (size_t i = 0; i < T * C; i++) {
        double c = apus_bf16_f32(out[i]);
        meas(&g_meas_conv, c, tr[i], te[i]);
        if (!tol_ok(c, tr[i], te[i], 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "conv in-test truth: %ld/%zu mismatches", bad, T * C);
    uint16_t *outd = malloc(T * C * 2);
    uint16_t *std_ = calloc(C * 3, 2);
    for (size_t t = 0; t < T; t++)
        apus_gdn_conv1d_step(x + t * C, w, outd + t * C, C, std_);
    CHECK(memcmp(out, outd, T * C * 2) == 0 && memcmp(st, std_, C * 3 * 2) == 0,
          "conv in-test prefill != decode (not bitwise)");
    /* zero input -> silu(0) = 0 -> +0 codes */
    uint16_t *zx = calloc(C, 2);
    uint16_t *zo = malloc(C * 2);
    uint16_t *zst = calloc(C * 3, 2);
    apus_gdn_conv1d_step(zx, w, zo, C, zst);
    long zbad = 0;
    for (size_t c = 0; c < C; c++) if (zo[c] != 0x0000) zbad++;
    CHECK(zbad == 0, "conv zero input: %ld nonzero outputs", zbad);
#ifdef __ARM_NEON
    /* BITWISE: scalar anchor == NEON (odd C exercises the tail) */
    uint16_t *outs = malloc(T * C * 2);
    uint16_t *outn = malloc(T * C * 2);
    uint16_t *sts = calloc(C * 3, 2);
    uint16_t *stn = calloc(C * 3, 2);
    for (size_t t = 0; t < T; t++) {
        apus_gdn_conv1d_step_scalar(x + t * C, w, outs + t * C, C, sts);
        apus_gdn_conv1d_step_neon(x + t * C, w, outn + t * C, C, stn);
    }
    CHECK(memcmp(outs, outn, T * C * 2) == 0 &&
          memcmp(sts, stn, C * 3 * 2) == 0,
          "conv scalar != NEON (not bitwise)");
    free(outs); free(outn); free(sts); free(stn);
#endif
    free(x); free(w); free(tr); free(te); free(out); free(st);
    free(outd); free(std_); free(zx); free(zo); free(zst);
}

/* =========================================================================*/
static double softplus_f64(double x) {
    return x > 20.0 ? x : log1p(exp(x));
}

static void decay_truth_f64(const uint16_t *a, const float *A_log,
                            const float *dt, double *g, size_t H) {
    for (size_t h = 0; h < H; h++) {
        double t = (double)apus_bf16_f32(a[h]) + (double)dt[h];
        g[h] = -exp((double)A_log[h]) * softplus_f64(t);
    }
}

static void test_decay_golden(void) {
    size_t H = (size_t)manifest_val("DECAY_H");
    size_t n;
    uint16_t *a = (uint16_t *)read_file("tests/m4a/golden/gdn_decay_a.bin", &n);
    float *A = (float *)read_file("tests/m4a/golden/gdn_decay_alog.bin", &n);
    float *dt = (float *)read_file("tests/m4a/golden/gdn_decay_dt.bin", &n);
    double *gg = (double *)read_file("tests/m4a/golden/gdn_decay_g.bin", &n);
    CHECK(a && A && dt && gg, "decay fixture load");
    if (!a || !A || !dt || !gg) return;
    float *g = malloc(H * sizeof(float));
    apus_gdn_decay(a, A, dt, g, H);
    long bad = 0;
    for (size_t i = 0; i < H; i++) {
        meas(&g_meas_decay, g[i], gg[i], 5.0);
        if (!tol_ok(g[i], gg[i], 5.0, 2e-6, 2e-7)) bad++;
    }
    CHECK(bad == 0, "decay golden: %ld/%zu mismatches", bad, H);
    /* corner: a+dt == 0 -> -exp(A_log)*log(2) */
    double c0 = -exp((double)A[0]) * log(2.0);
    CHECK(fabs((double)g[0] - c0) <= 1e-6 * fabs(c0),
          "decay a+dt=0 corner: got %.9g want %.9g", g[0], c0);
    /* corner: a+dt = 25 -> the x>20 identity branch */
    double c1 = -exp((double)A[1]) * 25.0;
    CHECK(fabs((double)g[1] - c1) <= 1e-5 * fabs(c1),
          "decay x>20 corner: got %.9g want %.9g", g[1], c1);
    digest_bytes(&g_digest, g, H * sizeof(float));
#ifdef __ARM_NEON
    float *gs = malloc(H * sizeof(float));
    float *gn = malloc(H * sizeof(float));
    apus_gdn_decay_scalar(a, A, dt, gs, H);
    apus_gdn_decay_neon(a, A, dt, gn, H);
    CHECK(memcmp(gs, gn, H * sizeof(float)) == 0,
          "decay scalar != NEON (not bitwise)");
    free(gs); free(gn);
#endif
    free(a); free(A); free(dt); free(gg); free(g);
}

static void test_decay_intest(void) {
    size_t H = 13;                 /* odd: exercises the NEON tail */
    uint16_t *a = malloc(H * 2);
    float *A = malloc(H * sizeof(float));
    float *dt = malloc(H * sizeof(float));
    double *tr = malloc(H * sizeof(double));
    float *g = malloc(H * sizeof(float));
    for (size_t i = 0; i < H; i++) a[i] = rng_bf16_scaled(2.0f);
    for (size_t h = 0; h < H; h++)
        A[h] = (float)((rng_u64() % 2000) / 1000.0 - 1.0);
    for (size_t i = 0; i < H; i++) dt[i] = rng_float() * 0.5f;
    decay_truth_f64(a, A, dt, tr, H);
    apus_gdn_decay(a, A, dt, g, H);
    long bad = 0;
    for (size_t i = 0; i < H; i++) {
        meas(&g_meas_decay, g[i], tr[i], 5.0);
        if (!tol_ok(g[i], tr[i], 5.0, 2e-6, 2e-7)) bad++;
    }
    CHECK(bad == 0, "decay in-test truth: %ld/%zu mismatches", bad, H);
    free(a); free(A); free(dt); free(tr); free(g);
}

/* =========================================================================*/
static void test_beta(void) {
    size_t H = (size_t)manifest_val("BETA_H");
    size_t n;
    uint16_t *b = (uint16_t *)read_file("tests/m4a/golden/gdn_beta_b.bin", &n);
    double *gb = (double *)read_file("tests/m4a/golden/gdn_beta_beta.bin", &n);
    CHECK(b && gb, "beta fixture load");
    if (!b || !gb) return;
    uint16_t *beta = malloc(H * 2);
    apus_gdn_beta(b, beta, H);
    long bad = 0;
    for (size_t i = 0; i < H; i++) {
        double c = apus_bf16_f32(beta[i]);
        meas(&g_meas_beta, c, gb[i], 0.0);
        if (!tol_ok(c, gb[i], 0.02, 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "beta golden: %ld/%zu mismatches", bad, H);
    /* exact corners: b=0 -> sigmoid 0.5 -> code 0x3F00; +20 -> rounds
     * to exactly 1.0 (0x3F80) */
    CHECK(beta[0] == 0x3F00, "beta b=0 corner: got 0x%04x", beta[0]);
    CHECK(beta[1] == 0x3F80, "beta b=+20 corner: got 0x%04x", beta[1]);
    CHECK(apus_bf16_f32(beta[2]) < 1e-6f,
          "beta b=-20 corner: got %.6g", apus_bf16_f32(beta[2]));
    digest_bytes(&g_digest, beta, H * 2);
#ifdef __ARM_NEON
    uint16_t *bs = malloc(H * 2);
    uint16_t *bn = malloc(H * 2);
    apus_gdn_beta_scalar(b, bs, H);
    apus_gdn_beta_neon(b, bn, H);
    CHECK(memcmp(bs, bn, H * 2) == 0, "beta scalar != NEON (not bitwise)");
    free(bs); free(bn);
#endif
    free(b); free(gb); free(beta);
}

/* =========================================================================*/
static void l2_truth_f64(const uint16_t *x, double *y, size_t H, size_t D,
                         double scale) {
    for (size_t h = 0; h < H; h++) {
        double ss = 0.0;
        for (size_t i = 0; i < D; i++) {
            double v = apus_bf16_f32(x[h * D + i]);
            ss += v * v;
        }
        double rs = 1.0 / sqrt(ss + 1e-6);
        for (size_t i = 0; i < D; i++)
            y[h * D + i] = (double)apus_bf16_f32(x[h * D + i]) * rs * scale;
    }
}

static void test_l2_golden(void) {
    size_t H = (size_t)manifest_val("L2_H");
    size_t D = (size_t)manifest_val("L2_D");
    size_t n;
    uint16_t *x = (uint16_t *)read_file("tests/m4a/golden/gdn_l2_x.bin", &n);
    double *gq = (double *)read_file("tests/m4a/golden/gdn_l2_yq.bin", &n);
    double *gk = (double *)read_file("tests/m4a/golden/gdn_l2_yk.bin", &n);
    CHECK(x && gq && gk, "l2 fixture load");
    if (!x || !gq || !gk) return;
    float qscale = (float)pow((double)D, -0.5);
    float *y = malloc(H * D * sizeof(float));
    apus_gdn_l2norm(x, y, H, D, qscale);
    long bad = 0;
    for (size_t i = 0; i < H * D; i++) {
        meas(&g_meas_l2, y[i], gq[i], 0.3);
        if (!tol_ok(y[i], gq[i], 0.3, 2e-6, 2e-6)) bad++;
    }
    CHECK(bad == 0, "l2 golden (q scale): %ld/%zu mismatches", bad, H * D);
    apus_gdn_l2norm(x, y, H, D, 1.0f);
    bad = 0;
    for (size_t i = 0; i < H * D; i++) {
        meas(&g_meas_l2, y[i], gk[i], 0.3);
        if (!tol_ok(y[i], gk[i], 0.3, 2e-6, 2e-6)) bad++;
    }
    CHECK(bad == 0, "l2 golden (k): %ld/%zu mismatches", bad, H * D);
    digest_bytes(&g_digest, y, H * D * sizeof(float));
#ifdef __ARM_NEON
    float *ys = malloc(H * D * sizeof(float));
    float *yn = malloc(H * D * sizeof(float));
    apus_gdn_l2norm_scalar(x, ys, H, D, qscale);
    apus_gdn_l2norm_neon(x, yn, H, D, qscale);
    CHECK(memcmp(ys, yn, H * D * sizeof(float)) == 0,
          "l2 scalar != NEON (not bitwise)");
    free(ys); free(yn);
#endif
    free(x); free(gq); free(gk); free(y);
}

static void test_l2_edges(void) {
    size_t H = 2, D = 64;
    uint16_t *x = malloc(H * D * 2);
    float *y = malloc(H * D * sizeof(float));
    double *tr = malloc(H * D * sizeof(double));
    /* zero vector -> 0 * rsqrt(1e-6) = +0 */
    memset(x, 0, H * D * 2);
    apus_gdn_l2norm(x, y, H, D, 1.0f);
    long bad = 0;
    for (size_t i = 0; i < H * D; i++) if (f32bits(y[i]) != 0) bad++;
    CHECK(bad == 0, "l2 zero vector: %ld nonzero", bad);
    /* random vs in-test f64 truth (odd D exercises the NEON tail) */
    size_t D2 = 65;
    uint16_t *x2 = malloc(H * D2 * 2);
    float *y2 = malloc(H * D2 * sizeof(float));
    double *tr2 = malloc(H * D2 * sizeof(double));
    for (size_t i = 0; i < H * D2; i++) x2[i] = rng_bf16_scaled(1.0f);
    l2_truth_f64(x2, tr2, H, D2, 1.0);
    apus_gdn_l2norm(x2, y2, H, D2, 1.0f);
    bad = 0;
    for (size_t i = 0; i < H * D2; i++) {
        meas(&g_meas_l2, y2[i], tr2[i], 0.3);
        if (!tol_ok(y2[i], tr2[i], 0.3, 2e-6, 2e-6)) bad++;
    }
    CHECK(bad == 0, "l2 in-test truth: %ld/%zu mismatches", bad, H * D2);
    (void)tr;
    free(x); free(y); free(tr); free(x2); free(y2); free(tr2);
}

/* =========================================================================*/
/* in-test f64 truth for the recurrence (same op order, double type);
 * tracks honest error scales: escS propagated as esc*dec + |k*delta|
 * per step, esco = sum_i |q_i * S_ij| (fp32 rounding scales with
 * absolute terms, not |out| — cancellation can drive |out| to ~0) */
static void rec_truth_f64(const float *q, const float *k,
                          const uint16_t *v, const float *g,
                          const uint16_t *beta, double *S, double *o,
                          double *escS, double *esco,
                          size_t T, size_t H, size_t D) {
    for (size_t t = 0; t < T; t++) {
        for (size_t h = 0; h < H; h++) {
            double *Sh = S + h * D * D;
            double *eh = escS + h * D * D;
            double dec = exp((double)g[t * H + h]);
            for (size_t i = 0; i < D * D; i++) {
                Sh[i] *= dec;
                eh[i] *= dec;
            }
            double b = (double)apus_bf16_f32(beta[t * H + h]);
            double delta[256];
            for (size_t j = 0; j < D; j++) {
                double acc = 0.0;
                for (size_t i = 0; i < D; i++)
                    acc += (double)k[(t * H + h) * D + i] * Sh[i * D + j];
                delta[j] = ((double)apus_bf16_f32(v[(t * H + h) * D + j])
                            - acc) * b;
            }
            for (size_t i = 0; i < D; i++) {
                double ki = (double)k[(t * H + h) * D + i];
                for (size_t j = 0; j < D; j++) {
                    Sh[i * D + j] += ki * delta[j];
                    eh[i * D + j] += fabs(ki * delta[j]);
                }
            }
            for (size_t j = 0; j < D; j++) {
                double acc = 0.0, sc = 0.0;
                for (size_t i = 0; i < D; i++) {
                    double p = (double)q[(t * H + h) * D + i] * Sh[i * D + j];
                    acc += p;
                    sc += fabs(p);
                }
                o[(t * H + h) * D + j] = acc;
                esco[(t * H + h) * D + j] = sc;
            }
        }
    }
}

static void test_rec_golden(void) {
    size_t H = (size_t)manifest_val("REC_H");
    size_t D = (size_t)manifest_val("REC_D");
    size_t T = (size_t)manifest_val("REC_T");
    size_t n;
    float *q = (float *)read_file("tests/m4a/golden/gdn_rec_q.bin", &n);
    float *k = (float *)read_file("tests/m4a/golden/gdn_rec_k.bin", &n);
    uint16_t *v = (uint16_t *)read_file("tests/m4a/golden/gdn_rec_v.bin", &n);
    float *g = (float *)read_file("tests/m4a/golden/gdn_rec_g.bin", &n);
    uint16_t *bt = (uint16_t *)read_file("tests/m4a/golden/gdn_rec_beta.bin",
                                         &n);
    double *go = (double *)read_file("tests/m4a/golden/gdn_rec_o.bin", &n);
    double *geo = (double *)read_file("tests/m4a/golden/gdn_rec_esco.bin", &n);
    double *gS = (double *)read_file("tests/m4a/golden/gdn_rec_S.bin", &n);
    double *geS = (double *)read_file("tests/m4a/golden/gdn_rec_escS.bin", &n);
    CHECK(q && k && v && g && bt && go && geo && gS && geS,
          "rec fixture load");
    if (!q || !k || !v || !g || !bt || !go || !geo || !gS || !geS) return;

    float *S = calloc(H * D * D, sizeof(float));
    float *o = malloc(T * H * D * sizeof(float));
    apus_gdn_recurrent(S, q, k, v, g, bt, o, T, H, D, D);
    long bad = 0;
    for (size_t i = 0; i < T * H * D; i++) {
        meas(&g_meas_reco, o[i], go[i], geo[i]);
        if (!tol_ok(o[i], go[i], geo[i], 1e-5, 1e-5)) bad++;
    }
    CHECK(bad == 0, "rec golden o: %ld/%zu mismatches", bad, T * H * D);
    bad = 0;
    for (size_t i = 0; i < H * D * D; i++) {
        meas(&g_meas_recS, S[i], gS[i], geS[i]);
        if (!tol_ok(S[i], gS[i], geS[i], 1e-5, 1e-5)) bad++;
    }
    CHECK(bad == 0, "rec golden S: %ld/%zu mismatches", bad, H * D * D);
    digest_bytes(&g_digest, o, T * H * D * sizeof(float));
    digest_bytes(&g_digest, S, H * D * D * sizeof(float));

    /* BITWISE: stepping apus_gdn_step == apus_gdn_recurrent */
    float *S2 = calloc(H * D * D, sizeof(float));
    float *o2 = malloc(T * H * D * sizeof(float));
    for (size_t t = 0; t < T; t++)
        apus_gdn_step(S2, q + t * H * D, k + t * H * D, v + t * H * D,
                      g + t * H, bt + t * H, o2 + t * H * D, H, D, D);
    CHECK(memcmp(S, S2, H * D * D * sizeof(float)) == 0 &&
          memcmp(o, o2, T * H * D * sizeof(float)) == 0,
          "rec prefill != step loop (not bitwise)");

    /* BITWISE: mt == sequential (thread-count independence is the
     * Makefile's stdout diff across APUS_THREADS) */
    float *S3 = calloc(H * D * D, sizeof(float));
    float *o3 = malloc(T * H * D * sizeof(float));
    for (size_t t = 0; t < T; t++)
        apus_gdn_step_mt(S3, q + t * H * D, k + t * H * D, v + t * H * D,
                         g + t * H, bt + t * H, o3 + t * H * D, H, D, D);
    CHECK(memcmp(S, S3, H * D * D * sizeof(float)) == 0 &&
          memcmp(o, o3, T * H * D * sizeof(float)) == 0,
          "rec step_mt != step (not bitwise)");
#ifdef __ARM_NEON
    /* BITWISE: scalar anchor == NEON head body */
    float *S4 = calloc(H * D * D, sizeof(float));
    float *o4 = malloc(T * H * D * sizeof(float));
    for (size_t t = 0; t < T; t++)
        for (size_t h = 0; h < H; h++)
            apus_gdn_step_head_scalar(S4 + h * D * D,
                                      q + (t * H + h) * D,
                                      k + (t * H + h) * D,
                                      v + (t * H + h) * D,
                                      g[t * H + h],
                                      apus_bf16_f32(bt[t * H + h]),
                                      o4 + (t * H + h) * D, D, D);
    float *S5 = calloc(H * D * D, sizeof(float));
    float *o5 = malloc(T * H * D * sizeof(float));
    for (size_t t = 0; t < T; t++)
        for (size_t h = 0; h < H; h++)
            apus_gdn_step_head_neon(S5 + h * D * D,
                                    q + (t * H + h) * D,
                                    k + (t * H + h) * D,
                                    v + (t * H + h) * D,
                                    g[t * H + h],
                                    apus_bf16_f32(bt[t * H + h]),
                                    o5 + (t * H + h) * D, D, D);
    CHECK(memcmp(S4, S5, H * D * D * sizeof(float)) == 0 &&
          memcmp(o4, o5, T * H * D * sizeof(float)) == 0,
          "rec scalar != NEON (not bitwise)");
    CHECK(memcmp(S, S4, H * D * D * sizeof(float)) == 0,
          "rec dispatched != scalar anchor (not bitwise)");
    free(S4); free(o4); free(S5); free(o5);
#endif
    free(q); free(k); free(v); free(g); free(bt);
    free(go); free(geo); free(gS); free(geS);
    free(S); free(o); free(S2); free(o2); free(S3); free(o3);
}

static void test_rec_intest(void) {
    size_t H = 2, D = 32, T = 10;
    float *q = malloc(T * H * D * sizeof(float));
    float *k = malloc(T * H * D * sizeof(float));
    uint16_t *v = malloc(T * H * D * 2);
    float *g = malloc(T * H * sizeof(float));
    uint16_t *bt = malloc(T * H * 2);
    for (size_t i = 0; i < T * H * D; i++) {
        q[i] = rng_float() * 0.25f;
        k[i] = rng_float() * 0.25f;
        v[i] = rng_bf16_scaled(1.5f);
    }
    for (size_t i = 0; i < T * H; i++) {
        g[i] = -5.0f * (float)((rng_u64() >> 40) / 16777216.0);
        bt[i] = apus_bf16_bits(
            (float)((rng_u64() >> 40) / 16777216.0));
    }
    float *S = calloc(H * D * D, sizeof(float));
    float *o = malloc(T * H * D * sizeof(float));
    apus_gdn_recurrent(S, q, k, v, g, bt, o, T, H, D, D);
    double *St = calloc(H * D * D, sizeof(double));
    double *ot = malloc(T * H * D * sizeof(double));
    double *etS = calloc(H * D * D, sizeof(double));
    double *eto = malloc(T * H * D * sizeof(double));
    rec_truth_f64(q, k, v, g, bt, St, ot, etS, eto, T, H, D);
    long bad = 0;
    for (size_t i = 0; i < T * H * D; i++) {
        meas(&g_meas_reco, o[i], ot[i], eto[i]);
        if (!tol_ok(o[i], ot[i], eto[i], 1e-5, 1e-5)) bad++;
    }
    CHECK(bad == 0, "rec in-test o: %ld/%zu mismatches", bad, T * H * D);
    bad = 0;
    for (size_t i = 0; i < H * D * D; i++) {
        meas(&g_meas_recS, S[i], St[i], etS[i]);
        if (!tol_ok(S[i], St[i], etS[i], 1e-5, 1e-5)) bad++;
    }
    CHECK(bad == 0, "rec in-test S: %ld/%zu mismatches", bad, H * D * D);

    /* decay extremes: g == 0 + beta == 0 -> S unchanged; g == -5 with
     * zero k -> exact decay of the pre-existing state */
    float *Sz = calloc(D * D, sizeof(float));
    for (size_t i = 0; i < D * D; i++) Sz[i] = (float)(i % 7) * 0.25f - 0.75f;
    float *Sref = malloc(D * D * sizeof(float));
    memcpy(Sref, Sz, D * D * sizeof(float));
    float *q1 = malloc(D * sizeof(float));
    float *k1 = malloc(D * sizeof(float));
    uint16_t *v1 = malloc(D * 2);
    float g0 = 0.0f;
    uint16_t b0 = apus_bf16_bits(0.0f);
    for (size_t i = 0; i < D; i++) {
        q1[i] = rng_float() * 0.25f;
        k1[i] = rng_float() * 0.25f;
        v1[i] = rng_bf16_scaled(1.0f);
    }
    float *o1 = malloc(D * sizeof(float));
    apus_gdn_step(Sz, q1, k1, v1, &g0, &b0, o1, 1, D, D);
    CHECK(memcmp(Sz, Sref, D * D * sizeof(float)) == 0,
          "rec g=0,beta=0: state changed");
    float gm5 = -5.0f;
    uint16_t b05 = apus_bf16_bits(0.5f);
    float dec = expf(-5.0f);
    /* a zero-k token decays the state exactly (kv = 0, delta = v*beta,
     * but the rank-1 update is k x delta = 0) */
    float *Sz2 = malloc(D * D * sizeof(float));
    memcpy(Sz2, Sref, D * D * sizeof(float));
    float *kz = calloc(D, sizeof(float));
    uint16_t *vz = calloc(D, 2);
    apus_gdn_step(Sz2, q1, kz, vz, &gm5, &b05, o1, 1, D, D);
    bad = 0;
    for (size_t i = 0; i < D * D; i++)
        if (f32bits(Sz2[i]) != f32bits(Sref[i] * dec)) bad++;
    CHECK(bad == 0, "rec g=-5 zero-k decay: %ld mismatches", bad);

    free(q); free(k); free(v); free(g); free(bt);
    free(S); free(o); free(St); free(ot); free(etS); free(eto);
    free(Sz); free(Sref); free(q1); free(k1); free(v1);
    free(o1); free(Sz2); free(kz); free(vz);
}

/* =========================================================================*/
static void onorm_truth_f64(const uint16_t *o, const uint16_t *z,
                            const uint16_t *w, double *y,
                            size_t H, size_t D) {
    for (size_t h = 0; h < H; h++) {
        double ss = 0.0;
        for (size_t d = 0; d < D; d++) {
            double v = apus_bf16_f32(o[h * D + d]);
            ss += v * v;
        }
        double rs = 1.0 / sqrt(ss / (double)D + 1e-6);
        for (size_t d = 0; d < D; d++) {
            double x1 = (double)apus_bf16_f32(apus_bf16_bits(
                (float)((double)apus_bf16_f32(o[h * D + d]) * rs)));
            double x2 = (double)apus_bf16_f32(apus_bf16_bits(
                (float)((double)apus_bf16_f32(w[d]) * x1)));
            double zv = apus_bf16_f32(z[h * D + d]);
            y[h * D + d] = x2 * (zv * (1.0 / (1.0 + exp(-zv))));
        }
    }
}

static void test_onorm_golden(void) {
    size_t H = (size_t)manifest_val("ONORM_H");
    size_t D = (size_t)manifest_val("ONORM_D");
    size_t n;
    uint16_t *o = (uint16_t *)read_file("tests/m4a/golden/gdn_onorm_o.bin",
                                        &n);
    uint16_t *z = (uint16_t *)read_file("tests/m4a/golden/gdn_onorm_z.bin",
                                        &n);
    uint16_t *w = (uint16_t *)read_file("tests/m4a/golden/gdn_onorm_w.bin",
                                        &n);
    double *gy = (double *)read_file("tests/m4a/golden/gdn_onorm_y.bin", &n);
    CHECK(o && z && w && gy, "onorm fixture load");
    if (!o || !z || !w || !gy) return;
    uint16_t *y = malloc(H * D * 2);
    apus_gdn_onorm(o, z, w, y, H, D);
    long bad = 0;
    for (size_t i = 0; i < H * D; i++) {
        double c = apus_bf16_f32(y[i]);
        meas(&g_meas_onorm, c, gy[i], 0.0);
        if (!tol_ok(c, gy[i], 0.02, 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "onorm golden: %ld/%zu mismatches", bad, H * D);
    digest_bytes(&g_digest, y, H * D * 2);
#ifdef __ARM_NEON
    uint16_t *ys = malloc(H * D * 2);
    uint16_t *yn = malloc(H * D * 2);
    apus_gdn_onorm_scalar(o, z, w, ys, H, D);
    apus_gdn_onorm_neon(o, z, w, yn, H, D);
    CHECK(memcmp(ys, yn, H * D * 2) == 0,
          "onorm scalar != NEON (not bitwise)");
    free(ys); free(yn);
#endif
    free(o); free(z); free(w); free(gy); free(y);
}

static void test_onorm_intest(void) {
    size_t H = 3, D = 96;
    uint16_t *o = malloc(H * D * 2);
    uint16_t *z = malloc(H * D * 2);
    uint16_t *w = malloc(D * 2);
    double *tr = malloc(H * D * sizeof(double));
    uint16_t *y = malloc(H * D * 2);
    for (size_t i = 0; i < H * D; i++) {
        o[i] = rng_bf16_scaled(1.5f);
        z[i] = rng_bf16_scaled(2.0f);
    }
    for (size_t i = 0; i < D; i++) w[i] = rng_bf16_scaled(0.8f);
    onorm_truth_f64(o, z, w, tr, H, D);
    apus_gdn_onorm(o, z, w, y, H, D);
    long bad = 0;
    for (size_t i = 0; i < H * D; i++) {
        double c = apus_bf16_f32(y[i]);
        meas(&g_meas_onorm, c, tr[i], 0.0);
        if (!tol_ok(c, tr[i], 0.02, 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "onorm in-test truth: %ld/%zu mismatches", bad, H * D);
    /* z gate extremes: z = +88 -> sigmoid saturates to 1.0 in fp32, so
     * silu(z) = z and with o=w=1 the output is bf16(88); z = -88 ->
     * silu(z) ~ -88 * 6e-39 -> ~0 */
    for (size_t i = 0; i < H * D; i++) {
        z[i] = apus_bf16_bits(88.0f);
        o[i] = apus_bf16_bits(1.0f);
    }
    for (size_t i = 0; i < D; i++) w[i] = apus_bf16_bits(1.0f);
    apus_gdn_onorm(o, z, w, y, H, D);
    CHECK(y[0] == apus_bf16_bits(88.0f),
          "onorm z=+88: got %.6g want 88", apus_bf16_f32(y[0]));
    for (size_t i = 0; i < H * D; i++) z[i] = apus_bf16_bits(-88.0f);
    apus_gdn_onorm(o, z, w, y, H, D);
    long zbad = 0;
    for (size_t i = 0; i < H * D; i++)
        if (fabsf(apus_bf16_f32(y[i])) > 1e-30f) zbad++;
    CHECK(zbad == 0, "onorm z=-88: %ld above 1e-30", zbad);
    free(o); free(z); free(w); free(tr); free(y);
}

int main(void) {
    printf("test_gdn: GDN kernel hard-gate tests (Qwen3.6-35B-A3B M4a)\n");
    fprintf(stderr, "  pool threads: %d\n", apus_pool_threads());

    if (manifest_val("CONV_C") < 0) {
        failures++;
        fprintf(stderr, "FAIL: could not load golden manifest\n");
        return 1;
    }

    test_conv_golden();
    test_conv_intest();
    test_decay_golden();
    test_decay_intest();
    test_beta();
    test_l2_golden();
    test_l2_edges();
    test_rec_golden();
    test_rec_intest();
    test_onorm_golden();
    test_onorm_intest();

    printf("  measured max err/(|gold|+esc): conv %.3g decay %.3g beta %.3g "
           "l2 %.3g rec_o %.3g rec_S %.3g onorm %.3g\n",
           g_meas_conv, g_meas_decay, g_meas_beta, g_meas_l2,
           g_meas_reco, g_meas_recS, g_meas_onorm);
    printf("  digest %016llx\n", (unsigned long long)g_digest);
    printf("test_gdn: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
