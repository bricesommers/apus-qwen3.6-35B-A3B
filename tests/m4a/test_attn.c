/*
 * tests/m4a/test_attn.c — hard-gate tests for c/attn.h (gated GQA
 * full-attention kernels, Qwen3.6-35B-A3B M4).
 *
 *   1. RMSNorm, the (1+w) zero-init variant, SINGLE rounding: golden
 *      f64 + in-test f64 at the real N=2048; zero vector.
 *   2. Partial NeoX RoPE (rot=16 of D=64, theta=1e7): golden f64 at
 *      positions 0/1/7/1000/131071/262143 (the golden replicates the
 *      fp32 inv_freq/angle before f64 cos/sin — libm ulps are the only
 *      remaining class), cos/sin bf16 codes, position 0 == identity,
 *      pass-through dims untouched.
 *   3. GQA eager attention (H=4 q heads, Hkv=2 KV heads, D=64): golden
 *      f64 with the per-step roundings replicated (A rnd-scale-rnd,
 *      P rnd), in-test f64 truth, BITWISE decode == last row of full
 *      recompute, mt == sequential.
 *   4. Elementwise sigmoid output gate: golden f64 + exact corners
 *      (logit 0 -> gate code 0x3F00; +20 -> 1.0; -20 -> ~0).
 *   5. BITWISE scalar anchor == NEON for every op (on __ARM_NEON).
 *
 * fp32-valued compares at 1e-5 of esc; bf16-out at ~2 bf16 ulp + esc.
 * Run from the repository root (fixtures under tests/m4a/golden/).
 */
#define APUS_BF16_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION
#include "bf16.h"
#include "attn.h"

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
static uint64_t rng_state = 0xC2B2AE3D27D4EB4Full;
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

static double g_meas_rms = 0, g_meas_rope = 0, g_meas_gqa = 0;
static double g_meas_og = 0;

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
/* (1+w) RMSNorm, single rounding */
static void rms_truth_f64(const uint16_t *x, const uint16_t *w, double *y,
                          size_t N) {
    double ss = 0.0;
    for (size_t i = 0; i < N; i++) {
        double v = apus_bf16_f32(x[i]);
        ss += v * v;
    }
    double rs = 1.0 / sqrt(ss / (double)N + 1e-6);
    for (size_t i = 0; i < N; i++) {
        double xn = (double)apus_bf16_f32(x[i]) * rs;
        y[i] = (double)apus_bf16_f32(apus_bf16_bits(
            (float)(xn * (1.0 + (double)apus_bf16_f32(w[i])))));
    }
}

static void test_rmsnorm(void) {
    size_t N = (size_t)manifest_val("RMS_N");
    size_t n;
    uint16_t *x = (uint16_t *)read_file("tests/m4a/golden/attn_rms_x.bin",
                                        &n);
    uint16_t *w = (uint16_t *)read_file("tests/m4a/golden/attn_rms_w.bin",
                                        &n);
    double *gy = (double *)read_file("tests/m4a/golden/attn_rms_y.bin", &n);
    CHECK(x && w && gy, "rmsnorm fixture load");
    if (!x || !w || !gy) return;
    uint16_t *y = malloc(N * 2);
    apus_attn_rmsnorm(x, w, y, N);
    long bad = 0;
    for (size_t i = 0; i < N; i++) {
        double c = apus_bf16_f32(y[i]);
        meas(&g_meas_rms, c, gy[i], 0.02);
        if (!tol_ok(c, gy[i], 0.02, 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "rmsnorm golden: %ld/%zu mismatches", bad, N);
    digest_bytes(&g_digest, y, N * 2);
    /* in-test f64 at the real H=2048 */
    size_t N2 = 2048;
    uint16_t *x2 = malloc(N2 * 2);
    uint16_t *w2 = malloc(N2 * 2);
    double *tr = malloc(N2 * sizeof(double));
    uint16_t *y2 = malloc(N2 * 2);
    for (size_t i = 0; i < N2; i++) {
        x2[i] = rng_bf16_scaled(2.0f);
        w2[i] = rng_bf16_scaled(0.1f);
    }
    rms_truth_f64(x2, w2, tr, N2);
    apus_attn_rmsnorm(x2, w2, y2, N2);
    bad = 0;
    for (size_t i = 0; i < N2; i++) {
        double c = apus_bf16_f32(y2[i]);
        meas(&g_meas_rms, c, tr[i], 0.02);
        if (!tol_ok(c, tr[i], 0.02, 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "rmsnorm in-test truth: %ld/%zu mismatches", bad, N2);
    /* zero vector -> +0 outputs */
    memset(x2, 0, N2 * 2);
    apus_attn_rmsnorm(x2, w2, y2, N2);
    bad = 0;
    for (size_t i = 0; i < N2; i++) if (y2[i] != 0x0000) bad++;
    CHECK(bad == 0, "rmsnorm zero vector: %ld nonzero", bad);
#ifdef __ARM_NEON
    /* odd N exercises the NEON tail */
    size_t N3 = 1023;
    uint16_t *x3 = malloc(N3 * 2);
    uint16_t *w3 = malloc(N3 * 2);
    uint16_t *ys = malloc(N3 * 2);
    uint16_t *yn = malloc(N3 * 2);
    for (size_t i = 0; i < N3; i++) {
        x3[i] = rng_bf16_scaled(2.0f);
        w3[i] = rng_bf16_scaled(0.1f);
    }
    apus_attn_rmsnorm_scalar(x3, w3, ys, N3);
    apus_attn_rmsnorm_neon(x3, w3, yn, N3);
    CHECK(memcmp(ys, yn, N3 * 2) == 0,
          "rmsnorm scalar != NEON (not bitwise)");
    free(x3); free(w3); free(ys); free(yn);
#endif
    free(x); free(w); free(gy); free(y);
    free(x2); free(w2); free(tr); free(y2);
}

/* =========================================================================*/
static void test_rope(void) {
    size_t P = (size_t)manifest_val("ROPE_P");
    size_t D = (size_t)manifest_val("ROPE_D");
    size_t ROT = (size_t)manifest_val("ROPE_ROT");
    size_t n;
    uint16_t *x = (uint16_t *)read_file("tests/m4a/golden/attn_rope_x.bin",
                                        &n);
    double *pos = (double *)read_file("tests/m4a/golden/attn_rope_pos.bin",
                                      &n);
    uint16_t *cs = (uint16_t *)read_file("tests/m4a/golden/attn_rope_cs.bin",
                                         &n);
    double *gy = (double *)read_file("tests/m4a/golden/attn_rope_y.bin", &n);
    CHECK(x && pos && cs && gy, "rope fixture load");
    if (!x || !pos || !cs || !gy) return;
    size_t R = ROT / 2;
    long flips = 0;
    for (size_t p = 0; p < P; p++) {
        uint16_t *y = malloc(D * 2);
        apus_attn_rope(x, y, D, ROT, (float)pos[p], 10000000.0);
        long bad = 0;
        for (size_t i = 0; i < D; i++) {
            double c = apus_bf16_f32(y[i]);
            meas(&g_meas_rope, c, gy[p * D + i], 0.02);
            if (!tol_ok(c, gy[p * D + i], 0.02, 0.008, 0.008)) bad++;
        }
        CHECK(bad == 0, "rope golden pos %.9g: %ld/%zu mismatches",
              pos[p], bad, D);
        /* cos/sin codes must match the golden's (libm class is sub-tie
         * on this libm — flips reported) */
        for (size_t i = 0; i < R; i++) {
            uint16_t cb = apus_bf16_bits(
                cosf((float)pos[p] *
                     (float)(1.0 / pow(10000000.0,
                                       2.0 * (double)i / (double)ROT))));
            uint16_t sb = apus_bf16_bits(
                sinf((float)pos[p] *
                     (float)(1.0 / pow(10000000.0,
                                       2.0 * (double)i / (double)ROT))));
            if (cb != cs[(p * 2 + 0) * R + i]) flips++;
            if (sb != cs[(p * 2 + 1) * R + i]) flips++;
        }
        /* pass-through dims are exact copies */
        long pbad = 0;
        for (size_t i = ROT; i < D; i++)
            if (y[i] != x[i]) pbad++;
        CHECK(pbad == 0, "rope pass-through pos %.9g: %ld diffs",
              pos[p], pbad);
        digest_bytes(&g_digest, y, D * 2);
        free(y);
    }
    printf("  rope cos/sin code flips vs golden: %ld/%zu\n",
           flips, P * 2 * R);
    CHECK(flips * 100 <= (long)(P * 2 * R),
          "rope cos/sin codes: %ld flips (>1%%)", flips);
    /* position 0 -> y == x bitwise (cos(0)=1, sin(0)=0 exactly) */
    uint16_t *y0 = malloc(D * 2);
    apus_attn_rope(x, y0, D, ROT, 0.0f, 10000000.0);
    CHECK(memcmp(y0, x, D * 2) == 0, "rope pos 0 != identity");
    free(y0);
#ifdef __ARM_NEON
    uint16_t *ys = malloc(D * 2);
    uint16_t *yn = malloc(D * 2);
    for (size_t p = 0; p < P; p++) {
        apus_attn_rope_scalar(x, ys, D, ROT, (float)pos[p], 10000000.0);
        apus_attn_rope_neon(x, yn, D, ROT, (float)pos[p], 10000000.0);
        CHECK(memcmp(ys, yn, D * 2) == 0,
              "rope scalar != NEON at pos %.9g (not bitwise)", pos[p]);
    }
    free(ys); free(yn);
#endif
    free(x); free(pos); free(cs); free(gy);
}

/* =========================================================================*/
/* in-test f64 truth for one GQA row (same op order, double type) */
static void gqa_row_truth_f64(const uint16_t *qh, const uint16_t *kc,
                              const uint16_t *vc, double *oh, double *esco,
                              size_t Hkv, size_t D, double scale,
                              size_t p) {
    double A[4096];
    for (size_t j = 0; j <= p; j++) {
        double acc = 0.0;
        for (size_t d = 0; d < D; d++)
            acc += (double)apus_bf16_f32(qh[d]) *
                   (double)apus_bf16_f32(kc[(j * Hkv) * D + d]);
        A[j] = (double)apus_bf16_f32(apus_bf16_bits(
            (float)((double)apus_bf16_f32(apus_bf16_bits((float)acc)) *
                    scale)));
    }
    double m = A[0];
    for (size_t j = 1; j <= p; j++)
        if (A[j] > m) m = A[j];
    double s = 0.0, e[4096];
    for (size_t j = 0; j <= p; j++) {
        e[j] = exp(A[j] - m);
        s += e[j];
    }
    double P[4096];
    for (size_t j = 0; j <= p; j++)
        P[j] = (double)apus_bf16_f32(apus_bf16_bits((float)(e[j] / s)));
    for (size_t d = 0; d < D; d++) {
        double acc = 0.0, sc = 0.0;
        for (size_t j = 0; j <= p; j++) {
            double pr = P[j] *
                        (double)apus_bf16_f32(vc[(j * Hkv) * D + d]);
            acc += pr;
            sc += fabs(pr);
        }
        oh[d] = acc;
        esco[d] = sc;
    }
}

static void test_gqa(void) {
    size_t H = (size_t)manifest_val("GQA_H");
    size_t HKV = (size_t)manifest_val("GQA_HKV");
    size_t T = (size_t)manifest_val("GQA_T");
    size_t D = (size_t)manifest_val("GQA_D");
    size_t n;
    uint16_t *q = (uint16_t *)read_file("tests/m4a/golden/attn_gqa_q.bin",
                                        &n);
    uint16_t *kc = (uint16_t *)read_file("tests/m4a/golden/attn_gqa_k.bin",
                                         &n);
    uint16_t *vc = (uint16_t *)read_file("tests/m4a/golden/attn_gqa_v.bin",
                                         &n);
    double *go = (double *)read_file("tests/m4a/golden/attn_gqa_o.bin", &n);
    double *geo = (double *)read_file("tests/m4a/golden/attn_gqa_esco.bin",
                                      &n);
    CHECK(q && kc && vc && go && geo, "gqa fixture load");
    if (!q || !kc || !vc || !go || !geo) return;
    float scale = (float)(1.0 / sqrt((double)D));
    uint16_t *abuf = malloc(T * 2);
    float *ebuf = malloc(T * sizeof(float));
    uint16_t *o = malloc(T * H * D * 2);
    apus_attn_gqa(q, kc, vc, o, T, T, H, HKV, D, scale, abuf, ebuf);
    long bad = 0;
    for (size_t i = 0; i < T * H * D; i++) {
        double c = apus_bf16_f32(o[i]);
        meas(&g_meas_gqa, c, go[i], geo[i]);
        if (!tol_ok(c, go[i], geo[i], 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "gqa golden: %ld/%zu mismatches", bad, T * H * D);
    digest_bytes(&g_digest, o, T * H * D * 2);

    /* BITWISE: decode == last row of full recompute (per head) */
    uint16_t *od = malloc(H * D * 2);
    apus_attn_gqa_decode(q + (T - 1) * H * D, kc, vc, od, T, H, HKV, D,
                         scale, abuf, ebuf);
    CHECK(memcmp(od, o + (T - 1) * H * D, H * D * 2) == 0,
          "gqa decode != full recompute last row (not bitwise)");
    /* BITWISE: mt == sequential */
    uint16_t *om = malloc(T * H * D * 2);
    apus_attn_gqa_mt(q, kc, vc, om, T, T, H, HKV, D, scale);
    CHECK(memcmp(om, o, T * H * D * 2) == 0,
          "gqa mt != sequential (not bitwise)");
#ifdef __ARM_NEON
    /* BITWISE: scalar anchor == NEON row body */
    uint16_t *os = malloc(T * H * D * 2);
    uint16_t *on = malloc(T * H * D * 2);
    size_t nrep = H / HKV;
    for (size_t t = 0; t < T; t++)
        for (size_t h = 0; h < H; h++) {
            apus_attn_gqa_row_scalar(q + (t * H + h) * D,
                                     kc + (h / nrep) * D,
                                     vc + (h / nrep) * D,
                                     os + (t * H + h) * D,
                                     HKV, D, scale, abuf, ebuf, t);
            apus_attn_gqa_row_neon(q + (t * H + h) * D,
                                   kc + (h / nrep) * D,
                                   vc + (h / nrep) * D,
                                   on + (t * H + h) * D,
                                   HKV, D, scale, abuf, ebuf, t);
        }
    CHECK(memcmp(os, on, T * H * D * 2) == 0,
          "gqa scalar != NEON (not bitwise)");
    CHECK(memcmp(os, o, T * H * D * 2) == 0,
          "gqa dispatched != scalar anchor (not bitwise)");
    free(os); free(on);
#endif
    /* in-test f64 truth (odd D exercises the NEON tails) */
    size_t D2 = 65, T2 = 7, H2 = 3, HK2 = 1;
    uint16_t *q2 = malloc(T2 * H2 * D2 * 2);
    uint16_t *k2 = malloc(T2 * HK2 * D2 * 2);
    uint16_t *v2 = malloc(T2 * HK2 * D2 * 2);
    uint16_t *o2 = malloc(T2 * H2 * D2 * 2);
    for (size_t i = 0; i < T2 * H2 * D2; i++) q2[i] = rng_bf16_scaled(1.0f);
    for (size_t i = 0; i < T2 * HK2 * D2; i++) {
        k2[i] = rng_bf16_scaled(1.0f);
        v2[i] = rng_bf16_scaled(1.2f);
    }
    float scale2 = (float)(1.0 / sqrt((double)D2));
    uint16_t *ab2 = malloc(T2 * 2);
    float *eb2 = malloc(T2 * sizeof(float));
    apus_attn_gqa(q2, k2, v2, o2, T2, T2, H2, HK2, D2, scale2, ab2, eb2);
    bad = 0;
    for (size_t t = 0; t < T2; t++)
        for (size_t h = 0; h < H2; h++) {
            double ot[128], oe[128];
            gqa_row_truth_f64(q2 + (t * H2 + h) * D2, k2, v2, ot, oe,
                              HK2, D2, (double)scale2, t);
            for (size_t d = 0; d < D2; d++) {
                double c = apus_bf16_f32(o2[(t * H2 + h) * D2 + d]);
                meas(&g_meas_gqa, c, ot[d], oe[d]);
                if (!tol_ok(c, ot[d], oe[d], 0.008, 0.008)) bad++;
            }
        }
    CHECK(bad == 0, "gqa in-test truth: %ld/%zu mismatches",
          bad, T2 * H2 * D2);
    free(q); free(kc); free(vc); free(go); free(geo);
    free(abuf); free(ebuf); free(o); free(od); free(om);
    free(q2); free(k2); free(v2); free(o2); free(ab2); free(eb2);
}

/* =========================================================================*/
static void test_outgate(void) {
    size_t N = (size_t)manifest_val("OG_N");
    size_t n;
    uint16_t *o = (uint16_t *)read_file("tests/m4a/golden/attn_og_o.bin",
                                        &n);
    uint16_t *gl = (uint16_t *)read_file("tests/m4a/golden/attn_og_gl.bin",
                                         &n);
    double *gy = (double *)read_file("tests/m4a/golden/attn_og_y.bin", &n);
    CHECK(o && gl && gy, "outgate fixture load");
    if (!o || !gl || !gy) return;
    uint16_t *y = malloc(N * 2);
    apus_attn_outgate(o, gl, y, N);
    long bad = 0;
    for (size_t i = 0; i < N; i++) {
        double c = apus_bf16_f32(y[i]);
        meas(&g_meas_og, c, gy[i], 0.02);
        if (!tol_ok(c, gy[i], 0.02, 0.008, 0.008)) bad++;
    }
    CHECK(bad == 0, "outgate golden: %ld/%zu mismatches", bad, N);
    /* exact corners: logit 0 -> gate code 0x3F00, o*0.5 exact halving;
     * +20 -> gate rounds to exactly 1.0 -> y == o code; -20 -> ~0 */
    CHECK(y[0] == apus_bf16_bits(apus_bf16_f32(o[0]) * 0.5f),
          "outgate logit=0 corner");
    CHECK(y[1] == o[1], "outgate logit=+20 corner: y != o");
    CHECK(apus_bf16_f32(y[2]) <
          1e-6f * (fabsf(apus_bf16_f32(o[2])) + 1e-6f),
          "outgate logit=-20 corner: got %.6g", apus_bf16_f32(y[2]));
    digest_bytes(&g_digest, y, N * 2);
#ifdef __ARM_NEON
    size_t N3 = 1023;                 /* odd: exercises the NEON tail */
    uint16_t *o3 = malloc(N3 * 2);
    uint16_t *g3 = malloc(N3 * 2);
    uint16_t *ys = malloc(N3 * 2);
    uint16_t *yn = malloc(N3 * 2);
    for (size_t i = 0; i < N3; i++) {
        o3[i] = rng_bf16_scaled(1.5f);
        g3[i] = rng_bf16_scaled(2.0f);
    }
    apus_attn_outgate_scalar(o3, g3, ys, N3);
    apus_attn_outgate_neon(o3, g3, yn, N3);
    CHECK(memcmp(ys, yn, N3 * 2) == 0,
          "outgate scalar != NEON (not bitwise)");
    free(o3); free(g3); free(ys); free(yn);
#endif
    free(o); free(gl); free(gy); free(y);
}

int main(void) {
    printf("test_attn: gated-GQA kernel hard-gate tests "
           "(Qwen3.6-35B-A3B M4a)\n");
    fprintf(stderr, "  pool threads: %d\n", apus_pool_threads());

    if (manifest_val("RMS_N") < 0) {
        failures++;
        fprintf(stderr, "FAIL: could not load golden manifest\n");
        return 1;
    }

    test_rmsnorm();
    test_rope();
    test_gqa();
    test_outgate();

    printf("  measured max err/(|gold|+esc): rms %.3g rope %.3g gqa %.3g "
           "outgate %.3g\n",
           g_meas_rms, g_meas_rope, g_meas_gqa, g_meas_og);
    printf("  digest %016llx\n", (unsigned long long)g_digest);
    printf("test_attn: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
