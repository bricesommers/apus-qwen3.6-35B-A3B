/*
 * tests/m3/test_bf16.c — hard-gate tests for c/bf16.h (BF16 kernels).
 *
 *   1. Exhaustive widen: all 65,536 bf16 codes (incl. subnormals, +-0, inf,
 *      NaN payloads) — scalar == the exact (code<<16) bit pattern; NEON row
 *      widen bitwise == scalar.
 *   2. Exhaustive narrow: 65,536 high codes x 6 low-16 patterns (393,216
 *      fp32 values: exact, ties, just-off-ties, subnormals, saturation to
 *      inf, inf, NaN) vs the numpy-oracle golden (float64 candidate-distance
 *      RNE, independent of the C bit-trick). Bitwise.
 *   3. Golden GEMM vs numpy f64 truth (gen_golden.py), esc-based tolerance;
 *      scalar vs NEON bitwise.
 *   4. Shape sweep (odd/small + the real Qwen3.6-35B-A3B shapes) x M in
 *      {1,2,4,7}: scalar vs NEON vs mt BITWISE; all vs in-test FP64 truth
 *      with esc-based tolerance; M-independence (GEMM row == GEMV, bitwise).
 *   5. LM head 248320x2048, light mode: slab-wise scalar/NEON/f64 checks +
 *      full NEON-vs-mt bitwise (avoids the full-scalar pass over 1.02 GB).
 *   6. Edge cases: K=1..3, O=1, tails, zero row -> +0, inf*0 -> NaN, fp32
 *      accumulation overflow -> +-inf, subnormal underflow.
 *
 * The mt paths exercise c/pool.h, so stdout (checks + digest) must be
 * identical at APUS_THREADS=1/4/8 — the Makefile diffs the runs.
 * Run from the repository root (golden fixtures under tests/m3/golden/).
 */
#define APUS_BF16_IMPLEMENTATION
#include "bf16.h"

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

static int g_verbose = 0;

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
static uint16_t rng_bf16(void) { /* realistic finite code via RNE narrow */
    return apus_bf16_bits(rng_float());
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

/* ---- FP64 ground truth: y[m,o] = sum_k f64(w)*f64(x) in ASCENDING k order
 *      (no reassociation), esc = sum_k |w*x|. FP32 accumulation error scales
 *      with esc, not |out| (cancellation can drive |out| to ~0), so
 *      tolerances are fractions of esc. ---- */
static void truth_f64(const uint16_t *w, const uint16_t *x,
                      double *out, double *esc, size_t M, size_t O, size_t K) {
    for (size_t m = 0; m < M; m++) {
        for (size_t o = 0; o < O; o++) {
            double acc = 0.0, scale = 0.0;
            for (size_t k = 0; k < K; k++) {
                double p = (double)apus_bf16_f32(w[o * K + k]) *
                           (double)apus_bf16_f32(x[m * K + k]);
                acc += p;
                scale += fabs(p);
            }
            out[m * O + o] = acc;
            if (esc) esc[m * O + o] = scale;
        }
    }
}

/* ---- file loading ---- */
static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *len = (size_t)sz;
    return buf;
}

/* =========================================================================*/
/* 1. exhaustive widen */
static void test_widen_exhaustive(void) {
    uint16_t *codes = malloc(65536 * sizeof(uint16_t));
    float *out_s = malloc(65536 * sizeof(float));
    for (int i = 0; i < 65536; i++) codes[i] = (uint16_t)i;
    for (int i = 0; i < 65536; i++) out_s[i] = apus_bf16_f32(codes[i]);
    long bad = 0;
    for (int i = 0; i < 65536; i++) {
        uint32_t want = (uint32_t)i << 16;
        CHECK(f32bits(out_s[i]) == want,
              "widen code %04x: got %08x want %08x", i, f32bits(out_s[i]), want);
        if (f32bits(out_s[i]) != want) bad++;
    }
    /* roundtrip through narrow is the identity on every code */
    for (int i = 0; i < 65536; i++) {
        if (apus_bf16_bits(out_s[i]) != (uint16_t)i) {
            CHECK(0, "narrow(widen(%04x)) != identity", i);
            bad++;
        }
        /* bf16_round (f32->f32) agrees with the bit helpers */
        if (f32bits(apus_bf16_round(out_s[i])) != (uint32_t)i << 16) {
            CHECK(0, "bf16_round(widen(%04x)) != identity", i);
            bad++;
        }
    }
#ifdef __ARM_NEON
    float *out_n = malloc(65536 * sizeof(float));
    apus_bf16_widen_neon(codes, out_n, 65536);
    CHECK(memcmp(out_s, out_n, 65536 * sizeof(float)) == 0,
          "widen NEON != scalar over all 65536 codes");
    /* odd length / tail handling */
    apus_bf16_widen_neon(codes + 3, out_n, 65533);
    CHECK(memcmp(out_s + 3, out_n, 65533 * sizeof(float)) == 0,
          "widen NEON tail != scalar");
    free(out_n);
#endif
    if (g_verbose) printf("  widen exhaustive: 65536 codes, %ld bad\n", bad);
    free(codes); free(out_s);
}

/* =========================================================================*/
/* 2. exhaustive narrow vs numpy-oracle golden */
static unsigned char *g_ni, *g_no;
static size_t g_narrow = 0;

static void test_narrow_exhaustive(void) {
    const uint32_t *in = (const uint32_t *)g_ni;
    const uint16_t *want = (const uint16_t *)g_no;
    long bad = 0;
    for (size_t i = 0; i < g_narrow; i++) {
        float f;
        uint32_t u = in[i];
        memcpy(&f, &u, 4);
        uint16_t got = apus_bf16_bits(f);
        CHECK(got == want[i],
              "narrow bits=%08x: got %04x want %04x", u, got, want[i]);
        if (got != want[i]) bad++;
    }
    if (g_verbose || bad)
        printf("  narrow exhaustive: %zu cases, %ld bad\n", g_narrow, bad);
}

/* =========================================================================*/
/* 3. golden GEMM */
static unsigned char *g_w, *g_x, *g_out, *g_esc;
static size_t g_M, g_O, g_K;

static int load_golden(void) {
    size_t len;
    unsigned char *man = read_file("tests/m3/golden/manifest.txt", &len);
    if (!man) { fprintf(stderr, "golden manifest missing — run gen_golden.py\n"); return 0; }
    if (sscanf((char *)man, "M=%zu\nO=%zu\nK=%zu\nNARROW=%zu",
               &g_M, &g_O, &g_K, &g_narrow) != 4) {
        fprintf(stderr, "bad manifest\n"); free(man); return 0;
    }
    free(man);
    size_t l1, l2, l3, l4, l5, l6;
    g_w   = read_file("tests/m3/golden/w.bin", &l1);
    g_x   = read_file("tests/m3/golden/x.bin", &l2);
    g_out = read_file("tests/m3/golden/out.bin", &l3);
    g_esc = read_file("tests/m3/golden/esc.bin", &l4);
    g_ni  = read_file("tests/m3/golden/narrow_in.bin", &l5);
    g_no  = read_file("tests/m3/golden/narrow_out.bin", &l6);
    if (!g_w || !g_x || !g_out || !g_esc || !g_ni || !g_no) {
        fprintf(stderr, "golden fixtures missing\n"); return 0;
    }
    if (l1 != g_O * g_K * 2 || l2 != g_M * g_K * 2 ||
        l3 != g_M * g_O * 8 || l4 != g_M * g_O * 8 ||
        l5 != g_narrow * 4 || l6 != g_narrow * 2) {
        fprintf(stderr, "golden fixture size mismatch\n"); return 0;
    }
    return 1;
}

/* FP32 sequential accumulation over K terms: linear error bound is
 * (K-1)*2^-24 of esc; random fixtures land near sqrt(K)*2^-24. The gate
 * leaves >10x headroom over the measured sweep maximum (see README). */
#define TOL_ERR 1e-4

/* Max error vs f64 truth, as a fraction of esc. *raw includes the bf16
 * output rounding (up to 2^-9 of |t|, part of the contract — torch rounds
 * to bf16 too); the returned masked value excludes it (differences within
 * 0.4% of |t| are output quantization, not accumulation error). The gate
 * asserts the masked value. */
static double check_vs_truth(const uint16_t *y, const double *truth,
                             const double *esc, size_t n, double *raw) {
    double mr = 0, rw = 0;
    for (size_t i = 0; i < n; i++) {
        double t = truth[i];
        double e = esc[i] > 1e-30 ? esc[i] : 1e-30;
        double got = (double)apus_bf16_f32(y[i]);
        double r = fabs(got - t) / e;
        if (r > rw) rw = r;
        double outq = fabs(got - t) <= 0.004 * fabs(t) ? 0.0 : r;
        if (outq > mr) mr = outq;
    }
    if (raw) *raw = rw;
    return mr;
}

static void test_golden_gemm(void) {
    const uint16_t *w = (const uint16_t *)g_w;
    const uint16_t *x = (const uint16_t *)g_x;
    const double *truth = (const double *)g_out;
    const double *esc = (const double *)g_esc;
    uint16_t *y_s = malloc(g_M * g_O * 2);
    uint16_t *y_n = malloc(g_M * g_O * 2);
    float *xf = malloc(g_M * g_K * sizeof(float));

    apus_bf16_gemm_scalar(w, x, y_s, g_M, g_O, g_K);
#ifdef __ARM_NEON
    apus_bf16_gemm_neon(w, x, xf, y_n, g_M, g_O, g_K);
#endif
    double raw = 0;
    double mr = check_vs_truth(y_s, truth, esc, g_M * g_O, &raw);
#ifdef __ARM_NEON
    double rn = check_vs_truth(y_n, truth, esc, g_M * g_O, &raw);
    if (rn > mr) mr = rn;
    CHECK(memcmp(y_s, y_n, g_M * g_O * 2) == 0,
          "golden gemm: NEON != scalar bitwise");
#endif
    digest_bytes(&g_digest, y_s, g_M * g_O * 2);
    printf("  golden gemm (M=%zu O=%zu K=%zu): max err/esc vs f64 = %.3g, "
           "raw (incl. bf16 out rounding) = %.3g (tol %.1g)\n",
           g_M, g_O, g_K, mr, raw, TOL_ERR);
    CHECK(mr < TOL_ERR, "golden gemm err/esc %.3g >= %.1g", mr, TOL_ERR);

    /* GEMV on each activation row == GEMM row, bitwise (M-independence) */
    uint16_t *yv = malloc(g_O * 2);
    for (size_t m = 0; m < g_M; m++) {
        apus_bf16_gemv_scalar(w, x + m * g_K, yv, g_O, g_K);
        CHECK(memcmp(yv, y_s + m * g_O, g_O * 2) == 0,
              "golden gemv row %zu != gemm row bitwise", m);
#ifdef __ARM_NEON
        apus_bf16_gemv_neon(w, x + m * g_K, xf, yv, g_O, g_K);
        CHECK(memcmp(yv, y_s + m * g_O, g_O * 2) == 0,
              "golden gemv NEON row %zu != gemm row bitwise", m);
#endif
    }
    free(yv); free(y_s); free(y_n); free(xf);
}

/* =========================================================================*/
/* 4. shape sweep: scalar vs NEON vs mt bitwise, all vs FP64 truth */
static void test_shapes(void) {
    static const struct { size_t O, K; } shapes[] = {
        {1, 1}, {1, 8}, {3, 5}, {5, 32}, {17, 160}, {64, 256},
        {127, 255}, {128, 512},
        /* the real Qwen3.6-35B-A3B shapes (O x K, all BF16, no bias):
         * GDN in_proj_qkv / attn q_proj 2048->8192; GDN in_proj_z
         * 2048->4096; GDN in_proj_b/a 2048->32; GDN out_proj / attn o_proj
         * 4096->2048; attn k/v_proj 2048->512; router 2048->256;
         * shared_expert_gate 2048->1; expert gate_up 2048->1024; expert
         * down / shared-expert second matmul 512->2048 */
        {8192, 2048}, {4096, 2048}, {32, 2048}, {2048, 4096},
        {512, 2048}, {256, 2048}, {1, 2048}, {1024, 2048}, {2048, 512},
    };
    static const size_t Ms[] = {1, 2, 4, 7};
    double gmax_err = 0, graw = 0;
    long gbit = 0;

    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        size_t O = shapes[si].O, K = shapes[si].K;
        uint16_t *w = malloc(O * K * 2);
        for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16();
        for (size_t o = 0; o < O; o += 13)   /* sprinkle subnormal rows */
            for (size_t k = 0; k < K; k++)
                w[o * K + k] = (uint16_t)(1 + (rng_u64() % 0x7F));

        for (size_t mi = 0; mi < sizeof(Ms) / sizeof(Ms[0]); mi++) {
            size_t M = Ms[mi];
            uint16_t *x = malloc(M * K * 2);
            uint16_t *y_s = malloc(M * O * 2);
            uint16_t *y_n = malloc(M * O * 2);
            uint16_t *y_t = malloc(M * O * 2);
            float *xf = malloc(M * K * sizeof(float));
            double *truth = malloc(M * O * sizeof(double));
            double *esc = malloc(M * O * sizeof(double));
            for (size_t i = 0; i < M * K; i++) x[i] = rng_bf16();
            if (M > 1) memset(x + K, 0, K * 2);   /* one all-zero act row */

            apus_bf16_gemm_scalar(w, x, y_s, M, O, K);
#ifdef __ARM_NEON
            apus_bf16_gemm_neon(w, x, xf, y_n, M, O, K);
            apus_bf16_gemm_mt(w, x, xf, y_t, M, O, K);
#else
            apus_bf16_gemm_mt(w, x, xf, y_t, M, O, K);
            memcpy(y_n, y_s, M * O * 2);
#endif
            truth_f64(w, x, truth, esc, M, O, K);

            /* the hard gate: bitwise across all three paths */
            long b1 = 0, b2 = 0;
            for (size_t i = 0; i < M * O; i++) {
                if (y_n[i] != y_s[i]) b1++;
                if (y_t[i] != y_s[i]) b2++;
            }
            gbit += b1 + b2;
            CHECK(b1 == 0, "gemm O=%zu K=%zu M=%zu: NEON != scalar %ld",
                  O, K, M, b1);
            CHECK(b2 == 0, "gemm O=%zu K=%zu M=%zu: mt != scalar %ld",
                  O, K, M, b2);

            double mr = check_vs_truth(y_s, truth, esc, M * O, &graw);
#ifdef __ARM_NEON
            double rn = check_vs_truth(y_n, truth, esc, M * O, &graw);
            if (rn > mr) mr = rn;
#endif
            if (mr > gmax_err) gmax_err = mr;
            CHECK(mr < TOL_ERR, "gemm O=%zu K=%zu M=%zu err/esc %.3g",
                  O, K, M, mr);
            if (g_verbose || (O >= 768 && M == 7))
                printf("  gemm O=%zu K=%zu M=%zu: err/esc=%.3g bitdiff=%ld\n",
                       O, K, M, mr, b1 + b2);

            /* M-independence: GEMV row r == GEMM row r, bitwise, both paths */
            if (M > 1) {
                size_t rows[2] = {0, M - 1};
                for (int ri = 0; ri < 2; ri++) {
                    size_t r = rows[ri];
                    uint16_t *yv = malloc(O * 2);
                    apus_bf16_gemv_scalar(w, x + r * K, yv, O, K);
                    CHECK(memcmp(yv, y_s + r * O, O * 2) == 0,
                          "M-indep scalar O=%zu K=%zu M=%zu row %zu",
                          O, K, M, r);
#ifdef __ARM_NEON
                    apus_bf16_gemv_neon(w, x + r * K, xf, yv, O, K);
                    CHECK(memcmp(yv, y_s + r * O, O * 2) == 0,
                          "M-indep NEON O=%zu K=%zu M=%zu row %zu",
                          O, K, M, r);
                    apus_bf16_gemv_mt(w, x + r * K, xf, yv, O, K);
                    CHECK(memcmp(yv, y_s + r * O, O * 2) == 0,
                          "M-indep gemv_mt O=%zu K=%zu M=%zu row %zu",
                          O, K, M, r);
#endif
                    free(yv);
                }
            }
            digest_bytes(&g_digest, y_t, M * O * 2);
            free(x); free(y_s); free(y_n); free(y_t);
            free(xf); free(truth); free(esc);
        }
        free(w);
    }
    printf("  shape sweep: max err/esc vs f64 = %.3g, raw (incl. bf16 out "
           "rounding) = %.3g (tol %.1g), bitwise diffs across paths = %ld\n",
           gmax_err, graw, TOL_ERR, gbit);
    CHECK(gbit == 0, "shape sweep bitwise diffs %ld", gbit);
}

/* =========================================================================*/
/* 5. LM head 248320x2048, light mode */
static void test_head_light(void) {
    const size_t O = 248320, K = 2048;
    uint16_t *w = malloc(O * K * 2);        /* 1.02 GB */
    uint16_t *x = malloc(K * 2);
    uint16_t *y_n = malloc(O * 2);
    uint16_t *y_t = malloc(O * 2);
    float *xf = malloc(K * sizeof(float));
    if (!w || !x || !y_n || !y_t || !xf) {
        fprintf(stderr, "head_light: allocation failed\n");
        failures++;
        return;
    }
    for (size_t i = 0; i < O * K; i += 4) {
        uint64_t r = rng_u64();
        w[i] = apus_bf16_bits((float)((double)(r >> 40) / (double)(1ull << 24) * 4.0 - 2.0));
        w[i + 1] = apus_bf16_bits((float)((double)((r >> 8) & 0xFFFFFF) / (double)(1ull << 24) * 4.0 - 2.0));
        w[i + 2] = apus_bf16_bits((float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0 - 2.0));
        w[i + 3] = apus_bf16_bits((float)((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0 - 2.0));
    }
    for (size_t i = 0; i < K; i++) x[i] = rng_bf16();

#ifdef __ARM_NEON
    apus_bf16_gemv_neon(w, x, xf, y_n, O, K);
#else
    apus_bf16_gemv_scalar(w, x, y_n, O, K);
#endif
    apus_bf16_gemv_mt(w, x, xf, y_t, O, K);
    CHECK(memcmp(y_n, y_t, O * 2) == 0,
          "head_light: mt != single-thread bitwise over 248320 rows");
    digest_bytes(&g_digest, y_t, O * 2);

    /* slab-wise scalar vs NEON vs f64 truth (rows are independent, so a
     * gemv on w + o0*K IS rows [o0, o0+SLAB) of the full gemv) */
    static const size_t slabs[] = {0, 123648, 247296};
    const size_t SLAB = 1024;
    uint16_t *y_s = malloc(SLAB * 2);
    double *truth = malloc(SLAB * sizeof(double));
    double *esc = malloc(SLAB * sizeof(double));
    double mr = 0;
    for (int s = 0; s < 3; s++) {
        size_t o0 = slabs[s];
        apus_bf16_gemv_scalar(w + o0 * K, x, y_s, SLAB, K);
        CHECK(memcmp(y_s, y_n + o0, SLAB * 2) == 0,
              "head_light slab %zu: scalar != NEON/mt bitwise", o0);
        truth_f64(w + o0 * K, x, truth, esc, 1, SLAB, K);
        double r = check_vs_truth(y_s, truth, esc, SLAB, NULL);
        if (r > mr) mr = r;
    }
    printf("  head light (O=%zu K=%zu): slab err/esc = %.3g, "
           "full mt-vs-single bitwise\n", O, K, mr);
    CHECK(mr < TOL_ERR, "head_light err/esc %.3g", mr);
    free(y_s); free(truth); free(esc);
    free(w); free(x); free(y_n); free(y_t); free(xf);
}

/* =========================================================================*/
/* 6. edge cases */
static void test_edges(void) {
    uint16_t y[8], y2[8];
    float xf[32];   /* K reaches 32 below */
    (void)y2; (void)xf;   /* NEON-only below; keep non-NEON builds clean */

    /* K=1 */
    {   uint16_t w[1] = {0x3F80}, x[1] = {0x4000};      /* 1.0 * 2.0 */
        apus_bf16_gemv_scalar(w, x, y, 1, 1);
        CHECK(y[0] == 0x4000, "edge K=1: got %04x", y[0]);
#ifdef __ARM_NEON
        apus_bf16_gemv_neon(w, x, xf, y2, 1, 1);
        CHECK(y2[0] == 0x4000, "edge K=1 NEON: got %04x", y2[0]);
#endif
    }
    /* K=3, mixed signs (tail-only NEON path) */
    {   uint16_t w[3] = {0x3F80, 0xBF80, 0x4040};       /* 1, -1, 3 */
        uint16_t x[3] = {0x4000, 0x4000, 0x3F80};       /* 2, 2, 1 */
        apus_bf16_gemv_scalar(w, x, y, 1, 3);           /* 2-2+3 = 3 */
        CHECK(y[0] == 0x4040, "edge K=3: got %04x", y[0]);
#ifdef __ARM_NEON
        apus_bf16_gemv_neon(w, x, xf, y2, 1, 3);
        CHECK(y2[0] == y[0], "edge K=3 NEON != scalar");
#endif
    }
    /* O=1, K=5 (chunk tail only) against f64 truth */
    {   uint16_t w[5], x[5];
        double t[1], e[1];
        for (int i = 0; i < 5; i++) { w[i] = rng_bf16(); x[i] = rng_bf16(); }
        apus_bf16_gemv_scalar(w, x, y, 1, 5);
        truth_f64(w, x, t, e, 1, 1, 5);
        CHECK(fabs((double)apus_bf16_f32(y[0]) - t[0]) <= 0.004 * fabs(t[0]),
              "edge O=1 K=5: got %a want ~%a", apus_bf16_f32(y[0]), t[0]);
    }
    /* zero activations -> exactly +0 (code 0x0000), both paths */
    {   uint16_t w[32], x[32];
        for (int i = 0; i < 32; i++) w[i] = rng_bf16();
        memset(x, 0, sizeof x);
        apus_bf16_gemv_scalar(w, x, y, 1, 32);
        CHECK(y[0] == 0x0000, "edge zero act: got %04x", y[0]);
#ifdef __ARM_NEON
        apus_bf16_gemv_neon(w, x, xf, y2, 1, 32);
        CHECK(y2[0] == 0x0000, "edge zero act NEON: got %04x", y2[0]);
#endif
    }
    /* inf * 0 -> NaN, propagates through narrow with sign/payload bits */
    {   uint16_t w[32], x[32];
        memset(w, 0, sizeof w); memset(x, 0, sizeof x);
        w[0] = 0x7F80;                       /* +inf */
        apus_bf16_gemv_scalar(w, x, y, 1, 32);
        CHECK((y[0] & 0x7FFF) > 0x7F80, "edge inf*0: expected NaN, got %04x", y[0]);
#ifdef __ARM_NEON
        apus_bf16_gemv_neon(w, x, xf, y2, 1, 32);
        CHECK((y2[0] & 0x7FFF) > 0x7F80, "edge inf*0 NEON: got %04x", y2[0]);
#endif
    }
    /* fp32 accumulation overflow -> +inf (0x7F80) / -inf (0xFF80) */
    {   uint16_t w[2] = {0x7F7F, 0x7F7F}, x[2] = {0x3F80, 0x3F80};
        apus_bf16_gemv_scalar(w, x, y, 1, 2);
        CHECK(y[0] == 0x7F80, "edge overflow +inf: got %04x", y[0]);
#ifdef __ARM_NEON
        apus_bf16_gemv_neon(w, x, xf, y2, 1, 2);
        CHECK(y2[0] == 0x7F80, "edge overflow +inf NEON: got %04x", y2[0]);
#endif
        w[0] = w[1] = 0xFF7F;
        apus_bf16_gemv_scalar(w, x, y, 1, 2);
        CHECK(y[0] == 0xFF80, "edge overflow -inf: got %04x", y[0]);
    }
    /* subnormal * subnormal underflows to exactly +0 */
    {   uint16_t w[1] = {0x0001}, x[1] = {0x0001};      /* 2^-133 each */
        apus_bf16_gemv_scalar(w, x, y, 1, 1);
        CHECK(y[0] == 0x0000, "edge subnormal underflow: got %04x", y[0]);
    }
    /* negative zero handling: -0 * 1 + 0 = +0 */
    {   uint16_t w[1] = {0x8000}, x[1] = {0x3F80};
        apus_bf16_gemv_scalar(w, x, y, 1, 1);
        CHECK(y[0] == 0x0000, "edge -0: got %04x", y[0]);
    }
}

int main(int argc, char **argv) {
    g_verbose = argc > 1 && strcmp(argv[1], "-v") == 0;
    printf("test_bf16: BF16 kernel hard-gate tests (Qwen3.6-35B-A3B)\n");
#ifdef __ARM_NEON
    printf("  NEON paths: enabled (bitwise-vs-scalar contract)\n");
#elif APUS_X86
    /* M12a-2: the mt/hot paths dispatch the c/x86.h AVX2 kernels here —
     * this suite's bitwise gates cover them against the scalar anchor. */
    printf("  NEON paths: N/A (x86: AVX2 dispatch %s, bitwise-vs-scalar "
           "contract)\n", apus_x86_have_avx2() ? "active" : "INACTIVE");
#else
    printf("  NEON paths: NOT compiled (scalar only)\n");
#endif
    fprintf(stderr, "  pool threads: %d\n", apus_pool_threads());

    test_widen_exhaustive();

    if (load_golden()) {
        test_narrow_exhaustive();
        test_golden_gemm();
    } else {
        failures++;
        fprintf(stderr, "FAIL: could not load golden fixtures\n");
    }

    test_shapes();
    test_head_light();
    test_edges();

    printf("  digest %016llx\n", (unsigned long long)g_digest);
    printf("test_bf16: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
