/*
 * tests/m12/test_m12a2.c — hard gates for the M12 AVX2 x86 kernels
 * (c/x86.h + the dispatch sites in c/bf16.h, c/moe.h),
 * ported from the Apus M12a-2 suite structure and adapted to this
 * engine's kernel set (pure-BF16 checkpoint). M12 RE-ANCHOR: the stale
 * Ling KDA/MLA AVX2 kernels are deleted from c/x86.h; the Qwen GDN/GQA
 * kernels deliberately run the scalar anchors on x86 (documented
 * fallback — c/x86.h's disposition note), and the shape sweep is
 * retargeted to the Qwen shapes.
 *
 * THE CONTRACT: every AVX2 kernel is BITWISE IDENTICAL to the normative
 * scalar kernel it replaces (c/x86.h: exact widen = 16-bit shift, the RNE
 * narrow bit-trick in integer lanes, staged single-rounded products,
 * scalar sequential summation order per output, NO FMA). This suite pins
 * that bitwise identity on every path, plus the thread-count-independence
 * digest (diffed across APUS_THREADS=1/4/8 by the Makefile), plus the
 * "AVX2 path was taken HERE" probe.
 *
 *   1. PROBE: report cpu support; after the battery the AVX2 hit counter
 *      must be > 0 when this CPU has AVX2 (and APUS_X86_DISABLE is unset).
 *   2. EXHAUSTIVE WIDEN: all 65536 BF16 codes, apus_bf16_widen_x86 ==
 *      apus_bf16_f32 bitwise. NARROW: directed IEEE specials (incl. RNE
 *      tie midpoints, overflow-to-inf boundary, NaN payloads) + 1M random
 *      FP32 bit patterns, apus_bf16_narrow8_x86 == apus_bf16_bits.
 *   3. GEMV/GEMM/mt/hot: AVX2 == scalar BITWISE on a shape sweep (odd
 *      tails, chain boundaries, the real Qwen shapes 8192x2048 /
 *      2048x4096 / 512x2048 / 1024x2048 / 2048x512 / 32x2048 /
 *      256x2048) + an IEEE-specials fill; the 248320x2048 head
 *      slab-wise (hot vs scalar); FP64-truth err/esc (masked m3 class,
 *      bound 1e-4) on the moderate shapes; M-independence.
 *   4. moe matvec (fp32 out): hot == scalar BITWISE.
 *   5. GDN recurrence step: DOCUMENTED SCALAR FALLBACK on x86 — the AVX2
 *      hit counter must NOT move across the calls (no stale AVX2 kernel
 *      is reachable), step_mt == step bitwise, repeat-run bitwise.
 *   6. Gated-GQA decode: same fallback gates (counter frozen,
 *      decode_mt == decode bitwise, repeat-run bitwise).
 *
 * Non-AVX2 x86-64: the AVX2-direct checks are skipped with a placeholder
 * (the scalar paths are the portable battery's business). Off x86-64
 * (APUS_X86 == 0) the suite compiles to a trivial pass — the macOS
 * battery includes it to keep the target list platform-uniform.
 *
 * Run from the repository root.
 */
#define APUS_BF16_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION
#define APUS_GDN_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION

#include "bf16.h"
#include "moe.h"
#include "gdn.h"
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

/* ---- digest state (FNV-1a; diffed across APUS_THREADS) ---- */
static uint64_t fnv = 0xcbf29ce484222325ull;

/* =========================================================================*/
#if APUS_X86

/* ---- deterministic PRNG (splitmix64) ---- */
static uint64_t rng_state = 0xC2B2AE3D27D4EB4Full;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_float(void) {   /* uniform in (-2, 2) */
    return ((double)(rng_u64() >> 40) / (double)(1ull << 24) * 4.0 - 2.0);
}
static uint16_t rng_bf16_scaled(float s) {
    return apus_bf16_bits(rng_float() * s);
}

static void digest(const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) { fnv ^= b[i]; fnv *= 0x100000001b3ull; }
}

/* ---- 2. exhaustive widen + narrow ----------------------------------------*/
static void test_widen_narrow(void) {
    static uint16_t codes[65536];
    static float out[65536];
    for (int base = 0; base < 65536; base += 8)
        for (int i = 0; i < 8; i++) codes[base + i] = (uint16_t)(base + i);
    apus_bf16_widen_x86(codes, out, 65536);
    long bad = 0;
    for (int i = 0; i < 65536; i++) {
        float s = apus_bf16_f32(codes[i]);
        uint32_t bs, bg;
        memcpy(&bs, &s, 4);
        memcpy(&bg, &out[i], 4);
        if (bs != bg) bad++;
    }
    CHECK(bad == 0, "widen: %ld/65536 codes differ from scalar", bad);

    /* narrow: directed IEEE specials, then a random bit-pattern sweep */
    static const uint32_t dir[] = {
        0x00000000, 0x80000000,             /* +-0 */
        0x7F800000, 0xFF800000,             /* +-inf */
        0x7FC00000, 0xFFC00000, 0x7F800001, /* NaN payloads */
        0x7FFFFFFF, 0xFFA00000,
        0x3F808000, 0x3F818000,             /* RNE tie midpoints (lo even/odd) */
        0x7F7F8000, 0x7F7FFFFF, 0x7F7F7FFF, /* overflow-to-inf boundary */
        0x00000001, 0x00007FFF, 0x00008000, /* fp32 subnormals -> bf16 0/tie */
        0x00018000, 0x33800000, 0x38808000,
        0xBF808000, 0xFF7F8000,             /* negative ties + neg overflow */
    };
    float in[8];
    uint16_t got[8];
    long badn = 0;
    for (size_t base = 0; base < sizeof dir / sizeof dir[0]; base += 8) {
        for (size_t i = 0; i < 8; i++) {
            uint32_t u = base + i < sizeof dir / sizeof dir[0]
                ? dir[base + i] : 0x3F800000;
            memcpy(&in[i], &u, 4);
        }
        apus_bf16_narrow8_x86(in, got);
        for (size_t i = 0; i < 8 && base + i < sizeof dir / sizeof dir[0];
             i++) {
            uint16_t e = apus_bf16_bits(in[i]);
            if (got[i] != e) badn++;
        }
    }
    CHECK(badn == 0, "narrow directed: %ld differ from scalar", badn);
    badn = 0;
    for (long t = 0; t < 1024 * 1024; t += 8) {
        for (int i = 0; i < 8; i++) {
            uint32_t u = (uint32_t)rng_u64();
            memcpy(&in[i], &u, 4);
        }
        apus_bf16_narrow8_x86(in, got);
        for (int i = 0; i < 8; i++) {
            uint16_t e = apus_bf16_bits(in[i]);
            if (got[i] != e) badn++;
        }
    }
    CHECK(badn == 0, "narrow random sweep: %ld/1M differ from scalar", badn);
}

/* ---- 3. bf16 GEMV/GEMM/mt/hot bitwise + FP64 truth ------------------------*/

/* the m3 masked metric: max |got-truth|/esc, masking differences within
 * 0.4% of |truth| (output quantization, not accumulation error). */
static double truth_err_esc(const uint16_t *w, const uint16_t *x,
                            const uint16_t *y, size_t O, size_t K) {
    double worst = 0;
    for (size_t o = 0; o < O; o++) {
        double acc = 0.0, esc = 0.0;
        for (size_t k = 0; k < K; k++) {
            double p = (double)apus_bf16_f32(w[o * K + k])
                     * (double)apus_bf16_f32(x[k]);
            acc += p;
            esc += fabs(p);
        }
        double r = fabs((double)apus_bf16_f32(y[o]) - acc)
                   / (esc > 1e-30 ? esc : 1e-30);
        double outq = fabs((double)apus_bf16_f32(y[o]) - acc)
                          <= 0.004 * fabs(acc) ? 0.0 : r;
        if (outq > worst) worst = outq;
    }
    return worst;
}

/* IEEE-specials fill: inf, subnormals, +-0, max-finite, smallest-normal
 * among regular values. NaN CODES are deliberately NOT used as inputs:
 * NaN payloads through multi-NaN arithmetic are CODEGEN-dependent on
 * x86 (gcc may commute a SIMD mul's operands — observed at -O2 — and
 * the accumulator's NaN+NaN payload selection shifts across opt levels:
 * this fill at -O1+UBSan / -O2+vectorizers diverged in payload bits
 * only). Every NaN in play here arises from 0*inf, the x86 default QNaN
 * 0xFFC00000 — bit-identical on every path, so NaN propagation through
 * the accumulator and the narrow IS still exercised deterministically.
 * NaN-code coverage lives in the exhaustive widen (all 65536 codes) and
 * the narrow directed/random sweeps. Normative inputs are finite BF16
 * (c/bf16.h); see the c/x86.h header note. */
static uint16_t special_code(void) {
    static const uint16_t sp[] = {
        0x0000, 0x8000, 0x7F80, 0xFF80, 0x7F7F, 0xFF7F,
        0x0001, 0x8001, 0x007F, 0x0080, 0xBF80, 0x3DCC,
    };
    if (rng_u64() % 4 == 0)
        return sp[rng_u64() % (sizeof sp / sizeof sp[0])];
    return rng_bf16_scaled(1.0f);
}

static void test_shape(size_t O, size_t K, int specials, int do_fp64) {
    uint16_t *w = malloc(O * K * 2);
    uint16_t *x = malloc(5 * K * 2);
    uint16_t *y_ref = malloc(5 * O * 2);
    uint16_t *y_got = malloc(5 * O * 2);
    float *xf = malloc(5 * K * sizeof(float));
    for (size_t i = 0; i < O * K; i++)
        w[i] = specials ? special_code() : rng_bf16_scaled(0.5f);
    for (size_t i = 0; i < 5 * K; i++)
        x[i] = specials ? special_code() : rng_bf16_scaled(1.0f);
    /* GEMV: AVX2 == scalar bitwise */
    apus_bf16_gemv_scalar(w, x, y_ref, O, K);
    apus_bf16_gemv_x86(w, x, xf, y_got, O, K);
    long bad = memcmp(y_ref, y_got, O * 2) != 0;
    CHECK(bad == 0, "gemv O=%zu K=%zu sp=%d: AVX2 != scalar", O, K, specials);
    /* mt (dispatched) == scalar bitwise */
    memset(y_got, 0, O * 2);
    apus_bf16_gemv_mt(w, x, xf, y_got, O, K);
    bad = memcmp(y_ref, y_got, O * 2) != 0;
    CHECK(bad == 0, "gemv_mt O=%zu K=%zu: != scalar", O, K);
    /* hot (dispatched, pool + arena) == scalar bitwise */
    {
        ApusScratchMark mk = apus_scratch_mark();
        apus_bf16_gemv_hot(w, x, y_got, O, K);
        apus_scratch_reset(mk);
    }
    bad = memcmp(y_ref, y_got, O * 2) != 0;
    CHECK(bad == 0, "gemv_hot O=%zu K=%zu: != scalar", O, K);
    /* GEMM at several M: AVX2 + mt == scalar bitwise */
    for (size_t M = 1; M <= 5; M += (M == 3 ? 2 : 1)) {   /* 1,2,3,5 */
        apus_bf16_gemm_scalar(w, x, y_ref, M, O, K);
        apus_bf16_gemm_x86(w, x, xf, y_got, M, O, K);
        bad = memcmp(y_ref, y_got, M * O * 2) != 0;
        CHECK(bad == 0, "gemm M=%zu O=%zu K=%zu: AVX2 != scalar", M, O, K);
        memset(y_got, 0, M * O * 2);
        apus_bf16_gemm_mt(w, x, xf, y_got, M, O, K);
        bad = memcmp(y_ref, y_got, M * O * 2) != 0;
        CHECK(bad == 0, "gemm_mt M=%zu O=%zu K=%zu: != scalar", M, O, K);
        ApusScratchMark mk = apus_scratch_mark();
        apus_bf16_gemm_hot(w, x, y_got, M, O, K);
        apus_scratch_reset(mk);
        bad = memcmp(y_ref, y_got, M * O * 2) != 0;
        CHECK(bad == 0, "gemm_hot M=%zu O=%zu K=%zu: != scalar", M, O, K);
    }
    /* M-independence: row 3 of the M=5 run == the M=1 run of that row */
    apus_bf16_gemm_x86(w, x + 3 * K, xf, y_got, 1, O, K);
    apus_bf16_gemm_x86(w, x, xf, y_ref, 5, O, K);
    bad = memcmp(y_ref + 3 * O, y_got, O * 2) != 0;
    CHECK(bad == 0, "M-independence O=%zu K=%zu", O, K);
    if (do_fp64) {
        apus_bf16_gemv_x86(w, x, xf, y_got, O, K);
        double e = truth_err_esc(w, x, y_got, O, K);
        CHECK(e < 1e-4, "gemv O=%zu K=%zu err/esc %.3e >= 1e-4", O, K, e);
        printf("  gemv O=%-6zu K=%-5zu sp=%d err/esc=%.2e\n", O, K, specials,
               e);
    }
    digest(y_got, O * 2);
    free(w); free(x); free(y_ref); free(y_got); free(xf);
}

/* head 248320x2048, slab-wise (full f64 truth is too slow under
 * emulation; 8 slabs, hot == scalar bitwise per slab) */
static void test_head_slab(void) {
    size_t V = 248320, K = 2048, SL = V / 8;
    uint16_t *w = malloc((size_t)SL * K * 2);
    uint16_t *x = malloc(K * 2);
    uint16_t *y1 = malloc(SL * 2);
    uint16_t *y2 = malloc(SL * 2);
    for (size_t i = 0; i < K; i++) x[i] = rng_bf16_scaled(1.0f);
    long bad = 0;
    for (size_t sl = 0; sl < 8; sl++) {
        for (size_t i = 0; i < (size_t)SL * K; i++)
            w[i] = (uint16_t)(rng_u64() & 0x3FFF) | 0x2E00;
        apus_bf16_gemv_scalar(w, x, y1, SL, K);
        ApusScratchMark mk = apus_scratch_mark();
        apus_bf16_gemv_hot(w, x, y2, SL, K);
        apus_scratch_reset(mk);
        if (memcmp(y1, y2, SL * 2) != 0) bad++;
        digest(y2, SL * 2);
    }
    CHECK(bad == 0, "head slabs 248320x2048: %ld/8 slabs hot != scalar",
          bad);
    printf("  head 248320x2048: 8 slabs hot == scalar bitwise\n");
    free(w); free(x); free(y1); free(y2);
}

/* ---- 4. moe router matvec (fp32 out) --------------------------------------*/
static void test_moe_matvec(void) {
    static const size_t shapes[][2] = {
        {256, 2048}, {32, 256}, {7, 3}, {100, 33}, {1, 1}, {64, 4096},
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        size_t O = shapes[s][0], K = shapes[s][1];
        uint16_t *w = malloc(O * K * 2);
        uint16_t *x = malloc(K * 2);
        float *y1 = malloc(O * sizeof(float));
        float *y2 = malloc(O * sizeof(float));
        for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16_scaled(0.5f);
        for (size_t i = 0; i < K; i++) x[i] = rng_bf16_scaled(1.0f);
        apus_moe_matvec_f32(w, x, y1, O, K);
        ApusScratchMark mk = apus_scratch_mark();
        apus_moe_matvec_f32_hot(w, x, y2, O, K);
        apus_scratch_reset(mk);
        CHECK(memcmp(y1, y2, O * sizeof(float)) == 0,
              "moe matvec hot != scalar at O=%zu K=%zu", O, K);
        digest(y2, O * sizeof(float));
        free(w); free(x); free(y1); free(y2);
    }
}

/* ---- 5/6. GDN / gated-GQA: DOCUMENTED SCALAR FALLBACK on x86 ----------
 * The Qwen GDN recurrence (c/gdn.h) and the gated-GQA row (c/attn.h) have
 * NO AVX2 port — the scalar anchors run directly (c/x86.h's disposition
 * note; the base's rule: scalar fallback is always legal, just slower).
 * These gates pin that contract: (a) the AVX2 hit counter must NOT move
 * across the calls — no stale Ling AVX2 kernel is reachable through them;
 * (b) the threaded variant is bitwise the sequential one (by construction,
 * verified); (c) a repeat run from identical inputs is bitwise identical
 * (determinism). Their semantic correctness against fixtures is the
 * m4a/m4b suites' business (in the portable battery). ------------------ */

static void test_gdn_fallback(void) {
    size_t H = 32, Dk = 128, Dv = 128;      /* the real Qwen GDN dims */
    float *S0 = malloc(H * Dk * Dv * sizeof(float));
    float *S1 = malloc(H * Dk * Dv * sizeof(float));
    float *S2 = malloc(H * Dk * Dv * sizeof(float));
    float *q = malloc(H * Dk * sizeof(float));
    float *k = malloc(H * Dk * sizeof(float));
    uint16_t *v = malloc(H * Dv * 2);
    float *g = malloc(H * sizeof(float));
    uint16_t *bt = malloc(H * 2);
    float *o0 = malloc(H * Dv * sizeof(float));
    float *o1 = malloc(H * Dv * sizeof(float));
    for (size_t i = 0; i < H * Dk; i++) { q[i] = rng_float(); k[i] = rng_float(); }
    for (size_t i = 0; i < H * Dv; i++) v[i] = rng_bf16_scaled(1.0f);
    for (size_t i = 0; i < H; i++) {
        g[i] = -5.0f * (float)((rng_u64() >> 40) / 16777216.0);
        bt[i] = apus_bf16_bits((float)((rng_u64() >> 40) / 16777216.0));
    }
    memset(S0, 0, H * Dk * Dv * sizeof(float));
    memcpy(S1, S0, H * Dk * Dv * sizeof(float));
    memcpy(S2, S0, H * Dk * Dv * sizeof(float));
    unsigned long h0 = apus_x86_avx2_hits();
    apus_gdn_step(S0, q, k, v, g, bt, o0, H, Dk, Dv);
    apus_gdn_step_mt(S1, q, k, v, g, bt, o1, H, Dk, Dv);
    CHECK(apus_x86_avx2_hits() == h0,
          "gdn step dispatched an AVX2 kernel (hits %lu -> %lu) — the x86 "
          "fallback must be the scalar anchor", h0, apus_x86_avx2_hits());
    CHECK(memcmp(S0, S1, H * Dk * Dv * sizeof(float)) == 0 &&
          memcmp(o0, o1, H * Dv * sizeof(float)) == 0,
          "gdn step_mt != step bitwise");
    apus_gdn_step(S2, q, k, v, g, bt, o1, H, Dk, Dv);
    CHECK(memcmp(S0, S2, H * Dk * Dv * sizeof(float)) == 0 &&
          memcmp(o0, o1, H * Dv * sizeof(float)) == 0,
          "gdn step repeat run != first run bitwise");
    digest(S0, H * Dk * Dv * sizeof(float));
    digest(o0, H * Dv * sizeof(float));
    printf("  gdn step H=%zu D=%zu: scalar fallback confirmed, mt == seq "
           "bitwise\n", H, Dk);
    free(S0); free(S1); free(S2); free(q); free(k); free(v); free(g);
    free(bt); free(o0); free(o1);
}

static void test_gqa_fallback(void) {
    size_t H = 16, Hkv = 2, D = 256, Tk = 300;   /* the real Qwen GQA dims */
    uint16_t *q = malloc(H * D * 2);
    uint16_t *kc = malloc(Tk * Hkv * D * 2);
    uint16_t *vc = malloc(Tk * Hkv * D * 2);
    uint16_t *o0 = malloc(H * D * 2);
    uint16_t *o1 = malloc(H * D * 2);
    uint16_t *abuf = malloc(Tk * 2);
    float *ebuf = malloc(Tk * sizeof(float));
    for (size_t i = 0; i < H * D; i++) q[i] = rng_bf16_scaled(1.0f);
    for (size_t i = 0; i < Tk * Hkv * D; i++) kc[i] = rng_bf16_scaled(1.0f);
    for (size_t i = 0; i < Tk * Hkv * D; i++) vc[i] = rng_bf16_scaled(1.0f);
    unsigned long h0 = apus_x86_avx2_hits();
    apus_attn_gqa_decode(q, kc, vc, o0, Tk, H, Hkv, D, 0.0625f, abuf, ebuf);
    apus_attn_gqa_decode_mt(q, kc, vc, o1, Tk, H, Hkv, D, 0.0625f);
    CHECK(apus_x86_avx2_hits() == h0,
          "gqa decode dispatched an AVX2 kernel (hits %lu -> %lu) — the x86 "
          "fallback must be the scalar anchor", h0, apus_x86_avx2_hits());
    CHECK(memcmp(o0, o1, H * D * 2) == 0, "gqa decode_mt != decode bitwise");
    apus_attn_gqa_decode(q, kc, vc, o1, Tk, H, Hkv, D, 0.0625f, abuf, ebuf);
    CHECK(memcmp(o0, o1, H * D * 2) == 0,
          "gqa decode repeat run != first run bitwise");
    digest(o0, H * D * 2);
    printf("  gqa decode H=%zu Hkv=%zu D=%zu Tk=%zu: scalar fallback "
           "confirmed, mt == seq bitwise\n", H, Hkv, D, Tk);
    free(q); free(kc); free(vc); free(o0); free(o1); free(abuf); free(ebuf);
}

/* =========================================================================*/
int main(void) {
    printf("test_m12a2: AVX2 kernels vs scalar anchors (bitwise contract)\n");
    printf("  cpu: avx2=%d fma=%d f16c=%d  APUS_X86_DISABLE=%s\n",
           __builtin_cpu_supports("avx2"), __builtin_cpu_supports("fma"),
           __builtin_cpu_supports("f16c"),
           getenv("APUS_X86_DISABLE") ? "set" : "unset");
    fprintf(stderr, "  pool threads: %d\n", apus_pool_threads());
    if (!apus_x86_have_avx2()) {
        /* non-AVX2 x86-64 (or APUS_X86_DISABLE set): the AVX2 kernels must
         * not execute — the engine dispatches to scalar. The AVX2-direct
         * checks are skipped; the scalar paths are the M12a-1 battery's
         * business. */
        printf("  no AVX2 dispatch on this machine — AVX2-direct checks "
               "skipped (placeholder)\n");
        CHECK(1, "placeholder");
        printf("digest %016llx\n", (unsigned long long)fnv);
        printf("test_m12a2: %ld checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    }

    test_widen_narrow();

    /* bf16 GEMV/GEMM/mt/hot: shape sweep (odd tails, chain boundaries,
     * the real Qwen shapes) + an IEEE-specials fill */
    test_shape(1, 1, 0, 1);
    test_shape(7, 3, 0, 1);
    test_shape(33, 100, 0, 1);
    test_shape(20, 2048, 0, 0);        /* 2x8 + 1x4 chain boundary */
    test_shape(129, 1032, 0, 0);
    test_shape(5, 1000, 0, 1);
    test_shape(64, 4096, 0, 1);
    test_shape(64, 256, 1, 0);         /* IEEE specials fill */
    test_shape(8192, 2048, 0, 0);      /* GDN qkv / GQA wq */
    test_shape(2048, 4096, 0, 0);      /* GDN out / GQA wo */
    test_shape(512, 2048, 0, 1);       /* GQA wk/wv, shared expert */
    test_shape(1024, 2048, 0, 0);      /* expert gate_up */
    test_shape(2048, 512, 0, 0);       /* expert down */
    test_shape(32, 2048, 0, 0);        /* GDN b/a */
    test_shape(256, 2048, 0, 0);       /* router scoring */
    test_head_slab();

    test_moe_matvec();

    /* sections 5/6: the Qwen GDN/GQA kernels run the DOCUMENTED scalar
     * fallback on x86 — pin that no AVX2 kernel is reachable through
     * them and that mt == sequential bitwise */
    test_gdn_fallback();
    test_gqa_fallback();

    /* the AVX2 path must actually have been taken on this machine (the
     * raw count is thread-count dependent — print only the boolean) */
    CHECK(apus_x86_avx2_hits() > 0,
          "AVX2 supported but no AVX2 kernel was dispatched (hits=%lu)",
          apus_x86_avx2_hits());
    printf("  avx2 path taken: %s\n", apus_x86_avx2_hits() > 0 ? "yes" : "no");

    printf("digest %016llx\n", (unsigned long long)fnv);
    printf("test_m12a2: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}

/* =========================================================================*/
#else  /* !APUS_X86: nothing to test off x86-64 (macOS uniform target) */

int main(void) {
    printf("test_m12a2: APUS_X86 == 0 on this platform — no AVX2 kernels, "
           "trivial pass\n");
    printf("digest %016llx\n", (unsigned long long)fnv);
    printf("test_m12a2: %ld checks, %d failures\n", checks, failures);
    return 0;
}

#endif /* APUS_X86 */
