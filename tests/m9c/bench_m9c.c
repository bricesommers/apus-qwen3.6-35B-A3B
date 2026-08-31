/*
 * tests/m9c/bench_m9c.c — ILP GEMV micro-variants at the real Qwen
 * decode shapes, INSIDE the approved reorder class only (re-anchored at
 * M9; the Ling V2 8-row grouping this bench originally picked is now
 * the SHIPPED c/bf16.h kernel, so V0 here is the current 8-row form).
 *
 * The class (tests/m9b, c/bf16.h header): per output row, four float32x4
 * accumulators acc[0..3]; per ascending 32-wide chunk, the eight product
 * vectors p_0..p_7 accumulate acc[q & 3] += p_q (q ascending); row end
 * s = (acc[0]+acc[1]) + (acc[2]+acc[3]), total = (s[0]+s[1]) + (s[2]+s[3]);
 * K tail appended scalar ascending. Any candidate must produce the
 * IDENTICAL bits as V0 (verified below with memcmp per shape — err/esc
 * is 0 by construction). Row-group width is pure interleave (the per-row
 * sequence is untouched), so grouping variants stay inside the class by
 * construction. c/bf16.h is NOT in the M9 edit scope: this bench is
 * informational — winners are recorded, nothing ships.
 *
 * Variants (all bit-preserving transforms of the per-row sequence):
 *   V0  the current kernel (8-row groups), verbatim copy.
 *   V1  V0 + software prefetch of the weight rows one row-group ahead.
 *   V2  16-row groups (same per-row sequence; more independent streams).
 *   V3  V0 + xf chunk loads hoisted out of the row loop (explicit CSE).
 *   V4  V1 + V3 combined.
 *
 * Single-thread row-function timing (kernel-level), best of 7, at the
 * real Qwen decode shapes. Informational — NOT a gate. `make bench-m9c`.
 */

/* NOTE: this file is macOS/ARM-specific (the ILP GEMV lives in NEON);
 * off-ARM it prints a stub message (the x86 hot path is the M12a-2
 * bitwise-sequential AVX2 kernel — its tuning is a separate story). */
#ifdef __ARM_NEON

#define APUS_BF16_IMPLEMENTATION
#include "bf16.h"

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float dot_ilp_v0(const uint16_t *wr, const float *xf, size_t K) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    size_t k = 0;
    for (; k + 32 <= K; k += 32) {
        for (int i = 0; i < 32; i += 8) {
            uint16x8_t h = vld1q_u16(wr + k + i);
            float32x4_t w0 = vreinterpretq_f32_u32(
                vshll_n_u16(vget_low_u16(h), 16));
            float32x4_t w1 = vreinterpretq_f32_u32(
                vshll_n_u16(vget_high_u16(h), 16));
            int q = i / 4;
            float32x4_t p0 = vmulq_f32(w0, vld1q_f32(xf + k + i));
            float32x4_t p1 = vmulq_f32(w1, vld1q_f32(xf + k + i + 4));
            if (q & 2) {
                a2 = vaddq_f32(a2, p0);
                a3 = vaddq_f32(a3, p1);
            } else {
                a0 = vaddq_f32(a0, p0);
                a1 = vaddq_f32(a1, p1);
            }
        }
    }
    float32x4_t s = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    float total = (vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1))
                + (vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3));
    for (; k < K; k++)
        total += apus_bf16_f32(wr[k]) * xf[k];
    return total;
}

/* N-row group body shared by every variant: R rows interleaved, the
 * identical per-row accumulation sequence (grouping is pure interleave).
 * PREFETCH/CSE are compile-time switches expanded per variant by the
 * macro below (no runtime branches in the hot loop). */

#define GEMV_ROWS_V(NAME, R, PF, CSE)                                       \
static void NAME(const uint16_t *w, const float *xf, uint16_t *y,           \
                 size_t K, size_t o0, size_t o1) {                          \
    size_t o = o0;                                                          \
    for (; o + R <= o1; o += R) {                                           \
        const uint16_t *wrs[R];                                             \
        for (int r = 0; r < R; r++) {                                       \
            wrs[r] = w + (o + r) * K;                                       \
            if (PF && o + 2 * R <= o1)                                      \
                __builtin_prefetch(wrs[r] + R * K, 0, 1);                   \
        }                                                                   \
        float32x4_t a[R][4];                                                \
        for (int r = 0; r < R; r++)                                         \
            for (int j = 0; j < 4; j++) a[r][j] = vdupq_n_f32(0.0f);        \
        size_t k = 0;                                                       \
        for (; k + 32 <= K; k += 32) {                                      \
            if (PF && k + 32 + 128 <= K)                                    \
                for (int r = 0; r < R; r++)                                 \
                    __builtin_prefetch(wrs[r] + k + 128, 0, 3);             \
            for (int i = 0; i < 32; i += 8) {                               \
                int q = i / 4;                                              \
                float32x4_t x0 = vld1q_f32(xf + k + i);                     \
                float32x4_t x1 = vld1q_f32(xf + k + i + 4);                 \
                for (int r = 0; r < R; r++) {                               \
                    uint16x8_t h = vld1q_u16(wrs[r] + k + i);               \
                    float32x4_t w0 = vreinterpretq_f32_u32(                 \
                        vshll_n_u16(vget_low_u16(h), 16));                  \
                    float32x4_t w1 = vreinterpretq_f32_u32(                 \
                        vshll_n_u16(vget_high_u16(h), 16));                 \
                    if (CSE) {                                              \
                        a[r][q & 3] = vaddq_f32(a[r][q & 3],                \
                            vmulq_f32(w0, x0));                             \
                        a[r][(q + 1) & 3] = vaddq_f32(a[r][(q + 1) & 3],    \
                            vmulq_f32(w1, x1));                             \
                    } else {                                                \
                        a[r][q & 3] = vaddq_f32(a[r][q & 3],                \
                            vmulq_f32(w0, vld1q_f32(xf + k + i)));          \
                        a[r][(q + 1) & 3] = vaddq_f32(a[r][(q + 1) & 3],    \
                            vmulq_f32(w1, vld1q_f32(xf + k + i + 4)));      \
                    }                                                       \
                }                                                           \
            }                                                               \
        }                                                                   \
        for (int r = 0; r < R; r++) {                                       \
            float32x4_t s = vaddq_f32(vaddq_f32(a[r][0], a[r][1]),          \
                                      vaddq_f32(a[r][2], a[r][3]));         \
            float total = (vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1))     \
                        + (vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3));    \
            for (size_t kk = k; kk < K; kk++)                               \
                total += apus_bf16_f32(wrs[r][kk]) * xf[kk];                \
            y[o + r] = apus_bf16_bits(total);                               \
        }                                                                   \
    }                                                                       \
    if (R > 8) {                                                            \
        for (; o + 8 <= o1; o += 8) {                                       \
            const uint16_t *wrs[8];                                         \
            for (int r = 0; r < 8; r++) wrs[r] = w + (o + r) * K;           \
            float32x4_t a[8][4];                                            \
            for (int r = 0; r < 8; r++)                                     \
                for (int j = 0; j < 4; j++) a[r][j] = vdupq_n_f32(0.0f);    \
            size_t k = 0;                                                   \
            for (; k + 32 <= K; k += 32) {                                  \
                for (int i = 0; i < 32; i += 8) {                           \
                    int q = i / 4;                                          \
                    for (int r = 0; r < 8; r++) {                           \
                        uint16x8_t h = vld1q_u16(wrs[r] + k + i);           \
                        float32x4_t w0 = vreinterpretq_f32_u32(             \
                            vshll_n_u16(vget_low_u16(h), 16));              \
                        float32x4_t w1 = vreinterpretq_f32_u32(             \
                            vshll_n_u16(vget_high_u16(h), 16));             \
                        a[r][q & 3] = vaddq_f32(a[r][q & 3],                \
                            vmulq_f32(w0, vld1q_f32(xf + k + i)));          \
                        a[r][(q + 1) & 3] = vaddq_f32(a[r][(q + 1) & 3],    \
                            vmulq_f32(w1, vld1q_f32(xf + k + i + 4)));      \
                    }                                                       \
                }                                                           \
            }                                                               \
            for (int r = 0; r < 8; r++) {                                   \
                float32x4_t s = vaddq_f32(vaddq_f32(a[r][0], a[r][1]),      \
                                          vaddq_f32(a[r][2], a[r][3]));     \
                float total = (vgetq_lane_f32(s, 0)                         \
                               + vgetq_lane_f32(s, 1))                      \
                            + (vgetq_lane_f32(s, 2)                         \
                               + vgetq_lane_f32(s, 3));                     \
                for (size_t kk = k; kk < K; kk++)                           \
                    total += apus_bf16_f32(wrs[r][kk]) * xf[kk];            \
                y[o + r] = apus_bf16_bits(total);                           \
            }                                                               \
        }                                                                   \
    }                                                                       \
    if (R > 4) {                                                            \
        for (; o + 4 <= o1; o += 4) {                                       \
            const uint16_t *wrs[4];                                         \
            for (int r = 0; r < 4; r++) wrs[r] = w + (o + r) * K;           \
            float32x4_t a[4][4];                                            \
            for (int r = 0; r < 4; r++)                                     \
                for (int j = 0; j < 4; j++) a[r][j] = vdupq_n_f32(0.0f);    \
            size_t k = 0;                                                   \
            for (; k + 32 <= K; k += 32) {                                  \
                for (int i = 0; i < 32; i += 8) {                           \
                    int q = i / 4;                                          \
                    for (int r = 0; r < 4; r++) {                           \
                        uint16x8_t h = vld1q_u16(wrs[r] + k + i);           \
                        float32x4_t w0 = vreinterpretq_f32_u32(             \
                            vshll_n_u16(vget_low_u16(h), 16));              \
                        float32x4_t w1 = vreinterpretq_f32_u32(             \
                            vshll_n_u16(vget_high_u16(h), 16));             \
                        a[r][q & 3] = vaddq_f32(a[r][q & 3],                \
                            vmulq_f32(w0, vld1q_f32(xf + k + i)));          \
                        a[r][(q + 1) & 3] = vaddq_f32(a[r][(q + 1) & 3],    \
                            vmulq_f32(w1, vld1q_f32(xf + k + i + 4)));      \
                    }                                                       \
                }                                                           \
            }                                                               \
            for (int r = 0; r < 4; r++) {                                   \
                float32x4_t s = vaddq_f32(vaddq_f32(a[r][0], a[r][1]),      \
                                          vaddq_f32(a[r][2], a[r][3]));     \
                float total = (vgetq_lane_f32(s, 0)                         \
                               + vgetq_lane_f32(s, 1))                      \
                            + (vgetq_lane_f32(s, 2)                         \
                               + vgetq_lane_f32(s, 3));                     \
                for (size_t kk = k; kk < K; kk++)                           \
                    total += apus_bf16_f32(wrs[r][kk]) * xf[kk];            \
                y[o + r] = apus_bf16_bits(total);                           \
            }                                                               \
        }                                                                   \
    }                                                                       \
    for (; o < o1; o++)                                                     \
        y[o] = apus_bf16_bits(dot_ilp_v0(w + o * K, xf, K));                \
}

GEMV_ROWS_V(gemv_rows_v0, 8, 0, 0)   /* current shipped kernel */
GEMV_ROWS_V(gemv_rows_v1, 8, 1, 0)   /* + prefetch */
GEMV_ROWS_V(gemv_rows_v2, 16, 0, 0)  /* 16-row groups */
GEMV_ROWS_V(gemv_rows_v3, 8, 0, 1)   /* + xf-load hoist */
GEMV_ROWS_V(gemv_rows_v4, 8, 1, 1)   /* V1 + V3 */

typedef void (*RowsFn)(const uint16_t *, const float *, uint16_t *,
                       size_t, size_t, size_t);

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

int main(void) {
    printf("bench_m9c: ILP GEMV variants (single-thread rows, best of 7)"
           " — V0 = the shipped 8-row kernel\n");
    static const size_t shapes[][2] = {
        {8192, 2048}, {2048, 4096}, {4096, 2048}, {512, 2048},
        {2048, 512}, {1024, 2048}, {256, 2048}, {32, 2048},
        {248320, 2048},
    };
    static const struct { const char *name; RowsFn fn; } vs[] = {
        { "V0 cur  ", gemv_rows_v0 },
        { "V1 pf   ", gemv_rows_v1 },
        { "V2 16row", gemv_rows_v2 },
        { "V3 cse  ", gemv_rows_v3 },
        { "V1+3    ", gemv_rows_v4 },
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        size_t O = shapes[s][0], K = shapes[s][1];
        uint16_t *w = malloc(O * K * 2);
        uint16_t *x = malloc(K * 2);
        float *xf = malloc(K * sizeof(float));
        uint16_t *y0 = malloc(O * 2);
        uint16_t *y1 = malloc(O * 2);
        for (size_t i = 0; i < O * K; i++)
            w[i] = (uint16_t)(0x2E00 | (rng_u64() % 1024));
        for (size_t i = 0; i < K; i++)
            x[i] = apus_bf16_bits((float)(i % 7) * 0.25f);
        for (size_t i = 0; i < K; i++) xf[i] = apus_bf16_f32(x[i]);
        gemv_rows_v0(w, xf, y0, K, 0, O);
        double best[8] = {0};
        for (size_t v = 0; v < sizeof vs / sizeof vs[0]; v++) {
            memset(y1, 0, O * 2);
            vs[v].fn(w, xf, y1, K, 0, O);
            if (memcmp(y0, y1, O * 2) != 0) {
                printf("  O=%-6zu K=%-5zu %s: BITWISE MISMATCH — "
                       "disqualified\n", O, K, vs[v].name);
                best[v] = -1.0;
                continue;
            }
            int iters = O >= 100000 ? 2 : 7;
            double b = 1e9;
            for (int r = 0; r < iters; r++) {
                double t = now_s();
                vs[v].fn(w, xf, y1, K, 0, O);
                double dt = now_s() - t;
                if (dt < b) b = dt;
            }
            best[v] = b;
        }
        printf("  O=%-6zu K=%-5zu", O, K);
        for (size_t v = 0; v < sizeof vs / sizeof vs[0]; v++) {
            if (best[v] < 0) printf("  %s DISQ", vs[v].name);
            else printf("  %s %7.3fms (%+.1f%%)", vs[v].name,
                        best[v] * 1e3, (best[0] / best[v] - 1.0) * 100.0);
        }
        printf("\n");
        free(w); free(x); free(xf); free(y0); free(y1);
    }
    printf("(+%% = faster than V0; informational only — c/bf16.h is "
           "outside the M9 edit scope, nothing ships)\n");
    return 0;
}

#else  /* !__ARM_NEON */
#include <stdio.h>
int main(void) {
    printf("bench_m9c: ARM-only (the ILP GEMV is the NEON hot path; x86 "
           "uses the M12a-2 AVX2 kernel)\n");
    return 0;
}
#endif
