/* tests/m9a/bench_m9a.c — A/B microbench: the staged-product bitwise
 * GEMV rows at 4-row chains vs 8-row chains (identical roundings:
 * products single-rounded and staged, adds strictly ascending k per
 * row), at the real Qwen decode shapes (GDN qkv projection 8192x2048
 * and a 16k-row slab of the 248320x2048 lm_head). The 8-chain form is
 * what c/bf16.h's bitwise kernel ships; informational. */
#ifdef __ARM_NEON
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define CHUNK 32u

/* --- current 4-row kernel (verbatim from c/bf16.h) --- */
static inline void prod_chunk(const uint16_t *wr, const float *xf, float *st) {
    for (int i = 0; i < (int)CHUNK; i += 8) {
        uint16x8_t h = vld1q_u16(wr + i);
        float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
        float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h), 16));
        vst1q_f32(st + i, vmulq_f32(w0, vld1q_f32(xf + i)));
        vst1q_f32(st + i + 4, vmulq_f32(w1, vld1q_f32(xf + i + 4)));
    }
}

static void gemv_rows_4(const uint16_t *w, const float *xf, float *y,
                        size_t K, size_t o0, size_t o1) {
    float st[4][CHUNK];
    size_t o = o0;
    for (; o + 4 <= o1; o += 4) {
        const uint16_t *w0 = w + (o + 0) * K;
        const uint16_t *w1 = w + (o + 1) * K;
        const uint16_t *w2 = w + (o + 2) * K;
        const uint16_t *w3 = w + (o + 3) * K;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        size_t k = 0;
        for (; k + CHUNK <= K; k += CHUNK) {
            prod_chunk(w0 + k, xf + k, st[0]);
            prod_chunk(w1 + k, xf + k, st[1]);
            prod_chunk(w2 + k, xf + k, st[2]);
            prod_chunk(w3 + k, xf + k, st[3]);
            for (int i = 0; i < (int)CHUNK; i++) {
                a0 += st[0][i]; a1 += st[1][i]; a2 += st[2][i]; a3 += st[3][i];
            }
        }
        y[o+0] = a0; y[o+1] = a1; y[o+2] = a2; y[o+3] = a3;
    }
}

/* --- 8-row variant: identical per-row add order --- */
static void gemv_rows_8(const uint16_t *w, const float *xf, float *y,
                        size_t K, size_t o0, size_t o1) {
    float st[8][CHUNK];
    size_t o = o0;
    for (; o + 8 <= o1; o += 8) {
        const uint16_t *wr[8];
        for (int r = 0; r < 8; r++) wr[r] = w + (o + r) * K;
        float a[8] = {0,0,0,0,0,0,0,0};
        size_t k = 0;
        for (; k + CHUNK <= K; k += CHUNK) {
            for (int r = 0; r < 8; r++) prod_chunk(wr[r] + k, xf + k, st[r]);
            for (int i = 0; i < (int)CHUNK; i++)
                for (int r = 0; r < 8; r++) a[r] += st[r][i];
        }
        for (int r = 0; r < 8; r++) y[o + r] = a[r];
    }
    for (; o < o1; o++) {
        const uint16_t *wr = w + o * K;
        float acc = 0.0f;
        float st1[CHUNK];
        for (size_t k = 0; k + CHUNK <= K; k += CHUNK) {
            prod_chunk(wr + k, xf + k, st1);
            for (int i = 0; i < (int)CHUNK; i++) acc += st1[i];
        }
        y[o] = acc;
    }
}

int main(void) {
    size_t O = 8192, K = 2048;
    uint16_t *w = malloc(O * K * 2);
    float *xf = malloc(K * 4);
    float *y = malloc(O * 4);
    for (size_t i = 0; i < O * K; i++) w[i] = (uint16_t)(0x2E00 | (i % 1024));
    for (size_t i = 0; i < K; i++) xf[i] = (float)(i % 7) * 0.25f;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now_s();
        for (int it = 0; it < 50; it++) gemv_rows_4(w, xf, y, K, 0, O);
        double t1 = now_s();
        for (int it = 0; it < 50; it++) gemv_rows_8(w, xf, y, K, 0, O);
        double t2 = now_s();
        printf("rep %d: 4-row %.2f ms, 8-row %.2f ms (%.2fx)\n", rep,
               (t1-t0)*20, (t2-t1)*20, (t1-t0)/(t2-t1));
    }
    /* also the DRAM-ish shape: a 16k-row slab of the 248320x2048 head */
    {
        size_t O2 = 16384, K2 = 2048;
        uint16_t *w2 = malloc(O2 * K2 * 2);
        for (size_t i = 0; i < O2 * K2; i++) w2[i] = (uint16_t)(0x2E00 | (i % 1024));
        double t0 = now_s();
        gemv_rows_4(w2, xf, y, K2, 0, O2);
        double t1 = now_s();
        gemv_rows_8(w2, xf, y, K2, 0, O2);
        double t2 = now_s();
        printf("head slab (16k rows x 2048): 4-row %.2f ms, 8-row %.2f ms (%.2fx)\n",
               (t1-t0)*1000, (t2-t1)*1000, (t1-t0)/(t2-t1));
        free(w2);
    }
    /* bitwise check */
    float *y1 = malloc(O * 4), *y2 = malloc(O * 4);
    gemv_rows_4(w, xf, y1, K, 0, O);
    gemv_rows_8(w, xf, y2, K, 0, O);
    printf("bitwise: %s\n", memcmp(y1, y2, O * 4) == 0 ? "IDENTICAL" : "DIFFER");
    return 0;
}

#else  /* !__ARM_NEON (M12: off-ARM stub, the Apus pattern) */
#include <stdio.h>
int main(void) {
    printf("bench_m9a: ARM/Darwin-only microbench (NEON/ILP/BLAS A/B)"
           " — see bench-m12a2 on x86\n");
    return 0;
}
#endif /* __ARM_NEON */
