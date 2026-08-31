/*
 * tests/m9b/bench_m9b.c — microbench for the M9b dispatch: the bitwise
 * 8-chain kernel (mt) vs the ILP kernel vs the Accelerate BLAS path, at
 * the real Qwen3.6 shapes (hidden 2048 projections, expert 1024/512,
 * a 16k-row slab of the 248320x2048 head). Informational.
 */
#ifdef __ARM_NEON
#define APUS_BF16_IMPLEMENTATION
#include "bf16.h"
#include "blas.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("bench_m9b: bf16 GEMV/GEMM dispatch timings (best of 5)\n");
    /* decode GEMV shapes: bitwise-mt vs ILP-mt */
    static const size_t gv[][2] = {
        {8192, 2048}, {2048, 4096}, {4096, 2048}, {512, 2048},
        {2048, 512}, {1024, 2048}, {256, 2048}, {16384, 2048},
    };
    for (int s = 0; s < 8; s++) {
        size_t O = gv[s][0], K = gv[s][1];
        uint16_t *w = malloc(O * K * 2);
        uint16_t *x = malloc(K * 2);
        float *xf = malloc(K * sizeof(float));
        uint16_t *y = malloc(O * 2);
        for (size_t i = 0; i < O * K; i++)
            w[i] = (uint16_t)(0x2E00 | (i % 1024));
        for (size_t i = 0; i < K; i++)
            x[i] = apus_bf16_bits((float)(i % 7) * 0.25f);
        double b1 = 1e9, b2 = 1e9;
        volatile uint16_t sink;
        for (int r = 0; r < 5; r++) {
            double t = now_s();
            apus_bf16_gemv_mt(w, x, xf, y, O, K);
            sink = y[r];
            if (now_s() - t < b1) b1 = now_s() - t;
            t = now_s();
            apus_bf16_gemv_ilp_neon(w, x, xf, y, O, K);
            sink = y[r];
            if (now_s() - t < b2) b2 = now_s() - t;
        }
        (void)sink;
        printf("gemv O=%-6zu K=%-5zu  bitwise-mt %7.3f ms  ILP-mt %7.3f ms"
               "  (%.2fx)\n", O, K, b1 * 1000, b2 * 1000, b1 / b2);
        free(w); free(x); free(xf); free(y);
    }
    /* prefill GEMM shapes: ILP-mt vs BLAS */
    static const size_t gm[][3] = {
        {128, 1024, 2048}, {256, 1024, 2048}, {512, 1024, 2048},
        {128, 2048, 4096}, {256, 2048, 4096}, {512, 2048, 4096},
        {128, 8192, 2048}, {256, 8192, 2048},
    };
    for (int s = 0; s < 8; s++) {
        size_t M = gm[s][0], O = gm[s][1], K = gm[s][2];
        uint16_t *w = malloc(O * K * 2);
        uint16_t *x = malloc(M * K * 2);
        float *xf = malloc(M * K * sizeof(float));
        uint16_t *y = malloc(M * O * 2);
        float *of = malloc(M * O * sizeof(float));
        for (size_t i = 0; i < O * K; i++)
            w[i] = (uint16_t)(0x2E00 | (i % 1024));
        for (size_t i = 0; i < M * K; i++)
            x[i] = apus_bf16_bits((float)(i % 7) * 0.25f);
        double b1 = 1e9, b2 = 1e9;
        volatile uint16_t sink;
        for (int r = 0; r < 5; r++) {
            double t = now_s();
            apus_bf16_gemm_ilp_neon(w, x, xf, y, M, O, K);
            sink = y[r];
            if (now_s() - t < b1) b1 = now_s() - t;
#if APUS_BLAS
            if (apus_blas_available()) {
                t = now_s();
                apus_bf16_gemm_blas(w, x, of, M, O, K);
                sink = (uint16_t)of[r];
                if (now_s() - t < b2) b2 = now_s() - t;
            }
#endif
        }
        (void)sink;
        printf("gemm M=%-4zu O=%-5zu K=%-5zu  ILP-mt %8.3f ms  BLAS %8.3f ms"
               "  (%.2fx)\n", M, O, K, b1 * 1000, b2 * 1000, b1 / b2);
        free(w); free(x); free(xf); free(y); free(of);
    }
    return 0;
}

#else  /* !__ARM_NEON (M12: off-ARM stub, the Apus pattern) */
#include <stdio.h>
int main(void) {
    printf("bench_m9b: ARM/Darwin-only microbench (NEON/ILP/BLAS A/B)"
           " — see bench-m12a2 on x86\n");
    return 0;
}
#endif /* __ARM_NEON */
