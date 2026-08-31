/*
 * c/blas.h — Accelerate.framework (vecLib/AMX) FP32 GEMM path for the
 * batch-M prefill shapes (M9b), ported from Apus c/blas.h and reduced to
 * this engine's format (pure-BF16 checkpoint: operands widened EXACTLY
 * to FP32). macOS/ARM only (Accelerate is a system framework, linked
 * like libSystem); elsewhere the entry points are no-op stubs and
 * apus_blas_available() returns 0, so dispatch sites stay on their
 * pinned paths at EVERY M (no new dependencies). Header-only (static
 * inline, like c/pool.h): no extra implementation macro needed.
 *
 * Dispatch contract:
 *   - M <  APUS_BLAS_M_MIN: caller keeps the NEON kernels. 128 is the
 *     gate-audited cutoff for THIS codebase: the largest batched M in
 *     any pre-M9b suite is m5's prefill_len64 (64); m4c 9, m6c 16, m7a
 *     ~32, m6a/m6b 8, m8 prompt 8 / verify 4 / mtp prefill 7 — all keep
 *     their pinned NEON paths, so every chunk-invariance and
 *     eager-vs-store bitwise gate is untouched. Decode (M=1) never
 *     dispatches.
 *   - M >= APUS_BLAS_M_MIN: cblas_sgemm from Accelerate, one vecLib
 *     thread per call (VECLIB_MAXIMUM_THREADS=1 pinned on first use),
 *     over a FIXED grid of <=8 MB weight row tiles distributed across
 *     the c/pool.h lanes (AMX is per-core). The tile grid depends only
 *     on (O, K), so outputs are deterministic and APUS_THREADS-
 *     independent by construction.
 *
 * Numerics contract (the accepted reorder class, user-approved
 * 2026-08-07): BF16 operands widen to FP32 EXACTLY (f32 bits =
 * code << 16), the GEMM is FP32, and outputs are RNE-rounded to BF16
 * once at the end (the anchor's rounding boundary) — the ONLY
 * difference vs the scalar anchor is the FP32 summation order inside
 * the dots. Measured err/esc vs FP64 truth on all real shapes in
 * tests/m9b (bound 2e-5, the m3 esc class).
 */
#ifndef APUS_BLAS_H
#define APUS_BLAS_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "bf16.h"

#if defined(__APPLE__) && defined(__ARM_NEON)
#define APUS_BLAS 1
#else
#define APUS_BLAS 0
#endif

/* Batch-M dispatch cutoff: above every pre-M9b suite's largest batched
 * M (m5 64, m7a ~32, m6c 16, m4c 9, m6a/m6b/m8 <= 8) so all existing
 * bitwise gates keep their pinned NEON paths; above the measured
 * NEON/BLAS crossover (~M=32..64, tests/m9b bench). */
#define APUS_BLAS_M_MIN 128

static inline int apus_blas_available(void);
static inline void apus_bf16_gemm_blas(const uint16_t *w,
                                     const uint16_t *x, float *out,
                                     size_t M, size_t O, size_t K);

/* The dispatch used by c/layer.h / c/cache.h batched-prefill call
 * sites: M >= APUS_BLAS_M_MIN with an available BLAS -> the AMX path;
 * else the bf16.h hot GEMM (M9b ILP / scalar anchor off-NEON).
 * M10: the fallback is the NO-HOOK variant — gemm_fast's call sites
 * include the tiered expert linears whose slab views are TRANSIENT
 * (LRU-evicted, reused), and the Metal backend's pointer-keyed
 * zero-copy cache must never see those (the stable-pointer invariant;
 * c/blas.h itself reads weights synchronously, so the AMX path is
 * slab-safe). Resident-weight batched GEMMs that should offload call
 * apus_bf16_gemm_hot directly (c/layer.h phase-A attention
 * projections). */
static inline void apus_bf16_gemm_fast(const uint16_t *w,
                                       const uint16_t *x, uint16_t *y,
                                       size_t M, size_t O, size_t K) {
#if APUS_BLAS
    if (M >= APUS_BLAS_M_MIN && apus_blas_available()) {
        ApusScratchMark mk = apus_scratch_mark();
        float *of = (float *)apus_scratch_alloc(M * O * sizeof(float));
        if (of) {
            apus_bf16_gemm_blas(w, x, of, M, O, K);
            /* single RNE narrowing per output — the anchor's rounding
             * boundary (integer-identical to apus_bf16_bits) */
            for (size_t i = 0; i < M * O; i++)
                y[i] = apus_bf16_bits(of[i]);
            apus_scratch_reset(mk);
            return;
        }
        apus_scratch_reset(mk);
    }
#endif
    apus_bf16_gemm_hot_cpu(w, x, y, M, O, K);
}

#if APUS_BLAS

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <Accelerate/Accelerate.h>
#include <stdlib.h>

static inline int apus_blas_available(void) {
    static int state = -1;
    if (state < 0) {
        const char *no = getenv("APUS_NO_BLAS");
        state = (no && no[0] == '1') ? 0 : 1;
        if (state)
            setenv("VECLIB_MAXIMUM_THREADS", "1", 0); /* before 1st sgemm */
    }
    return state;
}

/* Weight-tile scratch rows: largest multiple of 128 with
 * rows*K*4 <= cap. */
static inline size_t apus_blas_tile_rows(size_t O, size_t K,
                                         size_t cap_bytes) {
    size_t rows = cap_bytes / (K * sizeof(float));
    rows &= ~(size_t)127;
    if (rows < 128) rows = 128;
    return rows < O ? rows : O;
}

/* 8 MB per-lane tiles: big enough for efficient sgemm shapes, small
 * enough that 8 pool lanes cost 64 MB of TLS scratch. */
#define APUS_BLAS_TILE_CAP ((size_t)8 << 20)

typedef struct {
    const uint16_t *wb;     /* [O,K] BF16 */
    const float    *a;      /* widened activations [M,K] */
    float          *out;
    size_t M, O, K, OT;
} ApusBlasJob;

/* One weight tile [o0, o0+on): exact widen into a thread-local scratch
 * tile, then a single-thread sgemm. The tile grid is fixed by (O, K)
 * only, so every output element is produced by an sgemm of the same
 * shape regardless of lane count — deterministic and thread-count
 * independent by construction (asserted in tests/m9b). */
static inline void apus_blas_tile(const ApusBlasJob *j, size_t o0, size_t on) {
    size_t K = j->K;
    ApusScratchMark mk = apus_scratch_mark();
    float *wd = apus_scratch_alloc(on * K * sizeof(float));
    int heap = 0;
    if (!wd) {
        wd = malloc(on * K * sizeof(float));
        heap = 1;
        if (!wd) { apus_scratch_reset(mk); return; }
    }
    for (size_t oo = 0; oo < on; oo++)
        apus_bf16_widen_neon(j->wb + (o0 + oo) * K, wd + oo * K, K);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)j->M, (int)on, (int)K, 1.0f,
                j->a, (int)K, wd, (int)K, 0.0f, j->out + o0, (int)j->O);
    if (heap) free(wd);
    apus_scratch_reset(mk);
}

static inline void apus_blas_tiles(void *vjob, size_t t0, size_t t1) {
    const ApusBlasJob *j = vjob;
    for (size_t t = t0; t < t1; t++) {
        size_t o0 = t * j->OT, on = j->O - o0 < j->OT ? j->O - o0 : j->OT;
        apus_blas_tile(j, o0, on);
    }
}

static inline void apus_bf16_gemm_blas(const uint16_t *w,
                                     const uint16_t *x, float *out,
                                     size_t M, size_t O, size_t K) {
    ApusScratchMark mk = apus_scratch_mark();
    float *a = apus_scratch_alloc(M * K * sizeof(float));
    for (size_t m = 0; m < M; m++)
        apus_bf16_widen_neon(x + m * K, a + m * K, K);
    ApusBlasJob job = { w, a, out, M, O, K,
                        apus_blas_tile_rows(O, K, APUS_BLAS_TILE_CAP) };
    apus_pool_run((O + job.OT - 1) / job.OT, apus_blas_tiles, &job);
    apus_scratch_reset(mk);
}

#pragma clang diagnostic pop

#else  /* !APUS_BLAS: no-op stubs */

static inline int apus_blas_available(void) { return 0; }
static inline void apus_bf16_gemm_blas(const uint16_t *w,
                                     const uint16_t *x, float *out,
                                     size_t M, size_t O, size_t K) {
    (void)w; (void)x; (void)out; (void)M; (void)O; (void)K;
}

#endif /* APUS_BLAS */
#endif /* APUS_BLAS_H */
