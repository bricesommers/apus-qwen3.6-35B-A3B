/*
 * c/bf16.h — BF16 storage helpers and GEMV/GEMM kernels for Ling-3.0-flash
 * (a pure-BF16 checkpoint: no quantization anywhere). C11, libc only
 * (+ arm_neon.h on ARM, immintrin.h via c/x86.h on x86-64). Mirrors the
 * Apus M3 kernel discipline (c/fp4.h): scalar anchor first, SIMD proven
 * bitwise against it, threading bitwise at every thread count.
 *
 * Numerics contract (the normative in-engine anchor for every BF16 linear):
 *
 *   Storage:  W [O, K] BF16 row-major, x [M, K] BF16 row-major,
 *             y [M, O] BF16 row-major.
 *   Widen:    bf16 -> fp32 is EXACT: f32 bits = (uint32)code << 16
 *             (subnormals, inf, NaN payloads included).
 *   Narrow:   fp32 -> bf16 is round-to-nearest-even on the low 16 bits
 *             (u += 0x7FFF + ((u>>16)&1), then truncate). Overflow of a
 *             finite value rounds to +-inf per RNE (the 0x7F7F/0x7F80
 *             midpoint ties to the even code 0x7F80 = inf); NaN passes
 *             through as the high 16 bits (sign + payload preserved).
 *   GEMV/GEMM (scalar anchor — the semantic definition):
 *             acc = 0.0f
 *             for k in 0..K-1 (SEQUENTIAL, strictly ascending):
 *                 p    = f32(W[o,k]) * f32(x[m,k])   (one IEEE fp32 rounding)
 *                 acc += p                           (a second rounding)
 *             y[m,o] = bf16_narrow(acc)
 *             Two roundings per element, NO FMA anywhere; contraction is
 *             pinned off (-ffp-contract=off) so the compiler cannot fuse.
 *             This mirrors torch.nn.Linear's bf16 contract (fp32 accumulate,
 *             bf16 out) with our fixed summation order — torch itself leaves
 *             the order unspecified, so this anchor IS the in-engine
 *             definition every optimized path must reproduce.
 *   NEON kernels: BITWISE identical to the scalar anchor by construction:
 *             widening is exact, the per-element products are computed 4-wide
 *             (vmulq_f32 — the same single rounding as the scalar mul) and
 *             staged, and every output's adds run strictly in ascending k
 *             order. ILP comes only from interleaving INDEPENDENT output
 *             chains (4 rows at a time), never from reassociation. No
 *             reorder class is consumed on this platform (unlike Apus's M9a
 *             fp4 NEON kernels, which took the documented ILP-reorder
 *             class). The Apus M12a-2 AVX2 kernels prove this staged-product
 *             pattern at full SIMD width (tests/m12/README.md, "M12a-2
 *             contract"); here it is the PRIMARY SIMD path, not a port.
 *   x86 kernels (c/x86.h, M12a-2): same staged-product, strictly-
 *             sequential-adds pattern in AVX2 form — BITWISE identical to
 *             the scalar anchor (tests/m12 pins it). The x86 hot path is
 *             this bitwise-sequential AVX2 kernel, NOT the M9b ILP
 *             reorder kernel: digests are within-platform only (ARM hot =
 *             ILP reorder class, x86 hot = anchor bits).
 *   Threaded (mt): bitwise identical to the corresponding single-thread
 *             kernel at EVERY pool size (APUS_THREADS=1 included): x is
 *             widened once by the calling thread (exact), then output rows
 *             are partitioned contiguously over the c/pool.h lanes and each
 *             y[m,o] is computed entirely by one lane with the identical
 *             per-output accumulation order.
 *   Invariants: inputs are finite BF16 in normative use. IEEE propagation
 *             is deterministic and identical across all paths: inf and NaN
 *             widen/narrow exactly as above, 0*inf -> NaN, fp32 accumulation
 *             overflow -> +-inf -> bf16 inf. K has no alignment requirement
 *             (NEON tails are scalar, same roundings).
 *
 * Usage: #define APUS_BF16_IMPLEMENTATION in exactly one TU. Scalar paths
 * are always compiled; NEON paths are compiled when __ARM_NEON is defined
 * and are proven bitwise against the scalar paths in tests/m3.
 */
#ifndef APUS_BF16_H
#define APUS_BF16_H

#include <stddef.h>
#include <stdint.h>

#include "pool.h"
#include "x86.h"
#include "backend_metal.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* k staging granularity of the SIMD kernels (products staged per 32-wide
 * chunk before the strictly-sequential adds). Part of the implementation,
 * not the numerics contract. */
#define APUS_BF16_CHUNK 32u

/* --- scalar helpers ------------------------------------------------------*/

/* BF16 code -> FP32 (EXACT: bits << 16). */
float apus_bf16_f32(uint16_t b);

/* FP32 -> BF16 code, round-to-nearest-even. NaN passes through as the high
 * 16 bits; finite overflow rounds to +-inf (see header contract). */
uint16_t apus_bf16_bits(float x);

/* Round FP32 to BF16 precision and back to FP32 (apus_bf16_f32 of
 * apus_bf16_bits). */
float apus_bf16_round(float x);

/* Row widen: out[i] = apus_bf16_f32(b[i]), i < n. Exact. */
void apus_bf16_widen_scalar(const uint16_t *b, float *out, size_t n);
#ifdef __ARM_NEON
void apus_bf16_widen_neon(const uint16_t *b, float *out, size_t n);
#endif

/* --- scalar anchor (normative) -------------------------------------------*/

/* y[o] = bf16_narrow( sum_k f32(W[o,k]) * f32(x[k]) ), sequential k,
 * mul+add (two roundings per element). w: [O,K], x: [K], y: [O]. */
void apus_bf16_gemv_scalar(const uint16_t *w, const uint16_t *x,
                           uint16_t *y, size_t O, size_t K);

/* Same per output, over M activation rows. w: [O,K], x: [M,K], y: [M,O].
 * Every y[m,o] is bitwise the GEMV result for row m (M-independence). */
void apus_bf16_gemm_scalar(const uint16_t *w, const uint16_t *x,
                           uint16_t *y, size_t M, size_t O, size_t K);

/* --- NEON kernels (BITWISE == scalar anchor) ------------------------------*/
#ifdef __ARM_NEON
/* xf: scratch, K floats (widened x; filled by the kernel). */
void apus_bf16_gemv_neon(const uint16_t *w, const uint16_t *x, float *xf,
                         uint16_t *y, size_t O, size_t K);
/* xf: scratch, M*K floats. */
void apus_bf16_gemm_neon(const uint16_t *w, const uint16_t *x, float *xf,
                         uint16_t *y, size_t M, size_t O, size_t K);
#endif

/* --- threaded variants (c/pool.h; bitwise at every thread count) ---------*/

/* xf: scratch, K floats. Bitwise == apus_bf16_gemv_neon (or the scalar
 * anchor off-NEON) for any APUS_THREADS. */
void apus_bf16_gemv_mt(const uint16_t *w, const uint16_t *x, float *xf,
                       uint16_t *y, size_t O, size_t K);
/* xf: scratch, M*K floats. Bitwise == apus_bf16_gemm_neon (or scalar)
 * for any APUS_THREADS. */
void apus_bf16_gemm_mt(const uint16_t *w, const uint16_t *x, float *xf,
                       uint16_t *y, size_t M, size_t O, size_t K);

/* --- M9b ILP kernels (ADDITIVE — the user-approved bounded REORDER
 * class, 2026-08-07; the scalar anchor and the bitwise NEON kernels
 * above are the frozen contract reference and are what test-m3 pins).
 *
 * These kernels change ONLY the FP32 summation order inside each dot,
 * in a fixed documented way: per output row, four float32x4 vector
 * accumulators acc[0..3]; per 32-wide chunk (ascending chunk order),
 * the eight product vectors p_0..p_7 (each a single vmulq rounding,
 * exactly as the bitwise kernel stages them) are accumulated as
 * acc[q & 3] += p_q for q ascending 0..7. Row end:
 *   s     = (acc[0] + acc[1]) + (acc[2] + acc[3])   (vector adds)
 *   total = (s[0] + s[1]) + (s[2] + s[3])           (scalar lanes)
 * and any K tail (< 32) is appended with scalar ascending adds after
 * the combine. Every output row of the GEMM computes the identical
 * sequence as the GEMV (M-independence). Measured err/esc vs FP64
 * truth on all real shapes in tests/m9b (bound 2e-5, the m3 class). */
void apus_bf16_gemv_ilp_neon(const uint16_t *w, const uint16_t *x,
                             float *xf, uint16_t *y, size_t O, size_t K);
void apus_bf16_gemm_ilp_neon(const uint16_t *w, const uint16_t *x,
                             float *xf, uint16_t *y, size_t M, size_t O,
                             size_t K);

/* --- hot-path wrappers (M6c; ADDITIVE — every contract above is frozen) ---
 * The mt kernels with their widen scratch taken from the c/pool.h TLS
 * arena instead of a caller buffer. BITWISE identical to the scalar
 * anchor at every APUS_THREADS (the m3 mt contract); off NEON the mt
 * path IS the scalar anchor. On arena failure they fall back to the
 * scalar anchor. M9b: the NEON dispatch is the ILP kernel above (the
 * documented reorder class), row-partitioned over the pool — the
 * partition is a fixed function of (O,K), so T=1/4/8 stays BITWISE
 * identical within the path. M12a-2: the x86 dispatch is the mt variant
 * on the c/x86.h AVX2 rows (BITWISE == the scalar anchor at every
 * APUS_THREADS — the x86 hot path consumes NO reorder class).
 * M10: the hooked variants try the optional Metal backend first
 * (apus_backend_hooks — shaders replicate the ILP NEON kernels' rounding
 * sequences EXACTLY, so a taken hook is BITWISE identical to the CPU
 * dispatch; per-op fail-soft to the _cpu entry points). The _cpu entry
 * points skip the hook — c/cache.h's tiered expert calls use them
 * because store slabs are transient (the stable-pointer invariant). */
static inline void apus_bf16_gemv_hot_cpu(const uint16_t *w,
                                          const uint16_t *x, uint16_t *y,
                                          size_t O, size_t K) {
#ifdef __ARM_NEON
    float *xf = (float *)apus_scratch_alloc(K * sizeof(float));
    if (xf) {
        apus_bf16_gemv_ilp_neon(w, x, xf, y, O, K);
        return;
    }
#elif APUS_X86
    /* M12a-2: the x86 hot path is the BITWISE-sequential AVX2 kernel
     * (row-partitioned over the pool via the mt variant; == the scalar
     * anchor at every APUS_THREADS — no reorder class on this platform). */
    float *xf = (float *)apus_scratch_alloc(K * sizeof(float));
    if (xf) {
        apus_bf16_gemv_mt(w, x, xf, y, O, K);
        return;
    }
#endif
    apus_bf16_gemv_scalar(w, x, y, O, K);
}
static inline void apus_bf16_gemv_hot(const uint16_t *w, const uint16_t *x,
                                      uint16_t *y, size_t O, size_t K) {
    if (apus_backend_hooks.bf16_gemv
        && !apus_backend_hooks.bf16_gemv(w, x, y, O, K))
        return;
    apus_bf16_gemv_hot_cpu(w, x, y, O, K);
}
static inline void apus_bf16_gemm_hot_cpu(const uint16_t *w,
                                          const uint16_t *x, uint16_t *y,
                                          size_t M, size_t O, size_t K) {
#ifdef __ARM_NEON
    float *xf = (float *)apus_scratch_alloc(M * K * sizeof(float));
    if (xf) {
        apus_bf16_gemm_ilp_neon(w, x, xf, y, M, O, K);
        return;
    }
#elif APUS_X86
    float *xf = (float *)apus_scratch_alloc(M * K * sizeof(float));
    if (xf) {
        apus_bf16_gemm_mt(w, x, xf, y, M, O, K);
        return;
    }
#endif
    apus_bf16_gemm_scalar(w, x, y, M, O, K);
}
static inline void apus_bf16_gemm_hot(const uint16_t *w, const uint16_t *x,
                                      uint16_t *y, size_t M, size_t O,
                                      size_t K) {
    if (apus_backend_hooks.bf16_gemm
        && !apus_backend_hooks.bf16_gemm(w, x, y, M, O, K))
        return;
    apus_bf16_gemm_hot_cpu(w, x, y, M, O, K);
}

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_BF16_IMPLEMENTATION

#include <stdio.h>   /* snprintf (the weak apus_metal_enable stub) */
#include <string.h>

/* M10 backend hook table (c/backend_metal.h): all-NULL = CPU kernels.
 * Filled by apus_metal_enable() (strong definition in c/backend_metal.mm
 * in the metal=1 build). The weak stubs below keep the plain CPU binary
 * (and the Linux build) linking with identical behavior — APUS_METAL=1
 * there prints "not compiled in" and the engine continues on CPU. */
ApusBackendHooks apus_backend_hooks;

__attribute__((weak)) int apus_metal_enable(char *err, size_t errcap) {
    if (err && errcap)
        snprintf(err, errcap, "metal backend not compiled in");
    return -1;
}
__attribute__((weak)) void apus_metal_disable(void) {}
__attribute__((weak)) int apus_metal_is_enabled(void) { return 0; }
__attribute__((weak)) uint64_t apus_metal_bytes_wrapped(void) { return 0; }
__attribute__((weak)) uint64_t apus_metal_bytes_uploaded(void) { return 0; }
__attribute__((weak)) uint64_t apus_metal_dispatches(void) { return 0; }

float apus_bf16_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float x;
    memcpy(&x, &u, 4);
    return x;
}

uint16_t apus_bf16_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)(u >> 16); /* NaN */
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

float apus_bf16_round(float x) {
    return apus_bf16_f32(apus_bf16_bits(x));
}

void apus_bf16_widen_scalar(const uint16_t *b, float *out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = apus_bf16_f32(b[i]);
}

void apus_bf16_gemv_scalar(const uint16_t *w, const uint16_t *x,
                           uint16_t *y, size_t O, size_t K) {
    for (size_t o = 0; o < O; o++) {
        const uint16_t *wr = w + o * K;
        float acc = 0.0f;
        for (size_t k = 0; k < K; k++)
            acc += apus_bf16_f32(wr[k]) * apus_bf16_f32(x[k]);
        y[o] = apus_bf16_bits(acc);
    }
}

void apus_bf16_gemm_scalar(const uint16_t *w, const uint16_t *x,
                           uint16_t *y, size_t M, size_t O, size_t K) {
    for (size_t m = 0; m < M; m++)
        apus_bf16_gemv_scalar(w, x + m * K, y + m * O, O, K);
}

/* -------------------------------------------------------------------------*/
#ifdef __ARM_NEON

void apus_bf16_widen_neon(const uint16_t *b, float *out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t h = vld1q_u16(b + i);
        vst1q_f32(out + i, vreinterpretq_f32_u32(
            vshll_n_u16(vget_low_u16(h), 16)));
        vst1q_f32(out + i + 4, vreinterpretq_f32_u32(
            vshll_n_u16(vget_high_u16(h), 16)));
    }
    for (; i < n; i++)
        out[i] = apus_bf16_f32(b[i]);
}

/* Stage the 32 products f32(wr[k..k+31]) * xf[k..k+31] into st, each with
 * the same single IEEE fp32 rounding as the scalar mul (plain vmulq, never
 * FMA). Widening is exact (vshll 16). */
static inline void apus_bf16_prod_chunk_neon(const uint16_t *wr,
                                             const float *xf, float *st) {
    for (int i = 0; i < (int)APUS_BF16_CHUNK; i += 8) {
        uint16x8_t h = vld1q_u16(wr + i);
        float32x4_t w0 = vreinterpretq_f32_u32(
            vshll_n_u16(vget_low_u16(h), 16));
        float32x4_t w1 = vreinterpretq_f32_u32(
            vshll_n_u16(vget_high_u16(h), 16));
        vst1q_f32(st + i,     vmulq_f32(w0, vld1q_f32(xf + i)));
        vst1q_f32(st + i + 4, vmulq_f32(w1, vld1q_f32(xf + i + 4)));
    }
}

/* One output dot, single sequential chain (row tails). BITWISE the scalar
 * anchor: staged single-rounded products, adds strictly in ascending k. */
static float apus_bf16_dot_neon(const uint16_t *wr, const float *xf,
                                size_t K, float *st) {
    float acc = 0.0f;
    size_t k = 0;
    for (; k + APUS_BF16_CHUNK <= K; k += APUS_BF16_CHUNK) {
        apus_bf16_prod_chunk_neon(wr + k, xf + k, st);
        for (int i = 0; i < (int)APUS_BF16_CHUNK; i++)
            acc += st[i];
    }
    for (; k < K; k++)
        acc += apus_bf16_f32(wr[k]) * xf[k];
    return acc;
}

/* GEMV rows [o0, o1): 8 independent row chains for ILP (each chain IS the
 * scalar sequential sum for its row — no reassociation; M9a: widened from
 * 4 to 8 chains, a pure interleave change — every row keeps its staged
 * products + strictly-ascending adds, so outputs are bitwise identical;
 * gated by test-m3's scalar-vs-NEON bitwise checks and the m6c suite).
 * 4-chain and single-chain tails for the remainder. */
static void apus_bf16_gemv_rows_neon(const uint16_t *w, const float *xf,
                                     uint16_t *y, size_t K,
                                     size_t o0, size_t o1) {
    float st[8][APUS_BF16_CHUNK];
    size_t o = o0;
    for (; o + 8 <= o1; o += 8) {
        const uint16_t *wr[8];
        for (int r = 0; r < 8; r++) wr[r] = w + (o + r) * K;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        size_t k = 0;
        for (; k + APUS_BF16_CHUNK <= K; k += APUS_BF16_CHUNK) {
            apus_bf16_prod_chunk_neon(wr[0] + k, xf + k, st[0]);
            apus_bf16_prod_chunk_neon(wr[1] + k, xf + k, st[1]);
            apus_bf16_prod_chunk_neon(wr[2] + k, xf + k, st[2]);
            apus_bf16_prod_chunk_neon(wr[3] + k, xf + k, st[3]);
            apus_bf16_prod_chunk_neon(wr[4] + k, xf + k, st[4]);
            apus_bf16_prod_chunk_neon(wr[5] + k, xf + k, st[5]);
            apus_bf16_prod_chunk_neon(wr[6] + k, xf + k, st[6]);
            apus_bf16_prod_chunk_neon(wr[7] + k, xf + k, st[7]);
            for (int i = 0; i < (int)APUS_BF16_CHUNK; i++) {
                a0 += st[0][i];
                a1 += st[1][i];
                a2 += st[2][i];
                a3 += st[3][i];
                a4 += st[4][i];
                a5 += st[5][i];
                a6 += st[6][i];
                a7 += st[7][i];
            }
        }
        for (; k < K; k++) {
            a0 += apus_bf16_f32(wr[0][k]) * xf[k];
            a1 += apus_bf16_f32(wr[1][k]) * xf[k];
            a2 += apus_bf16_f32(wr[2][k]) * xf[k];
            a3 += apus_bf16_f32(wr[3][k]) * xf[k];
            a4 += apus_bf16_f32(wr[4][k]) * xf[k];
            a5 += apus_bf16_f32(wr[5][k]) * xf[k];
            a6 += apus_bf16_f32(wr[6][k]) * xf[k];
            a7 += apus_bf16_f32(wr[7][k]) * xf[k];
        }
        y[o + 0] = apus_bf16_bits(a0);
        y[o + 1] = apus_bf16_bits(a1);
        y[o + 2] = apus_bf16_bits(a2);
        y[o + 3] = apus_bf16_bits(a3);
        y[o + 4] = apus_bf16_bits(a4);
        y[o + 5] = apus_bf16_bits(a5);
        y[o + 6] = apus_bf16_bits(a6);
        y[o + 7] = apus_bf16_bits(a7);
    }
    for (; o + 4 <= o1; o += 4) {
        const uint16_t *w0 = w + (o + 0) * K;
        const uint16_t *w1 = w + (o + 1) * K;
        const uint16_t *w2 = w + (o + 2) * K;
        const uint16_t *w3 = w + (o + 3) * K;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        size_t k = 0;
        for (; k + APUS_BF16_CHUNK <= K; k += APUS_BF16_CHUNK) {
            apus_bf16_prod_chunk_neon(w0 + k, xf + k, st[0]);
            apus_bf16_prod_chunk_neon(w1 + k, xf + k, st[1]);
            apus_bf16_prod_chunk_neon(w2 + k, xf + k, st[2]);
            apus_bf16_prod_chunk_neon(w3 + k, xf + k, st[3]);
            for (int i = 0; i < (int)APUS_BF16_CHUNK; i++) {
                a0 += st[0][i];
                a1 += st[1][i];
                a2 += st[2][i];
                a3 += st[3][i];
            }
        }
        for (; k < K; k++) {
            a0 += apus_bf16_f32(w0[k]) * xf[k];
            a1 += apus_bf16_f32(w1[k]) * xf[k];
            a2 += apus_bf16_f32(w2[k]) * xf[k];
            a3 += apus_bf16_f32(w3[k]) * xf[k];
        }
        y[o + 0] = apus_bf16_bits(a0);
        y[o + 1] = apus_bf16_bits(a1);
        y[o + 2] = apus_bf16_bits(a2);
        y[o + 3] = apus_bf16_bits(a3);
    }
    for (; o < o1; o++)
        y[o] = apus_bf16_bits(
            apus_bf16_dot_neon(w + o * K, xf, K, st[0]));
}

/* GEMM rows [o0, o1) for all m: m-groups of 4 (the widened W chunk is
 * shared across the 4 activation rows), single-chain tail group. Every
 * output keeps the scalar anchor's strictly-sequential per-output order, so
 * values are M- and thread-partition-independent. */
static void apus_bf16_gemm_rows_neon(const uint16_t *w, const float *xf,
                                     uint16_t *y, size_t M, size_t O,
                                     size_t K, size_t o0, size_t o1) {
    float st[4][APUS_BF16_CHUNK];
    for (size_t m0 = 0; m0 < M; m0 += 4) {
        size_t mc = M - m0 < 4 ? M - m0 : 4;
        if (mc == 4) {
            const float *x0 = xf + (m0 + 0) * K;
            const float *x1 = xf + (m0 + 1) * K;
            const float *x2 = xf + (m0 + 2) * K;
            const float *x3 = xf + (m0 + 3) * K;
            for (size_t o = o0; o < o1; o++) {
                const uint16_t *wr = w + o * K;
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                size_t k = 0;
                for (; k + APUS_BF16_CHUNK <= K; k += APUS_BF16_CHUNK) {
                    for (int i = 0; i < (int)APUS_BF16_CHUNK; i += 8) {
                        uint16x8_t h = vld1q_u16(wr + k + i);
                        float32x4_t wv0 = vreinterpretq_f32_u32(
                            vshll_n_u16(vget_low_u16(h), 16));
                        float32x4_t wv1 = vreinterpretq_f32_u32(
                            vshll_n_u16(vget_high_u16(h), 16));
                        vst1q_f32(st[0] + i, vmulq_f32(wv0,
                            vld1q_f32(x0 + k + i)));
                        vst1q_f32(st[0] + i + 4, vmulq_f32(wv1,
                            vld1q_f32(x0 + k + i + 4)));
                        vst1q_f32(st[1] + i, vmulq_f32(wv0,
                            vld1q_f32(x1 + k + i)));
                        vst1q_f32(st[1] + i + 4, vmulq_f32(wv1,
                            vld1q_f32(x1 + k + i + 4)));
                        vst1q_f32(st[2] + i, vmulq_f32(wv0,
                            vld1q_f32(x2 + k + i)));
                        vst1q_f32(st[2] + i + 4, vmulq_f32(wv1,
                            vld1q_f32(x2 + k + i + 4)));
                        vst1q_f32(st[3] + i, vmulq_f32(wv0,
                            vld1q_f32(x3 + k + i)));
                        vst1q_f32(st[3] + i + 4, vmulq_f32(wv1,
                            vld1q_f32(x3 + k + i + 4)));
                    }
                    for (int i = 0; i < (int)APUS_BF16_CHUNK; i++) {
                        a0 += st[0][i];
                        a1 += st[1][i];
                        a2 += st[2][i];
                        a3 += st[3][i];
                    }
                }
                for (; k < K; k++) {
                    float wv = apus_bf16_f32(wr[k]);
                    a0 += wv * x0[k];
                    a1 += wv * x1[k];
                    a2 += wv * x2[k];
                    a3 += wv * x3[k];
                }
                y[(m0 + 0) * O + o] = apus_bf16_bits(a0);
                y[(m0 + 1) * O + o] = apus_bf16_bits(a1);
                y[(m0 + 2) * O + o] = apus_bf16_bits(a2);
                y[(m0 + 3) * O + o] = apus_bf16_bits(a3);
            }
        } else {
            for (size_t r = 0; r < mc; r++) {
                const float *xr = xf + (m0 + r) * K;
                for (size_t o = o0; o < o1; o++)
                    y[(m0 + r) * O + o] = apus_bf16_bits(
                        apus_bf16_dot_neon(w + o * K, xr, K, st[0]));
            }
        }
    }
}

void apus_bf16_gemv_neon(const uint16_t *w, const uint16_t *x, float *xf,
                         uint16_t *y, size_t O, size_t K) {
    apus_bf16_widen_neon(x, xf, K);
    apus_bf16_gemv_rows_neon(w, xf, y, K, 0, O);
}

void apus_bf16_gemm_neon(const uint16_t *w, const uint16_t *x, float *xf,
                         uint16_t *y, size_t M, size_t O, size_t K) {
    for (size_t m = 0; m < M; m++)
        apus_bf16_widen_neon(x + m * K, xf + m * K, K);
    apus_bf16_gemm_rows_neon(w, xf, y, M, O, K, 0, O);
}

#endif /* __ARM_NEON */

/* --- threaded variants -----------------------------------------------------*/

typedef struct {
    const uint16_t *w;
    float *xf;          /* widened x: K floats (GEMV) / M*K floats (GEMM) */
    uint16_t *y;
    size_t M, O, K;
} ApusBf16Job;

#ifdef __ARM_NEON
/* Row bodies delegate to the shared NEON row functions, so mt output is
 * bitwise the single-thread NEON kernel for any row partition. */
static void apus_bf16_gemv_neon_rows(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    apus_bf16_gemv_rows_neon(j->w, j->xf, j->y, j->K, o0, o1);
}
static void apus_bf16_gemm_neon_rows(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    apus_bf16_gemm_rows_neon(j->w, j->xf, j->y, j->M, j->O, j->K, o0, o1);
}
#else
/* Off-NEON fallback: the scalar anchor itself over the row range (xf is
 * exact widening, so this is bitwise apus_bf16_gem*_scalar). Also the
 * x86 runtime fallback when the CPU lacks AVX2 (APUS_X86). */
static void apus_bf16_gemv_scalar_rows(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    for (size_t o = o0; o < o1; o++) {
        const uint16_t *wr = j->w + o * j->K;
        float acc = 0.0f;
        for (size_t k = 0; k < j->K; k++)
            acc += apus_bf16_f32(wr[k]) * j->xf[k];
        j->y[o] = apus_bf16_bits(acc);
    }
}
static void apus_bf16_gemm_scalar_rows(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    for (size_t m = 0; m < j->M; m++) {
        const float *xr = j->xf + m * j->K;
        for (size_t o = o0; o < o1; o++) {
            const uint16_t *wr = j->w + o * j->K;
            float acc = 0.0f;
            for (size_t k = 0; k < j->K; k++)
                acc += apus_bf16_f32(wr[k]) * xr[k];
            j->y[m * j->O + o] = apus_bf16_bits(acc);
        }
    }
}
#if APUS_X86
/* M12a-2: the AVX2 row bodies (c/x86.h) — BITWISE == the scalar rows, so
 * the mt output is the scalar anchor's bits at every pool size. */
static void apus_bf16_gemv_x86_rows(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    apus_bf16_gemv_rows_x86(j->w, j->xf, j->y, j->K, o0, o1);
}
static void apus_bf16_gemm_x86_rows(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    apus_bf16_gemm_rows_x86(j->w, j->xf, j->y, j->M, j->O, j->K, o0, o1);
}
#endif
#endif

void apus_bf16_gemv_mt(const uint16_t *w, const uint16_t *x, float *xf,
                       uint16_t *y, size_t O, size_t K) {
    ApusBf16Job job = { w, xf, y, 1, O, K };
#ifdef __ARM_NEON
    apus_bf16_widen_neon(x, xf, K);
    apus_pool_run(O, apus_bf16_gemv_neon_rows, &job);
#elif APUS_X86
    if (apus_x86_have_avx2()) {
        apus_bf16_widen_x86(x, xf, K);
        apus_pool_run(O, apus_bf16_gemv_x86_rows, &job);
    } else {
        apus_bf16_widen_scalar(x, xf, K);
        apus_pool_run(O, apus_bf16_gemv_scalar_rows, &job);
    }
#else
    apus_bf16_widen_scalar(x, xf, K);
    apus_pool_run(O, apus_bf16_gemv_scalar_rows, &job);
#endif
}

void apus_bf16_gemm_mt(const uint16_t *w, const uint16_t *x, float *xf,
                       uint16_t *y, size_t M, size_t O, size_t K) {
    ApusBf16Job job = { w, xf, y, M, O, K };
#ifdef __ARM_NEON
    for (size_t m = 0; m < M; m++)
        apus_bf16_widen_neon(x + m * K, xf + m * K, K);
    apus_pool_run(O, apus_bf16_gemm_neon_rows, &job);
#elif APUS_X86
    if (apus_x86_have_avx2()) {
        for (size_t m = 0; m < M; m++)
            apus_bf16_widen_x86(x + m * K, xf + m * K, K);
        apus_pool_run(O, apus_bf16_gemm_x86_rows, &job);
    } else {
        for (size_t m = 0; m < M; m++)
            apus_bf16_widen_scalar(x + m * K, xf + m * K, K);
        apus_pool_run(O, apus_bf16_gemm_scalar_rows, &job);
    }
#else
    for (size_t m = 0; m < M; m++)
        apus_bf16_widen_scalar(x + m * K, xf + m * K, K);
    apus_pool_run(O, apus_bf16_gemm_scalar_rows, &job);
#endif
}

/* --- M9b ILP kernels (the approved bounded reorder class) -------------*/
#ifdef __ARM_NEON

/* Per-row engine: 4 float32x4 accumulators per row; per 32-chunk
 * (ascending), product vectors p_q (q=0..7, each one vmulq rounding)
 * accumulated acc[q & 3] += p_q; row end combines ((a0+a1)+(a2+a3))
 * vector-wise then ((x0+x1)+(x2+x3)) lane-wise; the K tail (< 32) is
 * appended with scalar ascending adds. Row groups (8 since M9c, was 4 —
 * grouping is pure interleave, the per-row sequence is unchanged, so
 * outputs are bitwise identical; measured in tests/m9c/bench_m9c.c) and
 * single-row tail groups compute the IDENTICAL per-row sequence
 * (M/O independence). */
static inline float apus_bf16_dot_ilp_neon(const uint16_t *wr,
                                           const float *xf, size_t K) {
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
            int q = i / 4;              /* two vectors per 8: q, q+1 */
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

static void apus_bf16_gemv_ilp_rows_neon(const uint16_t *w,
                                         const float *xf, uint16_t *y,
                                         size_t K, size_t o0, size_t o1) {
    /* M9c: 8-row groups (bench_m9c V2 — measured +14..+55% on the real
     * decode shapes, single-thread rows). Every row keeps the IDENTICAL
     * per-row accumulation sequence as the 4-row form (grouping is pure
     * interleave), so outputs are bitwise unchanged; 4-row and single-row
     * tails for the remainder. */
    size_t o = o0;
    for (; o + 8 <= o1; o += 8) {
        const uint16_t *wrs[8];
        for (int r = 0; r < 8; r++) wrs[r] = w + (o + r) * K;
        float32x4_t a[8][4];
        for (int r = 0; r < 8; r++)
            for (int j = 0; j < 4; j++) a[r][j] = vdupq_n_f32(0.0f);
        size_t k = 0;
        for (; k + 32 <= K; k += 32) {
            for (int i = 0; i < 32; i += 8) {
                int q = i / 4;
                for (int r = 0; r < 8; r++) {
                    uint16x8_t h = vld1q_u16(wrs[r] + k + i);
                    float32x4_t w0 = vreinterpretq_f32_u32(
                        vshll_n_u16(vget_low_u16(h), 16));
                    float32x4_t w1 = vreinterpretq_f32_u32(
                        vshll_n_u16(vget_high_u16(h), 16));
                    a[r][q & 3] = vaddq_f32(a[r][q & 3],
                        vmulq_f32(w0, vld1q_f32(xf + k + i)));
                    a[r][(q + 1) & 3] = vaddq_f32(a[r][(q + 1) & 3],
                        vmulq_f32(w1, vld1q_f32(xf + k + i + 4)));
                }
            }
        }
        for (int r = 0; r < 8; r++) {
            float32x4_t s = vaddq_f32(vaddq_f32(a[r][0], a[r][1]),
                                      vaddq_f32(a[r][2], a[r][3]));
            float total = (vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1))
                        + (vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3));
            for (size_t kk = k; kk < K; kk++)
                total += apus_bf16_f32(wrs[r][kk]) * xf[kk];
            y[o + r] = apus_bf16_bits(total);
        }
    }
    for (; o + 4 <= o1; o += 4) {
        const uint16_t *wrs[4];
        for (int r = 0; r < 4; r++) wrs[r] = w + (o + r) * K;
        float32x4_t a[4][4];
        for (int r = 0; r < 4; r++)
            for (int j = 0; j < 4; j++) a[r][j] = vdupq_n_f32(0.0f);
        size_t k = 0;
        for (; k + 32 <= K; k += 32) {
            for (int i = 0; i < 32; i += 8) {
                int q = i / 4;
                for (int r = 0; r < 4; r++) {
                    uint16x8_t h = vld1q_u16(wrs[r] + k + i);
                    float32x4_t w0 = vreinterpretq_f32_u32(
                        vshll_n_u16(vget_low_u16(h), 16));
                    float32x4_t w1 = vreinterpretq_f32_u32(
                        vshll_n_u16(vget_high_u16(h), 16));
                    a[r][q & 3] = vaddq_f32(a[r][q & 3],
                        vmulq_f32(w0, vld1q_f32(xf + k + i)));
                    a[r][(q + 1) & 3] = vaddq_f32(a[r][(q + 1) & 3],
                        vmulq_f32(w1, vld1q_f32(xf + k + i + 4)));
                }
            }
        }
        for (int r = 0; r < 4; r++) {
            float32x4_t s = vaddq_f32(vaddq_f32(a[r][0], a[r][1]),
                                      vaddq_f32(a[r][2], a[r][3]));
            float total = (vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1))
                        + (vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3));
            for (size_t kk = k; kk < K; kk++)
                total += apus_bf16_f32(wrs[r][kk]) * xf[kk];
            y[o + r] = apus_bf16_bits(total);
        }
    }
    for (; o < o1; o++)
        y[o] = apus_bf16_bits(
            apus_bf16_dot_ilp_neon(w + o * K, xf, K));
}

/* GEMM rows: every output (m,o) computes the identical sequence as the
 * GEMV row o (M-independence). 4 activation rows per output row share
 * each widened weight vector (single load, 4 single-rounded products). */
static void apus_bf16_gemm_ilp_rows_neon(const uint16_t *w,
                                         const float *xf, uint16_t *y,
                                         size_t M, size_t O, size_t K,
                                         size_t o0, size_t o1) {
    (void)O;
    for (size_t o = o0; o < o1; o++) {
        const uint16_t *wr = w + o * K;
        for (size_t m0 = 0; m0 < M; m0 += 4) {
            size_t mc = M - m0 < 4 ? M - m0 : 4;
            float32x4_t a[4][4];
            for (size_t r = 0; r < 4; r++)
                for (int j = 0; j < 4; j++) a[r][j] = vdupq_n_f32(0.0f);
            size_t k = 0;
            for (; k + 32 <= K; k += 32) {
                for (int i = 0; i < 32; i += 8) {
                    int q = i / 4;
                    uint16x8_t h = vld1q_u16(wr + k + i);
                    float32x4_t w0 = vreinterpretq_f32_u32(
                        vshll_n_u16(vget_low_u16(h), 16));
                    float32x4_t w1 = vreinterpretq_f32_u32(
                        vshll_n_u16(vget_high_u16(h), 16));
                    for (size_t r = 0; r < mc; r++) {
                        const float *xr = xf + (m0 + r) * K;
                        a[r][q & 3] = vaddq_f32(a[r][q & 3],
                            vmulq_f32(w0, vld1q_f32(xr + k + i)));
                        a[r][(q + 1) & 3] = vaddq_f32(a[r][(q + 1) & 3],
                            vmulq_f32(w1, vld1q_f32(xr + k + i + 4)));
                    }
                }
            }
            for (size_t r = 0; r < mc; r++) {
                const float *xr = xf + (m0 + r) * K;
                float32x4_t s = vaddq_f32(vaddq_f32(a[r][0], a[r][1]),
                                          vaddq_f32(a[r][2], a[r][3]));
                float total = (vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1))
                            + (vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3));
                for (size_t kk = k; kk < K; kk++)
                    total += apus_bf16_f32(wr[kk]) * xr[kk];
                y[(m0 + r) * O + o] = apus_bf16_bits(total);
            }
        }
    }
}

static void apus_bf16_gemv_ilp_rows_job(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    apus_bf16_gemv_ilp_rows_neon(j->w, j->xf, j->y, j->K, o0, o1);
}
static void apus_bf16_gemm_ilp_rows_job(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    apus_bf16_gemm_ilp_rows_neon(j->w, j->xf, j->y, j->M, j->O, j->K,
                                 o0, o1);
}

void apus_bf16_gemv_ilp_neon(const uint16_t *w, const uint16_t *x,
                             float *xf, uint16_t *y, size_t O, size_t K) {
    apus_bf16_widen_neon(x, xf, K);
    ApusBf16Job job = { w, xf, y, 1, O, K };
    apus_pool_run(O, apus_bf16_gemv_ilp_rows_job, &job);
}

void apus_bf16_gemm_ilp_neon(const uint16_t *w, const uint16_t *x,
                             float *xf, uint16_t *y, size_t M, size_t O,
                             size_t K) {
    for (size_t m = 0; m < M; m++)
        apus_bf16_widen_neon(x + m * K, xf + m * K, K);
    ApusBf16Job job = { w, xf, y, M, O, K };
    apus_pool_run(O, apus_bf16_gemm_ilp_rows_job, &job);
}

#endif /* __ARM_NEON (M9b ILP) */

#endif /* APUS_BF16_IMPLEMENTATION */
#endif /* APUS_BF16_H */
