/*
 * c/x86.h — M12: AVX2 kernels for the x86-64 hot paths, runtime-dispatched.
 * Ported from the Apus M12a-2 playbook (Apus c/x86.h) and adapted to this
 * engine's kernel set (pure-BF16 checkpoint: no FP8/FP4 expansion here).
 *
 * Kernel disposition (the M12 re-anchor, after the M4 model retarget):
 *
 *   LIVE AVX2 (bitwise == the scalar anchors, dispatched from c/bf16.h,
 *   c/moe.h, c/model.h):
 *     - widen / narrow (exact: 16-bit shift / integer-lane RNE bit-trick);
 *     - BF16 GEMV / GEMM row bodies (+ the mt/hot wrappers in c/bf16.h) —
 *       shape-generic: they carry EVERY dense Qwen projection (GDN qkv/z/
 *       b/a/out, GQA wq/wk/wv/wo), the eager + tiered expert gate_up/down
 *       (via c/cache.h's hot paths), the shared expert, the router scoring
 *       (apus_moe_route rides the gemv path), and the lm_head;
 *     - the MoE fp32-out matvec (c/moe.h) — generic machinery, pinned at
 *       kernel level regardless of model callers.
 *   DOCUMENTED SCALAR FALLBACK on x86 (no AVX2 port — the base's rule:
 *   scalar fallback is always legal, just slower):
 *     - the GDN ops (c/gdn.h: conv1d, l2norm, the recurrence step, the
 *       gated output norm) and the gated-GQA row + output gate (c/attn.h):
 *       fp32-state, libm-heavy, per-element-dominated kernels whose SIMD
 *       payoff is small against the GEMV-dominated hot path; on ARM they
 *       carry bitwise NEON variants, on x86 the scalar anchors run
 *       directly. tests/m12 gates that the fallback is what executes
 *       (the AVX2 hit counter does not move across them).
 *   DELETED at the M12 re-anchor: the Ling KDA recurrence step and
 *   gated-MLA row AVX2 kernels (stale since M4 — their scalar anchors
 *   c/kda.h / the Ling c/attn.h were deleted with the Qwen retarget; they
 *   were unreachable and are now gone, not merely quarantined).
 *
 * Design contract (see tests/m12/README.md):
 *
 *   Every AVX2 kernel in this tree is BITWISE IDENTICAL to the normative
 *   scalar kernel it replaces — the scalar anchors (c/bf16.h, c/moe.h) are
 *   the semantic definition on x86, and the AVX2 paths keep their exact
 *   rounding sequence:
 *     - BF16 widening is a pure 16-bit shift (EXACT, incl. subnormals,
 *       inf, NaN payloads); narrowing is the scalar RNE bit-trick in
 *       32-bit integer lanes (add 0x7FFF + ((u>>16)&1), truncate, NaN
 *       passthrough) — integer ops, no FP rounding involved;
 *     - per-element products are computed 8-wide (one IEEE fp32 mul per
 *       element, identical to the scalar mul's single rounding — NO FMA
 *       anywhere: the scalar anchors use separate mul + add, two
 *       roundings, and -ffp-contract=off keeps it that way);
 *     - every reduction keeps the scalar SEQUENTIAL order: independent
 *       output rows (GEMV/GEMM, incl. the fp32-out MoE matvec) are
 *       interleaved as separate accumulator chains — each
 *       chain is the scalar loop for its output, no reassociation.
 *   Consequence: on x86 the AVX2 paths reproduce the scalar anchor's bits
 *   exactly, dispatch is numerics-neutral, and every within-platform gate
 *   (scalar anchors, thread-count digests, tolerance classes) holds
 *   unchanged. NOTE the platform asymmetry (by design): the ARM hot path
 *   is the M9b ILP REORDER kernel (a documented, bounded reorder class);
 *   the x86 hot path is this bitwise-sequential AVX2 kernel. Digests are
 *   WITHIN-platform only — cross-platform bit-identity is not required.
 *
 * IEEE-propagation scope (matches the c/bf16.h invariants, one carve-
 * out): inf/NaN widen and narrow exactly, 0*inf yields the x86 default
 * QNaN 0xFFC00000 in every path (so NaNs arising INSIDE a dot propagate
 * deterministically), accumulation overflow rounds to +-inf. The ONE
 * case not covered bitwise is the payload of NaN-CODE inputs in mixed
 * NaN arithmetic: x86 NaN*NaN payload selection is operand-order-
 * dependent and the compiler may commute a SIMD mul's operands (gcc -O2
 * observed), and the accumulator's NaN+NaN payload selection shifts
 * across opt levels (this engine's tests/m12 caught payload-only
 * divergences at -O1+UBSan and -O2+vectorizers). Normative inputs are
 * finite BF16 (c/bf16.h), so this is outside the normative domain;
 * tests/m12's specials fill excludes NaN codes as inputs (NaN-code
 * coverage: the exhaustive widen and the narrow sweeps) and documents
 * the carve-out.
 *
 * Dispatch: per-function __attribute__((target("avx2"))) — no global
 * -mavx2, the binary still runs on baseline x86-64. Callers check
 * apus_x86_have_avx2() once per call and fall back to the scalar path.
 * APUS_X86_DISABLE=1 forces the scalar paths (bench/debug).
 *
 * The accumulators and row pointers are NAMED variables, never arrays
 * indexed by the loop variable: indexed acc[] spills to memory in the
 * inner loop (measured ~3x slower under Rosetta translation — the Apus
 * M12a-2 finding, tests/m12/README.md of that project).
 *
 * Everything is static inline (single-TU header pattern like c/pool.h,
 * c/blas.h). Self-contained: the scalar tails use the local apus_x86_f32 /
 * apus_x86_bits helpers (verbatim copies of the c/bf16.h anchor ops) so
 * this header does NOT depend on c/bf16.h include order. APUS_X86 == 0
 * off x86-64: the header compiles to nothing.
 */

#ifndef APUS_X86_H
#define APUS_X86_H

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define APUS_X86 1
#else
#define APUS_X86 0
#endif

#if APUS_X86

#include <immintrin.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>   /* getenv */
#include <string.h>   /* memcpy */

#define APUS_TGT_AVX2 __attribute__((target("avx2")))

/* Runtime AVX2 gate (cached; the write is idempotent, so the race is
 * benign). AVX2 alone suffices — no FMA is used (see header note). */
static inline int apus_x86_have_avx2(void) {
    static int cached = -1;
    if (cached < 0)
        cached = __builtin_cpu_supports("avx2")
              && !getenv("APUS_X86_DISABLE");
    return cached;
}

/* AVX2-path activity counter (tests/m12 "the AVX2 path was taken HERE"
 * probe). Incremented once per dispatched worker invocation. */
static _Atomic unsigned long apus_x86_hits;
static inline unsigned long apus_x86_avx2_hits(void) {
    return atomic_load(&apus_x86_hits);
}
static inline void apus_x86_hit(void) {
    atomic_fetch_add_explicit(&apus_x86_hits, 1, memory_order_relaxed);
}

/* --- self-contained scalar bf16 helpers (verbatim c/bf16.h anchor ops) ---*/

static inline float apus_x86_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float x;
    memcpy(&x, &u, 4);
    return x;
}

static inline uint16_t apus_x86_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)(u >> 16); /* NaN */
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

/* --- widen / narrow (EXACT) ----------------------------------------------*/

/* Widen 8 BF16 bit patterns -> 8 FP32 (exact: upper 16 bits). */
APUS_TGT_AVX2
static inline __m256 apus_bf16_widen8_x86(const uint16_t *p) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}

/* Row widen: out[i] = f32(b[i]), i < n. Exact, scalar tail. */
APUS_TGT_AVX2
static inline void apus_bf16_widen_x86(const uint16_t *b, float *out,
                                       size_t n) {
    apus_x86_hit();
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i, apus_bf16_widen8_x86(b + i));
    for (; i < n; i++)
        out[i] = apus_x86_f32(b[i]);
}

/* Narrow 8 FP32 -> 8 BF16 codes: the scalar RNE bit-trick in integer
 * lanes (u += 0x7FFF + ((u>>16)&1), truncate; NaN passes through as the
 * high 16 bits). INTEGER ops only — bitwise == apus_x86_bits per lane. */
APUS_TGT_AVX2
static inline void apus_bf16_narrow8_x86(const float *in, uint16_t *out) {
    __m256i u = _mm256_castps_si256(_mm256_loadu_ps(in));
    __m256i mag = _mm256_and_si256(u, _mm256_set1_epi32(0x7fffffff));
    /* mag <= 0x7fffffff (sign clear): signed compare is the unsigned one */
    __m256i isnan = _mm256_cmpgt_epi32(mag, _mm256_set1_epi32(0x7f800000));
    __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF),
        _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1)));
    __m256i code = _mm256_srli_epi32(_mm256_add_epi32(u, bias), 16);
    __m256i ncode = _mm256_srli_epi32(u, 16);
    __m256i r = _mm256_blendv_epi8(code, ncode, isnan);
    /* codes are 0..0xFFFF (positive int32, no packus saturation); pack
     * lane-wise then fix the 64-bit block order */
    __m256i pk = _mm256_permute4x64_epi64(_mm256_packus_epi32(r, r), 0xD8);
    _mm_storeu_si128((__m128i *)out, _mm256_castsi256_si128(pk));
}

/* --- staged-product sequential dots (the bitwise-scalar SIMD pattern) ----*/

/* k staging granularity (products staged per 32-wide chunk before the
 * strictly-sequential adds) — same granularity as the NEON kernels. */
#define APUS_X86_CHUNK 32u

/* Stage the 32 products f32(wr[k..k+31]) * xf[k..k+31] into st, each with
 * the same single IEEE fp32 rounding as the scalar mul (plain mul, never
 * FMA). Widening is exact. */
APUS_TGT_AVX2
static inline void apus_bf16_prod_chunk_x86(const uint16_t *wr,
                                            const float *xf, float *st) {
    for (int i = 0; i < (int)APUS_X86_CHUNK; i += 8)
        _mm256_storeu_ps(st + i, _mm256_mul_ps(apus_bf16_widen8_x86(wr + i),
                                               _mm256_loadu_ps(xf + i)));
}

/* One output dot, single sequential chain (row tails). BITWISE the scalar
 * anchor: staged single-rounded products, adds strictly in ascending k. */
APUS_TGT_AVX2
static inline float apus_bf16_dot_x86(const uint16_t *wr, const float *xf,
                                      size_t K, float *st) {
    float acc = 0.0f;
    size_t k = 0;
    for (; k + APUS_X86_CHUNK <= K; k += APUS_X86_CHUNK) {
        apus_bf16_prod_chunk_x86(wr + k, xf + k, st);
        for (int i = 0; i < (int)APUS_X86_CHUNK; i++)
            acc += st[i];
    }
    for (; k < K; k++)
        acc += apus_x86_f32(wr[k]) * xf[k];
    return acc;
}

/* GEMV rows [o0, o1): 8 independent row chains for ILP (each chain IS the
 * scalar sequential sum for its row — no reassociation; the same chain
 * structure as the c/bf16.h NEON kernel), 4-chain and single-chain tails.
 * NAMED accumulators (the Rosetta spilling trap, see header). */
APUS_TGT_AVX2
static inline void apus_bf16_gemv_rows_x86(const uint16_t *w,
                                           const float *xf, uint16_t *y,
                                           size_t K, size_t o0, size_t o1) {
    apus_x86_hit();
    float st[8][APUS_X86_CHUNK];
    size_t o = o0;
    for (; o + 8 <= o1; o += 8) {
        const uint16_t *w0 = w + (o + 0) * K;
        const uint16_t *w1 = w + (o + 1) * K;
        const uint16_t *w2 = w + (o + 2) * K;
        const uint16_t *w3 = w + (o + 3) * K;
        const uint16_t *w4 = w + (o + 4) * K;
        const uint16_t *w5 = w + (o + 5) * K;
        const uint16_t *w6 = w + (o + 6) * K;
        const uint16_t *w7 = w + (o + 7) * K;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        size_t k = 0;
        for (; k + APUS_X86_CHUNK <= K; k += APUS_X86_CHUNK) {
            apus_bf16_prod_chunk_x86(w0 + k, xf + k, st[0]);
            apus_bf16_prod_chunk_x86(w1 + k, xf + k, st[1]);
            apus_bf16_prod_chunk_x86(w2 + k, xf + k, st[2]);
            apus_bf16_prod_chunk_x86(w3 + k, xf + k, st[3]);
            apus_bf16_prod_chunk_x86(w4 + k, xf + k, st[4]);
            apus_bf16_prod_chunk_x86(w5 + k, xf + k, st[5]);
            apus_bf16_prod_chunk_x86(w6 + k, xf + k, st[6]);
            apus_bf16_prod_chunk_x86(w7 + k, xf + k, st[7]);
            for (int i = 0; i < (int)APUS_X86_CHUNK; i++) {
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
            a0 += apus_x86_f32(w0[k]) * xf[k];
            a1 += apus_x86_f32(w1[k]) * xf[k];
            a2 += apus_x86_f32(w2[k]) * xf[k];
            a3 += apus_x86_f32(w3[k]) * xf[k];
            a4 += apus_x86_f32(w4[k]) * xf[k];
            a5 += apus_x86_f32(w5[k]) * xf[k];
            a6 += apus_x86_f32(w6[k]) * xf[k];
            a7 += apus_x86_f32(w7[k]) * xf[k];
        }
        float accs[8] = { a0, a1, a2, a3, a4, a5, a6, a7 };
        apus_bf16_narrow8_x86(accs, y + o);
    }
    for (; o + 4 <= o1; o += 4) {
        const uint16_t *w0 = w + (o + 0) * K;
        const uint16_t *w1 = w + (o + 1) * K;
        const uint16_t *w2 = w + (o + 2) * K;
        const uint16_t *w3 = w + (o + 3) * K;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        size_t k = 0;
        for (; k + APUS_X86_CHUNK <= K; k += APUS_X86_CHUNK) {
            apus_bf16_prod_chunk_x86(w0 + k, xf + k, st[0]);
            apus_bf16_prod_chunk_x86(w1 + k, xf + k, st[1]);
            apus_bf16_prod_chunk_x86(w2 + k, xf + k, st[2]);
            apus_bf16_prod_chunk_x86(w3 + k, xf + k, st[3]);
            for (int i = 0; i < (int)APUS_X86_CHUNK; i++) {
                a0 += st[0][i];
                a1 += st[1][i];
                a2 += st[2][i];
                a3 += st[3][i];
            }
        }
        for (; k < K; k++) {
            a0 += apus_x86_f32(w0[k]) * xf[k];
            a1 += apus_x86_f32(w1[k]) * xf[k];
            a2 += apus_x86_f32(w2[k]) * xf[k];
            a3 += apus_x86_f32(w3[k]) * xf[k];
        }
        y[o + 0] = apus_x86_bits(a0);
        y[o + 1] = apus_x86_bits(a1);
        y[o + 2] = apus_x86_bits(a2);
        y[o + 3] = apus_x86_bits(a3);
    }
    for (; o < o1; o++)
        y[o] = apus_x86_bits(
            apus_bf16_dot_x86(w + o * K, xf, K, st[0]));
}

/* GEMM rows [o0, o1) for all m: m-groups of 4 (the widened W chunk is
 * shared across the 4 activation rows), single-chain tail group. Every
 * output keeps the scalar anchor's strictly-sequential per-output order,
 * so values are M- and thread-partition-independent. */
APUS_TGT_AVX2
static inline void apus_bf16_gemm_rows_x86(const uint16_t *w,
                                           const float *xf, uint16_t *y,
                                           size_t M, size_t O, size_t K,
                                           size_t o0, size_t o1) {
    apus_x86_hit();
    float st[4][APUS_X86_CHUNK];
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
                for (; k + APUS_X86_CHUNK <= K; k += APUS_X86_CHUNK) {
                    for (int i = 0; i < (int)APUS_X86_CHUNK; i += 8) {
                        __m256 wv = apus_bf16_widen8_x86(wr + k + i);
                        _mm256_storeu_ps(st[0] + i, _mm256_mul_ps(wv,
                            _mm256_loadu_ps(x0 + k + i)));
                        _mm256_storeu_ps(st[1] + i, _mm256_mul_ps(wv,
                            _mm256_loadu_ps(x1 + k + i)));
                        _mm256_storeu_ps(st[2] + i, _mm256_mul_ps(wv,
                            _mm256_loadu_ps(x2 + k + i)));
                        _mm256_storeu_ps(st[3] + i, _mm256_mul_ps(wv,
                            _mm256_loadu_ps(x3 + k + i)));
                    }
                    for (int i = 0; i < (int)APUS_X86_CHUNK; i++) {
                        a0 += st[0][i];
                        a1 += st[1][i];
                        a2 += st[2][i];
                        a3 += st[3][i];
                    }
                }
                for (; k < K; k++) {
                    float wv = apus_x86_f32(wr[k]);
                    a0 += wv * x0[k];
                    a1 += wv * x1[k];
                    a2 += wv * x2[k];
                    a3 += wv * x3[k];
                }
                y[(m0 + 0) * O + o] = apus_x86_bits(a0);
                y[(m0 + 1) * O + o] = apus_x86_bits(a1);
                y[(m0 + 2) * O + o] = apus_x86_bits(a2);
                y[(m0 + 3) * O + o] = apus_x86_bits(a3);
            }
        } else {
            for (size_t r = 0; r < mc; r++) {
                const float *xr = xf + (m0 + r) * K;
                for (size_t o = o0; o < o1; o++)
                    y[(m0 + r) * O + o] = apus_x86_bits(
                        apus_bf16_dot_x86(w + o * K, xr, K, st[0]));
            }
        }
    }
}

/* Single-thread entries (xf: caller widen scratch, K / M*K floats). */
static inline void apus_bf16_gemv_x86(const uint16_t *w, const uint16_t *x,
                                      float *xf, uint16_t *y,
                                      size_t O, size_t K) {
    apus_bf16_widen_x86(x, xf, K);
    apus_bf16_gemv_rows_x86(w, xf, y, K, 0, O);
}
static inline void apus_bf16_gemm_x86(const uint16_t *w, const uint16_t *x,
                                      float *xf, uint16_t *y,
                                      size_t M, size_t O, size_t K) {
    for (size_t m = 0; m < M; m++)
        apus_bf16_widen_x86(x + m * K, xf + m * K, K);
    apus_bf16_gemm_rows_x86(w, xf, y, M, O, K, 0, O);
}

/* --- MoE router matvec (fp32 out — c/moe.h op (0)) ------------------------*/

/* One fp32-out dot, single sequential chain (BITWISE c/moe.h's anchor). */
APUS_TGT_AVX2
static inline float apus_moe_dot_x86(const uint16_t *wr, const float *xf,
                                     size_t K, float *st) {
    float acc = 0.0f;
    size_t k = 0;
    for (; k + APUS_X86_CHUNK <= K; k += APUS_X86_CHUNK) {
        apus_bf16_prod_chunk_x86(wr + k, xf + k, st);
        for (int i = 0; i < (int)APUS_X86_CHUNK; i++)
            acc += st[i];
    }
    for (; k < K; k++)
        acc += apus_x86_f32(wr[k]) * xf[k];
    return acc;
}

/* Rows [o0, o1): 4 independent row chains (each the scalar sequential sum
 * for its row), single-chain tail. fp32 out, NO final narrowing. */
APUS_TGT_AVX2
static inline void apus_moe_matvec_rows_x86(const uint16_t *w,
                                            const float *xf, float *y,
                                            size_t K, size_t o0, size_t o1) {
    apus_x86_hit();
    float st[4][APUS_X86_CHUNK];
    size_t o = o0;
    for (; o + 4 <= o1; o += 4) {
        const uint16_t *w0 = w + (o + 0) * K;
        const uint16_t *w1 = w + (o + 1) * K;
        const uint16_t *w2 = w + (o + 2) * K;
        const uint16_t *w3 = w + (o + 3) * K;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        size_t k = 0;
        for (; k + APUS_X86_CHUNK <= K; k += APUS_X86_CHUNK) {
            apus_bf16_prod_chunk_x86(w0 + k, xf + k, st[0]);
            apus_bf16_prod_chunk_x86(w1 + k, xf + k, st[1]);
            apus_bf16_prod_chunk_x86(w2 + k, xf + k, st[2]);
            apus_bf16_prod_chunk_x86(w3 + k, xf + k, st[3]);
            for (int i = 0; i < (int)APUS_X86_CHUNK; i++) {
                a0 += st[0][i];
                a1 += st[1][i];
                a2 += st[2][i];
                a3 += st[3][i];
            }
        }
        for (; k < K; k++) {
            a0 += apus_x86_f32(w0[k]) * xf[k];
            a1 += apus_x86_f32(w1[k]) * xf[k];
            a2 += apus_x86_f32(w2[k]) * xf[k];
            a3 += apus_x86_f32(w3[k]) * xf[k];
        }
        y[o + 0] = a0;
        y[o + 1] = a1;
        y[o + 2] = a2;
        y[o + 3] = a3;
    }
    for (; o < o1; o++)
        y[o] = apus_moe_dot_x86(w + o * K, xf, K, st[0]);
}

#endif /* APUS_X86 */
#endif /* APUS_X86_H */
