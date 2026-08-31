/*
 * c/moe.h — MoE per-operation kernels for Qwen3.6-35B-A3B, milestone
 * M4. Replaces the Ling sigmoid/grouped-topk router and the clamped
 * SwiGLU variants. C11, libc/libm only. Tier discipline (the engine
 * rule): the scalar anchors are the normative semantic definition; the
 * __ARM_NEON variants are BITWISE identical to them (proven per-op in
 * tests/m4a), and the plain entry points dispatch to NEON when compiled
 * in. Follows docs/M4-CONTRACT.md section 5 and tools/oracle.py
 * (router_topk / moe_forward).
 *
 * All tensors BF16 storage unless noted; fp32 is the only compute type.
 * "rnd" is apus_bf16_bits (RNE narrow, c/bf16.h — the approved M3
 * anchor). -ffp-contract=off project-wide: every mul/add below is one
 * IEEE fp32 rounding, in the written order.
 *
 * Numerics contract (normative):
 *
 *   (0) fp32-OUTPUT matvec: identical accumulation to the c/bf16.h
 *       scalar anchor (acc = 0; for k ascending: acc +=
 *       f32(W[o,k])*f32(x[k]) — two roundings per element, no FMA) but
 *       the output is the raw fp32 accumulator, NOT narrowed to bf16.
 *       Generic engine machinery (the M10 Metal hook table references
 *       it); the Qwen router itself rounds its logits to bf16 (a), so
 *       this op is NOT on the router path anymore.
 *   (a) Router (HF:837-853; contract §5; oracle router_topk):
 *         logits = rnd(W_gate . x)        [E] BF16 (c/bf16.h GEMV)
 *         m = max_e f32(logits[e])  (ascending scan)
 *         e[e] = expf(f32(logits[e]) - m);  s = sum_e e[e] (ascending)
 *         probs[e] = e[e] / s                       (FP32 softmax)
 *         top-k over probs, selection order descending; tie-break:
 *           strict -> replacement => LOWEST index wins (the oracle's
 *           documented rule; goldens avoid exact ties)
 *         w[i] = rnd( probs[idx[i]] / sum_i probs[idx[i]] )  [k] BF16
 *         (renormalize fp32, ascending selection order, then rnd)
 *       NO bias, NO sigmoid, NO scaling factor, NO group-limit, NO
 *       jitter — all of those are Ling-only and do not exist here.
 *   (b) Routed-expert activation (HF:828-830; oracle moe_forward):
 *       fused gate_up output gu [2I] BF16 (gate = gu[:I], up = gu[I:]):
 *         a1[i] = rnd( silu_fp32(f32(gu[i])) )      (silu, rnd)
 *         act[i] = rnd( f32(a1[i]) * f32(gu[I+i]) ) (x up, rnd)
 *       TWO rounding points (torch materializes the silu output bf16).
 *       NO clamps — Ling's SwigluStepAndMul does not exist here.
 *   (b2) Shared-expert activation (HF:793; oracle moe_forward): g,u [I]
 *       BF16, SINGLE rounding:
 *         y[i] = rnd( silu_fp32(f32(g[i])) * f32(u[i]) )
 *       (plain SiLU MLP — a different rounding point count than the
 *       routed experts, replicated from the oracle).
 *   (c) Weighted combine (HF:831-834 via the oracle's DOCUMENTED
 *       realization choice — fp32 accumulate across the top-k experts
 *       with a single bf16 round; HF's index_add_ into a bf16 buffer
 *       rounds per expert, the documented reorder class absorbed by the
 *       m4b/m4c envelope):
 *         out[n] = rnd( sum_e f32(w[e]) * f32(y[e,n]) )
 *       experts in ascending slot order, w the BF16 routing weights
 *       from (a), fp32 throughout, single rnd.
 *
 * Invariants: finite inputs in normative use. libm (expf) is the
 * platform libm's — float64 goldens with documented tolerances, never
 * cross-libm bitwise (tests/m4a/README.md).
 *
 * Usage: #define APUS_MOE_IMPLEMENTATION in exactly one TU. Depends on
 * c/bf16.h (helpers only): that TU must also define
 * APUS_BF16_IMPLEMENTATION.
 */
#ifndef APUS_MOE_H
#define APUS_MOE_H

#include <stddef.h>
#include <stdint.h>

#include "bf16.h"

#ifdef __cplusplus
extern "C" {
#endif

/* (0) fp32-out matvec: y[o] = sum_k f32(w[o,k])*f32(x[k]), sequential
 * ascending k, mul+add (two roundings per element), NO final narrowing.
 * w: [O,K] BF16, x: [K] BF16, y: [O] fp32. */
void apus_moe_matvec_f32(const uint16_t *w, const uint16_t *x, float *y,
                         size_t O, size_t K);

/* (0) hot variants (ADDITIVE — the scalar anchor above is frozen).
 * NEON: BITWISE identical to the scalar anchor (staged single-rounded
 * products, adds strictly ascending k per row, ILP only across
 * independent rows — the c/bf16.h pattern). mt: row-partitioned over
 * the c/pool.h lanes, bitwise at every APUS_THREADS; xf widen scratch
 * comes from the TLS arena. */
#ifdef __ARM_NEON
void apus_moe_matvec_f32_neon(const uint16_t *w, const uint16_t *x,
                              float *xf, float *y, size_t O, size_t K);
#endif
void apus_moe_matvec_f32_hot(const uint16_t *w, const uint16_t *x,
                             float *y, size_t O, size_t K);

/* (a) Router. x: [K] BF16; wg: [E,K] BF16; idx: [topk] int32 out
 * (selection order, descending prob); w: [topk] BF16 out (renormalized
 * weights). Model values: E=256, topk=8. */
void apus_moe_route(const uint16_t *x, const uint16_t *wg,
                    int32_t *idx, uint16_t *w, size_t E, size_t K,
                    size_t topk);

/* (a2) Top-N prediction variant for the pilot/locality measurement: the
 * SAME scoring + selection machinery as apus_moe_route, but returns the
 * first n ids in selection order without computing weights.
 * topn <= topk is BITWISE the route selection. */
void apus_moe_route_topn(const uint16_t *x, const uint16_t *wg,
                         int32_t *idx, size_t n, size_t E, size_t K);

/* (b) Routed-expert activation: gu [2I] BF16 (fused gate|up),
 * act [I] BF16. Two rounding points (see header contract). */
void apus_moe_silu_act(const uint16_t *gu, uint16_t *act, size_t I);
void apus_moe_silu_act_scalar(const uint16_t *gu, uint16_t *act,
                              size_t I);
#ifdef __ARM_NEON
void apus_moe_silu_act_neon(const uint16_t *gu, uint16_t *act, size_t I);
#endif

/* (b2) Shared-expert plain SiLU-gated product: y[i] =
 * rnd(silu(g[i]) * u[i]), SINGLE rounding, fp32 compute from BF16
 * g, u [I]. */
void apus_moe_silu_mul(const uint16_t *g, const uint16_t *u, uint16_t *y,
                       size_t I);
void apus_moe_silu_mul_scalar(const uint16_t *g, const uint16_t *u,
                              uint16_t *y, size_t I);
#ifdef __ARM_NEON
void apus_moe_silu_mul_neon(const uint16_t *g, const uint16_t *u,
                            uint16_t *y, size_t I);
#endif

/* (c) Weighted combine: y [k,N] BF16 (per-expert outputs), w [k] BF16
 * (the router's renormalized weights), out [N] BF16. */
void apus_moe_combine(const uint16_t *y, const uint16_t *w, uint16_t *out,
                      size_t k, size_t N);
void apus_moe_combine_scalar(const uint16_t *y, const uint16_t *w,
                             uint16_t *out, size_t k, size_t N);
#ifdef __ARM_NEON
void apus_moe_combine_neon(const uint16_t *y, const uint16_t *w,
                           uint16_t *out, size_t k, size_t N);
#endif

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_MOE_IMPLEMENTATION

#include <math.h>

void apus_moe_matvec_f32(const uint16_t *w, const uint16_t *x, float *y,
                         size_t O, size_t K) {
    for (size_t o = 0; o < O; o++) {
        const uint16_t *wr = w + o * K;
        float acc = 0.0f;
        for (size_t k = 0; k < K; k++)
            acc += apus_bf16_f32(wr[k]) * apus_bf16_f32(x[k]);
        y[o] = acc;
    }
}

/* -------------------------------------------------------------------------*/
#ifdef __ARM_NEON
#include <arm_neon.h>

/* One output dot, single sequential chain. BITWISE the scalar anchor:
 * staged single-rounded products (plain vmulq), adds strictly ascending. */
static float apus_moe_dot32_neon(const uint16_t *wr, const float *xf,
                                 size_t K, float *st) {
    float acc = 0.0f;
    size_t k = 0;
    for (; k + 32 <= K; k += 32) {
        for (int i = 0; i < 32; i += 8) {
            uint16x8_t h = vld1q_u16(wr + k + i);
            float32x4_t w0 = vreinterpretq_f32_u32(
                vshll_n_u16(vget_low_u16(h), 16));
            float32x4_t w1 = vreinterpretq_f32_u32(
                vshll_n_u16(vget_high_u16(h), 16));
            vst1q_f32(st + i, vmulq_f32(w0, vld1q_f32(xf + k + i)));
            vst1q_f32(st + i + 4, vmulq_f32(w1, vld1q_f32(xf + k + i + 4)));
        }
        for (int i = 0; i < 32; i++)
            acc += st[i];
    }
    for (; k < K; k++)
        acc += apus_bf16_f32(wr[k]) * xf[k];
    return acc;
}

/* Rows [o0, o1): 4 independent row chains for ILP (each chain IS the
 * scalar sequential sum for its row — no reassociation), tail single. */
static void apus_moe_matvec_rows_neon(const uint16_t *w, const float *xf,
                                      float *y, size_t K,
                                      size_t o0, size_t o1) {
    float st[4][32];
    size_t o = o0;
    for (; o + 4 <= o1; o += 4) {
        const uint16_t *w0 = w + (o + 0) * K;
        const uint16_t *w1 = w + (o + 1) * K;
        const uint16_t *w2 = w + (o + 2) * K;
        const uint16_t *w3 = w + (o + 3) * K;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        size_t k = 0;
        for (; k + 32 <= K; k += 32) {
            for (int i = 0; i < 32; i += 8) {
                uint16x8_t h = vld1q_u16(w0 + k + i);
                vst1q_f32(st[0] + i, vmulq_f32(vreinterpretq_f32_u32(
                    vshll_n_u16(vget_low_u16(h), 16)),
                    vld1q_f32(xf + k + i)));
                vst1q_f32(st[0] + i + 4, vmulq_f32(vreinterpretq_f32_u32(
                    vshll_n_u16(vget_high_u16(h), 16)),
                    vld1q_f32(xf + k + i + 4)));
                h = vld1q_u16(w1 + k + i);
                vst1q_f32(st[1] + i, vmulq_f32(vreinterpretq_f32_u32(
                    vshll_n_u16(vget_low_u16(h), 16)),
                    vld1q_f32(xf + k + i)));
                vst1q_f32(st[1] + i + 4, vmulq_f32(vreinterpretq_f32_u32(
                    vshll_n_u16(vget_high_u16(h), 16)),
                    vld1q_f32(xf + k + i + 4)));
                h = vld1q_u16(w2 + k + i);
                vst1q_f32(st[2] + i, vmulq_f32(vreinterpretq_f32_u32(
                    vshll_n_u16(vget_low_u16(h), 16)),
                    vld1q_f32(xf + k + i)));
                vst1q_f32(st[2] + i + 4, vmulq_f32(vreinterpretq_f32_u32(
                    vshll_n_u16(vget_high_u16(h), 16)),
                    vld1q_f32(xf + k + i + 4)));
                h = vld1q_u16(w3 + k + i);
                vst1q_f32(st[3] + i, vmulq_f32(vreinterpretq_f32_u32(
                    vshll_n_u16(vget_low_u16(h), 16)),
                    vld1q_f32(xf + k + i)));
                vst1q_f32(st[3] + i + 4, vmulq_f32(vreinterpretq_f32_u32(
                    vshll_n_u16(vget_high_u16(h), 16)),
                    vld1q_f32(xf + k + i + 4)));
            }
            for (int i = 0; i < 32; i++) {
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
        y[o + 0] = a0;
        y[o + 1] = a1;
        y[o + 2] = a2;
        y[o + 3] = a3;
    }
    for (; o < o1; o++)
        y[o] = apus_moe_dot32_neon(w + o * K, xf, K, st[0]);
}

void apus_moe_matvec_f32_neon(const uint16_t *w, const uint16_t *x,
                              float *xf, float *y, size_t O, size_t K) {
    apus_bf16_widen_neon(x, xf, K);
    apus_moe_matvec_rows_neon(w, xf, y, K, 0, O);
}
#endif /* __ARM_NEON */

typedef struct {
    const uint16_t *w;
    const float *xf;
    float *y;
    size_t K;
} ApusMoeMvJob;

#ifdef __ARM_NEON
static void apus_moe_matvec_neon_rows(void *vjob, size_t o0, size_t o1) {
    const ApusMoeMvJob *j = vjob;
    apus_moe_matvec_rows_neon(j->w, j->xf, j->y, j->K, o0, o1);
}
#elif APUS_X86
/* M12a-2: the AVX2 row body (c/x86.h) — BITWISE == the scalar anchor. */
static void apus_moe_matvec_x86_rows(void *vjob, size_t o0, size_t o1) {
    const ApusMoeMvJob *j = vjob;
    apus_moe_matvec_rows_x86(j->w, j->xf, j->y, j->K, o0, o1);
}
#endif

void apus_moe_matvec_f32_hot(const uint16_t *w, const uint16_t *x,
                             float *y, size_t O, size_t K) {
    /* M10: try the Metal backend first (the shader replicates the NEON
     * kernel's rounding sequence EXACTLY — bitwise when taken), fail-soft
     * to the CPU kernels. */
    if (apus_backend_hooks.bf16_matvec_f32
        && !apus_backend_hooks.bf16_matvec_f32(w, x, y, O, K))
        return;
#ifdef __ARM_NEON
    float *xf = (float *)apus_scratch_alloc(K * sizeof(float));
    if (xf) {
        ApusMoeMvJob job = { w, xf, y, K };
        apus_bf16_widen_neon(x, xf, K);
        apus_pool_run(O, apus_moe_matvec_neon_rows, &job);
        return;
    }
#elif APUS_X86
    float *xf = (float *)apus_scratch_alloc(K * sizeof(float));
    if (xf && apus_x86_have_avx2()) {
        ApusMoeMvJob job = { w, xf, y, K };
        apus_bf16_widen_x86(x, xf, K);
        apus_pool_run(O, apus_moe_matvec_x86_rows, &job);
        return;
    }
#endif
    apus_moe_matvec_f32(w, x, y, O, K);
}

static inline float apus_moe_sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

/* --- (a) router -----------------------------------------------------------*/

/* FP32 softmax over the bf16 router logits + top-k selection (lowest
 * index on ties), shared by route and route_topn. logits [E] BF16 in,
 * probs [E] fp32 out. */
static void apus_moe_router_probs(const uint16_t *logits, float *probs,
                                  size_t E) {
    float m = apus_bf16_f32(logits[0]);
    for (size_t e = 1; e < E; e++) {
        float a = apus_bf16_f32(logits[e]);
        if (a > m) m = a;
    }
    float s = 0.0f;
    for (size_t e = 0; e < E; e++) {
        probs[e] = expf(apus_bf16_f32(logits[e]) - m);
        s += probs[e];
    }
    for (size_t e = 0; e < E; e++)
        probs[e] = probs[e] / s;
}

/* Top-n over probs, selection order descending; strict-> replacement =>
 * lowest index wins ties (the oracle's documented rule). */
static void apus_moe_router_select(const float *probs, int32_t *idx,
                                   size_t n, size_t E) {
    uint8_t sel[E];
    for (size_t e = 0; e < E; e++) sel[e] = 0;
    for (size_t i = 0; i < n; i++) {
        size_t best = 0;
        float bv = -INFINITY;
        for (size_t e = 0; e < E; e++) {
            if (!sel[e] && probs[e] > bv) {
                bv = probs[e];
                best = e;
            }
        }
        sel[best] = 1;
        idx[i] = (int32_t)best;
    }
}

void apus_moe_route(const uint16_t *x, const uint16_t *wg,
                    int32_t *idx, uint16_t *w, size_t E, size_t K,
                    size_t topk) {
    uint16_t logits[E];
    float probs[E];
    apus_bf16_gemv_hot(wg, x, logits, E, K);
    apus_moe_router_probs(logits, probs, E);
    apus_moe_router_select(probs, idx, topk, E);
    /* renormalize in fp32 (ascending selection order), rnd to bf16 */
    float sw = 0.0f;
    for (size_t i = 0; i < topk; i++)
        sw += probs[idx[i]];
    for (size_t i = 0; i < topk; i++)
        w[i] = apus_bf16_bits(probs[idx[i]] / sw);
}

void apus_moe_route_topn(const uint16_t *x, const uint16_t *wg,
                         int32_t *idx, size_t n, size_t E, size_t K) {
    /* identical scoring + selection as apus_moe_route (see it for the
     * contract); only the truncation point and the weight computation
     * differ. Regression gate: topn(topk) == route's selection bitwise. */
    uint16_t logits[E];
    float probs[E];
    apus_bf16_gemv_hot(wg, x, logits, E, K);
    apus_moe_router_probs(logits, probs, E);
    apus_moe_router_select(probs, idx, n, E);
}

/* --- (b) routed-expert activation (two rounding points) -------------------*/

void apus_moe_silu_act_scalar(const uint16_t *gu, uint16_t *act,
                              size_t I) {
    for (size_t i = 0; i < I; i++) {
        float gv = apus_bf16_f32(gu[i]);
        uint16_t a1 = apus_bf16_bits(gv * apus_moe_sigmoid(gv));
        act[i] = apus_bf16_bits(apus_bf16_f32(a1) *
                                apus_bf16_f32(gu[I + i]));
    }
}

#ifdef __ARM_NEON
/* BITWISE the scalar activation: widen is exact; the silu fp32 chain,
 * both rnd points per lane with the scalar helpers; the final mul is
 * one vmulq rounding per element. */
void apus_moe_silu_act_neon(const uint16_t *gu, uint16_t *act, size_t I) {
    size_t i = 0;
    for (; i + 4 <= I; i += 4) {
        float gf[4];
        vst1q_f32(gf, vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(gu + i), 16)));
        uint16_t a1[4];
        for (int l = 0; l < 4; l++)
            a1[l] = apus_bf16_bits(gf[l] * apus_moe_sigmoid(gf[l]));
        float32x4_t a1v = vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(a1), 16));
        float32x4_t uv = vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(gu + I + i), 16));
        float rf[4];
        vst1q_f32(rf, vmulq_f32(a1v, uv));
        for (int l = 0; l < 4; l++)
            act[i + (size_t)l] = apus_bf16_bits(rf[l]);
    }
    for (; i < I; i++) {
        float gv = apus_bf16_f32(gu[i]);
        uint16_t a1 = apus_bf16_bits(gv * apus_moe_sigmoid(gv));
        act[i] = apus_bf16_bits(apus_bf16_f32(a1) *
                                apus_bf16_f32(gu[I + i]));
    }
}
#endif

void apus_moe_silu_act(const uint16_t *gu, uint16_t *act, size_t I) {
#ifdef __ARM_NEON
    apus_moe_silu_act_neon(gu, act, I);
#else
    apus_moe_silu_act_scalar(gu, act, I);
#endif
}

/* --- (b2) shared-expert activation (single rounding) ----------------------*/

void apus_moe_silu_mul_scalar(const uint16_t *g, const uint16_t *u,
                              uint16_t *y, size_t I) {
    for (size_t i = 0; i < I; i++) {
        float gv = apus_bf16_f32(g[i]);
        float sil = gv * apus_moe_sigmoid(gv);
        y[i] = apus_bf16_bits(sil * apus_bf16_f32(u[i]));
    }
}

#ifdef __ARM_NEON
/* BITWISE the scalar silu_mul: widen is exact; the silu fp32 chain and
 * the product are per-lane scalar-mul/vmulq single roundings (the same
 * order as the anchor); the final rnd is per-lane scalar. */
void apus_moe_silu_mul_neon(const uint16_t *g, const uint16_t *u,
                            uint16_t *y, size_t I) {
    size_t i = 0;
    for (; i + 4 <= I; i += 4) {
        float gf[4];
        vst1q_f32(gf, vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(g + i), 16)));
        float32x4_t uv = vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(u + i), 16));
        float sil[4];
        for (int l = 0; l < 4; l++)
            sil[l] = gf[l] * apus_moe_sigmoid(gf[l]);
        float rf[4];
        vst1q_f32(rf, vmulq_f32(vld1q_f32(sil), uv));
        for (int l = 0; l < 4; l++)
            y[i + (size_t)l] = apus_bf16_bits(rf[l]);
    }
    for (; i < I; i++) {
        float gv = apus_bf16_f32(g[i]);
        float sil = gv * apus_moe_sigmoid(gv);
        y[i] = apus_bf16_bits(sil * apus_bf16_f32(u[i]));
    }
}
#endif

void apus_moe_silu_mul(const uint16_t *g, const uint16_t *u, uint16_t *y,
                       size_t I) {
#ifdef __ARM_NEON
    apus_moe_silu_mul_neon(g, u, y, I);
#else
    apus_moe_silu_mul_scalar(g, u, y, I);
#endif
}

/* --- (c) weighted combine ---------------------------------------------------*/

void apus_moe_combine_scalar(const uint16_t *y, const uint16_t *w,
                             uint16_t *out, size_t k, size_t N) {
    for (size_t n = 0; n < N; n++) {
        float acc = 0.0f;
        for (size_t e = 0; e < k; e++)
            acc += apus_bf16_f32(w[e]) * apus_bf16_f32(y[e * N + n]);
        out[n] = apus_bf16_bits(acc);
    }
}

#ifdef __ARM_NEON
/* BITWISE the scalar combine: 4 output lanes at a time, each lane
 * keeping the strictly-ascending-e add chain (broadcast-mul staged
 * products, plain vmulq/vaddq — never FMA); final rnd per lane. */
void apus_moe_combine_neon(const uint16_t *y, const uint16_t *w,
                           uint16_t *out, size_t k, size_t N) {
    size_t n = 0;
    for (; n + 4 <= N; n += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (size_t e = 0; e < k; e++) {
            float32x4_t wv = vdupq_n_f32(apus_bf16_f32(w[e]));
            float32x4_t yv = vreinterpretq_f32_u32(
                vshll_n_u16(vld1_u16(y + e * N + n), 16));
            acc = vaddq_f32(acc, vmulq_f32(wv, yv));
        }
        float rf[4];
        vst1q_f32(rf, acc);
        for (int l = 0; l < 4; l++)
            out[n + (size_t)l] = apus_bf16_bits(rf[l]);
    }
    for (; n < N; n++) {
        float acc = 0.0f;
        for (size_t e = 0; e < k; e++)
            acc += apus_bf16_f32(w[e]) * apus_bf16_f32(y[e * N + n]);
        out[n] = apus_bf16_bits(acc);
    }
}
#endif

void apus_moe_combine(const uint16_t *y, const uint16_t *w, uint16_t *out,
                      size_t k, size_t N) {
#ifdef __ARM_NEON
    apus_moe_combine_neon(y, w, out, k, N);
#else
    apus_moe_combine_scalar(y, w, out, k, N);
#endif
}

#endif /* APUS_MOE_IMPLEMENTATION */
#endif /* APUS_MOE_H */
