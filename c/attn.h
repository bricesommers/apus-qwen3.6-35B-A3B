/*
 * c/attn.h — gated GQA full-attention per-operation kernels for
 * Qwen3.6-35B-A3B, milestone M4. Replaces the Ling gated-MLA kernels.
 * C11, libc/libm only. Tier discipline (the engine rule): the scalar
 * anchors are the normative semantic definition; the __ARM_NEON
 * variants are BITWISE identical to them (proven per-op in tests/m4a),
 * and the plain entry points dispatch to NEON when compiled in.
 * Follows docs/M4-CONTRACT.md sections 2 and 4 and tools/oracle.py
 * (rmsnorm / rotary_cos_sin / apply_rope / attention_core).
 *
 * All tensors BF16 storage; fp32 is the only compute type. "rnd" is
 * apus_bf16_bits (RNE narrow, c/bf16.h — the approved M3 anchor).
 * -ffp-contract=off project-wide: every mul/add below is one IEEE fp32
 * rounding, in the written order.
 *
 * Numerics contract (normative):
 *
 *   (a) RMSNorm, the (1+w) zero-init variant (HF:878-892; contract §2 —
 *       used for input/post-attention/final norms AND the per-head
 *       attention q/k norms; the checkpoint stores gain-1, the +1 is
 *       applied at runtime; NOT the GDN output norm, which lives in
 *       c/gdn.h):
 *         ss = sum_i f32(x[i])^2 (ascending);  rs = 1/sqrtf(ss/N + 1e-6f)
 *         y[i] = rnd( (f32(x[i]) * rs) * (1.0f + f32(w[i])) )
 *       SINGLE rounding at the end (HF:889-891 "(x * w).to(float16)" —
 *       NOT Ling's normalize-rnd-then-weight block convention).
 *   (b) Partial RoPE, GPT-NeoX rotate_half pairing (HF:91-171,621-664;
 *       contract §4): the first `rot` dims only (rot = 0.25 * head_dim),
 *       dims rot..D pass through UNTOUCHED; pairs (i, i+rot/2) share
 *       frequency invf[i] = theta^(-2i/rot), computed in float64 and
 *       narrowed to fp32 (HF computes inv_freq fp32 in torch):
 *         ang = pos * invf[i]              (fp32; HF casts to float32)
 *         c = rnd(cosf(ang)), s = rnd(sinf(ang))     (cast bf16, HF:154)
 *         y[i]       = rnd( rnd( x[i]   *c) + rnd(-x[i+R]*s) )
 *         y[i+R]     = rnd( rnd( x[i+R] *c) + rnd( x[i]  *s) )
 *       with R = rot/2 and x widened fp32 (one rnd per mul, one per
 *       add — the torch opmath flow). attention_scaling = 1.0; mrope is
 *       the identity for text-only (contract §4).
 *   (c) GQA eager attention (HF:667-701; contract §4, decision 7: QK·
 *       scale and P·V rounded per HF eager order): per query row t
 *       (global pos p = Tk-Tq+t), per q-head h with kv head h/nrep
 *       (nrep = H/Hkv, the repeat_kv expansion), causal:
 *         dot  = sum_d f32(q[h,d])*f32(K[j,hk,d])   (ascending d)
 *         A[j] = rnd( f32(rnd(dot)) * scale )       (scale = D^-0.5)
 *         m    = max_j f32(A[j])  (ascending scan);  e[j] = expf(A[j]-m)
 *         s    = sum_j e[j] (ascending);  P[j] = rnd(e[j] / s)
 *         o[h,d] = rnd( sum_j f32(P[j]) * f32(V[j,hk,d]) )  (asc. j)
 *       over keys j = 0..p. KV cache: K,V are append-only [Tk,Hkv,D]
 *       BF16 buffers; the caller appends BEFORE calling, so
 *       apus_attn_gqa_decode (Tq=1) is BITWISE the last row of a full
 *       apus_attn_gqa recompute — each row's computation depends only
 *       on keys 0..p and the per-row code path is identical.
 *   (d) Elementwise sigmoid output gate (HF:774-777; contract §4 —
 *       applied AFTER attention, BEFORE o_proj; the attn_output_gate
 *       config key is decorative, the gate is unconditional):
 *         g[i] = rnd( 1/(1+expf(-f32(gate[i]))) )
 *         y[i] = rnd( f32(o[i]) * f32(g[i]) )
 *       gate [N] is the packed q_proj gate half (per-head [q|gate]).
 *
 * Invariants: finite inputs in normative use. libm (expf, sqrtf, cosf,
 * sinf, pow) is the platform libm's — float64 goldens with documented
 * tolerances, never cross-libm bitwise (tests/m4a/README.md).
 *
 * Usage: #define APUS_ATTN_IMPLEMENTATION in exactly one TU. Depends on
 * c/bf16.h (helpers only): that TU must also define
 * APUS_BF16_IMPLEMENTATION.
 */
#ifndef APUS_ATTN_H
#define APUS_ATTN_H

#include <stddef.h>
#include <stdint.h>

#include "bf16.h"

#ifdef __cplusplus
extern "C" {
#endif

#define APUS_ATTN_RMS_EPS 1e-6f
/* Default RoPE base (config rope_theta = 1e7) as float64 for the
 * inv_freq computation; the ops take theta as a parameter. */
#define APUS_ATTN_ROPE_THETA 10000000.0

/* (a) RMSNorm (1+w) variant: x, w, y [N] BF16. w is the RAW stored
 * value (gain-1); the +1 is applied inside. */
void apus_attn_rmsnorm(const uint16_t *x, const uint16_t *w, uint16_t *y,
                       size_t N);
void apus_attn_rmsnorm_scalar(const uint16_t *x, const uint16_t *w,
                              uint16_t *y, size_t N);
#ifdef __ARM_NEON
void apus_attn_rmsnorm_neon(const uint16_t *x, const uint16_t *w,
                            uint16_t *y, size_t N);
#endif

/* (b) Partial NeoX RoPE on one head vector: x, y [D] BF16; the first
 * rot dims rotated (pairs (i, i+rot/2)), the rest copied. pos is the
 * token position (fp32, as HF casts position_ids to float32). */
void apus_attn_rope(const uint16_t *x, uint16_t *y, size_t D, size_t rot,
                    float pos, double theta);
void apus_attn_rope_scalar(const uint16_t *x, uint16_t *y, size_t D,
                           size_t rot, float pos, double theta);
#ifdef __ARM_NEON
void apus_attn_rope_neon(const uint16_t *x, uint16_t *y, size_t D,
                         size_t rot, float pos, double theta);
#endif

/* (c) GQA eager attention, causal. q: [Tq,H,D] BF16; kc,vc: [Tk,Hkv,D]
 * BF16 (cache, already appended); o: [Tq,H,D] BF16. Query row t attends
 * keys 0..(Tk-Tq+t) of kv head h/(H/Hkv). scale is the caller's fp32
 * D^-0.5. abuf [Tk] u16, ebuf [Tk] f32: scratch. */
void apus_attn_gqa(const uint16_t *q, const uint16_t *kc,
                   const uint16_t *vc, uint16_t *o,
                   size_t Tq, size_t Tk, size_t H, size_t Hkv, size_t D,
                   float scale, uint16_t *abuf, float *ebuf);
/* (c) Decode: one query against the full cache. BITWISE == the last row
 * of apus_attn_gqa with Tq = 1 (and == the last row of a full recompute
 * over the same cache). */
void apus_attn_gqa_decode(const uint16_t *q, const uint16_t *kc,
                          const uint16_t *vc, uint16_t *o,
                          size_t Tk, size_t H, size_t Hkv, size_t D,
                          float scale, uint16_t *abuf, float *ebuf);

/* (c) Threaded variants: (row, head) units are independent, so the
 * c/pool.h lane partition is BITWISE identical to the sequential
 * variants at every APUS_THREADS by construction. Scratch is per-unit
 * stack, no caller buffers needed. */
void apus_attn_gqa_mt(const uint16_t *q, const uint16_t *kc,
                      const uint16_t *vc, uint16_t *o,
                      size_t Tq, size_t Tk, size_t H, size_t Hkv,
                      size_t D, float scale);
void apus_attn_gqa_decode_mt(const uint16_t *q, const uint16_t *kc,
                             const uint16_t *vc, uint16_t *o,
                             size_t Tk, size_t H, size_t Hkv, size_t D,
                             float scale);

/* Scalar/NEON single-row bodies (the m4a gate surface). p = last key
 * index (inclusive). */
void apus_attn_gqa_row_scalar(const uint16_t *qh, const uint16_t *kc,
                              const uint16_t *vc, uint16_t *oh,
                              size_t Hkv, size_t D, float scale,
                              uint16_t *abuf, float *ebuf, size_t p);
#ifdef __ARM_NEON
void apus_attn_gqa_row_neon(const uint16_t *qh, const uint16_t *kc,
                            const uint16_t *vc, uint16_t *oh,
                            size_t Hkv, size_t D, float scale,
                            uint16_t *abuf, float *ebuf, size_t p);
#endif

/* (d) Elementwise sigmoid output gate: o, gate, y [N] BF16. */
void apus_attn_outgate(const uint16_t *o, const uint16_t *gate,
                       uint16_t *y, size_t N);
void apus_attn_outgate_scalar(const uint16_t *o, const uint16_t *gate,
                              uint16_t *y, size_t N);
#ifdef __ARM_NEON
void apus_attn_outgate_neon(const uint16_t *o, const uint16_t *gate,
                            uint16_t *y, size_t N);
#endif

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_ATTN_IMPLEMENTATION

#include <math.h>

static inline float apus_attn_sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

/* --- (a) RMSNorm (1+w), single rounding ---------------------------------*/

void apus_attn_rmsnorm_scalar(const uint16_t *x, const uint16_t *w,
                              uint16_t *y, size_t N) {
    float ss = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float v = apus_bf16_f32(x[i]);
        ss += v * v;
    }
    float rs = 1.0f / sqrtf(ss / (float)N + APUS_ATTN_RMS_EPS);
    for (size_t i = 0; i < N; i++)
        y[i] = apus_bf16_bits((apus_bf16_f32(x[i]) * rs) *
                              (1.0f + apus_bf16_f32(w[i])));
}

#ifdef __ARM_NEON
#include <arm_neon.h>

/* BITWISE the scalar rmsnorm: ss staged 4 single-rounded squares then
 * added strictly ascending; the (x*rs)*(1+w) chain is two vmulq
 * roundings + one vaddq per element (the same order as the anchor);
 * the final rnd is per-lane scalar (the identical helper). */
void apus_attn_rmsnorm_neon(const uint16_t *x, const uint16_t *w,
                            uint16_t *y, size_t N) {
    float ss = 0.0f;
    size_t i = 0;
    for (; i + 4 <= N; i += 4) {
        float32x4_t v = vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(x + i), 16));
        float st[4];
        vst1q_f32(st, vmulq_f32(v, v));
        for (int l = 0; l < 4; l++)
            ss += st[l];
    }
    for (; i < N; i++) {
        float v = apus_bf16_f32(x[i]);
        ss += v * v;
    }
    float rs = 1.0f / sqrtf(ss / (float)N + APUS_ATTN_RMS_EPS);
    float32x4_t rsv = vdupq_n_f32(rs);
    float32x4_t onev = vdupq_n_f32(1.0f);
    i = 0;
    for (; i + 4 <= N; i += 4) {
        float32x4_t xv = vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(x + i), 16));
        float32x4_t wv = vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(w + i), 16));
        float32x4_t r = vmulq_f32(vmulq_f32(xv, rsv),
                                  vaddq_f32(onev, wv));
        float rf[4];
        vst1q_f32(rf, r);
        for (int l = 0; l < 4; l++)
            y[i + (size_t)l] = apus_bf16_bits(rf[l]);
    }
    for (; i < N; i++)
        y[i] = apus_bf16_bits((apus_bf16_f32(x[i]) * rs) *
                              (1.0f + apus_bf16_f32(w[i])));
}
#endif

void apus_attn_rmsnorm(const uint16_t *x, const uint16_t *w, uint16_t *y,
                       size_t N) {
#ifdef __ARM_NEON
    apus_attn_rmsnorm_neon(x, w, y, N);
#else
    apus_attn_rmsnorm_scalar(x, w, y, N);
#endif
}

/* --- (b) partial NeoX RoPE -----------------------------------------------*/

void apus_attn_rope_scalar(const uint16_t *x, uint16_t *y, size_t D,
                           size_t rot, float pos, double theta) {
    size_t R = rot / 2;
    for (size_t i = 0; i < R; i++) {
        /* float64 base^exponent narrowed to fp32 (the HF fp32 inv_freq;
         * the golden compares float64 truth — a documented tolerance
         * class, see header contract) */
        float invf = (float)(1.0 / pow(theta, 2.0 * (double)i /
                                           (double)rot));
        float ang = pos * invf;
        uint16_t cb = apus_bf16_bits(cosf(ang));
        uint16_t sb = apus_bf16_bits(sinf(ang));
        float c = apus_bf16_f32(cb), s = apus_bf16_f32(sb);
        float x0 = apus_bf16_f32(x[i]);
        float x1 = apus_bf16_f32(x[i + R]);
        y[i]     = apus_bf16_bits(apus_bf16_round(x0 * c) +
                                  apus_bf16_round(-x1 * s));
        y[i + R] = apus_bf16_bits(apus_bf16_round(x1 * c) +
                                  apus_bf16_round(x0 * s));
    }
    for (size_t i = rot; i < D; i++)
        y[i] = x[i];
}

#ifdef __ARM_NEON
/* BITWISE the scalar rope: one pair per lane, 4 pairs per iteration;
 * invf/ang and cosf/sinf are per-lane scalar (the identical libm
 * calls), the muls/adds are vmulq/vaddq single roundings in the
 * anchor's order, the rnd points per-lane scalar. */
void apus_attn_rope_neon(const uint16_t *x, uint16_t *y, size_t D,
                         size_t rot, float pos, double theta) {
    size_t R = rot / 2;
    size_t i = 0;
    for (; i + 4 <= R; i += 4) {
        float c4[4], s4[4], x04[4], x14[4];
        for (int l = 0; l < 4; l++) {
            float invf = (float)(1.0 / pow(theta,
                2.0 * (double)(i + (size_t)l) / (double)rot));
            float ang = pos * invf;
            c4[l] = apus_bf16_f32(apus_bf16_bits(cosf(ang)));
            s4[l] = apus_bf16_f32(apus_bf16_bits(sinf(ang)));
        }
        vst1q_f32(x04, vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(x + i), 16)));
        vst1q_f32(x14, vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(x + i + R), 16)));
        float32x4_t x0 = vld1q_f32(x04), x1 = vld1q_f32(x14);
        float32x4_t cv = vld1q_f32(c4), sv = vld1q_f32(s4);
        float m1[4], m2[4], m3[4], m4[4];
        vst1q_f32(m1, vmulq_f32(x0, cv));
        vst1q_f32(m2, vmulq_f32(vnegq_f32(x1), sv));
        vst1q_f32(m3, vmulq_f32(x1, cv));
        vst1q_f32(m4, vmulq_f32(x0, sv));
        for (int l = 0; l < 4; l++) {
            y[i + (size_t)l] = apus_bf16_bits(
                apus_bf16_round(m1[l]) + apus_bf16_round(m2[l]));
            y[i + R + (size_t)l] = apus_bf16_bits(
                apus_bf16_round(m3[l]) + apus_bf16_round(m4[l]));
        }
    }
    for (; i < R; i++) {
        float invf = (float)(1.0 / pow(theta, 2.0 * (double)i /
                                           (double)rot));
        float ang = pos * invf;
        uint16_t cb = apus_bf16_bits(cosf(ang));
        uint16_t sb = apus_bf16_bits(sinf(ang));
        float c = apus_bf16_f32(cb), s = apus_bf16_f32(sb);
        float x0 = apus_bf16_f32(x[i]);
        float x1 = apus_bf16_f32(x[i + R]);
        y[i]     = apus_bf16_bits(apus_bf16_round(x0 * c) +
                                  apus_bf16_round(-x1 * s));
        y[i + R] = apus_bf16_bits(apus_bf16_round(x1 * c) +
                                  apus_bf16_round(x0 * s));
    }
    for (size_t j = rot; j < D; j++)
        y[j] = x[j];
}
#endif

void apus_attn_rope(const uint16_t *x, uint16_t *y, size_t D, size_t rot,
                    float pos, double theta) {
#ifdef __ARM_NEON
    apus_attn_rope_neon(x, y, D, rot, pos, theta);
#else
    apus_attn_rope_scalar(x, y, D, rot, pos, theta);
#endif
}

/* --- (c) GQA eager attention ----------------------------------------------

 * Layout note: kc/vc are [Tk, Hkv, D]; the ROW base passed here is
 * already offset to the query head's kv head (kc + hk*D). */

void apus_attn_gqa_row_scalar(const uint16_t *qh, const uint16_t *kc,
                              const uint16_t *vc, uint16_t *oh,
                              size_t Hkv, size_t D, float scale,
                              uint16_t *abuf, float *ebuf, size_t p) {
    for (size_t j = 0; j <= p; j++) {
        const uint16_t *kr = kc + (j * Hkv) * D;
        float acc = 0.0f;
        for (size_t d = 0; d < D; d++)
            acc += apus_bf16_f32(qh[d]) * apus_bf16_f32(kr[d]);
        abuf[j] = apus_bf16_bits(
            apus_bf16_f32(apus_bf16_bits(acc)) * scale);
    }
    float m = apus_bf16_f32(abuf[0]);
    for (size_t j = 1; j <= p; j++) {
        float a = apus_bf16_f32(abuf[j]);
        if (a > m) m = a;
    }
    float s = 0.0f;
    for (size_t j = 0; j <= p; j++) {
        ebuf[j] = expf(apus_bf16_f32(abuf[j]) - m);
        s += ebuf[j];
    }
    for (size_t j = 0; j <= p; j++)
        abuf[j] = apus_bf16_bits(ebuf[j] / s);
    for (size_t d = 0; d < D; d++) {
        float acc = 0.0f;
        for (size_t j = 0; j <= p; j++)
            acc += apus_bf16_f32(abuf[j]) *
                   apus_bf16_f32(vc[(j * Hkv) * D + d]);
        oh[d] = apus_bf16_bits(acc);
    }
}

#ifdef __ARM_NEON
/* BITWISE the scalar row: the q.k dots stage 4 single-rounded products
 * then add them strictly in ascending d (the scalar order); the softmax
 * (max scan, expf, ascending sum, normalize) is the verbatim scalar
 * code; the P.V contraction is vectorized ACROSS the contiguous d
 * lanes, each lane keeping the scalar loop's ascending-j add chain. */
void apus_attn_gqa_row_neon(const uint16_t *qh, const uint16_t *kc,
                            const uint16_t *vc, uint16_t *oh,
                            size_t Hkv, size_t D, float scale,
                            uint16_t *abuf, float *ebuf, size_t p) {
    for (size_t j = 0; j <= p; j++) {
        const uint16_t *kr = kc + (j * Hkv) * D;
        float acc = 0.0f;
        size_t d = 0;
        for (; d + 4 <= D; d += 4) {
            float32x4_t qv = vreinterpretq_f32_u32(
                vshll_n_u16(vld1_u16(qh + d), 16));
            float32x4_t kv = vreinterpretq_f32_u32(
                vshll_n_u16(vld1_u16(kr + d), 16));
            float st[4];
            vst1q_f32(st, vmulq_f32(qv, kv));
            for (int l = 0; l < 4; l++)
                acc += st[l];
        }
        for (; d < D; d++)
            acc += apus_bf16_f32(qh[d]) * apus_bf16_f32(kr[d]);
        abuf[j] = apus_bf16_bits(
            apus_bf16_f32(apus_bf16_bits(acc)) * scale);
    }
    float m = apus_bf16_f32(abuf[0]);
    for (size_t j = 1; j <= p; j++) {
        float a = apus_bf16_f32(abuf[j]);
        if (a > m) m = a;
    }
    float s = 0.0f;
    for (size_t j = 0; j <= p; j++) {
        ebuf[j] = expf(apus_bf16_f32(abuf[j]) - m);
        s += ebuf[j];
    }
    for (size_t j = 0; j <= p; j++)
        abuf[j] = apus_bf16_bits(ebuf[j] / s);
    size_t d = 0;
    for (; d + 4 <= D; d += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (size_t j = 0; j <= p; j++) {
            float32x4_t pv = vdupq_n_f32(apus_bf16_f32(abuf[j]));
            float32x4_t vv = vreinterpretq_f32_u32(
                vshll_n_u16(vld1_u16(vc + (j * Hkv) * D + d), 16));
            acc = vaddq_f32(acc, vmulq_f32(pv, vv));
        }
        float tmp[4];
        vst1q_f32(tmp, acc);
        for (int l = 0; l < 4; l++)
            oh[d + (size_t)l] = apus_bf16_bits(tmp[l]);
    }
    for (; d < D; d++) {
        float acc = 0.0f;
        for (size_t j = 0; j <= p; j++)
            acc += apus_bf16_f32(abuf[j]) *
                   apus_bf16_f32(vc[(j * Hkv) * D + d]);
        oh[d] = apus_bf16_bits(acc);
    }
}
#endif

/* One attention row for one head: query q[h,:] against keys 0..p of
 * kv head hk. Dispatch shared by every variant (identical code, hence
 * bitwise identical). */
static void apus_attn_gqa_row(const uint16_t *qh, const uint16_t *kc,
                              const uint16_t *vc, uint16_t *oh,
                              size_t Hkv, size_t D, float scale,
                              uint16_t *abuf, float *ebuf, size_t p) {
#ifdef __ARM_NEON
    apus_attn_gqa_row_neon(qh, kc, vc, oh, Hkv, D, scale, abuf, ebuf, p);
#else
    apus_attn_gqa_row_scalar(qh, kc, vc, oh, Hkv, D, scale, abuf, ebuf,
                             p);
#endif
}

void apus_attn_gqa(const uint16_t *q, const uint16_t *kc,
                   const uint16_t *vc, uint16_t *o,
                   size_t Tq, size_t Tk, size_t H, size_t Hkv, size_t D,
                   float scale, uint16_t *abuf, float *ebuf) {
    size_t nrep = H / Hkv;
    for (size_t t = 0; t < Tq; t++) {
        size_t p = (Tk - Tq) + t;       /* global position of query t */
        for (size_t h = 0; h < H; h++) {
            size_t hk = h / nrep;
            apus_attn_gqa_row(q + (t * H + h) * D, kc + hk * D,
                              vc + hk * D, o + (t * H + h) * D,
                              Hkv, D, scale, abuf, ebuf, p);
        }
    }
}

void apus_attn_gqa_decode(const uint16_t *q, const uint16_t *kc,
                          const uint16_t *vc, uint16_t *o,
                          size_t Tk, size_t H, size_t Hkv, size_t D,
                          float scale, uint16_t *abuf, float *ebuf) {
    apus_attn_gqa(q, kc, vc, o, 1, Tk, H, Hkv, D, scale, abuf, ebuf);
}

/* --- threaded variants ---------------------------------------------------*/

/* One (row, head) unit with per-unit stack scratch — the identical code
 * as apus_attn_gqa_row, hence bitwise identical output. */
static void apus_attn_gqa_unit(const uint16_t *qh, const uint16_t *kc,
                               const uint16_t *vc, uint16_t *oh,
                               size_t Hkv, size_t D, float scale,
                               size_t Tk) {
    uint16_t abuf[Tk];
    float ebuf[Tk];
    apus_attn_gqa_row(qh, kc, vc, oh, Hkv, D, scale, abuf, ebuf, Tk - 1);
}

typedef struct {
    const uint16_t *q, *kc, *vc;
    uint16_t *o;
    size_t Tq, Tk, H, Hkv, D;
    float scale;
} ApusAttnGqaJob;

static void apus_attn_gqa_units(void *vjob, size_t u0, size_t u1) {
    const ApusAttnGqaJob *j = vjob;
    size_t nrep = j->H / j->Hkv;
    for (size_t u = u0; u < u1; u++) {
        size_t t = u / j->H, h = u % j->H;
        size_t p = (j->Tk - j->Tq) + t;
        size_t hk = h / nrep;
        apus_attn_gqa_unit(j->q + (t * j->H + h) * j->D,
                           j->kc + hk * j->D, j->vc + hk * j->D,
                           j->o + (t * j->H + h) * j->D,
                           j->Hkv, j->D, j->scale, p + 1);
    }
}

void apus_attn_gqa_mt(const uint16_t *q, const uint16_t *kc,
                      const uint16_t *vc, uint16_t *o,
                      size_t Tq, size_t Tk, size_t H, size_t Hkv,
                      size_t D, float scale) {
    ApusAttnGqaJob job = { q, kc, vc, o, Tq, Tk, H, Hkv, D, scale };
    apus_pool_run(Tq * H, apus_attn_gqa_units, &job);
}

void apus_attn_gqa_decode_mt(const uint16_t *q, const uint16_t *kc,
                             const uint16_t *vc, uint16_t *o,
                             size_t Tk, size_t H, size_t Hkv, size_t D,
                             float scale) {
    apus_attn_gqa_mt(q, kc, vc, o, 1, Tk, H, Hkv, D, scale);
}

/* --- (d) elementwise sigmoid output gate ----------------------------------*/

void apus_attn_outgate_scalar(const uint16_t *o, const uint16_t *gate,
                              uint16_t *y, size_t N) {
    for (size_t i = 0; i < N; i++) {
        uint16_t gb = apus_bf16_bits(
            apus_attn_sigmoid(apus_bf16_f32(gate[i])));
        y[i] = apus_bf16_bits(apus_bf16_f32(o[i]) * apus_bf16_f32(gb));
    }
}

#ifdef __ARM_NEON
/* BITWISE the scalar outgate: widen is exact; sigmoid + both rnd points
 * per lane with the scalar helpers; the final mul is one vmulq rounding
 * per element. */
void apus_attn_outgate_neon(const uint16_t *o, const uint16_t *gate,
                            uint16_t *y, size_t N) {
    size_t i = 0;
    for (; i + 4 <= N; i += 4) {
        float gf[4];
        vst1q_f32(gf, vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(gate + i), 16)));
        uint16_t gb[4];
        for (int l = 0; l < 4; l++)
            gb[l] = apus_bf16_bits(apus_attn_sigmoid(gf[l]));
        float32x4_t ov = vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(o + i), 16));
        float32x4_t gv = vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(gb), 16));
        float rf[4];
        vst1q_f32(rf, vmulq_f32(ov, gv));
        for (int l = 0; l < 4; l++)
            y[i + (size_t)l] = apus_bf16_bits(rf[l]);
    }
    for (; i < N; i++) {
        uint16_t gb = apus_bf16_bits(
            apus_attn_sigmoid(apus_bf16_f32(gate[i])));
        y[i] = apus_bf16_bits(apus_bf16_f32(o[i]) * apus_bf16_f32(gb));
    }
}
#endif

void apus_attn_outgate(const uint16_t *o, const uint16_t *gate,
                       uint16_t *y, size_t N) {
#ifdef __ARM_NEON
    apus_attn_outgate_neon(o, gate, y, N);
#else
    apus_attn_outgate_scalar(o, gate, y, N);
#endif
}

#endif /* APUS_ATTN_IMPLEMENTATION */
#endif /* APUS_ATTN_H */
