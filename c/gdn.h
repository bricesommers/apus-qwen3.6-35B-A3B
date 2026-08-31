/*
 * c/gdn.h — Gated DeltaNet (GDN) per-operation kernels for
 * Qwen3.6-35B-A3B, milestone M4. Replaces c/kda.h (Ling KDA). C11,
 * libc/libm only. Tier discipline (the engine rule): the scalar anchors
 * below are the normative semantic definition; the __ARM_NEON variants
 * are BITWISE identical to them (proven per-op in tests/m4a), and the
 * plain entry points dispatch to NEON when compiled in. Follows
 * docs/M4-CONTRACT.md section 3 and tools/oracle.py (the per-op entry
 * points these kernels are gated against).
 *
 * All tensors BF16 storage unless noted; fp32 is the only compute type.
 * "rnd" is apus_bf16_bits (RNE narrow, c/bf16.h — the approved M3
 * anchor, unchanged). Widening bf16->fp32 is exact. -ffp-contract=off
 * project-wide: every mul and add below is one IEEE fp32 rounding, in
 * the written order.
 *
 * Numerics contract (normative; oracle entry point in brackets):
 *
 *   (a) ONE fused depthwise causal conv1d k=4 over the fused [q|k|v]
 *       channels, no bias, then SiLU (HF:227-247, decode update 207-224;
 *       oracle causal_conv_silu). TWO rounding points (the torch opmath
 *       flow, NOT Ling's single-round conv):
 *         win[0..3] = x[c, t-3 .. t] (zero where t-3+i < 0)
 *         acc = 0; for i ascending: acc += f32(w[c,i]) * f32(win[i])
 *         co  = rnd(acc)                        (conv out bf16, rnd #1)
 *         out[c,t] = rnd( f32(co) * sigmoid(f32(co)) )   (SiLU, rnd #2)
 *       State: the last 3 PRE-conv inputs per channel, BF16, layout
 *       state[c*3+j] = win[j] for the next token (j=0 oldest); the
 *       oracle's [C,4] state is [dead, these 3] (its first column is
 *       shifted out before ever being read). Caller zero-inits.
 *       apus_gdn_conv1d (prefill) is a strict loop of the per-token
 *       body — BITWISE apus_gdn_conv1d_step (decode) token for token.
 *   (b) Decay gate, fp32 out, per-V-head SCALAR [H] (HF:574; oracle
 *       gdn_decay): a [H] BF16 (in_proj_a out), A_log/dt_bias [H] F32:
 *         g[h] = -expf(A_log[h]) * softplus(f32(a[h]) + dt_bias[h])
 *       NO lower bound, NOT the KDA sigmoid form. softplus =
 *       log1p(exp(x)), x for x > 20 (F.softplus/logaddexp branch; the
 *       tail is below fp32 resolution there — oracle header).
 *   (b2) Beta, BF16 sigmoid (HF:572; the deliberate difference from
 *       Ling's fp32 beta): beta[h] = rnd( 1/(1+expf(-f32(b[h]))) ).
 *   (c) Per-head L2 norm on q,k (HF:250-255,421-423; oracle l2norm):
 *       FP32 OUT (no bf16 rounding — the recurrence consumes fp32):
 *         ss = sum_i f32(x[i])^2  (ascending, mul+add roundings)
 *         y[i] = f32(x[i]) * (1/sqrtf(ss + 1e-6f))     (x * rsqrt —
 *       the HF formula, NOT Ling's literal division). The q scale
 *       (dk^-0.5, HF:426) is a SECOND fp32 rounding applied by the same
 *       op: y[i] = y[i] * scale (scale = 1.0f for k — an exact no-op).
 *   (d) Delta-rule recurrence, ALL fp32 (HF:395-455, CANONICAL — the
 *       chunk path HF:258-392 is a documented tolerance class, never
 *       the reference; oracle gdn_recurrence). Per token, per v-head h
 *       (state S fp32 [H,Dk,Dv], caller zero-inits; q,k fp32 from (c),
 *       v post-conv BF16, g fp32 from (b), beta BF16 from (b2)):
 *         dec = expf(g[h])                       (one scalar per head)
 *         for i,j ascending:   S[i,j] = S[i,j] * dec
 *         for j:  acc = 0; for i ascending: acc += k[i]*S[i,j]
 *                 delta[j] = (f32(v[j]) - acc) * f32(beta[h])
 *         for i,j ascending:   S[i,j] = S[i,j] + k[i]*delta[j]
 *         for j:  acc = 0; for i ascending: acc += q[i]*S[i,j]
 *                 o[j] = acc          (fp32 out, NO bf16 rounding here)
 *       o is computed AFTER the update (the current token included).
 *       apus_gdn_recurrent (prefill) is a strict loop of apus_gdn_step
 *       (decode) — BITWISE identical. The 16 K heads are repeat-
 *       interleaved x2 into the 32 V heads BEFORE this op (the layer
 *       l2norms the 16 K heads once and memcpys — bitwise identical to
 *       norming the duplicated heads).
 *   (e) GDN output norm per head (RMSNormGated, HF:175-192; oracle
 *       rmsnorm_gated): DIRECT weight (init ones, NO +1 — the OTHER
 *       convention from every other norm in the model). o,z [H*D] BF16
 *       (o is the recurrence output AFTER an explicit rnd at the layer;
 *       z the in_proj_z out), w [D] BF16 shared across heads:
 *         ss = sum_d f32(o[d])^2 (ascending); rs = 1/sqrtf(ss/D + 1e-6f)
 *         x1 = rnd( f32(o[d]) * rs )              (normalize, rnd)
 *         x2 = rnd( f32(x1) * f32(w[d]) )         (x direct weight, rnd)
 *         y[d] = rnd( f32(x2) * silu_fp32(f32(z[d])) )   (gate, rnd)
 *       Norm-before-gate; three rounding points (HF:189,190,192).
 *
 * Invariants: inputs finite in normative use. IEEE propagation follows
 * c/bf16.h's rules for the rnd steps. libm calls (expf, log1pf, sqrtf)
 * are the platform libm's; goldens are float64 truth with documented
 * tolerances (tests/m4a/README.md), never bitwise across libm
 * implementations.
 *
 * Usage: #define APUS_GDN_IMPLEMENTATION in exactly one TU. Depends on
 * c/bf16.h (helpers only): that TU must also define APUS_BF16_IMPLEMENTATION.
 */
#ifndef APUS_GDN_H
#define APUS_GDN_H

#include <stddef.h>
#include <stdint.h>

#include "bf16.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Conv kernel width and state depth (k-1). Part of the contract. */
#define APUS_GDN_CONV_K 4u
/* Norm epsilons (config rms_norm_eps / HF l2norm hardcoded 1e-6). */
#define APUS_GDN_L2NORM_EPS 1e-6f
#define APUS_GDN_ONORM_EPS  1e-6f

/* (a) Prefill: T tokens, channels C. x, out: [T,C] BF16 row-major;
 * w: [C, APUS_GDN_CONV_K] BF16; state: [C*(K-1)] BF16 in/out (pre-conv
 * inputs, j=0 oldest; caller zero-inits before the first chunk). */
void apus_gdn_conv1d(const uint16_t *x, const uint16_t *w, uint16_t *out,
                     size_t C, size_t T, uint16_t *state);
/* (a) Decode: one token. x, out: [C]. BITWISE == the per-token body of
 * apus_gdn_conv1d. */
void apus_gdn_conv1d_step(const uint16_t *x, const uint16_t *w,
                          uint16_t *out, size_t C, uint16_t *state);
/* Scalar anchors (the normative definition) and the BITWISE NEON
 * variants — exposed for the m4a scalar-vs-NEON gates. */
void apus_gdn_conv1d_step_scalar(const uint16_t *x, const uint16_t *w,
                                 uint16_t *out, size_t C, uint16_t *state);
#ifdef __ARM_NEON
void apus_gdn_conv1d_step_neon(const uint16_t *x, const uint16_t *w,
                               uint16_t *out, size_t C, uint16_t *state);
#endif

/* (b) Decay: a [H] BF16, A_log/dt_bias [H] F32, g [H] F32 out. */
void apus_gdn_decay(const uint16_t *a, const float *A_log,
                    const float *dt_bias, float *g, size_t H);
void apus_gdn_decay_scalar(const uint16_t *a, const float *A_log,
                           const float *dt_bias, float *g, size_t H);
#ifdef __ARM_NEON
void apus_gdn_decay_neon(const uint16_t *a, const float *A_log,
                         const float *dt_bias, float *g, size_t H);
#endif

/* (b2) Beta: b, beta [H] BF16. */
void apus_gdn_beta(const uint16_t *b, uint16_t *beta, size_t H);
void apus_gdn_beta_scalar(const uint16_t *b, uint16_t *beta, size_t H);
#ifdef __ARM_NEON
void apus_gdn_beta_neon(const uint16_t *b, uint16_t *beta, size_t H);
#endif

/* (c) L2 norm per head, FP32 out: x [H*D] BF16, y [H*D] F32. scale is
 * the q head-dim factor (dk^-0.5; pass 1.0f for k — exact no-op). */
void apus_gdn_l2norm(const uint16_t *x, float *y, size_t H, size_t D,
                     float scale);
void apus_gdn_l2norm_scalar(const uint16_t *x, float *y, size_t H,
                            size_t D, float scale);
#ifdef __ARM_NEON
void apus_gdn_l2norm_neon(const uint16_t *x, float *y, size_t H, size_t D,
                          float scale);
#endif

/* (d) One recurrence step (decode). S [H*Dk*Dv] F32 in/out;
 * q,k [H*Dk] F32 (post-l2norm, q pre-scaled); v [H*Dv] BF16; g [H] F32;
 * beta [H] BF16; o [H*Dv] F32 out. */
void apus_gdn_step(float *S, const float *q, const float *k,
                   const uint16_t *v, const float *g,
                   const uint16_t *beta, float *o,
                   size_t H, size_t Dk, size_t Dv);
/* (d) Threaded variant: pool over the independent heads; BITWISE
 * identical to apus_gdn_step at every APUS_THREADS. */
void apus_gdn_step_mt(float *S, const float *q, const float *k,
                      const uint16_t *v, const float *g,
                      const uint16_t *beta, float *o,
                      size_t H, size_t Dk, size_t Dv);
/* (d) Prefill: strict loop of apus_gdn_step over T tokens (BITWISE the
 * decode path stepped token by token). q,k,o: [T,H*Dk]/[T,H*Dv];
 * v: [T,H*Dv]; g,beta: [T,H]. */
void apus_gdn_recurrent(float *S, const float *q, const float *k,
                        const uint16_t *v, const float *g,
                        const uint16_t *beta, float *o,
                        size_t T, size_t H, size_t Dk, size_t Dv);
/* Scalar/NEON single-head bodies (the m4a gate surface). */
void apus_gdn_step_head_scalar(float *Sh, const float *qh,
                               const float *kh, const uint16_t *vh,
                               float g, float beta, float *oh,
                               size_t Dk, size_t Dv);
#ifdef __ARM_NEON
void apus_gdn_step_head_neon(float *Sh, const float *qh,
                             const float *kh, const uint16_t *vh,
                             float g, float beta, float *oh,
                             size_t Dk, size_t Dv);
#endif

/* (e) RMSNormGated: o, z, y [H*D] BF16; w [D] BF16 (direct, shared). */
void apus_gdn_onorm(const uint16_t *o, const uint16_t *z,
                    const uint16_t *w, uint16_t *y, size_t H, size_t D);
void apus_gdn_onorm_scalar(const uint16_t *o, const uint16_t *z,
                           const uint16_t *w, uint16_t *y,
                           size_t H, size_t D);
#ifdef __ARM_NEON
void apus_gdn_onorm_neon(const uint16_t *o, const uint16_t *z,
                         const uint16_t *w, uint16_t *y,
                         size_t H, size_t D);
#endif

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_GDN_IMPLEMENTATION

#include <math.h>

/* fp32 sigmoid, the pinned formula (overflow-safe: expf(-z) -> inf for
 * z << 0 gives 1/inf = +0, the correct limit). */
static inline float apus_gdn_sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

/* softplus: log1p(exp(x)), identity for x > 20 (the F.softplus/
 * logaddexp branch; the log1p(exp(-x)) tail is below fp32 resolution
 * there — oracle header note). */
static inline float apus_gdn_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

/* --- (a) fused causal conv1d k=4 + SiLU (two rounding points) ---------*/

/* One conv token, one channel: the shared computation. win[0..K-1] are
 * the BF16 window values, win[K-1] the current input. */
static inline uint16_t apus_gdn_conv_chan(const uint16_t *wc,
                                          const uint16_t *win) {
    float acc = 0.0f;
    for (size_t i = 0; i < APUS_GDN_CONV_K; i++)
        acc += apus_bf16_f32(wc[i]) * apus_bf16_f32(win[i]);
    uint16_t co = apus_bf16_bits(acc);              /* rnd #1 */
    float cof = apus_bf16_f32(co);
    return apus_bf16_bits(cof * apus_gdn_sigmoid(cof));  /* rnd #2 */
}

void apus_gdn_conv1d_step_scalar(const uint16_t *x, const uint16_t *w,
                                 uint16_t *out, size_t C,
                                 uint16_t *state) {
    for (size_t c = 0; c < C; c++) {
        uint16_t *st = state + c * (APUS_GDN_CONV_K - 1);
        uint16_t win[APUS_GDN_CONV_K];
        for (size_t j = 0; j < APUS_GDN_CONV_K - 1; j++)
            win[j] = st[j];
        win[APUS_GDN_CONV_K - 1] = x[c];
        out[c] = apus_gdn_conv_chan(w + c * APUS_GDN_CONV_K, win);
        /* state = last K-1 pre-conv inputs, j=0 oldest */
        for (size_t j = 0; j + 1 < APUS_GDN_CONV_K - 1; j++)
            st[j] = st[j + 1];
        st[APUS_GDN_CONV_K - 2] = x[c];
    }
}

#ifdef __ARM_NEON
#include <arm_neon.h>

/* BITWISE the scalar step: 4 channels per iteration; each lane runs the
 * channel's strictly-ascending 4-tap chain (one vmulq + one vaddq
 * rounding per tap, never FMA), then the two rnd points and the fp32
 * sigmoid per lane with the scalar helpers (identical code, hence
 * identical bits). */
void apus_gdn_conv1d_step_neon(const uint16_t *x, const uint16_t *w,
                               uint16_t *out, size_t C, uint16_t *state) {
    size_t c = 0;
    for (; c + 4 <= C; c += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        uint16_t wc[4], win[4];
        for (size_t i = 0; i < APUS_GDN_CONV_K; i++) {
            for (int l = 0; l < 4; l++) {
                size_t cc = c + (size_t)l;
                wc[l] = w[cc * APUS_GDN_CONV_K + i];
                win[l] = i + 1 < APUS_GDN_CONV_K
                    ? state[cc * (APUS_GDN_CONV_K - 1) + i] : x[cc];
            }
            uint16x4_t wh = vld1_u16(wc), xh = vld1_u16(win);
            float32x4_t wf = vreinterpretq_f32_u32(
                vshll_n_u16(wh, 16));
            float32x4_t xf = vreinterpretq_f32_u32(
                vshll_n_u16(xh, 16));
            acc = vaddq_f32(acc, vmulq_f32(wf, xf));
        }
        float a4[4];
        vst1q_f32(a4, acc);
        for (int l = 0; l < 4; l++) {
            uint16_t co = apus_bf16_bits(a4[l]);
            float cof = apus_bf16_f32(co);
            out[c + (size_t)l] = apus_bf16_bits(
                cof * apus_gdn_sigmoid(cof));
        }
        /* state shift (exact copies) */
        for (int l = 0; l < 4; l++) {
            uint16_t *st = state + (c + (size_t)l) * (APUS_GDN_CONV_K - 1);
            for (size_t j = 0; j + 1 < APUS_GDN_CONV_K - 1; j++)
                st[j] = st[j + 1];
            st[APUS_GDN_CONV_K - 2] = x[c + (size_t)l];
        }
    }
    for (; c < C; c++) {
        uint16_t *st = state + c * (APUS_GDN_CONV_K - 1);
        uint16_t winc[APUS_GDN_CONV_K];
        for (size_t j = 0; j < APUS_GDN_CONV_K - 1; j++)
            winc[j] = st[j];
        winc[APUS_GDN_CONV_K - 1] = x[c];
        out[c] = apus_gdn_conv_chan(w + c * APUS_GDN_CONV_K, winc);
        for (size_t j = 0; j + 1 < APUS_GDN_CONV_K - 1; j++)
            st[j] = st[j + 1];
        st[APUS_GDN_CONV_K - 2] = x[c];
    }
}
#endif /* __ARM_NEON */

void apus_gdn_conv1d_step(const uint16_t *x, const uint16_t *w,
                          uint16_t *out, size_t C, uint16_t *state) {
#ifdef __ARM_NEON
    apus_gdn_conv1d_step_neon(x, w, out, C, state);
#else
    apus_gdn_conv1d_step_scalar(x, w, out, C, state);
#endif
}

void apus_gdn_conv1d(const uint16_t *x, const uint16_t *w, uint16_t *out,
                     size_t C, size_t T, uint16_t *state) {
    for (size_t t = 0; t < T; t++)
        apus_gdn_conv1d_step(x + t * C, w, out + t * C, C, state);
}

/* --- (b) decay gate ----------------------------------------------------*/

void apus_gdn_decay_scalar(const uint16_t *a, const float *A_log,
                           const float *dt_bias, float *g, size_t H) {
    for (size_t h = 0; h < H; h++) {
        float t = apus_bf16_f32(a[h]) + dt_bias[h];
        g[h] = -expf(A_log[h]) * apus_gdn_softplus(t);
    }
}

#ifdef __ARM_NEON
/* BITWISE the scalar decay: 4 heads per iteration; the add is one vaddq
 * rounding per lane, the libm calls (expf, log1pf) are per-lane scalar —
 * identical code to the anchor, hence identical bits. */
void apus_gdn_decay_neon(const uint16_t *a, const float *A_log,
                         const float *dt_bias, float *g, size_t H) {
    size_t h = 0;
    for (; h + 4 <= H; h += 4) {
        float32x4_t af = vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(a + h), 16));
        float32x4_t t = vaddq_f32(af, vld1q_f32(dt_bias + h));
        float t4[4], al4[4];
        vst1q_f32(t4, t);
        vst1q_f32(al4, vld1q_f32(A_log + h));
        for (int l = 0; l < 4; l++)
            g[h + (size_t)l] =
                -expf(al4[l]) * apus_gdn_softplus(t4[l]);
    }
    for (; h < H; h++) {
        float t = apus_bf16_f32(a[h]) + dt_bias[h];
        g[h] = -expf(A_log[h]) * apus_gdn_softplus(t);
    }
}
#endif

void apus_gdn_decay(const uint16_t *a, const float *A_log,
                    const float *dt_bias, float *g, size_t H) {
#ifdef __ARM_NEON
    apus_gdn_decay_neon(a, A_log, dt_bias, g, H);
#else
    apus_gdn_decay_scalar(a, A_log, dt_bias, g, H);
#endif
}

/* --- (b2) beta (bf16 sigmoid) ------------------------------------------*/

void apus_gdn_beta_scalar(const uint16_t *b, uint16_t *beta, size_t H) {
    for (size_t h = 0; h < H; h++)
        beta[h] = apus_bf16_bits(
            apus_gdn_sigmoid(apus_bf16_f32(b[h])));
}

#ifdef __ARM_NEON
/* BITWISE the scalar beta: widen is exact; sigmoid + rnd per lane with
 * the scalar helpers. */
void apus_gdn_beta_neon(const uint16_t *b, uint16_t *beta, size_t H) {
    size_t h = 0;
    for (; h + 4 <= H; h += 4) {
        float32x4_t bf = vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(b + h), 16));
        float b4[4];
        vst1q_f32(b4, bf);
        for (int l = 0; l < 4; l++)
            beta[h + (size_t)l] = apus_bf16_bits(
                apus_gdn_sigmoid(b4[l]));
    }
    for (; h < H; h++)
        beta[h] = apus_bf16_bits(
            apus_gdn_sigmoid(apus_bf16_f32(b[h])));
}
#endif

void apus_gdn_beta(const uint16_t *b, uint16_t *beta, size_t H) {
#ifdef __ARM_NEON
    apus_gdn_beta_neon(b, beta, H);
#else
    apus_gdn_beta_scalar(b, beta, H);
#endif
}

/* --- (c) l2norm (fp32 out, x * rsqrt, optional q scale) ----------------*/

void apus_gdn_l2norm_scalar(const uint16_t *x, float *y, size_t H,
                            size_t D, float scale) {
    for (size_t h = 0; h < H; h++) {
        const uint16_t *xr = x + h * D;
        float *yr = y + h * D;
        float ss = 0.0f;
        for (size_t i = 0; i < D; i++) {
            float v = apus_bf16_f32(xr[i]);
            ss += v * v;
        }
        float rs = 1.0f / sqrtf(ss + APUS_GDN_L2NORM_EPS);
        for (size_t i = 0; i < D; i++)
            yr[i] = apus_bf16_f32(xr[i]) * rs * scale;
    }
}

#ifdef __ARM_NEON
/* BITWISE the scalar l2norm: ss staged 4 single-rounded squares then
 * added strictly ascending (the anchor's order); the normalize/scale
 * muls are two vmulq roundings per element. NOTE the anchor's
 * `xr[i] * rs * scale` is left-associative ((x*rs)*scale) — the NEON
 * mul order reproduces it exactly. */
void apus_gdn_l2norm_neon(const uint16_t *x, float *y, size_t H,
                          size_t D, float scale) {
    for (size_t h = 0; h < H; h++) {
        const uint16_t *xr = x + h * D;
        float *yr = y + h * D;
        float ss = 0.0f;
        size_t i = 0;
        for (; i + 4 <= D; i += 4) {
            float32x4_t v = vreinterpretq_f32_u32(
                vshll_n_u16(vld1_u16(xr + i), 16));
            float st[4];
            vst1q_f32(st, vmulq_f32(v, v));
            for (int l = 0; l < 4; l++)
                ss += st[l];
        }
        for (; i < D; i++) {
            float v = apus_bf16_f32(xr[i]);
            ss += v * v;
        }
        float rs = 1.0f / sqrtf(ss + APUS_GDN_L2NORM_EPS);
        float32x4_t rsv = vdupq_n_f32(rs);
        float32x4_t scv = vdupq_n_f32(scale);
        i = 0;
        for (; i + 4 <= D; i += 4) {
            float32x4_t v = vreinterpretq_f32_u32(
                vshll_n_u16(vld1_u16(xr + i), 16));
            vst1q_f32(yr + i, vmulq_f32(vmulq_f32(v, rsv), scv));
        }
        for (; i < D; i++)
            yr[i] = apus_bf16_f32(xr[i]) * rs * scale;
    }
}
#endif

void apus_gdn_l2norm(const uint16_t *x, float *y, size_t H, size_t D,
                     float scale) {
#ifdef __ARM_NEON
    apus_gdn_l2norm_neon(x, y, H, D, scale);
#else
    apus_gdn_l2norm_scalar(x, y, H, D, scale);
#endif
}

/* --- (d) delta-rule recurrence ------------------------------------------*/

/* One recurrence step for ONE head — the scalar anchor (the normative
 * definition). q,k fp32 [Dk]; v bf16 [Dv]; S fp32 [Dk,Dv]; o fp32 [Dv]. */
void apus_gdn_step_head_scalar(float *Sh, const float *qh,
                               const float *kh, const uint16_t *vh,
                               float g, float beta, float *oh,
                               size_t Dk, size_t Dv) {
    /* decay: S = S .* exp(g) (per-element mul, one rounding) */
    float dec = expf(g);
    for (size_t i = 0; i < Dk; i++) {
        float *Si = Sh + i * Dv;
        for (size_t j = 0; j < Dv; j++)
            Si[j] = Si[j] * dec;
    }
    /* delta = (v - k.S) * beta (over the decayed S) */
    float delta[Dv];
    for (size_t j = 0; j < Dv; j++) {
        float acc = 0.0f;
        for (size_t i = 0; i < Dk; i++)
            acc += kh[i] * Sh[i * Dv + j];
        delta[j] = (apus_bf16_f32(vh[j]) - acc) * beta;
    }
    /* rank-1 update: S += k outer delta (mul + add, two roundings) */
    for (size_t i = 0; i < Dk; i++) {
        float *Si = Sh + i * Dv;
        float ki = kh[i];
        for (size_t j = 0; j < Dv; j++)
            Si[j] = Si[j] + ki * delta[j];
    }
    /* o = q.S over the UPDATED S (q pre-scaled by the l2norm op) */
    for (size_t j = 0; j < Dv; j++) {
        float acc = 0.0f;
        for (size_t i = 0; i < Dk; i++)
            acc += qh[i] * Sh[i * Dv + j];
        oh[j] = acc;
    }
}

#ifdef __ARM_NEON
/* BITWISE the scalar step head: the decay and the rank-1 update are
 * per-element ops (one mul / one mul+add rounding per element — the
 * vector lanes reproduce them exactly); the two contractions (delta and
 * o) are vectorized ACROSS the contiguous output index j, so each lane
 * keeps the scalar loop's strictly-ascending-i add chain (staged
 * broadcast-mul products, plain vmulq/vaddq — never FMA). Scalar tails
 * for Dv % 4. expf stays the platform libm's (one scalar call). */
void apus_gdn_step_head_neon(float *Sh, const float *qh,
                             const float *kh, const uint16_t *vh,
                             float g, float beta, float *oh,
                             size_t Dk, size_t Dv) {
    float dec = expf(g);
    float32x4_t decv = vdupq_n_f32(dec);
    for (size_t i = 0; i < Dk; i++) {
        float *Si = Sh + i * Dv;
        size_t j = 0;
        for (; j + 4 <= Dv; j += 4)
            vst1q_f32(Si + j,
                      vmulq_f32(vld1q_f32(Si + j), decv));
        for (; j < Dv; j++)
            Si[j] = Si[j] * dec;
    }
    float delta[Dv];
    size_t j = 0;
    for (; j + 4 <= Dv; j += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (size_t i = 0; i < Dk; i++)
            acc = vaddq_f32(acc, vmulq_f32(vdupq_n_f32(kh[i]),
                                           vld1q_f32(Sh + i * Dv + j)));
        float32x4_t vf = vreinterpretq_f32_u32(
            vshll_n_u16(vld1_u16(vh + j), 16));
        vst1q_f32(delta + j,
                  vmulq_f32(vsubq_f32(vf, acc), vdupq_n_f32(beta)));
    }
    for (; j < Dv; j++) {
        float acc = 0.0f;
        for (size_t i = 0; i < Dk; i++)
            acc += kh[i] * Sh[i * Dv + j];
        delta[j] = (apus_bf16_f32(vh[j]) - acc) * beta;
    }
    for (size_t i = 0; i < Dk; i++) {
        float *Si = Sh + i * Dv;
        float32x4_t kiv = vdupq_n_f32(kh[i]);
        j = 0;
        for (; j + 4 <= Dv; j += 4)
            vst1q_f32(Si + j, vaddq_f32(vld1q_f32(Si + j),
                      vmulq_f32(kiv, vld1q_f32(delta + j))));
        for (; j < Dv; j++)
            Si[j] = Si[j] + kh[i] * delta[j];
    }
    j = 0;
    for (; j + 4 <= Dv; j += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (size_t i = 0; i < Dk; i++)
            acc = vaddq_f32(acc, vmulq_f32(vdupq_n_f32(qh[i]),
                                           vld1q_f32(Sh + i * Dv + j)));
        vst1q_f32(oh + j, acc);
    }
    for (; j < Dv; j++) {
        float acc = 0.0f;
        for (size_t i = 0; i < Dk; i++)
            acc += qh[i] * Sh[i * Dv + j];
        oh[j] = acc;
    }
}
#endif /* __ARM_NEON */

/* One step for one head: the dispatch shared by apus_gdn_step and the
 * threaded variant (identical code, hence bitwise identical). */
static void apus_gdn_step_head(float *Sh, const float *qh,
                               const float *kh, const uint16_t *vh,
                               float g, float beta, float *oh,
                               size_t Dk, size_t Dv) {
#ifdef __ARM_NEON
    apus_gdn_step_head_neon(Sh, qh, kh, vh, g, beta, oh, Dk, Dv);
#else
    apus_gdn_step_head_scalar(Sh, qh, kh, vh, g, beta, oh, Dk, Dv);
#endif
}

void apus_gdn_step(float *S, const float *q, const float *k,
                   const uint16_t *v, const float *g,
                   const uint16_t *beta, float *o,
                   size_t H, size_t Dk, size_t Dv) {
    for (size_t h = 0; h < H; h++)
        apus_gdn_step_head(S + h * Dk * Dv, q + h * Dk, k + h * Dk,
                           v + h * Dv, g[h], apus_bf16_f32(beta[h]),
                           o + h * Dv, Dk, Dv);
}

/* Threaded variant: heads are fully independent (per-head state,
 * per-head output), so the c/pool.h lane partition is BITWISE identical
 * to apus_gdn_step at every APUS_THREADS by construction. */
typedef struct {
    float *S;
    const float *q, *k;
    const uint16_t *v;
    const float *g;
    const uint16_t *beta;
    float *o;
    size_t Dk, Dv;
} ApusGdnStepJob;

static void apus_gdn_step_heads(void *vjob, size_t h0, size_t h1) {
    const ApusGdnStepJob *j = vjob;
    for (size_t h = h0; h < h1; h++)
        apus_gdn_step_head(j->S + h * j->Dk * j->Dv, j->q + h * j->Dk,
                           j->k + h * j->Dk, j->v + h * j->Dv, j->g[h],
                           apus_bf16_f32(j->beta[h]), j->o + h * j->Dv,
                           j->Dk, j->Dv);
}

void apus_gdn_step_mt(float *S, const float *q, const float *k,
                      const uint16_t *v, const float *g,
                      const uint16_t *beta, float *o,
                      size_t H, size_t Dk, size_t Dv) {
    ApusGdnStepJob job = { S, q, k, v, g, beta, o, Dk, Dv };
    apus_pool_run(H, apus_gdn_step_heads, &job);
}

void apus_gdn_recurrent(float *S, const float *q, const float *k,
                        const uint16_t *v, const float *g,
                        const uint16_t *beta, float *o,
                        size_t T, size_t H, size_t Dk, size_t Dv) {
    for (size_t t = 0; t < T; t++)
        apus_gdn_step(S, q + t * H * Dk, k + t * H * Dk,
                      v + t * H * Dv, g + t * H, beta + t * H,
                      o + t * H * Dv, H, Dk, Dv);
}

/* --- (e) RMSNormGated (direct weight, three rounding points) -----------*/

void apus_gdn_onorm_scalar(const uint16_t *o, const uint16_t *z,
                           const uint16_t *w, uint16_t *y,
                           size_t H, size_t D) {
    for (size_t h = 0; h < H; h++) {
        const uint16_t *oh = o + h * D;
        const uint16_t *zh = z + h * D;
        uint16_t *yh = y + h * D;
        float ss = 0.0f;
        for (size_t d = 0; d < D; d++) {
            float v = apus_bf16_f32(oh[d]);
            ss += v * v;
        }
        float rs = 1.0f / sqrtf(ss / (float)D + APUS_GDN_ONORM_EPS);
        for (size_t d = 0; d < D; d++) {
            uint16_t x1 = apus_bf16_bits(apus_bf16_f32(oh[d]) * rs);
            uint16_t x2 = apus_bf16_bits(apus_bf16_f32(x1) *
                                         apus_bf16_f32(w[d]));
            float zf = apus_bf16_f32(zh[d]);
            yh[d] = apus_bf16_bits(apus_bf16_f32(x2) *
                                   (zf * apus_gdn_sigmoid(zf)));
        }
    }
}

#ifdef __ARM_NEON
/* BITWISE the scalar onorm: ss staged 4 single-rounded squares then
 * added strictly ascending; the three rnd points and the fp32 sigmoid
 * are per-lane scalar (the identical helper calls, hence identical
 * bits). */
void apus_gdn_onorm_neon(const uint16_t *o, const uint16_t *z,
                         const uint16_t *w, uint16_t *y,
                         size_t H, size_t D) {
    for (size_t h = 0; h < H; h++) {
        const uint16_t *oh = o + h * D;
        const uint16_t *zh = z + h * D;
        uint16_t *yh = y + h * D;
        float ss = 0.0f;
        size_t d = 0;
        for (; d + 4 <= D; d += 4) {
            float32x4_t v = vreinterpretq_f32_u32(
                vshll_n_u16(vld1_u16(oh + d), 16));
            float st[4];
            vst1q_f32(st, vmulq_f32(v, v));
            for (int l = 0; l < 4; l++)
                ss += st[l];
        }
        for (; d < D; d++) {
            float v = apus_bf16_f32(oh[d]);
            ss += v * v;
        }
        float rs = 1.0f / sqrtf(ss / (float)D + APUS_GDN_ONORM_EPS);
        d = 0;
        for (; d + 4 <= D; d += 4) {
            float of[4], zf[4];
            vst1q_f32(of, vmulq_f32(vreinterpretq_f32_u32(
                vshll_n_u16(vld1_u16(oh + d), 16)), vdupq_n_f32(rs)));
            vst1q_f32(zf, vreinterpretq_f32_u32(
                vshll_n_u16(vld1_u16(zh + d), 16)));
            for (int l = 0; l < 4; l++) {
                uint16_t x1 = apus_bf16_bits(of[l]);
                uint16_t x2 = apus_bf16_bits(apus_bf16_f32(x1) *
                                             apus_bf16_f32(w[d + (size_t)l]));
                yh[d + (size_t)l] = apus_bf16_bits(
                    apus_bf16_f32(x2) *
                    (zf[l] * apus_gdn_sigmoid(zf[l])));
            }
        }
        for (; d < D; d++) {
            uint16_t x1 = apus_bf16_bits(apus_bf16_f32(oh[d]) * rs);
            uint16_t x2 = apus_bf16_bits(apus_bf16_f32(x1) *
                                         apus_bf16_f32(w[d]));
            float zf = apus_bf16_f32(zh[d]);
            yh[d] = apus_bf16_bits(apus_bf16_f32(x2) *
                                   (zf * apus_gdn_sigmoid(zf)));
        }
    }
}
#endif

void apus_gdn_onorm(const uint16_t *o, const uint16_t *z,
                    const uint16_t *w, uint16_t *y, size_t H, size_t D) {
#ifdef __ARM_NEON
    apus_gdn_onorm_neon(o, z, w, y, H, D);
#else
    apus_gdn_onorm_scalar(o, z, w, y, H, D);
#endif
}

#endif /* APUS_GDN_IMPLEMENTATION */
#endif /* APUS_GDN_H */
