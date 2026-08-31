/*
 * tests/m9c/profile_phasea.c — component profile for the M9 batched
 * prefill restoration, at the REAL Qwen3.6-35B-A3B dims (hidden 2048,
 * GDN 16 K heads / 32 V heads x 128, conv_dim 8192; MoE 256 experts
 * top-8, expert 1024<->2048<->512). Synthetic random weights, no I/O,
 * so the compute structure is isolated from the tiered store's NVMe
 * streaming (which dominates the real-model prefill wall).
 *
 *   A: T x per-token GDN mixer (replica of c/layer.h's static
 *      apus_layer_gdn_attn)
 *   B: the SHIPPED batched phase A (apus_layer_forward_hot on a
 *      synthetic GDN layer with a negligible MoE) vs the sequential
 *      apus_layer_forward — the end-to-end phase-A A/B, with a bitwise
 *      memcmp of the layer outputs
 *   C: the four GDN projections alone, batched (gemm_hot M=T)
 *   D: the sequential parts alone (conv/decay/beta/l2norm/step/onorm)
 *   E: phase-B MoE, real dims: per-token top-8 expert GEMVs (the
 *      sequential body) vs the unique-expert batched GEMMs (the M9
 *      restoration) at T=512, BITWISE identical outputs
 *
 * Informational — NOT a gate. macOS/ARM + Linux both build (the hot
 * dispatch picks ILP-NEON / AVX2 / scalar per platform).
 */
#define APUS_BF16_IMPLEMENTATION
#define APUS_GDN_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION
#define APUS_LAYER_IMPLEMENTATION
#include "bf16.h"
#include "gdn.h"
#include "attn.h"
#include "moe.h"
#include "layer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* replica of c/layer.h's static apus_layer_gdn_attn (the profile target
 * is the per-token body; kept in sync by eye — informational only) */
static void gdn_attn_tok(const ApusLayerCfg *c, const ApusLayerW *w,
                         ApusLayerState *st, const uint16_t *ln1,
                         uint16_t *ao) {
    size_t H = c->hidden;
    size_t hk = c->gdn_hk, hv = c->gdn_hv, dk = c->gdn_dk, dv = c->gdn_dv;
    size_t key_dim = hk * dk, value_dim = hv * dv;
    size_t conv_dim = 2 * key_dim + value_dim;
    float qscale = (float)pow((double)dk, -0.5);
    uint16_t *qkv = malloc(conv_dim * 2), *z = malloc(value_dim * 2);
    uint16_t *b = malloc(hv * 2), *a = malloc(hv * 2);
    uint16_t *qc = malloc(conv_dim * 2), *beta_b = malloc(hv * 2);
    uint16_t *ob = malloc(value_dim * 2), *on = malloc(value_dim * 2);
    float *g = malloc(hv * sizeof(float));
    float *reco = malloc(value_dim * sizeof(float));
    float *qn = malloc(hk * dk * sizeof(float));
    float *kn = malloc(hk * dk * sizeof(float));
    float *qr = malloc(hv * dk * sizeof(float));
    float *kr = malloc(hv * dk * sizeof(float));
    apus_bf16_gemv_hot(w->w_qkv, ln1, qkv, conv_dim, H);
    apus_bf16_gemv_hot(w->w_z, ln1, z, value_dim, H);
    apus_bf16_gemv_hot(w->w_b, ln1, b, hv, H);
    apus_bf16_gemv_hot(w->w_a, ln1, a, hv, H);
    apus_gdn_conv1d_step(qkv, w->conv_w, qc, conv_dim, st->conv_state);
    apus_gdn_beta(b, beta_b, hv);
    apus_gdn_decay(a, w->A_log, w->dt_bias, g, hv);
    apus_gdn_l2norm(qc, qn, hk, dk, qscale);
    apus_gdn_l2norm(qc + key_dim, kn, hk, dk, 1.0f);
    size_t rep = hv / hk;
    for (size_t h = 0; h < hk; h++)
        for (size_t r = 0; r < rep; r++) {
            memcpy(qr + (h * rep + r) * dk, qn + h * dk,
                   dk * sizeof(float));
            memcpy(kr + (h * rep + r) * dk, kn + h * dk,
                   dk * sizeof(float));
        }
    apus_gdn_step_mt(st->S, qr, kr, qc + 2 * key_dim, g, beta_b, reco,
                     hv, dk, dv);
    for (size_t i = 0; i < value_dim; i++)
        ob[i] = apus_bf16_bits(reco[i]);
    apus_gdn_onorm(ob, z, w->onorm_w, on, hv, dv);
    apus_bf16_gemv_hot(w->w_out, on, ao, H, value_dim);
    free(qkv); free(z); free(b); free(a); free(qc); free(beta_b);
    free(ob); free(on); free(g); free(reco); free(qn); free(kn);
    free(qr); free(kr);
}

static uint16_t *wfill(size_t n) {
    uint16_t *p = malloc(n * 2);
    for (size_t i = 0; i < n; i++)
        p[i] = (uint16_t)(0x2E00 | (rng_u64() % 1024));
    return p;
}
static uint16_t *nfill(size_t n) {   /* (1+w)-style small values */
    uint16_t *p = malloc(n * 2);
    for (size_t i = 0; i < n; i++) p[i] = apus_bf16_bits(0.01f);
    return p;
}

int main(void) {
    const size_t H = 2048, HK = 16, HV = 32, DK = 128, DV = 128, T = 512;
    const size_t key_dim = HK * DK, value_dim = HV * DV;
    const size_t conv_dim = 2 * key_dim + value_dim;
    printf("profile_phasea: GDN layer H=%zu conv_dim=%zu T=%zu "
           "(threads %d)\n", H, conv_dim, T, apus_pool_threads());
    ApusLayerCfg c;
    memset(&c, 0, sizeof c);
    c.kind = APUS_LAYER_GDN;
    c.hidden = H;
    c.gdn_hk = HK; c.gdn_hv = HV; c.gdn_dk = DK; c.gdn_dv = DV;
    /* negligible MoE (phase-A arms only through the layer) */
    c.experts = 4; c.moe_inter = 16; c.shared_inter = 16; c.top_k = 2;
    ApusLayerW w;
    memset(&w, 0, sizeof w);
    w.w_qkv = wfill(conv_dim * H);
    w.w_z = wfill(value_dim * H);
    w.w_b = wfill(HV * H);
    w.w_a = wfill(HV * H);
    w.conv_w = wfill(conv_dim * 4);
    w.onorm_w = nfill(DV);
    w.w_out = wfill(H * value_dim);
    w.ln1_w = nfill(H);
    w.ln2_w = nfill(H);
    w.rtr_w = wfill(c.experts * H);
    w.exp_gu = wfill(c.experts * 2 * c.moe_inter * H);
    w.exp_d = wfill(c.experts * H * c.moe_inter);
    w.sh_g = wfill(c.shared_inter * H);
    w.sh_u = wfill(c.shared_inter * H);
    w.sh_d = wfill(H * c.shared_inter);
    w.sh_gate = wfill(H);
    float *alog = malloc(HV * sizeof(float));
    float *dtb = malloc(HV * sizeof(float));
    for (size_t i = 0; i < HV; i++) { alog[i] = 1.0f; dtb[i] = 0.01f; }
    w.A_log = alog;
    w.dt_bias = dtb;
    uint16_t *x = malloc(T * H * 2), *ao = malloc(T * H * 2);
    uint16_t *ln1 = malloc(T * H * 2);
    for (size_t i = 0; i < T * H; i++)
        x[i] = apus_bf16_bits((float)(i % 13) * 0.0625f);
    for (size_t t = 0; t < T; t++)
        apus_attn_rmsnorm(x + t * H, w.ln1_w, ln1 + t * H, H);

    ApusLayerState st;
    memset(&st, 0, sizeof st);
    st.conv_state = calloc(conv_dim * 3, 2);
    st.S = calloc(HV * DK * DV, sizeof(float));

    /* A: per-token mixer loop (the M4 per-token body's attention) */
    double t0 = now_s();
    for (size_t t = 0; t < T; t++)
        gdn_attn_tok(&c, &w, &st, ln1 + t * H, ao + t * H);
    double tA = now_s() - t0;

    /* B: the SHIPPED layer paths end-to-end — sequential vs batched
     * (attention dominates; the negligible MoE rides along). Fresh
     * states, then a bitwise memcmp of the outputs. */
    uint16_t *yb = malloc(T * H * 2), *ys = malloc(T * H * 2);
    ApusLayerState sb, ss;
    memset(&sb, 0, sizeof sb);
    memset(&ss, 0, sizeof ss);
    sb.conv_state = calloc(conv_dim * 3, 2);
    sb.S = calloc(HV * DK * DV, sizeof(float));
    ss.conv_state = calloc(conv_dim * 3, 2);
    ss.S = calloc(HV * DK * DV, sizeof(float));
    t0 = now_s();
    apus_layer_forward(&c, &w, &ss, x, ys, T, NULL);
    double tBseq = now_s() - t0;
    t0 = now_s();
    apus_layer_forward_hot(&c, &w, &sb, x, yb, T, NULL);
    double tBbat = now_s() - t0;
    int bit_ok = memcmp(yb, ys, T * H * 2) == 0
        && memcmp(sb.S, ss.S, HV * DK * DV * sizeof(float)) == 0
        && memcmp(sb.conv_state, ss.conv_state, conv_dim * 3 * 2) == 0;

    /* C: projections alone, batched */
    uint16_t *qkv = malloc(T * conv_dim * 2);
    uint16_t *z = malloc(T * value_dim * 2);
    uint16_t *b = malloc(T * HV * 2), *a = malloc(T * HV * 2);
    t0 = now_s();
    apus_bf16_gemm_hot(w.w_qkv, ln1, qkv, T, conv_dim, H);
    apus_bf16_gemm_hot(w.w_z, ln1, z, T, value_dim, H);
    apus_bf16_gemm_hot(w.w_b, ln1, b, T, HV, H);
    apus_bf16_gemm_hot(w.w_a, ln1, a, T, HV, H);
    double tC = now_s() - t0;

    /* D: sequential parts alone */
    uint16_t *qc = malloc(T * conv_dim * 2);
    uint16_t *beta_b = malloc(T * HV * 2);
    uint16_t *ob = malloc(value_dim * 2), *on = malloc(value_dim * 2);
    float *g = malloc(T * HV * sizeof(float));
    float *qn = malloc(HK * DK * sizeof(float));
    float *kn = malloc(HK * DK * sizeof(float));
    float *qr = malloc(HV * DK * sizeof(float));
    float *kr = malloc(HV * DK * sizeof(float));
    float *reco = malloc(value_dim * sizeof(float));
    float qscale = (float)pow((double)DK, -0.5);
    memset(st.S, 0, HV * DK * DV * sizeof(float));
    t0 = now_s();
    apus_gdn_conv1d(qkv, w.conv_w, qc, conv_dim, T, st.conv_state);
    for (size_t t = 0; t < T; t++) {
        apus_gdn_beta(b + t * HV, beta_b + t * HV, HV);
        apus_gdn_decay(a + t * HV, alog, dtb, g + t * HV, HV);
        apus_gdn_l2norm(qc + t * conv_dim, qn, HK, DK, qscale);
        apus_gdn_l2norm(qc + t * conv_dim + key_dim, kn, HK, DK, 1.0f);
        for (size_t h = 0; h < HK; h++)
            for (size_t r = 0; r < HV / HK; r++) {
                memcpy(qr + (h * (HV / HK) + r) * DK, qn + h * DK,
                       DK * sizeof(float));
                memcpy(kr + (h * (HV / HK) + r) * DK, kn + h * DK,
                       DK * sizeof(float));
            }
        apus_gdn_step_mt(st.S, qr, kr, qc + t * conv_dim + 2 * key_dim,
                         g + t * HV, beta_b + t * HV, reco, HV, DK, DV);
        for (size_t i = 0; i < value_dim; i++)
            ob[i] = apus_bf16_bits(reco[i]);
        apus_gdn_onorm(ob, z + t * value_dim, w.onorm_w, on, HV, DV);
    }
    double tD = now_s() - t0;

    printf("  A per-token mixer loop (replica)    : %8.3f s\n", tA);
    printf("  B layer sequential (apus_layer_fwd) : %8.3f s\n", tBseq);
    printf("  B layer batched (forward_hot, M9)   : %8.3f s  (%.2fx, "
           "outputs+state %s)\n", tBbat, tBseq / tBbat,
           bit_ok ? "BITWISE" : "MISMATCH!");
    printf("  C projections only (4x gemm M=T)    : %8.3f s\n", tC);
    printf("  D sequential parts (conv/step/etc)  : %8.3f s\n", tD);
    printf("  -> C+D = %.3f s vs A = %.3f s (overhead check)\n",
           tC + tD, tA);
    free(qkv); free(z); free(b); free(a); free(qc); free(beta_b);
    free(ob); free(on); free(g); free(qn); free(kn); free(qr); free(kr);
    free(reco); free(yb); free(ys);
    free(sb.conv_state); free(sb.S);
    free(ss.conv_state); free(ss.S);
    free(st.conv_state); free(st.S);

    /* E: phase-B MoE at REAL dims (256 experts, top-8, expert
     * gate_up [1024,2048] + down [2048,512]): per-token GEMVs (the
     * sequential body) vs the unique-expert batched GEMMs (the M9
     * restoration). BITWISE compare of the per-(t,slot) outputs. */
    {
        const size_t E = 256, I = 512, TK = 8;
        printf("  E phase-B MoE: E=%zu top-%zu I=%zu T=%zu ...\n",
               E, TK, I, T);
        uint16_t *rtr = wfill(E * H);
        uint16_t *egu = wfill(E * 2 * I * H);
        uint16_t *ed = wfill(E * H * I);
        int32_t *idx = malloc(T * TK * sizeof(int32_t));
        uint16_t *rw = malloc(T * TK * 2);
        uint16_t *eo1 = malloc(T * TK * H * 2);
        uint16_t *eo2 = malloc(T * TK * H * 2);
        uint16_t *gu = malloc(2 * I * 2), *act = malloc(I * 2);
        t0 = now_s();
        for (size_t t = 0; t < T; t++) {
            apus_moe_route(ln1 + t * H, rtr, idx + t * TK, rw + t * TK,
                           E, H, TK);
            for (size_t j = 0; j < TK; j++) {
                size_t e = (size_t)idx[t * TK + j];
                apus_bf16_gemv_hot(egu + e * 2 * I * H, ln1 + t * H, gu,
                                   2 * I, H);
                apus_moe_silu_act(gu, act, I);
                apus_bf16_gemv_hot(ed + e * H * I, act,
                                   eo1 + (t * TK + j) * H, H, I);
            }
        }
        double tE1 = now_s() - t0;
        uint16_t *xg = malloc(T * TK * H * 2);
        uint16_t *gub = malloc(T * TK * 2 * I * 2);
        uint16_t *ab = malloc(T * TK * I * 2);
        t0 = now_s();
        for (size_t e = 0; e < E; e++) {
            size_t cnt = 0;
            for (size_t t = 0; t < T; t++)
                for (size_t j = 0; j < TK; j++)
                    if ((size_t)idx[t * TK + j] == e) {
                        memcpy(xg + cnt * H, ln1 + t * H, H * 2);
                        cnt++;
                    }
            if (!cnt) continue;
            apus_bf16_gemm_hot(egu + e * 2 * I * H, xg, gub, cnt,
                               2 * I, H);
            for (size_t r = 0; r < cnt; r++)
                apus_moe_silu_act(gub + r * 2 * I, ab + r * I, I);
            uint16_t *eb = malloc(cnt * H * 2);
            apus_bf16_gemm_hot(ed + e * H * I, ab, eb, cnt, H, I);
            size_t r = 0;
            for (size_t t = 0; t < T; t++)
                for (size_t j = 0; j < TK; j++)
                    if ((size_t)idx[t * TK + j] == e) {
                        memcpy(eo2 + (t * TK + j) * H, eb + r * H, H * 2);
                        r++;
                    }
            free(eb);
        }
        double tE2 = now_s() - t0;
        int eok = memcmp(eo1, eo2, T * TK * H * 2) == 0;
        printf("  E per-token expert GEMVs            : %8.3f s\n", tE1);
        printf("  E unique-expert batched GEMMs (M9)  : %8.3f s  (%.2fx, "
               "outputs %s)\n", tE2, tE1 / tE2,
               eok ? "BITWISE" : "MISMATCH!");
        free(rtr); free(egu); free(ed);
        free(idx); free(rw); free(eo1); free(eo2);
        free(gu); free(act); free(xg); free(gub); free(ab);
    }

    free(x); free(ao); free(ln1);
    free(alog); free(dtb);
    free((void *)w.w_qkv); free((void *)w.w_z); free((void *)w.w_b);
    free((void *)w.w_a); free((void *)w.conv_w); free((void *)w.onorm_w);
    free((void *)w.w_out); free((void *)w.ln1_w); free((void *)w.ln2_w);
    free((void *)w.rtr_w); free((void *)w.exp_gu); free((void *)w.exp_d);
    free((void *)w.sh_g); free((void *)w.sh_u); free((void *)w.sh_d);
    free((void *)w.sh_gate);
    return 0;
}
