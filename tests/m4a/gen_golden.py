#!/usr/bin/env python3
"""M4a golden generator — float64 numpy oracles for the GDN / GQA / MoE
per-operation kernels (c/gdn.h, c/attn.h, c/moe.h; docs/M4-CONTRACT.md,
tools/oracle.py). Replaces the Ling KDA/MLA generator.

The normative numerics anchors live in C; this script provides the
INDEPENDENT float64 truth the C tests are gated against. Every op is
evaluated here in float64, following the same operation order the C
header pins (ascending sequential sums; the same bf16 roundings where
the contract has them, applied with the m3 float64 candidate-distance
RNE oracle — deliberately NOT the C bit-trick, so a bug in one cannot
hide in the other). The remaining fp32-vs-f64 differences (one rounding
per fp32 op, libm expf/sinf/cosf/log1pf ulps) are the tolerance classes
documented in tests/m4a/README.md; where the C op is exact (decode ==
prefill, decode == recompute, scalar == NEON, mt == sequential) the C
tests gate bitwise without involving this oracle.

Fixture layout mirrors the C kernel signatures (row-major, heads folded
into the leading extent):

  gdn_conv_x.bin       u16 [T,C]       pre-conv activations (seq 1)
  gdn_conv_w.bin       u16 [C,4]       depthwise taps
  gdn_conv_out.bin     f64 [T,C]       silu(conv(x)) truth (two rounds)
  gdn_conv_esc.bin     f64 [T,C]       error scale sum|w*win|
  gdn_conv_x2.bin      u16 [T2,C]      continuation tokens (state wrap)
  gdn_conv_out2.bin    f64 [T2,C]      truth continuing the state
  gdn_decay_a/A/dt/g   u16/f32/f32/f64 [H]/[H]/[H]/[H]
  gdn_beta_b/beta      u16 [H] / f64 [H]  (bf16 sigmoid values)
  gdn_l2_x/yq/yk       u16 [H,D] / f64 [H,D] / f64 [H,D]
  gdn_rec_q/k          f32 [T,H,D]     (post-l2norm, q pre-scaled)
  gdn_rec_v/beta       u16 [T,H,D] / u16 [T,H]
  gdn_rec_g            f32 [T,H]       (per-head scalar decay)
  gdn_rec_o/esco       f64 [T,H,D]     o truth / error scale
  gdn_rec_S/escS       f64 [H,D,D]     final state truth / error scale
  gdn_onorm_o/z/w/y    u16/u16/u16/f64 [H,D] (w is [D], DIRECT)
  attn_rms_x/w/y       u16/u16/f64 [N]     ((1+w) variant, single round)
  attn_rope_x          u16 [64]; attn_rope_pos f64 [P]
  attn_rope_cs.bin     u16 [P,2,8]     bf16 cos/sin codes per position
  attn_rope_y.bin      f64 [P,64]      (D=64, rot=16, theta=1e7)
  attn_gqa_q/k/v       u16 [T,H,D] / [T,Hkv,D] / [T,Hkv,D]
  attn_gqa_A.bin       f64 [T,H,T]     post-scale A values (causal rows)
  attn_gqa_escA.bin    f64 [T,H,T]     scale * sum|q*k|
  attn_gqa_o/esco      f64 [T,H,D]
  attn_og_o/gl/y       u16 [N] / u16 [N] / f64 [N]  (elementwise gate)
  moe_rtr_x/wg         u16 [K] / u16 [E,K]
  moe_rtr_logits.bin   u16 [E]         (bf16 codes — the C rounds)
  moe_rtr_probs.bin    f64 [E]
  moe_rtr_idx.bin      i64 [8]; moe_rtr_w.bin f64 [8] (bf16-rounded)
  moe_act_gu/act       u16 [2I] / f64 [I]  (routed, two rounds)
  moe_silu_g/u/y       u16 [I] / f64 [I]   (shared, single round)
  moe_comb_y/w/out/esc u16 [k,N] / u16 [k] / f64 [N] / f64 [N]

Router fixture generation verifies the selection margin (top-8 vs 9th
prob) exceeds 1e-6 so fp32-vs-f64 softmax noise cannot flip the
selection — goldens avoid exact ties per the contract.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "golden")

# --- bf16 helpers (same float64-distance RNE oracle as tests/m3, kept
#     independent of the C bit-trick) ----------------------------------------

INF_TIE_THRESHOLD = np.float64(2.0 ** 128) - np.float64(2.0 ** 119)


def bf16_widen_f64(codes):
    """uint16 bf16 codes -> float64 values, exact (via float32)."""
    u = (codes.astype(np.uint32) << np.uint16(16).astype(np.uint32))
    return u.view(np.float32).astype(np.float64)


def bf16_narrow_oracle_f32(f32):
    """float32 array -> bf16 codes, RNE, float64 candidate distances."""
    return bf16_narrow_oracle_bits(np.asarray(f32, dtype=np.float32)
                                   .view(np.uint32))


def bf16_narrow_oracle_bits(u32):
    u = u32.astype(np.uint32)
    mag = u & np.uint32(0x7FFFFFFF)
    sign = (u >> np.uint32(31)) << np.uint32(15)
    is_nan = mag > np.uint32(0x7F800000)
    with np.errstate(invalid="ignore"):
        v = mag.view(np.float32).astype(np.float64)
        a = np.abs(v)
        lo = mag >> np.uint32(16)
        hi = lo + np.uint32(1)
        vlo = (lo << np.uint32(16)).view(np.float32).astype(np.float64)
        is_inf = mag == np.uint32(0x7F800000)
        sat = lo == np.uint32(0x7F7F)
        dlo = a - vlo
        vhi = (np.minimum(hi, np.uint32(0x7F7F)) << np.uint32(16))
        vhi = vhi.view(np.float32).astype(np.float64)
        dhi = vhi - a
    pick_hi_sat = a >= INF_TIE_THRESHOLD
    pick_hi = ((dhi < dlo) | ((dhi == dlo) & ((hi & np.uint32(1)) == 0)))
    pick_hi = np.where(sat, pick_hi_sat, pick_hi)
    pick_hi = np.where(is_inf, False, pick_hi)
    code = np.where(pick_hi, hi, lo) | sign
    return np.where(is_nan, (u >> np.uint32(16)), code).astype(np.uint16)


def rnd16(f64):
    """float64 -> bf16-rounded float64 (narrow oracle then widen)."""
    f32 = np.asarray(f64, dtype=np.float64).astype(np.float32)
    return bf16_widen_f64(bf16_narrow_oracle_f32(f32))


def sigmoid(x):
    with np.errstate(over="ignore"):   # exp -> inf gives the correct 0
        return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    """F.softplus: log(1+exp(x)), x for x>20 (logaddexp branch)."""
    return np.logaddexp(0.0, x)


def rand_bf16(rng, shape, scale=1.0):
    f = (rng.standard_normal(shape) * scale).astype(np.float32)
    return bf16_narrow_oracle_f32(f)


def dump(name, arr):
    np.asarray(arr).tofile(os.path.join(OUT, name))


# --- GDN ---------------------------------------------------------------------

def gen_gdn_conv(rng, mf):
    C, T, T2 = 64, 8, 3
    x = rand_bf16(rng, (T, C), 1.5)
    w = rand_bf16(rng, (C, 4), 0.7)
    x2 = rand_bf16(rng, (T2, C), 1.5)
    xw = bf16_widen_f64(x)
    ww = bf16_widen_f64(w)
    x2w = bf16_widen_f64(x2)

    def run(xseq, state):
        Tl = xseq.shape[0]
        out = np.zeros((Tl, C))
        esc = np.zeros((Tl, C))
        st = state.copy()
        for t in range(Tl):
            for c in range(C):
                win = list(st[c]) + [xseq[t, c]]
                acc = 0.0
                sc = 0.0
                for i in range(4):
                    p = ww[c, i] * win[i]
                    acc += p
                    sc += abs(p)
                co = rnd16(acc)                      # rnd #1
                out[t, c] = rnd16(co * sigmoid(co))  # rnd #2
                esc[t, c] = sc
            st = np.concatenate([st[:, 1:], xseq[t, :, None]], axis=1)
        return out, esc, st

    state0 = np.zeros((C, 3))
    out, esc, state1 = run(xw, state0)
    out2, esc2, _ = run(x2w, state1)

    dump("gdn_conv_x.bin", x)
    dump("gdn_conv_w.bin", w)
    dump("gdn_conv_x2.bin", x2)
    dump("gdn_conv_out.bin", out)
    dump("gdn_conv_esc.bin", esc)
    dump("gdn_conv_out2.bin", out2)
    dump("gdn_conv_esc2.bin", esc2)
    mf["CONV_C"] = C
    mf["CONV_T"] = T
    mf["CONV_T2"] = T2
    print(f"gdn conv: C={C} T={T}+{T2} out range "
          f"[{out.min():.6g}, {out.max():.6g}]")


def gen_gdn_decay(rng, mf):
    H = 8
    a = rand_bf16(rng, H, 2.0)
    A_log = np.log(rng.uniform(0.01, 16.0, H)).astype(np.float32)
    dt = (rng.standard_normal(H) * 1.0).astype(np.float32)
    aw = bf16_widen_f64(a)
    # corners: a+dt == 0 -> softplus(0) = log 2; a+dt = 25 -> the x>20
    # identity branch; a+dt = -25 -> the exp tail
    dt[0] = np.float32(-aw[0])
    a[1] = bf16_narrow_oracle_f32(np.float32(24.0))
    dt[1] = np.float32(1.0)
    a[2] = bf16_narrow_oracle_f32(np.float32(-24.0))
    dt[2] = np.float32(-1.0)
    aw = bf16_widen_f64(a)
    g = (-np.exp(A_log.astype(np.float64))
         * softplus(aw + dt.astype(np.float64)))
    dump("gdn_decay_a.bin", a)
    dump("gdn_decay_alog.bin", A_log)
    dump("gdn_decay_dt.bin", dt)
    dump("gdn_decay_g.bin", g)
    mf["DECAY_H"] = H
    print(f"gdn decay: H={H} g range [{g.min():.6g}, {g.max():.6g}]")


def gen_gdn_beta(rng, mf):
    H = 8
    b = rand_bf16(rng, H, 3.0)
    b[0] = np.uint16(0x0000)         # logit 0 -> sigmoid exactly 0.5
    b[1] = np.uint16(0x41A0)         # +20 -> sigmoid ~ 1
    b[2] = np.uint16(0xC1A0)         # -20 -> sigmoid ~ 0
    beta = rnd16(sigmoid(bf16_widen_f64(b)))
    dump("gdn_beta_b.bin", b)
    dump("gdn_beta_beta.bin", beta)
    mf["BETA_H"] = H
    print(f"gdn beta: H={H} beta range [{beta.min():.6g}, {beta.max():.6g}]")


def gen_gdn_l2(rng, mf):
    H, D = 2, 128
    x = rand_bf16(rng, (H, D), 1.5)
    x[1, :] = bf16_narrow_oracle_f32(
        (rng.standard_normal(D) * 1e-10).astype(np.float32))   # tiny row
    xw = bf16_widen_f64(x)
    ss = (xw ** 2).sum(axis=1)
    yn = xw * (1.0 / np.sqrt(ss + 1e-6))[:, None]
    yq = yn * np.float64(D) ** -0.5          # the q scale
    dump("gdn_l2_x.bin", x)
    dump("gdn_l2_yq.bin", yq)
    dump("gdn_l2_yk.bin", yn)
    mf["L2_H"] = H
    mf["L2_D"] = D
    print(f"gdn l2norm: H={H} D={D} y range [{yn.min():.6g}, {yn.max():.6g}]")


def gen_gdn_rec(rng, mf):
    H, D, T = 2, 8, 6
    # q,k fp32 (post-l2norm; q pre-scaled) — sane post-norm magnitudes
    q = (rng.standard_normal((T, H, D)) * 0.3).astype(np.float32)
    k = (rng.standard_normal((T, H, D)) * 0.3).astype(np.float32)
    v = rand_bf16(rng, (T, H, D), 1.5)
    g = (rng.uniform(-5.0, 0.0, (T, H))).astype(np.float32)
    g[0] = 0.0                       # no decay
    g[1] = -5.0                      # max decay
    beta = rand_bf16(rng, (T, H), 0.5)
    beta[2, 0] = np.uint16(0x0000)   # no update corner
    beta[3, 1] = bf16_narrow_oracle_f32(np.float32(1.0))  # full replace

    qw, kw = q.astype(np.float64), k.astype(np.float64)
    vw, bw = bf16_widen_f64(v), bf16_widen_f64(beta)
    S = np.zeros((H, D, D))
    o = np.zeros((T, H, D))
    esco = np.zeros((T, H, D))
    for t in range(T):
        for h in range(H):
            Sh = S[h]
            Sh *= np.exp(np.float64(g[t, h]))
            for j in range(D):
                acc = 0.0
                for i in range(D):
                    acc += kw[t, h, i] * Sh[i, j]
                delta = (vw[t, h, j] - acc) * bw[t, h]
                for i in range(D):
                    Sh[i, j] += kw[t, h, i] * delta
            for j in range(D):
                acc = 0.0
                sc = 0.0
                for i in range(D):
                    p = qw[t, h, i] * Sh[i, j]
                    acc += p
                    sc += abs(p)
                o[t, h, j] = acc
                esco[t, h, j] = sc
    escS = np.abs(S) + 1e-300

    dump("gdn_rec_q.bin", q)
    dump("gdn_rec_k.bin", k)
    dump("gdn_rec_v.bin", v)
    dump("gdn_rec_g.bin", g)
    dump("gdn_rec_beta.bin", beta)
    dump("gdn_rec_o.bin", o)
    dump("gdn_rec_esco.bin", esco)
    dump("gdn_rec_S.bin", S)
    dump("gdn_rec_escS.bin", escS)
    mf["REC_H"] = H
    mf["REC_D"] = D
    mf["REC_T"] = T
    print(f"gdn rec: H={H} D={D} T={T} o range [{o.min():.6g}, "
          f"{o.max():.6g}] S range [{S.min():.6g}, {S.max():.6g}]")


def gen_gdn_onorm(rng, mf):
    H, D = 2, 128
    o = rand_bf16(rng, (H, D), 1.5)
    o[1, :] = rand_bf16(rng, D, 1e-3)            # tiny-magnitude head
    z = rand_bf16(rng, (H, D), 2.0)
    w = rand_bf16(rng, D, 0.8)                   # [D] DIRECT weight
    ow, zw, ww = bf16_widen_f64(o), bf16_widen_f64(z), bf16_widen_f64(w)
    ss = (ow ** 2).sum(axis=1)
    rs = 1.0 / np.sqrt(ss / D + 1e-6)
    y = np.zeros((H, D))
    for h in range(H):
        for d in range(D):
            x1 = rnd16(ow[h, d] * rs[h])         # normalize, rnd
            x2 = rnd16(ww[d] * x1)               # x direct weight, rnd
            y[h, d] = rnd16(x2 * (zw[h, d] * sigmoid(zw[h, d])))
    dump("gdn_onorm_o.bin", o)
    dump("gdn_onorm_z.bin", z)
    dump("gdn_onorm_w.bin", w)
    dump("gdn_onorm_y.bin", y)
    mf["ONORM_H"] = H
    mf["ONORM_D"] = D
    print(f"gdn onorm: H={H} D={D} y range [{y.min():.6g}, {y.max():.6g}]")


# --- attention ---------------------------------------------------------------

def gen_attn_rms(rng, mf):
    N = 512
    x = rand_bf16(rng, N, 2.0)
    w = rand_bf16(rng, N, 0.1)       # (1+w) stored values: N(0, 0.1)
    xw = bf16_widen_f64(x)
    ss = (xw ** 2).sum()
    xh = xw * (1.0 / np.sqrt(ss / N + 1e-6))
    y = rnd16(xh * (1.0 + bf16_widen_f64(w)))   # SINGLE rounding
    dump("attn_rms_x.bin", x)
    dump("attn_rms_w.bin", w)
    dump("attn_rms_y.bin", y)
    mf["RMS_N"] = N
    print(f"attn rmsnorm: N={N} y range [{y.min():.6g}, {y.max():.6g}]")


def gen_attn_rope(rng, mf):
    P = 6
    D, ROT = 64, 16
    positions = np.array([0.0, 1.0, 7.0, 1000.0, 131071.0, 262143.0])
    x = rand_bf16(rng, D, 1.5)
    xw = bf16_widen_f64(x)
    i = np.arange(ROT // 2)
    # the contract computes freqs in fp32 (HF casts inv_freq and
    # position_ids to float32): replicate the fp32 inv_freq and the fp32
    # angle product, then evaluate cos/sin in float64 — isolating libm
    # ulps + bf16 rounding as the only remaining tolerance class
    invf32 = (1.0 / (10000000.0 ** (2.0 * i / ROT))).astype(np.float32)
    cs = np.zeros((P, 2, ROT // 2), dtype=np.uint16)
    y = np.zeros((P, D))
    for p in range(P):
        ang = (np.float32(positions[p]) * invf32).astype(np.float64)
        c = bf16_widen_f64(bf16_narrow_oracle_f32(np.cos(ang)))
        s = bf16_widen_f64(bf16_narrow_oracle_f32(np.sin(ang)))
        cs[p, 0] = bf16_narrow_oracle_f32(np.cos(ang))
        cs[p, 1] = bf16_narrow_oracle_f32(np.sin(ang))
        x0, x1 = xw[:ROT // 2], xw[ROT // 2:ROT]
        y[p, :ROT // 2] = rnd16(rnd16(x0 * c) + rnd16(-x1 * s))
        y[p, ROT // 2:ROT] = rnd16(rnd16(x1 * c) + rnd16(x0 * s))
        y[p, ROT:] = xw[ROT:]
    dump("attn_rope_x.bin", x)
    dump("attn_rope_pos.bin", positions)
    dump("attn_rope_cs.bin", cs)
    dump("attn_rope_y.bin", y)
    mf["ROPE_P"] = P
    mf["ROPE_D"] = D
    mf["ROPE_ROT"] = ROT
    print(f"attn rope: positions {positions.tolist()} y range "
          f"[{y.min():.6g}, {y.max():.6g}]")


def gen_attn_gqa(rng, mf):
    H, HKV, T, D = 4, 2, 6, 64
    q = rand_bf16(rng, (T, H, D), 1.0)
    k = rand_bf16(rng, (T, HKV, D), 1.0)
    v = rand_bf16(rng, (T, HKV, D), 1.2)
    qw, kw, vw = bf16_widen_f64(q), bf16_widen_f64(k), bf16_widen_f64(v)
    scale = np.float64(D) ** -0.5
    A = np.zeros((T, H, T))
    escA = np.zeros((T, H, T))
    o = np.zeros((T, H, D))
    esco = np.zeros((T, H, D))
    for t in range(T):
        for h in range(H):
            hk = h // (H // HKV)
            for j in range(t + 1):
                acc = 0.0
                sc = 0.0
                for d in range(D):
                    p = qw[t, h, d] * kw[j, hk, d]
                    acc += p
                    sc += abs(p)
                A[t, h, j] = rnd16(rnd16(acc) * scale)
                escA[t, h, j] = sc * scale
            m = A[t, h, :t + 1].max()
            e = np.exp(A[t, h, :t + 1] - m)
            ssum = e.sum()
            P = rnd16(e / ssum)
            for d in range(D):
                acc = 0.0
                sc = 0.0
                for j in range(t + 1):
                    p = P[j] * vw[j, hk, d]
                    acc += p
                    sc += abs(p)
                o[t, h, d] = acc
                esco[t, h, d] = sc
    dump("attn_gqa_q.bin", q)
    dump("attn_gqa_k.bin", k)
    dump("attn_gqa_v.bin", v)
    dump("attn_gqa_A.bin", A)
    dump("attn_gqa_escA.bin", escA)
    dump("attn_gqa_o.bin", o)
    dump("attn_gqa_esco.bin", esco)
    mf["GQA_H"] = H
    mf["GQA_HKV"] = HKV
    mf["GQA_T"] = T
    mf["GQA_D"] = D
    print(f"attn gqa: H={H} Hkv={HKV} T={T} o range "
          f"[{o.min():.6g}, {o.max():.6g}]")


def gen_attn_og(rng, mf):
    N = 512
    o = rand_bf16(rng, N, 1.5)
    gl = rand_bf16(rng, N, 2.0)
    gl[0] = np.uint16(0x0000)        # logit 0 -> sigmoid exactly 0.5
    gl[1] = np.uint16(0x41A0)        # +20 -> sigmoid ~ 1
    gl[2] = np.uint16(0xC1A0)        # -20 -> sigmoid ~ 0
    gb = rnd16(sigmoid(bf16_widen_f64(gl)))
    y = rnd16(bf16_widen_f64(o) * gb)
    dump("attn_og_o.bin", o)
    dump("attn_og_gl.bin", gl)
    dump("attn_og_y.bin", y)
    mf["OG_N"] = N
    print(f"attn outgate: N={N} y range [{y.min():.6g}, {y.max():.6g}]")


# --- MoE ---------------------------------------------------------------------

def gen_moe_router(rng, mf):
    E, K, TK = 256, 128, 8
    x = rand_bf16(rng, K, 1.0)
    wg = rand_bf16(rng, (E, K), 0.08)
    xw = bf16_widen_f64(x)
    wgw = bf16_widen_f64(wg)
    logits = np.zeros(E)
    for e in range(E):
        acc = 0.0
        for k in range(K):
            acc += wgw[e, k] * xw[k]
        logits[e] = acc
    logits_b = rnd16(logits)          # the router's logits are bf16
    z = logits_b - logits_b.max()
    e = np.exp(z)
    probs = e / e.sum()
    idx = np.argsort(-probs, kind="stable")[:TK]
    w = probs[idx] / probs[idx].sum()
    w = rnd16(w)

    # selection margin must dwarf fp32-vs-f64 softmax noise
    psort = np.sort(probs)[::-1]
    margin = psort[TK - 1] - psort[TK]
    assert margin > 1e-6, f"router margin too small: {margin}"

    dump("moe_rtr_x.bin", x)
    dump("moe_rtr_wg.bin", wg)
    dump("moe_rtr_logits.bin", bf16_narrow_oracle_f32(
        logits.astype(np.float32)))
    dump("moe_rtr_probs.bin", probs)
    dump("moe_rtr_idx.bin", idx.astype(np.int64))
    dump("moe_rtr_w.bin", w)
    mf["RTR_E"] = E
    mf["RTR_K"] = K
    mf["RTR_TK"] = TK
    print(f"moe router: E={E} K={K} idx {idx.tolist()} margin "
          f"{margin:.6g}")


def gen_moe_act(rng, mf):
    I = 768
    gu = rand_bf16(rng, 2 * I, 3.0)
    gw = bf16_widen_f64(gu)
    a1 = rnd16(gw[:I] * sigmoid(gw[:I]))     # rnd(silu(gate))
    act = rnd16(a1 * gw[I:])                 # rnd(a1 * up)
    dump("moe_act_gu.bin", gu)
    dump("moe_act_act.bin", act)
    mf["ACT_I"] = I
    print(f"moe act (routed): I={I} range [{act.min():.6g}, "
          f"{act.max():.6g}]")


def gen_moe_silu(rng, mf):
    I = 768
    g = rand_bf16(rng, I, 3.0)
    u = rand_bf16(rng, I, 3.0)
    gw, uw = bf16_widen_f64(g), bf16_widen_f64(u)
    y = rnd16(gw * sigmoid(gw) * uw)         # SINGLE rounding
    dump("moe_silu_g.bin", g)
    dump("moe_silu_u.bin", u)
    dump("moe_silu_y.bin", y)
    mf["SILU_I"] = I
    print(f"moe silu (shared): I={I} y range [{y.min():.6g}, "
          f"{y.max():.6g}]")


def gen_moe_combine(rng, mf):
    K_, N = 8, 768
    y = rand_bf16(rng, (K_, N), 2.0)
    w = rand_bf16(rng, K_, 0.4)              # bf16 routing weights
    yw = bf16_widen_f64(y)
    ww = bf16_widen_f64(w)
    out = np.zeros(N)
    esc = np.zeros(N)
    for n in range(N):
        acc = 0.0
        sc = 0.0
        for e in range(K_):
            p = ww[e] * yw[e, n]
            acc += p
            sc += abs(p)
        out[n] = acc
        esc[n] = sc
    dump("moe_comb_y.bin", y)
    dump("moe_comb_w.bin", w)
    dump("moe_comb_out.bin", out)
    dump("moe_comb_esc.bin", esc)
    mf["COMB_K"] = K_
    mf["COMB_N"] = N
    print(f"moe combine: k={K_} N={N} out range "
          f"[{out.min():.6g}, {out.max():.6g}]")


def main():
    rng = np.random.default_rng(20260829)
    os.makedirs(OUT, exist_ok=True)
    mf = {}
    gen_gdn_conv(rng, mf)
    gen_gdn_decay(rng, mf)
    gen_gdn_beta(rng, mf)
    gen_gdn_l2(rng, mf)
    gen_gdn_rec(rng, mf)
    gen_gdn_onorm(rng, mf)
    gen_attn_rms(rng, mf)
    gen_attn_rope(rng, mf)
    gen_attn_gqa(rng, mf)
    gen_attn_og(rng, mf)
    gen_moe_router(rng, mf)
    gen_moe_act(rng, mf)
    gen_moe_silu(rng, mf)
    gen_moe_combine(rng, mf)
    with open(os.path.join(OUT, "manifest.txt"), "w") as f:
        for k, v in mf.items():
            f.write(f"{k}={v}\n")
    print("manifest written")


if __name__ == "__main__":
    main()
