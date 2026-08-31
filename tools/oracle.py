#!/usr/bin/env python3
"""tools/oracle.py — numpy oracle for the Qwen3.6-35B-A3B text model
(model_type: qwen3_5_moe, text-only), milestone M0. Covers one decoder
layer of each kind and the stacked mini-model:

  gdn  : Gated DeltaNet linear attention + MoE   (layer_types
         "linear_attention": 30 of the real 40 layers)
  full : gated GQA full attention + MoE          (layer_types
         "full_attention": layers 3,7,...,39)

Written directly from the HF reference — deliberately NOT from the C
sources, so generator and C cannot share a consistent-but-wrong
convention (the Apus M4c lesson). Normative sources:

  reference/modeling_qwen3_5_moe.py (transformers main, self-contained
    text model):
    RMSNorm (1+w variant)            L878-895
    RMSNormGated (GDN output norm)   L175-192
    causal conv1d (prefill/decode)   L207-247
    l2norm                           L250-255
    recurrent gated delta rule       L395-455  (THE contract — the
      chunk path L258-392 is NOT bitwise equal and is NOT used)
    GatedDeltaNet mixer              L458-618
    rotate_half / partial RoPE       L621-664, rotary emb L91-171
    eager attention / repeat_kv      L667-701
    gated GQA attention              L704-778
    experts / router / MoE block     L781-875
    decoder layer wiring             L898-954
    text model (embed/norm/layers)   L1315-1400
    lm_head (no fp32 upcast)         L1900-1903
  reference/configuration_qwen3_5_moe.py — real config defaults
    (layer_types derivation L123-134, partial_rotary_factor 0.25 L124).
  reference/config.json — the real 40-layer config (schema the fixture
    config.json uses, nested text_config and all).
  reference/ling/fla/naive_gdr.py — fla naive recurrent gated delta
    rule (cross-check of the recurrence semantics only).

Structure and golden-IO conventions follow the Ling base oracle (the
file this replaces, git history): same bf16 helpers, same two numeric
modes, same deterministic _mm, same layer/model entry-point shape so
the tests/m4b/m5 generators can be adapted with minimal churn.

Two numeric modes (Apus M4b methodology):

  f32 — dtype-faithful, the C target: bf16 rounding (RNE, the
        candidate-distance oracle, NOT the C bit-trick) at every point
        where the HF BF16 implementation materializes a bf16 tensor,
        fp32 (st()) where HF keeps fp32 (router softmax, attention
        softmax input, GDN decay/recurrence/state, l2norm, norm
        variances), matmuls accumulated in f64 and rounded to the
        storage dtype (a deterministic, platform-independent "ideal
        fp32" realization).
  f64 — truth: identical algorithm, all arithmetic in float64, no
        bf16 rounding anywhere.

The per-stage |f32 - f64| envelope is the scale the C forward must
land inside of (C vs f32 golden <= envelope + realization slack).

Documented realization choices (reorder classes, absorbed by the
envelope, NOT semantic deviations):
  * MoE routed-expert weighted sum: accumulated fp32 across the top-k
    experts with a single bf16 round (HF index_add_ into a bf16
    buffer, modeling L816-834, rounds per expert; order and per-add
    rounding differ — same class as the base oracle's MoE combine).
  * torch's fp32 tree reductions (softmax denominators, state sums)
    are realized as f64 sequential sums then st(); per-op IEEE-exact,
    platform-identical.
  * F.softplus == logaddexp(0, x) including the x>20 branch (the
    log1p(exp(-x)) tail is below fp32 resolution there).
  * Router top-k tie-break: lowest index wins (stable argsort), the
    base oracle's documented convention.

Fixture mini-config (MINI_CFG, real config.json text schema): hidden
128; full attn 4 q heads / 2 kv heads x head_dim 64, partial rotary 16
(factor 0.25), theta 1e7; GDN 8 v heads / 4 k heads x 32 (repeat x2),
conv k=4; MoE 16 experts top-4, moe_inter 32, shared expert 32; 2
layers [linear_attention, full_attention]; vocab 256.

MTP (model mtp.*: fc, pre_fc_norm_embedding/hidden, layers.0 full
attention + MoE, norm; shared embeddings/lm_head) is OUT OF SCOPE for
M0 (deferred milestone M8) — extension point marked at the bottom.
Generation/sampling: out of scope.
"""

import json

import numpy as np

# --------------------------------------------------------------------------
# bf16 helpers: RNE via float64 candidate distances (independent of the C
# bit-trick, same as tests/m3 / tests/m4a oracles). Unchanged from the
# Ling base oracle — model-agnostic.

INF_TIE_THRESHOLD = np.float64(2.0 ** 128) - np.float64(2.0 ** 119)


def bf16_widen_f64(codes):
    u = codes.astype(np.uint32) << np.uint32(16)
    return u.view(np.float32).astype(np.float64)


def bf16_narrow_oracle_f32(f32):
    return bf16_narrow_oracle_bits(
        np.asarray(f32, dtype=np.float32).view(np.uint32))


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
    """float64 -> bf16-rounded float64 (values, not codes)."""
    f32 = np.asarray(f64, dtype=np.float64).astype(np.float32)
    return bf16_widen_f64(bf16_narrow_oracle_f32(f32))


def f32(f64):
    return np.asarray(f64, dtype=np.float64).astype(np.float32).astype(
        np.float64)


def _mm(a, b):
    """Cross-platform DETERMINISTIC matmul (M12a-1, the Apus M12b fix).

    numpy's `@`/einsum use BLAS (Accelerate/OpenBLAS) or SIMD reduction
    loops whose summation order varies by platform; the resulting ~1-ulp
    differences flip bf16 rounding ties and top-k selections, and those
    flips cascade through the random-weight fixtures. Here the reduction
    runs in the input dtype (float64 throughout this oracle) with a FIXED
    sequential k-order using only broadcast elementwise ops — per output
    element each step is a single IEEE-exact fp64 multiply or add,
    identical on every platform — so fixtures regenerate
    bitwise-identically on macOS and Linux.
    """
    dt = np.result_type(a, b)
    a_dt = np.asarray(a, dtype=dt)
    b_dt = np.asarray(b, dtype=dt)
    acc = np.zeros(a_dt.shape[:-1] + b_dt.shape[1:], dtype=dt)
    for k in range(a_dt.shape[-1]):
        acc += a_dt[..., k:k + 1] * b_dt[k]
    return acc


def sigmoid(x):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    """F.softplus: log(1+exp(x)), x for x>20. logaddexp(0,x) matches to
    below fp32 resolution in the tail (exp(-20) ~ 2e-9)."""
    return np.logaddexp(0.0, x)


# --------------------------------------------------------------------------
# config (real config.json text_config schema; MINI_CFG = the synthetic
# fixture config). All dims are read from cfg — the same code path runs
# the real 40-layer config and the tiny fixture config.

MINI_CFG = dict(
    model_type="qwen3_5_moe_text",
    vocab_size=256,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    linear_num_key_heads=4,
    linear_num_value_heads=8,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    num_experts=16,
    num_experts_per_tok=4,
    rms_norm_eps=1e-6,
    max_position_embeddings=262144,
    hidden_act="silu",
    attention_bias=False,
    attention_dropout=0.0,
    tie_word_embeddings=False,
    rope_parameters=dict(rope_type="default", rope_theta=10000000.0,
                         partial_rotary_factor=0.25),
    layer_types=["linear_attention", "full_attention"],
    # fixture sequencing (not config.json keys; ignored by parsers)
    prefill_len=7, decode_len=5,
)

KINDS = ["gdn", "full"]

_KIND_OF_TYPE = {"linear_attention": "gdn", "full_attention": "full"}


def config_from_json(path):
    """Load a real-schema config.json (nested text_config, as in
    reference/config.json) or a bare text config, fill the defaults the
    oracle relies on (configuration_qwen3_5_moe.py L90-134), and return
    the flat text-config dict that drives every oracle entry point."""
    with open(path) as f:
        obj = json.load(f)
    cfg = dict(obj.get("text_config", obj))
    cfg.setdefault("hidden_act", "silu")
    cfg.setdefault("rms_norm_eps", 1e-6)
    cfg.setdefault("attention_bias", False)
    cfg.setdefault("linear_conv_kernel_dim", 4)
    cfg.setdefault("head_dim",
                   cfg["hidden_size"] // cfg["num_attention_heads"])
    rp = dict(cfg.get("rope_parameters") or {})
    rp.setdefault("rope_theta", 10000.0)
    rp.setdefault("partial_rotary_factor", 0.25)   # BC default, L124
    cfg["rope_parameters"] = rp
    if "layer_types" not in cfg or cfg["layer_types"] is None:
        interval = cfg.get("full_attention_interval", 4)
        cfg["layer_types"] = [
            "linear_attention" if (i + 1) % interval else "full_attention"
            for i in range(cfg["num_hidden_layers"])]
    return cfg


def layer_kind(cfg, idx):
    return _KIND_OF_TYPE[cfg["layer_types"][idx]]


def gdn_dims(cfg):
    """Derived Gated DeltaNet dims (modeling L462-468)."""
    hk = cfg["linear_num_key_heads"]
    hv = cfg["linear_num_value_heads"]
    dk = cfg["linear_key_head_dim"]
    dv = cfg["linear_value_head_dim"]
    key_dim = hk * dk
    value_dim = hv * dv
    return dict(hk=hk, hv=hv, dk=dk, dv=dv, key_dim=key_dim,
                value_dim=value_dim, conv_dim=2 * key_dim + value_dim,
                rep=hv // hk, K=cfg["linear_conv_kernel_dim"])


def attn_dims(cfg):
    nh = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    d = cfg["head_dim"]
    rot = int(d * cfg["rope_parameters"]["partial_rotary_factor"])
    return dict(nh=nh, nkv=nkv, d=d, rot=rot, nrep=nh // nkv,
                theta=cfg["rope_parameters"]["rope_theta"])


# --------------------------------------------------------------------------
# op-level ports (HF semantics; `rnd` is rnd16 in f32 mode, identity in
# f64 mode; `st` is f32() in f32 mode, identity in f64 mode)

def rmsnorm(x, w, rnd, eps):
    """HF Qwen3_5MoeRMSNorm (L878-892), the (1+w) zero-init variant:
    fp32 normalize, fp32 x*(1+w), SINGLE cast to bf16 at the end (L889:
    "(x * w).to(float16)"). w is the RAW stored value (gain-1); the +1
    is applied here."""
    ss = (x ** 2).mean(axis=-1, keepdims=True)
    xn = x * (1.0 / np.sqrt(ss + eps))
    return rnd(xn * (1.0 + w))


def rmsnorm_gated(o, z, w, rnd, eps):
    """HF Qwen3_5MoeRMSNormGated (L183-192), per-head over the last
    dim. o: bf16-widened values [.., D]; z: bf16-widened gate [.., D];
    w: DIRECT weight (init ones, NO +1). Rounding points replicate the
    HF dtype flow: fp32 variance/normalize -> cast bf16 -> xw (bf16) ->
    xsilu(z fp32) (promotes fp32) -> cast bf16."""
    ss = (o ** 2).mean(axis=-1, keepdims=True)
    xn = o * (1.0 / np.sqrt(ss + eps))
    y = rnd(xn)                    # hidden_states.to(input_dtype), L189
    y = rnd(w * y)                 # weight * x (bf16 x bf16)
    y = rnd(y * silu(z))           # x fp32 silu(gate), cast bf16 L192
    return y


def linear(x, w, rnd):
    """bf16 GEMM: fp32(f64 here) accumulate, rnd out. w: [O, K]."""
    return rnd(_mm(x, w.T))


def causal_conv_silu(x, w, state, rnd):
    """HF causal conv (L207-247): ONE fused depthwise causal conv1d
    k=K over the fused [q|k|v] channels, no bias, then SiLU. x: [T, C]
    bf16-widened raw pre-conv values; w: [C, K]; state: [C, K] (the
    last K raw pre-conv inputs per channel, zero-padded at t=0 — the
    prefill path's left padding K-1 and the decode path's
    causal_conv1d_update are the same window). Conv accumulates fp32 ->
    bf16; SiLU on the bf16 value -> bf16 (torch opmath, single round
    each). Returns (out [T, C], new_state)."""
    T, C = x.shape
    K = w.shape[-1]
    out = np.zeros((T, C))
    stt = state.copy()
    for t in range(T):
        win = np.concatenate([stt[:, 1:], x[t, :, None]], axis=1)
        acc = np.zeros(C)
        for i in range(K):
            acc += w[:, i] * win[:, i]
        co = rnd(acc)
        out[t] = rnd(co * sigmoid(co))
        stt = win
    return out, stt


def gdn_decay(a, A_log, dt_bias, st):
    """HF L574: g = -exp(A_log.float()) * softplus(a.float() + dt_bias),
    all fp32, NO lower bound, per-V-head scalar. a: [T, Hv] bf16-widened;
    A_log/dt_bias: fp32 params [Hv]."""
    return st(-np.exp(A_log) * softplus(a + dt_bias[None, :]))


def l2norm(x, eps=1e-6):
    """HF l2norm (L250-255, eps hardcoded 1e-6 at L421-423): fp32
    x * rsqrt(sum(x^2) + eps) over the last dim. Caller applies st()."""
    ss = (x ** 2).sum(axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(ss + eps))


def gdn_recurrence(q, k, v, g, beta, S, st):
    """THE contract: torch_recurrent_gated_delta_rule (modeling
    L395-455), fp32 throughout, per token per v-head. q,k: [T, Hv, Dk]
    (post repeat_interleave, l2norm, q pre-scaled); v: [T, Hv, Dv];
    g,beta: [T, Hv]; S: [Hv, Dk, Dv]. Per step (L438-449):
      S <- S * exp(g_t); kv = sum_k k*S; S <- S + k x (beta*(v - kv));
      o_t = sum_k q*S  (AFTER the update — the current token included).
    Each materialized fp32 tensor gets st() in f32 mode. Returns
    (o [T, Hv, Dv], S_out)."""
    T, Hv, Dk = q.shape
    Dv = v.shape[-1]
    S = S.copy()
    o = np.zeros((T, Hv, Dv))
    for t in range(T):
        for h in range(Hv):
            Sh = S[h] * st(np.exp(g[t, h]))
            Sh = st(Sh)
            kv = st(_mm(k[t, h][None, :], Sh)[0])          # [Dv]
            delta = st(st(v[t, h] - kv) * beta[t, h])
            Sh = st(Sh + st(k[t, h][:, None] * delta[None, :]))
            S[h] = Sh
            o[t, h] = st(_mm(q[t, h][None, :], Sh)[0])
    return o, S


def rotary_cos_sin(pos, rot, theta, rnd, f32mode):
    """Partial-RoPE cos/sin at ONE position (modeling L91-171, default
    rope): inv_freq fp32 over rot/2 freqs, freqs = pos*inv_freq fp32,
    emb = cat(freqs, freqs) (NeoX layout: dims i and i+rot/2 share a
    frequency), attention_scaling 1.0, cast bf16 for application
    (L154). mrope = identity for text-only (position ids = arange; the
    three mrope rows are identical so apply_interleaved_mrope L156-171
    is a no-op)."""
    i = np.arange(rot // 2)
    if f32mode:
        invf = (1.0 / (theta ** (2.0 * i / rot))).astype(np.float32)
        ang = (np.float32(pos) * invf).astype(np.float64)
    else:
        invf = 1.0 / (theta ** (2.0 * i / rot))
        ang = float(pos) * invf
    c = rnd(np.cos(np.concatenate([ang, ang])))
    s = rnd(np.sin(np.concatenate([ang, ang])))
    return c, s


def apply_rope(xh, pos0, rot, theta, rnd, f32mode):
    """HF apply_rotary_pos_emb (L629-664): NeoX rotate_half pairing
    WITHIN the first `rot` dims, remaining dims pass through unchanged.
    xh: [T, H, D] bf16-widened. bf16 elementwise rounding per torch
    opmath semantics: rnd(x*c), rnd(rot(x)*s), rnd(sum)."""
    T = xh.shape[0]
    out = xh.copy()
    for t in range(T):
        c, s = rotary_cos_sin(pos0 + t, rot, theta, rnd, f32mode)
        xr = xh[t, :, :rot]
        x1 = xr[:, :rot // 2]
        x2 = xr[:, rot // 2:]
        rotx = np.concatenate([-x2, x1], axis=-1)
        out[t, :, :rot] = rnd(rnd(xr * c[None, :]) +
                              rnd(rotx * s[None, :]))
    return out


def attention_core(q, kall, vall, nrep, rnd, scale):
    """HF eager_attention_forward (L679-701) with causal mask. q:
    [Tq, H, D] (the LAST Tq positions), kall/vall: [Tk, Hkv, D] full
    cache. QK^T fp32-accumulated -> bf16, xscale (bf16) -> causal; mask
    = exclusion (HF adds finfo(bf16).min, which exp()s to exactly 0).
    softmax in FP32 (L696) -> bf16; P.V fp32-accumulated -> bf16.
    KV heads repeat_kv'd x nrep (L667-676: q head h uses kv head
    h // nrep)."""
    Tq, H, D = q.shape
    Tk = kall.shape[0]
    o = np.zeros((Tq, H, D))
    for t in range(Tq):
        p = (Tk - Tq) + t
        for h in range(H):
            hk = h // nrep
            A = np.zeros(p + 1)
            for j in range(p + 1):
                acc = 0.0
                for d in range(D):
                    acc += q[t, h, d] * kall[j, hk, d]
                A[j] = rnd(rnd(acc) * scale)
            m = A.max()
            e = np.exp(A - m)
            P = rnd(e / e.sum())
            for d in range(D):
                acc = 0.0
                for j in range(p + 1):
                    acc += P[j] * vall[j, hk, d]
                o[t, h, d] = rnd(acc)
    return o


def router_topk(x, wg, st, rnd, cfg):
    """HF Qwen3_5MoeTopKRouter (L837-853): bf16 logits -> FP32 softmax
    -> top-k -> renormalize to sum 1 (fp32) -> bf16. NO bias/sigmoid/
    scaling/group-limit. Deterministic tie-break: lowest index wins
    (stable argsort, the base oracle's convention)."""
    logits = rnd(_mm(x, wg.T))
    z = logits - logits.max()
    e = np.exp(z)
    probs = st(e / e.sum())
    tk = cfg["num_experts_per_tok"]
    idx = np.argsort(-probs, kind="stable")[:tk]
    w = st(probs[idx] / probs[idx].sum())
    return idx.astype(np.int64), rnd(w), logits, probs


def moe_forward(x, W, st, rnd, cfg):
    """HF Qwen3_5MoeSparseMoeBlock (L856-875) + experts (L797-834) for
    ONE token. x: [H] post-norm bf16-widened values. Fused experts:
    gate_up_proj [E, 2I, H] (chunk -> gate | up, L828), down_proj
    [E, H, I]; silu(gate)*up -> down, x routing weight. Weighted expert
    sum accumulated fp32, single bf16 round (documented reorder class —
    HF index_add_ into bf16). Shared expert: silu MLP gated by
    sigmoid(shared_expert_gate(x)) (L862, L871). Total = routed +
    shared."""
    idx, w, logits, probs = router_topk(x, W["rtr_w"], st, rnd, cfg)
    I = cfg["moe_intermediate_size"]
    acc = np.zeros(cfg["hidden_size"])
    for i, e in enumerate(idx):
        gu = linear(x, W["exp_gu"][e], rnd)
        act = rnd(silu(gu[:I]))
        act = rnd(act * gu[I:])
        y = linear(act, W["exp_d"][e], rnd)
        acc += rnd(w[i] * y)
    routed = rnd(acc)
    sg = linear(x, W["sh_g"], rnd)
    su = linear(x, W["sh_u"], rnd)
    sa = rnd(silu(sg) * su)
    shared = linear(sa, W["sh_d"], rnd)
    sgate = rnd(sigmoid(linear(x, W["sh_gate"], rnd)))
    shared = rnd(shared * sgate)
    out = rnd(routed + shared)
    return dict(idx=idx, w=w, routed=routed, shared=shared, out=out,
                logits=logits, probs=probs)


def moe_forward_per_token(x, W, st, rnd, cfg):
    """Per-token MoE (matches the C wiring): route, run selected
    experts, fp32 weighted combine, gated shared expert, bf16 add."""
    T = x.shape[0]
    H = cfg["hidden_size"]
    TK = cfg["num_experts_per_tok"]
    idxs = np.zeros((T, TK), dtype=np.int64)
    ws = np.zeros((T, TK))
    routed = np.zeros((T, H))
    shared = np.zeros((T, H))
    out = np.zeros((T, H))
    for t in range(T):
        r = moe_forward(x[t], W, st, rnd, cfg)
        idxs[t] = r["idx"]
        ws[t] = r["w"]
        routed[t] = r["routed"]
        shared[t] = r["shared"]
        out[t] = r["out"]
    return dict(rtr_idx=idxs, rtr_w=ws, moe_routed=routed,
                moe_shared=shared, moe_out=out)


# --------------------------------------------------------------------------
# mixers + layer forward (decoder wiring: modeling L911-954, pre-norm
# plain residuals: h = x + mixer(RMSNorm(x)); h = h + moe(RMSNorm(h)))

def gdn_mixer(x, w, state, rnd, st, cfg):
    """HF Qwen3_5MoeGatedDeltaNet (L458-618) on post-ln1 values
    x [T, H]. state: dict with conv [conv_dim, K] and S [Hv, Dk, Dv].
    Returns (attn_out, stages)."""
    gd = gdn_dims(cfg)
    T = x.shape[0]
    qkv = linear(x, w["w_qkv"], rnd)              # [T, conv_dim]
    z = linear(x, w["w_z"], rnd)                  # [T, value_dim]
    b = linear(x, w["w_b"], rnd)                  # [T, Hv]
    a = linear(x, w["w_a"], rnd)                  # [T, Hv]
    qc, state["conv"] = causal_conv_silu(qkv, w["conv_w"], state["conv"],
                                         rnd)
    q = qc[:, :gd["key_dim"]].reshape(T, gd["hk"], gd["dk"])
    k = qc[:, gd["key_dim"]:2 * gd["key_dim"]].reshape(T, gd["hk"],
                                                       gd["dk"])
    v = qc[:, 2 * gd["key_dim"]:].reshape(T, gd["hv"], gd["dv"])
    beta = rnd(sigmoid(b))                        # bf16 sigmoid, L572
    g = gdn_decay(a, w["A_log"], w["dt_bias"], st)
    q = np.repeat(q, gd["rep"], axis=1)           # repeat_interleave
    k = np.repeat(k, gd["rep"], axis=1)           # L575-577
    qn = st(l2norm(q))                            # fp32, L421-423
    kn = st(l2norm(k))
    qn = st(qn * float(gd["dk"]) ** -0.5)         # L426
    o, S = gdn_recurrence(qn, kn, v, g, beta, state["S"], st)
    state["S"] = S
    ob = rnd(o)                                   # .to(bf16), L454
    on = rmsnorm_gated(ob, z.reshape(T, gd["hv"], gd["dv"]),
                       w["onorm_w"], rnd, cfg["rms_norm_eps"])
    on = on.reshape(T, gd["value_dim"])
    attn_out = linear(on, w["w_out"], rnd)
    stages = dict(qkv_conv=qc, beta=beta, gdecay=g, rec_o=ob, onorm=on)
    return attn_out, stages


def attn_mixer(x, w, state, rnd, st, cfg, f32mode):
    """HF Qwen3_5MoeAttention (L704-778) on post-ln1 x [T, H]. state:
    kcache/vcache lists + pos. q_proj packs per-head [q|gate] (L744-747);
    per-head (1+w) q/k RMSNorm BEFORE RoPE; partial NeoX RoPE on the
    first rot dims; eager fp32 softmax; sigmoid output gate (L775);
    o_proj. Returns (attn_out, stages)."""
    ad = attn_dims(cfg)
    nh, nkv, d = ad["nh"], ad["nkv"], ad["d"]
    T = x.shape[0]
    qg = linear(x, w["wq"], rnd).reshape(T, nh, 2 * d)
    q, gate = qg[..., :d], qg[..., d:]
    q = rmsnorm(q, w["qn_w"], rnd, cfg["rms_norm_eps"])
    k = linear(x, w["wk"], rnd).reshape(T, nkv, d)
    k = rmsnorm(k, w["kn_w"], rnd, cfg["rms_norm_eps"])
    v = linear(x, w["wv"], rnd).reshape(T, nkv, d)
    pos0 = state["pos"]
    q = apply_rope(q, pos0, ad["rot"], ad["theta"], rnd, f32mode)
    k = apply_rope(k, pos0, ad["rot"], ad["theta"], rnd, f32mode)
    state["kcache"].append(k)
    state["vcache"].append(v)
    state["pos"] = pos0 + T
    kall = np.concatenate(state["kcache"], axis=0)
    vall = np.concatenate(state["vcache"], axis=0)
    o = attention_core(q, kall, vall, ad["nrep"], rnd,
                       float(d) ** -0.5)
    attno = o.reshape(T, nh * d)
    mgate = rnd(sigmoid(gate.reshape(T, nh * d)))
    og = rnd(attno * mgate)
    attn_out = linear(og, w["wo"], rnd)
    stages = dict(qf=q.reshape(T, nh * d), kf=k.reshape(T, nkv * d),
                  attno=attno, mgate=mgate)
    return attn_out, stages


def layer_forward(kind, w, x, state, mode, cfg):
    """One decoder layer on x [T, H] bf16-widened values (modeling
    L911-954). mode: 'f32' or 'f64'. Returns (out, stages)."""
    rnd = rnd16 if mode == "f32" else (lambda v: v)
    st = f32 if mode == "f32" else (lambda v: v)
    eps = cfg["rms_norm_eps"]

    ln1 = rmsnorm(x, w["ln1_w"], rnd, eps)
    if kind == "gdn":
        attn_out, astages = gdn_mixer(ln1, w, state, rnd, st, cfg)
    else:
        attn_out, astages = attn_mixer(ln1, w, state, rnd, st, cfg,
                                       mode == "f32")
    res1 = rnd(x + attn_out)
    ln2 = rmsnorm(res1, w["ln2_w"], rnd, eps)
    mstages = moe_forward_per_token(ln2, w, st, rnd, cfg)
    out = rnd(res1 + mstages["moe_out"])
    stages = dict(ln1=ln1, attn_out=attn_out, res1=res1, ln2=ln2,
                  out=out)
    stages.update(astages)
    stages.update(mstages)
    return out, stages


def new_state(kind, cfg):
    if kind == "gdn":
        gd = gdn_dims(cfg)
        return dict(conv=np.zeros((gd["conv_dim"], gd["K"])),
                    S=np.zeros((gd["hv"], gd["dk"], gd["dv"])))
    return dict(kcache=[], vcache=[], pos=0)


def state_arrays(kind, state):
    """Flattened state for fixture dumps (conv/S fp32-storable, KV
    caches bf16-storable)."""
    if "S" in state:
        return dict(conv_state=state["conv"], recurrent_state=state["S"])
    return dict(kcache=np.concatenate(state["kcache"], axis=0),
                vcache=np.concatenate(state["vcache"], axis=0),
                pos=np.array([state["pos"]], dtype=np.int64))


# --------------------------------------------------------------------------
# weight generation (fixed seed) + fixture IO
#
# Norm storage convention (hard gate: the oracle loads RAW values and
# applies the +1 itself): every RMSNorm EXCEPT the GDN gated output norm
# is the (1+w) zero-init variant — the checkpoint stores gain-1 (fixture:
# N(0, 0.1)). The GDN output norm weight is direct (fixture: 1+N(0,0.1)).
# A_log/dt_bias are fp32 params (A_log = log(U(0.01,16)) per L491/982,
# dt_bias near the ones init L488/978).

def _rb(rng, shape, sc):
    """bf16 codes of N(0, sc)."""
    return bf16_narrow_oracle_f32(
        (rng.standard_normal(shape) * sc).astype(np.float32))


def _rbn_offset(rng, shape):
    """(1+w) RMSNorm stored values: bf16 codes of N(0, 0.1)."""
    return bf16_narrow_oracle_f32(
        (rng.standard_normal(shape) * 0.1).astype(np.float32))


def _rbn_direct(rng, shape):
    """Direct norm weights (GDN gated output norm): 1.0 + N(0, 0.1)."""
    return bf16_narrow_oracle_f32(
        (1.0 + rng.standard_normal(shape) * 0.1).astype(np.float32))


def gen_layer_weights(rng, cfg, kind):
    """One layer's weight dict (base-oracle naming), bf16 codes + fp32
    params. MoE on EVERY layer (config: no dense layers)."""
    H = cfg["hidden_size"]
    E, I, Is = (cfg["num_experts"], cfg["moe_intermediate_size"],
                cfg["shared_expert_intermediate_size"])
    w = {}
    w["ln1_w"] = _rbn_offset(rng, H)
    w["ln2_w"] = _rbn_offset(rng, H)
    if kind == "gdn":
        gd = gdn_dims(cfg)
        w["w_qkv"] = _rb(rng, (gd["conv_dim"], H), 0.05)
        w["w_z"] = _rb(rng, (gd["value_dim"], H), 0.05)
        w["w_b"] = _rb(rng, (gd["hv"], H), 0.05)
        w["w_a"] = _rb(rng, (gd["hv"], H), 0.05)
        w["conv_w"] = _rb(rng, (gd["conv_dim"], gd["K"]), 0.3)
        w["A_log"] = np.log(rng.uniform(0.01, 16.0, gd["hv"])
                            ).astype(np.float32)
        w["dt_bias"] = (1.0 + rng.standard_normal(gd["hv"]) * 0.5
                        ).astype(np.float32)
        w["onorm_w"] = _rbn_direct(rng, gd["dv"])
        w["w_out"] = _rb(rng, (H, gd["value_dim"]), 0.05)
    else:
        ad = attn_dims(cfg)
        w["wq"] = _rb(rng, (ad["nh"] * 2 * ad["d"], H), 0.05)
        w["wk"] = _rb(rng, (ad["nkv"] * ad["d"], H), 0.05)
        w["wv"] = _rb(rng, (ad["nkv"] * ad["d"], H), 0.05)
        w["wo"] = _rb(rng, (H, ad["nh"] * ad["d"]), 0.05)
        w["qn_w"] = _rbn_offset(rng, ad["d"])
        w["kn_w"] = _rbn_offset(rng, ad["d"])
    w["rtr_w"] = _rb(rng, (E, H), 0.2)
    w["exp_gu"] = _rb(rng, (E, 2 * I, H), 0.05)
    w["exp_d"] = _rb(rng, (E, H, I), 0.05)
    w["sh_g"] = _rb(rng, (Is, H), 0.05)
    w["sh_u"] = _rb(rng, (Is, H), 0.05)
    w["sh_d"] = _rb(rng, (H, Is), 0.05)
    w["sh_gate"] = _rb(rng, (1, H), 0.05)
    return w


def gen_weights(rng, cfg):
    """bf16 codes (u16) + fp32 params, per kind."""
    return {kind: gen_layer_weights(rng, cfg, kind) for kind in KINDS}


def widen_w(w):
    """bf16 code arrays -> f64 values (fp32 params stay as-is)."""
    out = {}
    for k, v in w.items():
        if v.dtype == np.uint16:
            out[k] = bf16_widen_f64(v)
        else:
            out[k] = v.astype(np.float64)
    return out


# ==========================================================================
# full synthetic mini-model (stacked layers typed by cfg["layer_types"],
# REAL Qwen3.6 tensor naming — model.language_model.*, lm_head.*; the
# model.visual.* tower and mtp.* are excluded, same precedent as the
# GLM adapter's vision strip)

#: real checkpoint tensor names (reference/model.safetensors.index.json)
EMB_NAME = "model.language_model.embed_tokens.weight"
NORM_NAME = "model.language_model.norm.weight"
LMHEAD_NAME = "lm_head.weight"


def gen_full_weights(rng, cfg):
    """Real Qwen3.6 tensor names -> arrays (bf16 codes / fp32). Returns
    the flat named map AND the per-layer dicts (same arrays)."""
    H = cfg["hidden_size"]
    named = {}
    layers = []
    for L in range(cfg["num_hidden_layers"]):
        kind = layer_kind(cfg, L)
        w = gen_layer_weights(rng, cfg, kind)
        layers.append(w)
        p = f"model.language_model.layers.{L}."
        named[p + "input_layernorm.weight"] = w["ln1_w"]
        named[p + "post_attention_layernorm.weight"] = w["ln2_w"]
        if kind == "gdn":
            gd = gdn_dims(cfg)
            a = p + "linear_attn."
            named[a + "in_proj_qkv.weight"] = w["w_qkv"]
            named[a + "in_proj_z.weight"] = w["w_z"]
            named[a + "in_proj_b.weight"] = w["w_b"]
            named[a + "in_proj_a.weight"] = w["w_a"]
            named[a + "conv1d.weight"] = w["conv_w"].reshape(
                gd["conv_dim"], 1, gd["K"])
            named[a + "A_log"] = w["A_log"]
            named[a + "dt_bias"] = w["dt_bias"]
            named[a + "norm.weight"] = w["onorm_w"]
            named[a + "out_proj.weight"] = w["w_out"]
        else:
            a = p + "self_attn."
            named[a + "q_proj.weight"] = w["wq"]
            named[a + "k_proj.weight"] = w["wk"]
            named[a + "v_proj.weight"] = w["wv"]
            named[a + "o_proj.weight"] = w["wo"]
            named[a + "q_norm.weight"] = w["qn_w"]
            named[a + "k_norm.weight"] = w["kn_w"]
        named[p + "mlp.gate.weight"] = w["rtr_w"]
        named[p + "mlp.experts.gate_up_proj"] = w["exp_gu"]
        named[p + "mlp.experts.down_proj"] = w["exp_d"]
        named[p + "mlp.shared_expert.gate_proj.weight"] = w["sh_g"]
        named[p + "mlp.shared_expert.up_proj.weight"] = w["sh_u"]
        named[p + "mlp.shared_expert.down_proj.weight"] = w["sh_d"]
        named[p + "mlp.shared_expert_gate.weight"] = w["sh_gate"]
    named[EMB_NAME] = _rb(rng, (cfg["vocab_size"], H), 0.08)
    named[NORM_NAME] = _rbn_offset(rng, H)
    named[LMHEAD_NAME] = _rb(rng, (cfg["vocab_size"], H), 0.08)
    return named, layers


def full_layer_weights(named, L, kind, cfg):
    """Rebuild the per-layer dict from the real-name map (the golden
    path consumes weights read back from the container, never the
    generation arrays — the Apus M4c lesson)."""
    p = f"model.language_model.layers.{L}."
    w = {"ln1_w": named[p + "input_layernorm.weight"],
         "ln2_w": named[p + "post_attention_layernorm.weight"]}
    if kind == "gdn":
        gd = gdn_dims(cfg)
        a = p + "linear_attn."
        w.update({
            "w_qkv": named[a + "in_proj_qkv.weight"],
            "w_z": named[a + "in_proj_z.weight"],
            "w_b": named[a + "in_proj_b.weight"],
            "w_a": named[a + "in_proj_a.weight"],
            "conv_w": named[a + "conv1d.weight"].reshape(
                gd["conv_dim"], gd["K"]),
            "A_log": named[a + "A_log"],
            "dt_bias": named[a + "dt_bias"],
            "onorm_w": named[a + "norm.weight"],
            "w_out": named[a + "out_proj.weight"],
        })
    else:
        a = p + "self_attn."
        w.update({
            "wq": named[a + "q_proj.weight"],
            "wk": named[a + "k_proj.weight"],
            "wv": named[a + "v_proj.weight"],
            "wo": named[a + "o_proj.weight"],
            "qn_w": named[a + "q_norm.weight"],
            "kn_w": named[a + "k_norm.weight"],
        })
    w.update({
        "rtr_w": named[p + "mlp.gate.weight"],
        "exp_gu": named[p + "mlp.experts.gate_up_proj"],
        "exp_d": named[p + "mlp.experts.down_proj"],
        "sh_g": named[p + "mlp.shared_expert.gate_proj.weight"],
        "sh_u": named[p + "mlp.shared_expert.up_proj.weight"],
        "sh_d": named[p + "mlp.shared_expert.down_proj.weight"],
        "sh_gate": named[p + "mlp.shared_expert_gate.weight"],
    })
    return w


def full_forward(named, ids, mode, cfg, states=None):
    """Full mini-model: emb[id] (NO scaling) -> N layers (typed by
    cfg["layer_types"]) -> final RMSNorm -> lm_head (bf16 logits, NO
    fp32 upcast, L1900-1903), no softcap. named: real-name map of f64
    VALUES (bf16 widened). ids: int list. states: optional list of
    per-layer state dicts (decode chaining). Returns
    (logits [T, vocab], hidden traces [n_layers][T, H])."""
    rnd = rnd16 if mode == "f32" else (lambda v: v)
    x = named[EMB_NAME][np.asarray(ids)]
    traces = []
    for L in range(cfg["num_hidden_layers"]):
        kind = layer_kind(cfg, L)
        w = full_layer_weights(named, L, kind, cfg)
        st = states[L] if states is not None else new_state(kind, cfg)
        x, _ = layer_forward(kind, w, x, st, mode, cfg)
        if states is not None:
            states[L] = st
        traces.append(x)
    y = rmsnorm(x, named[NORM_NAME], rnd, cfg["rms_norm_eps"])
    logits = linear(y, named[LMHEAD_NAME], rnd)
    return logits, traces


# ==========================================================================
# MTP block (M8, contract §7). HF does NOT implement it; normative: the
# real checkpoint naming + vLLM qwen3_next_mtp.py (pre_fc norms applied
# directly to the incoming tensors — SINGLE norm each, no main-model
# final norm inside the block). Tensor names:
#   mtp.fc.weight [H, 2H]                (concat[e_normed, h_normed] proj)
#   mtp.pre_fc_norm_embedding.weight     ((1+w) RMSNorm)
#   mtp.pre_fc_norm_hidden.weight        ((1+w) RMSNorm)
#   mtp.layers.0.*                       (full_attention + full MoE,
#                                         same names as model.*.layers)
#   mtp.norm.weight                      ((1+w) RMSNorm)
# Embeddings/lm_head are SHARED with the main model
# (config mtp_use_dedicated_embeddings=false). The block does not affect
# the main-path logits (Phase A) — draft quality only.


def gen_mtp_weights(rng, cfg):
    """The mtp.* block weights (bf16 codes), real checkpoint names.
    The decoder layer is generated by the SAME gen_layer_weights("full")
    as the main layers."""
    H = cfg["hidden_size"]
    named = {
        "mtp.pre_fc_norm_embedding.weight": _rbn_offset(rng, H),
        "mtp.pre_fc_norm_hidden.weight": _rbn_offset(rng, H),
        "mtp.fc.weight": _rb(rng, (H, 2 * H), 0.05),
        "mtp.norm.weight": _rbn_offset(rng, H),
    }
    w = gen_layer_weights(rng, cfg, "full")
    p = "mtp.layers.0."
    named[p + "input_layernorm.weight"] = w["ln1_w"]
    named[p + "post_attention_layernorm.weight"] = w["ln2_w"]
    a = p + "self_attn."
    named[a + "q_proj.weight"] = w["wq"]
    named[a + "k_proj.weight"] = w["wk"]
    named[a + "v_proj.weight"] = w["wv"]
    named[a + "o_proj.weight"] = w["wo"]
    named[a + "q_norm.weight"] = w["qn_w"]
    named[a + "k_norm.weight"] = w["kn_w"]
    named[p + "mlp.gate.weight"] = w["rtr_w"]
    named[p + "mlp.experts.gate_up_proj"] = w["exp_gu"]
    named[p + "mlp.experts.down_proj"] = w["exp_d"]
    named[p + "mlp.shared_expert.gate_proj.weight"] = w["sh_g"]
    named[p + "mlp.shared_expert.up_proj.weight"] = w["sh_u"]
    named[p + "mlp.shared_expert.down_proj.weight"] = w["sh_d"]
    named[p + "mlp.shared_expert_gate.weight"] = w["sh_gate"]
    return named


def mtp_layer_weights(named, cfg):
    """Rebuild the MTP layer's per-layer dict from the real-name map
    (the golden path consumes weights read back from the container)."""
    p = "mtp.layers.0."
    a = p + "self_attn."
    return {
        "ln1_w": named[p + "input_layernorm.weight"],
        "ln2_w": named[p + "post_attention_layernorm.weight"],
        "wq": named[a + "q_proj.weight"],
        "wk": named[a + "k_proj.weight"],
        "wv": named[a + "v_proj.weight"],
        "wo": named[a + "o_proj.weight"],
        "qn_w": named[a + "q_norm.weight"],
        "kn_w": named[a + "k_norm.weight"],
        "rtr_w": named[p + "mlp.gate.weight"],
        "exp_gu": named[p + "mlp.experts.gate_up_proj"],
        "exp_d": named[p + "mlp.experts.down_proj"],
        "sh_g": named[p + "mlp.shared_expert.gate_proj.weight"],
        "sh_u": named[p + "mlp.shared_expert.up_proj.weight"],
        "sh_d": named[p + "mlp.shared_expert.down_proj.weight"],
        "sh_gate": named[p + "mlp.shared_expert_gate.weight"],
    }


def mtp_forward(named, h, ids, mode, cfg, state=None):
    """Contract §7: e = rmsnorm(emb[ids], pre_fc_norm_embedding);
    hh = rmsnorm(h, pre_fc_norm_hidden) — h is the PRE-final-norm main
    hidden (or the block's own normed output when chaining; the same
    single-norm path either way); z = fc(cat[e, hh]); one full layer;
    mtp.norm; SHARED lm_head. Returns (logits [T, V], y [T, H]) where y
    is the mtp.norm output (the draft-chain input). state: optional
    full-kind state dict (decode chaining)."""
    rnd = rnd16 if mode == "f32" else (lambda v: v)
    eps = cfg["rms_norm_eps"]
    e = rmsnorm(named[EMB_NAME][np.asarray(ids)],
                named["mtp.pre_fc_norm_embedding.weight"], rnd, eps)
    hh = rmsnorm(np.asarray(h, dtype=np.float64),
                 named["mtp.pre_fc_norm_hidden.weight"], rnd, eps)
    z = linear(np.concatenate([e, hh], axis=-1), named["mtp.fc.weight"],
               rnd)
    st = state if state is not None else new_state("full", cfg)
    out, _ = layer_forward("full", mtp_layer_weights(named, cfg), z, st,
                           mode, cfg)
    y = rmsnorm(out, named["mtp.norm.weight"], rnd, eps)
    logits = linear(y, named[LMHEAD_NAME], rnd)
    return logits, y


def mtp_chain(named, h0, emb0, steps, mode, cfg):
    """Draft chain from a seed pair (h0 = pre-final-norm main hidden
    [H], emb0): each step's h is the previous step's mtp.norm output
    (the depth-1 approximation — draft quality only). Returns
    [(argmax_token, logits_row), ...]."""
    state = new_state("full", cfg)
    h = np.asarray(h0, dtype=np.float64)[None, :]
    eid = emb0
    out = []
    for _ in range(steps):
        logits, y = mtp_forward(named, h, [eid], mode, cfg, state=state)
        tok = int(np.argmax(logits[0]))
        out.append((tok, logits[0]))
        h = y
        eid = tok
    return out
