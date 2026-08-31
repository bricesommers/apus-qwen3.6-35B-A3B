#!/usr/bin/env python3
"""tests/m4b/gen_fixtures.py — generate the M4b single-layer fixtures
(weights, inputs, per-stage f32/f64 goldens, decode chains, states) into
tests/m4b/fixtures/ for the Qwen3.6-35B-A3B mini config (oracle
MINI_CFG: hidden 128; GDN 4 K heads / 8 V heads x 32; full attn 4 q / 2
KV heads x 64, partial rotary 16; MoE 16 experts top-4; 2 layers
[linear_attention, full_attention]). Deterministic (fixed seed
20260829). numpy only.

Run from the repository root: .venv/bin/python tests/m4b/gen_fixtures.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle  # noqa: E402

OUT = os.path.join(HERE, "fixtures")

# stage lists captured per kind (order fixed; C test uses the same names)
STAGES = {
    "gdn": ["ln1", "qkv_conv", "beta", "gdecay", "rec_o", "onorm",
            "attn_out", "res1", "ln2", "rtr_idx", "rtr_w", "moe_routed",
            "moe_shared", "moe_out", "out"],
    "full": ["ln1", "qf", "kf", "attno", "mgate", "attn_out", "res1",
             "ln2", "rtr_idx", "rtr_w", "moe_routed", "moe_shared",
             "moe_out", "out"],
}


def dump(name, arr):
    np.asarray(arr).tofile(os.path.join(OUT, name))


def dump_state_codes(kind, state, cfg, suffix=""):
    """State after prefill, in storage dtypes (bf16 codes / f32 / i64).

    The C conv state holds the last K-1 PRE-conv inputs per channel;
    the oracle's [C, K] state is [dead_column, those K-1] (its first
    column is shifted out before ever being read), so it is sliced off
    here."""
    sa = oracle.state_arrays(kind, state)
    for k, v in sa.items():
        if k == "conv_state":
            dump(f"{kind}_state{suffix}_conv.bin",
                 oracle.bf16_narrow_oracle_f32(
                     v[:, 1:].astype(np.float32)))
        elif k == "recurrent_state":
            dump(f"{kind}_state{suffix}_S.bin", v.astype(np.float32))
        elif k in ("kcache", "vcache"):
            dump(f"{kind}_state{suffix}_{k}.bin",
                 oracle.bf16_narrow_oracle_f32(v.astype(np.float32)))
        elif k == "pos":
            dump(f"{kind}_state{suffix}_pos.bin", v.astype(np.int64))


def run_sequence(kind, w, x, state, mode, tag, cfg):
    """Run prefill or one decode step; dump all stage goldens."""
    out, stages = oracle.layer_forward(kind, w, x, state, mode, cfg)
    for name in STAGES[kind]:
        v = stages[name]
        if name == "rtr_idx":
            dump(f"{kind}_{tag}_{name}_{mode}.bin", v.astype(np.int64))
        else:
            dump(f"{kind}_{tag}_{name}_{mode}.bin",
                 np.asarray(v, dtype=np.float64))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = oracle.MINI_CFG
    rng = np.random.default_rng(20260829)

    W = oracle.gen_weights(rng, cfg)
    for kind in oracle.KINDS:
        for name, arr in W[kind].items():
            dump(f"w_{kind}_{name}.bin", arr)

    # inputs: prefill T=7, decode chain 5 tokens (positions 7..11);
    # own rng stream (seed documented) so weight edits don't shift them
    irng = np.random.default_rng(20260829)
    inputs = {}
    for kind in oracle.KINDS:
        xp = oracle.bf16_narrow_oracle_f32(
            irng.standard_normal((cfg["prefill_len"],
                                  cfg["hidden_size"])).astype(np.float32))
        xd = oracle.bf16_narrow_oracle_f32(
            irng.standard_normal((cfg["decode_len"],
                                  cfg["hidden_size"])).astype(np.float32))
        dump(f"{kind}_in_prefill.bin", xp)
        dump(f"{kind}_in_decode.bin", xd)
        inputs[kind] = (oracle.bf16_widen_f64(xp),
                        oracle.bf16_widen_f64(xd))

    for kind in oracle.KINDS:
        ww = oracle.widen_w(W[kind])
        xp, xd = inputs[kind]
        for mode in ("f32", "f64"):
            state = oracle.new_state(kind, cfg)
            run_sequence(kind, ww, xp, state, mode, "pre", cfg)
            if mode == "f32":
                dump_state_codes(kind, state, cfg)
            for i in range(cfg["decode_len"]):
                run_sequence(kind, ww, xd[i:i + 1], state, mode,
                             f"dec{i}", cfg)

    gd = oracle.gdn_dims(cfg)
    ad = oracle.attn_dims(cfg)
    with open(os.path.join(OUT, "manifest.txt"), "w") as f:
        f.write(f"hidden={cfg['hidden_size']}\n")
        f.write(f"gdn_hk={gd['hk']}\n")
        f.write(f"gdn_hv={gd['hv']}\n")
        f.write(f"gdn_dk={gd['dk']}\n")
        f.write(f"gdn_dv={gd['dv']}\n")
        f.write(f"gdn_conv_dim={gd['conv_dim']}\n")
        f.write(f"gdn_conv_k={gd['K']}\n")
        f.write(f"attn_nh={ad['nh']}\n")
        f.write(f"attn_nkv={ad['nkv']}\n")
        f.write(f"attn_d={ad['d']}\n")
        f.write(f"attn_rot={ad['rot']}\n")
        f.write(f"rope_theta={ad['theta']}\n")
        f.write(f"experts={cfg['num_experts']}\n")
        f.write(f"moe_inter={cfg['moe_intermediate_size']}\n")
        f.write(f"shared_inter={cfg['shared_expert_intermediate_size']}\n")
        f.write(f"top_k={cfg['num_experts_per_tok']}\n")
        f.write(f"prefill_len={cfg['prefill_len']}\n")
        f.write(f"decode_len={cfg['decode_len']}\n")
        f.write(f"kinds={','.join(oracle.KINDS)}\n")
        for kind in oracle.KINDS:
            f.write(f"stages_{kind}={','.join(STAGES[kind])}\n")
    print("fixtures written:", OUT)


if __name__ == "__main__":
    main()
