#!/usr/bin/env python3
"""tests/m4b/check_oracle.py — M4b oracle self-consistency checks
(Qwen3.6-35B-A3B mini config).

Runs entirely from the fixtures ON DISK (weights read back, never the
generation arrays — the Apus M4c lesson: independent realization paths).
Asserts (exit 1 on any failure):

  1. REPLAY: re-running the layer from disk-loaded weights reproduces the
     stored f32/f64 goldens exactly (IO integrity + determinism).
  2. SELECTION: router f32 and f64 selections are identical on every
     token (fixture margins large enough that C fp32 noise cannot flip;
     if this fails the fixture seed must change).
  3. CHUNK INVARIANCE (oracle-side): one-shot T=12 forward vs
     prefill_len 7 + 5 decode steps — f64 EXACT (<=1e-12); f32 measured
     and reported (the oracle realization is identical between paths, so
     zero flips expected).
  4. ENVELOPE: prints the per-stage |f32 - f64| table (the scale the C
     forward must land inside of) and writes envelope.txt.

Run from the repository root: .venv/bin/python tests/m4b/check_oracle.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle  # noqa: E402
from gen_fixtures import STAGES  # noqa: E402

FIX = os.path.join(HERE, "fixtures")
CFG = oracle.MINI_CFG

failures = 0


def check(cond, msg):
    global failures
    if not cond:
        failures += 1
        print(f"FAIL: {msg}")


def load(name, dtype=np.float64):
    return np.fromfile(os.path.join(FIX, name), dtype=dtype)


def load_weights(kind):
    cfg = CFG
    H = cfg["hidden_size"]
    E, I, Is = (cfg["num_experts"], cfg["moe_intermediate_size"],
                cfg["shared_expert_intermediate_size"])

    def u16(name):
        return load(f"w_{kind}_{name}.bin", np.uint16)

    def f32(name):
        return load(f"w_{kind}_{name}.bin", np.float32)

    w = {}
    w["ln1_w"] = u16("ln1_w").reshape(H)
    w["ln2_w"] = u16("ln2_w").reshape(H)
    if kind == "gdn":
        gd = oracle.gdn_dims(cfg)
        w["w_qkv"] = u16("w_qkv").reshape(gd["conv_dim"], H)
        w["w_z"] = u16("w_z").reshape(gd["value_dim"], H)
        w["w_b"] = u16("w_b").reshape(gd["hv"], H)
        w["w_a"] = u16("w_a").reshape(gd["hv"], H)
        w["conv_w"] = u16("conv_w").reshape(gd["conv_dim"], gd["K"])
        w["A_log"] = f32("A_log")
        w["dt_bias"] = f32("dt_bias")
        w["onorm_w"] = u16("onorm_w").reshape(gd["dv"])
        w["w_out"] = u16("w_out").reshape(H, gd["value_dim"])
    else:
        ad = oracle.attn_dims(cfg)
        w["wq"] = u16("wq").reshape(ad["nh"] * 2 * ad["d"], H)
        w["wk"] = u16("wk").reshape(ad["nkv"] * ad["d"], H)
        w["wv"] = u16("wv").reshape(ad["nkv"] * ad["d"], H)
        w["wo"] = u16("wo").reshape(H, ad["nh"] * ad["d"])
        w["qn_w"] = u16("qn_w").reshape(ad["d"])
        w["kn_w"] = u16("kn_w").reshape(ad["d"])
    w["rtr_w"] = u16("rtr_w").reshape(E, H)
    w["exp_gu"] = u16("exp_gu").reshape(E, 2 * I, H)
    w["exp_d"] = u16("exp_d").reshape(E, H, I)
    w["sh_g"] = u16("sh_g").reshape(Is, H)
    w["sh_u"] = u16("sh_u").reshape(Is, H)
    w["sh_d"] = u16("sh_d").reshape(H, Is)
    w["sh_gate"] = u16("sh_gate").reshape(1, H)
    return w


def seqs():
    yield "pre", CFG["prefill_len"]
    for i in range(CFG["decode_len"]):
        yield f"dec{i}", 1


def main():
    # --- replay from disk + selection identity + envelope --------------
    env_rows = []
    for kind in oracle.KINDS:
        w = oracle.widen_w(load_weights(kind))
        xp = oracle.bf16_widen_f64(load(f"{kind}_in_prefill.bin", np.uint16)
                                   .reshape(CFG["prefill_len"], -1))
        xd = oracle.bf16_widen_f64(load(f"{kind}_in_decode.bin", np.uint16)
                                   .reshape(CFG["decode_len"], -1))
        xin = {"pre": xp}
        for i in range(CFG["decode_len"]):
            xin[f"dec{i}"] = xd[i:i + 1]

        outs = {}
        for mode in ("f32", "f64"):
            state = oracle.new_state(kind, CFG)
            for tag, _ in seqs():
                out, stages = oracle.layer_forward(kind, w, xin[tag],
                                                   state, mode, CFG)
                outs[(tag, mode)] = out
                for name in STAGES[kind]:
                    gv = load(f"{kind}_{tag}_{name}_{mode}.bin",
                              np.int64 if name == "rtr_idx" else np.float64)
                    rv = stages[name].reshape(-1)
                    if name == "rtr_idx":
                        check(np.array_equal(rv, gv),
                              f"{kind}/{tag}/{name}/{mode}: replay mismatch")
                    else:
                        d = float(np.max(np.abs(rv - gv))) if gv.size else 0.0
                        check(d == 0.0,
                              f"{kind}/{tag}/{name}/{mode}: replay diff {d}")
        # selection identity f32 vs f64
        for tag, T in seqs():
            i32 = load(f"{kind}_{tag}_rtr_idx_f32.bin", np.int64)
            i64 = load(f"{kind}_{tag}_rtr_idx_f64.bin", np.int64)
            check(np.array_equal(i32, i64),
                  f"{kind}/{tag}: router selection f32 != f64")
        # envelope per stage (prefill sequence)
        for name in STAGES[kind]:
            if name == "rtr_idx":
                continue
            a = load(f"{kind}_pre_{name}_f32.bin")
            b = load(f"{kind}_pre_{name}_f64.bin")
            m = float(np.max(np.abs(a - b))) if a.size else 0.0
            sc = float(np.max(np.abs(b))) if b.size else 1.0
            env_rows.append((kind, name, m, m / max(sc, 1e-30)))
        # output diff f32 vs f64 across the whole chain (report)
        dmax = max(float(np.max(np.abs(outs[(t, "f32")] - outs[(t, "f64")])))
                   for t, _ in seqs())
        env_rows.append((kind, "out_CHAIN_MAX", dmax, dmax))

    # --- chunk invariance (oracle-side) ----------------------------------
    print("chunk invariance (one-shot T=12 vs 7+5):")
    for kind in oracle.KINDS:
        w = oracle.widen_w(load_weights(kind))
        xp = oracle.bf16_widen_f64(load(f"{kind}_in_prefill.bin", np.uint16)
                                   .reshape(CFG["prefill_len"], -1))
        xd = oracle.bf16_widen_f64(load(f"{kind}_in_decode.bin", np.uint16)
                                   .reshape(CFG["decode_len"], -1))
        xall = np.concatenate([xp, xd], axis=0)
        for mode in ("f32", "f64"):
            s1 = oracle.new_state(kind, CFG)
            out1, _ = oracle.layer_forward(kind, w, xall, s1, mode, CFG)
            s2 = oracle.new_state(kind, CFG)
            out2p, _ = oracle.layer_forward(kind, w, xp, s2, mode, CFG)
            outs2 = [out2p]
            for i in range(CFG["decode_len"]):
                o, _ = oracle.layer_forward(kind, w, xd[i:i + 1], s2,
                                            mode, CFG)
                outs2.append(o)
            out2 = np.concatenate(outs2, axis=0)
            d = float(np.max(np.abs(out1 - out2)))
            if mode == "f64":
                check(d <= 1e-12, f"{kind}: f64 chunk invariance {d}")
                print(f"  {kind}/f64: max diff {d:.3g}")
            else:
                bf1 = oracle.bf16_narrow_oracle_f32(out1.astype(np.float32))
                bf2 = oracle.bf16_narrow_oracle_f32(out2.astype(np.float32))
                flips = int(np.count_nonzero(bf1 != bf2))
                check(flips == 0,
                      f"{kind}: f32 chunk invariance {flips} code flips")
                print(f"  {kind}/f32: max diff {d:.3g}, "
                      f"code flips {flips}/{bf1.size}")

    # --- envelope table ---------------------------------------------------
    print("\nf32-vs-f64 envelope (prefill, per stage):")
    print(f"  {'stage':24s} {'kind':6s} {'max abs':>10s} {'rel':>10s}")
    for kind, name, m, rel in env_rows:
        print(f"  {name:24s} {kind:6s} {m:10.3g} {rel:10.3g}")
    with open(os.path.join(FIX, "envelope.txt"), "w") as f:
        for kind, name, m, rel in env_rows:
            f.write(f"{kind} {name} {m:.9g} {rel:.9g}\n")

    if failures:
        print(f"\ncheck_oracle: {failures} FAILURES")
        sys.exit(1)
    print("\ncheck_oracle: all checks passed")


if __name__ == "__main__":
    main()
