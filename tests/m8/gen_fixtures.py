#!/usr/bin/env python3
"""tests/m8/gen_fixtures.py — M8 fixtures: the M6a mini-model container
EXTENDED with the Qwen MTP block (contract §7), plus MTP oracle goldens.

  tests/m8/fixtures/model/      container: m6a contents (main shard,
                                apus.index.json v2) + a NEW mtp shard
                                group apus-qwen-mtp-00001.safetensors
                                holding every mtp.* tensor (dense first,
                                then the per-expert 2-member slab slices
                                mtp.layers.0.mlp.experts.{E}.{gate_up,
                                down}_proj.weight, contiguous per
                                expert), weight_map + apus.index.json
                                extended (shard_groups mtp; slab records
                                numbered layer = num_hidden_layers + 0
                                = 2, the convert.py convention).
                                config.json already carries
                                mtp_num_hidden_layers=1 (M5 fixture).
  mtp_pre_h.bin                 f32 [T,H] bf16 codes: pre-final-norm
                                main hiddens (random — the forward gate
                                does not need real main hiddens)
  mtp_pre_ids.bin               i64 [T]: emb ids
  mtp_pre_logits_{f32,f64}.bin  f64 [T,V]: oracle golden
  mtp_pre_gap.bin               f64 [T]: top1-top2 gap per row
  mtp_chain_*.bin               3-step draft chain goldens

Run from the repository root (make golden-m8; regenerates m5/m6a first
when missing). Deterministic (seeds documented inline).
"""

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests", "m1"))

import oracle  # noqa: E402
import stutil  # noqa: E402

M6A_MODEL = os.path.join(ROOT, "tests", "m6a", "fixtures", "model")
OUT = os.path.join(HERE, "fixtures")
MODEL = os.path.join(OUT, "model")
MAIN_SHARD = "apus-qwen-00001.safetensors"
MTP_SHARD = "apus-qwen-mtp-00001.safetensors"
CFG = oracle.MINI_CFG
MTP_SEED = 20260829

MAIN_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_up_proj|down_proj)\.weight$")


def build_container():
    """m6a container + the mtp shard group, convert.py layout rules."""
    if not os.path.exists(os.path.join(M6A_MODEL, "apus.index.json")):
        subprocess.check_call(
            [sys.executable,
             os.path.join(ROOT, "tests", "m6a", "gen_fixtures.py")],
            cwd=ROOT)
    os.makedirs(MODEL, exist_ok=True)
    for f in (MAIN_SHARD, "config.json", "apus.index.json"):
        shutil.copyfile(os.path.join(M6A_MODEL, f), os.path.join(MODEL, f))
    hdr, ds = stutil.read_shard(os.path.join(MODEL, MAIN_SHARD))
    raw = stutil.read_tensor_bytes(os.path.join(MODEL, MAIN_SHARD), hdr,
                                   ds)
    hdr6, ds6 = stutil.read_shard(os.path.join(M6A_MODEL, MAIN_SHARD))
    raw6 = stutil.read_tensor_bytes(os.path.join(M6A_MODEL, MAIN_SHARD),
                                    hdr6, ds6)
    for name in hdr6:
        assert raw[name] == raw6[name], f"{name}: payload changed"

    rng = np.random.default_rng(MTP_SEED)
    mtp = oracle.gen_mtp_weights(rng, CFG)
    E = CFG["num_experts"]

    # convert.py layout: non-expert tensors first (sorted), then the
    # per-expert 2-member slab slices by expert, gate_up then down
    # contiguous (the v2 slab invariants, asserted below)
    entries = []
    for name in sorted(n for n in mtp if ".experts." not in n):
        arr = mtp[name]
        dtype = "BF16" if arr.dtype == np.uint16 else "F32"
        entries.append((name, dtype, list(arr.shape), arr.tobytes()))
    gu = mtp["mtp.layers.0.mlp.experts.gate_up_proj"]     # [E, 2I, H]
    dn = mtp["mtp.layers.0.mlp.experts.down_proj"]        # [E, H, I]
    for e in range(E):
        entries.append((f"mtp.layers.0.mlp.experts.{e}.gate_up_proj."
                        f"weight", "BF16", list(gu[e].shape),
                        gu[e].tobytes()))
        entries.append((f"mtp.layers.0.mlp.experts.{e}.down_proj.weight",
                        "BF16", list(dn[e].shape), dn[e].tobytes()))
    stutil.write_shard(os.path.join(MODEL, MTP_SHARD), entries)
    mhdr, mds = stutil.read_shard(os.path.join(MODEL, MTP_SHARD))

    # weight_map: main tensors (m6a shard) + mtp tensors (mtp shard)
    weight_map = {name: MAIN_SHARD for name in hdr}
    weight_map.update({t[0]: MTP_SHARD for t in entries})
    with open(os.path.join(MODEL, "model.safetensors.index.json"),
              "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)

    # apus.index.json: extend the m6a v2 manifest with the mtp group
    with open(os.path.join(MODEL, "apus.index.json")) as f:
        manifest = json.load(f)
    assert manifest["format_version"] == 2
    for name, meta in mhdr.items():
        a, b = meta["data_offsets"]
        manifest["tensor_map"][name] = {
            "shard": MTP_SHARD, "offset": mds + a, "nbytes": b - a,
            "dtype": meta["dtype"], "shape": meta["shape"]}
    n_layers = CFG["num_hidden_layers"]
    slab_bytes = None
    for e in range(E):
        names = [f"mtp.layers.0.mlp.experts.{e}.gate_up_proj.weight",
                 f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"]
        offs = [(mds + mhdr[n]["data_offsets"][0],
                 mds + mhdr[n]["data_offsets"][1],
                 mhdr[n]["shape"]) for n in names]
        assert offs[0][1] == offs[1][0], \
            f"mtp expert {e}: gate_up/down not contiguous/in order"
        gu_sh, d_sh = offs[0][2], offs[1][2]
        assert gu_sh[0] == 2 * d_sh[1] and gu_sh[1] == d_sh[0]
        nb = offs[1][1] - offs[0][0]
        if slab_bytes is None:
            slab_bytes = nb
        assert nb == slab_bytes, f"mtp expert {e}: non-uniform slab"
        manifest["expert_slabs"].append({
            "layer": n_layers,          # num_hidden_layers + 0
            "expert": e, "shard": MTP_SHARD,
            "offset": offs[0][0], "nbytes": nb})
    manifest["shard_groups"] = {"main": [MAIN_SHARD],
                                "mtp": [MTP_SHARD]}
    manifest["ntensors"] = len(manifest["tensor_map"])
    with open(os.path.join(MODEL, "apus.index.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"  container: {len(hdr)} main + {len(entries)} mtp tensors, "
          f"{len(manifest['expert_slabs'])} slabs "
          f"(mtp layer {n_layers}, slab {slab_bytes} B)")


def load_named():
    """Read BOTH shards back (the golden path consumes ONLY container
    bytes); per-expert slab slices restacked into the fused names."""
    named = {}
    gu, dn = {}, {}
    for shard in (MAIN_SHARD, MTP_SHARD):
        hdr, ds = stutil.read_shard(os.path.join(MODEL, shard))
        raw = stutil.read_tensor_bytes(os.path.join(MODEL, shard), hdr,
                                       ds)
        for name, meta in hdr.items():
            b = raw[name]
            if meta["dtype"] == "BF16":
                val = oracle.bf16_widen_f64(
                    np.frombuffer(b, dtype=np.uint16)
                    .reshape(meta["shape"]))
            else:
                val = np.frombuffer(b, dtype=np.float32).reshape(
                    meta["shape"]).astype(np.float64)
            m = re.match(r"^(.*)\.mlp\.experts\.(\d+)\."
                         r"(gate_up_proj|down_proj)\.weight$", name)
            if m:
                key = (m.group(1), int(m.group(2)))
                (gu if m.group(3) == "gate_up_proj" else dn)[key] = val
            else:
                named[name] = val
    E = CFG["num_experts"]
    for pre in sorted({k[0] for k in gu}):
        named[pre + ".mlp.experts.gate_up_proj"] = np.stack(
            [gu[(pre, e)] for e in range(E)], axis=0)
        named[pre + ".mlp.experts.down_proj"] = np.stack(
            [dn[(pre, e)] for e in range(E)], axis=0)
    return named


def bf16_codes(f64):
    return oracle.bf16_narrow_oracle_f32(
        np.asarray(f64, dtype=np.float64).astype(np.float32))


def top2_gap(logits):
    s = np.sort(np.asarray(logits, dtype=np.float64))
    return float(s[-1] - s[-2])


def dump(name, arr):
    np.asarray(arr).tofile(os.path.join(OUT, name))


def main():
    os.makedirs(OUT, exist_ok=True)
    build_container()
    named = load_named()

    rng = np.random.default_rng(20260812)
    T, H, V = 12, CFG["hidden_size"], CFG["vocab_size"]
    # mtp_prefill golden: random bf16 pre-final-norm hiddens + emb ids
    h = oracle.rnd16(rng.standard_normal((T, H)) * 1.5)
    ids = rng.integers(0, V, T)
    dump("mtp_pre_h.bin", bf16_codes(h))
    dump("mtp_pre_ids.bin", ids.astype(np.int64))
    for mode in ("f32", "f64"):
        logits, _ = oracle.mtp_forward(named, h, ids, mode, CFG)
        dump(f"mtp_pre_logits_{mode}.bin", logits)
    l32 = np.fromfile(os.path.join(OUT, "mtp_pre_logits_f32.bin")
                      ).reshape(T, V)
    gaps = np.array([top2_gap(r) for r in l32])
    dump("mtp_pre_gap.bin", gaps)

    # mtp_chain golden: 3-step draft chain from a seed pair
    dump("mtp_chain_h0.bin", bf16_codes(h[0]))
    dump("mtp_chain_emb0.bin", np.array([int(ids[0])], dtype=np.int64))
    chain = oracle.mtp_chain(named, h[0].astype(np.float64),
                             int(ids[0]), 3, "f32", CFG)
    dump("mtp_chain_drafts.bin",
         np.array([c[0] for c in chain], dtype=np.int64))
    for i, (_, lg) in enumerate(chain):
        dump(f"mtp_chain_step{i}_logits.bin", lg)
        dump(f"mtp_chain_step{i}_gap.bin",
             np.array([top2_gap(lg)]))
    print("  goldens: prefill T=12, chain 3 steps, drafts",
          [c[0] for c in chain])

    with open(os.path.join(OUT, "manifest.txt"), "w") as f:
        f.write(f"T={T}\nV={V}\nH={H}\nmtp_seed={MTP_SEED}\n")
    print("m8 fixtures done")


if __name__ == "__main__":
    main()
