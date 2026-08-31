#!/usr/bin/env python3
"""tests/m0/test_oracle.py — M0 smoke self-check for the Qwen3.6 numpy
oracle (tools/oracle.py).

Builds a tiny 2-layer fixture (one Gated DeltaNet layer + one
full-attention layer, real config.json schema, real tensor naming),
writes it as a safetensors container + config.json, then runs ONLY from
weights read back from disk (the Apus M4c lesson: independent
realization paths). Checks:

  1. REPLAY: a fresh disk load + run reproduces the stored .npy goldens
     exactly (IO integrity + container round-trip).
  2. WIRING: the test's manual per-layer chain reproduces
     oracle.full_forward logits exactly.
  3. SELECTION: router f32/f64 top-k selections identical on every
     token (fixture margins large enough that C fp32 noise cannot
     flip; if this fails the fixture seed must change).
  4. CHUNK INVARIANCE: one-shot T=12 forward vs prefill 7 + 5 decode
     steps — f64 EXACT (<=1e-12); f32 zero bf16 code flips (the oracle
     realization is per-token, so both paths are the same computation).
  5. DETERMINISM (the M0 gate): two full runs produce identical
     digests.
  6. ENVELOPE (the M0 gate): f32-vs-f64 divergence on the fixture is
     small; measured max abs/rel printed, gated with headroom.

Run from the repository root:
  .venv/bin/python -m unittest discover -s tests/m0 -v
"""

import hashlib
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests", "m1"))

import oracle  # noqa: E402
import stutil  # noqa: E402

FIX = os.path.join(HERE, "fixtures")
MODEL = os.path.join(FIX, "model")
SEED = 20260831   # chosen so router top-k margins are clean (m4b policy)

# envelope gates (measured on this fixture: see README; gates carry
# ~4x headroom over the measured values)
ENV_GATE_ABS = 0.1
ENV_GATE_REL = 0.25


def write_container(named_codes):
    os.makedirs(MODEL, exist_ok=True)
    tensors = []
    weight_map = {}
    for name, arr in sorted(named_codes.items()):
        if arr.dtype == np.uint16:
            dtype = "BF16"
        elif arr.dtype == np.float32:
            dtype = "F32"
        else:
            raise ValueError(f"{name}: dtype {arr.dtype}")
        tensors.append((name, dtype, list(arr.shape),
                        arr.tobytes()))
        weight_map[name] = "apus-qwen36-00001.safetensors"
    stutil.write_shard(os.path.join(MODEL, "apus-qwen36-00001.safetensors"),
                       tensors)
    with open(os.path.join(MODEL, "model.safetensors.index.json"),
              "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    text_cfg = {k: v for k, v in oracle.MINI_CFG.items()
                if k not in ("prefill_len", "decode_len")}
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "dtype": "bfloat16",
        "text_config": text_cfg,
        "tie_word_embeddings": False,
    }
    with open(os.path.join(MODEL, "config.json"), "w") as f:
        json.dump(config, f, indent=1)


def load_container():
    """Read the container back (the golden path consumes ONLY this)."""
    path = os.path.join(MODEL, "apus-qwen36-00001.safetensors")
    hdr, data_start = stutil.read_shard(path)
    raw = stutil.read_tensor_bytes(path, hdr, data_start)
    named = {}
    for name, meta in hdr.items():
        b = raw[name]
        if meta["dtype"] == "BF16":
            codes = np.frombuffer(b, dtype=np.uint16).reshape(
                meta["shape"])
            named[name] = oracle.bf16_widen_f64(codes)
        elif meta["dtype"] == "F32":
            named[name] = np.frombuffer(b, dtype=np.float32).reshape(
                meta["shape"]).astype(np.float64)
        else:
            raise ValueError(meta["dtype"])
    return named


def run_chain(named, ids, mode, cfg, states=None):
    """full_forward with per-layer stages exposed (same wiring; the
    WIRING test asserts the logits match full_forward exactly)."""
    rnd = oracle.rnd16 if mode == "f32" else (lambda v: v)
    x = named[oracle.EMB_NAME][np.asarray(ids)]
    traces = []
    stages = []
    for L in range(cfg["num_hidden_layers"]):
        kind = oracle.layer_kind(cfg, L)
        w = oracle.full_layer_weights(named, L, kind, cfg)
        st = states[L] if states is not None else oracle.new_state(kind,
                                                                   cfg)
        x, stg = oracle.layer_forward(kind, w, x, st, mode, cfg)
        if states is not None:
            states[L] = st
        traces.append(x)
        stages.append(stg)
    y = oracle.rmsnorm(x, named[oracle.NORM_NAME], rnd,
                       cfg["rms_norm_eps"])
    logits = oracle.linear(y, named[oracle.LMHEAD_NAME], rnd)
    return logits, traces, stages


def digest_of(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


class OracleSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(FIX, exist_ok=True)
        rng = np.random.default_rng(SEED)
        named_codes, _ = oracle.gen_full_weights(rng, oracle.MINI_CFG)
        write_container(named_codes)
        cls.named = load_container()          # f64 values, from DISK
        cls.cfg = oracle.config_from_json(
            os.path.join(MODEL, "config.json"))
        cls.cfg["prefill_len"] = oracle.MINI_CFG["prefill_len"]
        cls.cfg["decode_len"] = oracle.MINI_CFG["decode_len"]

        irng = np.random.default_rng(SEED + 1)
        V = cls.cfg["vocab_size"]
        cls.ids = irng.integers(0, V, cls.cfg["prefill_len"]
                                + cls.cfg["decode_len"])
        np.save(os.path.join(FIX, "ids.npy"), cls.ids.astype(np.int64))

        # golden runs (f32 = C target, f64 = truth): prefill + decode
        cls.gold = {}
        for mode in ("f32", "f64"):
            states = [oracle.new_state(oracle.layer_kind(cls.cfg, L),
                                       cls.cfg)
                      for L in range(cls.cfg["num_hidden_layers"])]
            logits, traces, stages = run_chain(
                cls.named, cls.ids[:cls.cfg["prefill_len"]], mode,
                cls.cfg, states=states)
            cls.gold[(mode, "pre")] = (logits, traces, stages)
            np.save(os.path.join(FIX, f"prefill_logits_{mode}.npy"),
                    logits)
            for i in range(cls.cfg["decode_len"]):
                lg, tr, stg = run_chain(
                    cls.named, cls.ids[cls.cfg["prefill_len"] + i:
                                       cls.cfg["prefill_len"] + i + 1],
                    mode, cls.cfg, states=states)
                cls.gold[(mode, f"dec{i}")] = (lg, tr, stg)
                np.save(os.path.join(FIX,
                                     f"dec{i}_logits_{mode}.npy"), lg)
        with open(os.path.join(FIX, "manifest.txt"), "w") as f:
            f.write(f"seed={SEED}\n")
            f.write(f"prefill_len={cls.cfg['prefill_len']}\n")
            f.write(f"decode_len={cls.cfg['decode_len']}\n")
            f.write(f"layer_types={','.join(cls.cfg['layer_types'])}\n")

    def test_replay_from_disk(self):
        named = load_container()
        for mode in ("f32", "f64"):
            logits, traces = oracle.full_forward(
                named, self.ids[:self.cfg["prefill_len"]], mode, self.cfg)
            stored = np.load(os.path.join(
                FIX, f"prefill_logits_{mode}.npy"))
            self.assertTrue(np.array_equal(logits, stored),
                            f"{mode}: replay mismatch")

    def test_wiring_matches_full_forward(self):
        for mode in ("f32", "f64"):
            a, _ = oracle.full_forward(
                self.named, self.ids[:self.cfg["prefill_len"]], mode,
                self.cfg)
            b = self.gold[(mode, "pre")][0]
            self.assertTrue(np.array_equal(a, b),
                            f"{mode}: manual chain != full_forward")

    def test_router_selection_identity(self):
        for tag in ["pre"] + [f"dec{i}"
                              for i in range(self.cfg["decode_len"])]:
            for L in range(self.cfg["num_hidden_layers"]):
                i32 = self.gold[("f32", tag)][2][L]["rtr_idx"]
                i64 = self.gold[("f64", tag)][2][L]["rtr_idx"]
                self.assertTrue(
                    np.array_equal(i32, i64),
                    f"{tag}/L{L}: router selection f32 != f64")

    def test_chunk_invariance(self):
        for mode in ("f32", "f64"):
            one_shot, _ = oracle.full_forward(self.named, self.ids, mode,
                                              self.cfg)
            states = [oracle.new_state(oracle.layer_kind(self.cfg, L),
                                       self.cfg)
                      for L in range(self.cfg["num_hidden_layers"])]
            parts = [oracle.full_forward(
                self.named, self.ids[:self.cfg["prefill_len"]], mode,
                self.cfg, states=states)[0]]
            for i in range(self.cfg["decode_len"]):
                parts.append(oracle.full_forward(
                    self.named, self.ids[self.cfg["prefill_len"] + i:
                                         self.cfg["prefill_len"] + i + 1],
                    mode, self.cfg, states=states)[0])
            chained = np.concatenate(parts, axis=0)
            d = float(np.max(np.abs(one_shot - chained)))
            if mode == "f64":
                self.assertLessEqual(d, 1e-12, f"f64 chunk invariance {d}")
            else:
                c1 = oracle.bf16_narrow_oracle_f32(
                    one_shot.astype(np.float32))
                c2 = oracle.bf16_narrow_oracle_f32(
                    chained.astype(np.float32))
                flips = int(np.count_nonzero(c1 != c2))
                self.assertEqual(flips, 0,
                                 f"f32 chunk invariance: {flips} flips")
            print(f"\nchunk invariance {mode}: max diff {d:.3g}")

    def test_determinism_digest(self):
        digests = []
        for _ in range(2):
            states = [oracle.new_state(oracle.layer_kind(self.cfg, L),
                                       self.cfg)
                      for L in range(self.cfg["num_hidden_layers"])]
            arrays = [oracle.full_forward(
                self.named, self.ids[:self.cfg["prefill_len"]], "f32",
                self.cfg, states=states)[0]]
            for i in range(self.cfg["decode_len"]):
                arrays.append(oracle.full_forward(
                    self.named, self.ids[self.cfg["prefill_len"] + i:
                                         self.cfg["prefill_len"] + i + 1],
                    "f32", self.cfg, states=states)[0])
            for L, st in enumerate(states):
                arrays.extend(oracle.state_arrays(
                    oracle.layer_kind(self.cfg, L), st).values())
            digests.append(digest_of(arrays))
        self.assertEqual(digests[0], digests[1], "non-deterministic run")
        print(f"\ndigest(f32 prefill+decode+states): {digests[0]}")

    def test_envelope(self):
        rows = []
        tags = ["pre"] + [f"dec{i}"
                          for i in range(self.cfg["decode_len"])]
        for tag in tags:
            a32 = self.gold[("f32", tag)][0]
            a64 = self.gold[("f64", tag)][0]
            m = float(np.max(np.abs(a32 - a64)))
            r = m / float(np.max(np.abs(a64)))
            rows.append((f"logits/{tag}", m, r))
        for L in range(self.cfg["num_hidden_layers"]):
            a32 = self.gold[("f32", "pre")][1][L]
            a64 = self.gold[("f64", "pre")][1][L]
            m = float(np.max(np.abs(a32 - a64)))
            r = m / float(np.max(np.abs(a64)))
            rows.append((f"h_after_L{L}/pre", m, r))
        worst_abs = max(m for _, m, _ in rows)
        worst_rel = max(r for _, _, r in rows)
        print("\nf32-vs-f64 envelope (fixture):")
        for name, m, r in rows:
            print(f"  {name:18s} maxabs {m:10.4g} rel {r:10.4g}")
        print(f"  WORST              maxabs {worst_abs:10.4g} "
              f"rel {worst_rel:10.4g}")
        self.assertLessEqual(worst_abs, ENV_GATE_ABS,
                             f"envelope abs {worst_abs} > {ENV_GATE_ABS}")
        self.assertLessEqual(worst_rel, ENV_GATE_REL,
                             f"envelope rel {worst_rel} > {ENV_GATE_REL}")


if __name__ == "__main__":
    unittest.main()
