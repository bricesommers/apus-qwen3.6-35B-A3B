"""M1 test 5 — converter assumptions vs the REAL checkpoint index.

Parses reference/model.safetensors.index.json (the real 26-shard, 1,045-
tensor Qwen3.6-35B-A3B index) and asserts every assumption the converter
relies on:

  * the naming scheme matches the converter's regexes (fused expert
    tensors, language_model prefix, top-level mtp.* block),
  * every MoE layer (all 40 main layers + mtp.layers.0) has exactly the
    2-tensor fused group {gate_up_proj, down_proj},
  * cross-shard expert layers exist and are pinned: 17 of 41 layers have
    their two fused tensors in different input shards (the converter's
    deferral machinery exists for these; if a future repack changes the
    count, this test fails loudly and forces a re-audit),
  * BF16 layout arithmetic from config.json dims:
    gate_up_proj [E, 2M, H], down_proj [E, H, M], 2 bytes per element,
  * per-expert byte count (6 MiB) and total size bookkeeping,
  * the vision tower is exactly the model.visual.* prefix (stripped),
  * the hybrid layout: full attention every 4th layer, linear attention
    elsewhere.

NOTE (documented deviation): the real index maps tensor name -> shard file
only; it carries NO shapes/dtypes. Shapes/dtypes are read from each shard's
safetensors header at conversion time. This test therefore validates shapes
against config.json-derived expectations plus total-size arithmetic.
"""

import json
import os
import re
import sys
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))
import convert as apus_convert  # noqa: E402 — reuse the converter's regex

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "reference")


class TestIndexRealism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(REF_DIR, "model.safetensors.index.json")) as f:
            cls.index = json.load(f)
        cls.wm = cls.index["weight_map"]
        with open(os.path.join(REF_DIR, "config.json")) as f:
            cls.cfg = json.load(f)
        cls.text = cls.cfg["text_config"]

    def test_scale_and_counts(self):
        self.assertEqual(len(set(self.wm.values())), 26)
        self.assertEqual(len(self.wm), 1045)
        self.assertEqual(self.index["metadata"]["total_size"],
                         71903645408)

    def test_config_nested_text_config(self):
        """The real config nests the text model under text_config; the
        converter reads num_hidden_layers from there."""
        self.assertNotIn("num_hidden_layers", self.cfg)
        self.assertEqual(self.text["num_hidden_layers"], 40)
        self.assertEqual(self.cfg["model_type"], "qwen3_5_moe")
        self.assertEqual(apus_convert.num_hidden_layers(REF_DIR), 40)

    def test_fused_expert_groups_complete(self):
        """Every MoE layer carries exactly the 2 fused tensors; MoE is on
        ALL 40 main layers (no dense layers) + mtp.layers.0."""
        layers = defaultdict(set)
        for name in self.wm:
            m = apus_convert.FUSED_EXPERT_RE.match(name)
            if m:
                group, layer_idx = apus_convert._match_group_layer(m)
                layers[(group, layer_idx)].add(m.group(4))
        n_layers = self.text["num_hidden_layers"]
        self.assertEqual(len(layers), n_layers + 1)   # 41
        for layer in range(n_layers):
            self.assertIn(("main", layer), layers)
        self.assertIn(("mtp", 0), layers)
        for key, members in layers.items():
            self.assertEqual(members, {"gate_up_proj", "down_proj"},
                             f"expert layer {key} incomplete")
        # no OTHER tensor matches the fused prefix (e.g. no .weight suffix)
        for name in self.wm:
            if ".mlp.experts." in name:
                self.assertIsNotNone(
                    apus_convert.FUSED_EXPERT_RE.match(name), name)

    def test_expert_layers_split_across_shards_pinned(self):
        """17 of 41 layers have gate_up_proj and down_proj in DIFFERENT
        input shards — the converter must (and does) assemble these
        cross-shard, and the downloader must hold shards until consumed."""
        homes = defaultdict(set)
        for name, shard in self.wm.items():
            m = apus_convert.FUSED_EXPERT_RE.match(name)
            if m:
                homes[apus_convert._match_group_layer(m)].add(shard)
        self.assertEqual(len(homes), 41)
        split = {k: v for k, v in homes.items() if len(v) > 1}
        self.assertEqual(len(split), 17)

    def test_vision_tower_is_exactly_the_visual_prefix(self):
        visual = [n for n in self.wm if n.startswith("model.visual.")]
        self.assertEqual(len(visual), 333)
        self.assertEqual(len(self.wm) - len(visual), 712)
        for name in visual:
            self.assertEqual(apus_convert.classify_group(name), "strip")
        # nothing outside model.visual.* is stripped
        for name in self.wm:
            if not name.startswith("model.visual."):
                self.assertNotEqual(apus_convert.classify_group(name),
                                    "strip", name)

    def test_group_classification(self):
        for name in self.wm:
            g = apus_convert.classify_group(name)
            if name.startswith("model.visual."):
                self.assertEqual(g, "strip", name)
            elif name.startswith("mtp."):
                self.assertEqual(g, "mtp", name)
            else:
                self.assertEqual(g, "main", name)

    def test_per_shard_census_consistent(self):
        """Shard naming is sequential and the census sums to the total."""
        shards = sorted(set(self.wm.values()))
        pat = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
        idxs = []
        for s in shards:
            m = pat.match(s)
            self.assertIsNotNone(m, s)
            self.assertEqual(m.group(2), "00026", s)
            idxs.append(int(m.group(1)))
        self.assertEqual(idxs, list(range(1, 27)))
        census = defaultdict(int)
        for shard in self.wm.values():
            census[shard] += 1
        self.assertEqual(sum(census.values()), len(self.wm))

    def test_bf16_layout_and_per_expert_bytes(self):
        H = self.text["hidden_size"]              # 2048
        M = self.text["moe_intermediate_size"]    # 512
        E = self.text["num_experts"]              # 256
        self.assertEqual((H, M, E), (2048, 512, 256))
        # gate_up_proj [E, 2M, H] BF16; down_proj [E, H, M] BF16
        elem = 2
        gu_slice = 2 * M * H * elem
        d_slice = H * M * elem
        self.assertEqual(gu_slice, 4194304)       # 4 MiB
        self.assertEqual(d_slice, 2097152)        # 2 MiB
        per_expert = gu_slice + d_slice
        self.assertEqual(per_expert, 6291456)     # 6 MiB
        # Bookkeeping: 41 x 256 experts account for ~61.5 GiB of the
        # ~67.0 GiB total; the ~5.5 GiB remainder is the dense text set
        # (attention/embeddings/norms/router/shared expert) plus the
        # stripped vision tower.
        expert_total = 41 * E * per_expert
        total = self.index["metadata"]["total_size"]
        rest = total - expert_total
        self.assertEqual(expert_total, 66035122176)
        self.assertTrue(5 * 1024**3 < rest < 7 * 1024**3,
                        f"dense+visual remainder {rest / 1024**3:.2f} GiB "
                        f"unexpected")
        print(f"\nindex realism: 10,496 experts x {per_expert:,} B "
              f"= {expert_total / 1e9:.2f} GB; dense+visual remainder "
              f"{rest / 1e9:.2f} GB of {total / 1e9:.2f} GB total")

    def test_hybrid_attention_split(self):
        """Full-attention layers are 3, 7, ..., 39 (every 4th); every other
        layer uses the linear-attention (Gated DeltaNet) block."""
        n_layers = self.text["num_hidden_layers"]
        self.assertEqual(self.text["full_attention_interval"], 4)
        full = sorted(
            int(m.group(1)) for n in self.wm
            for m in [re.match(
                r"model\.language_model\.layers\.(\d+)\.self_attn\.q_proj"
                r"\.weight$", n)] if m)
        lin = sorted(
            int(m.group(1)) for n in self.wm
            for m in [re.match(
                r"model\.language_model\.layers\.(\d+)\.linear_attn\.A_log$",
                n)] if m)
        self.assertEqual(full, list(range(3, n_layers, 4)))
        self.assertEqual(sorted(set(range(n_layers)) - set(full)), lin)
        # mtp.layers.0 is a full-attention layer
        self.assertIn("mtp.layers.0.self_attn.q_proj.weight", self.wm)
        self.assertNotIn("mtp.layers.0.linear_attn.A_log", self.wm)

    def test_mtp_block(self):
        """The MTP block is a TOP-LEVEL mtp.* set: fc, two pre-FC norms,
        final norm, and one full layer (mtp.layers.0.*) with the same
        layout as a main layer."""
        for name in ("mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight",
                     "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight"):
            self.assertIn(name, self.wm)
        self.assertIn("mtp.layers.0.mlp.experts.gate_up_proj", self.wm)
        self.assertIn("mtp.layers.0.mlp.experts.down_proj", self.wm)
        # no main-model tensor claims the mtp prefix
        for name in self.wm:
            self.assertFalse(name.startswith("model.mtp."), name)

    def test_globals(self):
        self.assertIn("model.language_model.embed_tokens.weight", self.wm)
        self.assertIn("lm_head.weight", self.wm)
        self.assertIn("model.language_model.norm.weight", self.wm)

    def test_name_patterns_pinned(self):
        """Guard against silent naming drift in future checkpoints: the set
        of name patterns must stay exactly what the converter was built
        for (61 patterns including the vision tower, 46 without)."""
        pats = {re.sub(r"\d+", "N", n) for n in self.wm}
        self.assertEqual(len(pats), 61)
        nonvis = {p for p in pats if not p.startswith("model.visual.")}
        self.assertEqual(len(nonvis), 46)


if __name__ == "__main__":
    unittest.main(verbosity=2)
