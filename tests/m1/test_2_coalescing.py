"""M1 test 2 — coalesced per-expert layout (sliced from fused tensors).

Parses the output shard headers manually and asserts, for every expert:

  * its 2 slice tensors {gate_up_proj,down_proj}.weight appear consecutively
    in the shard header, in that exact order,
  * their data regions are contiguous and adjacent (end == next start),
  * the whole slab lives in one shard,
  * the manifest's slab record (layer, expert, shard, offset, nbytes)
    matches the actual header offsets exactly,
  * slab size equals the expected per-expert byte count,
  * slab bytes equal the corresponding SLICES of the fused source tensors
    (which may live in two different input shards),
  * MTP-block experts (slab layer == num_hidden_layers) live exclusively in
    apus-qwen-mtp-* shards, main experts in apus-qwen-*.
"""

import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))

import fixtures
import stutil
import convert as apus_convert

TARGET_BYTES = 64 * 1024
SLAB_ORDER = ["gate_up_proj.weight", "down_proj.weight"]


class TestCoalescing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.tmp.name, "src")
        cls.dst = os.path.join(cls.tmp.name, "out")
        os.makedirs(cls.src)
        fixtures.make_fixture_tree(cls.src)
        conv = apus_convert.Converter(cls.src, cls.dst,
                                      target_bytes=TARGET_BYTES)
        conv.convert()
        conv.finalize()
        with open(os.path.join(cls.dst, "apus.index.json")) as f:
            cls.manifest = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _experts_from_headers(self):
        """Rebuild per-expert layout facts from the raw output headers.
        Returns (slab_layer, eid) -> {"shard", "members", "positions"}."""
        experts = {}
        for shard in sorted(os.listdir(self.dst)):
            if not shard.endswith(".safetensors"):
                continue
            header, data_start = stutil.read_shard(os.path.join(self.dst,
                                                                shard))
            names = [n for n in header if n != "__metadata__"]
            for pos, name in enumerate(names):
                m = apus_convert.OUT_EXPERT_RE.match(name)
                if not m:
                    continue
                grp, lidx = apus_convert._match_group_layer(m)
                slab_layer = lidx if grp == "main" else \
                    fixtures.N_LAYERS + lidx
                key = (slab_layer, int(m.group(4)))
                begin, end = header[name]["data_offsets"]
                rec = experts.setdefault(
                    key, {"shard": shard, "members": [], "positions": []})
                self.assertEqual(rec["shard"], shard,
                                 f"expert {key} straddles shards")
                rec["members"].append(
                    (f"{m.group(5)}.weight", data_start + begin,
                     end - begin))
                rec["positions"].append(pos)
        return experts

    def test_slabs_contiguous_and_adjacent(self):
        experts = self._experts_from_headers()
        self.assertEqual(len(experts),
                         fixtures.N_EXPERT_LAYERS * fixtures.N_EXPERTS)
        for key, rec in experts.items():
            # consecutive positions in the header, in the fixed slab order
            positions = rec["positions"]
            self.assertEqual(positions, list(range(positions[0],
                                                   positions[0] + 2)),
                             f"{key}: slab members not adjacent in header")
            by_pos = [m for _, m in sorted(
                zip(positions, rec["members"]), key=lambda pm: pm[0])]
            self.assertEqual([s for s, _, _ in by_pos], SLAB_ORDER,
                             f"{key}: wrong intra-slab order")
            # data regions contiguous and adjacent
            (s0, off0, nb0), (s1, off1, nb1) = by_pos
            self.assertEqual(off0 + nb0, off1,
                             f"{key}: gap between {s0} and {s1}")
            total = nb0 + nb1
            self.assertEqual(total, fixtures.PER_EXPERT_BYTES,
                             f"{key}: unexpected per-expert byte count")

    def test_manifest_slab_records_match_headers(self):
        experts = self._experts_from_headers()
        slabs = {(s["layer"], s["expert"]): s
                 for s in self.manifest["expert_slabs"]}
        self.assertEqual(set(slabs), set(experts))
        for key, rec in experts.items():
            by_pos = [m for _, m in sorted(
                zip(rec["positions"], rec["members"]), key=lambda pm: pm[0])]
            slab_start = by_pos[0][1]
            slab_bytes = sum(nb for _, _, nb in by_pos)
            rec_m = slabs[key]
            self.assertEqual(rec_m["shard"], rec["shard"], key)
            self.assertEqual(rec_m["offset"], slab_start, key)
            self.assertEqual(rec_m["nbytes"], slab_bytes, key)

    def test_slab_bytes_equal_source_slices(self):
        """Every slab's bytes must equal the matching slices of the fused
        source tensors — including layers whose gate_up_proj and down_proj
        live in DIFFERENT input shards (fixture layers 1, 2 and mtp)."""
        with open(os.path.join(self.src,
                               "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        src_bytes = {}
        for shard in sorted(os.listdir(self.src)):
            if shard.endswith(".safetensors"):
                src_bytes.update(stutil.read_tensor_bytes(
                    os.path.join(self.src, shard)))
        slabs = {(s["layer"], s["expert"]): s
                 for s in self.manifest["expert_slabs"]}
        self.assertEqual(len(slabs),
                         fixtures.N_EXPERT_LAYERS * fixtures.N_EXPERTS)
        for (layer, eid), rec in slabs.items():
            if layer >= fixtures.N_LAYERS:
                pre = f"mtp.layers.{layer - fixtures.N_LAYERS}"
            else:
                pre = f"model.language_model.layers.{layer}"
            gu = src_bytes[f"{pre}.mlp.experts.gate_up_proj"]
            d = src_bytes[f"{pre}.mlp.experts.down_proj"]
            gu_slice = len(gu) // fixtures.N_EXPERTS
            d_slice = len(d) // fixtures.N_EXPERTS
            want = (gu[eid * gu_slice:(eid + 1) * gu_slice] +
                    d[eid * d_slice:(eid + 1) * d_slice])
            with open(os.path.join(self.dst, rec["shard"]), "rb") as f:
                f.seek(rec["offset"])
                got = f.read(rec["nbytes"])
            self.assertEqual(got, want,
                             f"expert L{layer}/{eid}: slab bytes differ "
                             f"from source slices")

    def test_manifest_offsets_match_headers_for_all_tensors(self):
        """Every tensor_map entry must point at the same bytes the output
        shard header describes."""
        tmap = self.manifest["tensor_map"]
        n = 0
        for shard in sorted(os.listdir(self.dst)):
            if not shard.endswith(".safetensors"):
                continue
            header, data_start = stutil.read_shard(os.path.join(self.dst,
                                                                shard))
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                rec = tmap[name]
                self.assertEqual(rec["shard"], shard, name)
                self.assertEqual(rec["offset"],
                                 data_start + meta["data_offsets"][0], name)
                self.assertEqual(rec["nbytes"],
                                 meta["data_offsets"][1]
                                 - meta["data_offsets"][0], name)
                n += 1
        self.assertEqual(n, len(tmap))

    def test_mtp_slabs_land_in_mtp_group(self):
        """Experts of the MTP block (slab layer == num_hidden_layers) must
        live exclusively in apus-qwen-mtp-* shards, main experts in
        apus-qwen-*."""
        experts = self._experts_from_headers()
        self.assertTrue(experts)
        # "apus-qwen-mtp-..." also startswith "apus-qwen-"; match the full
        # shard-name shape instead of a bare prefix.
        main_re = re.compile(rf"^{apus_convert.MAIN_PREFIX}-\d{{5}}"
                             r"\.safetensors$")
        mtp_re = re.compile(rf"^{apus_convert.MTP_PREFIX}-\d{{5}}"
                            r"\.safetensors$")
        for (layer, _eid), rec in experts.items():
            if layer >= fixtures.N_LAYERS:
                self.assertIsNotNone(mtp_re.match(rec["shard"]),
                                     f"MTP expert in shard {rec['shard']}")
            else:
                self.assertIsNotNone(main_re.match(rec["shard"]),
                                     f"main expert in shard {rec['shard']}")
        # every mtp tensor (dense included) lands in the mtp group
        for name, rec in self.manifest["tensor_map"].items():
            if name.startswith("mtp."):
                self.assertIsNotNone(mtp_re.match(rec["shard"]),
                                     f"{name} in shard {rec['shard']}")
            else:
                self.assertIsNotNone(main_re.match(rec["shard"]),
                                     f"{name} in shard {rec['shard']}")


if __name__ == "__main__":
    unittest.main()
