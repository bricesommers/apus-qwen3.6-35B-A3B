"""M1 test 1 — byte identity + vision strip.

Convert the synthetic fixture checkpoint, then compare EVERY output tensor's
bytes against the source bytes (dense tensors whole; fused expert tensors
slice by slice). Any difference fails. Also checks dtype and shape are
preserved verbatim, that the stripped vision tower (model.visual.*) is
absent from the output, that the manifest agrees with the output shard
headers, and cross-checks numpy-readable tensors via the safetensors
library (allowed for reads where dtypes permit).
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))

import fixtures
import stutil
import convert as apus_convert

TARGET_BYTES = 64 * 1024  # small shards -> exercises multi-shard packing


class TestByteIdentity(unittest.TestCase):
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

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _src_tensors(self):
        """name -> (payload, dtype, shape) across all source shards."""
        out = {}
        for shard in sorted(os.listdir(self.src)):
            if not shard.endswith(".safetensors"):
                continue
            spath = os.path.join(self.src, shard)
            header, _ = stutil.read_shard(spath)
            for name, payload in stutil.read_tensor_bytes(spath).items():
                out[name] = (payload, header[name]["dtype"],
                             header[name]["shape"])
        return out

    def test_tensor_count_preserved(self):
        """Output = source - stripped visual - fused tensors + slices."""
        src = self._src_tensors()
        n_visual = sum(1 for n in src if n.startswith("model.visual."))
        n_fused = sum(1 for n in src
                      if apus_convert.FUSED_EXPERT_RE.match(n))
        n_slices = n_fused * fixtures.N_EXPERTS
        expected = len(src) - n_visual - n_fused + n_slices
        with open(os.path.join(self.dst, "apus.index.json")) as f:
            manifest = json.load(f)
        self.assertEqual(manifest["ntensors"], expected)
        self.assertEqual(len(manifest["tensor_map"]), expected)
        self.assertEqual(manifest["stripped_tensors"], n_visual)
        self.assertEqual(n_visual, 4)

    def test_visual_stripped_absent(self):
        """No vision-tower tensor (or any byte region named after it) may
        appear in the output; the strip is counted in the manifest."""
        with open(os.path.join(self.dst, "apus.index.json")) as f:
            manifest = json.load(f)
        for name in manifest["tensor_map"]:
            self.assertFalse(name.startswith("model.visual."), name)
        for shard in sorted(os.listdir(self.dst)):
            if not shard.endswith(".safetensors"):
                continue
            header, _ = stutil.read_shard(os.path.join(self.dst, shard))
            for name in header:
                self.assertFalse(name.startswith("model.visual."), name)

    def test_every_tensor_byte_identical(self):
        with open(os.path.join(self.dst, "apus.index.json")) as f:
            manifest = json.load(f)
        tmap = manifest["tensor_map"]
        nchecked = 0
        for name, (payload, dtype, shape) in self._src_tensors().items():
            if name.startswith("model.visual."):
                continue
            m = apus_convert.FUSED_EXPERT_RE.match(name)
            if m:
                # Slice-level: each expert's output slice must equal the
                # matching region of the fused source tensor.
                group, layer_idx = apus_convert._match_group_layer(m)
                member = m.group(4)
                n_experts = shape[0]
                slice_nbytes = len(payload) // n_experts
                for e in range(n_experts):
                    out_name = apus_convert._slice_name(group, layer_idx,
                                                        e, member)
                    rec = tmap[out_name]
                    self.assertEqual(rec["dtype"], dtype, out_name)
                    self.assertEqual(rec["shape"], shape[1:], out_name)
                    self.assertEqual(rec["nbytes"], slice_nbytes, out_name)
                    with open(os.path.join(self.dst, rec["shard"]),
                              "rb") as f:
                        f.seek(rec["offset"])
                        out = f.read(rec["nbytes"])
                    want = payload[e * slice_nbytes:
                                   (e + 1) * slice_nbytes]
                    self.assertEqual(out, want,
                                     f"{out_name}: slice bytes differ")
                    nchecked += 1
                continue
            rec = tmap[name]
            self.assertEqual(rec["dtype"], dtype, name)
            self.assertEqual(rec["shape"], shape, name)
            self.assertEqual(rec["nbytes"], len(payload), name)
            with open(os.path.join(self.dst, rec["shard"]), "rb") as f:
                f.seek(rec["offset"])
                out = f.read(rec["nbytes"])
            self.assertEqual(out, payload,
                             f"{name}: bytes differ after conversion")
            nchecked += 1
        self.assertEqual(nchecked, len(tmap))

    def test_verify_helper(self):
        n = apus_convert.verify_source(self.src, self.dst,
                                       log=lambda m: None)
        self.assertGreater(n, 0)

    def test_safetensors_lib_crosscheck(self):
        """The safetensors library must parse our manually-written output
        shards, and every dtype must be one the Qwen checkpoint uses
        (BF16 weights + F32 GDN scalars)."""
        from safetensors import safe_open
        for shard in sorted(os.listdir(self.dst)):
            if not shard.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(self.dst, shard),
                           framework="numpy") as f:
                for name in f.keys():
                    meta = f.get_slice(name)
                    self.assertIn(meta.get_dtype(), ("BF16", "F32"))


if __name__ == "__main__":
    unittest.main()
