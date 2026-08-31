"""M1 test 3 — BF16 value spot-check.

Qwen3.6-35B-A3B ships BF16 weights (no quantization), so this test replaces
the original suite's MXFP4 dequant reference with the equivalent rigor for
the one numeric format the container actually carries:

  * BF16 decode reference: a BF16 value widens to FP32 by zero-extending the
    low 16 bits — bits = (uint16 << 16) viewed as float32. The C loader will
    rely on exactly this (bf16 -> f32 is a plain shift).
  * Hand-computed and boundary cases: zero, subnormals, min/max normal,
    infinities, NaN, and a rounding case (pi).
  * Conversion round-trip: a source shard holding known BF16 patterns —
    including a FUSED expert tensor whose per-expert slices are cut by the
    converter — is run through tools/convert.py; the output payload bytes
    must be bit-identical and must decode to exactly the same FP32 values.

The byte-identity contract guarantees exact round-trip (the converter never
interprets payloads); this test pins the decode semantics so a wrong
endianness or shift assumption in a future reader fails loudly here.
"""

import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))

import stutil
import convert as apus_convert


def decode_bf16(bits):
    """uint16 BF16 bit patterns -> float32 values (widen: << 16)."""
    bits = np.asarray(bits, dtype=np.uint16)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def encode_bf16_exact(values):
    """float32 values with zero low 16 bits -> uint16 BF16 patterns."""
    f32 = np.asarray(values, dtype=np.float32)
    assert np.all(f32.view(np.uint32) & np.uint32(0xFFFF) == 0), \
        "test values must be exactly representable in BF16"
    return (f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


class TestBF16Decode(unittest.TestCase):
    def test_hand_computed_values(self):
        cases = [
            (0x0000, 0.0),
            (0x8000, -0.0),
            (0x3F80, 1.0),                       # 0 01111111 0000000
            (0xBF80, -1.0),
            (0x4049, 3.140625),                  # pi truncated to BF16
            (0x0080, 2.0 ** -126),               # min normal
            (0x0001, 2.0 ** -133),               # min subnormal (2^-7 * 2^-126)
            (0x007F, 127.0 * 2.0 ** -133),       # max subnormal
            (0x7F7F, 255.0 * 2.0 ** 120),        # max normal ~3.39e38
        ]
        for bits, expected in cases:
            got = decode_bf16(np.array([bits]))[0]
            self.assertEqual(got, np.float32(expected), f"bits {bits:#06x}")
            # sign must survive for -0.0
            if expected == 0.0:
                self.assertEqual(np.signbit(got), np.signbit(expected),
                                 f"bits {bits:#06x}")

    def test_special_values(self):
        self.assertTrue(np.isinf(decode_bf16([0x7F80])[0]))
        self.assertTrue(np.isinf(decode_bf16([0xFF80])[0]))
        self.assertLess(decode_bf16([0xFF80])[0], 0.0)
        self.assertTrue(np.isnan(decode_bf16([0x7FC0])[0]))

    def test_structure_from_bit_definition(self):
        """Derive a value from the BF16 bit layout (s eeeeeeee mmmmmmm) and
        compare against the shift-widen reference, for random patterns."""
        rng = np.random.RandomState(42)
        for bits in rng.randint(0, 65536, size=1000):
            sign = -1.0 if bits & 0x8000 else 1.0
            exp = (bits >> 7) & 0xFF
            man = bits & 0x7F
            if exp == 0:
                val = sign * (man / 128.0) * 2.0 ** -126   # subnormal
            elif exp == 255:
                continue                                    # inf/nan
            else:
                val = sign * (1.0 + man / 128.0) * 2.0 ** (exp - 127)
            got = decode_bf16([bits])[0]
            self.assertEqual(got, np.float32(val), f"bits {bits:#06x}")

    def test_encode_decode_roundtrip_exact_values(self):
        """Exactly-representable values survive f32 -> bf16 -> f32."""
        vals = [0.0, 1.0, -1.0, 0.5, 3.140625, 2.0 ** -126,
                255.0 * 2.0 ** 120, 65536.0, -2.0 ** 100]
        bits = encode_bf16_exact(vals)
        back = decode_bf16(bits)
        np.testing.assert_array_equal(back, np.asarray(vals, dtype=np.float32))


class TestBF16ConversionRoundTrip(unittest.TestCase):
    """Known BF16 payloads through tools/convert.py: bytes and decoded
    values must be exactly preserved — including per-expert slices cut
    from a fused gate_up_proj tensor."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.tmp.name, "src")
        cls.dst = os.path.join(cls.tmp.name, "out")
        os.makedirs(cls.src)
        rng = np.random.RandomState(7)

        # Fused expert tensor [2 experts, 3, 4]: expert 0's slice gets the
        # boundary set, expert 1's slice gets exactly-representable values.
        # A second fused tensor [2, 4, 3] gets random bits; the converter
        # must cut both per expert and coalesce the slices.
        boundary = np.array(
            [0x0000, 0x8000, 0x3F80, 0xBF80, 0x4049, 0x0080, 0x0001,
             0x007F, 0x7F7F, 0x7F80, 0xFF80, 0x7FC0], dtype=np.uint16)
        exact = encode_bf16_exact([float(v) for v in range(-6, 6)])
        random_bits = rng.randint(0, 65536, size=24).astype(np.uint16)
        cls.gate_up = np.concatenate([boundary, exact])     # [2, 3, 4]
        cls.down = random_bits                               # [2, 4, 3]
        pre = "model.language_model.layers.2"
        tensors = [
            (f"{pre}.mlp.experts.gate_up_proj", "BF16", [2, 3, 4],
             cls.gate_up.tobytes()),
            (f"{pre}.mlp.experts.down_proj", "BF16", [2, 4, 3],
             cls.down.tobytes()),
            ("model.language_model.embed_tokens.weight", "BF16", [1, 1],
             np.array([0x3F80], dtype=np.uint16).tobytes()),
        ]
        stutil.write_shard(
            os.path.join(cls.src, "model-00001-of-00001.safetensors"),
            tensors)
        with open(os.path.join(cls.src, "model.safetensors.index.json"),
                  "w") as f:
            json.dump({
                "metadata": {"total_size": sum(
                    stutil.tensor_nbytes(d, s) for _, d, s, _ in tensors)},
                "weight_map": {n: "model-00001-of-00001.safetensors"
                               for n, _, _, _ in tensors},
            }, f)
        with open(os.path.join(cls.src, "config.json"), "w") as f:
            json.dump({"model_type": "qwen3_5_moe",
                       "text_config": {"num_hidden_layers": 40}}, f)

        conv = apus_convert.Converter(cls.src, cls.dst,
                                      target_bytes=64 * 1024)
        conv.convert()
        conv.finalize()
        with open(os.path.join(cls.dst, "apus.index.json")) as f:
            cls.manifest = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _read(self, name):
        rec = self.manifest["tensor_map"][name]
        with open(os.path.join(self.dst, rec["shard"]), "rb") as f:
            f.seek(rec["offset"])
            return rec, f.read(rec["nbytes"])

    def test_payload_bytes_bit_identical(self):
        pre = "model.language_model.layers.2.mlp.experts"
        for e in range(2):
            gu_want = self.gate_up[e * 12:(e + 1) * 12].tobytes()
            d_want = self.down[e * 12:(e + 1) * 12].tobytes()
            rec, out = self._read(f"{pre}.{e}.gate_up_proj.weight")
            self.assertEqual(rec["dtype"], "BF16")
            self.assertEqual(rec["shape"], [3, 4])
            self.assertEqual(out, gu_want,
                             f"expert {e} gate_up slice changed")
            rec, out = self._read(f"{pre}.{e}.down_proj.weight")
            self.assertEqual(rec["dtype"], "BF16")
            self.assertEqual(rec["shape"], [4, 3])
            self.assertEqual(out, d_want, f"expert {e} down slice changed")

    def test_decoded_values_exact(self):
        """Output payload decoded as BF16->FP32 equals the source values
        bit-for-bit (including -0.0 and the inf/nan specials)."""
        pre = "model.language_model.layers.2.mlp.experts"
        for e in range(2):
            _, out = self._read(f"{pre}.{e}.gate_up_proj.weight")
            src_f32 = decode_bf16(self.gate_up[e * 12:(e + 1) * 12])
            out_f32 = decode_bf16(
                np.frombuffer(out, dtype=np.uint16))
            same = (src_f32 == out_f32) | (np.isnan(src_f32)
                                           & np.isnan(out_f32))
            self.assertTrue(np.all(same),
                            f"expert {e}: decoded values differ")
            # -0.0 / +0.0 compare equal; check the sign bits explicitly
            np.testing.assert_array_equal(np.signbit(src_f32),
                                          np.signbit(out_f32))

    def test_little_endian_layout(self):
        """The safetensors data region is little-endian: the first two
        payload bytes of expert 0's gate_up slice are the LE encoding of
        0x0000, and the third/fourth of 0x8000."""
        rec, raw = self._read(
            "model.language_model.layers.2.mlp.experts.0."
            "gate_up_proj.weight")
        words = struct.unpack("<4H", raw[:8])
        self.assertEqual(words, (0x0000, 0x8000, 0x3F80, 0xBF80))


if __name__ == "__main__":
    unittest.main()
