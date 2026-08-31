# tests/m1 — converter + downloader (M1)

Dependency: the project venv at `../../.venv` (numpy; safetensors used for
read cross-checks only). No pytest — plain `unittest`. No network, no real
weights.

Run all (from the project root):

```sh
.venv/bin/python -m unittest discover -s tests/m1 -v
```

or individually, e.g.:

```sh
.venv/bin/python -m unittest discover -s tests/m1 -k test_3 -v
```

Target schema: **Qwen3.6-35B-A3B** (`model_type: qwen3_5_moe`), 26 shards,
1,045 tensors, 66.97 GiB, BF16. The schema differences vs the Ling base
that M1 retargets:

- **Fused expert tensors**: routed experts ship as two tensors per layer —
  `mlp.experts.gate_up_proj` [E, 2M, H] and `mlp.experts.down_proj`
  [E, H, M] (no `.weight` suffix, no per-expert tensors). The converter
  SLICES them per expert and coalesces: slab = `gate_up_proj[e]` +
  `down_proj[e]`, contiguous (6,291,456 B for the real model), recorded
  under synthesized per-expert output names
  `<layer prefix>.mlp.experts.<E>.{gate_up_proj,down_proj}.weight`.
- **Cross-shard expert layers**: 17 of the 41 MoE layers have their two
  fused tensors in DIFFERENT input shards. A layer's slabs are assembled
  during the pass of the LATER home shard (deterministic in sorted shard
  order); the downloader holds a source shard until every tensor in it is
  consumed + byte-verified (peak: output + ≤2 source shards).
- **Vision tower stripped**: `model.visual.*` (333 tensors) is skipped,
  counted, and reported in the manifest (`stripped_tensors`) — text-only
  scope.
- **MTP block**: top-level `mtp.*` tensors → `apus-qwen-mtp-*` shard group
  (lazy-load); slab records number `mtp.layers.K` as layer
  `num_hidden_layers + K` (= 40 for the real model).
- **Nested config**: `num_hidden_layers` lives under `text_config`.
- **Manifest format_version = 2**: v1 was the Ling container (3 whole
  tensors per slab, output names == source names); v2 changes slab
  semantics (2 slices cut from fused tensors, synthesized names), so a v1
  consumer must not silently read a v2 container. Top-level manifest keys
  otherwise unchanged.
- Bytes are never requantized/transcoded; the (1+w) RMSNorm convention is
  a RUNTIME concern — the converter is a byte-identical repack with
  rearrangement only.

What each file does:

- `stutil.py` — minimal *manual* safetensors reader/writer (8-byte LE header
  length + JSON header + raw data). The safetensors library is deliberately
  not used for writing: numpy has no BF16 dtype and the fixtures must be raw
  bytes we fully control. (Unchanged from the base; also reused by
  tools/oracle.py.)
- `fixtures.py` — synthetic Qwen3.6-35B-A3B checkpoint at tiny scale
  (H=64, M=16, E=8 experts on ALL layers, num_hidden_layers=4: GDN linear
  layers 0..2, full-attention layer 3, top-level MTP block with
  mtp.layers.0): same naming scheme and dtypes as the real index, BF16
  layout invariants intact (per-expert slab == 2 slices x 2 B = 6,144 B),
  fused expert tensors, 4 vision-tower tensors, 5 input shards with a
  weight_map index — with THREE expert layers deliberately split across
  input shards in both directions (layer 1: down earlier; layer 2: gate_up
  earlier; mtp: down earlier), mimicking the real checkpoint's 17 split
  layers. Payloads are random bytes — sufficient for byte-identity testing.
- `test_1_byte_identity.py` — every output tensor's bytes equal the source
  bytes (dense whole, fused per-expert slice by slice); dtype/shape
  preserved; vision tower absent from the output and counted in the
  manifest; manifest consistent; safetensors-lib parse cross-check.
- `test_2_coalescing.py` — each expert's 2 slice tensors are consecutive in
  the shard header, contiguous+adjacent in the data region, single-shard,
  in the right main/`mtp` output group (`apus-qwen-*` / `apus-qwen-mtp-*`);
  the manifest slab records (layer, expert, shard, offset, nbytes) match
  the actual header offsets; slab bytes equal the source slices even when
  the two fused tensors live in different input shards.
- `test_3_bf16_values.py` — BF16 semantics spot-check. Widen reference
  (uint16 << 16 viewed as float32), hand-computed and boundary cases
  (subnormals, max normal, inf, NaN, -0.0, pi), and a conversion round-trip
  proving known BF16 payloads — including per-expert slices cut from a
  fused tensor — arrive bit-identical and decode to exactly the same FP32
  values.
- `test_4_resume.py` — crash mid-conversion (progress-callback fault
  injection), crash mid expert-slab, crash mid DEFERRED cross-shard layer
  assembly, torn-write tail on the open shard, crash at seal boundary,
  no-op rerun: final output always byte-identical to the uninterrupted run.
- `test_5_index_realism.py` — validates the converter's assumptions against
  the REAL `reference/model.safetensors.index.json`: naming scheme, 26
  shards / 1,045 tensors / 66.97 GiB, fused 2-tensor expert groups on all
  41 MoE layers (40 main + mtp.layers.0), the pinned 17 cross-shard layers,
  333 vision tensors under exactly `model.visual.*`, BF16 shape arithmetic
  from the nested text_config, per-expert 6,291,456 B (6 MiB), full
  attention on layers 3,7,...,39, top-level mtp.* block, size bookkeeping.
  NOTE: the real index carries no shapes/dtypes (name -> shard only);
  shapes come from shard headers at conversion time.
- `test_6_download_driver.py` — offline (local "remote" dir) end-to-end
  download.py run: kill mid-download and mid-conversion, restart, partial
  `.part` resume, hold-until-consumed shard retention (landing zone peaks
  at 2 shards; a shard waiting on a cross-shard expert partner survives
  until the partner's pass), source shards deleted only after
  byte-verification, final container byte-identical to direct conversion.
