# tests/m6a — expert-store tiering (experts on NVMe, bounded RAM cache)

Milestone M6a: routed experts live on disk and are demand-loaded through
a bounded, pinned, LFRU-managed RAM cache — the Apus tiering design
re-anchored to the Qwen3.6 apus-qwen container (M1 **FORMAT_VERSION 2**:
2-member fused slabs). **Hard rule verified: memory pressure costs speed
only, never quality** — token streams and logits are bitwise identical
from "everything fits" down to "2 slots/layer under a 1-byte RSS
budget".

Run:

```
make test-m6a    # 65 store checks + 18 invariance checks, exit 0 iff pass
make ubsan-m6a   # same under -fsanitize=undefined
make bench-m6a   # slab-read bench + hit-rate/tok/s curve (informational)
make golden-m6a  # regenerate tests/m6a/fixtures deterministically
```

## Files

- `c/compat.h` — platform shims (RSS via mach task_info, F_NOCACHE fd
  marking, posix_fadvise shim, `APUS_*` env parsing).
- `c/cache.h` — the expert store + the tiered layer forward (below).
- `c/model.h` — `apus_model_load_ex(m, dir, max_seq, tiered, ...)`;
  tiered leaves `lw.exp_*` NULL; `apus_model_attach_store` +
  per-token dispatch of MoE layers through `apus_store_layer_forward`.
  `APUS_MODEL_IMPLEMENTATION` auto-pulls the cache/compat
  implementations so existing single-TU users link unchanged.
- `c/apus-qwen.c` — `--tiered` / `APUS_TIERED=1`: opens the store (env
  knobs below), attaches it, saves usage + prints store stats at exit.
- `tests/m6a/gen_fixtures.py` — copies the M5 fixture model (already
  the converter's v2 layout: per-expert `gate_up_proj.weight` [2I, H]
  then `down_proj.weight` [H, I], contiguous) and generates
  `apus.index.json` with `format_version: 2` + `expert_slabs` records
  ({layer:int, expert, shard, offset, nbytes}); v2 slab invariants
  (2 members, gate_up first, contiguity, uniform slab bytes, [2I,H]/
  [H,I] geometry) and payload byte-identity asserted in Python. Also
  writes `fixtures/model_v1` (a FORMAT_VERSION 1 manifest) for the
  rejection gate. Fixture geometry: 2 layers, MoE on BOTH, 16 experts,
  slab **24,576 B** (gate_up [64,128] + down [128,32] BF16).

## v2 slab support (c/cache.h, the M6 change)

**Addressing.** (layer, eid) → slab record: from `apus.index.json`
`expert_slabs` when present (member offsets then verified against the
shard headers), else derived from the headers (the 2 member slices
`model.language_model.layers.{L}.mlp.experts.{E}.{gate_up_proj,
down_proj}.weight` must share one shard and tile a contiguous byte
range — verified, not assumed; open fails loudly). A miss is **one
pread** into a 4 KiB-aligned buffer; `ApusBf16ExpertW` views
(`gate_up` [2I,H] + `down` [H,I] + dims) are zero-copy pointers into
it. The `gate_up` slice IS the fused [gate|up] layout the eager path's
`exp_gu` array holds — same bytes, same GEMV, hence bitwise.

**Format gate.** A PRESENT `apus.index.json` must have
`format_version == 2` — anything else (a v1 3-member-slab container)
fails `apus_store_open` with a clear "reconvert" error, never a silent
misread. A container with NO v2 slab slices at all (e.g. a v1
container without a manifest) fails open likewise: this model has MoE
on every layer, so a 0-MoE scan is a wrong-container error. MTP slab
records (converter layer numbering num_hidden_layers + K) are out of
range for the main store and skipped (the mtp tensor-name prefix is
M8's).

**Policy** (identical to Apus): per-layer LRU slots; misses load into a
per-forward working set, promoted at `layer_end` by swapping with the
coldest slots (empty slots first); batch-union overflow drops the excess
after use (one-shot streamers never flush the cache). Pins: never
evicted, seeded from the usage-history file (merge-with-max, atomic
rewrite), LFRU REPIN with 25%+4 hysteresis. RSS guard: over budget →
coldest LRU payloads freed in place (identity kept; pins/ws/in-flight
untouched; block boundaries only). I/O pool: generation-tagged jobs
(straggler drops payload, waiter re-submits), demand-class boost ahead
of speculative FIFO (`APUS_STORE_BOOST`), slab-buffer freelist
(`APUS_BUF_FREE`). Miss accounting: first consume of a hint-loaded entry
is a miss; in steady state misses == preads == unique (layer,eid) loads.

**Tiered forward.** c/layer.h is frozen (no hooks) and its attention
halves are static, so `apus_store_layer_forward` mirrors the per-token
wiring EXACTLY using only the public kernels (same ops, same order):
attention half → res1 → post_attn hook → ln2 → router → batch-union
demand hints → just-in-time resolves → act/combine → shared expert →
residuals. Same slab bytes + same kernels + same accumulation order ⇒
bitwise the eager path — and the invariance test is exactly that gate.

## Results (this Mac, M1 Pro)

**Quality invariance (THE hard test), `test_invariance`:** greedy
decode, 8-token prompt, 24 steps — eager vs store at 16 slots-per-layer
(all 16 experts fit) vs 8 vs 2 vs 2 slots + 1-byte RSS budget (**128
payload drops, 256 preads, 0 hits**) vs 4 slots + 2 seeded pins vs
synchronous I/O: **token streams identical and all 6,144 logits bitwise
identical in every configuration** (18/18 checks). Digest
`09fc53aee61f56c1` identical at APUS_IO_THREADS=1/4/8 (Makefile diffs).

**Unit tests, `test_store` (65 checks):** slab derivation + manifest
cross-check + one-pread-per-expert (instrumented), view byte-identity vs
direct member reads, **FORMAT_VERSION 1 manifest rejection** (clear
error naming the format), out-of-range layer/expert resolve failures,
hit/miss accounting, working-set promotion with mid-block overflow
(keeps newest 2, drops 2), LRU recency eviction, pin
seeding/first-touch/persistence across reopen, LFRU hysteresis (pin 10:
challenger 16 swaps; pin 20: 28 holds, 29 swaps), RSS guard (drops in
place, pins survive, reload on demand counted as miss), generation-tag
straggler (pre-claim hook; payload dropped, re-read, bytes correct),
concurrent-vs-serial byte identity (12 experts × 2 slices), hint
eviction guard (unconsumed hints never evict warm), demand boost
(completion order via the worker-side pre-claim hook — timing-
independent), usage decay at save (0.5: 100→50, 50→25, 33→16).

**Bench (`bench_m6a`, informational):** fixture slabs are 24 KB and
OS-cache warm, so decode tok/s is compute-bound — the meaningful number
is the hit-rate curve:

| slots/layer | hit rate | preads |
|---|---|---|
| 16 (all) | 87.5% | 32 |
| 8 | 52.3% | 122 |
| 4 | 27.7% | 185 |
| 2 | 13.7% | 221 |

Slab pread on this machine (warm 24 KB fixture slabs): ~2–6 GB/s
cached/F_NOCACHE. Real-model numbers need the real container (6 MiB
slabs, 256 experts × 40 layers ≈ 60 GiB routed); the machinery is
identical and config-driven (n_layers / n_experts from
`ApusStoreCfg`).

## Knobs

| knob | default | meaning |
|---|---|---|
| `APUS_EXPERT_CACHE_MB` | 4096 | LRU budget → slots/layer |
| `APUS_PIN_MB` | 512 | pin budget → pins/layer |
| `APUS_RSS_GUARD_MB` | 26624 | RSS guard threshold |
| `APUS_IO_THREADS` | 4 | I/O pool size; <0 = synchronous |
| `APUS_NOCACHE` | 1 | F_NOCACHE streaming reads |
| `APUS_STORE_BOOST` | 1 | demand-class overtakes speculative |
| `APUS_BUF_FREE` | 64 | slab freelist depth |
| `APUS_USAGE_DECAY` | 1.0 | heat decay at usage save |
| `APUS_TIERED` / `--tiered` | 0 | CLI: experts-on-demand |

All also explicit `ApusStoreCfg` fields (tests use those). Introspection:
`apus_store_stats`, `apus_store_resident_bytes`, `apus_store_debug_layer`,
`apus_store_debug_present/ready`.

## Notes

- Decode on the fixture is compute-bound; the store adds waits only on
  misses (`stats.waits/wait_ns`, `pread_ns` instrumented).
- The tiered forward mirrors c/layer.h per token — when the performance
  milestones (M9) optimize the eager path, the mirror must follow (the
  bitwise invariance gate catches drift).
- Real-model sizing: GDN S state is 32×128×128 fp32 = 2 MB/layer ×30
  GDN layers; full-attn KV caches scale with max_seq (2 kv × 256);
  expert RAM = slots × 6 MiB — the 4096/512 MB defaults target the
  32 GB dev Mac like Apus's.
