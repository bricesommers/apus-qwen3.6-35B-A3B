# tests/m8 — MTP speculative decoding

Milestone M8: the Qwen3.6-35B-A3B MTP block (`mtp.*`, contract §7) as a
draft proposer, plus an exact draft/verify decode loop whose emitted
token stream is **bitwise identical** to non-speculative decoding for
the same seed, greedy and sampled. Machinery adapted from the Ling
base's M8 engine (same accept rule, same rollback design).

Run:

```
make test-m8       # 25 checks x APUS_THREADS=1/4/8 (digests diffed)
make ubsan-m8      # same under -fsanitize=undefined
make golden-m8     # regenerate tests/m8/fixtures deterministically
bin/apus-qwen run --model DIR --spec [--spec-k K] ...   # real-model spec
```

`--spec` (also `APUS_SPEC=1`) defaults **OFF** — see the honest speed
verdict below. `--spec-k K` (also `APUS_SPEC_K`, default 2) is the
verify-batch size: K=2 means one speculative token per batch.

## The MTP block (loader + forward, c/mtp.h)

HF does NOT implement MTP (`_keys_to_ignore_on_load_unexpected`); the
normative structure comes from the real checkpoint naming
(`reference/model.safetensors.index.json`) + vLLM
`qwen3_next_mtp.py`. Tensor namespace `mtp.*` in the M1 container's
`apus-qwen-mtp-*` shard group, resolved lazily by `apus_mtp_load` only
when --spec (a default load never touches the group). Wiring
(contract §7):

```
e  = rmsnorm(emb[token], pre_fc_norm_embedding)   # (1+w), §2
hh = rmsnorm(h, pre_fc_norm_hidden)               # (1+w), §2 — SINGLE
     # norm: h is the PRE-final-norm main hidden (apus_model_forward_h);
     # vLLM applies pre_fc_norm_hidden directly to the incoming hidden,
     # no main-model final norm inside the block
z  = fc(cat[e, hh])                               # 2H->H
residual z -> input_layernorm -> gated GQA (§4, layer_type
     full_attention) -> +res -> post_attention_layernorm
     -> full MoE (§5) -> +res -> mtp.norm ((1+w))
     -> SHARED lm_head + SHARED embeddings
     (mtp_use_dedicated_embeddings=false)
```

RoPE positions UNSHIFTED (the layer state's own pos — the Ling base's
convention; HF has no MTP golden to pin the shift, and the position
choice is draft-quality-only: the emitted stream never depends on MTP
numerics). The layer body after the fc glue IS a standard FULL decoder
layer on input z, so the forward reuses `apus_layer_forward_hot`
(eager) / `apus_store_layer_forward` (tiered) unchanged. Draft chaining
(depth-1 approximation, draft-quality only): the next pair's h is
`mtp.norm(block_out)`.

**Expert store (c/cache.h):** the converter numbers mtp slab records
`layer = num_hidden_layers + K` (= 40 real; = 2 in the mini fixture)
with tensor names `mtp.layers.{K}.mlp.experts.{E}.*`. The store maps
both directions: `ApusStoreCfg.n_main_layers` (new; 0 = n_layers)
splits the layer space — `apus_parse_expert_name` parses the mtp
prefix into store layer `n_main + K`, and slab derivation builds the
tensor name from the mapped prefix. Default stores (n_main == n_layers)
skip the mtp records by range, exactly as at M6. The pilot does NOT
predict the MTP layer's router (gap list, as in the Ling base).

## The accept rule and step shape (the hard invariant)

A draft is accepted iff it equals the main model's OWN pick at that
position — argmax for greedy, the main model's own `apus_sample` draw
for sampled. One RNG uniform per emitted token in position order ==
the non-spec stream; drafts are always argmax and consume no RNG.

Step top: main state fed ≤ q−1; held token x_q (sampled from a valid
main row, not yet emitted); drafts z_1..z_{K-1} chained; the snapshot's
MTP part is the CLEAN point (pairs through the seed pair's position
q−1, saved BEFORE the chain steps — the chain is draft-quality dirty).

1. Verify batch `[x_q, z_1..z_{K-1}]` at positions q..q+K−1 → rows
   R[0..K−1] + hiddens H[0..K−1], ONE batched main forward from the
   carried decode state (bitwise "as if decoded one-by-one" — the M4
   per-token-body construction; no new code path).
2. Walk: emit x_q; for j=1..K−1 sample y from R[j−1]; accept z_j iff
   z_j == y; stop at the first mismatch (replacement held). Full match:
   bonus from R[K−1] held; main batch state kept.
3. Partial: restore the MAIN snapshot, re-feed batch[0..matched] in one
   batched call (bitwise the sequential state). BOTH cases: restore the
   MTP clean point, replay the true pairs at q..q+matched in one batched
   MTP call (pair hiddens from the verify batch; the last replay pair IS
   the next seed pair), re-snapshot, chain the remaining drafts.
4. Exit alignment: on a partial last step the verify batch fed rejected
   drafts too, so at exit the main state is restored + re-fed through
   exactly the emitted prefix — `apus_spec_run` returns with the model
   state fed exactly the emitted tokens (the state-digest gate).

**Snapshot/rollback for the hybrid state (the GDN complication):** GDN
layers carry a full copy of the conv state (last 3 pre-conv inputs per
channel) + the fp32 recurrent state S; FULL-attention layers (and the
MTP layer itself) need only a position rewind — the KV caches are
append-only and rows beyond pos are dead, never read. Rejected drafts
leave no trace in any state kind (the state-digest gate proves it).
On the real model that's ~70 MB of memcpys per step (30 GDN layers ×
2.3 MB), negligible against disk-bound decode.

## What the suite checks (25 checks, test_mtp.c)

1. `mtp_prefill` golden: C MTP forward vs oracle f32, rel 0.00465
   inside the f32-vs-f64 envelope 0.00465; 0/12 argmax flips (near-tie
   policy: excused iff golden gap ≤ 0.5).
2. `mtp_chain` golden: 3-step draft chain == oracle drafts bitwise
   ([108, 115, 42]).
3. **Equivalence (the gate):** spec K=2/3/4 vs non-spec, greedy AND
   sampled (temp 0.6/top_k 20/top_p 0.95, seed 42) — 24 emitted tokens
   BITWISE, and the post-run model-state digest (pos, GDN conv+S, GQA
   cache live rows) BITWISE vs non-spec — rejected drafts leave no
   trace.
4. **Forced draft patterns** (override): truth (16/16 accepted, 8/8
   full, 0 re-feeds), garbage (0/48, 23 re-feeds), mixed (12/24 — every
   draft-1 accepted, every draft-2 rejected) — streams bitwise, stats
   match the pattern exactly.
5. **Tiered spec** (store serves the mtp layer-2 slabs through the
   mtp.layers.0 prefix mapping) == eager, bitwise.
6. APUS_THREADS=1/4/8 digest diff (Makefile); UBSan clean (digest
   identical).

Fixture acceptance with random weights is ~0 (vocab 256) — the accept
paths are exercised deterministically via the forced patterns.

Fixture: the m6a container (m5 mini model + v2 manifest) plus a real
mtp shard group `apus-qwen-mtp-00001.safetensors` (dense mtp tensors +
the per-expert 2-member slab slices, slab records numbered layer 2 =
num_hidden_layers), mtp weights from `oracle.gen_mtp_weights` (seed
20260829), goldens from `oracle.mtp_forward`/`mtp_chain` — all consumed
from container bytes only.

CLI smoke (fixture model): non-spec and `--spec [--spec-k K]`, eager
and `--tiered`, all emit the identical stream
`186 26 88 97 193 139 54 229` (the m5 cli_greedy golden).

## Honest speed verdict (inherited from the Ling base's M8 measurement)

The Ling base measured spec on the REAL Ling model (tiered, greedy,
24 tokens): **~1.8× SLOWER** in the disk-bound regime (K=2: 0.23 tok/s
vs 0.41; K=3: 0.24 tok/s), despite 1.5–1.85 tok/batch and 42–50%
acceptance. Spec amortizes compute, not disk — and every draft step
pays a full extra layer PLUS the shared lm_head GEMV, while the verify
batch pulls more expert slabs per emitted token. **Spec stays OFF by
default.** The Qwen real-model speed verdict is an M9/M11 measurement
(same machinery, same expectation; the draft cost structure is
identical — one extra full layer + lm_head per draft row).

## Remaining gaps

- `serve` mode has no spec path (run-mode only), like the Ling base.
- The pilot does not predict the MTP layer's router (the mtp layer
  resolves on demand).
- Draft chaining uses the MTP block's own mtp.norm output as the next
  pair's h (the depth-1 approximation); `mtp_num_hidden_layers` > 1 is
  not supported by the loader (the real config has 1).
- The RoPE position convention (unshifted, the layer's own pos) is the
  Ling base's; HF ships no MTP golden. Draft-quality only — the
  bitwise gates are unaffected.

## Files

- `c/mtp.h` — MTP loader (real mtp.* names), batched MTP forward,
  ApusSnap snapshot/rollback, the `apus_spec_run` draft/verify engine,
  forced-draft hook.
- `c/model.h` — `apus_model_forward_h` (pre-final-norm hidden out,
  M5), `mtp_num_hidden_layers` parse.
- `c/cache.h` — `n_main_layers` + the `mtp.layers.{K}` prefix mapping
  (parse + slab-name derivation).
- `c/apus-qwen.c` — `--spec`/`--spec-k`, `APUS_SPEC`/`APUS_SPEC_K`,
  acceptance/tok-per-batch stats; store sized `n_layers + n_mtp` with
  `n_main_layers = n_layers` when spec is on.
- `tools/oracle.py` — the contract-§7 MTP oracle (`gen_mtp_weights`,
  `mtp_layer_weights`, `mtp_forward`, `mtp_chain`).
- `tests/m8/gen_fixtures.py` — container (m6a + mtp shard group, 48
  slabs) + goldens; `test_mtp.c`.
