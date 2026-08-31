# tests/m9c — batched-prefill restoration (M-independent-bitwise)

Milestone M9 (this adapter): the Ling M6c/M9c batched prefill is
**restored** for the Qwen layer body — this is the one place M9 allows
new hot-path structure, and it is kept STRICTLY inside the
M-independent-bitwise class: every batched linear runs at M=T through
`apus_bf16_gemm_hot` (the ILP GEMM whose M-independence — every GEMM
row bitwise the GEMV row — is gated in tests/m9b). **NO
`gemm_fast`/BLAS anywhere in the batched path**: the Ling M9c phase-B
FFN consumed the BLAS class at M≥128 (leaving the pinned realization
there); the Qwen restoration deliberately does not, so **prefill ==
decode stays BITWISE at every T on every platform** — no M≥128 escape,
no adjudication, no margin policy touched.

Run:

```
make test-m9c      # 97 checks x APUS_THREADS=1/4/8 (digests diffed)
make ubsan-m9c     # same under -fsanitize=undefined
make bench-m9c     # ILP GEMV variants at the Qwen shapes (informational)
make profile-m9c   # phase-A/phase-B component profile at real dims
```

## What batches (and what provably cannot change a bit)

**Phase A (chunked at `APUS_PREFILL_ATTN_CHUNK` = 256; the chunk only
bounds the scratch arena — every per-output bit is chunk-size
independent):**

- GDN layers: `w_qkv`/`w_z`/`w_b`/`w_a` projections at M=tc
  (gemm_hot, M-independent-bitwise); ONE `apus_gdn_conv1d` over the
  chunk (contract (a): bitwise the step loop, gated at m4a); the
  sequential state parts (beta/decay/l2norm/repeat×2/delta step/onorm)
  stay per token, ascending t; `w_out` at M=tc.
- FULL layers: `wq`/`wk`/`wv` at M=tc; per-token (1+w) qk-norms, RoPE
  at each token's own position, gate split, cache append (per-row,
  ascending t); ONE `apus_attn_gqa_mt` over the chunk after the appends
  (each (t,h) unit is the decode unit's identical body — the m4a
  "decode == full-recompute row" identity); per-token sigmoid outgate;
  `wo` at M=tc.

**Phase B (full T):** router per token (unchanged); shared expert
`sh_g`/`sh_u`/`sh_d`/`sh_gate` at M=T; each UNIQUE routed expert's
fused `gate_up`/`down` ONCE at M=count over the gathered rows
(gather/scatter are exact copies; silu acts are per-row; the per-token
combine keeps the ascending slot order; sigmoid shared-gate and
residuals are per-element). The tiered mirror (`c/cache.h`
`apus_store_prefill`) resolves each unique expert once — batch-union
demand hints first, the no-hook `_cpu` GEMMs on the transient slab
views, one `layer_end` at the end.

The GDN recurrence itself CANNOT batch (state-carrying per token by
definition) — it stays sequential; what batches is everything AROUND
it. A chunked-state GDN form would be a different numerics class
(documented tolerance class in docs/ARCHITECTURE.md §13) and is NOT
taken.

## Bitwise gates (this suite — 97 checks, T=1/4/8 digest)

1. **Layer level** (synthetic real-shaped layers of both kinds):
   `apus_layer_forward_hot` at T=256 and T=300 (crossing the chunk
   boundary) and T=64 vs the sequential `apus_layer_forward` — EVERY
   attention trace (ln1/attn_out/res1/ln2 + GDN
   qkv_conv/beta/gdecay/rec_o/onorm + FULL qf/kf/attno/mgate), the MoE
   traces (rtr_idx/rtr_w/moe_routed/moe_shared/moe_out), the layer
   output, and the layer state (conv state, S, KV caches, pos) BITWISE.
2. **Model level**: eager hot prefill T=64 == sequential decodes
   (logits + layer state) on the m5 fixture; tiered (m6a fixture)
   same.
3. **Tiered res1 stream** (post_attn hook capture): batched ==
   sequential BITWISE at T=64 AND T=256 — on macOS too (the phase-B
   restoration consumes no BLAS class, so the stream never leaves the
   pinned realization; the Ling suite could gate this only off-BLAS).

Gates: 97 checks, 0 failures; digest `30fd1b519a5c9bcc` identical at
APUS_THREADS=1/4/8 AND under UBSan; the model-level digests of every
existing suite are byte-identical (m5 `29a4612f3057cac2`, m6a/m6b
`09fc53aee61f56c1`, m6c `ff7fed2d52a79102`, m8 `cb171f7616055023`) —
the batched path is the default T>1 path and it reproduces the
sequential bits exactly.

**Stress note (the M9 hunt worth recording):** during re-anchoring, the
tiered gates failed ~5% of runs at APUS_THREADS=10 with
batched-vs-sequential mismatches at early tokens of layer 0 — on BOTH
sides. Root cause was NOT the engine: this suite (Ling-inherited) drew
token ids with `% 512` while the m5/m6a fixture vocab is 256, so half
the ids read past the 256-row embed table — heap garbage whose content
the process's own allocation churn occasionally mutates between the two
compared runs (Guard Malloc pinpointed the OOB read at the embed
memcpy, crashing 100% deterministically). Fixed here (`% 256`) and in
tests/m10/test_model.c (the same latent Ling-era line); 100/100 clean
at T=10 after the fix (and the ubsan/-O2 digests agree).

## Measured speed (profile_phasea.c, synthetic REAL Qwen dims, T=512,
threads 8, no I/O — compute structure isolated from the NVMe-bound
tiered wall)

| form | time | note |
|---|---|---|
| A per-token GDN mixer (replica loop) | 0.704 s/layer | the M4 body |
| B layer sequential (`apus_layer_forward`) | 0.795 s/layer | |
| B layer batched (`apus_layer_forward_hot`, M9) | 0.273 s/layer | **2.91×, outputs+state BITWISE** |
| C GDN projections only (4× gemm M=T) | 0.125 s | |
| D sequential parts (conv/decay/l2norm/step/onorm) | 0.085 s | the irreducible serial core |
| E phase-B per-token expert GEMVs (256 experts, top-8) | 0.734 s/layer | the M4 body |
| E phase-B unique-expert batched GEMMs (M9) | 0.134 s/layer | **5.49×, outputs BITWISE** |

Per GDN layer at T=512 the compute drops ~(0.795+0.734) →
(0.273+0.134) ≈ **3.8×** in the compute-bound regime. Honest framing
(inherited from the Ling M9c, whose real-model A/B was a wash): the
tiered real-model prefill is NVMe-bound (~1.9 GiB/token of cold expert
reads — half the Ling regime, so compute is a LARGER share here), and
the batched path's win materializes where prefill is compute-bound:
eager models, warm stores, long prompts, dense routing. The unique-
expert batching also CUTS the I/O-side per-expert resolution count
(each expert's slabs stream once per layer, not once per token-slot).

## bench-m9c (ILP GEMV variants, single-thread rows, best of 7, V0 =
the shipped 8-row kernel; memcmp-bitwise per shape — all variants
bitwise-identical, informational only: c/bf16.h is outside the M9 edit
scope, nothing ships)

| shape | V1 pf | V2 16row | V3 cse | V1+3 |
|---|---|---|---|---|
| 8192×2048 (qkv/q proj) | +16.9% | +9.3% | +16.2% | **+30.8%** |
| 2048×4096 (out/o proj) | +0.8% | -14.9% | -1.2% | +1.7% |
| 4096×2048 (GDN z) | +14.6% | -9.7% | +0.1% | +15.0% |
| 512×2048 (k/v, shared) | -3.9% | -18.7% | -0.0% | -2.6% |
| 2048×512 (down) | +3.8% | -13.7% | +0.0% | +3.8% |
| 1024×2048 (expert gu) | -3.3% | -19.1% | +0.0% | -3.3% |
| 256×2048 (router) | -2.6% | -19.6% | +0.0% | -2.6% |
| 32×2048 (GDN a/b) | +0.0% | -20.0% | +0.0% | +0.0% |
| 248320×2048 (head) | +15.6% | +4.1% | -0.7% | +15.6% |

Recorded, not shipped: the prefetch+CSE combination wins big on the
three largest streams (+15-31%) but loses on the small ones — a
size-conditional dispatch would be a candidate for a FUTURE milestone
(bitwise-neutral by construction, but c/bf16.h is outside the M9 edit
scope).

## Files

- `c/layer.h` — the M9 batched prefill (`apus_layer_attn_batch_{gdn,
  full}`, `apus_layer_moe_batch`, `apus_layer_prefill`,
  `APUS_PREFILL_ATTN_CHUNK`), header contract updated.
- `c/cache.h` — lockstep tiered mirror (`apus_store_attn_batch_*`,
  `apus_store_prefill`).
- `tests/m9c/test_m9c.c` — the bitwise gates (97 checks).
- `tests/m9c/bench_m9c.c` — ILP GEMV variants at the Qwen shapes.
- `tests/m9c/profile_phasea.c` — the phase-A/phase-B component profile.

## Honest remainder

- Phase-B scratch is proportional to T (the unique-expert gather
  buffers are T·top_k·H): at the real dims and a 256-chunk world this
  is tens of MB; very long prompts (T in the tens of thousands) grow it
  linearly — the Ling design had the same profile; chunking phase B is
  bitwise-neutral by construction and available if a real workload
  needs it.
- The ILP GEMM's W re-streaming (per 4-row activation group) is the
  prefill-side analog of the GEMV row-group squeeze — unmeasured here
  (out of the M9 edit scope; noted for a compute-bound future).
- The sequential GDN recurrence is now the largest non-batched
  attention term in isolation (profile arm D) — a chunked-state form
  would be a DIFFERENT numerics class (needs a fresh user gate).
- Everything on the real tiered workload remains store/IO-bound;
  compute levers move isolated-component numbers and the eager/warm
  cases, not the cold NVMe wall.
