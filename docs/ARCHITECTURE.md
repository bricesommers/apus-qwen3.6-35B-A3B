# apus-qwen36-35b-a3b — Architecture

Local inference engine for **Qwen/Qwen3.6-35B-A3B** (35B-total / 3B-active
hybrid-linear MoE, `model_type: qwen3_5_moe`, pure BF16, 66.97 GiB of
weights) on consumer hardware, streaming routed experts from disk through
a bounded RAM cache hierarchy. Adapter of the **apus-ling** engine
(Ling-3.0-flash-bf16 repo at `~/Desktop/AI-PROJECTS/Apus-Ling-3.0-Flash-bf16`,
commit `1a9b14f`), itself a port of the **apus** engine
(DeepSeek-V4-Flash), descended from [colibri](https://github.com/JustVugg/colibri)
(Apache-2.0).

**Status: M0–M13 complete; real model validated (M11, 2026-08-30).**
The standing real-model golden and all measured numbers are pinned in
`tests/m11/`; the normative numerics contract is `docs/M4-CONTRACT.md`;
the session log is `docs/STATUS.md`.

---

## 1. Goals, non-goals, hard invariants

Goals:
- Run Qwen3.6-35B-A3B (text-only) locally with an OpenAI-compatible
  server and a terminal chat client.
- C11 core, no dependencies beyond libc/libm/pthreads (macOS additionally
  links Accelerate.framework for the M≥128 prefill dispatch).
- Insufficient fast memory costs **speed only, never quality**.

Hard invariants (any violation requires explicit user approval):
- Never silently change numeric precision of any tensor or computation.
  Numerics follow `docs/M4-CONTRACT.md`. Scalar kernels are the semantic
  anchors; SIMD paths are bitwise-equal to them or belong to a
  user-approved, measured, bounded reorder class.
- Never change router semantics (fp32 softmax, top-8, renormalize — no
  bias, no sigmoid, no scaling, no group-limit).
- Expert weights stay BF16 exactly as shipped — the converter copies
  bytes, never requantizes or transcodes.
- Recurrent GDN is the canonical semantics; HF's chunk path is a
  documented tolerance class, never the golden source.

Non-goals: training/fine-tuning, multi-user high-throughput serving,
vision (the `model.visual.*` tower is stripped at conversion), native
Windows until M13.

## 2. Target hardware profile

- Primary machine: **MacBook Pro, Apple M1 Pro, 32 GB unified memory**
  (ARM NEON; persistent pthread pool; deterministic partitioning —
  thread-count independence is a bitwise gate everywhere threading
  exists).
- **Linux/x86_64 (M12)**: AVX2 kernels (`c/x86.h`) runtime-dispatched,
  bitwise == the scalar anchors. Docker harness
  `tools/docker/test-linux.sh`.
- **Windows (M13)**: shims + CI ported from the apus M15 engine
  (`../Apus`, `c/compat.h`) across the adapter seam — the engine-debt
  item flagged in Phase 0 (the Ling base has no Windows port).
- GPU: opt-in Metal backend (M10 re-point) — FP32 shaders replicating the
  CPU kernels' exact rounding sequence (bitwise offload). Base verdict:
  parity (expert-I/O-bound) → default OFF.
- I/O: `F_NOCACHE` pread + generation-tagged pthread I/O pool on macOS;
  cached pread + `posix_fadvise(DONTNEED)` on Linux.
- Storage: internal NVMe; the tiering design assumes multi-GB/s at
  ~6 MiB random reads (the Qwen expert slab size, §6).

## 3. Qwen3.6-35B-A3B — confirmed architecture reference

Sources: HF repo `config.json` / model card / chat template, and the
normative `reference/modeling_qwen3_5_moe.py` (transformers `main` —
absent from the v4.57.1 tag config.json names; a local copy is pinned).
Local copies in `reference/`. **Every numerics detail is pinned in
`docs/M4-CONTRACT.md`; this section is the map, not the contract.**

### 3.1 Top level

| Field | Value |
|---|---|
| Layers | 40 (+1 MTP layer, deferred to M8) |
| Hidden size | 2048 |
| Vocab | 248,320 padded (248,044 BPE + 26 added + pad) |
| Attention | 30 Gated DeltaNet + 10 full attention, 3:1 — full iff `(idx+1)%4==0` (layers 3,7,…,39) |
| GDN heads | 16 K heads / 32 V heads, head_dim 128, conv k=4 |
| Full-attn heads | 16 q / 2 KV, head_dim 256, partial RoPE 64 dims θ=1e7 |
| Routed experts | 256 per layer, ALL 40 layers = 10,240 total |
| Shared experts | 1 per layer (intermediate 512), sigmoid-gated |
| Top-k | 8, fp32 softmax, renormalized |
| Dense layers | none |
| Context | 262,144 native (YaRN-extensible), mrope = identity for text |
| Residual | plain ×1 pre-norm |

### 3.2 Router (all FP32 selection math)

logits = W_gate·x (BF16 matmul) → fp32 softmax → top-8 → renormalize
to sum 1 → BF16 weights. No bias, no sigmoid, no scaling, no
group-limit. Tie-break pinned lowest-index-wins (oracle rule; goldens
avoid exact ties). Aux loss is training-only.

### 3.3 Gated DeltaNet (30 layers)

Linear attention with per-head scalar decay and a delta-rule recurrence;
no RoPE, no KV cache — per-layer FP32 state S [32,128,128] (~2 MiB) plus
one fused k=4 conv state [8192,4]. Per token: in_proj_qkv → [q 2048 |
k 2048 | v 4096]; fused depthwise causal conv1d(k=4)+SiLU; β =
sigmoid(b) **in BF16**; decay g = −exp(A_log)·softplus(a+dt_bias) per
V-head scalar (FP32, **no lower bound** — unlike Ling KDA's per-channel
bounded sigmoid); 16 K heads repeat_interleave ×2 → 32; q,k L2-normed
(FP32, eps 1e-6); q ×= 128^-0.5; recurrence `S ← S·exp(g) + k⊗(β(v−kᵀS))`,
o = qᵀS (all FP32); per-head RMSNormGated (direct weight, silu(z) gate);
out_proj. The engine implements the recurrent form in both prefill and
decode — bitwise identical by construction; HF's chunk path is a
tolerance class.

### 3.4 Full attention (10 layers)

GQA: q_proj 2048→8192 packs per-head [q(256) | gate(256)]; k/v_proj
→512; per-head (1+w) RMSNorm on q/k BEFORE RoPE; partial RoPE on the
first 64 dims (GPT-NeoX rotate_half, θ=1e7, cos/sin cast BF16); scale
256^-0.5; eager attention, FP32 softmax, KV repeat ×8; **sigmoid output
gate** (second half of q_proj) applied after attention, before o_proj
4096→2048. KV cache: 2 heads × 256 dims per token per full-attn layer.

### 3.5 MTP layer (M8, deferred)

Not implemented by HF. From checkpoint names + vLLM: pre_fc_norms on
emb(token) and h_t → fc [4096→2048] → one FULL-ATTENTION decoder layer +
full 256-expert MoE → norm → shared lm_head + shared embeddings.
Does not affect main-model logits.

### 3.6 Weight formats and on-disk layout

Checkpoint: 26 safetensors shards, 1,045 tensors, **71,903,645,408 B =
66.97 GiB, 100% BF16**. No quantization anywhere. Text-only scope strips
333 `model.visual.*` tensors at conversion.

- **Routed experts — BF16 fused tensors**: `gate_up_proj [256,1024,2048]`
  + `down_proj [256,2048,512]` per layer. Per expert: 1024·2048·2 B +
  2048·512·2 B = **6,291,456 B (6.0 MiB)**. Coalesced 2-tensor slabs (§6).
  Routed total ≈ 60 GiB.
- **Dense** (BF16): embeddings/head [248320,2048] ×2 (~1.9 GiB), GDN
  projections (~2.3 GiB), full-attn projections, routers + shared
  experts, norms. Resident set ≈ **5.5 GiB** estimated.
- Norm weights are the **(1+w) zero-init variant** (checkpoint stores
  gain−1) everywhere except the GDN output norm (direct weight).

### 3.7 Tokenizer & message format

- Qwen2Tokenizer: GPT-2-style byte-level BPE, 248,044 vocab + 247,587
  merges, no byte fallback; **NFC normalizer** (same family as the Ling
  base — the C table-probing pipeline carries over); Qwen2-regex Split +
  ByteLevel pre-tokenizer; no auto BOS/EOS.
- Message format: `reference/chat_template.jinja` is the spec. ChatML
  (`<|im_start|>system|user|assistant\n…<|im_end|>`), thinking on by
  default (generation prompt `<|im_start|>assistant\n<think>\n`),
  `enable_thinking` / `preserve_thinking` kwargs, Qwen XML tool calls
  (`<tool_call><function=name><parameter=k>v</parameter></function>
  </tool_call>`), tool results in user-role `<tool_response>`. EOS =
  `<|im_end|>` (248046) or `<|endoftext|>` (248044). Generation
  defaults: temp 1.0, top_p 0.95, top_k 20 (+ presence_penalty 1.5
  recommended for thinking mode — sampler extension, user gate at M7).

## 4. apus-ling base: what carries over vs what is rewritten

Carried over unchanged (model-agnostic, proven):
- `st.h` safetensors shard index + pread (hardened parser).
- The expert store (`cache.h`) and pilot (`pilot.h`) — geometry changes
  only (6 MiB 2-tensor slabs, 256×40 experts).
- `bf16.h` / `blas.h` / `x86.h` kernels, `pool.h`, `compat.h`, `json.h`,
  `uni_tables.h` / `uni_nfc.h` (Unicode version re-verified at M2).
- The converter/downloader state machines, serving pattern
  (`apus-qwen serve` NDJSON stdio + stdlib gateway), `tools/chat.py`,
  the verification methodology, the milestone/test discipline.

Rewritten for Qwen (the actual adapter work):
- `c/kda.h` → `c/gdn.h` (M4): scalar softplus decay, fused conv, 16K/32V
  repeat_interleave, BF16 beta, silu-gated output norm. The delta-rule
  recurrence, conv mechanics, and FP32-state machinery carry over.
- `c/attn.h` (M4): gated MLA → GQA with [q|gate] packing, (1+w) qk-norm,
  partial RoPE-64 NeoX, sigmoid output gate.
- `c/moe.h` (M4): sigmoid/group router → fp32-softmax top-8 renorm;
  SwiGLU clamps dropped; sigmoid-gated shared expert.
- `c/model.h` / `c/layer.h` (M4/M5): qwen3_5_moe config schema, 3:1
  layer typing, hybrid state management.
- `c/tok.h` tables + `c/encoding.h` (M2): Qwen BPE + ChatML.
- `c/mtp.h` (M8): shared-embedding MTP with fc [4096→2048].
- `tools/oracle.py` (M0): rewritten from the HF reference, recurrent
  path only.

## 5. File layout

```
Apus-Qwen3.6-35B-A3B/
├── Makefile                     # the only build system (clang/gcc, single-TU)
├── AGENTS.md                    # agent guidance (gates, conventions)
├── HISTORY.md                   # inherited base session narrative
├── docs/
│   ├── ARCHITECTURE.md          # this file
│   ├── STATUS.md                # always-current technical log
│   ├── USAGE.md                 # terminal usage + env knobs (binary names current)
│   └── M4-CONTRACT.md           # THE numerics contract (normative)
├── reference/                   # Qwen spec files + pinned HF modeling source
│   └── ling/                    # inherited Ling references (KDA diff source)
├── c/                           # the engine: C11 stb-style single-TU headers
│   ├── apus-qwen.c              # CLI/driver (run/serve) → bin/apus-qwen
│   └── (per-header roles as in the base; see AGENTS.md)
├── tools/
│   ├── convert.py download.py   # HF shards → apus container (M1 retarget)
│   ├── oracle.py                # numpy golden generator (f32 target / f64 truth)
│   ├── server.py chat.py        # OpenAI gateway / terminal REPL
│   └── docker/                  # Linux test harness
├── tests/                       # one dir per milestone, each + README
└── weights/                     # the converted container (local, gitignored)
```

## 6. Weight container format

Same design as the base: **ordinary safetensors shards + a manifest**,
expert bytes byte-identical to the HF checkpoint.

- Main shards (`apus-qwen-NNNNN.safetensors`, ~5 GiB target) + separate
  MTP shard group so the speculative set lazy-loads only with `--spec`.
- **Coalesced per-expert layout**: each expert's {gate_up_proj slice,
  down_proj slice} contiguous → a cache miss is one 6,291,456 B pread
  into one slab, zero-copy views. The HF checkpoint stores experts as
  fused [256,…] tensors; the converter slices per expert and coalesces —
  bytes preserved, order rearranged (the same rearrangement class the
  base's 3-tensor coalescing used, byte-verified).
- Manifest `apus.index.json` (format version, config hash, tensor map
  with absolute offsets, per-expert slab records).
- The (1+w) RMSNorm convention is applied at RUNTIME; stored bytes stay
  the checkpoint's.
- Resumability and verify-then-delete download discipline as in the base
  (tests/m1 covers crash cases on synthetic fixtures).

## 7. Tiering design

Per-token expert demand (decode, no cache hits): 40 MoE layers × 8 =
320 routed experts × 6.0 MiB ≈ **1.9 GiB/token** worst-case cold reads —
about half the Ling regime (3.6 GiB), same design center. **Measured on
the real model (M11)**: 7320 preads × 6 MiB over 39 tokens ≈ **1.10
GiB/token** on the golden command (pilot hits + prefill batch-union
halve the cold bound), pilot recall 79.6% at K=8, zero demand loads.

Budget classes within the 32 GB pool (tunable, `APUS_*` env):
- Resident dense set: **~5.5 GiB** (estimated; M1 measures). Measured
  total peak RSS on the golden run: **8.29 GiB** (dense + cache + pins
  + state).
- GDN recurrent state: 30 layers × 2 MiB FP32 ≈ 60 MiB; conv states
  negligible; full-attn KV: 10 layers × 2×256×2 B/token ≈ 10 KiB/token.
- **Expert cache + pins: default 4 GiB + 0.5 GiB** ≈ ~680 experts of
  10,240.
- RSS guard `APUS_RSS_GUARD_MB 26624` (verified firing on the real
  model at 8000: 1980 drops, stream bitwise unchanged).
- macOS + working-set headroom.

Cache policy as in the base: per-layer LRU + end-of-block promotion,
LFRU pins + hysteresis, demand-boost I/O queue, batch-union in prefill.
**The dominant gate: eager vs store at ANY cache size, with or without
the pilot, produces bitwise-identical tokens and logits.** Pilot K
re-measured on the real model (M11, 64-token window): recall 67.6%@6 /
81.3%@8 / 92.2%@12; decode wall 23.2s@6 / 25.3s@8 / 32.8s@12 — **K=8
stays the default** (K=6's −8% wall isn't worth −13.7pt of recall
margin; K=12 trades bytes for waits at a negative rate).

## 8. Forward pass structure

Prefill and decode share one per-token body (bitwise identical by
construction; the GDN recurrent form makes this exact, unlike HF's
chunk prefill). Per layer: input_layernorm → attention (GDN recurrence
or full attention, per §3.3/§3.4) → +res → post_attention_layernorm →
MoE (router §3.2 → experts → weighted combine + sigmoid-gated shared
expert) → +res. Final RMSNorm → lm_head → logits. MTP per §3.5 when
implemented (--spec, M8).

The tiered forward (`c/cache.h`) mirrors the eager one (`c/layer.h`)
over the same public kernels; the bitwise invariance gates catch drift.

## 9. Kernel architecture (the numerics discipline in code)

Unchanged from the base — the tiers in `c/bf16.h` (+ `blas.h`, `x86.h`,
`backend_metal.mm`): scalar anchor (sequential ascending-k, one FP32 mul
+ one FP32 add per element, RNE BF16 out, `-ffp-contract=off` pinned) →
bitwise NEON/mt → user-approved bounded reorder dispatch (M9b ILP GEMV /
Accelerate BLAS, err ≤ 1e-4 vs FP64) → AVX2 (bitwise == scalar) → Metal
(bitwise offload). New shapes re-gated at M3; the M9b reorder approval
carries over only after re-measurement on Qwen shapes (M9).

Per-op kernels (gdn/attn/moe) follow the same rule: scalar semantics
first, vectorization only across independent units, no FMA contraction.

## 10. Serving

The engine opens no sockets. `apus-qwen serve` speaks NDJSON stdio;
`tools/server.py` (stdlib only): `/v1/chat/completions` (+SSE,
`reasoning_content`), `/v1/completions`, `/v1/models`, `/health`,
`/debug/encode`; optional `APUS_API_KEY`; thinking on by default;
Qwen XML tool calls ↔ OpenAI both directions; `preserve_thinking`
kwarg passthrough (M7). `tools/chat.py`: terminal REPL.

## 11. Verification methodology

Unchanged from the base:
- **Golden-I/O**: `tools/oracle.py` written from the HF reference, never
  from the C. Two modes: f32 (C target, BF16 RNE at contract points) and
  f64 (truth); C must land inside the f32-vs-f64 envelope per stage.
- **Bitwise gates**: tokenizer vs HF byte-exact probes, widen/narrow
  exhaustive, prefill == decode, eager == tiered == pilot-on, spec ==
  non-spec, thread-count independence (T=1/4/8 digests).
- **Bounded classes**: measured err/esc vs FP64, recorded per milestone
  README; token flips require near-tie margin evidence.
- Negative results documented with numbers, never deleted.
- UBSan + `leaks` (ASan unusable on the dev Mac).

## 12. Milestones (plan — Phase B, 2026-08-29; ALL COMPLETE)

- **M0** — oracle rewrite + docs (this file, M4-CONTRACT) + binary
  rename (`apus-ling` → `apus-qwen`).
- **M1** — converter + downloader retarget (fused slabs, strip vision,
  MTP shard group).
- **M2** — tokenizer + ChatML encoding (byte-exact probes, jinja goldens).
- **M3** — BF16 kernel battery on Qwen shapes.
- **M4** — op kernels: `gdn.h`, `attn.h`, `moe.h` (the semantic core).
- **M5** — full model forward on synthetic fixtures (prefill == decode,
  eager == tiered).
- **M6** — store/pilot retarget (6 MiB slabs; pilot K re-measured).
- **M7** — serving (ChatML, thinking/preserve, tool calls).
- **M8** — MTP speculative decoding (exact; measured ~1.9× slower on the
  real model — OFF by default).
- **M9** — performance campaign (reorder classes re-approved on Qwen
  shapes).
- **M10** — Metal backend re-point (opt-in; real-model parity, bitwise).
- **M11** — real-model validation DONE (2026-08-30): standing golden
  (`tests/m11/golden.txt` + `check_golden.sh`), determinism across
  threads/cache/RSS-guard, pilot-K curve, spec/Metal/BLAS verdicts,
  per-token traffic 1.10 GiB/token. External hosted-reference
  validation stays OPEN (needs a user API key).
- **M12** — Linux/x86_64 battery.
- **M13** — Windows port + tri-platform CI (apus M15 shims across the
  seam).

## 13. Open items / future work

- External (R1-style) validation of the real model against a hosted
  reference — needs a user API key; open since M11.
- presence_penalty in the sampler (recommended Qwen defaults) — user
  gate at M7.
- vLLM/SGLang cross-check of the numerics contract (the base reconciled
  HF vs vLLM vs fla; here only HF exists so far) — M4 task.
- KV/state reuse across chat turns (the chunk-invariance property makes
  it safe); chunked GDN for long-prefill speed would be a NEW reorder
  class requiring a fresh user gate.
