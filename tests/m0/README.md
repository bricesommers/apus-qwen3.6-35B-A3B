# tests/m0 — Qwen3.6 oracle smoke self-check (fixture + f32/f64 envelope)

Milestone M0: the normative numpy oracle (`tools/oracle.py`) for
**Qwen3.6-35B-A3B** (`model_type: qwen3_5_moe`, text-only), written from
the HF reference (`reference/modeling_qwen3_5_moe.py`) — never from the
C sources. This test is the oracle's smoke gate: it builds a tiny
synthetic fixture, runs both numeric modes, and checks determinism and
the f32-vs-f64 envelope before any C exists.

Run:

```
.venv/bin/python -m unittest discover -s tests/m0 -v
```

## The synthetic mini-model

`tools/oracle.py` `MINI_CFG` — the real `config.json` text_config
schema with tiny dims, every structural feature preserved: hidden 128,
**2 layers typed by the real `layer_types` rule** (`L0 =
linear_attention` Gated DeltaNet + MoE, `L1 = full_attention` gated
GQA + MoE — MoE on every layer, like the real model), GDN 8 v heads /
4 k heads × 32 (repeat ×2, conv k=4), full attn 4 q / 2 kv heads ×
head_dim 64, partial RoPE 16 dims (factor 0.25, NeoX, θ=1e7), 16
experts top-4, moe_inter 32, shared expert 32, vocab 256, eps 1e-6.
Container = safetensors shard + `model.safetensors.index.json` +
nested-schema `config.json` (tests/m1/stutil.py writer), REAL Qwen3.6
tensor naming (`model.language_model.*`, `lm_head.weight`). Norm
storage convention honored: (1+w) RMSNorms stored as gain−1
(`N(0, 0.1)`), GDN gated output norm direct (`1+N(0, 0.1)`), A_log /
dt_bias fp32. Master seed **20260831** (chosen per the m4b policy so
router top-k margins are clean: f32/f64 selections identical on every
token); ids seeded 20260832. The golden path consumes ONLY weights
read back from the container (the Apus M4c lesson).

## Gates (what CI/M0 requires)

1. **REPLAY** — fresh disk load + run reproduces the stored `.npy`
   goldens exactly (container IO integrity).
2. **WIRING** — the test's manual per-layer chain reproduces
   `oracle.full_forward` logits exactly.
3. **SELECTION** — router f32/f64 top-k selections identical on every
   token (if this fails after an oracle edit, the fixture seed must
   change — m4b policy).
4. **CHUNK INVARIANCE** — one-shot T=12 vs prefill 7 + 5 decode steps:
   f64 ≤ 1e-12 (measured 0), f32 zero bf16 code flips (measured 0).
   The oracle is recurrent-path-only (`torch_recurrent_gated_delta_rule`
   is the contract; the HF chunk path is NOT bitwise equal and is not
   used), so decode goldens are as clean as prefill goldens.
5. **DETERMINISM (M0 gate)** — two full runs (prefill + decode +
   states) produce identical SHA-256 digests.
6. **ENVELOPE (M0 gate)** — f32-vs-f64 divergence on the fixture:
   gated at maxabs ≤ 0.1, rel ≤ 0.25.

## Measured (this MacBook Pro M1, seed 20260831)

| stage | maxabs | rel |
|---|---|---|
| logits prefill (7 tok) | 0.040 | 0.012 |
| logits decode (worst step) | 0.061 | 0.023 |
| h after L0 (gdn) | 0.0063 | 0.0067 |
| h after L1 (full) | 0.010 | 0.0096 |

Digest (f32 prefill+decode+states):
`d6a70357b286f07cf0af80f981519e352183ee6f7380f5188998f74ddb4f88fc`.

## Methodology

Two numeric modes, as in the Ling base oracle: **f32** (the C target —
bf16 RNE rounding at every point HF materializes a bf16 tensor, fp32
where HF keeps fp32: attention/router softmax, GDN decay + recurrence
+ state, l2norm, norm variances; matmuls f64-accumulated via the
deterministic sequential `_mm`) and **f64** (truth). The per-stage
|f32−f64| envelope is the scale the C forward must land inside of at
later milestones (C vs f32 golden ≤ envelope + realization slack).
Documented realization choices (reorder classes, absorbed by the
envelope) live in the oracle's header: MoE expert weighted-sum fp32
accumulate with a single bf16 round (HF `index_add_` into bf16), fp32
tree reductions as f64 sequential + `st()`, stable-argsort top-k
tie-break.

MTP and generation/sampling: out of scope for M0 (extension point
marked at the bottom of `tools/oracle.py`).

## Files

- `tools/oracle.py` — the oracle (per-op entry points for the later
  m4-analogues: `rmsnorm`, `rmsnorm_gated`, `causal_conv_silu`,
  `gdn_decay`, `l2norm`, `gdn_recurrence`, `apply_rope`,
  `attention_core`, `router_topk`, `moe_forward`, `gdn_mixer`,
  `attn_mixer`, `layer_forward`, `full_forward`)
- `tests/m0/test_oracle.py` — this test
- `tests/m0/fixtures/` — generated (gitignored)
