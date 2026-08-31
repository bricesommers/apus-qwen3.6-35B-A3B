# M4 numerics contract — Qwen3.6-35B-A3B forward pass

Supersedes the inherited Ling contract (in git history). Source:
HF transformers `main` `reference/modeling_qwen3_5_moe.py` ("HF:NNN" =
line in that file), cross-checked against the checkpoint index
(`reference/model.safetensors.index.json`) and, for MTP only, vLLM
`qwen3_next_mtp.py`. Single implementation — no HF/vLLM disagreement to
reconcile yet; vLLM/SGLang cross-check is an M4 milestone task.

Weights BF16 unless noted (A_log, dt_bias are F32 in the Ling sense —
here they are BF16-loadable smalls; the checkpoint is pure BF16).
"rnd" = round to BF16 (RNE). Layer pattern: `layer_types` =
[linear,linear,linear,full]×10 — full-attn layers are idx 3,7,…,39.

## 1. Embedding
`emb[id]` bf16, NO scaling (HF:1346–1347). lm_head untied, no bias;
logits bf16, NOT upcast (HF:1901–1903). No softcap, no logit scaling.

## 2. RMSNorm(x, w) — input/post_attention/final + attention q/k norm
fp32: `x̂ = x·rsqrt(mean(x²)+1e-6)`; then `x̂·(1.0 + w)` — the weight is
**zero-init, checkpoint stores gain−1** (HF:878–895, init L986). The
converter keeps bytes identical; the +1 is applied at runtime.
NOTE: the GDN output norm below is the OTHER convention (direct weight).

## 3. Gated DeltaNet layer (30 linear layers)
Dims: key_dim 2048 (16 heads × 128), value_dim 4096 (32 heads × 128),
conv_dim 8192.
- in_proj_qkv: 2048→8192 → [q 2048 | k 2048 | v 4096] (HF:499,558–566);
  in_proj_z 2048→4096; in_proj_b 2048→32; in_proj_a 2048→32. No biases.
- ONE fused depthwise causal conv1d k=4 over the 8192 channels (no bias),
  then SiLU; fp32 internal, out bf16. Conv state = last 4 PRE-conv inputs
  per channel (HF:227–247; decode update 207–224).
- β = `sigmoid(b)` — computed in **BF16** (HF:572). (Ling KDA used fp32 —
  deliberate difference.)
- Decay, fp32, per-V-head SCALAR [32]:
  `g = −exp(A_log.float()) · softplus(a.float() + dt_bias)` (HF:574).
  NO lower bound, NOT the KDA sigmoid form.
- q,k: 16 K heads → repeat_interleave ×2 → 32 (HF:575–577); L2-norm per
  head fp32, `x·rsqrt(Σx²+1e-6)` (HF:252–255,301–303); q ×= 128^-0.5.
- Recurrence (CANONICAL semantics — the chunk path HF:258–392 is used by
  HF for any seq_len>1 but is NOT bitwise equal; the engine and oracle
  implement the recurrent form only, HF:438–449), all fp32, per v-head
  state S[128,128]:
  `S ← S·exp(g_t)`; `kv = k_tᵀS`; `S ← S + k_t⊗(β_t·(v_t − kv))`;
  `o_t = q_tᵀS`. State fp32 [32,128,128] per layer.
- Output norm per head (128 dims), RMSNormGated: fp32 variance →
  x̂ = x·rsqrt(var+1e-6) → **rnd(x̂)** → ×w (DIRECT weight, init ones, no
  +1) → ×silu(z.float()) in fp32 → rnd (HF:176–192). Norm-before-gate.
- out = rnd(out_proj(flatten(o))), 4096→2048 (HF:495).
- Residual add bf16. No RoPE, no attention mask semantics.

## 4. Full attention layer (layers 3,7,…,39)
16 q heads, 2 KV heads, head_dim 256.
- q_proj 2048→8192, viewed [16,512], per-head chunk → [q(256) | gate(256)]
  (HF:716–718,744–746). k/v_proj 2048→512 each; o_proj 4096→2048. No
  biases.
- Per-head q/k RMSNorm (contract #2, (1+w) variant) BEFORE RoPE
  (HF:728–731,749–750).
- Partial RoPE: first 64 dims only (partial_rotary_factor 0.25 × 256);
  GPT-NeoX rotate_half pairing (i, i+32) within the rotary part; θ=1e7;
  inv_freq fp32, cos/sin cast bf16 before application; pass dims
  untouched (HF:124–132,154,621–625,629–664). attention_scaling = 1.0.
  mrope: text-only ⇒ all 3 position rows = arange ⇒ identity
  (HF:1352–1364,1541–1546) — plain partial RoPE-64 for the engine.
- Eager attention: A = softmax_fp32(QKᵀ·256^-0.5 + causal mask) → rnd →
  ·V (HF:679–701). KV repeat ×8.
- Output gate: `o ⊙= sigmoid(gate)` in bf16, AFTER attention, BEFORE
  o_proj (HF:774–777). The `attn_output_gate` config key is decorative —
  the gate is unconditional.
- Residual add bf16.

## 5. MoE (ALL 40 layers — no dense layers)
- Router: logits = rnd(x·W_gateᵀ) [256] bf16 matmul → **fp32 softmax** →
  top-8 → **renormalize w /= Σw** → rnd (HF:837–853). NO bias, NO
  sigmoid, NO scaling factor, NO group-limit, NO jitter. Tie-break:
  oracle pins lowest-index-wins; goldens avoid exact ties.
- Experts (fused tensors): gate_up_proj [256,1024,2048], down_proj
  [256,2048,512]; per token: gate‖up = rnd(x·gate_up_eᵀ).chunk(2);
  silu(gate)⊙up (NO clamp — Ling's SwigluStepAndMul does not exist here);
  rnd(·down_eᵀ); ×w_e; fp32 sum (HF:797–834).
- Shared expert: plain SiLU MLP 2048→512→2048, output gated by
  `sigmoid(shared_expert_gate(x))` ([1,2048] linear, no bias)
  (HF:861–862,871).
- out = rnd(experts + shared) (HF:873). Residual add bf16.

## 6. Final norm → lm_head
Final RMSNorm (contract #2); lm_head bf16 GEMM fp32 accum, logits bf16.

## 7. MTP (deferred to M8 — does not affect main-model logits)
HF does NOT implement MTP (keys ignored on load, HF:965,1832). From
checkpoint names + vLLM qwen3_next_mtp.py: e = pre_fc_norm_embedding(
emb(ids)); h = pre_fc_norm_hidden(h_t); x = fc(cat[e,h]) [4096→2048] →
one FULL-ATTENTION decoder layer (contract #4) + full MoE (contract #5)
→ mtp.norm → SHARED lm_head + SHARED embeddings. All MTP norms are the
(1+w) variant.

## Rounding-order decisions
1. Recurrent GDN is canonical; HF's chunk path is a tolerance class,
   never the golden source (Phase A decision, mirrors the Ling KDA
   precedent).
2. β sigmoid in bf16 (HF:572) — not fp32.
3. GDN output norm: rnd AFTER normalize, BEFORE weight (HF:188–192).
4. All non-GDN norms: (1+w) at runtime, bytes untouched.
5. Router: fp32 softmax + renorm; ties lowest-index (oracle rule).
6. Prefill == decode bitwise: the engine runs the same per-token
   recurrent body in both; HF's prefill chunk path is not the reference.
7. Attention softmax fp32; QK·scale and P·V rounded per HF eager order.

## Things that do NOT exist in this model
No KDA per-channel decay, no lower-bound gate, no MLA/compressed KV, no
sliding window, no DSA/indexer, no attention sinks, no softcap, no mHC,
no sigmoid/bias/group router, no SwiGLU clamps, no dense layers, no
quantization of any kind. `attn_output_gate` is a decorative config key.
Vision tower (`model.visual.*`) is stripped at conversion — text-only
scope.
