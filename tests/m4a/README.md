# tests/m4a — GDN / GQA / MoE per-operation kernels (c/gdn.h, c/attn.h, c/moe.h) hard gate

Milestone M4a, Qwen3.6-35B-A3B: the per-operation kernels of the forward
pass (docs/M4-CONTRACT.md sections 2–5), each unit-golden-tested. M4b/c
assembles them into the single-layer forward. Tier discipline: the
scalar C11 anchors are the normative semantic definition; the NEON
variants are gated **BITWISE == scalar** per op, and the plain entry
points dispatch to NEON when compiled in (the engine rule — an
optimized path is verified against its scalar anchor, never trusted).
The mt variants (recurrence step, GQA) are gated bitwise == sequential;
the Makefile diffs stdout across APUS_THREADS=1/4/8 (thread
independence, like test-m3).

## Numerics path (normative — the C headers are the anchor)

Every op: bf16 storage, fp32 compute, sequential ascending accumulation,
one IEEE fp32 rounding per mul and per add (`-ffp-contract=off` pinned),
`rnd` = apus_bf16_bits (RNE). Full per-op contracts are in the header
comment blocks of c/gdn.h, c/attn.h, c/moe.h; the salient pins:

- **conv1d k=4 + SiLU, ONE fused depthwise conv over the [q|k|v]
  channels** (c/gdn.h): fp32 tap accumulation over the causal window
  (oldest first), then **TWO rounding points** (torch opmath; NOT
  Ling's single round): `co = rnd(acc)`, `out = rnd(co·sigmoid(co))`.
  State = last 3 PRE-conv inputs (bf16), zero-init.
- **decay**: `g[h] = −expf(A_log[h])·softplus(f32(a[h])+dt_bias[h])`,
  per-V-head scalar, fp32 out, NO lower bound (softplus = log1p(exp x),
  identity x>20 — NOT the KDA sigmoid form).
- **beta**: `rnd(sigmoid(b))` — computed in **BF16** (the deliberate
  difference from Ling's fp32 beta).
- **l2norm**: `x·rsqrt(Σx²+1e-6)` per head (multiply-by-reciprocal, NOT
  Ling's literal division), **fp32 out** (no bf16 rounding), q scaled
  `dk^-0.5` as a second fp32 rounding.
- **delta-rule recurrence** (CANONICAL; the HF chunk path is a
  tolerance class, never the reference): all fp32. Per token per
  v-head: `S = S·expf(g)` (one scalar per head); `delta = (v − k·S)·β`;
  `S += k⊗delta`; `o = q·S` on the UPDATED S. fp32 o out, no rnd.
- **RMSNormGated (GDN output norm)**: DIRECT weight (no +1), three
  rounding points: rnd(x·rsqrt) → rnd(w·x) → rnd(x·silu(z)).
- **RMSNorm (1+w) variant** (c/attn.h — input/post-attention/final +
  attention q/k norms; checkpoint stores gain−1): fp32 normalize, then
  **SINGLE rounding** `rnd((x·rs)·(1+w))` — NOT Ling's
  normalize-rnd-then-weight block convention.
- **Partial RoPE**: NeoX rotate_half pairs (i, i+rot/2) within the
  first rot=64 dims only (pass dims untouched), θ=1e7, fp32
  inv_freq/angle, fp32 cos/sin cast bf16, one rnd per mul + one per add.
- **GQA eager attention**: 16 q heads / 2 KV heads (kv = h/8),
  `A = rnd(rnd(qkᵀ)·256^-0.5)`; softmax fp32 (max-subtract, ascending);
  `P = rnd(e/s)`; `o = rnd(P·V)`; causal; KV cache append BEFORE
  attention (decode == full-recompute row bitwise).
- **output gate**: elementwise `g = rnd(sigmoid(gate))`, `rnd(o·g)`
  (the gate is the packed q_proj [q|gate] half; unconditional).
- **router**: `logits = rnd(x·W_gate)` (bf16 matmul) → **fp32 softmax**
  → top-8 (strict-`>` replacement ⇒ lowest index on ties) → renormalize
  → **BF16 weights**. NO bias/sigmoid/scaling/group-limit.
- **routed-expert act** (fused gate_up): TWO rounds — `rnd(silu(g))`
  then `rnd(·up)`; **shared-expert act**: SINGLE round
  `rnd(silu(g)·u)`. NO clamps (SwigluStepAndMul does not exist).
- **combine**: `rnd(Σ_e f32(w_e)·f32(y_e))` — fp32 accumulate across
  the top-k with a SINGLE bf16 round (the oracle's documented
  realization class vs HF's per-expert index_add_ rounding; absorbed by
  the m4b/m4c envelope).

## What is tested / measured (this MacBook Pro M1 Pro, `make test-m4a`)

126 checks total, 0 failures (`make ubsan-m4a` clean; ASan is broken on
this machine, same as m2/m3). Stdout digests identical at
APUS_THREADS=1/4/8 (the Makefile diffs them).

### test_gdn (45 checks)

- **conv**: golden f64 (C=64, T=8 zero-state + T2=3 continuation, two
  rounding points) with esc = Σ|w·win|; state == last-3 pre-conv codes
  (exact); **BITWISE prefill == decode-stepping** (outputs AND state);
  in-test f64 truth (C=37, T=9); zero input → +0; **BITWISE scalar ==
  NEON** (odd C exercises the tail). Max err = **1.30e-03**.
- **decay**: golden f64 (H=8) + corners: a+dt=0 → −exp(A_log)·log2
  exactly; a+dt=25 → the x>20 identity branch; a+dt=−25 → the exp tail.
  In-test f64 (odd H=13). Max err = **6.04e-08**.
- **beta**: golden f64 + exact corners: b=0 → code 0x3F00 (0.5);
  b=+20 → code 0x3F80 (rounds to exactly 1.0); b=−20 → ~0. Max err 0.
- **l2norm**: golden f64 (incl. a 1e-10-magnitude row) at the q scale
  AND scale=1; zero vector → +0; in-test f64 (odd D=65). Max err =
  **6.44e-08**.
- **recurrence**: golden f64 for o and the final state S (H=2, D=8,
  T=6; g rows at exactly 0 and −5, β corners 0 and 1), esc-tracked
  (esc_S propagated as esc·dec + |k·delta| per step; esc_o = Σ|q·S|).
  **BITWISE apus_gdn_recurrent == stepping apus_gdn_step**, **mt ==
  sequential**, **scalar == NEON head body** (and dispatched ==
  anchor). In-test f64 (D=32, T=10); decay extremes (g=0/β=0 state
  unchanged; g=−5 zero-k exact decay). Max err: o **1.89e-07**, S
  **2.04e-06** (fp32-out, assert 1e-5 of esc).
- **onorm (RMSNormGated)**: golden f64 (incl. tiny-magnitude head);
  z=+88 → silu(z)=z (sigmoid saturates to 1.0) so output is bf16(88);
  z=−88 → ~0. In-test f64. Max err = **3.68e-03**.

### test_attn (39 checks)

- **rmsnorm (1+w, single round)**: golden f64 (N=512) + in-test f64 at
  the real N=2048; zero vector; **scalar == NEON** (odd N=1023). Max
  err = **6.99e-03**.
- **rope** (D=64, rot=16, θ=1e7): positions 0/1/7/1000/131071/262143.
  The golden replicates the fp32 angle (fp32 inv_freq, fp32 product)
  before f64 cos/sin — cos/sin bf16 codes match the oracle **0/96
  flips**. Pass-through dims exact copies; **position 0 == identity
  bitwise**; **scalar == NEON** per position. Max err = **0**.
- **gqa** (H=4, Hkv=2, D=64, T=6): golden f64 with the per-step
  roundings replicated (A rnd–scale–rnd, P rnd); in-test f64 truth
  (odd D=65, Hkv=1); **BITWISE decode == last row of full recompute**,
  **mt == sequential**, **scalar == NEON row** (and dispatched ==
  anchor). Max err = **1.91e-03**.
- **outgate**: golden f64 + corners: logit 0 → gate code 0x3F00,
  o·0.5 exact halving; +20 → gate rounds to exactly 1.0 → y == o
  codes; −20 → ~0. Max err = **0**.

### test_moe (42 checks)

- **fp32-out matvec** (generic machinery): shape sweep incl. {1×1,
  3×5, 64×256, 256×2048, 7×1, 33×100} vs in-test f64, esc = Σ|w·x|;
  **hot == scalar bitwise**, **NEON == scalar bitwise**. Max err/esc =
  **3.75e-08**.
- **router golden** (E=256, K=128): probs (from the bf16-rounded
  logits) err **1.37e-16**, selection **EXACT** (fixture margin 1.04e-4
  asserted by the generator), weights bf16-rounded exact. **topn(topk)
  == route selection bitwise**; topn(topk+4) prefix == route bitwise.
- **router tie-breaks** (constructed, C-side exact): all-equal scores →
  idx = 0..7, w = 0.125 exactly (renormalized uniform); expert tie at
  the cut → lowest index wins.
- **routed act (two rounds) / shared silu_mul (one round)**: golden
  f64 + in-test (odd I=767); silu(0) = +0; **scalar == NEON**. Max err
  = **3.74e-03 / 3.81e-03**.
- **combine** (bf16 weights): golden f64 (k=8, N=768, esc = Σ|w·y|) +
  in-test at odd N=767; k=1 → rnd(w·y); zero weights → +0 codes (IEEE
  (+0)+(−0) = +0); **scalar == NEON**. Max err = **1.56e-03**.

## Tolerance classes and rationale

- **fp32-out ops** (decay, l2norm, recurrence o/S, fp32 matvec,
  softmax probs): one fp32 rounding per op vs f64 truth ⇒ assert err ≤
  1e-5 of (|gold|+esc); measured ≤ 2e-6.
- **bf16-out ops**: the single final RND dominates: ≤ 2^-8 relative to
  the output (up to 2 bf16 ulp at binade edges) plus fp32 internal
  noise ~1e-6·esc ⇒ assert 0.008·(|gold|+esc); measured ≤ 7e-3.
- **esc** is always the sum of ABSOLUTE products, never |dot| —
  cancellation drives |dot| to ~0 while fp32 rounding scales with the
  absolute terms (the Apus m4a finding).
- **bitwise gates** (exact, no oracle involved): conv prefill ==
  decode-stepping; recurrence prefill == step loop; GQA decode == full
  recompute row; RoPE pos 0 == identity; router selection + tie-breaks;
  conv state codes; **scalar anchor == NEON for every op**; mt ==
  sequential (thread-count independence via the Makefile's
  APUS_THREADS=1/4/8 stdout diff).
- **libm class**: expf/log1pf/sqrtf/cosf/sinf are the platform libm's.
  Goldens evaluate in f64, so libm ulps live inside the tolerances
  above; a ulp landing on a bf16 tie flips one code by ≤ 2^-8 of its
  term, covered by the esc. No cross-platform bitwise claim (the Apus
  M12c finding).

## Surprises / notes for the M4b integration implementer

- **Two norm conventions coexist.** Every RMSNorm EXCEPT the GDN
  output norm is the (1+w) zero-init variant with a SINGLE end rounding
  (c/attn.h). The GDN output norm (c/gdn.h) is DIRECT weight with THREE
  rounding points (rnd after normalize, after weight, after gate).
- **The GDN recurrence consumes FP32 q,k** (l2norm does NOT round to
  bf16) and its output is rounded to bf16 BEFORE the gated norm
  (HF:454) — do not skip that rnd.
- **beta is bf16** (HF:572), not fp32 — and the recurrence widens it.
- **Conv has TWO rounding points** (acc → bf16, silu → bf16); Ling's
  conv had one. Same for the routed-expert activation (rnd(silu) THEN
  rnd(·up)) — but the shared expert has ONE. The asymmetry is the
  oracle's, replicated deliberately.
- **Router logits are bf16** (the GEMV rounds) — softmax reads the
  ROUNDED logits widened. Weights are bf16 after renormalization.
- **Fixture filename caveat (macOS case-insensitive APFS)**:
  `gdn_decay_a.bin` and `gdn_decay_A.bin` COLLIDE — the A_log fixture
  is `gdn_decay_alog.bin`.
- **q,k l2norm before repeat_interleave**: norming the 16 K heads and
  memcpy-ing x2 is bitwise identical to norming the duplicated 32 heads
  (per-head determinism) — the layer exploits this; c/gdn.h documents
  it.

## Files

- `c/gdn.h` — fused conv1d+SiLU, decay, beta, l2norm, delta-rule
  recurrence, RMSNormGated (`APUS_GDN_IMPLEMENTATION`)
- `c/attn.h` — RMSNorm (1+w), partial NeoX RoPE, GQA eager attention +
  KV-cache decode, elementwise output gate (`APUS_ATTN_IMPLEMENTATION`)
- `c/moe.h` — fp32-out matvec (generic), fp32-softmax top-k router,
  routed/shared activations, weighted combine
  (`APUS_MOE_IMPLEMENTATION`)
- `tests/m4a/gen_golden.py` — f64 numpy oracles (fixed seed 20260829)
- `tests/m4a/test_gdn.c`, `test_attn.c`, `test_moe.c` — hard-gate tests
- `tests/m4a/golden/` — generated fixtures (via `make golden-m4a` /
  `test-m4a`)
