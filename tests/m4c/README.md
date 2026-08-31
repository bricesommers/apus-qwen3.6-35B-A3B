# tests/m4c — C single-layer forward, verified against the M4b goldens

Milestone M4c, Qwen3.6-35B-A3B: `c/layer.h` assembles the M3/M4a
kernels into ONE decoder layer of each kind (gdn = Gated DeltaNet +
MoE, full = gated GQA full attention + MoE — MoE on every layer, no
dense layers), verified stage-by-stage against the numpy oracle
fixtures (tests/m4b).

Run:

```
make test-m4c     # 246 checks, exit 0 iff all pass
make ubsan-m4c    # same under -fsanitize=undefined (ASan broken on this Mac)
```

`make golden-m4b` regenerates the fixtures (deterministic) and runs the
oracle self-checks. The mt kernel paths (GDN recurrence step, GQA
decode) use the pool, so the Makefile diffs stdout across
APUS_THREADS=1/4/8 (thread independence).

## Wiring (c/layer.h, contract §3-5, HF:911-954)

`res1 = rnd(x + mixer(rmsnorm(x)))`, `out = rnd(res1 +
moe(rmsnorm(res1)))`, all residuals bf16 adds, all norms the (1+w)
single-round variant. GDN mixer: fused qkv/z/b/a projections → ONE
fused depthwise conv1d k=4 + SiLU over the [q|k|v] channels (state,
two rounding points) → beta bf16 sigmoid, decay softplus fp32 → q,k
l2norm fp32 over the 16 K heads (q scaled dk^-0.5) → repeat ×2 to the
32 V heads (bitwise — see the c/gdn.h note) → delta-rule step (fp32
state S) → rnd → RMSNormGated (direct weight) → out_proj. FULL mixer:
q_proj packed per-head [q|gate] → per-head (1+w) q/k norms BEFORE RoPE
→ partial NeoX RoPE (rot dims, absolute position) → cache append →
eager GQA (fp32 softmax, kv head h/nrep) → elementwise sigmoid output
gate → o_proj. MoE: router (bf16 logits → fp32 softmax → top-k lowest
index → renorm → bf16 weights) → per-expert fused gate_up / two-round
act / down → fp32 weighted combine single round (the oracle's
documented realization class) → shared expert single-round silu-MLP
gated by rnd(sigmoid(rnd(W_gate·x))) → bf16 add.

**Decode == prefill by construction**: `apus_layer_forward` runs every
token through the single per-token body (per-row GEMVs + the M4a
state-carrying kernels), so any prefill/decode chunking is bitwise
identical. `apus_layer_forward_hot` IS `apus_layer_forward` at M4 (the
Ling M6c/M9c batched prefill is deferred to this adapter's own
performance milestones).

## Gates (per tests/m4b/README.md methodology)

- **Stage goldens** (every named intermediate, prefill + each of the 5
  decode steps × 2 kinds): `max|C − f32| ≤ envelope + slack·scale`,
  where envelope = `max|f32 − f64|` from the fixture pair and
  slack = 0.005 (bf16-valued stages) / 0.001 (fp32-valued: gdecay) of
  scale = max|f32 golden|. I.e. C must land inside the oracle's own
  dtype-divergence envelope, plus fp32 accumulation-order slack (which
  includes the ARM gemv hot path's user-approved M9b ILP reorder
  class). **rtr_idx: EXACT** (fixture selection margins verified by
  the oracle self-check).
- **State after prefill** vs the state fixtures: GDN conv state and
  FULL KV caches as bf16 codes (≤1% single-code tie flips allowed —
  the documented realization class, C sequential fp32 vs the oracle's
  f64-accumulated ideal; measured 0/1536 and 0/896+0/896), S fp32 at
  1e-3·scale (measured 1.5e-8 / 0.226), pos exact.
- **Chunk invariance (C-side)**: one-shot T=12 vs prefill 7 + 5
  decodes — outputs, conv state, S, caches, pos **BITWISE identical**
  (both kinds). Plus a rerun-determinism bitwise check.
- **Digest** FNV over outputs, identical at APUS_THREADS=1/4/8 and
  under UBSan (c441a570f6343039).

## Measured C-vs-f32 vs envelope (worst rows, this MacBook Pro M1 Pro)

Most stages are **bitwise-equal** to the f32 golden (C−f32 = 0):
every ln1/res1/ln2 row, all full-attention mixer stages, the MoE
shared stage, most outputs. Nonzero worst rows are single bf16 tie
flips or fp32 noise, always far inside the envelope:

| stage (worst) | C−f32 | envelope | scale |
|---|---|---|---|
| gdn out (pre) | 7.8e-3 | 0.0154 | 3.3 |
| gdn moe_out | 2.0e-3 | 1.5e-3 | 0.18 |
| gdn moe_routed | 9.8e-4 | 1.2e-3 | 0.18 |
| gdn gdecay (fp32) | 1.9e-6 | 0.024 | 27.8 |
| full out (pre) | 0.0156 | 0.0144 | 3.1 |
| full moe_out | 4.9e-4 | 1.6e-3 | 0.15 |

Router selections: exact on all 24 tokens × 2 kinds. Envelope itself
(f32 vs f64): 0.09%–1.2% rel per stage (see `make golden-m4b` output /
fixtures/envelope.txt).

## Surprises / notes for M5

- **Norm storage convention is a hard gate**: every RMSNorm EXCEPT the
  GDN gated output norm is the (1+w) zero-init variant — the
  checkpoint stores gain−1 (fixture: N(0, 0.1) codes) and the +1 is
  applied at runtime; the GDN output norm is DIRECT (fixture:
  1+N(0, 0.1)). The oracle loads RAW values and applies the +1 itself;
  the C kernels do the same.
- **The C conv state holds K−1 columns** (last 3 pre-conv inputs); the
  oracle's [C,K] state's first column is dead weight — the fixture
  dump slices it off (tests/m4b/gen_fixtures.py dump_state_codes).
- **Router near-ties are a fixture property, not a tolerance** (same
  lesson as the Ling base): the selection-identity oracle gate pins
  the seed, never a wider C tolerance.
- **The q/k l2norm happens BEFORE repeat_interleave** (norming the 16
  K heads and memcpy-ing ×2 is bitwise the norm of the duplicated 32
  heads) — the layer and c/gdn.h both document this.
- **Projection GEMVs are the M9b ILP reorder class on ARM** (the
  gemv_hot dispatch) — absorbed by the stage envelopes, and the reason
  the envelope slack is 0.005 rather than bitwise.
- c/layer.h is correctness-first: scalar per-token body, no scratch
  arena, VLAs. M5 owns the full-model rewiring (c/model.h is
  compile-only Ling-shaped glue at M4); the performance batching
  returns with the adapter's own M6c/M9c.

## Files

- `c/layer.h` — the layer assembly (`APUS_LAYER_IMPLEMENTATION`;
  ApusLayerCfg/ApusLayerW/ApusLayerState/ApusLayerTrace).
- `tools/oracle.py` — the M4b numpy oracle (see tests/m4b/README.md).
- `tests/m4b/{gen_fixtures.py,check_oracle.py,fixtures/}` — goldens.
- `tests/m4c/test_layer.c` — the hard-gate test.
