# tests/m10 — Metal GPU backend: dense BF16 compute on the Apple GPU (optional)

Milestone M10 (roadmap lever #7, the Apus M7b / Ling M10 pattern carried
across the adapter seam): an **optional** Metal backend
(`c/backend_metal.mm` + `c/backend_metal.h`) that offloads the dense BF16
matmuls of the Qwen3.6 forward pass — the GDN and gated-GQA attention
projections (decode GEMV AND the M9 batched prefill GEMM), the router
scoring matvec, the eager expert + shared-expert linears, and the LM
head — to the Apple GPU. FP32-only shaders (fast math DISABLED),
zero-copy unified-memory weight buffers, per-op fail-soft to the CPU
kernels, `APUS_METAL_DENSE_MB` budget (default 8192 MB), opt-in only
(`APUS_METAL=1` / `--metal`, or the separate `bin/apus-qwen-metal`
binary).

```
make metal               # builds bin/apus-qwen-metal (== make bin/apus-qwen-metal)
make test-m10            # kernel bitwise gates + model gate
make ubsan-m10           # model gate, UBSan on the C side
make bench-m10           # CPU-vs-Metal microbench (informational)
bin/apus-qwen-metal run --model DIR --tiered --metal ...   # or APUS_METAL=1
```

**The CPU binary stays the default and is untouched behaviorally**:
`bin/apus-qwen` never links the backend (weak stubs); with no
`--metal`/`APUS_METAL=1` the Metal binary runs the identical CPU kernels;
the CPU binary with `APUS_METAL=1` prints "metal backend not compiled in"
and continues on CPU. The Linux build sees none of this (stubs serve
there; `test-m10` is a clear macOS-only stub off-Darwin and is NOT in the
container battery).

## Backend interface (the wiring design)

A **function-pointer table**, `ApusBackendHooks apus_backend_hooks`
(`c/backend_metal.h`), defined zero-init in the `APUS_BF16_IMPLEMENTATION`
TU (bf16.h is in every binary) with weak stubs for
`apus_metal_enable/disable/...`. `apus_metal_enable()` (strong, in
`c/backend_metal.mm`) initializes Metal and fills the table. Call sites
try the hook first and run the CPU kernel when the pointer is NULL or the
call returns nonzero:

| hook | call site | semantics |
|---|---|---|
| `bf16_gemv` | `c/bf16.h` `apus_bf16_gemv_hot` — every dense decode GEMV: GDN projections (qkv 8192x2048, z 4096x2048, b/a 32x2048, out 2048x4096), gated-GQA projections (wq 8192x2048, wk/wv 512x2048, wo 2048x4096), **router scoring** (`apus_moe_route`, 256x2048), eager expert gate_up/down (1024x2048 / 2048x512), shared expert (512x2048, 2048x512, gate 1x2048), lm_head (248320x2048) | the M9b ILP NEON kernel's per-row rounding sequence, exactly |
| `bf16_gemm` | `c/bf16.h` `apus_bf16_gemm_hot` — the M9 batched prefill phase-A projections and phase-B expert/shared-expert batches (M=T) | same per-row sequence per (m,o) — M-independent, like the CPU kernel |
| `bf16_matvec_f32` | `c/moe.h` `apus_moe_matvec_f32_hot` | the fp32-out anchor (staged products, ascending adds). **Kept generic machinery: the Qwen router scores through the bf16 gemv hook** (fp32 softmax over widened bf16 logits), so this kernel has no model caller — the hook stays wired and is pinned at kernel level regardless |

NOT hooked: `apus_bf16_gemm_fast` (kept machinery with no production
caller in Qwen — the M9 batched path deliberately consumes NO
BLAS/gemm_fast; its M<128 fallback is the no-hook
`apus_bf16_gemm_hot_cpu`), and **tiered expert compute** — store slabs
are transient (LRU), so `c/cache.h`'s per-expert calls use the no-hook
`apus_bf16_gemv_hot_cpu` / `apus_bf16_gemm_hot_cpu` entry points and
never reach the pointer-keyed zero-copy cache (the stable-pointer
invariant: resident weights only; `apus_metal_disable()` drops the cache
— call it before freeing/reloading model weights).

## Numerics contract — a BITWISE offload (not a reorder)

Shaders are FP32 with **fast math disabled** (`MTLMathModeSafe`: no
contraction, no reassociation) and replicate the DISPATCHED CPU kernels'
rounding sequences exactly:

- `bf16_gemv` / `bf16_gemm` (`dot_ilp` shader): four float4 accumulators
  per output row; per ascending 32-wide chunk the eight single-rounded
  product vectors (exact `u16<<16` widening) accumulate `acc[q & 3] +=
  p_q` (q ascending); fixed combine `((a0+a1)+(a2+a3))` then
  `((s.x+s.y)+(s.z+s.w))`; K tail appended with scalar ascending adds
  (mul + add, two roundings — never fma); one RNE bf16 narrow at the end
  (the integer bit-trick). **Bitwise identical to `apus_bf16_gemv_hot` /
  `apus_bf16_gemm_hot` on every tested shape** (test_kernels below). The
  M9 re-anchor re-measured the ILP class on the Qwen shapes without
  changing the kernels' per-row sequence, so the base's shaders carry
  verbatim — the gate below PROVES the carry rather than assuming it.
- `bf16_matvec_f32`: staged single-rounded products per 32-wide chunk,
  then the 32 strictly ascending scalar adds; fp32 out. Bitwise
  identical to `apus_moe_matvec_f32_hot`.

Because the offload is bitwise, the CPU battery digests are definitionally
untouched, the model gate can demand full-logits bitwise equality (not
just token identity), and no tolerance adjudication was needed anywhere.
(IEEE specials note: like tests/m12, the gate's specials fill excludes
NaN codes as inputs — GPU NaN payload selection in mixed NaN arithmetic
is outside the contract; normative inputs are finite BF16. inf/±0/
subnormals and 0*inf -> default NaN are covered and bitwise.)

## Gate results (this machine, Apple M-series)

### Kernel level (`test_kernels`: 86 checks, 0 failures)

- `bf16_gemv`: Metal == dispatched CPU **BITWISE** on the shape sweep
  (odd tails, O=20 group boundary, the real Qwen shapes 8192x2048 /
  4096x2048 / 32x2048 / 2048x4096 / 512x2048 / 1024x2048 / 2048x512 /
  1x2048 / 256x2048), an IEEE-specials fill, and the 248320x2048 head
  slab-wise (8/8).
- `bf16_gemm`: BITWISE at M = 1,2,3,5,64,300 on Qwen shapes;
  M-independence (Metal GEMM row == Metal GEMM at M=1).
- `bf16_matvec_f32`: BITWISE on all six shapes (256x2048 + tails).
- Fail-soft: `APUS_METAL_DENSE_MB=1` — the 32 MB weight returns
  "unsupported" and the hooked call falls back to the CPU kernel
  **bitwise**; the small weight still goes to the GPU. Hook wiring:
  hooked `gemv_hot` == `_cpu` entry bitwise; GPU determinism bitwise.
- Zero-copy: 304.9 MB wrapped, **0 B uploaded** (kernel battery).
- Digest `3fbe7d90787aaa8c` identical at APUS_THREADS=1/4/8.

### Model level (`test_model`: 12 checks, 0 failures)

- **A.** Eager m5 fixture, T=64: Metal-hooked forward == CPU forward —
  logits **BITWISE** (the offload is bitwise, so the whole model is).
- **B. THE GATE — greedy teacher-forced stream, 24 tokens, CPU == Metal
  IDENTICAL.**
- **C.** Repeated Metal prefill bitwise; chunk invariance
  (prefill(64) == prefill(32) + 32 decodes) bitwise — Metal through both
  the batched-GEMM hook (M9 phase A) and the decode-GEMV hook.
- **D. Tiered slab safety** (m6a fixture, cache_bytes=1 = heavy
  eviction): Metal on, T=64 logits AND per-layer state **BITWISE** vs
  CPU — transient expert slabs compute on the CPU via the no-hook
  variants; any stale-buffer aliasing would flip these bits.
- Digest `3d00af73bcbf48f0` identical at APUS_THREADS=1/4/8 and under
  UBSan (the C side; the .mm object is shared, the Apus m7b precedent).
- CLI level: `bin/apus-qwen` vs `bin/apus-qwen-metal --metal` on the m5
  fixture, 16 greedy tokens — stream IDENTICAL; the CPU binary with
  `APUS_METAL=1` prints "metal backend not compiled in" and continues on
  CPU (fail-soft).

### Retarget notes (vs the Ling base suite)

- Shapes re-pointed to the Qwen set (above); the head slab test moved to
  248320x2048.
- The fixture vocab is 256 (the stale suite's hardcoded V=512 was the
  Ling fixture's — fixed; the tiered store config now reads the expert
  count from the loaded model instead of the Ling 32).
- The router hook moved: the Ling router used the fp32-out matvec; the
  Qwen router scores via `apus_bf16_gemv_hot` (bf16 logits -> fp32
  softmax), so router offload rides the gemv hook at 256x2048. The
  fp32-out matvec hook is kept (generic machinery) and pinned at kernel
  level.
- The base's two real-model A/B bugs stay fixed and gated: the
  `gemm_fast` M<128 fallback is the no-hook `_cpu` variant (transient
  tiered slabs never reach the pointer cache), and the backend
  serializes whole ops with a mutex (the tiered pilot's prediction
  thread calls the hooked router CONCURRENTLY with the forward thread).

### Real model

Not measured at M10 — weight download is in explicit WAIT (gate 8) and
the real-model A/B is M11. Fixture-level expectation: **PARITY** (the
base measured PARITY on the real Ling model; the bench below shows the
same on every Qwen shape, and the unified-memory argument — both sides
stream the same DRAM — is shape-independent). Record the real-model
verdict in the M11 entry.

## Measured performance (`make bench-m10`, threads=8, best of N)

Decode GEMV (M=1), effective weight-streaming bandwidth:

| op | shape | CPU (M9b/M9c ILP) | Metal | speedup |
|---|---|---|---|---|
| gdn qkv / attn wq | 8192x2048 | 0.620 ms (54.1 GB/s) | 0.613 ms (54.7 GB/s) | x1.01 |
| gdn out / attn wo | 2048x4096 | 0.556 ms (30.2 GB/s) | 0.573 ms (29.3 GB/s) | x0.97 |
| wk/wv + shared g/u | 512x2048 | 0.387 ms (5.4 GB/s) | 0.402 ms (5.2 GB/s) | x0.96 |
| expert gate_up | 1024x2048 | 0.403 ms (10.4 GB/s) | 0.384 ms (10.9 GB/s) | x1.05 |
| expert down | 2048x512 | 0.247 ms (8.5 GB/s) | 0.251 ms (8.4 GB/s) | x0.98 |
| router scoring | 256x2048 | 0.388 ms (2.7 GB/s) | 0.379 ms (2.8 GB/s) | x1.02 |
| lm head | 248320x2048 | 5.902 ms (172.3 GB/s) | 5.966 ms (170.5 GB/s) | x0.99 |
| matvec-f32 (kept) | 256x2048 | 0.253 ms | 0.275 ms | x0.92 |

Prefill-class GEMM (the M9 batched projections, M=512):

| op | shape | CPU | Metal | speedup |
|---|---|---|---|---|
| gdn qkv / attn wq | 512x8192x2048 | 75.5 ms (227 GF/s) | 76.9 ms (224 GF/s) | x0.98 |
| gdn out / attn wo | 512x2048x4096 | 31.3 ms (274 GF/s) | 31.3 ms (274 GF/s) | x1.00 |
| expert gate_up | 512x1024x2048 | 8.00 ms (269 GF/s) | 8.04 ms (267 GF/s) | x0.99 |
| expert down | 512x2048x512 | 4.16 ms (258 GF/s) | 4.14 ms (260 GF/s) | x1.01 |

**Verdict (mirrors the base's M10 finding exactly): PARITY.** On unified
memory both sides stream the same DRAM; the CPU hot path is already
multi-threaded and reaches 3-172 GB/s (GEMV) / 224-274 GF/s (GEMM)
depending on shape, and the naive one-thread-per-row shaders match but
do not beat it. Small shapes show the ~30-50 µs dispatch overhead
(x0.92-0.98). The wins are architectural, not raw: dense compute off
the CPU (freeing cores for expert resolve/disk I/O), and a proven
bitwise hook point for a future fused/multi-op or expert-GEMV
(GPU-resident experts) milestone. **Recommendation: keep opt-in,
default OFF.**

## Remaining gaps / future work

- Shaders are naive (thread-per-row, no simdgroup reductions or
  vectorized `half4`/`ushort4` loads — a per-row sequence-preserving
  restructure could lift bandwidth; the current parity makes it
  unmotivated).
- Tiered expert offload would need generation-tagged buffer entries
  keyed by (slab id, generation) or an explicit invalidation call from
  the store's eviction path (the Apus m7b README's analysis applies
  verbatim), plus a batched multi-expert dispatch to amortize the
  per-op overhead (~30-50 µs x the per-token expert GEMVs).
- Real-model A/B (golden text, tiered prefill+decode) at M11.
