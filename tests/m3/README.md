# tests/m3 — BF16 kernel hard gate (c/bf16.h)

Milestone M3: BF16 GEMV/GEMM kernels for Qwen3.6-35B-A3B, a **pure-BF16
checkpoint** (no quantization anywhere — where Apus needed MXFP4/FP8
kernels, this engine needs plain BF16 matmuls). Scalar reference + NEON +
threaded (c/pool.h), C11 + libc + arm_neon.h + pthreads only. Mirrors the
Apus M3 discipline: exhaustive format tests, numpy-oracle goldens, FP64
truth with esc/err classes, shape sweeps, thread-count independence.
The kernel layer (c/bf16.h, c/blas.h, c/x86.h) is inherited PROVEN from
the Ling base — this milestone re-gates the battery onto the Qwen shapes.

## Numerics contract (normative — c/bf16.h header is the authority)

Storage: `W [O,K]` BF16 row-major, `x [M,K]` BF16, `y [M,O]` BF16.

- **Widen** bf16→fp32 is EXACT: `f32 bits = (uint32)code << 16`
  (subnormals, ±0, inf, NaN payloads included).
- **Narrow** fp32→bf16 is round-to-nearest-even on the low 16 bits
  (`u += 0x7FFF + ((u>>16)&1)`, truncate). Finite overflow rounds to ±inf
  per RNE (the 0x7F7F/0x7F80 midpoint ties to the even code 0x7F80 = inf);
  NaN passes through as the high 16 bits.
- **The scalar anchor IS the semantic definition** of every BF16 linear in
  this engine:
  ```
  acc = 0.0f
  for k in 0..K-1 (strictly ascending):
      p    = f32(W[o,k]) * f32(x[m,k])   # one IEEE fp32 rounding
      acc += p                           # a second rounding — NO FMA
  y[m,o] = bf16_narrow(acc)
  ```
  Two roundings per element, sequential k order; contraction is pinned off
  (`-ffp-contract=off`) so no compiler can fuse mul+add. This mirrors
  torch.nn.Linear's bf16 contract (fp32 accumulate, bf16 out); torch leaves
  the summation order unspecified, so our fixed order is the normative
  in-engine anchor.
- **NEON kernels: BITWISE == scalar** (the design decision of this
  milestone; justification below). Widening is exact (`vshll_n_u16` 16),
  products are computed 4-wide (`vmulq_f32` — the same single rounding as
  the scalar mul) and staged in 32-wide chunks, and every output's adds run
  strictly in ascending k order. ILP comes only from interleaving
  INDEPENDENT output chains (4 rows at a time, named scalar accumulators —
  the Apus M12a-2 Rosetta finding), never from reassociation.
- **Threaded (`*_mt`): bitwise == the single-thread path at EVERY pool
  size** (APUS_THREADS=1 included): x is widened once by the calling
  thread (exact), output rows are partitioned contiguously over c/pool.h
  lanes, and each `y[m,o]` is computed entirely by one lane with the
  identical per-output accumulation order.
- **IEEE edge behavior** (deterministic, identical across all paths):
  inf/NaN widen/narrow exactly, `0*inf → NaN`, fp32 accumulation overflow
  `→ ±inf → bf16 inf`, subnormal underflow → ±0. Normative use assumes
  finite inputs.

### Bitwise vs reorder-class decision

**All NEON and mt paths are bitwise-identical to the scalar anchor; no
reorder class is consumed anywhere in M3.** Apus's M9a fp4 NEON kernels
took the documented 4-accumulator ILP-reorder class (gated by err/esc
bounds) because their per-32-block dots could not be vectorized without
reassociation. Here the per-element work (widen + one mul) vectorizes
exactly and the sequential fadd chain can stay untouched while ILP comes
from independent output rows — the pattern Apus's M12a-2 AVX2 kernels
prove at full SIMD width (tests/m12/README.md "M12a-2 contract": staged
products, scalar sequential order, no FMA). Bitwise costs no correctness
budget and little speed (the kernel is memory-bound multithreaded; see
bench), so the tolerance class stays unspent for wherever it is actually
needed.

## What is tested

1. **Widen exhaustive** — all 65,536 bf16 codes (subnormals, ±0, inf, NaN
   payloads): scalar == the exact `(code<<16)` pattern per code;
   `narrow(widen(b)) == b` and `bf16_round(widen(b)) == widen(b)`
   identities; NEON row widen bitwise == scalar (full array + odd-offset
   tail).
2. **Narrow exhaustive vs numpy oracle** — all 65,536 codes as high bits ×
   6 low-16 patterns {0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF}
   (393,216 fp32 values: exact hits, subnormals, tie midpoints,
   just-below/above-tie, saturation-to-inf, inf, NaN payloads).
   `gen_golden.py` computes the oracle with float64 candidate distances
   and explicit tie-to-even — deliberately NOT the C bit-trick, so a bug
   in one cannot hide in the other. **0 mismatches.**
3. **Golden GEMM vs numpy f64 truth** (fixed seed 20260806; M=3, O=64,
   K=256; corner rows: all-zero, all-negative, bf16 subnormals, 2^100
   magnitudes, ±1 alternating): y_f64 = ascending-k double sum; tolerance
   as a fraction of the per-output error scale `esc = Σ|w·x|` (fp32
   accumulation error scales with esc, not |out|, which cancellation can
   drive to ~0). Measured: accumulation err/esc = **0** beyond the bf16
   output quantization; raw err/esc (incl. the 2⁻⁹ output rounding) =
   6.07e-4. Scalar vs NEON **bitwise**; GEMV row == GEMM row bitwise for
   every m.
4. **Shape sweep** — O×K ∈ {1×1, 1×8, 3×5, 5×32, 17×160, 64×256, 127×255,
   128×512} (generic edge shapes) plus the real Qwen shapes
   {8192×2048 (GDN in_proj_qkv / attn q_proj), 4096×2048 (GDN in_proj_z),
   32×2048 (GDN in_proj_b/a), 2048×4096 (GDN out_proj / attn o_proj),
   512×2048 (attn k/v_proj), 256×2048 (router), 1×2048
   (shared_expert_gate), 1024×2048 (expert gate_up), 2048×512 (expert
   down)} × M ∈ {1,2,4,7},
   random bf16 + sprinkled subnormal weight rows + an all-zero activation
   row. Gates per (shape, M): scalar vs NEON vs mt **bitwise** (0 diffs
   across the whole sweep), err/esc vs in-test FP64 truth < 1e-4,
   M-independence (GEMM rows 0 and M-1 == GEMV, bitwise, scalar + NEON +
   gemv_mt). Measured max err/esc = **6.35e-08** (≈ √K·2⁻²⁴ at K=4096;
   the linear adversarial bound is (K−1)·2⁻²⁴ ≈ 2.4e-4 — the 1e-4 gate
   sits between the √K random-data scaling and that bound, >1500× over
   measured); raw incl. output rounding 5.25e-4.
5. **LM head 248320×2048, light mode** — full 1.02 GB weight tensor, M=1:
   mt vs single-thread NEON bitwise over all 248,320 rows; three
   1024-row slabs (start/middle/end) scalar vs NEON bitwise + f64 truth
   (rows are independent, so a GEMV on `w + o0*K` IS rows [o0, o0+1024) of
   the full GEMV — no API change needed). Slab err/esc = 0 beyond output
   quantization.
6. **Edge cases** — K=1, K=3, O=1 with K=5 (chunk-tail-only paths), zero
   activations → exactly +0 (0x0000), inf·0 → NaN (sign/payload through
   narrow), fp32 accumulation overflow → ±inf (0x7F80/0xFF80), subnormal ×
   subnormal underflow → +0, −0 handling.

`make test-m3`: **459,292 checks, 0 failures**, digest `99bb4ee4e1509abb`,
stdout diffed **identical across APUS_THREADS=1/4/8**. `make ubsan-m3`
clean, same digest (UBSan found a real bug during the base's development —
a stack-buffer overflow in the test's own edge-case scratch — caught here
before any forward-pass integration). ASan intentionally absent (broken on
the dev Mac, as in Apus); `leaks --atExit` on the test binary: **0 leaks**.

## Benchmark (this MacBook Pro M1, `make bench-m3`; GB/s = bf16 weight bytes streamed)

| shape (O×K)           | path   | time (µs) | GB/s  | GFLOP/s |
|-----------------------|--------|-----------|-------|---------|
| 8192×2048 (qkv/q)     | scalar | 15342     | 2.19  | 2.19    |
|                       | NEON   | 3715      | 9.03  | 9.03    |
|                       | mt (8) | 834       | 40.2  | 40.2    |
| 4096×2048 (in_proj_z) | scalar | 7422      | 2.26  | 2.26    |
|                       | NEON   | 1599      | 10.5  | 10.5    |
|                       | mt (8) | 409       | 41.0  | 41.0    |
| 2048×4096 (o/out)     | scalar | 7592      | 2.21  | 2.21    |
|                       | NEON   | 1474      | 11.4  | 11.4    |
|                       | mt (8) | 377       | 44.5  | 44.5    |
| 512×2048 (k/v)        | scalar | 932       | 2.25  | 2.25    |
|                       | NEON   | 182       | 11.5  | 11.5    |
|                       | mt (8) | 89        | 23.5  | 23.5    |
| 1024×2048 (gate_up)   | scalar | 1909      | 2.20  | 2.20    |
|                       | NEON   | 375       | 11.2  | 11.2    |
|                       | mt (8) | 126       | 33.3  | 33.3    |
| 2048×512 (down)       | scalar | 784       | 2.68  | 2.68    |
|                       | NEON   | 186       | 11.3  | 11.3    |
|                       | mt (8) | 88        | 23.9  | 23.9    |
| 256×2048 (router)     | scalar | 477       | 2.20  | 2.20    |
|                       | NEON   | 94        | 11.2  | 11.2    |
|                       | mt (8) | 69        | 15.1  | 15.1    |

Single-thread NEON is ~4–5× scalar and sits at ~9–11.5 GB/s: it is
compute-bound on the strictly-sequential fp32 add chain (4-cycle latency
hidden only 4-way by the independent row chains ≈ 1 elem/cycle) — the
deliberate price of the bitwise contract (a multi-accumulator reorder
would go faster but is NOT taken; see the decision above). Multithreaded
the kernels are memory-bound at ~23–44 GB/s (small tensors are
latency-bound below the ceiling), i.e. at/near the M1's single-socket
streaming ceiling on the big projections, which is what matters for
decode: the bitwise choice costs nothing where it counts.

## Surprises / notes for the M4 forward-pass implementer

- **No quantization step exists.** Unlike Apus's fp4 path (act-quant to
  FP8 was mandatory and lossy), these kernels take activations already in
  bf16 and produce bf16. The only rounding decisions are the two fp32
  roundings per element and the final RNE to bf16 — all pinned by the
  contract above.
- **Scratch:** NEON/mt GEMV needs K floats; GEMM/mt needs M·K floats
  (widened activations, filled by the kernel — exact, so any fill order is
  fine). Scalar paths take no scratch. Pool lanes never write scratch.
- **Row independence is total** (per-output sequential contract): a GEMV
  on `w + o0*K` computes exactly rows [o0, o0+n) of the full GEMV,
  bitwise. Slab/tiled weight streaming can rely on this.
- **M-independence is bitwise**: `apus_bf16_gemm_*` row m ==
  `apus_bf16_gemv_*` on the same row. Decode (M=1) and prefill (M>1)
  therefore agree exactly on shared prefixes wherever the surrounding
  layers also do.
- **K needs no alignment** (NEON tails are scalar, same roundings) — but
  all real Qwen K dims (2048, 4096, 512) are multiples of 32 anyway.
- **Inf/NaN inputs are handled deterministically** (IEEE propagation, edge
  tests pin the bits) but are outside the normative contract — the
  forward pass should treat non-finite activations as a bug, not a case
  to optimize.
- The raw-vs-truth error includes the bf16 output rounding (up to 2⁻⁹ of
  |out|); when comparing against an external f32/f64 reference, budget
  for it explicitly — it is part of the torch-compatible contract, not
  accumulation noise.

## Files

- `c/bf16.h` — kernels (`APUS_BF16_IMPLEMENTATION` single-TU pattern)
- `c/pool.h` — persistent pthread pool, ported verbatim from Apus (M6c)
- `tests/m3/gen_golden.py` — numpy oracle (narrow RNE + f64 GEMM truth)
- `tests/m3/test_bf16.c` — hard-gate tests
- `tests/m3/bench_bf16.c` — GEMV benchmark
- `tests/m3/golden/` — generated fixtures (via `make golden-m3` / `test-m3`)
