# tests/m9b — reorder-class re-measurement on the Qwen3.6 shapes

Milestone M9 (this adapter): the two inherited user-approved bounded
reorder classes (Ling M9b, hard-gate-3 decision 2026-08-07) —
a multi-accumulator ILP rework of the bf16 GEMV/GEMM inner loop, and an
Accelerate (AMX) dispatch for batch-M prefill — **re-measured against
FP64 truth on every real Qwen3.6-35B-A3B shape**. The approval carries
across the adapter seam only after this re-measurement
(docs/ARCHITECTURE.md §9; this suite IS that measurement). The bf16.h
scalar anchor and the bitwise NEON kernels stay the frozen contract
reference (test-m3 pins them unchanged).

Run:

```
make test-m9b    # 48 checks x APUS_THREADS=1/4/8 (digests diffed)
make ubsan-m9b   # same under -fsanitize=undefined
make bench-m9b   # dispatch timings at the Qwen shapes
```

## Lever 1: ILP bf16 GEMV/GEMM (the dispatch the engine runs)

Per output row: four float32x4 vector accumulators; per 32-wide chunk
(ascending chunk order) the eight product vectors p_0..p_7 (each one
vmulq rounding, exactly as the bitwise kernel stages them) accumulate
`acc[q & 3] += p_q` for q ascending 0..7. Row end:
`s = (acc0+acc1)+(acc2+acc3)` (vector adds), `total = (s0+s1)+(s2+s3)`
(scalar lanes); any K tail (< 32) is appended with scalar ascending
adds after the combine. Every GEMM output row computes the identical
sequence as the GEMV row (M-independence, asserted). Pool partitioning
is a fixed function of (O, K) — T=1/4/8 stays BITWISE within the path.

Measured err/esc vs FP64 truth on the Qwen shapes (m3 masked metric —
differences within 0.4% of |truth| masked as output quantization;
bound 1e-4, the approved bound):

| shape (O×K) | role | ILP masked | ILP raw |
|---|---|---|---|
| 8192×2048 | GDN in_proj_qkv / attn q_proj | 0 | 2.46e-4 |
| 4096×2048 | GDN z | 0 | 2.49e-4 |
| 2048×4096 | GDN out_proj / attn o_proj | 0 | 2.41e-4 |
| 512×2048 | attn k/v, shared g/u | 0 | 2.32e-4 |
| 2048×512 | expert/shared down | 0 | 4.66e-4 |
| 1024×2048 | fused expert gate_up | 0 | 2.42e-4 |
| 256×2048 | router | 0 | 2.38e-4 |
| 32×2048 | GDN a/b | 0 | 2.21e-4 |
| 248320×2048 | lm_head (8 slabs) | 1.09e-8 | — |
| gemm M=1..127 | 1024×2048 + 2048×512 | ≤5.07e-9 | ≤7.23e-4 |

M-independence: every GEMM row bitwise the GEMV row at M =
1/2/5/8/64/127 (asserted, 0 diffs).

**Verdict: masked err/esc = 0 on every projection shape (raw ≤ 7.2e-4,
all inside output quantization), 1.09e-8 on the lm_head — every shape
far inside the 1e-4 approved bound. NO shape falls back to the bitwise
tier. The inherited ILP approval CARRIES to the Qwen shapes.**

## Lever 2: Accelerate dispatch for M ≥ 128 (c/blas.h)

Gate-audited cutoff: the largest batched M in any pre-M9 suite is m5's
prefill_len64 (64; m4c 9, m6c 16, m7a ~32, m6a/m6b/m8 ≤ 8), so every
existing bitwise gate keeps its pinned path. bf16 operands widen EXACTLY
to fp32, one `cblas_sgemm` per fixed ≤8 MB tile over the pool (grid
depends only on (O,K): deterministic, APUS_THREADS-independent bitwise),
VECLIB_MAXIMUM_THREADS=1, single RNE bf16 rounding at the end — the
ONLY difference vs the anchor is fp32 summation order. Decode (M=1)
never dispatches. APUS_NO_BLAS=1 disables. Linked `-framework
Accelerate` on Darwin for all targets (no-op stub elsewhere).

Measured err/esc vs FP64 truth (bound 1e-4): M=128: ≤2.4e-8; M=256:
≤3.4e-8; M=512: ≤3.7e-8 (expert gate_up 1024×2048 and out-proj
2048×4096).

**Consumption status (Qwen M9): the BLAS class is measured and stays
approved, but NOTHING in the default path dispatches to it — the M9
batched prefill (tests/m9c) runs every batched linear through the
M-independent-bitwise `apus_bf16_gemm_hot` precisely so prefill ==
decode stays BITWISE at every T (the Ling M9c phase-B BLAS consumption
at M≥128 is deliberately NOT replicated). gemm_fast remains for future
phases that can tolerate the class.**

## Speed

Microbench (best-of-5, this MacBook Pro M1 Pro, mt paths at
APUS_THREADS=8; full table via `make bench-m9b`):

| shape | bitwise-mt | ILP-mt | BLAS |
|---|---|---|---|
| gemv 8192×2048 (qkv/q proj) | 0.615 ms | 0.372 ms (**1.65×**) | — |
| gemv 2048×4096 (out/o proj) | 0.308 ms | 0.154 ms (**2.00×**) | — |
| gemv 4096×2048 (GDN z) | 0.423 ms | 0.252 ms (**1.68×**) | — |
| gemv 512×2048 (k/v, shared) | 0.067 ms | 0.046 ms (**1.46×**) | — |
| gemv 2048×512 (down) | 0.060 ms | 0.042 ms (**1.43×**) | — |
| gemv 1024×2048 (expert gu) | 0.091 ms | 0.056 ms (**1.63×**) | — |
| gemv 256×2048 (router) | 0.050 ms | 0.035 ms (**1.43×**) | — |
| gemv 16384×2048 (head slab) | 1.207 ms | 0.922 ms (**1.31×**) | — |
| gemm M=128 1024×2048 | — | 2.18 ms | 0.99 ms (**2.21×**) |
| gemm M=256 1024×2048 | — | 4.46 ms | 1.56 ms (**2.85×**) |
| gemm M=512 1024×2048 | — | 9.13 ms | 2.82 ms (**3.24×**) |
| gemm M=128 2048×4096 | — | 8.53 ms | 2.89 ms (**2.95×**) |
| gemm M=256 2048×4096 | — | 17.5 ms | 4.04 ms (**4.33×**) |
| gemm M=512 2048×4096 | — | 40.9 ms | 6.01 ms (**6.81×**) |
| gemm M=128 8192×2048 | — | 18.1 ms | 5.50 ms (**3.29×**) |
| gemm M=256 8192×2048 | — | 35.3 ms | 7.88 ms (**4.47×**) |

## Adjudications

None needed: every existing suite digest is byte-identical to its
milestone value (the ILP class was already the live dispatch — m6c
measured it at 0 err/esc on these shapes; this milestone re-measures
and re-documents), and the m9c restoration consumed NO BLAS class, so
no gate moved and no margin policy was touched.

## Gates evidence

- 48 checks, 0 failures; digest `52a99b08e5d57d92` identical at
  APUS_THREADS=1/4/8 AND under UBSan.
- Full battery digests byte-identical to their milestones (m3
  `99bb4ee4e1509abb`, m4c `c441a570f6343039`, m5 `29a4612f3057cac2`,
  m6a/m6b `09fc53aee61f56c1`, m6c `ff7fed2d52a79102`, m8
  `cb171f7616055023`) — the default path did not move a bit.

## Files

- `tests/m9b/test_m9b.c`, `bench_m9b.c` — the re-anchored gates and
  timings (Qwen shapes).
- `c/bf16.h`, `c/blas.h` — unchanged (the classes themselves were
  carried at M3/M6; this milestone measures, it does not modify).

## M12 platform note (inherited)

Off-Darwin the BLAS dispatch is a no-op stub by design (c/blas.h), so
the section-3 BLAS gates collapse to a single placeholder check. The
M-independence comparisons use `apus_bf16_gemv_x86` on x86 (the hot
path there is the bitwise-sequential AVX2 kernel). Digests are
within-platform.

## M10 platform note (inherited)

The M9b ILP GEMV/GEMM kernels have a Metal twin (tests/m10): the
backend's shaders replicate the kernels' per-row rounding sequence
exactly — a BITWISE offload. This suite's digests are definitionally
untouched: the CPU binary never links the backend.
