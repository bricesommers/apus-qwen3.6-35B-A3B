# tests/m9a — strictly-bitwise perf gates + the M9 forward profile

Milestone M9 (this adapter): re-anchor the Ling M9a suite to the
Qwen3.6 engine — re-run the bitwise hot-kernel wiring checks on the
Qwen shapes, and profile the CURRENT forward (glue vs kernel
breakdown) on the Qwen fixture model. No kernel changes ship from this
suite (c/bf16.h is outside the M9 edit scope); it measures and
documents.

Run:

```
make test-m9a     # 23 checks x APUS_THREADS=1/4/8 (digests diffed)
make ubsan-m9a    # same under -fsanitize=undefined
make bench-m9a    # the 4-chain vs 8-chain GEMV microbench (Qwen shapes)
make profile-m9a  # the fixture-forward profile driver (sample(1) target)
```

## Step 1: the profile (fixture model + real-dim components)

The real model can't be profiled yet (weights are M11), so the profile
comes in two honest halves:

**(a) Fixture forward under sample(1)** — `tests/m9a/profile_sample.txt`:
`profile_m9a` loops 32-token prefill + 96-token decode episodes on the
m5 Qwen fixture (2 layers, hidden 128, 16 experts, vocab 512), 10 s
window, APUS_THREADS=8, ~26k samples over all threads. Named compute
frames (pool-idle `__psynch_cvwait` excluded — the fixture GEMVs are
microseconds, so the pool lanes dominate the wall):

| frame | samples | share of named compute | what |
|---|---|---|---|
| `apus_bf16_gemv_ilp_rows_job` | 403 | 58% | decode GEMV chains (ILP dispatch) |
| `apus_attn_gqa_row_neon` | 65 | 9.4% | FULL attention rows |
| `apus_bf16_gemm_ilp_rows_job` | 53 | 7.7% | prefill GEMMs (the M9 batched path) |
| `apus_gdn_step_head_neon` | 48 | 6.9% | GDN recurrence |
| `apus_pool_worker`+`apus_pool_run` | 57 | 8.2% | dispatch machinery |
| **glue** (conv1d 14, rmsnorm 10, onorm 6, expf 19, layer 5) | 54 | **7.8%** | the serial glue |

The glue share here (7.8%) is an UPPER bound: the fixture's hidden is
128, so the O(H) glue is artificially large next to the O(H²) GEMVs.
At the real dims the kernel share dominates by construction (hidden
2048: 16× wider GEMVs, same-width norms).

**(b) Real-dim component profile** — `tests/m9c/profile_phasea.c`
(synthetic REAL Qwen dims, T=512): the GDN layer's serial parts
(conv/decay/beta/l2norm/step/onorm — everything that is NOT a
projection GEMM) are 0.085 s vs 0.704 s for the whole per-token mixer
(**12%**), and the projections are 88%. Per-token glue inside the
mixer (norms, elementwise) is below the per-op timing floor.

**Conclusion (same shape as the Ling M9a <0.25%-glue verdict, restated
for Qwen):** the serial glue is not a lever; the GEMV/GEMM chains and
the batching structure around the GDN recurrence are. M9's lever was
therefore structural (tests/m9c), not glue surgery.

## Step 2: the bitwise wiring checks (re-anchored to Qwen shapes)

23 checks, 0 failures; digest `cab576057b380b27` identical at
APUS_THREADS=1/4/8 and under UBSan:

- `gemv_hot` (the live ILP dispatch) err/esc vs in-test FP64 truth,
  bound 1e-4, at every real Qwen shape: 8192×2048, 4096×2048,
  2048×4096, 1024×2048, 2048×512, 512×2048, 256×2048, 32×2048, plus
  the 8/4/1-row tail boundaries ({4,5,8,9,12,13}×33, {1,3}×1, 17×100)
  and the 248320×2048 lm_head — all inside the bound (see tests/m9b
  for the full table).
- The frozen bitwise 8-chain kernel (`apus_bf16_gemv_neon`) == the
  scalar anchor BITWISE at a partial-chain boundary (O=20: 2×8 + 1×4
  groups, K=2048).
- Logits widen path: SIMD widen == scalar widen (exact) at V=248320.

## Microbench (make bench-m9a, single thread, bitwise-staged GEMV
4-chain vs the shipped 8-chain, Qwen shapes)

| shape | 4-chain | 8-chain | speedup |
|---|---|---|---|
| 8192×2048 (qkv proj) | 4.6–5.0 ms | 3.6–3.7 ms | 1.25–1.38× |
| 16384×2048 head slab | 10.97 ms | 7.98 ms | 1.38× |

bitwise: IDENTICAL (the 8-chain form is what c/bf16.h ships since the
Ling M9a; re-measured here on the Qwen shapes).

## Fixture-level numbers (pilot/bench harness, requirement: bench
targets run on the fixture)

- `make profile-m9a`: 2000 episodes (256,000 tokens, 25% prefill / 75%
  decode) in 154.7 s — **1655 tok/s** on the m5 fixture (threads 8).
- `make bench-m6a`: pread 1.99 GB/s (cached) / 3.12 GB/s (F_NOCACHE);
  hit-rate curve 85.5% @ 16 slots → 45.9% @ 8 → 23.6% @ 4 → 11.4% @ 2
  (843 → 1300 tok/s on the fixture).
- Pilot recall harness (tests/m6b, `make test-m6b`): live counters ==
  Python recompute EXACTLY (96.9% on the random fixture — machinery
  gated, value is fixture noise; real-model pilot-K tuning stays M11).

## Files

- `tests/m9a/test_m9a.c` — the re-anchored bitwise gates.
- `tests/m9a/bench_m9a.c` — the 4/8-chain A/B at Qwen shapes.
- `tests/m9a/profile_m9a.c` — the fixture-forward profile driver.
- `tests/m9a/profile_sample.txt` — the raw sample(1) output.
