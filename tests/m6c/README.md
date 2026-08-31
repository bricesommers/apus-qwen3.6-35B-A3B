# tests/m6c — hot-path equivalence (Qwen3.6 re-anchor)

Milestone M6c, re-anchored for the Qwen3.6 adapter: the hot kernels on
the REAL Qwen shapes stay inside their numerics class, and the tiered
forward is BITWISE the eager one across cache sizes AND I/O thread
counts. All threading is fixed-partition + per-row accumulation (the m3
mt contract), so results are thread-count independent **bitwise** (this
test's digest is diffed across APUS_THREADS=1/4/8 by the Makefile).

Scope note vs the Ling base's M6c: this adapter's forward processes
every token through the ONE per-token body (c/layer.h M4 contract) —
the Ling batched-prefill machinery was removed at M4 and is deferred to
this adapter's own performance milestones (M9). What M6 re-anchors here
is therefore the hot-kernel class gates on Qwen shapes + the
tiered==eager gate at the model level, not a batching gate (there is no
batching to be M-independent about yet; when M9 adds batching it must
be M-independent-bitwise, same discipline as the base).

Run:

```
make test-m6c        # builds, runs APUS_THREADS=1/4/8, diffs digests
make ubsan-m6c       # same under -fsanitize=undefined
```

## Test coverage (61 checks, `test_m6c.c`)

- `gemv_hot` on the real projection shapes (GDN qkv 8192×2048, z
  4096×2048, out 2048×4096, b/a 32×2048; expert gate_up 1024×2048,
  down 2048×512; shared 512×2048 / 2048×512; router 256×2048) and odd
  tails, plus the real lm_head slab 248320×2048 spot rows — gated by
  the m3 masked err/esc metric vs in-test FP64 truth (bound 1e-4): the
  dispatch runs the **inherited user-approved M9b ILP NEON kernel**
  (the Ling base's documented bounded reorder class, carried across the
  adapter seam; its re-approval on Qwen shapes is M9's business —
  measured max err/esc here: **0**).
- `gemm_hot`, M sweep (M=8/9/3/16/2) — same err/esc class.
- `matvec_f32 NEON/hot == scalar` BITWISE (256×2048 router, 16×128
  fixture router, odd).
- `gdn_step_mt == gdn_step` BITWISE (H=32/D=128 real + odd shapes,
  random states, T=6).
- `gqa_mt == gqa` BITWISE at the real gated-GQA shape (16 q heads, 2 KV
  heads, head_dim 256, T=64) + decode == full row.
- **Tiered == eager at the model level** (T=16 one-shot prefill vs 16
  decode steps): logits AND layer state (S, conv states, KV caches)
  bitwise in eager mode; tiered one-shot AND tiered sequential ==
  eager logits at cache sizes {16 slots, 2 slots} × I/O pool
  {sync, 1, 4, 8} — the I/O pool never perturbs compute bits.
- Scratch-arena LIFO smoke (dead-segment grow + reuse).

Digest `ff7fed2d52a79102` identical at APUS_THREADS=1/4/8 and under
UBSan.

## Result

`make test-m6c`: 61 checks, 0 failures; gemv_hot ILP max err/esc 0
(bound 1e-4); thread-independence diffs clean. `make ubsan-m6c` clean.

## Digest evidence (this adapter's battery)

| suite | digest | status |
|---|---|---|
| m3 | 99bb4ee4e1509abb | green (459,292 checks) |
| m4c | c441a570f6343039 | green (246 checks) |
| m5 | 29a4612f3057cac2 | green (81 checks) |
| m6a/m6b invariance | 09fc53aee61f56c1 | green (18+18 checks) |
| m6c | ff7fed2d52a79102 | green (61 checks) |

## Platform notes

On Linux/x86_64 the hot dispatch is the M12a-2 bitwise-sequential AVX2
kernel (c/x86.h), NOT the ILP reorder kernel — the err/esc gates above
are path-agnostic and hold unchanged, and digests are within-platform
(M12 re-anchors). `test_router_matvec` pins `matvec_f32 AVX2 == scalar`
in the NEON check's slot (same check count on both platforms;
placeholder when AVX2 is absent).
