# tests/m12 — M12: the Linux/x86_64 port battery (re-anchored to Qwen)

Ported from the Apus/Ling M12 playbook, re-anchored at M12 of this
adapter to the Qwen kernel set: the AVX2 hot set is the generic BF16
GEMV/GEMM (every dense projection, the experts, the router scoring, the
lm_head), the exact widen/narrow, and the fp32-out MoE matvec; the Qwen
GDN recurrence (c/gdn.h) and the gated-GQA row (c/attn.h) run the
**documented scalar fallback** on x86 (see "Kernel disposition" below);
the stale Ling KDA/gated-MLA AVX2 kernels are **deleted** from c/x86.h.

M12 has two halves, as in the base:

- **M12a-1**: the engine builds and passes the full portable battery on
  Linux/x86_64 (gcc, libc + pthreads only — no BLAS dependency), verified
  in a linux/amd64 Docker container on the Apple-Silicon dev host
  (Rosetta translation).
- **M12a-2**: AVX2 kernels (`c/x86.h`) for the x86-64 hot paths, runtime
  dispatched, **bitwise identical to the scalar anchors** they replace.

## Running the harness

```sh
tools/docker/test-linux.sh                 # the full portable battery
tools/docker/test-linux.sh test-m3         # selected suites
tools/docker/test-linux.sh test-m0 test-m1 # the python (unittest) suites
```

`tools/docker/Dockerfile.dev` is ubuntu:24.04 + gcc + make + python3
(numpy / safetensors / tokenizers / jinja2 for the fixture generators and
the m0/m1/m2/m7a suites). `test-linux.sh` builds the image (cached),
mounts the repo READ-ONLY at `/repo`, copies `c tests tools reference
Makefile` (NOT `weights/`) into the container's own `/src`, deletes the
copied `bin/` dirs (they hold Mach-O binaries fresh enough that make
would try to EXECUTE them), and runs `make -s PY=python3 <targets>`.
The default target list is the current portable battery: test-m2, m3,
m4a, m4b, m4c, m5, m6a, m6b, m6c, m7a, m8, m9a, m9b, m9c, m12a2 plus the
m0/m1 unittest suites (special-cased: they are `python3 -m unittest
discover`, not make targets).

Note: linux/amd64 runs under Rosetta emulation on Apple Silicon — expect
5-20x slowdowns on compute-heavy suites (the full battery is ~30+ min).

## Platform shims (the complete list)

| Site | macOS | Linux shim |
|---|---|---|
| `Makefile` | `CC := clang`, `-framework Accelerate` | `CC := gcc`, no framework; `CFLAGS += -D_GNU_SOURCE -ffp-contract=off -fno-tree-vectorize -fno-tree-slp-vectorize` (`_GNU_SOURCE`: glibc hides `pread`/`posix_memalign`/`strdup`/`clock_gettime`/`posix_fadvise` under `-std=c11`; contract=off pins FP mul+add contraction off everywhere; the no-vectorize pair works around a Rosetta -O2 mistranslation SIGTRAP — the Apus m8 finding, proven numerics-noop there, and our pinned kernels are explicit intrinsics / scalar anyway) |
| `c/compat.h apus_rss_bytes` | mach `task_info` resident_size | `/proc/self/statm` resident pages × 4096 (current RSS; `getrusage ru_maxrss` is the PEAK — fallback only). Pre-existing from the M6a port |
| `c/compat.h apus_fd_nocache` | `fcntl(fd, F_NOCACHE, 1)` | no-op returning -1 ("cached"; O_DIRECT's alignment rules are not trivially safe for the slab pread path). Pre-existing |
| `c/compat.h apus_fadvise_dontneed` | deliberate no-op (F_NOCACHE covers) | `posix_fadvise(POSIX_FADV_DONTNEED)` when `fd >= 0`. Pre-existing |
| `c/pool.h` thread count | `sysctlbyname(hw.perflevel0.physicalcpu)` | `sysconf(_SC_NPROCESSORS_ONLN)`. Pre-existing |
| `c/blas.h` | Accelerate `cblas_sgemm` for the M>=128 prefill dispatch (`APUS_BLAS == 1` only under `__APPLE__ && __ARM_NEON`) | `APUS_BLAS == 0`: no-op stubs, `apus_blas_available()` returns 0 → `apus_bf16_gemm_fast` stays on the hot kernel at EVERY M. No OpenBLAS (libc + pthreads constraint) |

Scalar fallbacks that simply engage on x86 (no edits needed): every
`#ifdef __ARM_NEON` site in `c/gdn.h` / `c/attn.h` (the Qwen GDN ops and
the gated-GQA row carry bitwise NEON variants on ARM only) and the
`c/bf16.h`/`c/moe.h` scalar anchors under the AVX2 dispatch.

## The deterministic oracle (`tools/oracle.py` `_mm`)

numpy's `@` uses BLAS (Accelerate/OpenBLAS) whose summation order varies
by platform; ~1-ulp differences flip bf16 rounding ties and top-k
selections and cascade through the random-weight fixtures. `linear()`,
the router logits, and the recurrence contractions reduce in the input
dtype (float64 throughout the oracle) with a FIXED sequential k-order
using only broadcast elementwise ops — each step is a single IEEE-exact
fp64 multiply or add, identical on every platform — so fixtures
regenerate bitwise-identically on macOS and Linux. (The base's M12b
fix, carried since the M0 oracle; the Qwen fixtures were generated with
the fixed oracle from the start.)

## Test-file adaptations (x86 anchors, same check structure)

The rule: where a suite pins `kernel == NEON` bitwise on ARM, on x86 it
pins `kernel == the scalar anchor` (M12a-1) or `kernel == the AVX2
kernel, which is itself pinned == scalar in test_m12a2` (M12a-2).

- `tests/m6c/test_m6c.c` — `test_router_matvec`: the NEON-kernel check
  slot is taken by `apus_moe_matvec_rows_x86` on x86 (same bitwise
  contract; placeholder when AVX2 is absent). Retargeted to the Qwen
  shapes (256x2048 router + odd).
- `tests/m3/test_bf16.c` — the main() probe prints the AVX2 dispatch
  state on x86; the mt/hot bitwise gates cover the AVX2 rows against
  the scalar anchor. The `#ifdef __ARM_NEON`-only sections compile out
  on x86, so the Linux check count is lower than macOS (every compiled
  check is the scalar anchor itself — the digest gates are
  within-platform).
- `tests/m9b/test_m9b.c` — the BLAS section is `#if APUS_BLAS`-only; off
  BLAS a placeholder keeps the check slot. The M-independence comparison
  uses `apus_bf16_gemv_x86` on x86. The six BLAS err/esc shape gates
  genuinely do not exist off-Darwin (lower Linux check count).
- `tests/m4a/*` — NOT adapted (Apus precedent): their `#ifdef
  __ARM_NEON` NEON-vs-scalar sections compile out on x86; the scalar
  anchors are the normative definition there and the fixture goldens
  gate them. The x86-specific bitwise coverage lives in `test_m12a2`.

## Digest policy (within-platform only) + the hot-path asymmetry

The FNV digests the Makefiles diff across `APUS_THREADS=1/4/8` are
**within-platform** invariance gates — identical across thread counts of
the same build, NOT across platforms. And by design the hot path itself
differs across platforms:

- **ARM hot path**: the M9b ILP NEON kernel — the user-approved bounded
  REORDER class (4 vector accumulators per row; err/esc vs FP64 truth
  gated in tests/m9b, bound 1e-4, re-measured on the Qwen shapes at M9).
- **x86 hot path**: the M12 bitwise-sequential AVX2 kernel — NO reorder
  class consumed; outputs are the scalar anchor's bits.

Consequence: suites whose digest streams include hot-path outputs (m4c,
m5, m6a/m6b, m6c, m8, m9a, m9b, m9c) have different digests on Linux than
on macOS — AND the Linux values coincide with what a pure-scalar macOS
run would produce. Golden-tolerance gates (m4c/m5/m8 fixture comparisons,
margin policies) pass identically because the scalar anchor is the
semantic definition on both platforms. Cross-platform bit-identity is
explicitly NOT required.

## In-container results (linux/amd64, emulated)

Full battery via `tools/docker/test-linux.sh` (17 targets: 15 make
suites + the m0/m1 unittest suites). "Linux digest" values are the
AVX2-active run:

| Suite | Linux | macOS | Notes |
|---|---|---|---|
| test-m2 | PASS (161 + 225) | PASS (same) | tokenizer/chat goldens byte-identical cross-platform |
| test-m3 | PASS (459077, `99bb4ee4e1509abb`) | PASS (459292, same digest) | count diff = NEON-only sections; every compiled check is the scalar anchor → digest identical cross-platform |
| test-m4a | PASS (38/29/33) | PASS (45/39/42) | NEON sections compile out; gdn digest `d5553a623e4b3165` differs (fp32-exact state exposes glibc-vs-libSystem expf ulps); attn `4adce6b629c1dc32` + moe `f9d28d029faabf4e` identical |
| test-m4b | PASS | PASS | oracle fixture goldens |
| test-m4c | PASS (246, `c441a570f6343039`) | PASS (246, same) | digest identical cross-platform |
| test-m5 | PASS (81, `3794b9ece894d2f8`) | PASS (81, `29a4612f3057cac2`) | Linux == the scalar anchor's bits; macOS runs the M9b ILP reorder class — within-platform digests by design |
| test-m6a/m6b | PASS (65+18 / 26+18, `0787b931bdffd61d`) | PASS (`09fc53aee61f56c1`) | store/pilot invariance BITWISE; recall counters == Python recompute (96.9% on the random fixture) |
| test-m6c | PASS (61 x T=1/4/8, `85c1568149949586`) | PASS (61, `ff7fed2d52a79102`) | same count (the AVX2 matvec check fills the NEON slot) |
| test-m7a | PASS (39) | PASS (39) | HTTP suite against the Linux binary |
| test-m8 | PASS (25, `cb171f7616055023`) | PASS (25, same) | spec == non-spec BITWISE (greedy+sampled K=2/3/4, forced patterns, tiered); digest identical cross-platform |
| test-m9a | PASS (23 x T=1/4/8, `7c5e3315e74ca441`) | PASS (23, `cab576057b380b27`) | scalar-anchor bits vs the ILP reorder class |
| test-m9b | PASS (42 x T=1/4/8, `383c214627eb8d86`) | PASS (48, `52a99b08e5d57d92`) | -6 = the non-existent BLAS shape gates |
| test-m9c | PASS (97 x T=1/4/8, `e2cd9bcbc8186d77`) | PASS (97, `30fd1b519a5c9bcc`) | batched prefill == sequential BITWISE at every T |
| test-m12a2 | PASS (263 x T=1/4/8, `fcddbb22e6330478`) | trivial pass | the AVX2 bitwise + fallback suite (below) |
| test-m0 (python) | PASS | PASS | oracle suite |
| test-m1 (python) | PASS (38 tests) | PASS (38) | converter suite |

Wall time: the full 17-target battery is ~8.5 min in-container on the
M1 Pro dev host (faster than the base's ~30 min — the Qwen synthetic
fixtures are smaller than the Ling ones).

## Excluded BY DESIGN

- **m9b's Accelerate/BLAS dispatch**: macOS system framework; on Linux
  the dispatch stubs out and prefill stays on the hot kernel at every M
  (placeholder check). No OpenBLAS — libc + pthreads only.
- **Real-model runs** (the 35B weights): the container harness excludes
  `weights/`; the portable battery is the synthetic-fixture set.
- **bench-\***: not gates. The NEON benches are ARM-only; `bench-m12a2`
  is the x86 one (EMULATED numbers below).
- **m10 (Metal)**: macOS-only by construction (the CI runs it in the
  macos job only).

## Windows

Native Windows is OUT (the engine relies on pthreads in `c/pool.h` and
POSIX file I/O — `pread`, `posix_fadvise`, `posix_memalign` — with
`c/compat.h` shims covering macOS and Linux only; a native port needs a
thread-pool + file-I/O shim layer first). **WSL2 is the supported Windows
path**: the same linux/amd64 binary, the same Docker image, and
`tools/docker/test-linux.sh` work unchanged under WSL2 (Docker Desktop
or a native dockerd), and on real x86-64 hardware the AVX2 dispatch runs
natively instead of under Rosetta.

---

# M12a-2: the AVX2 kernels (`c/x86.h`)

## Kernel disposition (the M12 re-anchor)

Per the base's rule — AVX2 consumes NO reorder class; scalar fallback is
always legal, just slower — each kernel family is either ported bitwise
== the scalar anchor or explicitly left on the scalar anchor:

| Kernel family | Disposition |
|---|---|
| widen / narrow | **AVX2, live** (exact: 16-bit shift / integer-lane RNE bit-trick; exhaustive + sweep-pinned) |
| BF16 GEMV / GEMM row bodies (+ mt/hot wrappers, `c/bf16.h`) | **AVX2, live** — shape-generic; carry every dense Qwen projection (GDN qkv 8192x2048, z 4096x2048, b/a 32x2048, out 2048x4096; GQA wq 8192x2048, wk/wv 512x2048, wo 2048x4096), the router scoring (256x2048 — `apus_moe_route` rides the gemv path), the eager + tiered expert gate_up/down (1024x2048 / 2048x512, via `c/cache.h`'s hot paths), the shared expert, and the lm_head (248320x2048) |
| MoE fp32-out matvec (`c/moe.h`) | **AVX2, live** — generic machinery (in Qwen the router rides the bf16 gemv path, NOT this matvec); pinned at kernel level regardless of model callers |
| GDN ops (`c/gdn.h`: conv1d, l2norm, the recurrence step, the gated output norm) | **documented scalar fallback on x86** — fp32-state, libm-heavy, per-element-dominated kernels; the SIMD payoff is small against the GEMV-dominated hot path and x86 is the portability target, not the perf target (dev target is Apple Silicon, where these carry bitwise NEON variants). `test_m12a2` §5 pins the fallback (the AVX2 hit counter must not move across the calls) |
| Gated-GQA row + output gate (`c/attn.h`) | **documented scalar fallback on x86** — same rationale; `test_m12a2` §6 pins it |
| Ling KDA recurrence step, gated-MLA row | **DELETED** (stale since M4 — the scalar anchors `c/kda.h` / the Ling `c/attn.h` were deleted with the Qwen retarget; the AVX2 bodies were unreachable and are now gone, not merely quarantined) |

## The contract (why bitwise, not tolerance-class)

Every AVX2 kernel is BITWISE IDENTICAL to the normative scalar kernel it
replaces:

- widen = 16-bit shift (exact, all 65536 codes pinned in test_m12a2);
  narrow = the scalar RNE bit-trick in 32-bit integer lanes (no FP
  rounding; directed IEEE specials + 1M random patterns pinned);
- per-element products 8-wide, one IEEE fp32 mul each — NO FMA (the
  scalar anchors are mul+add, two roundings);
- every reduction keeps the scalar SEQUENTIAL order per output: GEMV/GEMM
  (and the fp32-out MoE matvec) interleave independent ROW chains
  (8/4/1, the c/bf16.h chain structure).

Dispatch: per-function `__attribute__((target("avx2")))` (no global
-mavx2 — the binary still runs on baseline x86-64), runtime-gated by
`apus_x86_have_avx2()` (cached `__builtin_cpu_supports` +
`APUS_X86_DISABLE=1` escape hatch), scalar fallback otherwise. Dispatch
sites: `c/bf16.h` (mt row bodies + the hot wrappers), `c/moe.h`
(`apus_moe_matvec_f32_hot`), `c/model.h` (the lm_head widen). An activity
counter feeds the suite's "AVX2 was taken HERE" probe — and, inverted,
the §5/§6 "the fallback was taken HERE" gates.

## Emulator findings (linux/amd64 on Apple Silicon, Rosetta)

- AVX2/FMA/F16C survive translation and `__builtin_cpu_supports("avx2")`
  works in-container (bare-macOS Rosetta does NOT expose AVX2 — the
  x86_64 clang side-check on the dev host takes the placeholder branch;
  the real gate is the Docker battery).
- Accumulator ARRAYS indexed by the loop variable spill to memory under
  translation (the Apus ~3x finding): all kernels use NAMED accumulators
  (a0..a7) and named row pointers.
- `-fno-tree-vectorize -fno-tree-slp-vectorize` stay on for Linux (the
  Apus Rosetta -O2 mistranslation workaround; numerics-noop — our SIMD
  is explicit intrinsics).
- **NaN-payload carve-out** (the base's finding, re-confirmed by this
  suite's design): NaN payloads through mixed NaN arithmetic are
  codegen-dependent on x86 — gcc -O2 may commute a SIMD mul's operands
  (NaN\*NaN product payload), and the accumulator's NaN+NaN payload
  selection shifts across opt levels. Everything else — inf, subnormals,
  ±0, 0·inf → the default QNaN (bit-identical on every path),
  accumulation overflow — is bitwise identical across paths AND
  codegens. Normative inputs are finite BF16, so the carve-out is
  outside the normative domain; the suite's IEEE-specials fill excludes
  NaN codes as inputs (NaN-code coverage: the exhaustive widen + the
  narrow directed/random sweeps), and `c/x86.h`'s header documents the
  rule.

## Tests (`tests/m12/test_m12a2.c` — `make test-m12a2`)

263 checks, diffed across APUS_THREADS=1/4/8:

1. probe (cpu features: in-container `avx2=1024 fma=16384 f16c=131072`)
   + AVX2-hit counter > 0 after the battery;
2. exhaustive widen (all 65536 codes) + narrow (directed specials incl.
   RNE tie midpoints and the overflow-to-inf boundary, then 1M random
   FP32 bit patterns) — bitwise vs the scalar helpers;
3. GEMV/GEMM/mt/hot == scalar BITWISE on the shape sweep (odd tails,
   chain boundaries, the Qwen shapes 8192x2048 / 2048x4096 / 512x2048 /
   1024x2048 / 2048x512 / 32x2048 / 256x2048), an IEEE-specials fill,
   the 248320x2048 head slab-wise (hot == scalar), M-independence,
   FP64-truth err/esc (masked m3 class, bound 1e-4 — measured 0 on
   every gated shape);
4. MoE router matvec hot == scalar bitwise (256x2048 router-class +
   odd shapes);
5. GDN recurrence step (H=32, Dk=Dv=128 — the real dims): the AVX2 hit
   counter must NOT move across the calls (the documented scalar
   fallback is what executes; no stale AVX2 kernel is reachable),
   step_mt == step BITWISE, repeat-run BITWISE;
6. gated-GQA decode (H=16, Hkv=2, D=256, Tk=300): same fallback gates,
   decode_mt == decode BITWISE.

Off x86-64 the suite compiles to a trivial pass (the macOS battery keeps
the target list platform-uniform); on non-AVX2 x86-64 the AVX2-direct
checks skip with a placeholder (the scalar battery still gates the
fallback paths).

## Bench (in-container — EMULATED under Rosetta, indicative only; real
x86-64 hardware numbers will differ)

`make bench-m12a2`, APUS_THREADS=1, per-call times:

| Kernel | scalar | AVX2 | speedup |
|---|---|---|---|
| gemv 8192x2048 (GDN qkv / GQA wq) | 26.1 ms | 7.3 ms | 3.6x |
| gemv 2048x4096 (GDN out / GQA wo) | 13.2 ms | 3.4 ms | 3.9x |
| gemv 1024x2048 (expert gate_up) | 3.3 ms | 0.85 ms | 3.9x |
| gemv 2048x512 (expert down) | 1.6 ms | 0.43 ms | 3.6x |
| gemv 248320x2048 (LM head) | 797.7 ms | 222.8 ms | 3.6x |
| gdn step H=32 D=128 | 1.42 ms (scalar fallback — no AVX2 port) | | |
| gqa decode H=16 d=256 Tk=4096 | 51.7 ms (scalar fallback — no AVX2 port) | | |

(the gemv columns are the direct scalar vs `apus_bf16_gemv_x86` calls;
the GDN/GQA rows time the dispatched path, which on x86 IS the scalar
anchor — the documented fallback.)

## macOS re-verification

`c/x86.h` compiles to nothing off x86-64 (`APUS_X86 == 0`); the dispatch
sites are `#ifdef __ARM_NEON / #elif APUS_X86` — the ARM branches are
textually untouched. Full macOS battery after the re-anchor: all suites
green, every digest byte-identical to the M10 values (m3
`99bb4ee4e1509abb`, m4c `c441a570f6343039`, m5 `29a4612f3057cac2`, m6a
`09fc53aee61f56c1`, m6c `ff7fed2d52a79102`, m8 `cb171f7616055023`, m9a
`cab576057b380b27`, m9b `52a99b08e5d57d92`, m9c `30fd1b519a5c9bcc`, m10
`3fbe7d90787aaa8c` / `3d00af73bcbf48f0`); test-m12a2 is the trivial pass
there.
