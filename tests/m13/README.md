# tests/m13 — Windows port (MinGW-w64/UCRT64) + tri-platform CI

M13 ports the **Apus M15 Windows support** across the adapter seam
(donor: `../Apus`, the DeepSeek-V4-Flash M15 engine — READ-ONLY; shim
code copied with attribution, POSIX call sites map 1:1). The Ling base
this repo was seeded from had no Windows port (Phase-0 engine-debt
flag); the engine has since diverged from the donor (GDN/attn/moe
kernels, v2 container, batched prefill), so M13 ports the **shim layer
and CI only** — no model code.

Unlike every other milestone, M13 has no local C test binary: the dev
machine is macOS and the Windows path cannot be compiled or run here.
**The M13 gate is (a) this review matrix, (b) the full macOS battery
byte-identical before/after the port (POSIX paths must not move), and
(c) the `windows` CI job — the real green lands on the first CI run
after push** (pushing is user-gated). This README is the record of
(a) and the checklist for (c).

## What was ported (per file)

- `c/compat.h` — the M15 `_WIN32` static-inline shim block, verbatim
  from the donor: `apus_ncpu` (GetSystemInfo), `apus_aligned_alloc` /
  `apus_aligned_free` (`_aligned_malloc`/`_aligned_free` — storage that
  must NOT pass through plain `free()`), `apus_sys_open_ro`
  (CreateFileA + FILE_FLAG_OVERLAPPED + `_open_osfhandle`, `_O_BINARY`),
  `apus_sys_fsize` (`_fstati64`), `apus_sys_fsync` (`_commit`),
  `apus_sys_rename` (MoveFileExA + MOVEFILE_REPLACE_EXISTING),
  `apus_sys_mkdir_p`, `apus_sys_pread` (ReadFile + per-call OVERLAPPED,
  1 GiB chunks, ERROR_HANDLE_EOF → short read). RSS: psapi
  `GetProcessMemoryInfo` WorkingSetSize (current, not peak) added as the
  `_WIN32` branch of `apus_rss_bytes`; the getrusage fallback include is
  POSIX-guarded. Non-Windows builds map every shim 1:1 onto the POSIX
  calls (open/fstat/pread/posix_memalign/free/fsync/rename/sysconf).
- `c/st.h` — all `open`/`fstat`/`pread` call sites converted to
  `apus_sys_open_ro` / `apus_sys_fsize` / `apus_sys_pread` (same edit as
  the donor's st.h; this file was still line-identical to the donor's
  pre-M15 ancestor). The F_NOCACHE twin fd keeps its existing
  `__APPLE__`/`#else` guard (no F_NOCACHE equivalent on Windows → the
  twin closes and reads stay cached, like Linux).
- `c/pool.h` — thread count via `apus_ncpu()` off-Apple; the scratch
  arena's segment alloc/free pairs go through `apus_aligned_alloc` /
  `apus_aligned_free` (both the fresh-segment and the grow-in-place
  paths — a mismatched pair is the M15 heap-corruption bug class).
- `c/cache.h` — `apus_slab_alloc` (4096-aligned slabs) →
  `apus_aligned_alloc`; EVERY slab-buffer release paired with
  `apus_aligned_free` (buf_free recycle list, RSS-guard drops, store
  close: slots/pins/wait-slots/buf_free); the usage-history save uses
  `apus_sys_fsync` + `apus_sys_rename` (Windows rename() fails with
  EEXIST on an existing destination — M15 bug #2).
- `c/apus-qwen.c` — `_setmode(_fileno(stdin|stdout), _O_BINARY)` at
  `main()` entry under `_WIN32` (MSVCRT text mode would corrupt the
  NDJSON serve protocol with \r\n).
- `Makefile` — `ifeq ($(OS),Windows_NT)` branch (donor-shaped):
  `-std=gnu11` (STRICT_ANSI hides strdup/clock_gettime), no
  `_GNU_SOURCE`, the same `-fno-tree-vectorize` pins as Linux,
  `LDLIBS += -lpsapi -static` (psapi for GetProcessMemoryInfo; -static
  bundles winpthreads — libwinpthread-1.dll is not reliably on PATH,
  silent exit 127 when missing). The engine link rules gain an explicit
  `-lpthread` (no-op on macOS/Linux, required by winpthreads).
- `tools/chat.py` — pins `encoding="utf-8", errors="replace"` on the
  engine pipe (Windows Python defaults to cp1252).
- `tests/m2/gen_golden.py` — golden `.txt`/manifest writes with
  `newline=""` (Windows text mode writes \r\n; the C tests byte-compare)
  and JSON/template reads with `encoding="utf-8"`.
- `.gitattributes` (new) — `* -text` (autocrlf CRLF'd the m2
  conformance pairs on checkout in the donor's M15; same hazard here).
- `.github/workflows/ci.yml` — the `windows` job (below).

## POSIX-ism review matrix (whole `c/` tree swept)

| POSIX-ism | Sites | Windows disposition |
|---|---|---|
| `pread` | `c/st.h` (5 call sites) | `apus_sys_pread` (overlapped ReadFile) |
| `open(O_RDONLY)` | `c/st.h` (3, incl. F_NOCACHE twin) | `apus_sys_open_ro` (overlapped, `_O_BINARY`) |
| `fstat` | `c/st.h` (1) | `apus_sys_fsize` (`_fstati64`) |
| `posix_memalign` | `c/cache.h` slab alloc | `apus_aligned_alloc` (`_aligned_malloc`) |
| `aligned_alloc`/`free`+`malloc` pairing | `c/pool.h` scratch arena | `apus_aligned_alloc`/`apus_aligned_free` pairs |
| `sysconf(_SC_NPROCESSORS_ONLN)` | `c/pool.h` | `apus_ncpu` (GetSystemInfo) |
| `fsync(fileno(f))` | `c/cache.h` usage save | `apus_sys_fsync` (`_commit`) |
| `rename` (replace semantics) | `c/cache.h` usage save | `apus_sys_rename` (MoveFileExA) |
| `fcntl(F_NOCACHE)` | `c/compat.h`, `c/st.h` twin | `__APPLE__`-guarded already; `#else` branch = cached reads (like Linux) |
| `clock_gettime(CLOCK_MONOTONIC)` | `c/cache.h`, `c/apus-qwen.c` | MinGW-w64 provides it (`-std=gnu11` exposes it) |
| `strdup`, `fileno`, `remove` | `c/json.h`, `c/st.h`, `c/cache.h` | provided by UCRT; strdup exposed by `-std=gnu11` |
| `<unistd.h>` / `<sys/stat.h>` / `<fcntl.h>` includes | `c/st.h`, `c/cache.h`, `tests/m6a/test_store.c` | MinGW-w64 ships compatible subsets (donor-proven) |
| `sysctlbyname`, mach `task_info` | `c/pool.h`, `c/compat.h` | `__APPLE__`-only already |
| `getrusage` | `c/compat.h` (RSS fallback) | POSIX-guarded by the port; Windows uses psapi |
| pthreads (`pthread_*`, cond vars) | `c/pool.h`, `c/cache.h`, `c/pilot.h` | winpthreads (MSYS2 gcc, `-lpthread -static`) |
| `pthread_set_qos_class_self_np` | `c/pool.h` | `__APPLE__`-guarded already |
| stdio text mode on the serve protocol | `c/apus-qwen.c` | `_setmode(..., _O_BINARY)` under `_WIN32` |
| `nanosleep` in tests | `tests/m6a/test_store.c`, `tests/m6b/test_pilot.c` | already ≥ 1 ms polls (the M15 sub-ms-rounding lesson) |
| Accelerate / `c/blas.h` | Makefile Darwin-only link | no-op stub off-Darwin (Linux CI proves the stub; Windows takes the same path) |
| Metal (`c/backend_metal.mm`) | Makefile Darwin-only targets | stubbed off-Darwin; excluded from the Windows battery |
| `c/x86.h` AVX2 intrinsics | x86_64 only | MinGW gcc on x86_64 = the same code path Linux CI runs |
| `mmap`, `fork`, `signal`, `alarm`, `select` on fds | — | NOT USED anywhere in `c/` (mmap deliberately avoided in `c/st.h` by design) |
| `system("mkdir -p")` in tests | — | none in this repo (the donor converted five; nothing to port) |
| `select()` on pipes in Python tooling | — | not used (server/chat use blocking text pipes; the M15 select-on-pipe bug class has no call site here) |

## CI workflow shape (`.github/workflows/ci.yml`)

Three jobs: `linux` (ubuntu-latest, gcc), `macos` (macos-latest, clang +
the Metal suite `test-m10`), `windows` (windows-latest, MSYS2 UCRT64,
MinGW-w64 gcc; native Windows Python 3.12 for the fixture oracles via
`path-type: inherit`; `PYTHONUTF8=1`; `update: false` on setup-msys2 —
full upgrades kill their own shell on GH runners; the 2026 pacman
package name `mingw-w64-ucrt-x86_64-gcc`). The Windows battery is the
same 15-target list as linux (`test-m2 … test-m12a2` + the m0/m1
unittest suites) minus nothing else: `test-m10` was never in the
portable list (Metal is macOS-only) and the `ubsan-*` targets are not
part of CI on any OS (MinGW has no sanitizer runtimes — the donor's
exclusion, moot here since CI never ran them).

## Verification status

- macOS battery m2–m12a2 + test-m10 + m0/m1 unittests: run BEFORE and
  AFTER the port; every digest byte-identical (see the M13 entry in
  `docs/STATUS.md`). `make all` warning-free.
- The Linux path is untouched (no shared-code change outside `_WIN32`
  guards + the proven digest-neutral shim substitutions), so the Docker
  battery was not re-run.
- **Unverifiable until the first real Windows CI run** (the honest
  boundary of M13): the MinGW compile itself (warning surface, gnu11
  feature exposure), winpthreads pool behavior under load, overlapped
  pread correctness/performance on real shard files, the RSS guard via
  psapi, fixture regeneration under native Windows Python, and any
  Windows-only runtime bug of the M15 class that review cannot see
  (the donor found three only by running: heap pairing, rename
  semantics, select-on-pipe — the first two have call sites here and
  are shimmed; the third has none).
