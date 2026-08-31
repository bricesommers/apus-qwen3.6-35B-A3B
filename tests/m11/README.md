# tests/m11 — real-model validation (Qwen3.6-35B-A3B, 66.97 GiB container)

M11 turns "fixtures green" into "the real model is proven". Everything
here ran against `weights/apus-qwen/` (format v2, 10,496 expert slabs,
READ-ONLY) on the dev Mac (M1 Pro, 32 GB) on 2026-08-30. Raw logs:
`logs/<name>.out` / `logs/<name>.err`.

## The standing golden

Golden command (the sanity gate for every future engine change):

```sh
./bin/apus-qwen run --model weights/apus-qwen --tiered \
    --prompt "The capital of France is" --max-tokens 24 --greedy
```

- **`golden.txt`** — the exact emitted bytes (24 greedy tokens, thinking
  mode, truncated mid-sentence by --max-tokens):

  ```
  Here's a thinking process:

  1.  **Analyze User Input:** The user asks "The capital of France
  ```

- **`golden.ids.json`** — the exact token stream: 15 prompt ids (ChatML,
  thinking on) + 24 generated ids, captured through the engine's own
  `serve` NDJSON encode/generate path; concatenated per-token text
  re-decodes byte-identical to `golden.txt`.
- **Reference stats** (`logs/golden.err`): load 2.22s; prefill 15 tok in
  4.65s (3.2 tok/s); decode 24 tok in 11.31s (2.1 tok/s); pilot recall
  **5960/7488 = 79.6%**; store hits 3679 / misses 6174 / **preads 7320,
  all pilot hints, 0 demand loads, 0 rss_drops**, waits 1088. Peak RSS
  8.29 GiB (`/usr/bin/time -l`, separate identical-stream run).

**GATE**: any future engine change must reproduce `golden.txt`
**bitwise** under this command (any thread count, any cache size, any
pilot K, spec on/off, Metal on/off). `check_golden.sh` re-runs and
diffs. Wall times and store stats are informational (usage-file hot
pinning shifts them); the stream is the gate. The only legal exceptions
are the documented reorder classes in the relevant milestone READMEs.

## Determinism on the real model (all streams BITWISE == golden)

| run | prefill | decode 24 tok | notes |
|---|---|---|---|
| golden (APUS_THREADS default 10) | 4.65s | 11.31s | reference |
| repeat | 4.65s-class | 11.31s | identical stream + identical recall |
| APUS_THREADS=1 | 6.28s | 16.91s | thread-independence gate, real model |
| APUS_THREADS=4 | — | 11.43s | identical |
| APUS_THREADS=8 | — | 9.80s | identical |
| APUS_EXPERT_CACHE_MB=512 | 4.02s | 8.30s | preads 10346 vs 7320, **0 demand loads** — costs-speed-not-quality proven on the real model |
| APUS_RSS_GUARD_MB=20000 | 4.72s | 9.14s | rss_drops 0 (peak RSS 8.29 GiB never nears 20 GB) |
| APUS_RSS_GUARD_MB=10000 | 4.53s | 9.54s | rss_drops 0 |
| APUS_RSS_GUARD_MB=8000 | 4.55s | 9.69s | **rss_drops 1980**, 0 demand loads — the guard fires on the real model, stream still bitwise golden |

Pilot recall was 5960/7488 in EVERY golden-command run — the pilot is
deterministic.

## Pilot-K A/B (real model)

`--prompt "Explain photosynthesis" --max-tokens 64 --greedy`, default
threads, two runs per K for K=6/8 (repeat walls in parentheses). All
streams bitwise identical across K (hints never affect numerics).

| K | decode wall, 64 tok | recall | waits | preads |
|---|---|---|---|---|
| 6 | 23.21s (23.70s) | 13492/19968 = **67.6%** | 3326 | 13306 |
| 8 | 25.25s (25.56s) | 16240/19968 = **81.3%** | 2667 | 14852 |
| 12 | 32.78s | 18405/19968 = **92.2%** | 2111 | 20473 |

Verdict: **APUS_PILOT_K=8 stays the default.** K=6 is ~8% faster on this
quiet-machine window but gives up 13.7 points of recall — that margin is
the insurance against demand stalls when RAM is tight (rss guard / tiny
cache). K=12 buys recall at +30% wall (deeper speculation trades bytes
for waits at a negative rate — the Ling/Apus regime repeats on Qwen).
0 demand loads at every K.

## Backend A/Bs (golden command, greedy; streams BITWISE == golden)

### MTP speculative decoding (`--spec`)

| mode | decode wall | tok/s | accepted | tok/batch |
|---|---|---|---|---|
| non-spec | 11.31s | 2.12 | — | — |
| `--spec --spec-k 2` | 22.35s | 1.07 | 8/16 drafts = 50.0% | 1.50 |
| `--spec --spec-k 4` | 21.84s | 1.10 | 15/27 = 55.6% | 2.67 |

Verdict: **spec stays OFF** — ~1.9× slower, same class as the Ling
finding. Qwen's trained MTP does lift acceptance (55.6% at K=4, 2.67
tok/batch, vs Ling's 42.3%/1.85 at K=3), but decode is disk/DRAM-bound:
the verify batch reads MORE expert slabs and each draft pays the full
MTP layer + shared lm_head. Exactness holds: both streams bitwise golden.

### Metal (`bin/apus-qwen-metal --metal`)

prefill 4.99s / decode 10.93s (2.2 tok/s) vs CPU 4.65s / 11.31s —
**parity**, stream bitwise golden. The real-model A/B confirms the
fixture verdict: expert-I/O-bound, so the GPU changes nothing. Stays
opt-in (`--metal` / `APUS_METAL=1`).

### BLAS prefill (`APUS_NO_BLAS`)

1450-token prompt (`logs/long_prompt.txt`), `--max-tokens 2`:
prefill **59.79s (24.3 tok/s)** default vs **59.34s (24.4 tok/s)** with
`APUS_NO_BLAS=1`; streams bitwise identical. This is the expected NO-OP:
per the M9c decision the Accelerate class is measured-and-approved but
UNCONSUMED — no production call site dispatches to `apus_bf16_gemm_fast`
(prefill runs the M-independent-bitwise ILP GEMM at every T, so
prefill == decode is bitwise everywhere). The real-model A/B confirms
the design; `APUS_NO_BLAS` currently changes nothing.

## Per-token expert traffic (real model)

Golden run: 7320 preads × 6,291,456 B/slab = **42.9 GiB over 39 tokens
(15 prefill + 24 decode) ≈ 1.10 GiB/token** — vs the Phase A worst-case
estimate of 1.9 GiB/token (320 cold experts × 6 MiB). Pilot-hit reuse +
prefill batch-union halves cold traffic. 0 demand loads in every run
above (the pilot + demand-boost queue hide all I/O at default budgets).

## Not done (open items)

- External R1-style validation vs a hosted reference (needs a user API
  key — out of scope; no key was sought or used).
- `apus.usage` hot-pinning makes wall times drift between sessions;
  the stream gate is immune to it.
