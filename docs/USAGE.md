# apus-qwen — terminal usage

Qwen3.6-35B-A3B (35B), fully local on this Mac. Nothing leaves the machine.

## Interactive chat

```bash
cd ~/Desktop/AI-PROJECTS/Apus-Qwen3.6-35B-A3B
.venv/bin/python tools/chat.py --model weights/apus-qwen --tiered
```

In-chat commands:

| command | effect |
|---|---|
| `/quit` | exit (Ctrl-D also works) |
| `/reset` | clear conversation history |
| `/system <text>` | set a system message |
| `/thinking on\|off` | thinking mode (default on; off = direct answers) |
| `/preserve on\|off` | keep thinking in ALL history turns (default off) |
| `/temp 0` | greedy/deterministic (default 1.0; top_p 0.95, top_k 20) |
| `/max 2000` | max tokens per reply (default 2048) |
| `/raw <text>` | raw completion for this turn (no chat template) |
| `/help` | show all commands |

**Reply budget vs thinking:** the per-reply cap (default **2048**
tokens) counts thinking + answer together, and thinking is spent FIRST.
A long-form request (trip planning, code, essays) can burn the whole
budget on reasoning and stop with the `[N tok, length]` marker before
the visible answer starts. That marker means the reply budget ran out —
NOT the context window (32,768, shared by the whole conversation).
Raise it in-session with `/max 8192` or at launch:

```bash
.venv/bin/python tools/chat.py --model weights/apus-qwen --tiered --max-tokens 8192
```

Trade-off: at ~2–4 tok/s, 8,192 tokens ≈ 35–40 min of generation. For
quicker long-form answers, ask the model to "answer directly, keep it
concise" — that shrinks the thinking phase. `/thinking off` skips it
entirely.

## One-shot generation

```bash
cd ~/Desktop/AI-PROJECTS/Apus-Qwen3.6-35B-A3B
./bin/apus-qwen run --model weights/apus-qwen --tiered \
    --prompt "Your prompt here" --max-tokens 100 --greedy
```

`--greedy` = deterministic argmax (or `--temp/--top-p/--top-k/--seed`).
Add `--spec` to try MTP speculative decoding (exact — the emitted
stream is bitwise the non-spec one — but measured ~1.9× SLOWER on the
real model: decode is disk-bound, so the verify batch's extra expert
reads dominate even at 55.6% draft acceptance, K=4; off by default,
see tests/m11). On macOS, `make metal` builds
`bin/apus-qwen-metal`; add `--metal` for the GPU backend (bitwise-equal
output — the shaders replicate the dispatched CPU kernels' rounding
sequences exactly, gated in tests/m10 AND on the real model in
tests/m11 — but parity speed on unified memory, so it's off by
default).

## Linux / Windows

The engine builds and runs on Linux/x86_64 (AVX2, runtime-dispatched,
bitwise-equal to the scalar kernels; scalar fallback on older CPUs. The
dense GEMV/GEMM hot paths — projections, experts, router, lm_head — are
the AVX2 kernels; the GDN recurrence and the gated-GQA row run the
scalar anchors on x86, a documented fallback, see c/x86.h):

```bash
make bin/apus-qwen        # gcc; Linux flags are automatic
./bin/apus-qwen run --model weights/apus-qwen --tiered --prompt "..." --greedy
```

On Windows there are two routes. **WSL2** (Ubuntu): the Linux
instructions apply as-is. **Native Windows** (M13, the Apus M15 shim
layer): MSYS2 UCRT64 with `mingw-w64-ucrt-x86_64-gcc` + `make` — the
Makefile's Windows branch is automatic (`-std=gnu11`, `-lpsapi
-static`, the `c/compat.h` `_WIN32` shims: overlapped-I/O pread,
`_aligned_malloc`, MoveFileEx rename, winpthreads). The full portable
test battery runs in the CI `windows` job; the Metal backend and the
`ubsan-*` targets are macOS/POSIX-only. `tools/docker/test-linux.sh`
runs the full test battery in a container (any OS with Docker). The
weights container is platform-independent — copy `weights/apus-qwen/`
or rebuild it with `tools/download.py`.

## OpenAI-compatible server (for other apps)

```bash
cd ~/Desktop/AI-PROJECTS/Apus-Qwen3.6-35B-A3B
.venv/bin/python tools/server.py --model weights/apus-qwen --tiered --port 8080
```

Then point any OpenAI client at `http://localhost:8080/v1` (any non-empty
API key). Endpoints: `/v1/chat/completions` (SSE streaming,
`reasoning_content`, tool calls), `/v1/completions`, `/v1/models`,
`/health`. Quick check:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "apus-qwen", "stream": true,
  "messages": [{"role": "user", "content": "Hello"}]}'
```

Thinking is on by default; per request:
`"chat_template_kwargs": {"enable_thinking": false}` (the model card's
API; `"preserve_thinking": true` keeps reasoning in all history turns).
Tool calls: pass OpenAI `tools`; the model's XML `<tool_call>` output is
translated back to OpenAI `tool_calls` (use non-streaming for clean
parsing; in streaming mode the markup arrives raw in deltas).

Sampling defaults follow the model's generation_config (temperature 1.0,
top_p 0.95, top_k 20). `presence_penalty` is accepted (OpenAI/vLLM
semantics: penalizes tokens already generated at least once) but
**defaults to 0.0 = OFF**; the model card's recommended 1.5 for thinking
mode is a pending opt-in decision — pass it explicitly if you want it.

Context and length: `serve` runs with **max_seq 32768** by default
(override with `--max-seq N` or `APUS_MAX_SEQ`; KV cache 671 MB at the
default). If `max_tokens` is omitted the engine generates until EOS or
the context fills (OpenAI semantics).

### Open WebUI (browser chat)

```bash
# one-time
docker pull ghcr.io/open-webui/open-webui:main
docker run -d --name open-webui -p 3000:8080 \
  -e ENABLE_OLLAMA_API=false \
  -e OPENAI_API_BASE_URLS='http://host.docker.internal:8000/v1' \
  -e OPENAI_API_KEYS='apus' \
  -v open-webui:/app/backend/data --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
# later: docker start open-webui   (stop: docker stop open-webui)
```

Then open http://localhost:3000 (first signup = local admin) and select
`apus-qwen`. **Required one-time fix**: Open WebUI injects ~7K tokens of
built-in tool definitions into every request for models without a model
entry (`meta.builtinTools` defaults all-on) — that turns a 1-minute
reply into 7 minutes of prefill. Create the model entry once (Admin
Panel → Models → apus-qwen → Edit → Built-in Tools → all off), or in
the DB:

```bash
docker exec open-webui python3 -c "
import sqlite3, json, time
db = sqlite3.connect('/app/backend/data/webui.db')
cats = ['automations','calendar','channels','chats','code_interpreter',
        'files','image_generation','knowledge','memory','notes',
        'notifications','subagents','tasks','time','user_input','web_search']
meta = {'builtinTools': {c: False for c in cats}}
now = int(time.time())
db.execute('insert or replace into model (id, user_id, name, params, meta, updated_at, created_at, is_active) values (?,?,?,?,?,?,?,1)',
           ('apus-qwen', db.execute(\"select id from user where role='admin'\").fetchone()[0],
            'apus-qwen', '{}', json.dumps(meta), now, now))
db.commit()"
docker restart open-webui
```

Also recommended: Admin Panel → Settings → Interface → off: Title
Auto-Generation, Tags, Follow-Up Generation, Autocomplete (each fires an
extra engine request that queues on the single engine). Hard-reload the
page (Cmd-Shift-R) after config changes — the SPA caches capabilities.

## What to expect

Measured on the dev Mac (M1 Pro, 32 GB), real model, tiered defaults
(tests/m11 for the full tables):

- First turn: ~2 s model load, then ~3 tok/s prefill on short prompts
  (24 tok/s batched on long ones), **2–3 tok/s decode** depending on
  free RAM. A 100-token answer takes ~40–50 s. Normal for 67 GiB of
  weights on 32 GB RAM.
- Peak RAM ~8.3 GiB — **close heavy apps (browsers!) for the biggest
  speedup**; free RAM is the #1 speed factor.
- Multi-turn chat re-reads the conversation each turn (no KV reuse yet);
  use `/reset` for new topics.
- Stop the server/chat with Ctrl-C; expert usage history is saved on
  exit and makes later runs slightly faster (hot pinning).
- `weights/apus-qwen/` is the ONLY copy of the converted weights —
  never delete it (re-downloadable via `tools/download.py`, ~4 h).

## Useful knobs (environment variables)

| var | default | effect |
|---|---|---|
| `APUS_EXPERT_CACHE_MB` | 4096 | expert RAM cache; try 8192 with free RAM (512 stays bitwise-exact, just re-reads more) |
| `APUS_THREADS` | P-cores | compute threads (stream is bitwise at any count) |
| `APUS_PILOT_K` | 8 | prefetch depth — measured optimum (real model: recall 67.6%@6 / 81.3%@8 / 92.2%@12; K=6 −8% wall but thin recall margin, K=12 +30% wall) |
| `APUS_RSS_GUARD_MB` | 26624 | current-RSS budget; drops cached slabs (never pinned bytes) when exceeded — bitwise-safe, verified firing on the real model |
| `APUS_SPEC` | 0 | MTP speculative decoding (exact, measured ~1.9× slower here) |
| `APUS_NO_BLAS` | unset | disable the Accelerate prefill dispatch (currently a no-op: the BLAS class is approved but unconsumed — see tests/m11) |
| `APUS_API_KEY` | unset | server Bearer auth |
| `APUS_THINKING` | 1 | server thinking default |
| `APUS_MAX_SEQ` | 32768 | serve context cap (`--max-seq` overrides) |

If numbers look off, compare with the baselines in `docs/STATUS.md`
(golden command: `./bin/apus-qwen run --model weights/apus-qwen --tiered
--prompt "The capital of France is" --max-tokens 24 --greedy` — must
emit `tests/m11/golden.txt` byte-for-byte, ~20–30 s wall on a quiet
machine; `tests/m11/check_golden.sh` automates the diff).
