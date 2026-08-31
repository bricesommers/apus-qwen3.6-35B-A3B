# tests/m7a — OpenAI-compatible serving (M7)

Milestone M7: OpenAI-compatible local serving for apus-qwen, following
the colibri/Apus split — a **Python stdlib-only HTTP gateway**
(`tools/server.py`) driving the **C engine** (`bin/apus-qwen serve`) as a
persistent subprocess. No new dependencies (gateway: `http.server`,
`json`, `subprocess`, `threading`; engine: libc/pthreads only). No
numerics/router changes: the serve path reuses `c/encoding.h` (Qwen
ChatML rendering, the M2 port), `c/model.h` (forward), and `c/sample.h`
(sampling) exactly as the `run` CLI does. The only engine-side serving
additions are the **dual-EOS stop** and the **opt-in presence_penalty**
(below).

Run:

```
make golden-m7a     # regenerate tests/m7a/fixtures (scripted parrot models)
make test-m7a       # 39 tests, exit 0 iff all pass
make ubsan-m7a      # same suite with bin/apus-qwen built -fsanitize=undefined
```

Manual use:

```
bin/apus-qwen serve --model DIR [--tiered]                  # engine, stdio
.venv/bin/python tools/server.py --model DIR --port 8000    # gateway, HTTP
.venv/bin/python tools/chat.py --model DIR                  # terminal REPL
```

## Architecture: stdio NDJSON (engine stays socket-free)

The C engine speaks **one-JSON-object-per-line on stdin/stdout**; the
gateway owns all networking (binds localhost). The engine stays libc-only
(no socket code in C), the process model is gateway-drives-engine, and a
single engine process serves exactly one gateway.

### Engine protocol (request → `bin/apus-qwen serve` stdin)

```json
{"id": <any>, "cmd": "encode",
 "messages": [...], "tools": [...]|null, "thinking": true|false,
 "preserve_thinking": true|false}

{"id": <any>, "cmd": "generate",
 "messages": [...] | "text": "raw prompt" | "ids": [1,2,3],
 "tools": [...]|null, "thinking": bool, "preserve_thinking": bool,
 "max_tokens": int, "temperature": float, "top_p": float,
 "top_k": int, "seed": uint, "stop": [str, ...],
 "presence_penalty": float}
```

- `tools` (OpenAI format) is passed to `ling_encode_messages` as the
  template's `tools` variable — the template renders the `<|im_start|>system`
  block with the `<tools>` section (c/encoding.h, jinja-faithful, M2).
- `preserve_thinking` maps to the template's `preserve_thinking` kwarg
  (thinking kept in ALL assistant history turns; default off = only the
  turns after the last real user query keep it).
- `"text"` is tokenized verbatim (no chat template, no BOS — the
  `/v1/completions` path). `"ids"` feeds raw ids (synthetic models
  without a tokenizer).
- Defaults: `max_tokens` unset = fill the remaining context (OpenAI
  semantics — generate until EOS/stop), `temperature` 1.0, `top_p` 0.95,
  `top_k` 20, `seed` 0, `presence_penalty` 0.0 (the Qwen
  generation_config sampling defaults; `temperature <= 0` = greedy).
  Generation is clamped to `max_seq`.

### Engine protocol (events → stdout)

```json
{"id","type":"encoded","text","ids"}                 encode reply
{"id","type":"prompt","prompt_tokens"}               generate, first
{"id","type":"token","token_id","text"}              per generated token
{"id","type":"done","finish_reason","prompt_tokens",
 "completion_tokens","text"}                         terminal
{"id","type":"error","message"}                      request failed
```

- **EOS (M7)**: generation stops on ANY eos id — the config.json
  `eos_token_id` (**int or array**; the real config has 248044
  `<|endoftext|>`, the fixtures use the array form `[257, 258]`) plus
  `<|im_end|>` (248046) resolved through the tokenizer, mirroring the
  generation_config list `[248046, 248044]`. A terminating EOS is never
  emitted as a `token` event; `finish_reason` is `"stop"` (EOS),
  `"length"`, or `"stop_string"`.
- Stop strings are matched against the assembled decoded text; on a
  match the text is truncated at the match start (a partial last-token
  piece is emitted if it precedes the match). `text` fields require a
  tokenizer in the model dir.
- The process stays alive across requests; every request gets a fresh
  model state (multi-turn context is re-prefilled by the client side;
  KV reuse across turns is a later optimization).

### presence_penalty (M7, opt-in, default OFF)

OpenAI/vLLM semantics: when `presence_penalty != 0`, its value is
subtracted from the logit of every token already **generated** at least
once in this completion (once per unique token; the prompt is not
counted — vLLM's `logits -= presence_penalty * output_mask`), applied
to the raw logits **before** temperature/top-k/top-p (the HF/vLLM
processors-before-warpers order — verified against transformers
`_get_logits_processor`, which appends penalty processors before the
Temperature→TopK→TopP warpers, and vLLM's sampler pipeline).
`0.0` (the default) is exactly the pre-M7 pipeline, bitwise. The model
card recommends `presence_penalty=1.5` for thinking mode; making that
the served default is a **pending user-gate decision** — the plumbing
ships inert.

## Gateway endpoints (`tools/server.py`)

| Endpoint | Notes |
|---|---|
| `GET /health` | `{"status","model","engine"}`; no auth |
| `GET /v1/models` | single-model OpenAI list |
| `POST /v1/chat/completions` | full OpenAI shape; `stream:true` → SSE |
| `POST /v1/completions` | `prompt` (string only); stream supported |
| `POST /debug/encode` | NON-STANDARD test endpoint: rendered prompt text + token ids (conformance/usage verification) |

Honored chat fields: `messages`, `tools`, `temperature`, `top_p`,
`top_k`, `presence_penalty`, `max_tokens`/`max_completion_tokens`,
`seed`, `stream`, `stream_options.include_usage`, `stop` (string or
list), `model`, `chat_template_kwargs.enable_thinking` (legacy alias
`thinking`) and `chat_template_kwargs.preserve_thinking`. Unknown fields
are ignored.

**Thinking mode.** The Qwen template default is thinking ON (generation
prompt `<|im_start|>assistant\n<think>\n`); the gateway defaults the
same (override per request with
`chat_template_kwargs: {"enable_thinking": false}` →
`...<think>\n\n</think>\n\n`, or globally with `--no-thinking` /
`APUS_THINKING=0`). Responses expose reasoning as `reasoning_content`
(OpenAI/DeepSeek-API style).

**SSE streaming.** `chat.completion.chunk` deltas: first
`{"role":"assistant"}`, then `reasoning_content` deltas (buffered until
`</think>` is unambiguous), then `content` deltas (raw — tool_call
markup included, see below), a final empty-delta chunk with
`finish_reason`, an optional usage chunk
(`stream_options.include_usage`, `choices: []`), and `data: [DONE]`.
`Connection: close` (no chunked encoding).

**Qwen XML ↔ OpenAI tool calls.** Request side: `tools` render into the
system block via `ling_encode_messages` (jinja-faithful, M2-conformant);
assistant history `tool_calls` render as
`<tool_call>\n<function=name>\n<parameter=k>\nv\n</parameter>\n
</function>\n</tool_call>` and `role: "tool"` results coalesce into one
user turn of `<tool_response>` blocks — all in c/encoding.h. Response
side: the gateway parses the model's `<tool_call>` markup back into
OpenAI `tool_calls` (`id` = `call_<24 hex>`, `type: "function"`,
`function.arguments` = JSON string of the raw-value dict). The parser is
*tolerant*: EOS not required, unterminated thinking → all
`reasoning_content`, malformed markup → plain content; text before the
opener is newline-stripped into `content`. `finish_reason`: `"length"`
for length, `"tool_calls"` when calls were parsed, else `"stop"`.

**Errors.** `400` malformed JSON / bad fields, `404` unknown model or
path, `401` auth failure — all in the OpenAI shape
`{"error": {"message","type","param","code"}}`.

**Concurrency.** One engine ⇒ requests are serialized through a single
lock (FIFO queue, no interleaving, no batching). **Auth:** env
`APUS_API_KEY` → `Authorization: Bearer` required on `/v1/*` and
`/debug/*` (`/health` open).

## The scripted "parrot" fixtures (`tests/m7a/fixtures/`)

Random-weight models can't produce a *known* token stream, which the
server tests need (Qwen XML tool-call output, dual EOS, stop strings,
exact usage counts). `gen_fixtures.py` builds two mini-models on the M5
dims/schedule (2 layers typed by the real rules: GDN+MoE, full-attn+MoE)
with **all layer weights zero** (every sublayer output vanishes; the
bf16 residual is preserved exactly, so the final hidden state depends
only on the last embedded token), random embed rows, and a **scripted
head**: for each transition `a → b`, `head[b] = 8·embed[a]` (min logit
margin >500, checked at gen time). Chain tokens with multi-character
content are added tokens (ids 300+), plus a byte-level vocab (ids
0–255, GPT-2 alphabet, no merges) and the ChatML specials (256–261:
`<|im_start|>`, `<|im_end|>` = EOS 257, `<|endoftext|>` = EOS 258,
`<think>`, `</think>`, and `</think>\n\n` = 261). The config's
`eos_token_id` is the **array** `[257, 258]` — exercising the M7 list
parsing; `vocab_size` is 512 (embed/head rebuilt at the wider vocab).

The `</think>\n\n` added token is what lets the parrot tell the two
generation prompts apart: leftmost-longest added-token matching makes
the thinking-OFF prompt end in token 261 while the thinking-ON prompt
ends in `<think>` (259) + `\n` byte (10) — so thinking mode enters the
reasoning chain and chat mode enters the answer-only chain, like the
real model.

- `model_chat/`: thinking reply `"reasoning: thinking it over.</think>
  The answer is STOP right here."` + `<|im_end|>`; chat-mode reply is
  the answer alone; raw chains: `">"` → the thinking stream, `"#"` →
  `"ALT"` + EOS **258** (dual-EOS proof), `"~"` → the two bytes of `é`
  as separate tokens + EOS (UTF-8-guard proof).
- `model_tools/`: reasoning + a complete Qwen XML `get_weather` tool
  call + `<|im_end|>`.

## Test coverage (39 checks, `test_server.py`)

- **Pipe protocol (8):** encode thinking/chat (exact prompt strings;
  thinking prompt ends `<think>\n` = ids [259, 10], chat prompt ends in
  the off-tail token 261); malformed line / unknown cmd / error
  recovery; generate event sequence, EOS non-emission, exact scripted
  text; usage == `encode` id count; stop-string truncation; max_tokens
  clamp; ids-only path on the tokenizer-less M5 fixture incl. clean
  error for message requests; **dual EOS** (the `#` chain stops on
  `<|endoftext|>` 258); **presence_penalty** (0.0 == unset bitwise;
  −1e5 loops the already-generated token).
- **Qwen XML parser (7):** thinking/chat splits; tool-call shape
  (`<function=name>`, `<parameter=k>`, `call_` id, arguments JSON);
  multiple calls + CJK values; multi-line parameter values; bare
  name-only call; malformed name raises; tolerant tails (unterminated
  think, malformed markup → plain content, `<|im_end|>` cut, inter-block
  garbage); `ThinkSplitter` char-by-char exactness.
- **HTTP chat (15 on model_chat):** health/models; thinking mode
  (shape, reasoning_content, usage); chat mode via
  `chat_template_kwargs.enable_thinking` AND the legacy `thinking`
  alias; usage == `/debug/encode` ids; stop strings (list & bare
  string); seed determinism; SSE (chunk shape, event order, reassembly
  == non-stream, usage chunk, `[DONE]`); 6-way concurrency; error shapes
  (400/404); `/v1/completions` (+SSE); dual EOS over HTTP; UTF-8
  split-char guard (é over 2 tokens, streamed and not);
  presence_penalty (off-default == 0.0; −1e5 loops → `length`);
  gateway-vs-pipe encode equality on a multi-turn `reasoning_content`
  conversation; **8 jinja2-rendered conformance cases through
  `/debug/encode` byte-exact** (thinking on/off, system+user, multi-turn
  reasoning strip, preserve_thinking, embedded `<think>` split, tools,
  tool round-trip).
- **HTTP tools (3 on model_tools):** full tool-call round trip
  (`tool_calls` JSON, `finish_reason: "tool_calls"`, usage); streaming
  finish reason; `role="tool"` follow-up rendering (`<tool_response>`
  user coalescing, byte-exact vs the jinja golden).
- **Auth (1):** `APUS_API_KEY` 401/200, open `/health`.
- **chat.py (1):** scripted session — `/system`, thinking turn (dimmed
  reasoning with the `---` boundary), `/thinking off`, `/preserve on`,
  `/raw`, `/reset` — output shape and per-turn stats lines.

## `tools/chat.py`

Terminal REPL over the same stdio protocol (no HTTP): `/quit /reset
/system /thinking /preserve /temp /max /raw /help`, thinking reasoning
printed dimmed, per-turn `[N tok, Ts, X tok/s, finish_reason]` stats.
Defaults temp 1.0 / top_p 0.95 / top_k 20 / thinking on
(`APUS_THINKING=0`, `--no-thinking`), preserve_thinking off
(`--preserve-thinking`), `APUS_BIN` overrides the engine path.

## Known limitations

- No KV reuse across turns — every request re-prefills the full
  conversation (single-user local serving).
- Single engine: requests serialized (throughput is a non-goal).
- Streaming tool calls arrive as raw `<tool_call>` markup in `content`
  deltas; parse client-side or use non-streaming.
- `/v1/completions` accepts a single string prompt (no batching,
  `echo`/`logprobs`/`suffix`).
- Engine stop strings are matched on decoded text only (needs a
  tokenizer); at most 16 per request; generation clamped to `max_seq`.
- `presence_penalty` counts generated tokens only (OpenAI/vLLM
  semantics), not prompt tokens (HF `input_ids` semantics) — the served
  API is the OpenAI one. `frequency_penalty` and `repetition_penalty`
  are not implemented.

## Files

- `c/apus-qwen.c` — `serve` subcommand (NDJSON protocol) + shared
  engine context with `run` (dual-EOS, presence_penalty plumbing).
- `c/model.h` — config `eos_token_id` int-or-array → `eos_ids[]`.
- `tools/server.py` — the OpenAI gateway (stdlib only).
- `tools/chat.py` — terminal REPL.
- `tests/m7a/gen_fixtures.py` — parrot models + jinja conformance goldens.
- `tests/m7a/test_server.py` — the 39-check suite.
