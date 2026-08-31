#!/usr/bin/env python3
"""apus-qwen OpenAI-compatible serving gateway (M7). Python STDLIB ONLY.

Follows the colibri/Apus split: this process owns HTTP/JSON/SSE; the C
engine (`bin/apus-qwen serve --model DIR`) is spawned as a subprocess and
driven over stdin/stdout with the NDJSON protocol documented in
tests/m7a/README.md. The model loads once (engine stays up); every
request re-prefills (no KV reuse across turns yet).

Endpoints:
  GET  /health                    -> {"status": "ok", ...}
  GET  /v1/models                 -> OpenAI model list (single model)
  POST /v1/chat/completions       -> chat.completion (or SSE chunks when
                                     "stream": true, ending in data: [DONE])
  POST /v1/completions            -> text_completion (stream supported)
  POST /debug/encode              -> NON-STANDARD test endpoint: the exact
                                     prompt text + token ids the engine
                                     renders for a message list.

Request fields honored (chat): messages, tools, temperature, top_p,
top_k, presence_penalty, max_tokens (or max_completion_tokens), seed,
stream, stream_options.include_usage, stop (str | [str]), model,
chat_template_kwargs.enable_thinking (bool; default from --no-thinking /
APUS_THINKING, default ON — thinking is the Qwen template default;
"thinking" accepted as a legacy alias) and
chat_template_kwargs.preserve_thinking (bool, default False — the model
card's API). Unknown fields are ignored.

Sampling defaults are the Qwen generation_config defaults (temperature
1.0, top_p 0.95, top_k 20). presence_penalty is OPTIONAL and defaults
to 0.0 = OFF (bitwise the pre-M7 pipeline when unset; the model card
recommends 1.5 for thinking mode — that default awaits the user's gate
decision). Semantics: OpenAI/vLLM — subtracted from the logits of tokens
already generated at least once, before temperature/top-k/top-p.

Concurrency: ONE engine, so requests are serialized through a single lock
— concurrent HTTP requests queue in lock-acquisition order and are
processed one at a time. No interleaving, no batching.

Auth: if env APUS_API_KEY is set, /v1/* and /debug/* require
`Authorization: Bearer $APUS_API_KEY` (401 otherwise); /health stays open.
Off by default.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

THINK_END = "</think>"
TC_START = "<tool_call>"
TC_END = "</tool_call>"
FN_START = "<function="
FN_END = "</function>"
PAR_START = "<parameter="
PAR_END = "</parameter>"
EOS_STR = "<|im_end|>"

# ---------------------------------------------------------------------------
# Qwen XML tool-call parsing (response side): the inverse of c/encoding.h's
# assistant tool_calls rendering (reference/chat_template.jinja):
#   <tool_call>\n<function=NAME>\n
#   <parameter=k1>\nv1\n</parameter>\n ...
#   </function>\n</tool_call>
# Tolerant by design: EOS is not required (length/stop finishes),
# unterminated thinking -> all reasoning_content, malformed markup ->
# plain content.
# ---------------------------------------------------------------------------


def parse_qwen_tool_call(text):
    """Parse ONE <tool_call>...</tool_call> body. Returns
    (name, args: {key: raw_value}); raises ValueError on malformed."""
    s = text.strip("\n")
    if not s.startswith(FN_START):
        raise ValueError(f"missing {FN_START!r} opener")
    gt = s.find(">", len(FN_START))
    if gt < 0:
        raise ValueError("unterminated <function=")
    name = s[len(FN_START):gt]
    if not name or "<" in name:
        raise ValueError(f"tool name format error: {name!r}")
    pos = gt + 1
    if s.startswith("\n", pos):
        pos += 1
    args = {}
    while s.startswith(PAR_START, pos):
        gt = s.find(">", pos + len(PAR_START))
        if gt < 0:
            raise ValueError("unterminated <parameter=")
        key = s[pos + len(PAR_START):gt]
        if not key:
            raise ValueError("empty parameter name")
        vs = gt + 1
        if s.startswith("\n", vs):
            vs += 1
        ve = s.find("\n" + PAR_END, vs)
        if ve < 0:
            raise ValueError("unterminated parameter value")
        val = s[vs:ve]
        if key in args:
            raise ValueError(f"duplicate parameter: {key!r}")
        args[key] = val
        pos = ve + 1 + len(PAR_END)
        if s.startswith("\n", pos):
            pos += 1
    if s[pos:] != FN_END:
        raise ValueError(f"expected {FN_END!r}, got {s[pos:]!r}")
    return name, args


def parse_tool_calls(text):
    """Parse a run of consecutive <tool_call> blocks (newline-separated).
    Returns OpenAI-shaped tool_calls; raises ValueError on malformed
    markup."""
    calls = []
    pos = 0
    while True:
        i = text.find(TC_START, pos)
        if i < 0:
            break
        if text[pos:i].strip("\n"):
            raise ValueError(f"garbage between tool_calls: "
                             f"{text[pos:i]!r}")
        e = text.find(TC_END, i)
        if e < 0:
            raise ValueError("unterminated tool_call")
        body = text[i + len(TC_START):e]
        name, args = parse_qwen_tool_call(body)
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args, ensure_ascii=False)},
        })
        pos = e + len(TC_END)
    if not calls:
        raise ValueError("no tool_call block")
    if text[pos:].strip("\n"):
        raise ValueError(f"trailing garbage after tool_call: "
                         f"{text[pos:]!r}")
    return calls


def parse_completion(text, thinking):
    """Split a raw completion into (reasoning_content, content, tool_calls).
    Tolerant port of the reference decode semantics."""
    i = text.find(EOS_STR)
    if i >= 0:
        text = text[:i]
    reasoning, content, tool_calls = "", "", []
    rest = text
    if thinking:
        i = rest.find(THINK_END)
        if i < 0:
            return text, "", []          # unterminated: all reasoning
        reasoning, rest = rest[:i], rest[i + len(THINK_END):]
    i = rest.find(TC_START)
    if i < 0:
        return reasoning, rest, []
    try:
        tool_calls = parse_tool_calls(rest[i:])
        content = rest[:i].strip("\n")
    except ValueError:
        content, tool_calls = rest, []   # malformed markup: plain content
    return reasoning, content, tool_calls


class ThinkSplitter:
    """Incremental thinking/content split for SSE streaming. Reasoning is
    buffered until "</think>" is seen unambiguously; after that, text is
    forwarded as content deltas raw (tool_call markup included — see
    README)."""

    _KEEP = len(THINK_END) - 1

    def __init__(self, thinking):
        self.closed = not thinking
        self.pending = ""

    def feed(self, piece):
        out = []
        s = self.pending + piece
        self.pending = ""
        if not self.closed:
            i = s.find(THINK_END)
            if i >= 0:
                if s[:i]:
                    out.append(("reasoning_content", s[:i]))
                self.closed = True
                s = s[i + len(THINK_END):]
            else:
                if len(s) > self._KEEP:
                    out.append(("reasoning_content", s[:-self._KEEP]))
                    s = s[-self._KEEP:]
                self.pending = s
                return out
        if s:
            out.append(("content", s))
        return out

    def flush(self):
        if not self.pending:
            return []
        kind = "content" if self.closed else "reasoning_content"
        p, self.pending = self.pending, ""
        return [(kind, p)]


# ---------------------------------------------------------------------------
# Engine subprocess (single instance, serialized access)
# ---------------------------------------------------------------------------


class EngineError(Exception):
    pass


class ApusEngine:
    def __init__(self, apus_bin, model_dir, tiered=False):
        cmd = [apus_bin, "serve", "--model", model_dir]
        if tiered:
            cmd.append("--tiered")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.lock = threading.Lock()
        self._next_id = 0

    def alive(self):
        return self.proc.poll() is None

    def request(self, payload):
        """Send one request; yield engine events up to and including the
        terminal one (done/error/encoded). Holds the engine lock for the
        whole exchange: one request in flight, others queue. If the caller
        abandons the generator (client disconnect), the remaining events
        are drained so the protocol stays in sync."""
        with self.lock:
            self._next_id += 1
            payload = dict(payload, id=self._next_id)
            try:
                self.proc.stdin.write(json.dumps(payload) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError) as e:
                raise EngineError(f"engine not writable: {e}")
            terminal = False
            try:
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        raise EngineError("engine exited mid-request")
                    ev = json.loads(line)
                    if ev.get("type") in ("done", "error", "encoded"):
                        terminal = True
                    yield ev
                    if terminal:
                        return
            finally:
                if not terminal:       # abandoned: drain to the sentinel
                    while True:
                        line = self.proc.stdout.readline()
                        if not line:
                            break
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            continue
                        if ev.get("type") in ("done", "error", "encoded"):
                            break

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------------------
# HTTP gateway
# ---------------------------------------------------------------------------


def openai_error(message, err_type="invalid_request_error", param=None,
                 code=None):
    return {"error": {"message": message, "type": err_type,
                      "param": param, "code": code}}


class Gateway(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, engine, model_id, default_thinking=True):
        super().__init__(addr, Handler)
        self.engine = engine
        self.model_id = model_id
        self.default_thinking = default_thinking
        self.api_key = os.environ.get("APUS_API_KEY") or None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "apus-qwen-m7"

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt, *args):
        sys.stderr.write("gateway: " + fmt % args + "\n")

    def _send_json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status, message, err_type="invalid_request_error"):
        self._send_json(status, openai_error(message, err_type))

    def _auth_ok(self):
        key = self.server.api_key
        if not key:
            return True
        if self.headers.get("Authorization") == f"Bearer {key}":
            return True
        self._send_error(401, "missing or invalid API key",
                         "authentication_error")
        return False

    def _read_body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0 or n > 64 * 1024 * 1024:
            return None
        return self.rfile.read(n)

    # -- GET ---------------------------------------------------------------

    def do_GET(self):
        if self.path == "/health":
            eng = "up" if self.server.engine.alive() else "down"
            self._send_json(200 if eng == "up" else 503,
                            {"status": "ok" if eng == "up" else "degraded",
                             "model": self.server.model_id, "engine": eng})
            return
        if self.path == "/v1/models":
            if not self._auth_ok():
                return
            self._send_json(200, {"object": "list", "data": [{
                "id": self.server.model_id, "object": "model",
                "created": 0, "owned_by": "apus-qwen"}]})
            return
        self._send_error(404, f"unknown path: {self.path}",
                         "not_found_error")

    # -- POST --------------------------------------------------------------

    def do_POST(self):
        if not self._auth_ok():
            return
        raw = self._read_body()
        try:
            body = json.loads(raw) if raw else None
        except ValueError as e:
            self._send_error(400, f"malformed JSON body: {e}")
            return
        if not isinstance(body, dict):
            self._send_error(400, "request body must be a JSON object")
            return
        if self.path == "/v1/chat/completions":
            self._chat_completions(body)
        elif self.path == "/v1/completions":
            self._completions(body)
        elif self.path == "/debug/encode":
            self._debug_encode(body)
        else:
            self._send_error(404, f"unknown path: {self.path}",
                             "not_found_error")

    # -- shared request translation ---------------------------------------

    def _check_model(self, body):
        m = body.get("model")
        if m is not None and m != self.server.model_id:
            self._send_error(404, f"unknown model: {m!r}",
                             "not_found_error")
            return False
        return True

    def _sampling_params(self, body):
        req = {}
        mt = body.get("max_tokens", body.get("max_completion_tokens"))
        if mt is not None:
            req["max_tokens"] = int(mt)
        for k in ("temperature", "top_p", "top_k", "seed",
                  "presence_penalty"):
            if body.get(k) is not None:
                req[k] = body[k]
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        if isinstance(stop, list) and all(isinstance(s, str) for s in stop):
            req["stop"] = stop
        return req

    def _template_kwargs(self, body):
        """The model card's chat_template_kwargs API: enable_thinking
        (default ON / server default; "thinking" accepted as a legacy
        alias) and preserve_thinking (default False)."""
        kw = body.get("chat_template_kwargs")
        if not isinstance(kw, dict):
            kw = {}
        thinking = kw.get("enable_thinking",
                          kw.get("thinking", self.server.default_thinking))
        return bool(thinking), bool(kw.get("preserve_thinking", False))

    # -- /v1/chat/completions ----------------------------------------------

    def _chat_completions(self, body):
        if not self._check_model(body):
            return
        messages = body.get("messages")
        if not isinstance(messages, list) or not all(
                isinstance(m, dict) for m in messages):
            self._send_error(400, "messages must be an array of objects")
            return
        thinking, preserve = self._template_kwargs(body)
        req = {"cmd": "generate", "messages": messages,
               "thinking": thinking, "preserve_thinking": preserve}
        if body.get("tools") is not None:
            req["tools"] = body["tools"]
        req.update(self._sampling_params(body))
        if body.get("stream"):
            self._chat_stream(req, thinking, body.get("stream_options") or {})
        else:
            self._chat_once(req, thinking)

    def _chat_once(self, req, thinking):
        try:
            done = None
            for ev in self.server.engine.request(req):
                if ev.get("type") == "done":
                    done = ev
                elif ev.get("type") == "error":
                    self._send_error(500, ev.get("message", "engine error"),
                                     "engine_error")
                    return
        except EngineError as e:
            self._send_error(503, str(e), "engine_error")
            return
        reasoning, content, tool_calls = parse_completion(
            done.get("text", ""), thinking)
        message = {"role": "assistant"}
        if reasoning:
            message["reasoning_content"] = reasoning
        message["content"] = content if (content or not tool_calls) else None
        if tool_calls:
            message["tool_calls"] = tool_calls
        p, c = done["prompt_tokens"], done["completion_tokens"]
        resp = {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.server.model_id,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self._finish_reason(done, tool_calls),
            }],
            "usage": {"prompt_tokens": p, "completion_tokens": c,
                      "total_tokens": p + c},
        }
        self._send_json(200, resp)

    @staticmethod
    def _finish_reason(done, tool_calls):
        if done["finish_reason"] == "length":
            return "length"
        if tool_calls:
            return "tool_calls"
        return "stop"          # both EOS and stop_string map to "stop"

    def _sse_begin(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse_send(self, obj):
        self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False)
                          + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def _chat_stream(self, req, thinking, stream_opts):
        include_usage = bool(stream_opts.get("include_usage"))
        cmpl_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        def chunk(delta, finish=None):
            return {"id": cmpl_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.server.model_id,
                    "choices": [{"index": 0, "delta": delta,
                                 "finish_reason": finish}]}

        self._sse_begin()
        splitter = ThinkSplitter(thinking)
        full = []
        done = None
        try:
            self._sse_send(chunk({"role": "assistant"}))
            for ev in self.server.engine.request(req):
                t = ev.get("type")
                if t == "token":
                    piece = ev.get("text", "")
                    full.append(piece)
                    for kind, text in splitter.feed(piece):
                        self._sse_send(chunk({kind: text}))
                elif t == "done":
                    done = ev
                elif t == "error":
                    self._sse_send(openai_error(
                        ev.get("message", "engine error"), "engine_error"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
            for kind, text in splitter.flush():
                self._sse_send(chunk({kind: text}))
            if done is None:
                self.wfile.write(b"data: [DONE]\n\n")
                return
            reasoning, content, tool_calls = parse_completion(
                "".join(full), thinking)
            self._sse_send(chunk({}, self._finish_reason(done, tool_calls)))
            if include_usage:
                p, c = done["prompt_tokens"], done["completion_tokens"]
                self._sse_send({"id": cmpl_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": self.server.model_id,
                                "choices": [],
                                "usage": {"prompt_tokens": p,
                                          "completion_tokens": c,
                                          "total_tokens": p + c}})
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass               # client left; engine generator drains itself
        except EngineError:
            pass

    # -- /v1/completions ----------------------------------------------------

    def _completions(self, body):
        if not self._check_model(body):
            return
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            self._send_error(400, "prompt must be a string "
                                  "(batched prompts not supported)")
            return
        req = {"cmd": "generate", "text": prompt}
        req.update(self._sampling_params(body))
        if body.get("stream"):
            self._compl_stream(req)
            return
        try:
            done = None
            for ev in self.server.engine.request(req):
                if ev.get("type") == "done":
                    done = ev
                elif ev.get("type") == "error":
                    self._send_error(500, ev.get("message", "engine error"),
                                     "engine_error")
                    return
        except EngineError as e:
            self._send_error(503, str(e), "engine_error")
            return
        p, c = done["prompt_tokens"], done["completion_tokens"]
        self._send_json(200, {
            "id": "cmpl-" + uuid.uuid4().hex[:24],
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.server.model_id,
            "choices": [{"text": done.get("text", ""), "index": 0,
                         "finish_reason": ("length" if done["finish_reason"]
                                           == "length" else "stop")}],
            "usage": {"prompt_tokens": p, "completion_tokens": c,
                      "total_tokens": p + c},
        })

    def _compl_stream(self, req):
        cmpl_id = "cmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        self._sse_begin()
        done = None
        try:
            for ev in self.server.engine.request(req):
                t = ev.get("type")
                if t == "token":
                    self._sse_send({
                        "id": cmpl_id, "object": "text_completion",
                        "created": created, "model": self.server.model_id,
                        "choices": [{"text": ev.get("text", ""), "index": 0,
                                     "finish_reason": None}]})
                elif t == "done":
                    done = ev
                elif t == "error":
                    self._sse_send(openai_error(
                        ev.get("message", "engine error"), "engine_error"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
            fr = "length" if done["finish_reason"] == "length" else "stop"
            self._sse_send({"id": cmpl_id, "object": "text_completion",
                            "created": created, "model": self.server.model_id,
                            "choices": [{"text": "", "index": 0,
                                         "finish_reason": fr}]})
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError, EngineError):
            pass

    # -- /debug/encode (non-standard; test/conformance endpoint) ------------

    def _debug_encode(self, body):
        messages = body.get("messages")
        if not isinstance(messages, list):
            self._send_error(400, "messages must be an array")
            return
        thinking, preserve = self._template_kwargs(body)
        req = {"cmd": "encode", "messages": messages,
               "thinking": thinking, "preserve_thinking": preserve}
        if body.get("tools") is not None:
            req["tools"] = body["tools"]
        try:
            for ev in self.server.engine.request(req):
                if ev.get("type") == "encoded":
                    self._send_json(200, {"text": ev.get("text", ""),
                                          "ids": ev.get("ids", [])})
                    return
                if ev.get("type") == "error":
                    self._send_error(400, ev.get("message", "encode error"),
                                     "engine_error")
                    return
        except EngineError as e:
            self._send_error(503, str(e), "engine_error")


def main():
    ap = argparse.ArgumentParser(
        description="apus-qwen OpenAI-compatible gateway")
    ap.add_argument("--model", required=True, help="model dir")
    ap.add_argument("--apus",
                    default=os.environ.get("APUS_BIN", "bin/apus-qwen"),
                    help="path to the apus-qwen binary (env APUS_BIN)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tiered", action="store_true",
                    help="engine expert-store tiering (M6)")
    ap.add_argument("--model-id", default=None,
                    help="served model id (default: model dir basename)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="default to non-thinking mode (per-request "
                         "chat_template_kwargs.enable_thinking overrides)")
    args = ap.parse_args()

    default_thinking = (os.environ.get("APUS_THINKING", "1") != "0"
                        and not args.no_thinking)
    model_id = args.model_id or os.path.basename(
        os.path.normpath(args.model)) or "apus-qwen"
    engine = ApusEngine(args.apus, args.model, args.tiered)
    srv = Gateway((args.host, args.port), engine, model_id, default_thinking)
    sys.stderr.write(f"apus-qwen gateway: http://{args.host}:{args.port} "
                     f"model={model_id} engine={' '.join([args.apus, 'serve'])}"
                     f" thinking_default={default_thinking}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()


if __name__ == "__main__":
    main()
