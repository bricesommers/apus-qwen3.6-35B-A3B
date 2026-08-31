#!/usr/bin/env python3
"""tests/m7a/test_server.py — serving milestone (M7) tests: NDJSON pipe
protocol, Qwen XML tool-call parser, full HTTP path (OpenAI endpoints,
SSE, thinking on/off, preserve_thinking, tool-call round trip, dual-EOS
stop, presence_penalty, UTF-8 guard, concurrency, errors, auth),
chat-template conformance byte-exactness, and tools/chat.py.

Run from the repository root: .venv/bin/python tests/m7a/test_server.py
Fixtures: tests/m7a/fixtures (make golden-m7a).
"""

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import server as gw  # noqa: E402

BIN = os.environ.get("APUS_BIN", os.path.join(ROOT, "bin", "apus-qwen"))
SERVER_PY = os.path.join(ROOT, "tools", "server.py")
CHAT_PY = os.path.join(ROOT, "tools", "chat.py")
FIX = os.path.join(HERE, "fixtures")
MODEL_CHAT = os.path.join(FIX, "model_chat")
MODEL_TOOLS = os.path.join(FIX, "model_tools")
MODEL_M5 = os.path.join(ROOT, "tests", "m5", "fixtures", "model")

THINK_TEXT = "reasoning: thinking it over."
ANSWER_TEXT = "The answer is STOP right here."
TOOL_REASONING = "I should check the weather."

PASS = 0
FAIL = 0


def check(cond, name, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


# ---------------------------------------------------------------------------
# engine pipe helpers

class Pipe:
    def __init__(self, model):
        self.proc = subprocess.Popen(
            [BIN, "serve", "--model", model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def request(self, req):
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        events = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            ev = json.loads(line)
            events.append(ev)
            if ev.get("type") in ("done", "error", "encoded"):
                return events

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Http:
    def __init__(self, model, env_extra=None, no_thinking=False):
        self.port = free_port()
        env = dict(os.environ)
        env.update(env_extra or {})
        cmd = [sys.executable, SERVER_PY, "--model", model,
               "--port", str(self.port)]
        if no_thinking:
            cmd.append("--no-thinking")
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=env)
        for _ in range(200):
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port,
                                               timeout=2)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    c.close()
                    break
                c.close()
            except OSError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("gateway did not come up")

    def call(self, method, path, body=None, headers=None, raw=False):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        h = dict(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body)
            h["Content-Type"] = "application/json"
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        blob = r.read()
        c.close()
        if raw:
            return r.status, blob
        try:
            return r.status, json.loads(blob)
        except ValueError:
            return r.status, blob

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------------------
# 1. pipe protocol

def t_pipe_encode():
    p = Pipe(MODEL_CHAT)
    ev = p.request({"id": 1, "cmd": "encode",
                    "messages": [{"role": "user", "content": "hello"}],
                    "thinking": True})[0]
    p.close()
    want = ("<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n")
    check(ev["type"] == "encoded" and ev["text"] == want
          and ev["ids"][-2:] == [259, 10] and len(ev["ids"]) == 26,
          "pipe/encode thinking", json.dumps(ev)[:200])
    p = Pipe(MODEL_CHAT)
    ev = p.request({"id": 2, "cmd": "encode",
                    "messages": [{"role": "user", "content": "hello"}],
                    "thinking": False})[0]
    p.close()
    check(ev["type"] == "encoded"
          and ev["text"].endswith("<|im_start|>assistant\n"
                                  "<think>\n\n</think>\n\n")
          and ev["ids"][-1] == 261,
          "pipe/encode chat mode (off-tail token 261)", ev["text"][-40:])


def t_pipe_errors():
    p = Pipe(MODEL_CHAT)
    p.proc.stdin.write("this is not json\n")
    p.proc.stdin.flush()
    ev = json.loads(p.proc.stdout.readline())
    ok1 = ev.get("type") == "error"
    ev = p.request({"id": 9, "cmd": "frobnicate"})[0]
    ok2 = ev.get("type") == "error"
    ev = p.request({"id": 10, "cmd": "encode",
                    "messages": [{"role": "user", "content": "hello"}],
                    "thinking": True})[0]
    p.close()
    check(ok1 and ok2 and ev.get("type") == "encoded",
          "pipe/error recovery (malformed line, unknown cmd)")


def t_pipe_generate():
    p = Pipe(MODEL_CHAT)
    evs = p.request({"id": 1, "cmd": "generate",
                     "messages": [{"role": "user", "content": "hello"}],
                     "thinking": True, "max_tokens": 10})
    p.close()
    types = [e["type"] for e in evs]
    texts = "".join(e.get("text", "") for e in evs if e["type"] == "token")
    done = evs[-1]
    check(types[0] == "prompt" and types[-1] == "done"
          and types.count("token") == 5
          and all(e.get("token_id") not in (257, 258) for e in evs
                  if e["type"] == "token")
          and done["finish_reason"] == "stop"
          and done["completion_tokens"] == 5
          and texts == THINK_TEXT + "</think>" + ANSWER_TEXT
          and done["text"] == texts,
          "pipe/generate sequence + EOS non-emission + text",
          f"{types} {texts!r}")


def t_pipe_usage():
    p = Pipe(MODEL_CHAT)
    enc = p.request({"id": 1, "cmd": "encode",
                     "messages": [{"role": "user", "content": "hello"}],
                     "thinking": True})[0]
    done = p.request({"id": 2, "cmd": "generate",
                      "messages": [{"role": "user", "content": "hello"}],
                      "thinking": True})[-1]
    p.close()
    check(done["prompt_tokens"] == len(enc["ids"]),
          "pipe/usage == encode id count",
          f"{done['prompt_tokens']} != {len(enc['ids'])}")


def t_pipe_stop_string():
    p = Pipe(MODEL_CHAT)
    evs = p.request({"id": 1, "cmd": "generate",
                     "messages": [{"role": "user", "content": "hello"}],
                     "thinking": True, "stop": ["STOP"]})
    p.close()
    pieces = [e.get("text", "") for e in evs if e["type"] == "token"]
    done = evs[-1]
    check(done["finish_reason"] == "stop_string"
          and all("STOP" not in x for x in pieces)
          and done["text"] == THINK_TEXT + "</think>The answer is ",
          "pipe/stop string truncation", f"{pieces} {done['text']!r}")


def t_pipe_max_tokens_and_ids():
    p = Pipe(MODEL_CHAT)
    done = p.request({"id": 1, "cmd": "generate",
                      "messages": [{"role": "user", "content": "hello"}],
                      "thinking": True, "max_tokens": 2})[-1]
    p.close()
    check(done["finish_reason"] == "length"
          and done["completion_tokens"] == 2,
          "pipe/max_tokens clamp", json.dumps(done)[:160])
    # tokenizer-less model: messages request fails cleanly, ids works
    p = Pipe(MODEL_M5)
    ev = p.request({"id": 2, "cmd": "generate",
                    "messages": [{"role": "user", "content": "hello"}]})[0]
    ok1 = ev.get("type") == "error"
    done = p.request({"id": 3, "cmd": "generate", "ids": [3, 1, 4, 1],
                      "max_tokens": 2})[-1]
    p.close()
    check(ok1 and done.get("type") == "done"
          and done["prompt_tokens"] == 4,
          "pipe/ids path on tokenizer-less model + clean error",
          f"{ev.get('message')} / {json.dumps(done)[:120]}")


def t_pipe_dual_eos():
    # the "alt#" raw chain ends with <|endoftext|> (258) — the second id
    # of the config's eos_token_id array [257, 258]
    p = Pipe(MODEL_CHAT)
    evs = p.request({"id": 1, "cmd": "generate", "text": "alt#"})
    p.close()
    done = evs[-1]
    check(done["finish_reason"] == "stop"
          and done["completion_tokens"] == 1
          and done["text"] == "ALT"
          and all(e.get("token_id") != 258 for e in evs
                  if e["type"] == "token"),
          "pipe/dual EOS (<|endoftext|> 258 stops, never emitted)",
          json.dumps(done)[:200])


def t_pipe_presence_penalty():
    p = Pipe(MODEL_CHAT)
    base = {"id": 1, "cmd": "generate",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": True, "max_tokens": 4}
    d_unset = p.request(dict(base, id=1))[-1]
    d_zero = p.request(dict(base, id=2, presence_penalty=0.0))[-1]
    p.close()
    ok1 = (d_unset["text"] == d_zero["text"]
           and d_unset["finish_reason"] == d_zero["finish_reason"])
    # huge NEGATIVE penalty: every already-generated token gets boosted,
    # so the first scripted chunk repeats forever (proves the penalty is
    # applied to generated tokens at all); positive-ish default path is
    # the unset/0.0 case above (off = bitwise pre-M7).
    p = Pipe(MODEL_CHAT)
    d_neg = p.request(dict(base, id=3, presence_penalty=-100000.0))[-1]
    p.close()
    ok2 = (d_neg["finish_reason"] == "length"
           and d_neg["completion_tokens"] == 4
           and d_neg["text"] == THINK_TEXT * 4)
    check(ok1 and ok2,
          "pipe/presence_penalty (0.0 == unset; -1e5 loops the seen token)",
          f"{d_zero.get('text')!r} / {d_neg.get('text')!r}")


# ---------------------------------------------------------------------------
# 2. Qwen XML parser (unit, via tools/server.py)

def t_parser_thinking():
    r, c, tc = gw.parse_completion(
        "reasoning here.</think>The answer.", True)
    check(r == "reasoning here." and c == "The answer." and tc == [],
          "parser/thinking split", f"{r!r} {c!r}")
    r, c, tc = gw.parse_completion("The answer.", False)
    check(r == "" and c == "The answer." and tc == [],
          "parser/chat mode", f"{r!r} {c!r}")


def t_parser_tool_call():
    text = ("\n\n<tool_call>\n<function=get_weather>\n"
            "<parameter=location>\nBeijing\n</parameter>\n"
            "</function>\n</tool_call>")
    r, c, tc = gw.parse_completion("reasoning.</think>" + text, True)
    ok = (r == "reasoning." and c == "" and len(tc) == 1
          and tc[0]["type"] == "function"
          and tc[0]["id"].startswith("call_")
          and tc[0]["function"]["name"] == "get_weather"
          and json.loads(tc[0]["function"]["arguments"])
          == {"location": "Beijing"})
    check(ok, "parser/tool call shape", json.dumps(tc)[:200])


def t_parser_multi_and_typed():
    text = ("<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n"
            "</function>\n</tool_call>\n"
            "<tool_call>\n<function=g>\n<parameter=b>\n中文\n</parameter>\n"
            "</function>\n</tool_call>")
    r, c, tc = gw.parse_completion(text, False)
    ok = (len(tc) == 2
          and json.loads(tc[0]["function"]["arguments"]) == {"a": "1"}
          and json.loads(tc[1]["function"]["arguments"]) == {"b": "中文"})
    check(ok, "parser/multiple calls + CJK", json.dumps(tc,
          ensure_ascii=False)[:200])


def t_parser_multiline_value():
    body = ("\n<function=note>\n<parameter=text>\nline one\nline two\n"
            "</parameter>\n</function>\n")
    name, args = gw.parse_qwen_tool_call(body)
    check(name == "note" and args == {"text": "line one\nline two"},
          "parser/multi-line parameter value", f"{name!r} {args!r}")


def t_parser_tolerant():
    r, c, tc = gw.parse_completion("never closed thinking", True)
    ok1 = r == "never closed thinking" and c == ""
    r, c, tc = gw.parse_completion(
        "x</think>pre<tool_call>broken<parameter=", True)
    ok2 = c == "pre<tool_call>broken<parameter=" and tc == []
    r, c, tc = gw.parse_completion(
        "a</think>b<|im_end|>TRAILING", True)
    ok3 = c == "b"
    # garbage between two tool_call blocks -> malformed -> plain content
    r, c, tc = gw.parse_completion(
        "<tool_call>\n<function=f>\n</function>\n</tool_call>junk"
        "<tool_call>\n<function=g>\n</function>\n</tool_call>", False)
    ok4 = tc == [] and "<tool_call>" in c
    check(ok1 and ok2 and ok3 and ok4,
          "parser/tolerant tails (unterminated, malformed, EOS cut, "
          "inter-block garbage)", f"{r!r} {c!r}")


def t_parser_bare_call():
    name, args = gw.parse_qwen_tool_call("\n<function=f>\n</function>\n")
    check(name == "f" and args == {}, "parser/bare name-only call")
    try:
        gw.parse_qwen_tool_call("\n<function=<\n</function>")
        ok = False
    except ValueError:
        ok = True
    check(ok, "parser/malformed name raises")


def t_think_splitter():
    text = THINK_TEXT + "</think>" + ANSWER_TEXT
    whole = gw.ThinkSplitter(True)
    acc = []
    for piece in [text]:
        acc += whole.feed(piece)
    acc += whole.flush()
    ch = gw.ThinkSplitter(True)
    acc2 = []
    for ch_piece in text:
        acc2 += ch.feed(ch_piece)
    acc2 += ch.flush()

    def joined(events):
        r = "".join(t for k, t in events if k == "reasoning_content")
        c = "".join(t for k, t in events if k == "content")
        return r, c
    r1, c1 = joined(acc)
    r2, c2 = joined(acc2)
    check(r1 == r2 == THINK_TEXT and c1 == c2 == ANSWER_TEXT,
          "ThinkSplitter char-by-char exactness", f"{r2!r} {c2!r}")


# ---------------------------------------------------------------------------
# 3. HTTP chat (model_chat)

def t_http_basics(h):
    s, b = h.call("GET", "/health")
    ok1 = s == 200 and b["status"] == "ok" and b["engine"] == "up"
    s, b = h.call("GET", "/v1/models")
    ok2 = (s == 200 and b["object"] == "list"
           and b["data"][0]["id"] == "model_chat")
    check(ok1 and ok2, "http/health + models")


def t_http_chat_thinking(h):
    s, b = h.call("POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hello"}]})
    m = b["choices"][0]["message"]
    ok = (s == 200 and b["object"] == "chat.completion"
          and m["role"] == "assistant"
          and m.get("reasoning_content") == THINK_TEXT
          and m["content"] == ANSWER_TEXT
          and b["choices"][0]["finish_reason"] == "stop"
          and b["usage"]["completion_tokens"] == 5
          and b["usage"]["prompt_tokens"] > 0)
    check(ok, "http/chat thinking shape + usage",
          json.dumps(b, ensure_ascii=False)[:300])


def t_http_chat_mode(h):
    # model card API: chat_template_kwargs.enable_thinking
    s, b = h.call("POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": {"enable_thinking": False}})
    m = b["choices"][0]["message"]
    ok1 = (s == 200 and "reasoning_content" not in m
           and m["content"] == ANSWER_TEXT
           and b["usage"]["completion_tokens"] == 3)
    # legacy alias "thinking" still honored
    s, b = h.call("POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": {"thinking": False}})
    m = b["choices"][0]["message"]
    ok2 = (s == 200 and "reasoning_content" not in m
           and m["content"] == ANSWER_TEXT)
    check(ok1 and ok2,
          "http/chat mode (enable_thinking false + legacy alias)",
          json.dumps(m)[:200])


def t_http_usage(h):
    s, enc = h.call("POST", "/debug/encode", {
        "messages": [{"role": "user", "content": "hello"}]})
    s, b = h.call("POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hello"}]})
    check(b["usage"]["prompt_tokens"] == len(enc["ids"]),
          "http/usage == /debug/encode ids")


def t_http_stop(h):
    for stop in (["STOP"], "STOP"):
        s, b = h.call("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hello"}],
            "stop": stop})
        ok = (b["choices"][0]["message"]["content"] == "The answer is "
              and b["choices"][0]["finish_reason"] == "stop")
    check(ok, "http/stop strings (list + bare string)",
          json.dumps(b["choices"][0])[:200])


def t_http_seed(h):
    body = {"messages": [{"role": "user", "content": "hello"}],
            "seed": 7}
    s, a = h.call("POST", "/v1/chat/completions", body)
    s, b = h.call("POST", "/v1/chat/completions", body)
    check(a["choices"][0]["message"] == b["choices"][0]["message"],
          "http/seed determinism")


def t_http_sse(h):
    c = http.client.HTTPConnection("127.0.0.1", h.port, timeout=60)
    body = json.dumps({"messages": [{"role": "user", "content": "hello"}],
                       "stream": True,
                       "stream_options": {"include_usage": True}})
    c.request("POST", "/v1/chat/completions", body=body,
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    blob = r.read().decode()
    c.close()
    lines = [l[6:] for l in blob.split("\n") if l.startswith("data: ")]
    chunks = [json.loads(l) for l in lines if l != "[DONE]"]
    ok = lines[-1] == "[DONE]" and r.status == 200
    deltas = [ch["choices"][0]["delta"] for ch in chunks
              if ch.get("choices")]
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    finish = [ch for ch in chunks
              if ch.get("choices")
              and ch["choices"][0].get("finish_reason")]
    usage = [ch for ch in chunks if not ch.get("choices")]
    ok = (ok and deltas[0] == {"role": "assistant"}
          and reasoning == THINK_TEXT
          and content == ANSWER_TEXT
          and finish and finish[-1]["choices"][0]["finish_reason"] == "stop"
          and usage and usage[-1]["usage"]["completion_tokens"] == 5)
    check(ok, "http/SSE shape, order, reassembly, usage chunk",
          blob[:300].replace("\n", "\\n"))


def t_http_concurrency(h):
    results = [None] * 6

    def work(i):
        results[i] = h.call("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hello"}]})

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = all(s == 200
             and b["choices"][0]["message"]["content"] == ANSWER_TEXT
             and b["choices"][0]["message"].get("reasoning_content")
             == THINK_TEXT
             for s, b in results)
    check(ok, "http/6-way concurrency serialization")


def t_http_errors(h):
    s, b = h.call("POST", "/v1/chat/completions", None,
                  headers={"Content-Type": "application/json"}, raw=False)
    ok1 = s == 400 and b["error"]["type"] == "invalid_request_error"
    s, b = h.call("POST", "/v1/chat/completions", {"messages": "nope"})
    ok2 = s == 400
    s, b = h.call("POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "other"})
    ok3 = s == 404 and b["error"]["type"] == "not_found_error"
    s, b = h.call("POST", "/v1/nope", {})
    ok4 = s == 404
    s, b = h.call("GET", "/nope")
    ok5 = s == 404
    check(ok1 and ok2 and ok3 and ok4 and ok5,
          "http/error shapes (400/404)", json.dumps(b)[:160])


def t_http_completions(h):
    s, b = h.call("POST", "/v1/completions", {"prompt": "weather>"})
    ok1 = (s == 200 and b["object"] == "text_completion"
           and b["choices"][0]["text"] == (THINK_TEXT + "</think>"
                                           + ANSWER_TEXT)
           and b["choices"][0]["finish_reason"] == "stop")
    s, blob = h.call("POST", "/v1/completions",
                     {"prompt": "weather>", "stream": True}, raw=True)
    text = blob.decode()
    ok2 = (s == 200 and "data: [DONE]" in text
           and "text_completion" in text)
    check(ok1 and ok2, "http//v1/completions (+SSE)",
          json.dumps(b, ensure_ascii=False)[:200])


def t_http_dual_eos(h):
    # "alt#" raw chain ends with <|endoftext|> (258): the second id of
    # the config eos list [257, 258] must also stop generation
    s, b = h.call("POST", "/v1/completions", {"prompt": "alt#"})
    ok = (s == 200 and b["choices"][0]["text"] == "ALT"
          and b["choices"][0]["finish_reason"] == "stop"
          and b["usage"]["completion_tokens"] == 1)
    check(ok, "http/dual EOS (<|endoftext|> 258 stops)",
          json.dumps(b)[:200])


def t_http_utf8_guard(h):
    # "tilde~" chain emits the two bytes of "é" as separate tokens; the
    # engine's UTF-8 guard must hold the partial first byte and complete
    # it on the second token
    s, b = h.call("POST", "/v1/completions", {"prompt": "tilde~"})
    ok1 = (s == 200 and b["choices"][0]["text"] == "é"
           and b["usage"]["completion_tokens"] == 2)
    s, blob = h.call("POST", "/v1/completions",
                     {"prompt": "tilde~", "stream": True}, raw=True)
    text = blob.decode("utf-8")     # would raise on invalid UTF-8
    pieces = [json.loads(l[6:])["choices"][0]["text"]
              for l in text.split("\n")
              if l.startswith("data: ") and l[6:] != "[DONE]"]
    ok2 = s == 200 and "".join(pieces) == "é"
    check(ok1 and ok2, "http/UTF-8 split-char guard (é over 2 tokens)",
          f"{pieces!r}")


def t_http_presence_penalty(h):
    base = {"messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4}
    s, a = h.call("POST", "/v1/chat/completions", dict(base))
    s, b = h.call("POST", "/v1/chat/completions",
                  dict(base, presence_penalty=0.0))
    ok1 = a["choices"][0]["message"] == b["choices"][0]["message"]
    s, c = h.call("POST", "/v1/chat/completions",
                  dict(base, presence_penalty=-100000.0))
    m = c["choices"][0]["message"]
    ok2 = (m.get("reasoning_content", "") == THINK_TEXT * 4
           and c["choices"][0]["finish_reason"] == "length")
    check(ok1 and ok2,
          "http/presence_penalty (off-default == 0.0; -1e5 loops)",
          json.dumps(m)[:200])


def t_http_encode_parity(h):
    msgs = [{"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1",
             "reasoning_content": "r1"},
            {"role": "user", "content": "q2"}]
    s, a = h.call("POST", "/debug/encode", {"messages": msgs})
    p = Pipe(MODEL_CHAT)
    ev = p.request({"id": 1, "cmd": "encode", "messages": msgs,
                    "thinking": True})[0]
    p.close()
    check(a["text"] == ev["text"] and a["ids"] == ev["ids"],
          "http/gateway-vs-pipe encode equality (multi-turn "
          "reasoning_content)")


def t_http_conformance(h):
    cases = json.load(open(os.path.join(FIX, "conformance.json")))
    nbad = 0
    for c in cases:
        kwargs = {"enable_thinking": bool(c.get("enable_thinking", 1))}
        if c.get("preserve_thinking"):
            kwargs["preserve_thinking"] = True
        body = {"messages": c["messages"],
                "chat_template_kwargs": kwargs}
        if c.get("tools"):
            body["tools"] = c["tools"]
        s, b = h.call("POST", "/debug/encode", body)
        if s != 200 or b["text"] != c["prompt"]:
            nbad += 1
            print(f"    conformance mismatch in {c['name']!r}:")
            print(f"      got:  {b.get('text', b)!r}")
            print(f"      want: {c['prompt']!r}")
    check(nbad == 0, f"http/chat-template conformance ({len(cases)} "
          "jinja goldens byte-exact, enable_thinking/preserve_thinking)")


# ---------------------------------------------------------------------------
# 4. HTTP tools (model_tools)

def gw_tools():
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location.",
            "parameters": {"type": "object",
                           "properties": {"location": {"type": "string"}},
                           "required": ["location"]},
        },
    }]


def t_tools_roundtrip(h):
    s, b = h.call("POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "weather in Beijing?"}],
        "tools": gw_tools()})
    m = b["choices"][0]["message"]
    tc = m.get("tool_calls", [])
    ok = (s == 200 and len(tc) == 1
          and tc[0]["type"] == "function"
          and tc[0]["function"]["name"] == "get_weather"
          and json.loads(tc[0]["function"]["arguments"])
          == {"location": "Beijing"}
          and m.get("reasoning_content") == TOOL_REASONING
          and b["choices"][0]["finish_reason"] == "tool_calls"
          and b["usage"]["completion_tokens"] == 3)
    check(ok, "tools/round trip (tool_calls JSON + finish_reason)",
          json.dumps(b, ensure_ascii=False)[:300])


def t_tools_stream(h):
    c = http.client.HTTPConnection("127.0.0.1", h.port, timeout=60)
    body = json.dumps({
        "messages": [{"role": "user", "content": "weather in Beijing?"}],
        "tools": gw_tools(), "stream": True})
    c.request("POST", "/v1/chat/completions", body=body,
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    blob = r.read().decode()
    c.close()
    lines = [json.loads(l[6:]) for l in blob.split("\n")
             if l.startswith("data: ") and l[6:] != "[DONE]"]
    finish = [ch for ch in lines
              if ch.get("choices")
              and ch["choices"][0].get("finish_reason")]
    ok = (finish
          and finish[-1]["choices"][0]["finish_reason"] == "tool_calls")
    check(ok, "tools/streaming finish_reason", blob[-200:].replace("\n",
          "\\n"))


def t_tools_followup(h):
    cases = json.load(open(os.path.join(FIX, "conformance.json")))
    want = [c for c in cases if c["name"] == "tool_roundtrip"][0]
    s, b = h.call("POST", "/debug/encode",
                  {"messages": want["messages"], "tools": want["tools"]})
    ok = (s == 200
          and "\n<tool_response>\nsunny, 20C\n</tool_response>" in b["text"]
          and b["text"] == want["prompt"])
    check(ok, "tools/role=tool follow-up rendering (<tool_response>, "
          "byte-exact vs jinja)", b.get("text", "")[-120:])


# ---------------------------------------------------------------------------
# 5. auth

def t_auth():
    h = Http(MODEL_CHAT, env_extra={"APUS_API_KEY": "sekret"})
    s, b = h.call("GET", "/health")
    ok1 = s == 200
    s, b = h.call("GET", "/v1/models")
    ok2 = s == 401 and b["error"]["type"] == "authentication_error"
    s, b = h.call("GET", "/v1/models",
                  headers={"Authorization": "Bearer sekret"})
    ok3 = s == 200
    s, b = h.call("POST", "/v1/chat/completions",
                  {"messages": [{"role": "user", "content": "hi"}]},
                  headers={"Authorization": "Bearer wrong"})
    ok4 = s == 401
    h.close()
    check(ok1 and ok2 and ok3 and ok4,
          "auth/APUS_API_KEY (401/200, open /health)")


# ---------------------------------------------------------------------------
# 6. tools/chat.py

def t_chat_py():
    script = ("/system You are helpful.\n"
              "hello\n"
              "/thinking off\n"
              "/preserve on\n"
              "again\n"
              "/raw weather>\n"
              "/reset\n"
              "/quit\n")
    p = subprocess.Popen([sys.executable, CHAT_PY, "--model", MODEL_CHAT],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True)
    out, _ = p.communicate(script, timeout=120)
    ok = ("(system message set)" in out
          and THINK_TEXT in out
          and "\n---\n" in out           # thinking dimming boundary
          and "(thinking off)" in out
          and "(preserve_thinking on)" in out
          and out.count(ANSWER_TEXT) >= 3
          and "finish_reason" not in out
          and "[5 tok" in out
          and "[3 tok" in out
          and "(history cleared)" in out)
    check(ok, "chat.py commands, thinking dimming, preserve, multi-turn, "
          "raw, reset", out[:300].replace("\033", "<ESC>"))


# ---------------------------------------------------------------------------

def main():
    print("test_server: serving milestone tests (apus-qwen M7)")

    print("-- pipe protocol")
    t_pipe_encode()
    t_pipe_errors()
    t_pipe_generate()
    t_pipe_usage()
    t_pipe_stop_string()
    t_pipe_max_tokens_and_ids()
    t_pipe_dual_eos()
    t_pipe_presence_penalty()

    print("-- Qwen XML parser")
    t_parser_thinking()
    t_parser_tool_call()
    t_parser_multi_and_typed()
    t_parser_multiline_value()
    t_parser_tolerant()
    t_parser_bare_call()
    t_think_splitter()

    print("-- HTTP chat (model_chat)")
    h = Http(MODEL_CHAT)
    t_http_basics(h)
    t_http_chat_thinking(h)
    t_http_chat_mode(h)
    t_http_usage(h)
    t_http_stop(h)
    t_http_seed(h)
    t_http_sse(h)
    t_http_concurrency(h)
    t_http_errors(h)
    t_http_completions(h)
    t_http_dual_eos(h)
    t_http_utf8_guard(h)
    t_http_presence_penalty(h)
    t_http_encode_parity(h)
    t_http_conformance(h)
    h.close()

    print("-- HTTP tools (model_tools)")
    h = Http(MODEL_TOOLS)
    t_tools_roundtrip(h)
    t_tools_stream(h)
    t_tools_followup(h)
    h.close()

    print("-- auth")
    t_auth()

    print("-- chat.py")
    t_chat_py()

    print(f"test_server: {PASS + FAIL} checks, {FAIL} failures")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
