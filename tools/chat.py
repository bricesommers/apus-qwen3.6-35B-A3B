#!/usr/bin/env python3
"""apus-qwen interactive terminal chat — colibri `coli chat` style.

Drives `bin/apus-qwen serve` (NDJSON stdio protocol, see
tests/m7a/README.md) directly: no HTTP gateway, no port, no deps beyond
the Python stdlib. The engine loads once; each turn re-prefills the
conversation (no KV reuse yet), so later turns get slower — keep history
short or /reset. Conversation history is kept client-side and re-encoded
through the Qwen ChatML template each turn (c/encoding.h splits the
reasoning back out of the stored reply on the next encode).

Usage:
    python3 tools/chat.py --model weights/apus-qwen --tiered
    python3 tools/chat.py --model tests/m7a/fixtures/model_chat

In-chat commands:
    /quit              exit (also Ctrl-D / Ctrl-C)
    /reset             clear conversation history
    /system <text>     set/replace the system message
    /thinking on|off   toggle thinking mode (default on)
    /preserve on|off   preserve thinking in ALL history turns
                       (chat_template_kwargs.preserve_thinking; default off)
    /temp X            sampling temperature (0 = greedy)
    /max N             max tokens per reply
    /raw <text>        raw completion mode (no chat template) for this turn
    /help              show commands
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class Engine:
    def __init__(self, model, apus, tiered):
        cmd = [apus, "serve", "--model", model]
        if tiered:
            cmd.append("--tiered")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
        self.next_id = 0

    def request(self, req):
        self.next_id += 1
        req["id"] = self.next_id
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return self.next_id

    def events(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                yield json.loads(line)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except BaseException:          # timeout AND a second Ctrl-C
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except BaseException:
                pass


def main():
    ap = argparse.ArgumentParser(description="apus-qwen terminal chat")
    ap.add_argument("--model", required=True)
    ap.add_argument("--apus",
                    default=os.environ.get(
                        "APUS_BIN", os.path.join(ROOT, "bin", "apus-qwen")))
    ap.add_argument("--tiered", action="store_true")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--preserve-thinking", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    thinking = (os.environ.get("APUS_THINKING", "1") != "0"
                and not args.no_thinking)
    preserve = args.preserve_thinking
    temp, max_tokens = args.temp, args.max_tokens
    messages = []
    eng = Engine(args.model, args.apus, args.tiered)

    print("apus-qwen chat — /help for commands, /quit to exit", flush=True)

    try:
        while True:
            try:
                line = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                parts = line.split(None, 1)
                cmd = parts[0]
                arg = parts[1] if len(parts) > 1 else ""
                if cmd == "/quit":
                    break
                elif cmd == "/reset":
                    messages = []
                    print("(history cleared)")
                elif cmd == "/system":
                    messages = [m for m in messages
                                if m.get("role") != "system"]
                    if arg:
                        messages.insert(0, {"role": "system", "content": arg})
                    print("(system message set)" if arg
                          else "(system message removed)")
                elif cmd == "/thinking":
                    thinking = arg.strip().lower() not in ("off", "0", "false")
                    print(f"(thinking {'on' if thinking else 'off'})")
                elif cmd == "/preserve":
                    preserve = arg.strip().lower() in ("on", "1", "true")
                    print(f"(preserve_thinking "
                          f"{'on' if preserve else 'off'})")
                elif cmd == "/temp":
                    temp = float(arg)
                    print(f"(temperature {temp})")
                elif cmd == "/max":
                    max_tokens = int(arg)
                    print(f"(max_tokens {max_tokens})")
                elif cmd == "/raw":
                    if not arg:
                        continue
                    req = {"cmd": "generate", "text": arg,
                           "max_tokens": max_tokens, "temperature": temp,
                           "top_p": args.top_p, "top_k": args.top_k,
                           "seed": args.seed}
                    _stream_turn(eng, req, None)
                elif cmd == "/help":
                    print(__doc__)
                else:
                    print(f"(unknown command {cmd}; /help)")
                continue

            messages.append({"role": "user", "content": line})
            req = {"cmd": "generate", "messages": messages,
                   "thinking": thinking, "preserve_thinking": preserve,
                   "max_tokens": max_tokens,
                   "temperature": temp, "top_p": args.top_p,
                   "top_k": args.top_k, "seed": args.seed}
            reply = _stream_turn(eng, req, thinking)
            if reply is not None:
                messages.append({"role": "assistant", "content": reply})
    finally:
        eng.close()


def _stream_turn(eng, req, thinking):
    """Send one generate request, stream tokens to the terminal.

    Returns the assembled reply text, or None on error. When `thinking`
    is true the reasoning (up to </think>) is printed dimmed."""
    rid = eng.request(req)
    DIM, RESET = "\033[2m", "\033[0m"
    parts = []
    in_think = bool(thinking)
    printed_any = False
    t0 = time.time()
    print("apus-qwen> ", end="", flush=True)
    if in_think:
        print(DIM, end="", flush=True)
    done = None
    for ev in eng.events():
        if ev.get("id") != rid:
            continue
        t = ev.get("type")
        if t == "token":
            text = ev.get("text", "")
            parts.append(text)
            if in_think and "</think>" in text:
                pre, post = text.split("</think>", 1)
                print(pre, end="", flush=True)
                print(RESET + "\n---\napus-qwen> " + post, end="",
                      flush=True)
                in_think = False
            else:
                print(text, end="", flush=True)
            printed_any = True
        elif t == "done":
            done = ev
            break
        elif t == "error":
            print(RESET + f"\n(engine error: {ev.get('message')})")
            return None
    if in_think:
        print(RESET, end="", flush=True)
    dt = time.time() - t0
    if done:
        n = done.get("completion_tokens", 0)
        print(f"\n[{n} tok, {dt:.0f}s, {n/dt:.2f} tok/s, "
              f"{done.get('finish_reason')}]", flush=True)
    elif printed_any:
        print(flush=True)
    return "".join(parts)


if __name__ == "__main__":
    main()
