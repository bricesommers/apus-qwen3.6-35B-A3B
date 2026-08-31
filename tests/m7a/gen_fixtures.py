#!/usr/bin/env python3
"""tests/m7a/gen_fixtures.py — scripted "parrot" mini-models for the
serving tests, plus chat-template conformance goldens.

The server tests need a model whose generated token stream is a fixed,
known script (Qwen XML tool-call output, dual EOS, stop strings, exact
usage counts). Random weights cannot do that. Instead:

  * All layer weights are ZERO (attention/GDN + MoE weights, convs,
    A_log/dt_bias): every sublayer output is 0 and the bf16 residual
    preserves x exactly, so each of the 2 layers (GDN+MoE, full-attn+MoE)
    is the identity and the final hidden state depends ONLY on the last
    embedded token. The final RMSNorm (weight 1) rescales; logits =
    head @ f(embed[last]).
  * embed rows random N(0,1); head all zero except: for each scripted
    transition a -> b, head[b] = 8 * embed[a]. Decoding follows the
    scripted Markov chain (margin checked at gen time against the stored
    bf16 values).
  * Chain tokens with multi-character content are added tokens (ids
    300+). ChatML specials: 256-261 (<|im_start|>, <|im_end|> = EOS 257,
    <|endoftext|> = EOS 258, <think>, </think>, and "</think>\n\n" =
    261 — the non-thinking generation prompt tail; leftmost-longest
    added-token matching makes the chat-mode prompt end in 261 while the
    thinking-mode prompt ends in "<think>" + "\\n" byte 10, so the two
    modes get different chain entries). Byte-level vocab ids 0-255
    (GPT-2 bytes_to_unicode, no merges).
  * config.json eos_token_id = [257, 258] (the ARRAY form — exercises
    the M7 dual-EOS list parsing in c/model.h; <|im_end|> 257 is also
    the tokenizer-lookup EOS in c/apus-qwen.c).

Variants:
  fixtures/model_chat/   thinking reply with a stop-string trigger, a
                         shared chat-mode entry, a raw-completions entry
                         ('>'), an EOS-258 chain ('#'), and a UTF-8
                         split-char chain ('~')
  fixtures/model_tools/  thinking + a complete Qwen XML tool call
                         (get_weather) + EOS

Also writes fixtures/conformance.json: conversations rendered with
jinja2 from reference/chat_template.jinja (the tests/m2 approach) for
byte-exact comparison through the full server path.

Run from the repository root. Deterministic (seed 20260910).
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests", "m1"))

import stutil  # noqa: E402

SEED = 20260910
M5_MODEL = os.path.join(ROOT, "tests", "m5", "fixtures", "model")
M5_SHARD = "apus-qwen-00001.safetensors"
VOCAB = 512

EMBED_T = "model.language_model.embed_tokens.weight"
NORM_T = "model.language_model.norm.weight"
HEAD_T = "lm_head.weight"

# ChatML specials (mirror the real model's marker roles)
IM_START, IM_END, ENDOFTEXT = 256, 257, 258
THINK_START, THINK_END, THINK_OFF_TAIL = 259, 260, 261
SP = {
    IM_START: "<|im_start|>",
    IM_END: "<|im_end|>",
    ENDOFTEXT: "<|endoftext|>",
    THINK_START: "<think>",
    THINK_END: "</think>",
    THINK_OFF_TAIL: "</think>\n\n",
}
EOS_IDS = [IM_END, ENDOFTEXT]          # config eos_token_id (array form)
NL_BYTE = 0x0A        # thinking-mode generation prompt ends "<think>\n"
GT_BYTE = 0x3E        # ">" : raw-completions chain entry
HASH_BYTE = 0x23      # "#" : EOS-258 chain entry
TILDE_BYTE = 0x7E     # "~" : UTF-8 split-char chain entry

CHAT_CHUNKS = {
    300: "reasoning: thinking it over.",
    301: "The answer is ",
    302: "STOP",
    303: " right here.",
    304: "ALT",
}
# thinking chain: "reasoning: thinking it over.</think>The answer is
# STOP right here." + <|im_end|>
CHAT_THINK_CHAIN = [300, THINK_END, 301, 302, 303, IM_END]
# chat mode (thinking off): the same answer without the reasoning part
CHAT_OFF_CHAIN = [301, 302, 303, IM_END]
# EOS-258 chain: "ALT" + <|endoftext|>
ALT_CHAIN = [304, ENDOFTEXT]
# UTF-8 guard chain: bytes of "é" split across two tokens + EOS
UTF8_CHAIN = [0xC3, 0xA9, ENDOFTEXT]

CHAT_ENTRIES = {
    NL_BYTE: CHAT_THINK_CHAIN,        # thinking-mode prompt tail
    THINK_OFF_TAIL: CHAT_OFF_CHAIN,   # chat-mode prompt tail
    GT_BYTE: CHAT_THINK_CHAIN,        # raw "weather>" prompt
    HASH_BYTE: ALT_CHAIN,             # raw "alt#" prompt
    TILDE_BYTE: UTF8_CHAIN,           # raw "tilde~" prompt
}

TOOLS_CHUNKS = {
    300: "I should check the weather.",
    301: "\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\n"
         "Beijing\n</parameter>\n</function>\n</tool_call>",
}
# thinking text: "I should check the weather.</think>\n\n<tool_call>
#   \n<function=get_weather>\n<parameter=location>\nBeijing\n</parameter>
#   \n</function>\n</tool_call>" + <|im_end|>
TOOLS_THINK_CHAIN = [300, THINK_END, 301, IM_END]
TOOLS_OFF_CHAIN = [301, IM_END]

TOOLS_ENTRIES = {
    NL_BYTE: TOOLS_THINK_CHAIN,
    THINK_OFF_TAIL: TOOLS_OFF_CHAIN,
}

VARIANTS = {
    "model_chat": (CHAT_CHUNKS, CHAT_ENTRIES),
    "model_tools": (TOOLS_CHUNKS, TOOLS_ENTRIES),
}


def bytes_to_unicode():
    bs = (list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD))
          + list(range(0xAE, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = bytes_to_unicode()


def write_tokenizer(path, chunks):
    vocab = {B2U[b]: b for b in range(256)}
    added = [
        {"id": i, "content": c, "special": True, "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False}
        for i, c in sorted(SP.items())
    ]
    added += [
        {"id": i, "content": c, "special": False, "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False}
        for i, c in sorted(chunks.items())
    ]
    tok = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                          "trim_offsets": True, "use_regex": False},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                    "trim_offsets": True, "use_regex": False},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False,
                  "byte_fallback": False, "vocab": vocab, "merges": []},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False, indent=1)


def bf16_narrow(f32):
    """float32 -> bf16 codes (numpy round-to-nearest-even)."""
    f = np.asarray(f32, dtype=np.float32)
    u = f.view(np.uint32).copy()
    u += np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (u >> np.uint32(16)).astype(np.uint16)


def bf16_bytes(f32):
    return bf16_narrow(np.asarray(f32, dtype=np.float32)).tobytes()


def build_variant(name, chunks, entries):
    out_dir = os.path.join(HERE, "fixtures", name)
    os.makedirs(out_dir, exist_ok=True)
    hdr, ds = stutil.read_shard(os.path.join(M5_MODEL, M5_SHARD))
    raw = stutil.read_tensor_bytes(os.path.join(M5_MODEL, M5_SHARD), hdr, ds)
    with open(os.path.join(M5_MODEL, "config.json")) as f:
        config = json.load(f)
    tc = config["text_config"]
    dim = tc["hidden_size"]

    rng = np.random.default_rng(SEED)
    embed = rng.standard_normal((VOCAB, dim)).astype(np.float32)
    head = np.zeros((VOCAB, dim), dtype=np.float32)
    transitions = {}
    for entry, chain in entries.items():
        transitions[entry] = chain[0]
        for a, b in zip(chain, chain[1:]):
            assert a not in transitions or transitions[a] == b, (
                f"{name}: conflicting successor for {a}")
            transitions[a] = b
    for a, b in transitions.items():
        head[b] += np.float32(8.0) * embed[a]

    tensors = []
    for tname in sorted(hdr):
        if tname in (EMBED_T, NORM_T, HEAD_T):
            continue                     # replaced below (vocab resized)
        meta = hdr[tname]
        tensors.append((tname, meta["dtype"], meta["shape"],
                        b"\x00" * len(raw[tname])))
    tensors.append((EMBED_T, "BF16", (VOCAB, dim), bf16_bytes(embed)))
    tensors.append((NORM_T, "BF16", (dim,),
                    bf16_bytes(np.ones(dim, dtype=np.float32))))
    tensors.append((HEAD_T, "BF16", (VOCAB, dim), bf16_bytes(head)))
    tensors.sort(key=lambda t: t[0])
    stutil.write_shard(os.path.join(out_dir, M5_SHARD), tensors)
    with open(os.path.join(out_dir, "model.safetensors.index.json"),
              "w") as f:
        json.dump({"metadata": {},
                   "weight_map": {t[0]: M5_SHARD for t in tensors}}, f)
    tc["vocab_size"] = VOCAB
    tc["eos_token_id"] = EOS_IDS         # M7: dual EOS, array form
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    write_tokenizer(os.path.join(out_dir, "tokenizer.json"), chunks)

    # margin self-check with the stored bf16 values
    def widen(b):
        u = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32)
    e = widen(bf16_bytes(embed)).reshape(VOCAB, dim)
    h = widen(bf16_bytes(head)).reshape(VOCAB, dim)
    yn = e / np.sqrt((e * e).mean(axis=1, keepdims=True) + 1e-6)
    logits = yn @ h.T
    worst = min(logits[a, b] - np.max(np.delete(logits[a], b))
                for a, b in transitions.items())
    print(f"  {name}: {len(transitions)} transitions, "
          f"min logit margin {worst:.1f}")
    assert worst > 100, "scripted chain margin too small"


# --- chat-template conformance goldens (tests/m2 approach) -------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string",
                             "description": "The city name."},
            },
            "required": ["location"],
        },
    },
}


def gen_conformance():
    import jinja2
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def tojson(x, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True,
                                        lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    tpl = env.from_string(
        open(os.path.join(ROOT, "reference", "chat_template.jinja")).read())

    cases = [
        {"name": "simple_thinking",
         "messages": [{"role": "user", "content": "hello"}]},
        {"name": "simple_chat",
         "messages": [{"role": "user", "content": "hello"}],
         "enable_thinking": 0},
        {"name": "system_user",
         "messages": [{"role": "system", "content": "You are helpful."},
                      {"role": "user", "content": "hi"}]},
        {"name": "multiturn_reasoning",
         "messages": [{"role": "user", "content": "q1"},
                      {"role": "assistant", "content": "a1",
                       "reasoning_content": "r1"},
                      {"role": "user", "content": "q2"}]},
        {"name": "preserve_thinking",
         "messages": [{"role": "user", "content": "q1"},
                      {"role": "assistant", "content": "a1",
                       "reasoning_content": "r1"},
                      {"role": "user", "content": "q2"}],
         "preserve_thinking": 1},
        {"name": "embedded_think",
         "messages": [{"role": "user", "content": "q"},
                      {"role": "assistant",
                       "content": "<think>\nembedded reasoning\n</think>\n"
                                  "final answer"}]},
        {"name": "tools_thinking",
         "messages": [{"role": "user", "content": "weather?"}],
         "tools": [WEATHER_TOOL]},
        {"name": "tool_roundtrip",
         "messages": [
             {"role": "user", "content": "weather in Beijing?"},
             {"role": "assistant", "content": "",
              "tool_calls": [{"id": "call_1", "type": "function",
                              "function": {"name": "get_weather",
                                           "arguments": {
                                               "location": "Beijing"}}}]},
             {"role": "tool", "content": "sunny, 20C"}],
         "tools": [WEATHER_TOOL]},
    ]
    out = []
    for c in cases:
        kwargs = {"messages": c["messages"], "add_generation_prompt": True}
        if c.get("tools"):
            kwargs["tools"] = c["tools"]
        kwargs["enable_thinking"] = bool(c.get("enable_thinking", 1))
        if c.get("preserve_thinking"):
            kwargs["preserve_thinking"] = True
        prompt = tpl.render(**kwargs)
        out.append(dict(c, prompt=prompt))
    with open(os.path.join(HERE, "fixtures", "conformance.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"  conformance: {len(out)} jinja-rendered cases")


def main():
    if not os.path.exists(os.path.join(M5_MODEL, M5_SHARD)):
        import subprocess
        subprocess.check_call(
            [sys.executable,
             os.path.join(ROOT, "tests", "m5", "gen_fixtures.py")], cwd=ROOT)
    for name, (chunks, entries) in VARIANTS.items():
        build_variant(name, chunks, entries)
    gen_conformance()
    print("m7a fixtures done")


if __name__ == "__main__":
    main()
