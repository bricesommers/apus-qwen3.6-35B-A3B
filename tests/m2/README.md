# M2 — BPE tokenizer + chat/message format (Qwen3.6-35B-A3B)

Port of the M2 milestone to `Qwen/Qwen3.6-35B-A3B` (Qwen2Tokenizer): the C11
BPE tokenizer (`c/tok.h`, with `c/json.h`, `c/uni_tables.h`, `c/uni_nfc.h`)
and the ChatML chat/message encoder (`c/encoding.h`), validated byte-exactly
against the Hugging Face reference.

## Layout

```
c/json.h                 JSON parser/serializer (Python json.dumps-compatible)
c/uni_tables.h           GENERATED: Unicode class ranges (N / L∪M / WS) for
                         the pre-tokenizer regex — do not edit
c/uni_nfc.h              GENERATED: NFC tables (decomp / ccc / comp) — do not edit
c/tok.h                  tokenizer: NFC -> Split(regex) -> byte-level BPE,
                         26 added tokens, ByteLevel decode
c/encoding.h             Qwen ChatML format (reference/chat_template.jinja port)
tests/m2/gen_uni_tables.py   generates c/uni_tables.h from the reference
                             tokenizer's own pre-tokenizer behavior
tests/m2/gen_nfc_tables.py   generates c/uni_nfc.h (+ tests/m2/nfc_meta.json)
tests/m2/gen_golden.py       generates tests/m2/golden/* (HF tokenizers + jinja2)
tests/m2/test_tok.c          tokenizer conformance battery
tests/m2/test_encoding.c     chat-format conformance battery
```

## What the Qwen tokenizer actually is (reference/tokenizer.json)

- `normalizer`: **NFC** (Rust unicode-normalization crate). Applied per span
  between added tokens (all 26 added tokens have `normalized=false`, so they
  are matched on the raw text first — the HF AddedVocabulary order).
- `pre_tokenizer`: `Split(regex, Isolated)` then
  `ByteLevel(add_prefix_space=false, use_regex=false)`. Single regex stage
  (GPT-2 style, Qwen2 flavor): contractions `'(?i:'s|'t|'re|'ve|'m|'ll|'d)`
  (the `(?i:)` fold is Unicode-aware — U+017F behaves as `s`), optional
  non-letter prefix + **letter/mark run `[\p{L}\p{M}]+`**, **`\p{N}` single
  digit char** (numbers split per digit), optional space + punct/symbol run
  + trailing CR/LFs, then the three whitespace alternatives. Leftmost match,
  alternation priority, greedy.
- `model`: byte-level BPE (GPT-2 `bytes_to_unicode` alphabet), 248,044 vocab
  entries + 247,587 merges (space-joined `"left right"` strings), ids
  0..248043.
- `added_tokens`: 26, ids 248044..248069 (`<|endoftext|>`, `<|im_start|>`,
  `<|im_end|>`, vision specials, `<tool_call>`, `<tool_response>`, `<think>`,
  FIM/repo tokens; `tok_n_tokens == 248070`; the config's `vocab_size 248320`
  is padded). Matched leftmost-longest.
- `decoder`/`post_processor`: ByteLevel; **no automatic BOS/EOS**
  (`add_bos_token=false`). EOS for generation is `<|im_end|>` (248046) or
  `<|endoftext|>` (248044) — a sampling concern, not M2.

### Differences from the Ling base's tokenizer

- Contraction alternatives: `'s|'t|'re|'ve|'m|'ll|'d` (was `[sdmt]|ll|ve|re`).
- The letter class is `[\p{L}\p{M}]`: **combining marks join letter runs**,
  not punct/symbol runs (in Ling they were "other"). The `uni_tables.h` "L"
  ranges therefore cover `\p{L}\p{M}` (744 ranges vs Ling's 675; the N and
  WS range sets are unchanged).
- `\s*[\r\n]+` vs Ling's `\s*[\r\n]`: identical matches (both end after the
  last newline of the whitespace run) — the C matcher is unchanged there.
- No possessive quantifiers; irrelevant for these disjoint classes.

## Class and NFC tables are probed, not databased

Both generated headers derive from the reference implementation's own
behavior, so they match whatever Unicode version the Rust crates use —
Python's unicodedata (15.1) provably differs (e.g. U+08CF is a ccc=230 mark
for Python but a plain starter for the crate; 108 such marks).

- `uni_tables.h`: per-codepoint probes classify N / L∪M / WS / OTHER purely
  by how the reference pre-tokenizer splits probe strings (see the docstring
  of `gen_uni_tables.py`). ASCII is hardcoded in `tok.h`.
- `uni_nfc.h`: decompositions probed from the crate's NFD (batched, `|`
  separator — a ccc=0 starter that blocks all cross-probe interaction);
  composition pairs admitted iff the crate's NFC maps the pair back (the
  composition-exclusion list falls out of ground truth; candidates include
  the non-recursive decompositions so chained compositions like
  U+01D5 <- (U+00DC, U+0304) work); the mark set (ccc != 0) detected
  behaviorally over every codepoint (`NFD(U+0300 X U+0316)` reorders iff X
  is a mark); ccc values from unicodedata plus ladder probes for unknowns;
  Hangul algorithmic (TR15 part 3). The generator self-checks a table-only
  Python NFC against the crate over all singles, all 810x810 ordered mark
  pairs, blocking triples and 20k random strings before writing anything.

**NFC carry-over verdict (2026-08-29):** regenerated from the Qwen
tokenizer's normalizer, `c/uni_nfc.h` + `tests/m2/nfc_meta.json` came out
**byte-identical** to the Ling base's tables (same crate: 810 marks, 1,339
composition pairs, 2,060 decomposition entries). The NFC machinery is
unchanged; only the class table changed (marks moved into L).

## Chat format (reference/chat_template.jinja — ChatML)

The repo ships **no conformance pairs**, so goldens are generated by rendering
the shipped template with jinja2 in the same environment HF
`apply_chat_template` uses: `ImmutableSandboxedEnvironment(trim_blocks=True,
lstrip_blocks=True, extensions=[loopcontrols])`, `tojson = json.dumps(...,
ensure_ascii=False)` (transformers overrides jinja2's HTML-escaping default),
and a `raise_exception` global. jinja2 is a Python-side test dependency only
(installed into `.venv`); the C engine stays dependency-free.

Semantics implemented in `c/encoding.h` (all verified byte-exact):

- **Tools preamble** (non-empty `tools` array): `<|im_start|>system\n` +
  `# Tools\n\nYou have access to the following functions:\n\n<tools>` +
  `\n` + `tojson(tool)` per tool + `\n</tools>` + the Qwen tool-call
  instruction text (the `<function=...>`/`<parameter=...>` XML format and
  the `<IMPORTANT>` reminders), then `\n\n` + trimmed system content **only
  if** messages[0] is a system message whose content trims to non-empty,
  then `<|im_end|>\n`.
- **Plain system**: `<|im_start|>system\n` + trimmed content + `<|im_end|>\n`
  (messages[0] only; a later system message is an error, as the template
  raises).
- **User**: `<|im_start|>user\n` + trimmed content + `<|im_end|>\n`.
- **Assistant thinking**: reasoning comes from `reasoning_content` (iff it is
  a string — null/absent falls through) or is split out of content on
  `</think>`/`<think>` with the template's exact Python
  `split/rstrip/lstrip` semantics; then `|trim`'d. It is **preserved** as
  `<|im_start|>assistant\n<think>\nRC\n</think>\n\nCONTENT` iff
  `preserve_thinking` or the turn sits after the last real user query
  (`last_query_index`: the last user message whose trimmed content is not
  wrapped in `<tool_response>...</tool_response>`; if none exists the
  template raises "No user query found in messages." — so do we). Older
  turns render thinking-free: `<|im_start|>assistant\nCONTENT`.
- **Tool calls**: `<tool_call>\n<function=NAME>\n` + per argument
  `<parameter=k>\nv\n</parameter>\n` (insertion order; `v` verbatim if
  string, else `json.dumps(v, ensure_ascii=False)`; arguments must be a JSON
  object) + `</function>\n</tool_call>`; the first call is preceded by
  `\n\n` iff the (trimmed) content is non-empty, later calls by `\n`.
  A `function` wrapper object is unwrapped when present and truthy.
- **Tool results**: consecutive `tool` messages group under ONE
  `<|im_start|>user` … `<|im_end|>\n` turn, each wrapped in
  `\n<tool_response>\n…\n</tool_response>`. A tool message in first position
  opens no `<|im_start|>user` (the template guards on `loop.previtem`).
- **Generation prompt**: `<|im_start|>assistant\n<think>\n` (thinking on,
  the default) or `<|im_start|>assistant\n<think>\n\n</think>\n\n`
  (`enable_thinking=False`).
- **Trimming**: every message content passes the template's `|trim` —
  Python `str.strip()` semantics (CPython `isspace`: 0x09-0x0D, 0x1C-0x1F,
  0x20, 0x85, 0xA0, 0x1680, 0x2000-0x200A, 0x2028, 0x2029, 0x202F, 0x205F,
  0x3000), NOT the regex `\s` class. `c/encoding.h` implements that set
  exactly.
- **Content shapes** (text-only scope): string, null (renders empty), or a
  list of `{"text": ...}` items; image/video items fail cleanly (the
  project strips the vision tower). Unknown roles, non-string scalar
  content, empty message lists, late system messages and missing user
  queries are clean errors, matching the template's `raise_exception`s.
- No BOS is ever added.

## Gates (all hard, byte-exact)

`make test-m2` (regenerates tables + goldens, then runs both batteries):

1. **Text cases** (8): encode twice (determinism), ids vs HF exactly;
   decode vs HF exactly; round-trip modulo NFC (`decode(encode(x)) ==
   nfc(x)`); nosplit variants (special strings as plain text).
2. **Specials** (26 records): every added token encodes to exactly its own
   id and decodes back to its content; no automatic BOS/EOS.
3. **NFC battery** (`nfc.bin`, 2,167,490 records): `tok_nfc` bytes vs the
   reference normalizer — all 1,111,936 single codepoints, all 656,100
   ordered mark pairs, every composition pair + blocking triples + blocked
   samples, all 399 L×V + 10,773 LV×T Hangul compositions, 100k random
   strings. **0 mismatches.**
4. **Codepoint battery** (`codepoints.bin`, `--exhaustive`, 1,242,655
   records): 5 probe strings per codepoint (0x00..0x2FFFF dense, stride 17
   above), ids vs HF exactly. **0 mismatches.**
5. **Chat battery** (27 cases): rendered prompt vs jinja2 golden
   byte-exactly (determinism checked by double render), token ids vs HF
   exactly, `ling_encode_ids` wrapper agreement, plus clean-failure paths
   (non-string scalar content, NULL/empty messages, unknown role, late
   system message, missing/wrapped user query, vision items).
6. **UBSan** (`make ubsan-m2`): both batteries under
   `-fsanitize=undefined -fno-omit-frame-pointer`, clean. ASan is
   intentionally absent (broken on the dev Mac, as in Apus); `leaks` is run
   on the test binaries instead.

## Results (2026-08-29, Apple Silicon, clang, `-std=c11 -O2 -Wall -Wextra -ffp-contract=off`)

- `test_tok`: **0 failures** — specials 26/26, nfc
  2,167,490 records 0 mismatches, codepoints 1,242,655 records 0 mismatches.
- `test_encoding`: **0 failures** — 27/27 chat cases byte-exact.
- `-Wall -Wextra` clean; UBSan clean; `leaks` clean.

## Probe scale vs the Ling base's M2

- Same scales: 1,242,655 codepoint probe records, 2,167,490 NFC records
  (0 mismatches both).
- The class table regenerated (marks moved into the L ranges); the NFC
  tables verified byte-identical to the base's (same Rust crate) and
  carried over.
- Chat goldens grew 20 -> 27 cases: the Qwen template's thinking stripping
  (last_query_index), `preserve_thinking`, the ChatML tools preamble,
  `<function=...>/<parameter=...>` tool calls, tool_response coalescing and
  the `|trim` semantics all get dedicated cases; the unknown-role and
  late-system cases moved to the error-path checks (this template raises
  where Ling's rendered nothing).
