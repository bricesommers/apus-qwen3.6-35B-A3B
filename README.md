# apus-qwen — Qwen3.6-35B-A3B on consumer hardware

A dependency-free C11 inference engine for
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
(35B-total / 3B-active hybrid-linear MoE, 67 GiB of BF16 weights,
text-only), running **fully locally on a 32 GB MacBook** by streaming
routed experts from NVMe through a bounded RAM cache. Nothing leaves the
machine.

Qwen3.6-35B-A3B uses a hybrid architecture: 30 Gated DeltaNet
linear-attention layers + 10 gated full-attention layers (3:1), 256
routed experts per MoE layer on all 40 layers (top-8, fp32-softmax
router), one MTP layer. This engine implements all of it from scratch in
C — including the gated delta-rule recurrence — against a numerics
contract written from the HF modeling code (`docs/M4-CONTRACT.md`).

**Status: M0–M13 complete, real-model validated (M11, 2026-08-30).**
Measured on the dev Mac (M1 Pro, 32 GB, tiered defaults): **~2–3 tok/s
decode**, 3.2 tok/s short-prompt / 24.3 tok/s long-prompt (batched)
prefill, ~2 s load, **8.3 GiB peak RSS**, ~1.1 GiB/token of expert
NVMe traffic at 79.6% pilot recall with zero demand stalls. The standing
golden stream (`tests/m11/golden.txt`) is bitwise-reproducible across
thread counts, cache sizes, pilot depths, speculative decoding, and the
Metal backend; `tests/m11/check_golden.sh` re-gates it. See
`docs/STATUS.md` for the milestone log and `docs/ARCHITECTURE.md` for
the full design.

- **Quality invariant**: insufficient fast memory costs *speed only,
  never quality* — the tiered store, pilot prefetch, and speculative
  decoding are gated to produce **bitwise-identical tokens** to the eager
  path. Numerics are pinned by a scalar anchor with SIMD paths proven
  bitwise or inside a measured, bounded reorder class.
- **Platforms**: macOS/Apple Silicon (primary), Linux/x86_64 (AVX2;
  Docker test harness), and native Windows (MSYS2 UCRT64; CI-gated).
- **Serving**: OpenAI-compatible HTTP gateway (SSE streaming, thinking
  mode, tool calls) + a terminal chat client; a browser setup (Open
  WebUI on Docker) is documented in `docs/USAGE.md`. The engine itself
  opens no sockets.

## Requirements

The model lives on disk, not in RAM — routed experts are streamed from
NVMe through a bounded cache (~1.1 GiB/token of reads), so RAM needs
are modest and **insufficient RAM costs speed only, never quality**
(proven on the real model: an 8 GB RSS budget with 1,980 forced expert
evictions still produces the bitwise-golden token stream).

- **RAM**: 8.3 GiB peak at defaults (measured) — **16 GB machines work**,
  that is the practical floor: close heavy apps and shrink the cache,
  e.g. `APUS_EXPERT_CACHE_MB=1024 APUS_RSS_GUARD_MB=11264`. 24 GB+ needs
  no care; the published numbers were measured on 32 GB.
- **Disk**: 67 GiB for the converted weights (~135 GiB peak during
  download+convert; the container is platform-independent, so it can be
  built on another machine and copied, or live on an external drive via
  symlink).
- **Speed expectations**: ~2–3 tok/s decode on an M1 Pro; a base M1/M2
  (⅓ the memory bandwidth) lands around ~1–1.5 tok/s. This engine
  trades speed for the ability to run a 35B MoE locally at all.

## Quickstart (macOS / Linux)

```sh
git clone <this repo>
cd Apus-Qwen3.6-35B-A3B

python3 -m venv .venv && .venv/bin/pip install numpy safetensors tokenizers huggingface_hub jinja2

# Download + convert the weights (~67 GiB, resumable, byte-verified;
# takes ~1 h). Peak disk ≈ 135 GiB.
.venv/bin/python tools/download.py --repo Qwen/Qwen3.6-35B-A3B \
    --work weights/work-qwen --out weights/apus-qwen

make bin/apus-qwen        # macOS: also `make metal` for the GPU backend

# Chat
.venv/bin/python tools/chat.py --model weights/apus-qwen --tiered

# Or serve an OpenAI-compatible API on localhost:8080
.venv/bin/python tools/server.py --model weights/apus-qwen --tiered --port 8080
```

Details, knobs, and expectations: `docs/USAGE.md`. Design:
`docs/ARCHITECTURE.md`. Numerics contract: `docs/M4-CONTRACT.md`.

## Testing

Every subsystem is gated (details in `tests/*/README.md`). The full
milestone battery (M0–M13) is green on macOS and Linux/x86_64, with
Windows gated in CI; `tests/m11/` pins the standing real-model golden.

## License & attribution

Source-available under **PolyForm Noncommercial 1.0.0**
(`LICENCE.PolyForm-Noncommercial`; noncommercial use only). Third-party
components: colibri (Apache-2.0), fla-core (MIT), vLLM (Apache-2.0),
Qwen3.6-35B-A3B model (Apache-2.0) — see `NOTICE`. The model weights are
downloaded separately from Hugging Face under Qwen's own terms.
