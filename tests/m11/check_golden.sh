#!/bin/sh
# M11 standing-golden gate: re-run the golden command on the real model
# and require the emitted stream to be byte-identical to golden.txt.
# Wall time ~20-30 s (model load + 15-token prefill + 24-token decode).
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if [ ! -d weights/apus-qwen ]; then
    echo "check_golden: weights/apus-qwen not present — skipping (weights are local-only)" >&2
    exit 2
fi
out="$(mktemp)"
trap 'rm -f "$out"' EXIT
./bin/apus-qwen run --model weights/apus-qwen --tiered \
    --prompt "The capital of France is" --max-tokens 24 --greedy \
    > "$out" 2>/dev/null
if cmp -s tests/m11/golden.txt "$out"; then
    echo "check_golden: OK — stream bitwise identical to tests/m11/golden.txt"
else
    echo "check_golden: FAIL — stream diverged from the standing golden:" >&2
    diff tests/m11/golden.txt "$out" >&2 || true
    exit 1
fi
