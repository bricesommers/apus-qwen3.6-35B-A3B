#!/bin/bash
# tools/docker/test-linux.sh — M12 Linux/x86_64 test harness (ported
# from the Apus/Ling tools/docker/test-linux.sh).
#
# Builds the dev image (cached; pip deps are NOT reinstalled unless
# tools/docker/Dockerfile.dev changes), mounts the repo READ-ONLY at /repo,
# copies the build/test inputs (c, tests, tools, reference, Makefile —
# NOT the weights/) into the container's own /src, and runs the given make
# targets there with the system python3.
#
# The read-only mount + in-container copy is deliberate: it keeps ELF build
# artifacts out of the macOS work tree (an in-place Linux build would
# overwrite bin/ and tests/*/bin, leaving the Mac side broken).
#
# Usage:
#   tools/docker/test-linux.sh                 # the full portable battery
#   tools/docker/test-linux.sh test-m3 test-m4c
#   tools/docker/test-linux.sh test-m1         # m0/m1 are python (unittest) suites
# Env: APUS_LINUX_IMAGE overrides the image tag.
#
# Note: linux/amd64 runs under Rosetta emulation on Apple Silicon — expect
# 5-20x slowdowns on compute-heavy suites.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${APUS_LINUX_IMAGE:-apus-qwen-linux-dev:m12a1}"

if [ $# -eq 0 ]; then
    set -- test-m2 test-m3 test-m4a test-m4b test-m4c test-m5 test-m6a \
           test-m6b test-m6c test-m7a test-m8 test-m9a test-m9b test-m9c \
           test-m12a2 test-m0 test-m1
fi

echo ">> building image $IMAGE (linux/amd64)"
docker build --platform linux/amd64 -t "$IMAGE" \
    -f "$ROOT/tools/docker/Dockerfile.dev" "$ROOT/tools/docker"

echo ">> running ${*} in-container"
docker run --rm -i --platform linux/amd64 \
    -v "$ROOT:/repo:ro" \
    "$IMAGE" bash -s -- "$@" <<'CONTAINER_EOF'
set -u
mkdir -p /src
cp -a /repo/c /repo/tests /repo/tools /repo/reference /repo/Makefile /src/
# Purge copied build artifacts: tests/*/bin and bin/ hold macOS (Mach-O)
# binaries fresh enough that make would consider them up-to-date and try to
# EXECUTE them (Exec format error). Deleting them forces clean ELF builds.
find /src -type d -name bin -exec rm -rf {} + 2>/dev/null || true
rm -rf /src/bin
cd /src
pass=""; fail=""
for t in "$@"; do
    echo
    echo "=== $t ==="
    if [ "$t" = "test-m0" ] || [ "$t" = "test-m1" ]; then
        # m0/m1 are python unittest suites, not make targets (tests/m0,
        # tests/m1 READMEs)
        if python3 -m unittest discover -s "tests/${t#test-}" 2>&1 | tail -5 \
            && [ "${PIPESTATUS[0]}" -eq 0 ]; then
            pass="$pass $t"
        else
            fail="$fail $t"
        fi
    elif make -s PY=python3 "$t"; then
        pass="$pass $t"
    else
        fail="$fail $t"
    fi
done
echo
echo "==================== summary ===================="
echo "pass:$pass"
echo "fail:$fail"
[ -z "$fail" ]
CONTAINER_EOF
