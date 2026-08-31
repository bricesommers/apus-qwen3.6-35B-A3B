# apus-qwen — root Makefile
# Milestone M2: tokenizer (c/tok.h) + chat/message encoding (c/encoding.h).
# Milestone M3: BF16 kernels (c/bf16.h + c/pool.h).

UNAME   := $(shell uname)
# clang on macOS, gcc on Linux/x86_64 (same warning set, as Apus).
# M13 (Apus M15, ported across the adapter seam): Windows via MinGW-w64
# gcc (MSYS2 UCRT64 shell: uname is MINGW64_NT-* and OS=Windows_NT is
# always set). No MSVC support (C11 + GNU extensions). Windows uses
# -std=gnu11 (not strict c11): MinGW hides strdup/clock_gettime et al.
# behind __STRICT_ANSI__; gnu11 exposes them. FP semantics unchanged
# (-ffp-contract=off stays pinned).
ifeq ($(OS),Windows_NT)
STD     := -std=gnu11
else
STD     := -std=c11
endif
ifeq ($(UNAME),Darwin)
CC      := clang
else
CC      := gcc
endif
CFLAGS  ?= $(STD) -O2 -Wall -Wextra
# Pin FP mul+add contraction OFF on every platform (Apus convention: same
# flags, same bits, on CI and dev machines). No FP in M2, but the flag stays.
CFLAGS  += -ffp-contract=off
ifeq ($(OS),Windows_NT)
# M13: MinGW-w64. -D_GNU_SOURCE does not exist here; the POSIX surface is
# shimmed in c/compat.h. The -fno-tree-vectorize pair below mirrors the
# Linux flags (numerics no-op; keeps the x86 anchor builds consistent).
CFLAGS  += -fno-tree-vectorize -fno-tree-slp-vectorize
else ifneq ($(UNAME),Darwin)
# Linux: under -std=c11 glibc hides pread/posix_memalign/strdup/
# clock_gettime behind feature-test macros; _GNU_SOURCE exposes them.
CFLAGS  += -D_GNU_SOURCE
# M12a-1 (Apus playbook): gcc -O2's auto-vectorizers emit sequences that
# Rosetta's x86_64 translator mis-executes (SIGTRAP under linux/amd64
# emulation; proven numerics-noop on native x86_64 — the pinned kernels
# are explicit intrinsics / scalar, nothing relies on auto-vectorization).
CFLAGS  += -fno-tree-vectorize -fno-tree-slp-vectorize
endif
LDLIBS  := -lm
ifeq ($(OS),Windows_NT)
# M13: -lpsapi for GetProcessMemoryInfo (compat.h RSS). -static bundles
# the winpthreads/mingw runtime — a MinGW .exe otherwise needs
# libwinpthread-1.dll on PATH (silent exit 127 when missing), and static
# linking makes the binaries portable for end users anyway.
LDLIBS  += -lpsapi -static
endif
# M9b: Accelerate (system AMX BLAS) on macOS for the M>=128 prefill
# dispatch (c/blas.h); a no-op stub elsewhere.
ifeq ($(UNAME),Darwin)
LDLIBS  += -framework Accelerate
endif

M2   := tests/m2
BIN  := $(M2)/bin
M3   := tests/m3
BIN3 := $(M3)/bin
M4A  := tests/m4a
BIN4 := $(M4A)/bin
M4C  := tests/m4c
BIN5 := $(M4C)/bin
M5   := tests/m5
BIN6 := $(M5)/bin
PY   := .venv/bin/python

M2_DEPS := c/json.h c/tok.h c/uni_tables.h c/uni_nfc.h
M3_DEPS := c/bf16.h c/pool.h c/x86.h
M4A_DEPS := c/bf16.h c/gdn.h c/attn.h c/moe.h c/x86.h
M4C_DEPS := $(M4A_DEPS) c/layer.h
M5_DEPS := $(M4C_DEPS) c/st.h c/model.h c/sample.h c/json.h

.PHONY: all test-m2 ubsan-m2 golden golden-exhaustive uni-tables clean \
        test-m3 ubsan-m3 bench-m3 golden-m3 \
        test-m4a ubsan-m4a golden-m4a \
        test-m4b test-m4c ubsan-m4c golden-m4b \
        test-m5 ubsan-m5 golden-m5 \
        test-m6a ubsan-m6a bench-m6a golden-m6a \
        test-m6b ubsan-m6b golden-m6b \
        test-m7a ubsan-m7a golden-m7a test-m6c ubsan-m6c \
        test-m8 ubsan-m8 golden-m8 test-m9a ubsan-m9a bench-m9a \
        profile-m9a \
        test-m9b ubsan-m9b bench-m9b \
        test-m12a2 ubsan-m12a2 bench-m12a2 \
        test-m9c ubsan-m9c bench-m9c profile-m9c \
        metal test-m10 ubsan-m10 bench-m10

all: $(BIN)/test_tok $(BIN)/test_encoding $(BIN3)/test_bf16 $(BIN3)/bench_bf16 \
     $(BIN4)/test_gdn $(BIN4)/test_attn $(BIN4)/test_moe $(BIN5)/test_layer \
     bin/apus-qwen $(BIN6)/test_full

$(BIN):
	mkdir -p $(BIN)

$(BIN)/test_tok: $(M2)/test_tok.c $(M2_DEPS) | $(BIN)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS)

$(BIN)/test_encoding: $(M2)/test_encoding.c $(M2_DEPS) c/encoding.h | $(BIN)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS)

# Unicode/NFC tables are GENERATED from the reference tokenizer's own
# behavior (see tests/m2/README.md); regenerate on every golden run.
uni-tables:
	$(PY) $(M2)/gen_uni_tables.py .
	$(PY) $(M2)/gen_nfc_tables.py .

golden: uni-tables
	$(PY) $(M2)/gen_golden.py

golden-exhaustive: uni-tables
	$(PY) $(M2)/gen_golden.py --exhaustive

test-m2: all golden
	./$(BIN)/test_tok
	./$(BIN)/test_encoding

# UBSan-only variant (Apple's ASan runtime is broken on this machine;
# ASan is intentionally absent, like Apus)
ubsan-m2: CFLAGS = -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer
ubsan-m2: $(BIN)/test_tok_ubsan $(BIN)/test_encoding_ubsan golden
	./$(BIN)/test_tok_ubsan
	./$(BIN)/test_encoding_ubsan

$(BIN)/test_tok_ubsan: $(M2)/test_tok.c $(M2_DEPS) | $(BIN)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS)

$(BIN)/test_encoding_ubsan: $(M2)/test_encoding.c $(M2_DEPS) c/encoding.h | $(BIN)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS)

# --- M3: BF16 kernels (c/bf16.h) + thread pool (c/pool.h) ------------------
# Hard gate: NEON/mt kernels BITWISE == the scalar anchor (c/bf16.h
# contract); exhaustive widen/narrow (all 65,536 codes), numpy f64 goldens,
# shape sweep incl. the real Ling shapes. The mt paths use the pool, so the
# Makefile diffs stdout across APUS_THREADS=1/4/8 (thread independence).

$(BIN3):
	mkdir -p $(BIN3)

$(BIN3)/test_bf16: $(M3)/test_bf16.c $(M3_DEPS) | $(BIN3)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN3)/bench_bf16: $(M3)/bench_bf16.c $(M3_DEPS) | $(BIN3)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN3)/test_bf16_ubsan: $(M3)/test_bf16.c $(M3_DEPS) | $(BIN3)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m3:
	$(PY) $(M3)/gen_golden.py

test-m3: $(BIN3)/test_bf16 golden-m3
	APUS_THREADS=1 ./$(BIN3)/test_bf16 > $(BIN3)/out_t1.txt
	APUS_THREADS=4 ./$(BIN3)/test_bf16 > $(BIN3)/out_t4.txt
	APUS_THREADS=8 ./$(BIN3)/test_bf16 > $(BIN3)/out_t8.txt
	diff $(BIN3)/out_t1.txt $(BIN3)/out_t4.txt
	diff $(BIN3)/out_t1.txt $(BIN3)/out_t8.txt
	cat $(BIN3)/out_t4.txt

# UBSan-only variant (Apple's ASan runtime is broken on this machine;
# ASan is intentionally absent, like Apus and ubsan-m2)
ubsan-m3: $(BIN3)/test_bf16_ubsan golden-m3
	APUS_THREADS=1 ./$(BIN3)/test_bf16_ubsan > $(BIN3)/out_u1.txt
	APUS_THREADS=4 ./$(BIN3)/test_bf16_ubsan > $(BIN3)/out_u4.txt
	diff $(BIN3)/out_u1.txt $(BIN3)/out_u4.txt
	cat $(BIN3)/out_u4.txt

bench-m3: $(BIN3)/bench_bf16
	./$(BIN3)/bench_bf16

# --- M4a: GDN/GQA/MoE per-op kernels (c/gdn.h, c/attn.h, c/moe.h) ---------
# Hard gates: numpy f64 goldens (esc-based tolerances; fp32-out ops at
# 1e-5 of esc, bf16-out ops at ~2 bf16 ulp), BITWISE scalar-anchor ==
# NEON for every new op, BITWISE path-equivalence gates (conv prefill ==
# decode-stepping, recurrence prefill == step loop, GQA decode ==
# full-recompute row, mt == sequential), deterministic router
# tie-breaks. The mt paths use the pool, so the Makefile diffs stdout
# across APUS_THREADS=1/4/8 (thread independence, like test-m3).

$(BIN4):
	mkdir -p $(BIN4)

$(BIN4)/test_gdn: $(M4A)/test_gdn.c $(M4A_DEPS) | $(BIN4)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN4)/test_attn: $(M4A)/test_attn.c $(M4A_DEPS) | $(BIN4)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN4)/test_moe: $(M4A)/test_moe.c $(M4A_DEPS) | $(BIN4)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN4)/test_gdn_ubsan: $(M4A)/test_gdn.c $(M4A_DEPS) | $(BIN4)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN4)/test_attn_ubsan: $(M4A)/test_attn.c $(M4A_DEPS) | $(BIN4)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN4)/test_moe_ubsan: $(M4A)/test_moe.c $(M4A_DEPS) | $(BIN4)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m4a:
	$(PY) $(M4A)/gen_golden.py

test-m4a: $(BIN4)/test_gdn $(BIN4)/test_attn $(BIN4)/test_moe golden-m4a
	APUS_THREADS=1 ./$(BIN4)/test_gdn > $(BIN4)/out_gdn_t1.txt
	APUS_THREADS=4 ./$(BIN4)/test_gdn > $(BIN4)/out_gdn_t4.txt
	APUS_THREADS=8 ./$(BIN4)/test_gdn > $(BIN4)/out_gdn_t8.txt
	diff $(BIN4)/out_gdn_t1.txt $(BIN4)/out_gdn_t4.txt
	diff $(BIN4)/out_gdn_t1.txt $(BIN4)/out_gdn_t8.txt
	cat $(BIN4)/out_gdn_t4.txt
	APUS_THREADS=1 ./$(BIN4)/test_attn > $(BIN4)/out_attn_t1.txt
	APUS_THREADS=4 ./$(BIN4)/test_attn > $(BIN4)/out_attn_t4.txt
	APUS_THREADS=8 ./$(BIN4)/test_attn > $(BIN4)/out_attn_t8.txt
	diff $(BIN4)/out_attn_t1.txt $(BIN4)/out_attn_t4.txt
	diff $(BIN4)/out_attn_t1.txt $(BIN4)/out_attn_t8.txt
	cat $(BIN4)/out_attn_t4.txt
	APUS_THREADS=1 ./$(BIN4)/test_moe > $(BIN4)/out_moe_t1.txt
	APUS_THREADS=4 ./$(BIN4)/test_moe > $(BIN4)/out_moe_t4.txt
	APUS_THREADS=8 ./$(BIN4)/test_moe > $(BIN4)/out_moe_t8.txt
	diff $(BIN4)/out_moe_t1.txt $(BIN4)/out_moe_t4.txt
	diff $(BIN4)/out_moe_t1.txt $(BIN4)/out_moe_t8.txt
	cat $(BIN4)/out_moe_t4.txt

# UBSan-only variant (Apple's ASan runtime is broken on this machine;
# ASan is intentionally absent, like ubsan-m2/ubsan-m3)
ubsan-m4a: $(BIN4)/test_gdn_ubsan $(BIN4)/test_attn_ubsan $(BIN4)/test_moe_ubsan golden-m4a
	./$(BIN4)/test_gdn_ubsan
	./$(BIN4)/test_attn_ubsan
	./$(BIN4)/test_moe_ubsan

# --- M4b/c: single-layer forward (c/layer.h + tools/oracle.py) -------------
# Hard gate: per-stage goldens from the numpy oracle (f32 = C target, f64
# = truth; C must land inside the f32-vs-f64 envelope + realization
# slack), oracle self-checks (replay determinism, router selection
# identity, oracle chunk invariance), and C-side chunk invariance
# BITWISE (one per-token body). The mt kernel paths use the pool, so the
# Makefile diffs stdout across APUS_THREADS=1/4/8 (thread independence).

$(BIN5):
	mkdir -p $(BIN5)

$(BIN5)/test_layer: $(M4C)/test_layer.c $(M4C_DEPS) | $(BIN5)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN5)/test_layer_ubsan: $(M4C)/test_layer.c $(M4C_DEPS) | $(BIN5)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m4b:
	$(PY) tests/m4b/gen_fixtures.py
	$(PY) tests/m4b/check_oracle.py

# M4b IS the oracle-fixture gate (generation + oracle self-checks)
test-m4b: golden-m4b

test-m4c: $(BIN5)/test_layer golden-m4b
	APUS_THREADS=1 ./$(BIN5)/test_layer > $(BIN5)/out_t1.txt
	APUS_THREADS=4 ./$(BIN5)/test_layer > $(BIN5)/out_t4.txt
	APUS_THREADS=8 ./$(BIN5)/test_layer > $(BIN5)/out_t8.txt
	diff $(BIN5)/out_t1.txt $(BIN5)/out_t4.txt
	diff $(BIN5)/out_t1.txt $(BIN5)/out_t8.txt
	cat $(BIN5)/out_t4.txt

# UBSan-only variant (same rationale as ubsan-m4a)
ubsan-m4c: $(BIN5)/test_layer_ubsan golden-m4b
	./$(BIN5)/test_layer_ubsan

# --- M5: full mini-model forward + sampling --------------------------------
# Hard gate: full-model logits vs the f32 oracle inside the f32-vs-f64
# envelope per sequence, greedy near-tie / sampled CDF-margin flip
# policies (0 unexcused), chunk invariance + determinism BITWISE, CLI
# smoke on the fixture model (the token stream diffed against the oracle
# golden). The mt kernel paths use the pool, so stdout is diffed across
# APUS_THREADS=1/4/8 (thread independence, like test-m4c).

bin:
	mkdir -p bin

bin/apus-qwen: c/apus-qwen.c $(wildcard c/*.h) | bin
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN6):
	mkdir -p $(BIN6)

$(BIN6)/test_full: $(M5)/test_full.c $(M5_DEPS) | $(BIN6)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN6)/test_full_ubsan: $(M5)/test_full.c $(M5_DEPS) | $(BIN6)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m5:
	$(PY) $(M5)/gen_fixtures.py

test-m5: $(BIN6)/test_full bin/apus-qwen golden-m5
	APUS_THREADS=1 ./$(BIN6)/test_full > $(BIN6)/out_t1.txt
	APUS_THREADS=4 ./$(BIN6)/test_full > $(BIN6)/out_t4.txt
	APUS_THREADS=8 ./$(BIN6)/test_full > $(BIN6)/out_t8.txt
	diff $(BIN6)/out_t1.txt $(BIN6)/out_t4.txt
	diff $(BIN6)/out_t1.txt $(BIN6)/out_t8.txt
	cat $(BIN6)/out_t4.txt
	./bin/apus-qwen run --model $(M5)/fixtures/model \
	    --ids "3,1,4,1,5,9,2,6" --max-tokens 8 --greedy > $(BIN6)/cli_greedy.txt
	diff $(BIN6)/cli_greedy.txt $(M5)/fixtures/cli_greedy.txt
	cat $(BIN6)/cli_greedy.txt

# UBSan-only variant (same rationale as ubsan-m4a)
ubsan-m5: $(BIN6)/test_full_ubsan golden-m5
	./$(BIN6)/test_full_ubsan

# --- M6a: expert store (c/cache.h) ------------------------------------------
# Hard gate: eager vs store at every cache size (32/16/2 slots, 2 + RSS
# 1-byte drops, 4 + 2 pins, synchronous I/O) — tokens AND all logits
# BITWISE identical. Store unit tests on the fixture container. The
# invariance digest must be identical at APUS_IO_THREADS=1/4/8 (the I/O
# pool must not perturb compute bits).

M6A  := tests/m6a
BIN7 := $(M6A)/bin
M6B  := tests/m6b
BIN8 := $(M6B)/bin
M6_DEPS := $(M5_DEPS) c/compat.h c/cache.h c/pilot.h

$(BIN7):
	mkdir -p $(BIN7)

$(BIN8):
	mkdir -p $(BIN8)

$(BIN7)/test_store: $(M6A)/test_store.c $(M6_DEPS) | $(BIN7)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN7)/test_invariance: $(M6A)/test_invariance.c $(M6_DEPS) | $(BIN7)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN7)/bench_m6a: $(M6A)/bench_m6a.c $(M6_DEPS) | $(BIN7)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN7)/test_store_ubsan: $(M6A)/test_store.c $(M6_DEPS) | $(BIN7)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN7)/test_invariance_ubsan: $(M6A)/test_invariance.c $(M6_DEPS) | $(BIN7)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m6a: golden-m5
	$(PY) $(M6A)/gen_fixtures.py

test-m6a: $(BIN7)/test_store $(BIN7)/test_invariance golden-m6a
	./$(BIN7)/test_store
	APUS_IO_THREADS=1 ./$(BIN7)/test_invariance > $(BIN7)/out_i1.txt 2>/dev/null
	APUS_IO_THREADS=4 ./$(BIN7)/test_invariance > $(BIN7)/out_i4.txt 2>/dev/null
	APUS_IO_THREADS=8 ./$(BIN7)/test_invariance > $(BIN7)/out_i8.txt 2>/dev/null
	diff $(BIN7)/out_i1.txt $(BIN7)/out_i4.txt
	diff $(BIN7)/out_i1.txt $(BIN7)/out_i8.txt
	cat $(BIN7)/out_i4.txt

ubsan-m6a: $(BIN7)/test_store_ubsan $(BIN7)/test_invariance_ubsan golden-m6a
	./$(BIN7)/test_store_ubsan
	./$(BIN7)/test_invariance_ubsan

bench-m6a: $(BIN7)/bench_m6a golden-m6a
	./$(BIN7)/bench_m6a

# --- M6b: router-lookahead pilot (c/pilot.h) --------------------------------
# Hard gate: pilot on/off x cache sizes — BITWISE tokens+logits; recall
# counters == Python recompute exactly; ring/hook unit tests.

$(BIN8)/test_pilot: $(M6B)/test_pilot.c $(M6_DEPS) | $(BIN8)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN8)/test_invariance: $(M6B)/test_invariance.c $(M6_DEPS) | $(BIN8)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN8)/test_recall: $(M6B)/test_recall.c $(M6_DEPS) | $(BIN8)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN8)/test_pilot_ubsan: $(M6B)/test_pilot.c $(M6_DEPS) | $(BIN8)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN8)/test_invariance_ubsan: $(M6B)/test_invariance.c $(M6_DEPS) | $(BIN8)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN8)/test_recall_ubsan: $(M6B)/test_recall.c $(M6_DEPS) | $(BIN8)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m6b: golden-m6a

# the pilot "issued/stale" counts are thread-timing dependent, so the
# I/O-thread diff covers everything EXCEPT those informational lines
test-m6b: $(BIN8)/test_pilot $(BIN8)/test_invariance $(BIN8)/test_recall golden-m6b
	./$(BIN8)/test_pilot
	APUS_IO_THREADS=1 ./$(BIN8)/test_invariance > $(BIN8)/out_i1.txt 2>/dev/null
	APUS_IO_THREADS=4 ./$(BIN8)/test_invariance > $(BIN8)/out_i4.txt 2>/dev/null
	APUS_IO_THREADS=8 ./$(BIN8)/test_invariance > $(BIN8)/out_i8.txt 2>/dev/null
	grep -v "pilot:" $(BIN8)/out_i1.txt > $(BIN8)/out_i1.d
	grep -v "pilot:" $(BIN8)/out_i4.txt > $(BIN8)/out_i4.d
	grep -v "pilot:" $(BIN8)/out_i8.txt > $(BIN8)/out_i8.d
	diff $(BIN8)/out_i1.d $(BIN8)/out_i4.d
	diff $(BIN8)/out_i1.d $(BIN8)/out_i8.d
	cat $(BIN8)/out_i4.txt
	./$(BIN8)/test_recall
	$(PY) $(M6B)/check_recall.py

ubsan-m6b: $(BIN8)/test_pilot_ubsan $(BIN8)/test_invariance_ubsan $(BIN8)/test_recall_ubsan golden-m6b
	./$(BIN8)/test_pilot_ubsan
	./$(BIN8)/test_invariance_ubsan
	./$(BIN8)/test_recall_ubsan
	$(PY) $(M6B)/check_recall.py

# --- M6c: performance pass (NEON/mt wiring + batched prefill) ---------------
# Hard gate: every hot kernel BITWISE == its scalar anchor; batched
# prefill == sequential (logits AND state, eager AND tiered). The FNV
# digest must be identical at APUS_THREADS=1/4/8.

M6C  := tests/m6c
BIN9 := $(M6C)/bin

$(BIN9):
	mkdir -p $(BIN9)

$(BIN9)/test_m6c: $(M6C)/test_m6c.c $(M6_DEPS) | $(BIN9)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN9)/test_m6c_ubsan: $(M6C)/test_m6c.c $(M6_DEPS) | $(BIN9)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

test-m6c: $(BIN9)/test_m6c golden-m5 golden-m6a
	APUS_THREADS=1 ./$(BIN9)/test_m6c > $(BIN9)/out_t1.txt 2>/dev/null
	APUS_THREADS=4 ./$(BIN9)/test_m6c > $(BIN9)/out_t4.txt 2>/dev/null
	APUS_THREADS=8 ./$(BIN9)/test_m6c > $(BIN9)/out_t8.txt 2>/dev/null
	diff $(BIN9)/out_t1.txt $(BIN9)/out_t4.txt
	diff $(BIN9)/out_t1.txt $(BIN9)/out_t8.txt
	cat $(BIN9)/out_t4.txt

ubsan-m6c: $(BIN9)/test_m6c_ubsan golden-m5 golden-m6a
	./$(BIN9)/test_m6c_ubsan

# --- M9b: reorder-class levers, re-measured on the Qwen shapes ---------
# The inherited user-approved bounded reorder classes (Ling M9b,
# 2026-08-07 — the approval carries only after re-measurement on the
# Qwen shapes, docs/ARCHITECTURE.md §9; THIS suite is that
# re-measurement): err/esc vs FP64 truth on all real Qwen shapes (bound
# 1e-4, the m3 class); T=1/4/8 thread-independence stays bitwise (fixed
# pool partition / tile grid).

M9B  := tests/m9b
BINC := $(M9B)/bin
M9B_DEPS := $(M6_DEPS) c/blas.h c/mtp.h

$(BINC):
	mkdir -p $(BINC)

$(BINC)/test_m9b: $(M9B)/test_m9b.c $(M9B_DEPS) | $(BINC)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BINC)/test_m9b_ubsan: $(M9B)/test_m9b.c $(M9B_DEPS) | $(BINC)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BINC)/bench_m9b: $(M9B)/bench_m9b.c $(M3_DEPS) c/blas.h | $(BINC)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

test-m9b: $(BINC)/test_m9b golden-m6a
	APUS_THREADS=1 ./$(BINC)/test_m9b > $(BINC)/out_t1.txt 2>/dev/null
	APUS_THREADS=4 ./$(BINC)/test_m9b > $(BINC)/out_t4.txt 2>/dev/null
	APUS_THREADS=8 ./$(BINC)/test_m9b > $(BINC)/out_t8.txt 2>/dev/null
	diff $(BINC)/out_t1.txt $(BINC)/out_t4.txt
	diff $(BINC)/out_t1.txt $(BINC)/out_t8.txt
	cat $(BINC)/out_t4.txt

ubsan-m9b: $(BINC)/test_m9b_ubsan golden-m6a
	./$(BINC)/test_m9b_ubsan

bench-m9b: $(BINC)/bench_m9b
	./$(BINC)/bench_m9b

# --- M9a: strictly-bitwise perf-pass gates, re-anchored to Qwen ---------
# The hot-GEMV wiring checks (err/esc vs in-test FP64 at every real Qwen
# shape + the 8/4/1-row tail boundaries; the frozen bitwise 8-chain
# kernel == the scalar anchor; the logits widen path) plus the
# fixture-forward profile driver (profile-m9a, sample(1) target).

M9A  := tests/m9a
BINB := $(M9A)/bin

$(BINB):
	mkdir -p $(BINB)

$(BINB)/test_m9a: $(M9A)/test_m9a.c $(M6_DEPS) | $(BINB)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BINB)/test_m9a_ubsan: $(M9A)/test_m9a.c $(M6_DEPS) | $(BINB)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BINB)/bench_m9a: $(M9A)/bench_m9a.c | $(BINB)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BINB)/profile_m9a: $(M9A)/profile_m9a.c $(M6_DEPS) | $(BINB)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

test-m9a: $(BINB)/test_m9a golden-m6a
	APUS_THREADS=1 ./$(BINB)/test_m9a > $(BINB)/out_t1.txt 2>/dev/null
	APUS_THREADS=4 ./$(BINB)/test_m9a > $(BINB)/out_t4.txt 2>/dev/null
	APUS_THREADS=8 ./$(BINB)/test_m9a > $(BINB)/out_t8.txt 2>/dev/null
	diff $(BINB)/out_t1.txt $(BINB)/out_t4.txt
	diff $(BINB)/out_t1.txt $(BINB)/out_t8.txt
	cat $(BINB)/out_t4.txt

ubsan-m9a: $(BINB)/test_m9a_ubsan golden-m6a
	./$(BINB)/test_m9a_ubsan

bench-m9a: $(BINB)/bench_m9a
	./$(BINB)/bench_m9a

profile-m9a: $(BINB)/profile_m9a golden-m5
	./$(BINB)/profile_m9a

# --- M9c: batched-prefill restoration gates (M-independent-bitwise) -----
# Hard gate: the M9 batched prefill (phase A: projection GEMMs at M=tc
# through the M-independent-bitwise gemm_hot, ONE batched conv1d, ONE
# chunked gqa_mt; phase B: shared expert at M=T + unique-expert
# gate_up/down at M=count — NO BLAS/gemm_fast anywhere) == the per-token
# body BITWISE (attention AND MoE traces + state at T=256/300 on
# synthetic layers of both kinds; model-level eager+tiered at T=64;
# tiered res1 hook stream at T=64 AND T=256). The FNV digest must be
# identical at APUS_THREADS=1/4/8.

M9C  := tests/m9c
BINE := $(M9C)/bin

$(BINE):
	mkdir -p $(BINE)

$(BINE)/test_m9c: $(M9C)/test_m9c.c $(M9B_DEPS) | $(BINE)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BINE)/test_m9c_ubsan: $(M9C)/test_m9c.c $(M9B_DEPS) | $(BINE)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BINE)/bench_m9c: $(M9C)/bench_m9c.c $(M3_DEPS) c/blas.h | $(BINE)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BINE)/profile_phasea: $(M9C)/profile_phasea.c $(M4C_DEPS) | $(BINE)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

test-m9c: $(BINE)/test_m9c golden-m5 golden-m6a
	APUS_THREADS=1 ./$(BINE)/test_m9c > $(BINE)/out_t1.txt 2>/dev/null
	APUS_THREADS=4 ./$(BINE)/test_m9c > $(BINE)/out_t4.txt 2>/dev/null
	APUS_THREADS=8 ./$(BINE)/test_m9c > $(BINE)/out_t8.txt 2>/dev/null
	diff $(BINE)/out_t1.txt $(BINE)/out_t4.txt
	diff $(BINE)/out_t1.txt $(BINE)/out_t8.txt
	cat $(BINE)/out_t4.txt

ubsan-m9c: $(BINE)/test_m9c_ubsan golden-m5 golden-m6a
	./$(BINE)/test_m9c_ubsan

bench-m9c: $(BINE)/bench_m9c
	./$(BINE)/bench_m9c

profile-m9c: $(BINE)/profile_phasea
	./$(BINE)/profile_phasea

# --- M10: Metal GPU backend (c/backend_metal.mm) ---------------------------
# macOS-only, opt-in: bin/apus-qwen-metal links the backend; the plain CPU
# binary never links it and behaves exactly as before (weak stubs). Hard
# gates: kernel-level Metal == the DISPATCHED CPU kernels BITWISE (the
# shaders replicate the M9b ILP rounding sequences exactly), model-level
# greedy tokens CPU == Metal identical on the m5 fixture, tiered slab-safety
# (transient expert slabs never touch the pointer cache). Off-Darwin the
# targets are clear stubs (the Linux battery does not include them).

M10  := tests/m10
BINM := $(M10)/bin
METAL_LDLIBS := $(LDLIBS) -lpthread -lc++ -framework Metal -framework Foundation

ifeq ($(UNAME),Darwin)

$(BINM):
	mkdir -p $(BINM)

# The backend compiles separately as Objective-C++ (no -std=c11; the
# same warning/FP flags otherwise). UBSan builds reuse the same object —
# sanitization stays on the C side (the Apus m7b precedent).
$(BINM)/backend_metal.o: c/backend_metal.mm c/backend_metal.h | $(BINM)
	$(CC) -O2 -Wall -Wextra -ffp-contract=off -Ic -c -o $@ c/backend_metal.mm

bin/apus-qwen-metal: c/apus-qwen.c $(wildcard c/*.h) $(BINM)/backend_metal.o | bin
	$(CC) $(CFLAGS) -Ic -o $@ c/apus-qwen.c $(BINM)/backend_metal.o $(METAL_LDLIBS)

$(BINM)/test_kernels: $(M10)/test_kernels.c $(M9B_DEPS) $(BINM)/backend_metal.o | $(BINM)
	$(CC) $(CFLAGS) -Ic -o $@ $(M10)/test_kernels.c $(BINM)/backend_metal.o $(METAL_LDLIBS)

$(BINM)/test_model: $(M10)/test_model.c $(M9B_DEPS) $(BINM)/backend_metal.o | $(BINM)
	$(CC) $(CFLAGS) -Ic -o $@ $(M10)/test_model.c $(BINM)/backend_metal.o $(METAL_LDLIBS)

$(BINM)/test_model_ubsan: $(M10)/test_model.c $(M9B_DEPS) $(BINM)/backend_metal.o | $(BINM)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $(M10)/test_model.c $(BINM)/backend_metal.o $(METAL_LDLIBS)

$(BINM)/bench_metal: $(M10)/bench_metal.c $(M3_DEPS) c/backend_metal.h $(BINM)/backend_metal.o | $(BINM)
	$(CC) $(CFLAGS) -Ic -o $@ $(M10)/bench_metal.c $(BINM)/backend_metal.o $(METAL_LDLIBS)

metal: bin/apus-qwen-metal

test-m10: $(BINM)/test_kernels $(BINM)/test_model golden-m5 golden-m6a
	./$(BINM)/test_kernels
	./$(BINM)/test_model

ubsan-m10: $(BINM)/test_model_ubsan golden-m5 golden-m6a
	./$(BINM)/test_model_ubsan

bench-m10: $(BINM)/bench_metal
	./$(BINM)/bench_metal

else

metal test-m10 ubsan-m10 bench-m10:
	@echo "$@: the Metal backend is macOS-only — skipped on this platform"

endif

# --- M12: AVX2 x86-64 kernels (c/x86.h) --------------------------------------
# Hard gate: every AVX2 kernel BITWISE == the scalar anchor it replaces
# (staged single-rounded products, scalar sequential order per output, no
# FMA); exhaustive widen (all 65536 codes); the FNV digest must be
# identical at APUS_THREADS=1/4/8. The Qwen GDN/GQA kernels run the
# DOCUMENTED scalar fallback on x86 — the suite pins that no AVX2 kernel
# is reachable through them (the hit counter must not move). Off x86-64
# the suite is a trivial pass (the macOS battery keeps the target list
# platform-uniform).

M12  := tests/m12
BIND := $(M12)/bin
M12_DEPS := c/bf16.h c/moe.h c/gdn.h c/attn.h c/pool.h c/x86.h

$(BIND):
	mkdir -p $(BIND)

$(BIND)/test_m12a2: $(M12)/test_m12a2.c $(M12_DEPS) | $(BIND)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIND)/test_m12a2_ubsan: $(M12)/test_m12a2.c $(M12_DEPS) | $(BIND)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIND)/bench_m12a2: $(M12)/bench_m12a2.c $(M12_DEPS) | $(BIND)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

test-m12a2: $(BIND)/test_m12a2
	APUS_THREADS=1 ./$(BIND)/test_m12a2 > $(BIND)/out_t1.txt 2>/dev/null
	APUS_THREADS=4 ./$(BIND)/test_m12a2 > $(BIND)/out_t4.txt 2>/dev/null
	APUS_THREADS=8 ./$(BIND)/test_m12a2 > $(BIND)/out_t8.txt 2>/dev/null
	diff $(BIND)/out_t1.txt $(BIND)/out_t4.txt
	diff $(BIND)/out_t1.txt $(BIND)/out_t8.txt
	cat $(BIND)/out_t4.txt

ubsan-m12a2: $(BIND)/test_m12a2_ubsan
	./$(BIND)/test_m12a2_ubsan

bench-m12a2: $(BIND)/bench_m12a2
	APUS_THREADS=1 ./$(BIND)/bench_m12a2

# --- M8: MTP speculative decoding (c/mtp.h) ---------------------------------
# Hard gate: spec == non-spec token streams BITWISE at K=2/3/4, greedy +
# sampled; state digests bitwise after rollback; forced draft patterns;
# tiered (store layer-4 slabs) bitwise; digest diffed at APUS_THREADS=
# 1/4/8.

M8   := tests/m8
BINA := $(M8)/bin

$(BINA):
	mkdir -p $(BINA)

$(BINA)/test_mtp: $(M8)/test_mtp.c $(M6_DEPS) c/mtp.h | $(BINA)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BINA)/test_mtp_ubsan: $(M8)/test_mtp.c $(M6_DEPS) c/mtp.h | $(BINA)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m8: golden-m6a
	$(PY) $(M8)/gen_fixtures.py

test-m8: $(BINA)/test_mtp golden-m8
	APUS_THREADS=1 ./$(BINA)/test_mtp > $(BINA)/out_t1.txt 2>/dev/null
	APUS_THREADS=4 ./$(BINA)/test_mtp > $(BINA)/out_t4.txt 2>/dev/null
	APUS_THREADS=8 ./$(BINA)/test_mtp > $(BINA)/out_t8.txt 2>/dev/null
	diff $(BINA)/out_t1.txt $(BINA)/out_t4.txt
	diff $(BINA)/out_t1.txt $(BINA)/out_t8.txt
	cat $(BINA)/out_t4.txt

ubsan-m8: $(BINA)/test_mtp_ubsan golden-m8
	./$(BINA)/test_mtp_ubsan

# --- M7: serving (serve subcommand + tools/server.py + tools/chat.py) -------
# Full HTTP path through the scripted parrot fixtures: pipe protocol,
# Qwen XML tool-call parser, OpenAI endpoints + SSE, thinking on/off,
# preserve_thinking, tool-call round trip, dual-EOS stop, presence_penalty,
# UTF-8 guard, concurrency serialization, errors, auth, jinja byte-exact
# conformance, chat.py session.

M7A  := tests/m7a

golden-m7a: golden-m5
	$(PY) $(M7A)/gen_fixtures.py

bin/apus-qwen_ubsan: c/apus-qwen.c $(M6_DEPS) c/tok.h c/encoding.h c/uni_tables.h c/uni_nfc.h | bin
	$(CC) -std=c11 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

test-m7a: bin/apus-qwen golden-m7a
	$(PY) $(M7A)/test_server.py

ubsan-m7a: bin/apus-qwen_ubsan golden-m7a
	APUS_BIN=bin/apus-qwen_ubsan $(PY) $(M7A)/test_server.py

clean:
	rm -rf $(BIN) $(M2)/golden $(M2)/nfc_meta.json $(BIN3) $(M3)/golden \
	    $(BIN4) $(M4A)/golden $(BIN5) tests/m4b/fixtures \
	    $(BIN6) $(M5)/fixtures bin/apus-qwen $(BIN7) $(M6A)/fixtures \
	    $(BIN8) $(M7A)/fixtures bin/apus-qwen_ubsan $(BIN9) $(BINA) \
	    $(M8)/fixtures $(BINB) $(BINC) $(BIND) $(BINE) $(BINM) \
	    bin/apus-qwen-metal
