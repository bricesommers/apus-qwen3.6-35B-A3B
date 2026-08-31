# tests/m6b — router-lookahead prefetch ("pilot")

Milestone M6b: the prefetch layer that feeds the M6a expert store,
ported from Apus (SPSC ring + pilot thread) and re-anchored to the
Qwen3.6 router (fp32-softmax top-k — NO bias/sigmoid/group stage; the
Ling router surface was dropped at M4). **Hard rule verified: the pilot
changes only WHEN/WHETHER an expert is in RAM, never numerics** — token
streams and logits are bitwise identical with pilot ON vs OFF across
every cache size.

Run:

```
make test-m6b    # 26 pilot checks + 18 invariance checks + recall audit
make ubsan-m6b   # same under -fsanitize=undefined
```

Fixtures: the M6a container (tests/m6a/fixtures, via golden-m6a).

## Pilot design (c/pilot.h)

**Prediction (dL=1).** The c/cache.h tiered forward fires a post-attention
hook with res1 (bf16) of layer L; the pilot computes layer L+1's router
input — L+1's OWN post_attention_layernorm (`apus_attn_rmsnorm`, the
block convention) applied to res1 — and scores it with `apus_moe_route`
(c/moe.h — the real router's own code path: bf16 matvec, FP32 softmax,
top-k with stable lowest-index ties, renormalize). The predicted set is
the first `pilot_k` entries of the router's top-k output (`pilot_k` >
top_k uses the same selection machinery at a deeper truncation,
`apus_moe_route_topn` — this is the fixture case: top_k 4, pilot_k 8).
NOTE: the proxy input lacks layer L+1's attention output, so the
prediction is approximate by design — see the recall caveat below.

**Delivery.** Predictions go onto a bounded SPSC lock-free ring
(`(pos:32|layer:16|eid:16)` slots, acquire/release head/tail) consumed by
a dedicated pilot thread that calls `apus_store_hint` (thread-safe,
dedup'ed, eviction-guarded). The compute thread never blocks: a full ring
**drops the newest** entry (FIFO issue order == time-to-need order, so
the oldest queued hint is the most urgent); hints strictly behind the
compute thread's (pos, layer) watermark are dropped at the consumer as
stale.

**Recall accounting (decode, s==1).** The router_actual hook (fired by
the tiered MoE with the actual top-k) compares against the pending
prediction at the same (pos, layer) — recall = actual_hits /
actual_experts is measurable live from `apus_pilot_stats`. Store-side
accounting note (M6a): first consume of a hint-loaded entry counts as a
miss — prefetch effectiveness shows up as `demand_loads == 0` /
present-at-ask, not as hits.

**Prefill.** `prefill_last_only` (default 1) predicts only for the last
token of a multi-token forward (the state that flows into decode).

## Knobs (defaults and why)

| knob | default | meaning / rationale |
|---|---|---|
| `APUS_PILOT` | 1 in tiered mode | master switch (`--tiered` only; a store is required) |
| `APUS_PILOT_K` | 8 | predicted top-N cap. **Provisional** — the real value comes from a recall-curve run on the real model (M9/M11); the fixture harness below is what that run uses |
| `APUS_PILOT_RING` | 4096 | ring capacity (pow2) |
| `APUS_PILOT_DUMP` | off | NDJSON P/A dump path (decode path; used by test_recall) |
| `prefill_last_only` | 1 | prefill predicts the last token only |

## Results (this Mac, M1 Pro; fixture = random weights — see caveat)

**Invariance (THE hard test), `test_invariance`:** greedy decode,
8-token prompt, 24 steps: eager vs store-only vs pilot ON at 16 / 8 / 2
slots-per-layer vs 2 slots + RSS budget 1 byte (128 drops) vs 4 slots +
2 pins — **token streams identical, all 6,144 logits bitwise identical
in every configuration** (18/18 checks), with the pilot thread issuing
real loads through the I/O pool. Digest `09fc53aee61f56c1` — the SAME
digest as m6a (identical greedy run) — identical at
APUS_IO_THREADS=1/4/8 (Makefile diffs; the informational pilot
issued/stale lines are excluded — thread-timing dependent).

**Pilot correctness, `test_pilot` (26 checks):** ring FIFO order,
drop-newest with exact counts (16/4), wraparound; `apus_pilot_predict`
bitwise equals the direct `rmsnorm + apus_moe_route` computation (n ==
top_k) AND `rmsnorm + apus_moe_route_topn` (n > top_k) — the shared
code paths; out-of-range and unattached layers not predictable (-1);
`prefill_last_only` (t<s-1 skipped, t=s-1 predicts, 8 ring entries per
prediction); recall accounting 8/8 hits on a matching actual set, no
accounting without a pending prediction, s>1 actuals ignored; consumer
thread drain (issued == enqueued), stale watermark drop,
destroy-with-backlog.

**Recall end-to-end, `test_recall` + `check_recall.py`:** piloted greedy
decode with the NDJSON dump: live counters
`predictions=24 actual_experts=96 actual_hits=93` (recall 96.9%)
recomputed from the dumped P/A sets in Python **match exactly**.

> **CAVEAT — the recall VALUE is random-weight noise, NOT a tuning
> input.** The fixture model has random weights over only 16 experts
> and the pilot predicts 8 of them per token, so "recall" is
> structurally high regardless of any real locality. What is validated:
> the accounting machinery (live counters == Python recompute), the
> dump format, the prediction code path (bitwise the real router). Real
> recall numbers come from the identical run on the real container
> (M9/M11 — `--tiered` + `APUS_PILOT_DUMP` + check_recall.py).

**UBSan:** `make ubsan-m6b` clean (all three binaries + checker).

## Notes

- The pilot only ever reads hidden states and issues hints — the
  bitwise invariance gate holds by construction (same slab bytes, same
  kernels), but it is asserted at every cache size anyway.
- Pilot hooks live on the c/cache.h tiered forward (`ApusStoreFwdHooks`
  post_attn / router_actual + `apus_store_fwd_set_batch`); the eager
  path has no hooks (c/layer.h is frozen).
- Prediction cost per token is one extra rmsnorm + router matvec per
  layer boundary (16×128 on the fixture; 256×2048 on the real model —
  fold into the mt GEMM when M9 optimizes).
- Real-model tuning (recall curve → APUS_PILOT_K, pin budget →
  APUS_PIN_MB) requires the real container; the machinery for the run
  is `--tiered` + `APUS_PILOT_DUMP` + tests/m6b/check_recall.py.
