#!/usr/bin/env python3
"""tests/m6b/check_recall.py — recompute the pilot recall from the NDJSON
P/A dump and compare against the live C counters (must match EXACTLY).

The dump has one JSON object per line:
  {"type":"P","pos":..,"layer":..,"eids":[..]}   predicted set (top-8)
  {"type":"A","pos":..,"layer":..,"eids":[..]}   actual chosen set (8)

Recall accounting (mirrors c/pilot.h router_actual): for each A record
at (pos, layer) with a pending P at the same (pos, layer), count
actual += |A|, hits += |A ∩ P|. The pending P for a (pos, layer) is the
most recent P record with that (pos, layer) (dL=1: written during the
previous layer of the same token).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "bin", "pilot_dump.ndjson")
COUNTERS = os.path.join(HERE, "bin", "pilot_counters.txt")


def main():
    pending = {}
    actual = 0
    hits = 0
    predictions = 0
    pred_experts = 0
    with open(DUMP) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["pos"], rec["layer"])
            if rec["type"] == "P":
                pending[key] = set(rec["eids"])
                predictions += 1
                pred_experts += len(rec["eids"])
            elif rec["type"] == "A":
                if key in pending:
                    a = set(rec["eids"])
                    actual += len(a)
                    hits += len(a & pending[key])
    counters = {}
    with open(COUNTERS) as f:
        for line in f:
            k, v = line.strip().split("=")
            counters[k] = int(v)
    ok = True
    for k, mine in (("predictions", predictions),
                    ("pred_experts", pred_experts),
                    ("actual_experts", actual),
                    ("actual_hits", hits)):
        if counters.get(k) != mine:
            print(f"MISMATCH {k}: live {counters.get(k)} vs recomputed "
                  f"{mine}")
            ok = False
    if not ok:
        sys.exit(1)
    print(f"check_recall: live counters == Python recompute exactly "
          f"(hits {hits}/{actual} = "
          f"{100.0 * hits / actual:.1f}%, {predictions} predictions)")


if __name__ == "__main__":
    main()
