#!/usr/bin/env python3
"""Generate c/uni_nfc.h: NFC normalization tables for the apus tokenizer.

The Qwen tokenizer.json has normalizer = NFC (the Rust unicode-normalization
crate via HF `tokenizers`). These tables let c/tok.h reproduce it byte-exactly.

Everything behavioral is DERIVED FROM THE REFERENCE IMPLEMENTATION, not from a
Unicode database, because the crate's Unicode version differs from Python's
unicodedata (probes showed e.g. U+08CF is a ccc=230 mark for Python 15.1 but
a plain starter for the crate):

  decomp: cp -> full canonical decomposition, probed from the crate's NFD
          (batched, "|" separator — a ccc=0 starter that normalization never
          introduces, so probes cannot interact).
  comp:   (a, b) -> c composition pairs: candidates are the NON-recursive
          canonical decompositions (unicodedata.decomposition, untagged,
          length 2 — chained compositions like U+01D5 <- (U+00DC, U+0304) are
          invisible in the recursive NFD), cross-checked with length-2 NFD
          results; a pair is admitted iff the crate's NFC maps a+b back to c
          (the composition-exclusion list falls out of this ground truth).
  marks:  the set of codepoints the crate treats as ccc != 0, detected
          behaviorally over EVERY codepoint: NFD(U+0300 + X + U+0316) changes
          the order of the outer marks iff X is a mark (a starter X isolates
          the segments); ccc VALUES come from Python's unicodedata (stable
          across versions for existing chars), with a ladder probe
          (binary search against reference marks) for chars Python does
          not know.
  Hangul: algorithmic (TR15 part 3), not table-driven.

Self-checks before writing anything: NFC re-implemented in Python using only
these tables must match the crate over all single codepoints, ALL ordered
mark pairs, a sample of blocking triples, and 20k random strings. The
independent C-side battery (tests/m2 nfc.bin via gen_golden.py) re-verifies
all of it against the compiled tables.
"""

import random
import sys
import unicodedata

from tokenizers.normalizers import NFC, NFD

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT_PATH = ROOT + "/c/uni_nfc.h"

CPS = [cp for cp in range(0x110000) if not (0xD800 <= cp <= 0xDFFF)]

# Hangul constants (TR15)
SBASE, LBASE, VBASE, TBASE = 0xAC00, 0x1100, 0x1161, 0x11A7
LCOUNT, VCOUNT, TCOUNT = 19, 21, 28
NCOUNT = VCOUNT * TCOUNT
SCOUNT = LCOUNT * NCOUNT

K230 = "̀"  # ccc 230, ancient
M220 = "̖"  # ccc 220, ancient


def batched_norm(norm, strings, sep="|"):
    """normalize_str over many strings at once; sep must be a ccc=0 starter
    that normalization never introduces, so probes cannot interact.
    Probes containing sep (e.g. the sep codepoint itself) go individually."""
    out = [None] * len(strings)

    def flush(batch, idxs):
        if not batch:
            return
        parts = norm.normalize_str(sep.join(batch)).split(sep)
        if len(parts) != len(batch):
            raise RuntimeError(f"probe desync: {len(parts)} != {len(batch)}")
        for j, p in zip(idxs, parts):
            out[j] = p

    batch, idxs = [], []
    for i, s in enumerate(strings):
        if sep in s:
            out[i] = norm.normalize_str(s)
            continue
        batch.append(s)
        idxs.append(i)
        if len(batch) == 65536:
            flush(batch, idxs)
            batch, idxs = [], []
    flush(batch, idxs)
    return out


def main():
    nfd = NFD()
    nfc = NFC()

    # ---- decomp: NFD probes over every codepoint ----
    print("probing NFD for all codepoints...")
    singles = [chr(cp) for cp in CPS]
    nfd_out = batched_norm(nfd, singles)
    decomp = {}  # cp -> list of cps (fully recursive, canonically ordered)
    for cp, s in zip(CPS, nfd_out):
        if s != chr(cp):
            decomp[cp] = [ord(c) for c in s]
    print(f"  {len(decomp)} codepoints with canonical decomposition")

    # ---- comp: length-2 canonical decompositions the crate maps back ----
    print("probing composition pairs...")
    cand = {}  # (a, b) -> c
    for cp in CPS:
        d = unicodedata.decomposition(chr(cp))
        if not d or d.startswith("<"):
            continue
        parts = [int(x, 16) for x in d.split()]
        if len(parts) == 2:
            cand[(parts[0], parts[1])] = cp
    for c, d in decomp.items():
        if len(d) == 2:
            cand.setdefault((d[0], d[1]), c)
    keys = [(a, b, c) for (a, b), c in cand.items()]
    nfc_out = batched_norm(nfc, [chr(a) + chr(b) for a, b, _ in keys])
    comp = {}  # (a, b) -> c
    for (a, b, c), s in zip(keys, nfc_out):
        if s == chr(c):
            comp[(a, b)] = c
    print(f"  {len(comp)} composition pairs (of {len(keys)} candidates)")

    # ---- marks: behavioral detection of ccc != 0 over every codepoint ----
    # (codepoints with a decomposition are replaced before ccc is ever
    # consulted, so their own class is irrelevant and they are skipped)
    print("detecting mark set (ccc != 0) behaviorally...")
    cand_marks = [cp for cp in CPS if cp not in decomp]
    probes = [K230 + chr(cp) + M220 for cp in cand_marks]
    got = batched_norm(nfd, probes)
    rust_marks = set()
    for cp, s, p in zip(cand_marks, got, probes):
        if s != p:
            rust_marks.add(cp)
    print(f"  {len(rust_marks)} marks")

    # ccc values: Python unicodedata where known; ladder probe otherwise
    py_ccc = {cp: unicodedata.combining(chr(cp)) for cp in rust_marks}
    unknown = sorted(cp for cp in rust_marks if not py_ccc[cp])
    if unknown:
        print(f"  {len(unknown)} marks unknown to unicodedata; ladder-probing ccc...")
        ref = {}
        for cp in sorted(py_ccc):
            v = py_ccc[cp]
            if v and v not in ref:
                ref[v] = cp
        ladder_vals = sorted(ref)
        for cp in unknown:
            lo, hi = 0, len(ladder_vals)  # find first v with ccc(cp) <= v
            while lo < hi:
                mid = (lo + hi) // 2
                k = ref[ladder_vals[mid]]
                # NFD(X + K): swaps to K + X iff ccc(X) > ccc(K)
                s = nfd.normalize_str(chr(cp) + chr(k))
                if s == chr(k) + chr(cp):
                    lo = mid + 1
                else:
                    hi = mid
            if lo >= len(ladder_vals):
                raise RuntimeError(f"ccc ladder overflow for U+{cp:04X}")
            py_ccc[cp] = ladder_vals[lo]
            print(f"    U+{cp:04X}: ccc={ladder_vals[lo]}")
    ccc = {cp: v for cp, v in py_ccc.items() if v}
    dropped = sorted(set(rust_marks) - set(ccc))
    if dropped:
        print(f"  WARNING: {len(dropped)} marks with no ccc value: "
              + " ".join(f"U+{cp:04X}" for cp in dropped[:20]))
    py_only = [cp for cp in CPS
               if unicodedata.combining(chr(cp)) and cp not in rust_marks and cp not in decomp]
    print(f"  {len(py_only)} unicodedata marks the crate treats as starters "
          f"(ccc forced to 0), e.g. {[f'U+{cp:04X}' for cp in py_only[:8]]}")

    # ---- NFC in Python using ONLY these tables (self-check engine) ----
    def t_ccc(cp):
        return ccc.get(cp, 0)

    def t_decompose(cp, out):
        if SBASE <= cp < SBASE + SCOUNT:
            s = cp - SBASE
            out.append(LBASE + s // NCOUNT)
            out.append(VBASE + (s % NCOUNT) // TCOUNT)
            if s % TCOUNT:
                out.append(TBASE + s % TCOUNT)
            return
        d = decomp.get(cp)
        if d is None:
            out.append(cp)
        else:
            out.extend(d)

    def t_comp(a, b):
        if LBASE <= a < LBASE + LCOUNT and VBASE <= b < VBASE + VCOUNT:
            return SBASE + ((a - LBASE) * VCOUNT + (b - VBASE)) * TCOUNT
        if SBASE <= a < SBASE + SCOUNT and (a - SBASE) % TCOUNT == 0 \
                and TBASE < b < TBASE + TCOUNT:
            return a + (b - TBASE)
        return comp.get((a, b))

    def t_nfd(s):
        seq = []
        for ch in s:
            t_decompose(ord(ch), seq)
        cccs = [t_ccc(c) for c in seq]
        i = 1
        while i < len(seq):
            if cccs[i] and cccs[i - 1] > cccs[i]:
                seq[i - 1], seq[i] = seq[i], seq[i - 1]
                cccs[i - 1], cccs[i] = cccs[i], cccs[i - 1]
                if i > 1:
                    i -= 1
                    continue
            i += 1
        return "".join(chr(c) for c in seq)

    def t_nfc(s):
        seq = []
        for ch in s:
            t_decompose(ord(ch), seq)
        cccs = [t_ccc(c) for c in seq]
        i = 1
        while i < len(seq):
            if cccs[i] and cccs[i - 1] > cccs[i]:
                seq[i - 1], seq[i] = seq[i], seq[i - 1]
                cccs[i - 1], cccs[i] = cccs[i], cccs[i - 1]
                if i > 1:
                    i -= 1
                    continue
            i += 1
        out = []
        starter_idx = -1
        prev_ccc = 0
        for c, cc in zip(seq, cccs):
            if starter_idx >= 0 and (prev_ccc == 0 or prev_ccc < cc):
                m = t_comp(out[starter_idx], c)
                if m is not None:
                    out[starter_idx] = m
                    continue
            if cc == 0:
                starter_idx = len(out)
            prev_ccc = cc
            out.append(c)
        return "".join(chr(c) for c in out)

    print("self-check: all single codepoints (NFC + NFD)...")
    got_nfc = batched_norm(nfc, singles)
    bad = 0
    for cp, want_nfd, want_nfc in zip(CPS, nfd_out, got_nfc):
        c = chr(cp)
        if t_nfd(c) != want_nfd or t_nfc(c) != want_nfc:
            if bad < 10:
                print(f"  MISMATCH U+{cp:04X}: nfd {t_nfd(c)!r}/{want_nfd!r} "
                      f"nfc {t_nfc(c)!r}/{want_nfc!r}")
            bad += 1
    if bad:
        raise RuntimeError(f"{bad} single-codepoint mismatches")

    print("self-check: ALL ordered mark pairs (NFD reorder + NFC)...")
    marks = sorted(ccc)
    pair_strs = [chr(a) + chr(b) for a in marks for b in marks]
    got_nfd = batched_norm(nfd, pair_strs)
    got_nfc = batched_norm(nfc, pair_strs)
    bad = 0
    for s, wn, wc in zip(pair_strs, got_nfd, got_nfc):
        if t_nfd(s) != wn or t_nfc(s) != wc:
            if bad < 10:
                print(f"  MISMATCH {s!r}: nfd {t_nfd(s)!r}/{wn!r} nfc {t_nfc(s)!r}/{wc!r}")
            bad += 1
    if bad:
        raise RuntimeError(f"{bad} mark-pair mismatches")

    print("self-check: blocking triples sample + 20k random strings...")
    rng = random.Random(1234)
    triples = []
    for (a, b), c in comp.items():
        cb = ccc.get(b, 0)
        cands = [m for m in marks if 0 < ccc[m] < cb] if cb else []
        for m in rng.sample(cands, min(4, len(cands))):
            triples.append(chr(a) + chr(m) + chr(b))
        triples.append(chr(a) + chr(b))
    interesting = [chr(cp) for cp in rng.sample(CPS, 4000)] + \
        [chr(m) for m in marks] + [chr(c) for c in decomp] + \
        [chr(cp) for cp in range(0x1100, 0x1113)] + ["각", "a", " ", "|", "\n"]
    triples += ["".join(rng.choice(interesting) for _ in range(rng.randrange(1, 9)))
                for _ in range(20000)]
    got = batched_norm(nfc, triples)
    bad = 0
    for s, w in zip(triples, got):
        if t_nfc(s) != w:
            if bad < 10:
                print(f"  MISMATCH {s!r}: {t_nfc(s)!r} != {w!r}")
            bad += 1
    if bad:
        raise RuntimeError(f"{bad} triple/random mismatches")
    print("self-check: all clean")

    # ---- emit header ----
    ccc_ranges = []
    start = prev = None
    cur = 0
    for cp in sorted(ccc):
        v = ccc[cp]
        if start is not None and cp == prev + 1 and v == cur:
            prev = cp
            continue
        if start is not None:
            ccc_ranges.append((start, prev, cur))
        start = prev = cp
        cur = v
    if start is not None:
        ccc_ranges.append((start, prev, cur))

    # Hangul syllables are algorithmic in tok.h (TR15 part 3): drop them from
    # the decomp table (11,172 entries) and the L+V comp pairs (399) — the
    # self-check above exercised the same Hangul-first code path.
    decomp_cps = [cp for cp in sorted(decomp) if not (SBASE <= cp < SBASE + SCOUNT)]
    comp_emit = {(a, b): c for (a, b), c in sorted(comp.items())
                 if not (LBASE <= a < LBASE + LCOUNT)}
    flat = []
    for cp in decomp_cps:
        flat.extend(decomp[cp])

    with open(OUT_PATH, "w") as f:
        f.write("/* Generated by tests/m2/gen_nfc_tables.py — do not edit.\n")
        f.write("   NFC tables matching the reference tokenizer's normalizer\n")
        f.write("   (Rust unicode-normalization crate via HF tokenizers).\n")
        f.write("   Decompositions are probed from the crate's NFD; composition\n")
        f.write("   pairs are admitted iff the crate's NFC maps the pair back\n")
        f.write("   (exclusion list by ground truth); the mark set (ccc != 0) is\n")
        f.write("   detected behaviorally per codepoint and ccc values come from\n")
        f.write("   unicodedata/ladder probes. Hangul is algorithmic in tok.h.\n")
        f.write("   Verified: all singles, all ordered mark pairs, blocking\n")
        f.write("   triples sample, 20k random strings — plus the independent\n")
        f.write("   C-side battery (tests/m2 nfc.bin). */\n")
        f.write("#ifndef APUS_UNI_NFC_H\n#define APUS_UNI_NFC_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write("/* cp -> [off,off+len) into uni_nfc_decomp_data, sorted by cp */\n")
        f.write("static const struct { uint32_t cp, off, len; } uni_nfc_decomp[] = {\n")
        off = 0
        for cp in decomp_cps:
            f.write(f"    {{0x{cp:06X}u, {off}u, {len(decomp[cp])}u}},\n")
            off += len(decomp[cp])
        f.write("};\n#define UNI_NFC_DECOMP_N " + str(len(decomp_cps)) + "u\n\n")
        f.write("static const uint32_t uni_nfc_decomp_data[] = {\n")
        for i in range(0, len(flat), 8):
            f.write("    " + "".join(f"0x{x:06X}u, " for x in flat[i:i + 8]).rstrip() + "\n")
        f.write("};\n\n")

        f.write("/* canonical combining class != 0, sorted ranges */\n")
        f.write("static const struct { uint32_t lo, hi, ccc; } uni_nfc_ccc[] = {\n")
        for lo, hi, v in ccc_ranges:
            f.write(f"    {{0x{lo:06X}u, 0x{hi:06X}u, {v}u}},\n")
        f.write("};\n#define UNI_NFC_CCC_N " + str(len(ccc_ranges)) + "u\n\n")

        f.write("/* canonical composition (a,b) -> c, sorted by (a,b) */\n")
        f.write("static const struct { uint32_t a, b, c; } uni_nfc_comp[] = {\n")
        for (a, b), c in sorted(comp_emit.items()):
            f.write(f"    {{0x{a:06X}u, 0x{b:06X}u, 0x{c:06X}u}},\n")
        f.write("};\n#define UNI_NFC_COMP_N " + str(len(comp_emit)) + "u\n\n")
        f.write("#endif\n")

    print(f"decomp: {len(decomp_cps)} entries, {len(flat)} data words")
    print(f"ccc: {len(ccc_ranges)} ranges, {len(ccc)} marks")
    print(f"comp: {len(comp_emit)} pairs (+{len(comp) - len(comp_emit)} algorithmic Hangul)")
    print("wrote", OUT_PATH)

    # sidecar for tests/m2/gen_golden.py: the probe sets (marks + ccc values,
    # composition pairs incl. algorithmic Hangul) for the nfc.bin battery
    import json as _json
    meta = {
        "marks": {str(cp): ccc[cp] for cp in sorted(ccc)},
        "comp": [[a, b, c] for (a, b), c in sorted(comp.items())],
    }
    meta_path = ROOT + "/tests/m2/nfc_meta.json"
    with open(meta_path, "w") as f:
        _json.dump(meta, f)
    print("wrote", meta_path)


if __name__ == "__main__":
    main()
