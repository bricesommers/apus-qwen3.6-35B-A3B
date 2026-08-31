#!/usr/bin/env python3
"""M3 golden generator — numpy oracle for the BF16 kernels (c/bf16.h).

The normative numerics anchor lives in C (c/bf16.h scalar kernels); this
script provides the INDEPENDENT oracle the C tests are gated against:

  * BF16 narrow (fp32 -> bf16, RNE): computed here with float64 candidate
    distances and an explicit tie-to-even rule — deliberately NOT the C
    bit-trick (u += 0x7FFF + ((u>>16)&1)) — so a bug in one cannot hide in
    the other. Cases: every bf16 code as high bits x low-16 patterns
    {0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF} (covers exact values,
    subnormals, tie midpoints, just-below/above-tie, saturation-to-inf,
    inf and NaN payloads).
  * GEMM f64 truth: y_f64[m,o] = sum over k IN ASCENDING ORDER of
    f64(w)*f64(x) (w, x the exactly-widened bf16 values), plus the error
    scale esc[m,o] = sum_k |w*x|. fp32 accumulation error scales with esc,
    not |out| (cancellation can drive |out| to ~0), so C tolerances are
    fractions of esc.

Fixtures (tests/m3/golden/):
  manifest.txt     M= / O= / K= / NARROW=
  w.bin            uint16 [O, K]   bf16 weight codes
  x.bin            uint16 [M, K]   bf16 activation codes
  out.bin          float64 [M, O]  sequential-k f64 truth
  esc.bin          float64 [M, O]  f64 error scale
  narrow_in.bin    uint32 [NARROW] f32 bit patterns
  narrow_out.bin   uint16 [NARROW] expected bf16 codes (this oracle)

Weights/activations are built to exercise the corners: all-zero rows,
all-negative rows, bf16-subnormal codes, near-max magnitudes, alternating
signs. All finite (inf/NaN propagation is a C-side edge test, not an
oracle comparison).
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "golden")

LOW_PATTERNS = [0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF]
INF_TIE_THRESHOLD = np.float64(2.0 ** 128) - np.float64(2.0 ** 119)


def bf16_widen_f64(codes):
    """uint16 bf16 codes -> float64 values, exact (via float32)."""
    u = (codes.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return u.astype(np.float64)


def bf16_narrow_oracle(u32):
    """fp32 bit patterns -> bf16 codes, RNE, via float64 candidate
    distances (independent of the C bit-trick)."""
    u = u32.astype(np.uint32)
    mag = u & np.uint32(0x7FFFFFFF)
    sign = (u >> np.uint32(31)) << np.uint32(15)
    is_nan = mag > np.uint32(0x7F800000)

    with np.errstate(invalid="ignore"):   # NaN patterns are handled explicitly
        v = mag.view(np.float32).astype(np.float64)
        a = np.abs(v)
        lo = mag >> np.uint32(16)                      # truncation toward 0
        hi = lo + np.uint32(1)
        vlo = (lo << np.uint32(16)).view(np.float32).astype(np.float64)

        is_inf = mag == np.uint32(0x7F800000)          # a == inf -> lo == 0x7F80
        sat = lo == np.uint32(0x7F7F)                  # next code would be inf
        dlo = a - vlo
        # normal case: dhi = vhi - a (clamp hi so the sat case below reuses the
        # array; it is overridden there anyway)
        vhi = (np.minimum(hi, np.uint32(0x7F7F)) << np.uint32(16))
        vhi = vhi.view(np.float32).astype(np.float64)
        dhi = vhi - a
    # at the finite/inf boundary the "next grid value" is 2^128: round up
    # iff a >= midpoint(max_finite, 2^128) (tie -> 0x7F80, the even code)
    pick_hi_sat = a >= INF_TIE_THRESHOLD
    pick_hi = ((dhi < dlo) | ((dhi == dlo) & ((hi & np.uint32(1)) == 0)))
    pick_hi = np.where(sat, pick_hi_sat, pick_hi)
    pick_hi = np.where(is_inf, False, pick_hi)     # inf stays inf (lo)
    code = np.where(pick_hi, hi, lo) | sign
    return np.where(is_nan, (u >> np.uint32(16)), code).astype(np.uint16)


def gemm_truth_f64(w64, x64):
    """y[m,o] = sum_k w64[o,k]*x64[m,k] in ASCENDING k order, float64;
    esc[m,o] = sum_k |w64[o,k]*x64[m,k]|. Loop keeps the order explicit —
    no BLAS reassociation."""
    O, K = w64.shape
    M = x64.shape[0]
    out = np.zeros((M, O), dtype=np.float64)
    esc = np.zeros((M, O), dtype=np.float64)
    for m in range(M):
        for o in range(O):
            acc = 0.0
            scale = 0.0
            for k in range(K):
                p = w64[o, k] * x64[m, k]
                acc += p
                scale += abs(p)
            out[m, o] = acc
            esc[m, o] = scale
    return out, esc


def main():
    rng = np.random.default_rng(20260806)
    os.makedirs(OUT, exist_ok=True)

    # --- narrow oracle cases -------------------------------------------------
    highs = np.arange(65536, dtype=np.uint32)
    cases = (highs[:, None] << np.uint32(16)) | \
        np.array(LOW_PATTERNS, dtype=np.uint32)[None, :]
    cases = cases.reshape(-1)
    expected = bf16_narrow_oracle(cases)
    cases.tofile(os.path.join(OUT, "narrow_in.bin"))
    expected.tofile(os.path.join(OUT, "narrow_out.bin"))

    # --- GEMM golden ---------------------------------------------------------
    M, O, K = 3, 64, 256

    def rand_codes(shape):
        """Random FINITE bf16 codes: random float32 in a sane range, RNE
        narrowed by this oracle (so the codes are realistic, not uniform
        over the code space)."""
        f = (rng.standard_normal(shape) * 2.0).astype(np.float32)
        return bf16_narrow_oracle(f.view(np.uint32))

    w = rand_codes((O, K))
    w[0, :] = 0x0000                                   # all-zero row
    w[1, :] = 0x8000 | (w[1, :] & 0x7FFF)              # all-negative row
    w[2, :32] = rng.integers(1, 0x80, 32, dtype=np.uint16)   # subnormals
    w[3, :32] = np.uint16(0x7180)                      # 2^100 (big, f32-safe)
    w[4, :32] = np.where(np.arange(32) % 2 == 0,
                         np.uint16(0x3F80), np.uint16(0xBF80))  # +-1 alt

    x = rand_codes((M, K))
    x[1, :] = 0x0000                                   # all-zero act row
    x[2, 128:] = rng.integers(1, 0x80, K - 128, dtype=np.uint16)

    w64 = bf16_widen_f64(w)
    x64 = bf16_widen_f64(x)
    out, esc = gemm_truth_f64(w64, x64)

    w.tofile(os.path.join(OUT, "w.bin"))
    x.tofile(os.path.join(OUT, "x.bin"))
    out.tofile(os.path.join(OUT, "out.bin"))
    esc.tofile(os.path.join(OUT, "esc.bin"))

    with open(os.path.join(OUT, "manifest.txt"), "w") as f:
        f.write(f"M={M}\nO={O}\nK={K}\nNARROW={cases.size}\n")

    print(f"golden: M={M} O={O} K={K} narrow={cases.size}")
    print(f"  out range: [{out.min():.6g}, {out.max():.6g}]")
    print(f"  esc range: [{esc.min():.6g}, {esc.max():.6g}]")
    n_inf = int(np.count_nonzero((expected & 0x7FFF) == 0x7F80))
    n_nan = int(np.count_nonzero((expected & 0x7FFF) > 0x7F80))
    print(f"  narrow oracle: {n_inf} inf results, {n_nan} NaN results")


if __name__ == "__main__":
    main()
