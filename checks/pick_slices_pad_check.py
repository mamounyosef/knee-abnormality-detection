"""pick_slices_pad_check.py - exhaustive logic test of shared_code.pick_slices with short_stack "pad" (CPU, seconds).

For every stack length 1..80 and n_want in (18, 24, 30), both modes (even / random), with and without reverse:
  - never a repeated real slice; every index is a real slice (0..n-1) or -1 (padding); exactly n_want picks
  - n >= n_want: no padding at all, picks sorted (or reversed), spread over the whole stack
  - n <  n_want: EVERY real slice exactly once, padding = n_want - n
  - n large enough for the margin (lo..hi holds n_want), evaluation ("even") mode: identical to the old behaviour;
    training ("random") mode uses disjoint bins under "pad" (the old bins can share a boundary slice = a repeat)
  - "repeat" (default) is byte-identical to before for every case (the knob must not alter old runs)

    python checks/pick_slices_pad_check.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runs"))
import shared_code as sc  # noqa: E402


def old_pick(n_available, n_want, mode, rng, margin=0.0, offset=0.0, reverse=False):
    """The pre-change pick_slices, verbatim, for the no-regression check."""
    lo = int(n_available * margin)
    hi = n_available - lo
    if hi - lo >= n_want:
        edges = np.linspace(lo, hi, n_want + 1)
        if mode == "random":
            a_, b_ = np.floor(edges[:-1]).astype(int), np.ceil(edges[1:]).astype(int)
            idx = np.array([rng.integers(a, max(a + 1, b)) for a, b in zip(a_, b_)])
        else:
            centres = (edges[:-1] + edges[1:]) / 2.0
            if offset:
                centres = centres + float(offset) * (edges[1] - edges[0])
            idx = np.clip(centres.astype(int), 0, n_available - 1)
    else:
        idx = np.round(np.linspace(0, n_available - 1, n_want)).astype(int)
    return idx[::-1].copy() if reverse else idx


fails, cases = [], 0
for want in (18, 24, 30):
    for n in range(1, 81):
        for mode in ("even", "random"):
            for rev in (False, True):
                for margin in (0.0, 0.05):
                    cases += 1
                    r1, r2 = np.random.default_rng(n), np.random.default_rng(n)
                    new_rep = sc.pick_slices(n, want, mode, r1, margin, 0.0, rev)            # default = "repeat"
                    old = old_pick(n, want, mode, r2, margin, 0.0, rev)
                    if not np.array_equal(new_rep, old):
                        fails.append(f"repeat changed: n={n} want={want} {mode} rev={rev} m={margin}")
                    p = sc.pick_slices(n, want, mode, np.random.default_rng(n), margin, 0.0, rev, "pad")
                    real = p[p >= 0]
                    tag = f"n={n} want={want} {mode} rev={rev} m={margin}"
                    if len(p) != want:
                        fails.append(f"{tag}: {len(p)} picks")
                    if ((p < -1) | (p >= n)).any():
                        fails.append(f"{tag}: index out of range {p}")
                    if len(set(real.tolist())) != len(real):
                        fails.append(f"{tag}: repeated real slice {p}")
                    if n >= want and (p < 0).any():
                        fails.append(f"{tag}: padding although the stack is long enough")
                    if n < want and (sorted(real.tolist()) != list(range(n)) or (p < 0).sum() != want - n):
                        fails.append(f"{tag}: short stack must give every slice once + {want - n} pads, got {p}")
                    ordered = real[::-1] if rev else real
                    if not np.all(np.diff(ordered) > 0):
                        fails.append(f"{tag}: picks not in slice order {p}")
                    lo = int(n * margin)
                    if n - 2 * lo >= want and mode == "even":      # random+pad uses disjoint bins on purpose (no repeats)
                        r3 = np.random.default_rng(n)
                        if not np.array_equal(p, old_pick(n, want, mode, r3, margin, 0.0, rev)):
                            fails.append(f"{tag}: long-stack path differs from the old behaviour")
                    if want <= n < want + 2 * lo + 2 and mode == "even":
                        span = real.max() - real.min()
                        if span < 0.85 * (n - 1):
                            fails.append(f"{tag}: picks do not span the stack ({real.min()}..{real.max()})")
print(f"{cases} cases checked")
print("\n".join(fails[:20]) if fails else "ALL OK")
