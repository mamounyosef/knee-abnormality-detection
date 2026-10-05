"""compare_runs.py - do two public-side runs give the same raw reader outputs? (e.g. sequential vs parallel CoAt readers)
Compares every float array in the .npz files both output folders hold (same name), plus diagnostics/*.npz.

    python runs/others/ensembling/public_side/compare_runs.py <dirA> <dirB>
"""
import sys
from pathlib import Path

import numpy as np

a, b = Path(sys.argv[1]), Path(sys.argv[2])
for fa in sorted(list(a.glob("*.npz")) + list(a.glob("diagnostics/*.npz"))):
    fb = b / fa.relative_to(a)
    if not fb.exists():
        continue
    za, zb = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
    for k in za.files:
        if k in zb.files and za[k].dtype.kind == "f" and za[k].shape == zb[k].shape:
            d = float(np.abs(za[k] - zb[k]).max())
            print(f"{str(fa.relative_to(a)):55s} {k:20s} {str(za[k].shape):14s} max abs diff {d:.2e}")
