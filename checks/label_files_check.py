"""label_files_check.py - label CSVs against a reference set: same studies, same columns, no NaN,
values in [0, 1], and how far each finding moved.

    python checks/label_files_check.py <reference.csv> <candidate.csv> [...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent.parent / "data"
ref = pd.read_csv(D / sys.argv[1]).set_index("StudyInstanceUID")
tasks = [c for c in ref.columns if ref[c].dtype.kind == "f"][:12]
bad = 0
for f in sys.argv[2:]:
    c = pd.read_csv(D / f).set_index("StudyInstanceUID")
    same_set = set(c.index) == set(ref.index)
    same_order = list(c.index) == list(ref.index)
    cols = list(c.columns) == list(ref.columns)
    v = c.loc[ref.index, tasks].values if same_set else c[tasks].values
    nan, lo, hi = np.isnan(v).any(), v.min(), v.max()
    other = [k for k in ref.columns if k not in tasks]
    meta_same = same_set and ref[other].equals(c.loc[ref.index, other])
    d = np.abs(v - ref[tasks].values) if same_set else None
    ok = same_set and cols and not nan and lo >= 0 and hi <= 1 and meta_same
    bad += not ok
    print(f"{f}: {len(c)} rows | same studies {same_set} (same order {same_order}) | same columns {cols} "
          f"| non-label columns identical {meta_same} | NaN {nan} | range {lo:.4f}..{hi:.4f}  "
          f"{'OK' if ok else 'FAIL'}")
    if d is not None:
        print("   mean |diff| vs reference: " + ", ".join(f"{t} {d[:, j].mean():.3f}" for j, t in enumerate(tasks)))
sys.exit(1 if bad else 0)
