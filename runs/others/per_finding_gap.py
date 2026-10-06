"""per_finding_gap.py - where does the shipped ensemble lose? Per-finding AUROC of the scaffold fourteen (logit mean)
next to our report labels' own AUROC on the same panel studies (the label ceiling), on audit-58 and clean-121.

    python runs/others/per_finding_gap.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
src = (HERE / "old_label_members_ensemble.py").read_text(encoding="utf-8")
g = {"__file__": str(HERE / "old_label_members_ensemble.py"), "__name__": "olm_scaffold"}
exec(compile(src[:src.index("pool = sorted(")], "old_label_members_ensemble.py", "exec"), g)
T, BASE, SURF, get = g["T"], g["BASE"], g["SURF"], g["get"]
au = pd.read_csv(HERE.parents[1] / "data/05_audit58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
for s, (ids, Y) in SURF.items():
    P = np.stack([get(m, ids).loc[ids, T].astype(float).to_numpy() for m in BASE]).mean(0)
    y = Y.loc[ids, T].to_numpy().astype(int)
    ens = [roc_auc_score(y[:, j], P[:, j]) if 0 < y[:, j].sum() < len(y) else np.nan for j in range(len(T))]
    print(f"\n{s}: ensemble macro {np.nanmean(ens):.4f}  (positives per finding: {y.sum(0).tolist()})")
    lab = None
    if s == "audit58":
        lab = [roc_auc_score(y[:, j], au.loc[ids, T[j]]) for j in range(len(T))]
    for j, t in enumerate(T):
        print(f"  {t:18s} ensemble {ens[j]:.3f}" + (f"   report labels {lab[j]:.3f}" if lab else ""))
