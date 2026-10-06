"""zoom_trajectory.py - the joint-zoom runs' validations step by step against their parent at the same steps:
overall val / audit, and the 5 joint-line findings (ACL, both menisci, medial/lateral OA) as their own macro.

    python runs/others/crop_rois/zoom_trajectory.py [run ...]
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUNS = sys.argv[1:] or ["convnext_base_25d_v5_joint_zoom", "convnext_base_25d_v5_joint_zoom_dual", "convnext_base_25d_v5_joint_zoom_030"]
PARENT = "convnext_base_25d_18slices_v5_aug_base"
JOINT = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"]
OTHER = ["MCL", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def load(run):
    f = ROOT / "runs" / run / "results" / "val_log.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    for pre in ("val", "audit"):
        d[f"{pre}_joint"] = d[[f"{pre}_auroc_{t}" for t in JOINT]].mean(1)
        d[f"{pre}_other"] = d[[f"{pre}_auroc_{t}" for t in OTHER]].mean(1)
    return d.drop_duplicates("step", keep="last").set_index("step")


cols = ["val_macro_auroc", "audit_macro_auroc", "val_joint", "audit_joint", "val_other", "audit_other"]
par = load(PARENT)
best = par.loc[5900]
print(f"PARENT best (step 5900): " + " ".join(f"{c} {best[c]:.4f}" for c in cols))
for run in RUNS:
    d = load(run)
    if d is None:
        print(f"\n{run}: no val_log yet")
        continue
    print(f"\n{run}")
    print(f"  {'step':>5s} " + " ".join(f"{c[:-6] if c.endswith('_auroc') else c:>12s}" for c in cols) + "   (parent at the same step: val / val_joint)")
    for st, r in d.iterrows():
        p = par.loc[st] if st in par.index else None
        ps = f"{p['val_macro_auroc']:.4f} / {p['val_joint']:.4f}" if p is not None else "-"
        print(f"  {st:5d} " + " ".join(f"{r[c]:12.4f}" for c in cols) + f"   {ps}")
    last = d.iloc[-1]
    print("  per joint finding, last step vs parent best (val | audit):")
    for t in JOINT:
        print(f"    {t:17s} {last[f'val_auroc_{t}']:.4f} ({last[f'val_auroc_{t}'] - best[f'val_auroc_{t}']:+.4f}) | "
              f"{last[f'audit_auroc_{t}']:.4f} ({last[f'audit_auroc_{t}'] - best[f'audit_auroc_{t}']:+.4f})")
