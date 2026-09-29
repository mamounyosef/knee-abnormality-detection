"""local_vs_modal_check.py - do LOCAL (RTX 4060) runs train worse than Modal ones?

Scores every saved validation prediction of each run against ONE fixed label set (default the
v3 val labels 17_val_qwen-mg-raw, and the v2 05_val_qwen24t-med) so runs trained on different
labels become comparable, and prints the curve per step. The Modal MedGemma twin and the local
OOF / ensemble-label twins share model, schedule and split; only labels and machine differ.

    python runs/others/local_vs_modal_check.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RUNS = {"convnext_base_25d_18slices (Modal, v2)": "convnext_base_25d_18slices",
        "mglabels_raw (Modal, v3)": "convnext_base_25d_18slices_mglabels_raw",
        "oof (LOCAL, 18_*)": "convnext_base_25d_18slices_oof",
        "ens (LOCAL, 20_*)": "convnext_base_25d_18slices_ens"}
LABELS = sys.argv[1:] or ["17_val_qwen-mg-raw.csv", "05_val_qwen24t-med.csv"]
labs = {f: pd.read_csv(ROOT / "data" / f).set_index("StudyInstanceUID") for f in LABELS}
for name, run in RUNS.items():
    pdir = ROOT / "runs" / run / "results/predictions"
    files = sorted(pdir.glob("run*_val_s*.npz"), key=lambda p: int(re.search(r"_s(\d+)", p.name).group(1)))
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        ids = [str(s) for s in z["study_ids"]]
        tasks = [str(t) for t in z["tasks"]] if "tasks" in z else None
        lg = z["logits"]
        rec = {"step": int(re.search(r"_s(\d+)", f.name).group(1))}
        for lf, L in labs.items():
            t = tasks or [c for c in L.columns if L[c].dtype.kind == "f"][:12]
            y = (L.loc[ids, t].values >= 0.5).astype(int)
            rec[lf] = np.mean([roc_auc_score(y[:, j], lg[:, j]) for j in range(len(t)) if 0 < y[:, j].sum() < len(y)])
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"\n{name}: {len(df)} validations")
    for lf in LABELS:
        b = df.loc[df[lf].idxmax()]
        print(f"  vs {lf}: peak {b[lf]:.4f} at step {int(b.step)}; last {df[lf].iloc[-1]:.4f} (step {int(df.step.iloc[-1])})")
