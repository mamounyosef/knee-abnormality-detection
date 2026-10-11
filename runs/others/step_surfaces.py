"""step_surfaces.py - one run's val (v5, 404), audit-58 and clean-121 at EVERY saved validation step (11 Oct).

clean-121 = the 58 audit studies + the 63 v5-val studies no member trained on (24_val_clean121.csv), from the val npz.

    python runs/others/step_surfaces.py <run>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
run = sys.argv[1]
R = ROOT / "runs" / run / "results"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
clean = pd.read_csv(ROOT / "data/24_val_clean121.csv").set_index("StudyInstanceUID")
v = pd.read_csv(R / "val_log.csv").drop_duplicates("step", keep="last")
v = v[v.step > 0].set_index("step")


def mac(Y, P):
    return np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)])


rows = []
for f in sorted((R / "predictions").glob("*_val_s*.npz")):
    step = int(f.stem.split("_s")[-1])
    if step not in v.index:
        continue
    z = np.load(f, allow_pickle=True)
    P = pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T)
    ids = [s for s in clean.index if s in P.index]
    if len(ids) < 121:
        continue
    Y = (clean.loc[ids, T].values >= 0.5).astype(int)
    rows.append((step, v.loc[step, "val_macro_auroc"], v.loc[step, "audit_macro_auroc"], mac(Y, P.loc[ids, T].values)))
d = pd.DataFrame(rows, columns=["step", "val", "audit58", "clean121"]).sort_values("step")
d["sum3"] = d.val + d.audit58 + d.clean121
print(d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
for c in ("val", "audit58", "clean121", "sum3"):
    b = d.loc[d[c].idxmax()]
    print(f"best by {c:8s}: step {int(b.step):5d}  val {b.val:.4f}  audit {b.audit58:.4f}  clean {b.clean121:.4f}")
