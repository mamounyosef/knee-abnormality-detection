"""score_audit58_manual.py - score a hand-made audit-58 label file against the expert panel.

Validates the file (58 audit studies, 12 findings, values in [0, 1]), then prints macro and
per-finding AUROC next to the Qwen labeller on the same studies.

    python data_code/score_audit58_manual.py [C:\\temp1\\audit58_labeling\\labels.csv]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
         "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
f = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\temp1\audit58_labeling\labels.csv")
aud = pd.read_csv(ROOT / "data/05_audit58.csv").set_index("StudyInstanceUID")
lab = pd.read_csv(f).set_index("StudyInstanceUID")
missing = [t for t in TASKS if t not in lab.columns]
assert not missing, f"missing columns {missing}"
assert set(lab.index) == set(aud.index), (f"studies differ: {len(set(aud.index) - set(lab.index))} "
                                          f"missing, {len(set(lab.index) - set(aud.index))} extra")
v = lab.loc[aud.index, TASKS].astype(float).values
assert not np.isnan(v).any() and v.min() >= 0 and v.max() <= 1, "values must be in [0, 1], no blanks"
y = aud[["official_" + t for t in TASKS]].values
qwen = aud[TASKS].values
rows = []
for j, t in enumerate(TASKS):
    rows.append((t, roc_auc_score(y[:, j], v[:, j]), roc_auc_score(y[:, j], qwen[:, j]), int(y[:, j].sum())))
df = pd.DataFrame(rows, columns=["finding", "manual", "qwen", "positives"]).set_index("finding")
print(df.round(4).to_string())
print(f"\nmacro AUROC  manual {df.manual.mean():.4f}  |  Qwen {df.qwen.mean():.4f}  "
      f"({df.manual.mean() - df.qwen.mean():+.4f})  |  current v3 labels 0.9311 (reference)")
print(f"distinct values used: {len(np.unique(v))}  (few distinct values = coarse ranking)")
