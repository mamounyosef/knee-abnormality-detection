"""26_make_ttt_pseudo.py - pseudo-labels for the test-time-training SIMULATION (7 Oct).

The idea under test: inside the Kaggle notebook, the ensemble labels the hidden test studies, one member fine-tunes
on those studies with those soft labels, then predicts them again. Here the 404 v5 val studies (data/23_val_v3_audit.csv:
346 v3-labelled + the 58 audit) stand in for the test set, and the teacher is fit-10's core restricted to members that
NEVER trained on them (all v5 split): convnext_base_v5, dinov3_convnext_base_v5, convnext_base_25d_18slices_v5_aug_base,
convnext_base_25d_poolattn_v5 on all 12, convnext_base_25d_v5_joint_zoom_030 on the 5 joint findings only (as fit-10).
Flat logit mean per finding (count-normalised), sigmoid -> soft label. The real labels are never read for the labels.

Writes data/26_ttt_val_pseudo.csv (same columns as the split CSVs) and prints the teacher's AUROC on the real val
labels and on audit-58 (what the student is distilling).

    python data_code/26_make_ttt_pseudo.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "runs" / "others"
src = (HERE / "zoom_ensemble_test.py").read_text(encoding="utf-8")
z = {"__file__": str(HERE / "zoom_ensemble_test.py"), "__name__": "zoom_scaffold"}
exec(compile(src[:src.index("data = {}")], "zoom_ensemble_test.py", "exec"), z)
g, T, JOINT = z["g"], z["T"], z["JOINT"]
TEACH = ["convnext_base_v5", "dinov3_convnext_base_v5", "convnext_base_25d_18slices_v5_aug_base",
         "convnext_base_25d_poolattn_v5"]
ZOOM = z["ZOOM"]["z030"]

val = pd.read_csv(ROOT / "data" / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
ids = list(val.index)
num = sum(g["get"](m, ids).loc[ids, T].astype(float).to_numpy() for m in TEACH)
den = np.full(len(T), float(len(TEACH)))
jm = np.array([t in JOINT for t in T], dtype=float)
num = num + g["get"](ZOOM, ids).loc[ids, T].astype(float).to_numpy() * jm[None]
den = den + jm
logit = num / den[None]
prob = 1.0 / (1.0 + np.exp(-logit))

out = val.copy()
out[T] = prob
dst = ROOT / "data" / "26_ttt_val_pseudo.csv"
out.reset_index().to_csv(dst, index=False)

y = (val[T].to_numpy() > 0.5).astype(int)
aud = val.source.eq("audit").to_numpy()


def macro(yy, pp):
    return np.mean([roc_auc_score(yy[:, j], pp[:, j]) for j in range(len(T)) if 0 < yy[:, j].sum() < len(yy)])


print(f"teacher on real val-404 labels {macro(y, logit):.4f} | audit-58 {macro(y[aud], logit[aud]):.4f}")
print(f"pseudo-label mean per finding: " + ", ".join(f"{t} {prob[:, j].mean():.2f}" for j, t in enumerate(T)))
assert ((out[T] >= 0) & (out[T] <= 1)).all().all() and len(out) == 404 and out.index.is_unique
print(f"wrote {dst} ({len(out)} studies)")
