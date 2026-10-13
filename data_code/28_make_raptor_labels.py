"""
28_make_raptor_labels.py - the DEFAULT labels from 13 Oct (user): the w50 labels moved 10% toward the raptor folds.

    new logit = 0.9 * logit(w50 label) + 0.1 * raptor OOF logit

w50 label = data/22_train_full_qwen-mg-convnext-mii-w50.csv (the "v4" labels every current member trains on).
Raptor OOF = for each study, the ONE fold of convnext_base_v5_raptor96_f0..f4 that did not train on it (round-2 split,
27_fold{k}), at that fold's best-val checkpoint (v3-graded fold val). The 58 audit studies are in no fold and keep the
panel's 0/1 labels, exactly as before.

WHY (runs/others/raptor_teacher_audit.py, 13 Oct): audit-58 of the labels, current w50 0.9453 -> 0.9467 (+0.0014,
paired bootstrap better in 91%, 90% interval [-0.0003, +0.0033]). Raptor is the best single teacher (0.9257 alone).

Every training csv in use is a row subset of the 22_* file with identical labels, so each gets a twin with the new
labels (same rows, same order, same other columns):
    22_train_full_qwen-mg-convnext-mii-w50.csv -> 28_train_full_w50r10.csv
    23_train_w50.csv                           -> 28_train_w50r10.csv            (v5 split, 8% val)
    24_train_w50_minus_clean121.csv            -> 28_train_w50r10_minus_clean121.csv
    27_fold{k}_train.csv                       -> 28_fold{k}_train.csv           (round-2 folds)
Validation files are NOT relabelled (23_val_v3_audit / 21_fold*_val / 24_val_clean121 stay v3 + panel labels).

    python data_code/28_make_raptor_labels.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
sys.path.insert(0, str(ROOT / "runs/others/ensembling"))
import sweep_new_runs as sw  # noqa: E402

TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
A = 0.1
FAM = "convnext_base_v5_raptor96_f{}"
PAIRS = {"22_train_full_qwen-mg-convnext-mii-w50.csv": "28_train_full_w50r10.csv",
         "23_train_w50.csv": "28_train_w50r10.csv",
         "24_train_w50_minus_clean121.csv": "28_train_w50r10_minus_clean121.csv",
         **{f"27_fold{k}_train.csv": f"28_fold{k}_train.csv" for k in range(5)}}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# raptor OOF over all 4,342 fold studies
parts = []
for k in range(5):
    va = set(pd.read_csv(D / f"21_fold{k}_val.csv").StudyInstanceUID)
    tr = set(pd.read_csv(D / f"27_fold{k}_train.csv").StudyInstanceUID)
    f = sw.best_npz(FAM.format(k), "val")
    z = np.load(f, allow_pickle=True)
    p = pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=TASKS)
    assert set(p.index) == va and not (va & tr), f"fold {k}: val npz != fold val, or val in train"
    parts.append(p)
    print(f"  fold {k}: {Path(f).name}  {len(p)} studies")
oof = pd.concat(parts)
assert oof.index.is_unique and len(oof) == 4342

full = pd.read_csv(D / "22_train_full_qwen-mg-convnext-mii-w50.csv").set_index("StudyInstanceUID")
new = full.copy()
rows = [s for s in full.index if full.loc[s, "source"] != "audit"]
assert set(rows) == set(oof.index), "non-audit rows != the 4,342 fold studies"
new.loc[rows, TASKS] = sigmoid((1 - A) * logit(full.loc[rows, TASKS].values) + A * oof.loc[rows, TASKS].values)
aud = full.index[full.source == "audit"]
assert (new.loc[aud, TASKS].values == full.loc[aud, TASKS].values).all()
d = (new.loc[rows, TASKS] - full.loc[rows, TASKS]).abs()
print(f"\nrelabelled {len(rows)} studies (audit {len(aud)} unchanged): mean |change| {d.values.mean():.4f}, "
      f"max {d.values.max():.4f}")
print("mean label per finding, w50 -> new:")
for t in TASKS:
    print(f"  {t:17s} {full.loc[rows, t].mean():.4f} -> {new.loc[rows, t].mean():.4f}")

for src, dst in PAIRS.items():
    s = pd.read_csv(D / src)
    o = s.copy()
    ids = s.StudyInstanceUID.values
    assert np.allclose(s[TASKS].values, full.loc[ids, TASKS].values), f"{src} labels are not the w50 labels"
    o[TASKS] = new.loc[ids, TASKS].values
    o.to_csv(D / dst, index=False)
    print(f"  {src:45s} -> {dst:38s} {len(o)} rows")
