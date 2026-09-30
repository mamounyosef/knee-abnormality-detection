"""21_make_v3_folds.py - the 5 cross-validation folds of 13_fold*, relabelled with the v3 labels.

Same study membership as data/13_fold{k}_{train,val}.csv (so the out-of-fold predictions cover
the same 4,342 train + val studies exactly once, audit-58 never inside), labels taken from
17_train_full_qwen-mg-raw.csv (v3: Qwen + MedGemma 1.5 on the MedGemma five). All other
columns are kept from the 13_ files.

Writes data/21_fold{k}_train.csv and data/21_fold{k}_val.csv, then checks:
  - membership identical to 13_fold, row order identical
  - fold val sets disjoint and together exactly the 4,342 non-audit studies
  - train and val of each fold disjoint; no audit study anywhere
  - labels equal the v3 file on every row, no NaN, all in [0, 1]

    python data_code/21_make_v3_folds.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent.parent / "data"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
v3 = pd.read_csv(D / "17_train_full_qwen-mg-raw.csv").set_index("StudyInstanceUID")
audit = set(pd.read_csv(D / "05_audit58.csv").StudyInstanceUID)
vals = []
for k in range(5):
    for part in ("train", "val"):
        old = pd.read_csv(D / f"13_fold{k}_{part}.csv")
        new = old.copy()
        ids = new.StudyInstanceUID
        missing = set(ids) - set(v3.index)
        assert not missing, f"fold {k} {part}: {len(missing)} studies missing from the v3 file"
        new[T] = v3.loc[ids, T].values
        assert list(new.StudyInstanceUID) == list(old.StudyInstanceUID)
        assert list(new.columns) == list(old.columns)
        v = new[T].values
        assert not np.isnan(v).any() and v.min() >= 0 and v.max() <= 1
        assert np.abs(v - v3.loc[ids, T].values).max() == 0
        assert not (set(ids) & audit), f"fold {k} {part} contains audit studies"
        new.to_csv(D / f"21_fold{k}_{part}.csv", index=False)
        if part == "val":
            vals.append(set(ids))
        d = np.abs(v - old[T].values)
        print(f"fold {k} {part}: {len(new)} rows, labels changed on {int((d.max(1) > 1e-9).sum())} rows "
              f"(mean |diff| {d.mean():.4f})")
    tr = set(pd.read_csv(D / f"21_fold{k}_train.csv").StudyInstanceUID)
    assert not (tr & vals[-1]), f"fold {k}: train and val overlap"
union = set().union(*vals)
assert sum(len(v) for v in vals) == len(union), "fold val sets overlap"
assert union == set(v3.index) - audit, "fold vals are not exactly the non-audit studies"
print(f"ALL OK: 5 folds, val sets disjoint, together {len(union)} studies = every non-audit study")
