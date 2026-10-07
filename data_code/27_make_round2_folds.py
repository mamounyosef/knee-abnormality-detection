"""27_make_round2_folds.py - fold CSVs for the ROUND-2 teachers (7 Oct).

Round 1 (data_code/22_*): teachers trained on v3 labels in the 5 folds of data/21_fold{k}_*, their out-of-fold
predictions blended 50/50 into v3 = the v4 labels (22_*-w50) every current member trains on.
Round 2: the SAME 5 folds (same studies, so every study is still scored by models that never trained on it; the 58
audit studies are in no fold, so the teacher's audit-58 score stays clean), but the labels are v4 for both train and
val. Writes data/27_fold{k}_train.csv and data/27_fold{k}_val.csv with the 21_* columns.

    python data_code/27_make_round2_folds.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
v4 = pd.read_csv(D / "22_train_full_qwen-mg-convnext-mii-w50.csv").set_index("StudyInstanceUID")
aud = set(pd.read_csv(D / "05_audit58.csv").StudyInstanceUID)
seen = set()
for k in range(5):
    for part in ("train", "val"):
        f = pd.read_csv(D / f"21_fold{k}_{part}.csv")
        ids = f.StudyInstanceUID
        assert ids.isin(v4.index).all(), f"fold {k} {part}: studies without v4 labels"
        assert not (set(ids) & aud), "an audit study in a fold"
        f[T] = v4.loc[ids, T].to_numpy()
        assert f[T].notna().all().all() and ((f[T] >= 0) & (f[T] <= 1)).all().all()
        f.to_csv(D / f"27_fold{k}_{part}.csv", index=False)
        if part == "val":
            assert not (set(ids) & seen), "a study in two folds' val"
            seen |= set(ids)
        print(f"fold {k} {part}: {len(f)} studies, mean label {f[T].to_numpy().mean():.3f}")
    tr = set(pd.read_csv(D / f"27_fold{k}_train.csv").StudyInstanceUID)
    va = set(pd.read_csv(D / f"27_fold{k}_val.csv").StudyInstanceUID)
    assert not (tr & va)
print(f"val union {len(seen)} studies (+ {len(aud)} audit in no fold) = {len(seen) + len(aud)}; v4 has {len(v4)}")
assert len(seen) + len(aud) == len(v4)
