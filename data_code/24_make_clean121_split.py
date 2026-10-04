"""
24_make_clean121_split.py - split for WARM-STARTING an old-split model (first: SwinV2) on the v4 labels.

WHY (user, 4 Oct). A model trained on the old 05 split has already seen 283 of the 346 v5 val studies,
so the v5 val is leaky for it. The studies it never saw: the 63 v5 val studies outside
05_train_qwen24t-med.csv, plus the 58 audit studies = 121 "clean" studies. Validate on those 121
(early stopping), train on everything else (4,279 = 97% of the data).

LABELS (same rule as 23_make_split8.py)
  train rows  v4 (w50, 22_train_full_qwen-mg-convnext-mii-w50.csv)
  val rows    the v5 val's own labels: v3 for the 63, the panel's 0/1 for the 58 audit

    python data_code/24_make_clean121_split.py

Outputs: data/24_train_w50_minus_clean121.csv, data/24_val_clean121.csv
"""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent.parent / "data"
full = pd.read_csv(D / "22_train_full_qwen-mg-convnext-mii-w50.csv")
v5val = pd.read_csv(D / "23_val_v3_audit.csv")
old_train = set(pd.read_csv(D / "05_train_qwen24t-med.csv").StudyInstanceUID)

val = v5val[(v5val.source == "audit") | ~v5val.StudyInstanceUID.isin(old_train)].reset_index(drop=True)
train = full[~full.StudyInstanceUID.isin(set(val.StudyInstanceUID))].reset_index(drop=True)

# checks
assert full.StudyInstanceUID.is_unique and len(full) == 4400
assert (val.source == "audit").sum() == 58 and len(val) == 121, len(val)
assert len(train) + len(val) == 4400
assert not set(val.StudyInstanceUID) & old_train, "a clean study was in the old training set"
shared = set(train.group) & set(val.group)                 # report-text groups must not straddle
assert not shared, f"{len(shared)} report groups straddle train/val"
assert list(train.columns) == list(val.columns)

train.to_csv(D / "24_train_w50_minus_clean121.csv", index=False)
val.to_csv(D / "24_val_clean121.csv", index=False)
print(f"train {len(train)} (v4 labels) | val {len(val)} = {(val.source != 'audit').sum()} v3 + "
      f"{(val.source == 'audit').sum()} audit | no shared report groups")
labels = list(full.columns[4:])
print("positive rate, train vs val:")
for t in labels:
    print(f"  {t:17s} {train[t].mean():.3f}  {(val[t] > 0.5).mean():.3f}")
