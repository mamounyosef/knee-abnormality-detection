"""external_labels_audit.py - how good is the public pilkwang LLM label set (CC0, Claude-read reports) against the
58 panel labels, alone and blended with our report labels (v3, 05_audit58.csv) and our model teacher (data/19 OOF).

    python runs/others/external_labels_audit.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
au = pd.read_csv(ROOT / "data/05_audit58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
pk = pd.read_csv(ROOT / "data/external/pilkwang_llm_labels/report_labels_v2.csv",
                 dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
print("audit studies without a pilkwang row:", (~au.index.isin(pk.index)).sum(), "(dropped)")
au = au[au.index.isin(pk.index)]
pk = pk.loc[au.index]
Y = au[["official_" + t for t in T]].to_numpy()
ours = au[T].to_numpy()                     # v3 report labels (qwen + medgemma)
pil = pk[T].to_numpy()


def macro(p):
    return np.mean([roc_auc_score(Y[:, j], p[:, j]) for j in range(len(T)) if 0 < Y[:, j].sum() < len(Y)])


def per(p):
    return [roc_auc_score(Y[:, j], p[:, j]) if 0 < Y[:, j].sum() < len(Y) else np.nan for j in range(len(T))]


def rk(p):
    return rankdata(p, axis=0) / len(p)


rows = {"ours v3 labels": ours, "pilkwang labels": pil}
for w in (0.25, 0.5, 0.75):
    rows[f"rank blend pilkwang w{w}"] = (1 - w) * rk(ours) + w * rk(pil)
    rows[f"prob blend pilkwang w{w}"] = (1 - w) * ours + w * pil
print(f"{'source':32s} macro   " + " ".join(f"{t[:7]:>7s}" for t in T))
for k, p in rows.items():
    print(f"{k:32s} {macro(p):.4f}  " + " ".join(f"{v:7.3f}" for v in per(p)))
# how much the two labelers agree on the 4,407 (logit-free: rank correlation per finding)
v3 = pd.read_csv(ROOT / "data/23_train_w50.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
j = v3.index.intersection(pk.index)
print("\nspearman v4 (w50 train labels) vs pilkwang on", len(j), "train studies:")
print(" ".join(f"{t[:7]}={pd.Series(v3.loc[j, t]).rank().corr(pd.Series(pk.loc[j, t]).rank()):.2f}" for t in T))
