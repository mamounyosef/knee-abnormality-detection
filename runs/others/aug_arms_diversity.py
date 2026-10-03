"""aug_arms_diversity.py - are the three augmentation arms different enough to help an ensemble?

1. per finding, rank correlation between the arms' logits on the v5 val (404 studies); for scale,
   the same correlation between base and two models of OTHER families on the 58 audit studies
2. the 3-arm average vs each arm alone on the v5 val (paired bootstrap)
3. audit-58: the current 14 (split analogues, as in relabel_batch_plan.py) + one or all arms, flat.
   Caveat: the arms' val included the 58, so their audit numbers are slightly optimistic.

    python runs/others/aug_arms_diversity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ensembling"))
import sweep_new_runs as sw  # noqa: E402
from ten_mii_folds_audit import TEN_SPLIT  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ARMS = ["convnext_base_25d_18slices_v5_aug_base", "convnext_base_25d_18slices_v5_aug_strong",
        "convnext_base_25d_18slices_v5_aug_strong_extra"]
SH = {a: a.replace("convnext_base_25d_18slices_v5_aug_", "") for a in ARMS}
D = RUNS.parent / "data"
rng = np.random.default_rng(0)


def preds(run, split, ids):
    z = np.load(sw.best_npz(run, split), allow_pickle=True)
    return pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T).loc[ids].values


def macro(Y, P):
    return np.nanmean([roc_auc_score(Y[:, j], P[:, j]) if 0 < Y[:, j].sum() < len(Y) else np.nan for j in range(12)])


def rho(a, b):
    return [pd.Series(a[:, j]).corr(pd.Series(b[:, j]), method="spearman") for j in range(12)]


val = pd.read_csv(D / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
vid = list(val.index)
Yv = (val.loc[vid, T].values > 0.5).astype(int)
V = {a: preds(a, "val", vid) for a in ARMS}

print("1. rank correlation per finding, v5 val 404 studies (1.0 = identical ranking)")
pairs = [(ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])]
R = {f"{SH[a]}~{SH[b]}": rho(V[a], V[b]) for a, b in pairs}
aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
aid = list(aud.index)
Ya = (aud.loc[aid, ["official_" + t for t in T]].values >= 0.5).astype(int)
A = {a: preds(a, "audit", aid) for a in ARMS}
for other in ("medimageinsight_davit_huge_336_v3_f0", "mricore_vitb_cross_attention"):
    R[f"base~{other[:22]} (audit 58)"] = rho(A[ARMS[0]], preds(other, "audit", aid))
tab = pd.DataFrame(R, index=T).round(3)
tab.loc["MEAN"] = tab.mean().round(3)
pd.set_option("display.width", 250)
print(tab.to_string())

print("\n2. averaging the three arms (logit mean), v5 val")
avg = np.mean([V[a] for a in ARMS], 0)
m_avg = macro(Yv, avg)
for a in ARMS:
    d = []
    for _ in range(2000):
        r = rng.integers(0, len(vid), len(vid))
        d.append(macro(Yv[r], avg[r]) - macro(Yv[r], V[a][r]))
    d = np.array(d)
    print(f"  3-arm average {m_avg:.4f} vs {SH[a]:12s} {macro(Yv, V[a]):.4f}: {m_avg - macro(Yv, V[a]):+.4f}"
          f"  90% [{np.percentile(d, 5):+.4f}, {np.percentile(d, 95):+.4f}]")

print("\n3. audit-58, the current 14 (split analogues) plus arms, flat logit mean")
NINE = [k for k in TEN_SPLIT if k != "medimageinsight_davit_huge_336_fulldata"]
FOLDS = [f"medimageinsight_davit_huge_336_v3_f{k}" for k in range(5)]
mem = {n: preds(TEN_SPLIT[n], "audit", aid) for n in NINE}
mem.update({f: preds(f, "audit", aid) for f in FOLDS})
base14 = list(mem.values())
m14 = macro(Ya, np.mean(base14, 0))
print(f"  the 14                       {m14:.4f}")
for lab, add in [(f"+ {SH[a]}", [A[a]]) for a in ARMS] + [("+ all three arms", [A[a] for a in ARMS])]:
    m = macro(Ya, np.mean(base14 + add, 0))
    print(f"  {lab:28s} {m:.4f} ({m - m14:+.4f})")
