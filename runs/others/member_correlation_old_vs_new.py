"""member_correlation_old_vs_new.py - did the relabel (v4 labels) make the ensemble members more ALIKE?

Hypothesis (user, 4 Oct): batches 1 and 2 each improved members a lot alone, yet the leaderboard stayed at 0.942. If the
v4 labels (half of each label is a teacher-model prediction: ConvNeXt folds + MedImageInsight folds) pulled every member
toward the same answers, the ensemble gains less from each one.

On audit-58 (the only studies none of these members trained on), for the OLD fourteen (split analogues of the 0.942
submission) and the NEW fourteen (batch 2), per finding then averaged:
  1. mean pairwise correlation of member logits (Pearson and Spearman)
  2. each changed member's correlation with the rest of its ensemble, and with the MedImageInsight-fold mean (a v4 teacher)
  3. the ensemble's gain over its average member: macro AUROC(ensemble) - mean macro AUROC(members)
  4. error correlation: pairwise correlation of residuals (panel label - member probability)
Plus the same diversity numbers among the 6 v5 members on the 346 clean val (more studies, only new members).

    python runs/others/member_correlation_old_vs_new.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ensembling"))
import sweep_new_runs as sw  # noqa: E402
from ten_mii_folds_audit import TEN_SPLIT  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
STEP = {"convnext_base_25d_poolattn_v5": 3700}
SWAP = {"dinov3_convnext_base": "dinov3_convnext_base_v5", "convnext_base_relabel": "convnext_base_v5",
        "convnext_base_25d_all_knees_same_orientation": "convnext_base_25d_18slices_v5_aug_base",
        "convnext_base_25d_poolattn": "convnext_base_25d_poolattn_v5",
        "mricore_vitb_cross_attention": "mricore_vitb_cross_attention_v5",
        "mricore_vitb_mae_cross_attention": "mricore_vitb_mae_cross_attention_v5"}
FOLDS = [f"medimageinsight_davit_huge_336_v3_f{k}" for k in range(5)]
OLD = [TEN_SPLIT[n] for n in TEN_SPLIT if n != "medimageinsight_davit_huge_336_fulldata"] + FOLDS
NEW = [SWAP.get(m, m) for m in OLD]


def load(run, split):
    d = R / run / "results" / "predictions"
    f = next(d.glob(f"*_{split}_s{STEP[run]}.npz")) if run in STEP else Path(sw.best_npz(run, split))
    z = np.load(f, allow_pickle=True)
    return pd.DataFrame(z["logits"], index=[str(x) for x in z["study_ids"]], columns=T)


aud = pd.read_csv(R.parent / "data" / "05_audit58.csv").set_index("StudyInstanceUID")
u = list(aud.index)
Y = pd.DataFrame((aud[["official_" + t for t in T]].values >= .5).astype(int), index=aud.index, columns=T).loc[u]
P = {m: load(m, "audit").loc[u] for m in set(OLD) | set(NEW)}
sig = lambda x: 1 / (1 + np.exp(-x))   # noqa: E731


def mean_pair(ms, fn):
    return np.mean([np.mean([fn(P[a][t].values, P[b][t].values) for t in T]) for a, b in itertools.combinations(ms, 2)])


pear = lambda a, b: np.corrcoef(a, b)[0, 1]          # noqa: E731
spear = lambda a, b: spearmanr(a, b).correlation      # noqa: E731


def err_pair(ms):
    return np.mean([np.mean([np.corrcoef(Y[t] - sig(P[a][t]), Y[t] - sig(P[b][t]))[0, 1] for t in T])
                    for a, b in itertools.combinations(ms, 2)])


def macro(df):
    return np.mean([roc_auc_score(Y[t], df[t]) for t in T])


print("=== audit-58, all 14 members (old = the 0.942 fourteen's split analogues, new = batch 2)\n")
print(f"{'':28s} {'old':>8s} {'new':>8s} {'change':>8s}")
for name, f in (("mean pairwise Pearson", lambda ms: mean_pair(ms, pear)),
                ("mean pairwise Spearman", lambda ms: mean_pair(ms, spear)),
                ("mean pairwise ERROR corr", err_pair)):
    a, b = f(OLD), f(NEW)
    print(f"{name:28s} {a:8.4f} {b:8.4f} {b - a:+8.4f}")
for name, ms in (("old", OLD), ("new", NEW)):
    ens = sum(P[m] for m in ms) / len(ms)
    mm = np.mean([macro(P[m]) for m in ms])
    print(f"{name}: ensemble {macro(ens):.4f}, mean member {mm:.4f}, ensemble gain over the average member {macro(ens) - mm:+.4f}")

mii = sum(P[f] for f in FOLDS) / 5
print("\nthe 6 changed members: correlation with the REST of their ensemble, and with the MedImageInsight-fold mean (a v4 teacher)")
print(f"{'old -> new':72s} {'vs rest old':>11s} {'vs rest new':>11s} {'vs MII old':>10s} {'vs MII new':>10s} {'AUROC old':>9s} {'AUROC new':>9s}")
for o, n in SWAP.items():
    rest_o = sum(P[m] for m in OLD if m != o) / 13
    rest_n = sum(P[m] for m in NEW if m != n) / 13
    c = lambda x, y: np.mean([pear(x[t].values, y[t].values) for t in T])   # noqa: E731
    print(f"{o + ' -> ' + n.replace('convnext_base_25d_18slices_', ''):72s} {c(P[o], rest_o):11.4f} {c(P[n], rest_n):11.4f} "
          f"{c(P[o], mii):10.4f} {c(P[n], mii):10.4f} {macro(P[o]):9.4f} {macro(P[n]):9.4f}")
six_o, six_n = list(SWAP), list(SWAP.values())
print(f"\nwithin the 6 changed members only: pairwise Pearson old {mean_pair(six_o, pear):.4f} -> new {mean_pair(six_n, pear):.4f}; "
      f"error corr old {err_pair(six_o):.4f} -> new {err_pair(six_n):.4f}")
print(f"the 6 changed members vs the 8 unchanged: mean cross-correlation old "
      f"{np.mean([np.mean([pear(P[a][t].values, P[b][t].values) for t in T]) for a in six_o for b in OLD if b not in six_o]):.4f} -> new "
      f"{np.mean([np.mean([pear(P[a][t].values, P[b][t].values) for t in T]) for a in six_n for b in NEW if b not in six_n]):.4f}")


# does keeping BOTH versions (diversity) beat replacing? audit-58, paired bootstrap vs the new fourteen
from scipy.stats import rankdata  # noqa: E402
rng = np.random.default_rng(0)
I = np.array([i for i in (rng.integers(0, 58, 58) for _ in range(3000)) if all(0 < Y[t].values[i].sum() < 58 for t in T)])


def bmac(df):
    out = np.zeros(len(I))
    for t in T:
        y, s = Y[t].values[I], df[t].values[I]
        r = rankdata(s, axis=1); n1 = y.sum(1)
        out += ((r * y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * (y.shape[1] - n1))
    return out / len(T)


new_ens = sum(P[m] for m in NEW) / len(NEW)
bn = bmac(new_ens)
print("\nkeep both versions? (audit-58, vs the new fourteen", f"{macro(new_ens):.4f})")
cands = {
    "old fourteen (the 0.942 one)": sum(P[m] for m in OLD) / len(OLD),
    "nested: each of the 6 = mean(old, new) as one vote": sum((P[o] + P[n]) / 2 for o, n in SWAP.items()) / 14
                                                          + sum(P[m] for m in OLD if m not in SWAP) / 14,
    "union: old fourteen + the 6 new (20 members)": (sum(P[m] for m in OLD) + sum(P[n] for n in SWAP.values())) / 20,
}
for k, df in cands.items():
    d = bmac(df) - bn
    print(f"  {k:52s} {macro(df):.4f} ({macro(df) - macro(new_ens):+.4f})  better in {np.mean(d > 0):.0%}")
