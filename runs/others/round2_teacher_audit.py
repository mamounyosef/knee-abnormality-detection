"""round2_teacher_audit.py - do the ROUND-2 ConvNeXt folds (v4 labels, flipped, 8 Oct) make better labels?

Same audit-58 estimate as data_code/22_make_convnext_mii_labels.py, for both ConvNeXt teachers:
  old  convnext_base_25d_18slices_f{k}          (v2 labels, unflipped) - the 22_* teacher
  new  convnext_base_25d_18slices_v4_flip_f{k}  (v4 labels, flipped, best step picked on the v3-GRADED val rows only)
each blended 50/50 with the MII folds, then into the v3 labels at weight w. Also the teachers alone.
To beat: the current w50 labels, 0.9453.

    python runs/others/round2_teacher_audit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "data"
sys.path.insert(0, str(ROOT / "runs/others/ensembling"))
import sweep_new_runs as sw  # noqa: E402

TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
FAM = {"old ConvNeXt": "convnext_base_25d_18slices_f{}",
       "new ConvNeXt": "convnext_base_25d_18slices_v4_flip_f{}",
       "MII": "medimageinsight_davit_huge_336_v3_f{}"}


def best_npz(run, split):
    """Round-2 folds: best step over the val rows graded by v3 (21_fold val) ONLY. Their first segment validated
    on 27_fold val = v4 labels (half round-1 teacher, ~0.95), so the whole-log argmax would pick a v4-graded step."""
    if "v4_flip" not in run:
        return sw.best_npz(run, split)
    r = ROOT / "runs" / run / "results"
    ok = {f.name[:6] for f in (r / "train_config").glob("run0*.txt")
          if "val_csv: 21_fold" in f.read_text(encoding="utf-8", errors="replace")}
    v = pd.read_csv(r / "val_log.csv")
    v = v[v.run_id.isin(ok)]
    b = v.loc[v.val_macro_auroc.idxmax()]
    f = r / "predictions" / f"{b.run_id}_{split}_s{int(b.step)}.npz"
    assert f.exists(), f
    return f


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
u = list(aud.index)
Y = aud.loc[u, ["official_" + t for t in TASKS]].values
q = pd.read_csv(D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID").loc[u, TASKS].values
mg = pd.read_csv(D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID").loc[u, TASKS].values
bi = [TASKS.index(t) for t in ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]]
v3 = q.copy()
v3[:, bi] = sigmoid(0.5 * logit(q[:, bi]) + 0.5 * logit(mg[:, bi]))
mac = lambda P: np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(12)])   # noqa: E731
per = lambda P: [roc_auc_score(Y[:, j], P[:, j]) for j in range(12)]            # noqa: E731

A = {}
for name, pat in FAM.items():
    fold = []
    for k in range(5):
        f = best_npz(pat.format(k), "audit")
        z = np.load(f, allow_pickle=True)
        fold.append(pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=TASKS).loc[u].values)
        print(f"  {name:13s} f{k}: {Path(f).name}")
    A[name] = np.mean(fold, 0)

print(f"\naudit-58 v3 labels alone {mac(v3):.4f} (expected 0.9311)")
for name in FAM:
    print(f"  teacher alone, {name:13s} {mac(A[name]):.4f}")
print("\n  w     old ConvNeXt + MII   new ConvNeXt + MII   new - old")
res = {}
for w in (0.3, 0.4, 0.5, 0.6, 0.7):
    r = []
    for c in ("old ConvNeXt", "new ConvNeXt"):
        two = 0.5 * A[c] + 0.5 * A["MII"]
        r.append(mac(sigmoid((1 - w) * logit(v3) + w * two)))
    res[w] = r
    print(f"  {w:.1f}   {r[0]:.4f}               {r[1]:.4f}               {r[1] - r[0]:+.4f}")
print("\n  three teachers (old ConvNeXt, new ConvNeXt, MII, a third each):")
three = (A["old ConvNeXt"] + A["new ConvNeXt"] + A["MII"]) / 3
for w in (0.3, 0.4, 0.5, 0.6, 0.7):
    r3 = mac(sigmoid((1 - w) * logit(v3) + w * three))
    print(f"  {w:.1f}   {r3:.4f}  ({r3 - res[w][0]:+.4f} vs old ConvNeXt + MII)")
print("\nper finding at w=0.5 (old -> new):")
po = per(sigmoid(0.5 * logit(v3) + 0.5 * (0.5 * A["old ConvNeXt"] + 0.5 * A["MII"])))
pn = per(sigmoid(0.5 * logit(v3) + 0.5 * (0.5 * A["new ConvNeXt"] + 0.5 * A["MII"])))
for t, a, b in zip(TASKS, po, pn):
    print(f"  {t:17s} {a:.4f} -> {b:.4f} ({b - a:+.4f})")
assert abs(res[0.5][0] - 0.9453) < 5e-4, "old w50 does not reproduce 0.9453"

# ---- why: per-fold scores, per-finding teacher scores, how much each teacher repeats v3 / MII ----
print("\nper fold, audit-58 of each fold model alone (old -> new):")
for k in range(5):
    sc_ = []
    for name in ("old ConvNeXt", "new ConvNeXt"):
        z = np.load(best_npz(FAM[name].format(k), "audit"), allow_pickle=True)
        sc_.append(mac(pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=TASKS).loc[u].values))
    print(f"  f{k}  {sc_[0]:.4f} -> {sc_[1]:.4f} ({sc_[1] - sc_[0]:+.4f})")
print("\nper finding, teacher alone on audit-58: old ConvNeXt | new ConvNeXt | MII | v3 labels")
for t, a, b, m, l in zip(TASKS, per(A["old ConvNeXt"]), per(A["new ConvNeXt"]), per(A["MII"]), per(v3)):
    print(f"  {t:17s} {a:.4f} | {b:.4f} ({b - a:+.4f}) | {m:.4f} | {l:.4f}")


def oof(pat):
    parts = []
    for k in range(5):
        z = np.load(best_npz(pat.format(k), "val"), allow_pickle=True)
        parts.append(pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=TASKS))
    return pd.concat(parts)


O = {n: oof(p) for n, p in FAM.items()}
ids = O["MII"].index
lab = pd.read_csv(D / "17_train_full_qwen-mg-raw.csv").set_index("StudyInstanceUID").loc[ids, TASKS]
L = logit(lab.values)
rk = lambda a, b: np.mean([pd.Series(a[:, j]).corr(pd.Series(b[:, j]), method="spearman") for j in range(12)])  # noqa: E731
print(f"\nout-of-fold, all {len(ids)} training studies: mean rank correlation per finding")
for n in ("old ConvNeXt", "new ConvNeXt"):
    x = O[n].loc[ids].values
    print(f"  {n:13s} with the v3 labels {rk(x, L):.3f}   with MII {rk(x, O['MII'].loc[ids].values):.3f}")
print(f"  MII           with the v3 labels {rk(O['MII'].loc[ids].values, L):.3f}")

rng = np.random.default_rng(0)
bo = sigmoid(0.5 * logit(v3) + 0.5 * (0.5 * A["old ConvNeXt"] + 0.5 * A["MII"]))
bn = sigmoid(0.5 * logit(v3) + 0.5 * (0.5 * A["new ConvNeXt"] + 0.5 * A["MII"]))
d, wins = [], 0
for _ in range(2000):
    i = rng.integers(0, len(u), len(u))
    try:
        x = np.mean([roc_auc_score(Y[i, j], bn[i, j]) - roc_auc_score(Y[i, j], bo[i, j]) for j in range(12)])
    except ValueError:
        continue
    d.append(x)
d = np.array(d)
print(f"\npaired bootstrap of the 58 studies, new - old labels at w=0.5: mean {d.mean():+.4f}, "
      f"90% interval [{np.percentile(d, 5):+.4f}, {np.percentile(d, 95):+.4f}], new better in {np.mean(d > 0):.0%}")
