"""raptor_folds_eval.py - what the 5 raptor96 folds are worth, solo and inside raptor4 (13 Oct, user: all 5 folds done).

Fold family = convnext_base_v5_raptor96_f0..f4, each at its best-val step.
  per fold      own fold val (v3-graded), audit-58
  family        audit-58 = fold mean; clean-121 = fold mean on the 58 + each fold's out-of-fold prediction on the 63;
                OOF-4342 = every study scored by the one fold that did not train on it (v3 labels)
  reference     convnext_base_25d_18slices_v4_flip_f0..f4 (same round-2 split) and the single raptor96_local
Ensembles: raptor4 (LB 0.942) with the single raptor swapped for the family, or sharing its weight, and a share sweep.
P(a) / P(c) = paired-bootstrap probability of beating raptor4 on audit-58 / clean-121.

    python runs/others/ensembling/raptor_folds_eval.py [n_bootstrap, default 2000]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent.parent
D = R.parent / "data"
sys.path.insert(0, str(HERE))
import sweep_new_runs as sw  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
FAM = "convnext_base_v5_raptor96"
REF = "convnext_base_25d_18slices_v4_flip"
SINGLE = "convnext_base_v5_raptor96_local"
STEP = {"convnext_base_v5_raptor96_local": 9000, "medimageinsight_davit_huge_v5_raptor96_sam": 5300,
        "swinv2_base_in22k_v5_warm": 2000}
RAPTOR4 = json.loads((HERE / "raptor4_weights.json").read_text(encoding="utf-8"))
W4 = RAPTOR4["weights_per_class"]["ACL"]

aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
v5 = pd.read_csv(D / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
old_train = set(pd.read_csv(D / "05_train_qwen24t-med.csv").StudyInstanceUID)
A58 = list(aud.index)
C63 = [s for s in v5.index if v5.loc[s, "source"] != "audit" and s not in old_train]
YA = (aud.loc[A58, ["official_" + t for t in T]].values >= .5).astype(int)
YC = np.vstack([YA, (v5.loc[C63, T].values > .5).astype(int)])


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(12) if 0 < y[:, j].sum() < len(y)]))


def load(run, split):
    d = R / run / "results" / "predictions"
    if run in STEP:
        f = sorted(d.glob(f"*_{split}_s{STEP[run]}.npz"))[-1]
    else:
        f = sw.best_npz(run, split)
    z = np.load(f, allow_pickle=True)
    return pd.DataFrame(z["logits"], index=[str(x) for x in z["study_ids"]], columns=T), f.stem


def best_val(run):
    v = pd.read_csv(R / run / "results" / "val_log.csv")
    r = v.loc[v.val_macro_auroc.idxmax()]
    return int(r.step), float(r.val_macro_auroc)


def family(base, report=True):
    aud_p, oof, oy = [], [], []
    for k in range(5):
        run = f"{base}_f{k}"
        va = pd.read_csv(D / f"21_fold{k}_val.csv").set_index("StudyInstanceUID")
        a, _ = load(run, "audit")
        v, stem = load(run, "val")
        step, bv = best_val(run)
        aud_p.append(a.loc[A58, T].values)
        ids = [s for s in va.index if s in v.index]
        oof.append(v.loc[ids, T]); oy.append(va.loc[ids, T])
        if report:
            print(f"   f{k}  best step {step:5d}  fold val {bv:.4f}  audit-58 {auc(YA, a.loc[A58, T].values):.4f}  "
                  f"({len(ids)}/{len(va)} val studies, {stem})")
    o = pd.concat(oof)
    y = (pd.concat(oy).values > .5).astype(int)
    am = np.mean(aud_p, 0)
    clean = np.vstack([am, o.loc[C63, T].values])
    return am, clean, auc(y, o.values), len(o)


def single(run):
    a, _ = load(run, "audit")
    v, _ = load(run, "val")
    return a.loc[A58, T].values, np.vstack([a.loc[A58, T].values, v.loc[C63, T].values])


rng = np.random.default_rng(0)


def boots(n, y, k=2000):
    b = rng.integers(0, n, (k, n))
    return b[[all(0 < y[i, j].sum() < n for j in range(12)) for i in b]]


BA, BC = boots(len(A58), YA, int(sys.argv[1]) if len(sys.argv) > 1 else 2000), \
    boots(len(YC), YC, int(sys.argv[1]) if len(sys.argv) > 1 else 2000)


def pbetter(p, q, y, B):
    from sklearn.metrics import roc_auc_score
    w = 0
    for i in B:
        yy = y[i]
        a = np.mean([roc_auc_score(yy[:, j], p[i, j]) for j in range(12)])
        c = np.mean([roc_auc_score(yy[:, j], q[i, j]) for j in range(12)])
        w += a > c
    return w / len(B)


print(f"RAPTOR96 FOLDS ({FAM}_f0..f4)")
fa, fc, foof, n = family(FAM)
print(f"   family: audit-58 {auc(YA, fa):.4f} | clean-121 {auc(YC, fc):.4f} | OOF-{n} {foof:.4f}")
print(f"\nREFERENCE {REF}_f0..f4 (same split, 18-slice recipe)")
ra, rc, roof, rn = family(REF)
print(f"   family: audit-58 {auc(YA, ra):.4f} | clean-121 {auc(YC, rc):.4f} | OOF-{rn} {roof:.4f}")
sa, sc = single(SINGLE)
print(f"\nSINGLE {SINGLE} (LB 0.939): audit-58 {auc(YA, sa):.4f} | clean-121 {auc(YC, sc):.4f}")

M = {m: single(m) for m in W4}
M["FOLDS"] = (fa, fc)


def blend(w):
    a = sum(x * M[m][0] for m, x in w.items()) / sum(w.values())
    c = sum(x * M[m][1] for m, x in w.items()) / sum(w.values())
    return a, c


base_a, base_c = blend(W4)
print(f"\nENSEMBLES (logit mean; P = bootstrap P(beats raptor4))")
print(f"   {'variant':52s} {'audit-58':>8} {'clean-121':>9} {'P(a)':>5} {'P(c)':>5}")


def row(name, w):
    a, c = blend(w)
    pa = pbetter(a, base_a, YA, BA) if w != W4 else float("nan")
    pc = pbetter(c, base_c, YC, BC) if w != W4 else float("nan")
    print(f"   {name:52s} {auc(YA, a):8.4f} {auc(YC, c):9.4f} {pa:5.0%} {pc:5.0%}")


rest = {m: x for m, x in W4.items() if m != SINGLE}
row("raptor4 as submitted (LB 0.942)", W4)
row("raptor4, single raptor -> folds (0.42)", {**rest, "FOLDS": 0.42})
row("raptor4, single 0.21 + folds 0.21", {**rest, SINGLE: 0.21, "FOLDS": 0.21})
for s in (0.30, 0.50, 0.60):
    k = (1 - s) / (1 - 0.42)
    row(f"folds {s:.2f}, others rescaled (no single)", {**{m: x * k for m, x in rest.items()}, "FOLDS": s})
    row(f"folds {s / 2:.2f} + single {s / 2:.2f}, others rescaled", {**{m: x * k for m, x in rest.items()},
                                                                  "FOLDS": s / 2, SINGLE: s / 2})
for s in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
    row(f"raptor4 + folds {s:.2f} (all 4 kept, x{1 - s:.2f})", {**{m: x * (1 - s) for m, x in W4.items()}, "FOLDS": s})
row("pair: folds 0.7 + MII 0.3",{"FOLDS": 0.7, "medimageinsight_davit_huge_v5_raptor96_sam": 0.3})
row("pair: single 0.7 + MII 0.3 (LB 0.940)", {SINGLE: 0.7, "medimageinsight_davit_huge_v5_raptor96_sam": 0.3})
