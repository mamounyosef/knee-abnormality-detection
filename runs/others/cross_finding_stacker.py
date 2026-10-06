"""cross_finding_stacker.py - can a second stage that sees ALL 12 ensemble outputs beat the plain per-finding logit mean?
(co-occurrence: ACL tears come with contusions, OA with meniscus tears and effusion, ...)

Base: the batch-3 scaffold fourteen (old_label_members_ensemble.py), flat logit mean = what we ship.
Stacker per finding j: logistic regression on the 12 ensemble-mean logits, its own logit fixed as an offset feature with
strong L2 on the others, so it can only ADD cross-finding evidence. Out-of-sample by repeated stratified K-fold
over the pooled 179 studies (audit-58 panel labels + clean-121 report labels); scored per surface.

    python runs/others/cross_finding_stacker.py
"""
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
src = (HERE / "old_label_members_ensemble.py").read_text(encoding="utf-8")
g = {"__file__": str(HERE / "old_label_members_ensemble.py"), "__name__": "olm_scaffold"}
exec(compile(src[:src.index("pool = sorted(")], "old_label_members_ensemble.py", "exec"), g)
T, BASE, SURF, get = g["T"], g["BASE"], g["SURF"], g["get"]

Xs, Ys, S = [], [], []
for s, (ids, Y) in SURF.items():
    L = np.stack([get(m, ids).loc[ids, T].astype(float).to_numpy() for m in BASE]).mean(0)
    Xs.append(L); Ys.append(Y.loc[ids, T].to_numpy().astype(int)); S += [s] * len(ids)
X, Y, S = np.vstack(Xs), np.vstack(Ys), np.array(S)
Xz = (X - X.mean(0)) / X.std(0)


def auc(y, p):
    r = rankdata(p); pos = y == 1; n1 = pos.sum(); n0 = len(y) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def stack_oof(Cs, reps=10):
    out = np.zeros((reps,) + X.shape)
    for rep in range(reps):
        for j in range(len(T)):
            others = [k for k in range(len(T)) if k != j]
            strat = Y[:, j] * 2 + (S == "audit58")
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, strat):
                m = LogisticRegression(C=Cs, max_iter=2000)
                # own logit as a free offset (scaled copy so L2 barely touches it), others penalised
                F = np.column_stack([Xz[:, j] * 10, Xz[:, others]])
                m.fit(F[tr], Y[tr, j])
                out[rep, te, j] = m.decision_function(F[te])
    return out


def report(name, P):
    res = {}
    for s in ("audit58", "clean121"):
        idx = S == s
        reps = P if P.ndim == 3 else P[None]
        res[s] = np.mean([np.nanmean([auc(Y[idx, j], r[idx, j]) for j in range(len(T))]) for r in reps])
    print(f"{name:28s} audit {res['audit58']:.4f} | clean-121 {res['clean121']:.4f}")
    return res


base = report("logit mean (shipped)", X)
for C in (0.01, 0.03, 0.1, 0.3):
    r = report(f"stacker C={C}", stack_oof(C))
    print(f"{'':28s} delta audit {r['audit58'] - base['audit58']:+.4f} | clean {r['clean121'] - base['clean121']:+.4f}")
