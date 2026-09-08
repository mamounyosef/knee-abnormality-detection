"""Does convnext_base_seed1337 earn an ensemble slot?

It loses solo by 0.0023, which settles nothing - dinov3_convnext_base loses solo by 0.0039
and earns the biggest slot added since the relabel. The question is only whether its ERRORS
differ from its parent's, and two runs off the same trunk and the same data are the most
correlated pair this project has ever measured.

Flat blending in logit space, scored on val, targets binarized at 0.5.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path("d:/My Projects/knee_abnormality_detection")

S = "convnext_small_relabel"
B = "convnext_base_relabel"
D = "dinov3_convnext_base"
C = "mricore_vitb_cross_attention"
Z = "convnext_base_seed1337"

COMBOS = [
    ("LB0.928-b   S+B+C+D", [S, B, C, D]),
    ("+seed1337   S+B+C+D+Z", [S, B, C, D, Z]),
    ("swap B->Z   S+Z+C+D", [S, Z, C, D]),
    ("pair        B+Z", [B, Z]),
    ("pair        B+D  (reference)", [B, D]),
    ("solo        B", [B]),
    ("solo        Z", [Z]),
]

_cache = {}


def load(run):
    """That run's best validation, the one ensemble.py would pick as its member."""
    if run in _cache:
        return _cache[run]
    best, best_a = None, -1
    for f in (REPO / "runs" / run / "results/predictions").glob("*_val_*.npz"):
        z = np.load(f, allow_pickle=True)
        lg, y = z["logits"], z["targets"]
        aus = []
        for i in range(len(z["tasks"])):
            m = ~np.isnan(y[:, i])
            yb = (y[m, i] > 0.5).astype(int)
            if len(np.unique(yb)) > 1:
                aus.append(roc_auc_score(yb, lg[m, i]))
        a = float(np.mean(aus))
        if a > best_a:
            best, best_a = f, a
    z = np.load(best, allow_pickle=True)
    o = np.argsort(np.asarray([str(s) for s in z["study_ids"]]))
    _cache[run] = (z["logits"][o], z["targets"][o], [str(t) for t in z["tasks"]], best_a)
    return _cache[run]


def macro(logits, Y):
    aus = []
    for i in range(logits.shape[1]):
        m = ~np.isnan(Y[:, i])
        yb = (Y[m, i] > 0.5).astype(int)
        if len(np.unique(yb)) > 1:
            aus.append(roc_auc_score(yb, logits[m, i]))
    return float(np.mean(aus))


rows = []
for tag, members in COMBOS:
    L, Y = [], None
    for r in members:
        lg, y, tk, _ = load(r)
        Y = y if Y is None else Y
        L.append(lg)
    rows.append((tag, macro(np.stack(L).mean(axis=0), Y)))

base = dict(rows)["LB0.928-b   S+B+C+D"]
print("%-30s %9s %11s" % ("combination", "val", "vs LB0.928"))
print("-" * 54)
for tag, a in rows:
    print("%-30s %9.4f %11s" % (tag, a, "%+.4f" % (a - base)))
print("-" * 54)

# how correlated are the two seeds, really? residual correlation is what decides a slot
bl, Y, _, _ = load(B)
zl, _, _, _ = load(Z)
dl, _, _, _ = load(D)
for other, name in ((zl, "seed1337"), (dl, "dinov3")):
    cs = []
    for i in range(Y.shape[1]):
        m = ~np.isnan(Y[:, i])
        yb = (Y[m, i] > 0.5).astype(float)
        # residual against the label, per finding, then correlate the two members'
        ra = bl[m, i] - yb * bl[m, i].std()
        rb = other[m, i] - yb * other[m, i].std()
        cs.append(np.corrcoef(ra, rb)[0, 1])
    print(f"mean per-finding logit correlation, convnext_base vs {name}: {np.mean(cs):.3f}")
