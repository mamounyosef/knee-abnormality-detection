"""pair_blend_test.py - blend TWO runs alone (mean logit at share w of B), scored on val-346 (v3), audit-58 and
clean-121 from each run's val npz at the given step, with a paired bootstrap vs A alone (11 Oct).

    python runs/others/pair_blend_test.py <runA>:<step> <runB>:<step>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]


def load(arg):
    run, step = arg.split(":")
    f = sorted((ROOT / "runs" / run / "results/predictions").glob(f"*_val_s{step}.npz"))[-1]
    z = np.load(f, allow_pickle=True)
    return run, pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T)


(na, A), (nb, B) = load(sys.argv[1]), load(sys.argv[2])
val = pd.read_csv(ROOT / "data/23_val_v3_audit.csv").set_index("StudyInstanceUID")
aud = set(pd.read_csv(ROOT / "data/05_audit58.csv").StudyInstanceUID)
clean = pd.read_csv(ROOT / "data/24_val_clean121.csv").set_index("StudyInstanceUID")
ids = [s for s in A.index if s in B.index]
SURF = {"val-346": val.loc[[s for s in ids if s not in aud], T],
        "audit-58": val.loc[[s for s in ids if s in aud], T],
        "clean-121": clean.loc[[s for s in clean.index if s in ids], T]}
rng = np.random.default_rng(0)


def auc_cols(y, p, I):
    out = np.zeros(len(I)); cnt = np.zeros(len(I))
    for j in range(y.shape[1]):
        yy, pp = y[I, j], p[I, j]
        n1 = yy.sum(1); n0 = yy.shape[1] - n1
        ok = (n1 > 0) & (n0 > 0)
        r = rankdata(pp, axis=1)
        out += np.where(ok, ((r * yy).sum(1) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1), 0); cnt += ok
    return out / cnt


print(f"A = {na}\nB = {nb}\nlogit r(A, B) on all {len(ids)}: {np.corrcoef(A.loc[ids].values.ravel(), B.loc[ids].values.ravel())[0, 1]:.3f}")
W = [0.0, 0.3, 0.4, 0.5, 0.6, 1.0]
res = {w: [] for w in W}
for s, Ydf in SURF.items():
    y = (Ydf.values >= 0.5).astype(int)
    I = rng.integers(0, len(y), (2000, len(y)))
    full = np.arange(len(y))[None]
    a, b = A.loc[Ydf.index].values, B.loc[Ydf.index].values
    a0, aI = auc_cols(y, a, full)[0], auc_cols(y, a, I)
    for w in W:
        p = (1 - w) * a + w * b
        v = auc_cols(y, p, full)[0]
        res[w].append((v, v - a0, (auc_cols(y, p, I) - aI > 0).mean()))
print(f"{'share of B':10s} " + " | ".join(f"{s:>26s}" for s in SURF))
for w in W:
    print(f"{w:10.1f} " + " | ".join(f"{v:.4f} ({d:+.4f}, {p:4.0%})" for v, d, p in res[w]))
