"""blend_space_test.py - how should the members be averaged? logits (today) vs probabilities vs percentile RANKS (what the
public 0.943 pipeline does) vs ranks of a z-scored logit, on the batch-3 base (14 members, split analogues + SwinV2 v5).

Two surfaces: audit-58 and clean-121 (same construction as old_label_members_ensemble.py, MII folds out-of-fold on the 63).
Paired bootstrap vs the logit mean.

    python runs/others/blend_space_test.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("olm_scaffold", HERE / "old_label_members_ensemble.py")
src = (HERE / "old_label_members_ensemble.py").read_text(encoding="utf-8")
src = src[:src.index("pool = sorted(")]                     # reuse its loaders and surfaces, not its experiment
g = {"__file__": str(HERE / "old_label_members_ensemble.py"), "__name__": "olm_scaffold"}
exec(compile(src, "old_label_members_ensemble.py", "exec"), g)
T, BASE, SURF, get = g["T"], g["BASE"], g["SURF"], g["get"]
sig = lambda x: 1 / (1 + np.exp(-x))   # noqa: E731
rng = np.random.default_rng(0)
for surf, (ids, Y) in SURF.items():
    P = {m: get(m, ids).astype(float) for m in BASE}
    I = np.array([i for i in (rng.integers(0, len(ids), len(ids)) for _ in range(3000))
                  if all(0 < Y[t].values[i].sum() < len(i) for t in T)])

    def bmac(df):
        out = np.zeros(len(I))
        for t in T:
            y, s = Y[t].values[I], df[t].values[I]
            r = rankdata(s, axis=1); n1 = y.sum(1)
            out += ((r * y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * (y.shape[1] - n1))
        return out / len(T)

    mac = lambda df: np.mean([roc_auc_score(Y[t], df[t]) for t in T])   # noqa: E731
    logit = sum(P.values()) / len(P)
    prob = sum(sig(p) for p in P.values()) / len(P)
    rank = sum(p.rank(pct=True) for p in P.values()) / len(P)
    z = sum((p - p.mean()) / p.std() for p in P.values()) / len(P)
    geo = sum(np.log(sig(p)) for p in P.values()) / len(P)
    b = bmac(logit)
    print(f"\n=== {surf} ({len(ids)} studies), {len(P)} members")
    for name, df in (("logit mean (today)", logit), ("probability mean", prob), ("percentile-rank mean (public)", rank),
                     ("z-scored logit mean", z), ("geometric mean of probabilities", geo)):
        d = bmac(df) - b
        print(f"  {name:34s} {mac(df):.4f} ({mac(df) - mac(logit):+.4f})  better in {np.mean(d > 0):.0%}")
