"""ortho_ensemble_test.py - does orthofoundation_vitl_cross_attention_v5 (any saved val step) add to the ensemble?

Surfaces, both unseen by every member involved:
  clean-346  the v5 val minus audit, with the six v5-split members (same set as protocol_features_test.py)
  audit-58   panel labels, with the batch-3 scaffold base (14, old_label_members_ensemble.py)
Variants: + Ortho at weight 1; Ortho in place of mricore_vitb_cross_attention_v5 (its donor). Flat logit mean, paired
bootstrap vs the base. Also Ortho's logit correlation with each base member on clean-346.

    python runs/others/ortho_ensemble_test.py [step ...]      (default: every saved val step >= 1200)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
R = HERE.parent
src = (HERE / "old_label_members_ensemble.py").read_text(encoding="utf-8")
g = {"__file__": str(HERE / "old_label_members_ensemble.py"), "__name__": "olm"}
exec(compile(src[:src.index("pool = sorted(")], "olm", "exec"), g)
T, BASE, get = g["T"], g["BASE"], g["get"]
D = R.parent / "data"
v5 = pd.read_csv(D / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
c346 = list(v5.index[v5.source != "audit"])
Yv = (v5.loc[c346, T] > .5).astype(int)
Ya = pd.DataFrame((aud[["official_" + t for t in T]].values >= .5).astype(int), index=aud.index, columns=T)
V5_SIX = ["convnext_base_v5", "dinov3_convnext_base_v5", "convnext_base_25d_18slices_v5_aug_base",
          "convnext_base_25d_poolattn_v5", "mricore_vitb_cross_attention_v5", "mricore_vitb_mae_cross_attention_v5"]
STEP = {"convnext_base_25d_poolattn_v5": 3700}
ORTHO = "orthofoundation_vitl_cross_attention_v5"


def npz(run, step=None):
    d = R / run / "results" / "predictions"
    if step is None:
        if run in STEP:
            step = STEP[run]
        else:
            v = pd.read_csv(R / run / "results" / "val_log.csv")
            step = int(v.loc[v.val_macro_auroc.idxmax(), "step"])
    z = np.load(next(d.glob(f"*_val_s{step}.npz")), allow_pickle=True)
    return pd.DataFrame(z["logits"], index=[str(x) for x in z["study_ids"]], columns=T)


steps = [int(s) for s in sys.argv[1:]] or sorted(
    int(p.stem.split("_s")[-1]) for p in (R / ORTHO / "results" / "predictions").glob("*_val_s*.npz")
    if int(p.stem.split("_s")[-1]) >= 1200)
rng = np.random.default_rng(0)


def surface(name, members, Y, getter):
    ids = list(Y.index)
    n = len(ids)
    I = np.array([i for i in (rng.integers(0, n, n) for _ in range(3000)) if all(0 < Y[t].values[i].sum() < n for t in T)])

    def bmac(df):
        out = np.zeros(len(I))
        for t in T:
            y, s = Y[t].values[I], df.loc[ids, t].values[I]
            r = rankdata(s, axis=1); n1 = y.sum(1)
            out += ((r * y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * (y.shape[1] - n1))
        return out / len(T)

    mac = lambda df: np.mean([roc_auc_score(Y[t], df.loc[ids, t]) for t in T])   # noqa: E731
    P = {m: getter(m).loc[ids].astype(float) for m in members}
    base = sum(P.values()) / len(P)
    b0, bb = mac(base), bmac(base)
    print(f"\n=== {name}: base {len(P)} members {b0:.4f}")
    for st in steps:
        o = npz(ORTHO, st).loc[ids].astype(float)
        add = (base * len(P) + o) / (len(P) + 1)
        donor = next((m for m in P if "mricore_vitb_cross_attention" in m and "mae" not in m), None)
        swap = (base * len(P) - P[donor] + o) / len(P) if donor else None
        line = f"  step {st:5d}: alone {mac(o):.4f} | + Ortho {mac(add):.4f} ({mac(add) - b0:+.4f}, {np.mean(bmac(add) - bb > 0):.0%})"
        if swap is not None:
            line += f" | for {donor.split('_v5')[0][-24:]} {mac(swap):.4f} ({mac(swap) - b0:+.4f}, {np.mean(bmac(swap) - bb > 0):.0%})"
        print(line)
    return P


P = surface("clean-346 (six v5 members)", V5_SIX, Yv, lambda m: npz(m))
surface("audit-58 (batch-3 base)", BASE, Ya, lambda m: get(m, list(Ya.index)).astype(float)
        if m != ORTHO else npz(m))
o = npz(ORTHO, steps[-1]).loc[c346]
print(f"\nlogit correlation of Ortho step {steps[-1]} with the v5 members (clean-346, mean over findings):")
for m, df in P.items():
    print(f"  {m:45s} {np.mean([np.corrcoef(o[t], df[t])[0, 1] for t in T]):.3f}")
