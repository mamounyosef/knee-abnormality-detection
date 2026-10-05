"""protocol_features_test.py - does WHAT WAS SCANNED predict the findings? (the public pipeline's calibrator uses protocol counts)

Per study, from the series index (all derivable at test time from test_series.csv flags and the DICOM headers the engine
already reads): series counts per view and per plane, slices / thickness / pixel size / field of view per plane, vendor,
and series-description keywords (3D, DESS, STIR, T1, T2, PD, cartilage, mapping...). No pixels.
Models per finding: logistic regression and histogram gradient boosting, trained on the 3,996 v5-train studies (v4 labels
> 0.5), never on any study scored here.
Scored: alone (macro AUROC) on audit-58, the 346 clean v5 val, and blended into the 6-v5-member ensemble on the 346
(logit space, weights 0.25-1 of one member) and into the batch-3 base on audit-58.

    python runs/others/protocol_features_test.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
R, D = HERE.parent, HERE.parent.parent / "data"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
idx = pd.read_csv(D / "07_series_index.csv")
KW = {"3d": r"3d|cube|space|vista|dess|merge|vibe", "stir": r"stir|tirm", "t1": r"t1", "t2": r"t2", "pd": r"\bpd|dp|proton",
      "fs": r"fs|fat|spir|spair|sat|tirm|stir", "cart": r"cart|chondr|map|t2map|t2\*", "gre": r"gre|fl3d|medic|fiesta|grad"}
idx["desc"] = idx.series_desc.fillna("").str.lower()
rows = {}
for st, g in idx.groupby("study"):
    f = {"n_series": len(g)}
    for s in ("sag_plain", "sag_fluid", "cor_plain", "cor_fluid", "ax_plain", "ax_fluid"):
        f[f"n_{s}"] = int((g.slot == s).sum())
    for p in ("sagittal", "coronal", "axial"):
        gp = g[g.plane == p]
        f[f"slices_{p}"] = gp.n_slices.max() if len(gp) else 0
        f[f"thick_{p}"] = gp.slice_thickness.min() if len(gp) else 0
        f[f"px_{p}"] = gp.pixel_spacing_c.min() if len(gp) else 0
        f[f"fov_{p}"] = gp.fov_w_mm.max() if len(gp) else 0
    for k, rx in KW.items():
        f[f"kw_{k}"] = int(g.desc.str.contains(rx, regex=True).sum())
    man = str(g.manufacturer.mode().iloc[0]).lower() if g.manufacturer.notna().any() else ""
    for v in ("siemens", "ge", "philips", "toshiba", "canon", "hitachi"):
        f[f"vendor_{v}"] = int(v in man)
    rows[st] = f
X = pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(float)
v4 = pd.read_csv(D / "22_train_full_qwen-mg-convnext-mii-w50.csv").set_index("StudyInstanceUID")
tr = list(pd.read_csv(D / "23_train_w50.csv").StudyInstanceUID)
v5 = pd.read_csv(D / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
c346 = list(v5.index[v5.source != "audit"])
Ya = pd.DataFrame((aud[["official_" + t for t in T]].values >= .5).astype(int), index=aud.index, columns=T)
Yv = (v5.loc[c346, T] > .5).astype(int)
print(f"{X.shape[1]} protocol features for {len(X)} studies; train {len(tr)}")

PRED = {}
for name, mk in (("logreg", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))),
                 ("gboost", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                                   l2_regularization=1.0, random_state=0))):
    pa, pv = pd.DataFrame(index=aud.index, columns=T, dtype=float), pd.DataFrame(index=c346, columns=T, dtype=float)
    for t in T:
        m = mk().fit(X.loc[tr].values, (v4.loc[tr, t] > .5).astype(int).values)
        pa[t] = m.decision_function(X.loc[aud.index].values) if hasattr(m, "decision_function") else 0
        pv[t] = m.decision_function(X.loc[c346].values) if hasattr(m, "decision_function") else 0
    PRED[name] = (pa, pv)
    sa = np.mean([roc_auc_score(Ya[t], pa[t]) for t in T]); sv = np.mean([roc_auc_score(Yv[t], pv.loc[c346, t]) for t in T])
    print(f"\n{name}: ALONE macro AUROC audit-58 {sa:.4f} | 346 clean {sv:.4f}")
    print("  per finding (346): " + ", ".join(f"{t} {roc_auc_score(Yv[t], pv.loc[c346, t]):.2f}" for t in T))

# blend into the ensembles
import importlib.util  # noqa: E402
src = (HERE / "old_label_members_ensemble.py").read_text(encoding="utf-8")
g = {"__file__": str(HERE / "old_label_members_ensemble.py"), "__name__": "olm"}
exec(compile(src[:src.index("pool = sorted(")], "olm", "exec"), g)
BASE, get = g["BASE"], g["get"]
V5_SIX = ["convnext_base_v5", "dinov3_convnext_base_v5", "convnext_base_25d_18slices_v5_aug_base",
          "convnext_base_25d_poolattn_v5", "mricore_vitb_cross_attention_v5", "mricore_vitb_mae_cross_attention_v5"]
STEP = {"convnext_base_25d_poolattn_v5": 3700}


def v5pred(m):
    d = R / m / "results" / "predictions"
    if m in STEP:
        f = next(d.glob(f"*_val_s{STEP[m]}.npz"))
    else:
        v = pd.read_csv(R / m / "results" / "val_log.csv"); s = int(v.loc[v.val_macro_auroc.idxmax(), "step"])
        f = next(d.glob(f"*_val_s{s}.npz"))
    z = np.load(f, allow_pickle=True)
    return pd.DataFrame(z["logits"], index=[str(x) for x in z["study_ids"]], columns=T).loc[c346]


rng = np.random.default_rng(0)
for surf, ens, Y, which in (("audit58 (batch-3 base, 14)", sum(get(m, list(aud.index)).astype(float) for m in BASE) / len(BASE), Ya, 0),
                            ("346 clean (6 v5 members)", sum(v5pred(m) for m in V5_SIX) / 6, Yv, 1)):
    n = len(Y)
    I = np.array([i for i in (rng.integers(0, n, n) for _ in range(3000)) if all(0 < Y[t].values[i].sum() < n for t in T)])

    def bmac(df):
        out = np.zeros(len(I))
        for t in T:
            y, s = Y[t].values[I], df[t].values[I]
            r = rankdata(s, axis=1); n1 = y.sum(1)
            out += ((r * y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * (y.shape[1] - n1))
        return out / len(T)

    mac = lambda df: np.mean([roc_auc_score(Y[t], df[t]) for t in T])   # noqa: E731
    b0, bb = mac(ens), bmac(ens)
    k = 14 if which == 0 else 6
    print(f"\n=== {surf}: {b0:.4f}")
    for name, (pa, pv) in PRED.items():
        p = (pa if which == 0 else pv).loc[Y.index].astype(float)
        p = (p - p.mean()) / p.std() * ens.std()              # put it on the ensemble's logit scale
        for w in (0.25, 0.5, 1.0):
            P = (ens * k + w * p) / (k + w)
            print(f"  + {name} w{w:<4} {mac(P):.4f} ({mac(P) - b0:+.4f})  better in {np.mean(bmac(P) - bb > 0):.0%}")
