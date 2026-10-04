"""specialists_v5_ensemble_val346.py - the specialists inside an ensemble, on the 346 clean v5 val studies (6x audit-58).

Only members trained on the v5 split are clean on that val, so this uses the SIX v5 members of the batch-2 ensemble
(convnext_base_v5, dinov3_convnext_base_v5, convnext_base_25d_18slices_v5_aug_base, convnext_base_25d_poolattn_v5 @3700,
mricore_vitb_cross_attention_v5, mricore_vitb_mae_cross_attention_v5) as a proxy of the full ensemble. Each specialist
votes ONLY on its findings at weight w (0 off, 1 equal share, 2, 3, 5) or takes over; paired bootstrap of each w vs off
over the 346, and a cross-fitted choice of w (chosen on half, scored on the other half, 500 splits).
CAVEAT: the specialists early-stopped on this val (346 + 58), so their numbers are slightly optimistic.

    python runs/others/specialists_v5_ensemble_val346.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUNS = Path(__file__).resolve().parent.parent
MEMBERS = {"convnext_base_v5": None, "dinov3_convnext_base_v5": None, "convnext_base_25d_18slices_v5_aug_base": None,
           "convnext_base_25d_poolattn_v5": 3700, "mricore_vitb_cross_attention_v5": None,
           "mricore_vitb_mae_cross_attention_v5": None}
SPECS = [f"convnext_base_25d_v5_spec_{g}" for g in ("lateral", "pfoa", "fluid", "trauma")]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
WS = [0, 1, 2, 3, 5]

# a per-finding weight plan, scored as one ensemble (findings not listed: specialist off)
PLAN = {"Lateral Meniscus": 1, "PF OA": 3, "Synovitis": 1, "Baker's": 3, "Fracture": 2, "Contusion": 2}


def best_step(run):
    v = pd.read_csv(RUNS / run / "results" / "val_log.csv")
    return int(v.loc[v.val_macro_auroc.idxmax(), "step"])


def load(run, step):
    f = next((RUNS / run / "results" / "predictions").glob(f"*_val_s{step}.npz"))
    z = np.load(f, allow_pickle=True)
    cols = [str(t) for t in z["tasks"]] if "tasks" in z.files else T
    return pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=cols)


v5 = pd.read_csv(RUNS.parent / "data" / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
ids = list(v5.index[v5.source != "audit"])
assert len(ids) == 346
Y = (v5.loc[ids, T] > 0.5).astype(int)
P = {r: load(r, s if s is not None else best_step(r)).loc[ids] for r, s in MEMBERS.items()}
n = len(P)
esum = sum(P[r][T] for r in P)
print(f"six v5 members, 346 clean val: macro {np.mean([roc_auc_score(Y[t], esum[t]) for t in T]):.4f}\n")
rng = np.random.default_rng(0)
tot = 0.0
for r in SPECS:
    S = load(r, best_step(r)).loc[ids]
    for t in S.columns:
        y = Y[t].values
        cols = {w: (esum[t].values + w * S[t].values) / (n + w) for w in WS}
        cols["take"] = S[t].values
        sc = {k: roc_auc_score(y, c) for k, c in cols.items()}
        boot = {w: [] for w in (1, 2, 3)}
        for _ in range(1000):
            i = rng.integers(0, 346, 346)
            b0 = roc_auc_score(y[i], cols[0][i])
            for w in boot:
                boot[w].append(roc_auc_score(y[i], cols[w][i]) - b0)
        cf = []
        for _ in range(500):
            perm = rng.permutation(346)
            a, b = perm[:173], perm[173:]
            pick = max(cols, key=lambda k: roc_auc_score(y[a], cols[k][a]))
            cf.append(roc_auc_score(y[b], cols[pick][b]) - roc_auc_score(y[b], cols[0][b]))
        tot += sc[1] - sc[0]
        print(f"{t:17s} " + "  ".join(f"w{w}={sc[w]:.4f}" for w in WS) + f"  take={sc['take']:.4f}")
        print(f"{'':17s} " + "  ".join(f"w{w} {sc[w] - sc[0]:+.4f} (better {np.mean(np.array(boot[w]) > 0):.0%})" for w in boot)
              + f" | cross-fitted {np.mean(cf):+.4f} (better {np.mean(np.array(cf) > 0):.0%})")
print(f"\nmacro over 12 with every specialist at w=1: {tot / 12:+.4f}")

Sall = {}
for r in SPECS:
    s = load(r, best_step(r)).loc[ids]
    for t in s.columns:
        Sall[t] = s[t].values
plan_cols = {t: ((esum[t].values + PLAN[t] * Sall[t]) / (n + PLAN[t]) if t in PLAN else esum[t].values) for t in T}
m0 = np.mean([roc_auc_score(Y[t], esum[t]) for t in T]); m1 = np.mean([roc_auc_score(Y[t], plan_cols[t]) for t in T])
d = []
for _ in range(1000):
    i = rng.integers(0, 346, 346)
    d.append(np.mean([roc_auc_score(Y[t].values[i], plan_cols[t][i]) - roc_auc_score(Y[t].values[i], esum[t].values[i]) for t in T]))
print(f"\nPLAN {PLAN}: 346 clean val macro {m0:.4f} -> {m1:.4f} ({m1 - m0:+.4f}), better in {np.mean(np.array(d) > 0):.0%}")
