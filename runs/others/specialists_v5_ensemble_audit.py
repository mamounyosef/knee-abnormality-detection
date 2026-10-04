"""specialists_v5_ensemble_audit.py - do the v5 specialists help the batch-2 ensemble, finding by finding?

Base: the batch-2 fourteen (LB pending, ref 56824782) as split-data analogues on audit-58 (the only fair surface;
same construction as relabel_batch2_swap_audit.py). Each specialist votes ONLY on its own findings:
    ensemble logit = (sum of the 14 members + w x specialist) / (14 + w)     w = 0 (off), 1 (equal share), 2, 3, 5
    takeover       = the specialist alone
Reported per finding: audit-58 AUROC for each w, paired bootstrap of w=1 vs off, and a CROSS-FITTED choice of w
(pick the best w on a random half of the 58, score it on the other half, 500 splits) so the weight is not picked
on noise. Also each specialist vs its parent on the 346 clean v5 val studies (the 58 audit removed), alone.
Specialists at their best-val step (the 58 sit inside the val they early-stopped on: slightly optimistic).

    python runs/others/specialists_v5_ensemble_audit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ensembling"))
import sweep_new_runs as sw  # noqa: E402
from ten_mii_folds_audit import TEN_SPLIT  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
B12 = {"dinov3_convnext_base": "dinov3_convnext_base_v5", "convnext_base_relabel": "convnext_base_v5",
       "convnext_base_25d_all_knees_same_orientation": "convnext_base_25d_18slices_v5_aug_base",
       "convnext_base_25d_poolattn": "convnext_base_25d_poolattn_v5",
       "mricore_vitb_cross_attention": "mricore_vitb_cross_attention_v5",
       "mricore_vitb_mae_cross_attention": "mricore_vitb_mae_cross_attention_v5"}
STEP = {"convnext_base_25d_poolattn_v5": 3700}
PARENT = "convnext_base_25d_18slices_v5_aug_base"
SPECS = {f"convnext_base_25d_v5_spec_{g}": None for g in ("lateral", "pfoa", "fluid", "trauma")}
WS = [0, 1, 2, 3, 5]

# a per-finding weight plan, scored as one ensemble (findings not listed: specialist off)
PLAN = {"Lateral Meniscus": 1, "PF OA": 3, "Synovitis": 1, "Baker's": 3, "Fracture": 2, "Contusion": 2}

aud = pd.read_csv(RUNS.parent / "data" / "05_audit58.csv").set_index("StudyInstanceUID")
u = list(aud.index)
Y = pd.DataFrame((aud.loc[u, ["official_" + t for t in T]].values >= 0.5).astype(int), index=u, columns=T)


def best_step(run):
    v = pd.read_csv(RUNS / run / "results" / "val_log.csv")
    return int(v.loc[v.val_macro_auroc.idxmax(), "step"])


def load(run, split, step=None):
    d = RUNS / run / "results" / "predictions"
    f = next(d.glob(f"*_{split}_s{step}.npz")) if step is not None else Path(sw.best_npz(run, split))
    z = np.load(f, allow_pickle=True)
    cols = [str(t) for t in z["tasks"]] if "tasks" in z.files else T
    return pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=cols), f.name


old14 = [TEN_SPLIT[n] for n in TEN_SPLIT if n != "medimageinsight_davit_huge_336_fulldata"] + \
        [f"medimageinsight_davit_huge_336_v3_f{k}" for k in range(5)]
base = [B12.get(m, m) for m in old14]
assert len(base) == 14
P = {r: load(r, "audit", STEP.get(r))[0].loc[u] for r in base}
S, Sname = {}, {}
for r in SPECS:
    S[r], Sname[r] = load(r, "audit", best_step(r))
    S[r] = S[r].loc[u]
ens_sum = sum(P[r][T] for r in base)            # 58 x 12, sum of the 14 member logits


def auc(y, p):
    return roc_auc_score(y, p)


rng = np.random.default_rng(0)
print("files:", {r.replace("convnext_base_25d_v5_spec_", ""): Sname[r] for r in S})
print(f"\nbatch-2 fourteen (split analogues), audit-58 macro {np.mean([auc(Y[t], ens_sum[t]) for t in T]):.4f}\n")
gain_fixed = {}
for r in SPECS:
    for t in S[r].columns:
        y = Y[t].values
        cols = {w: (ens_sum[t].values + w * S[r][t].values) / (14 + w) for w in WS}
        cols["take"] = S[r][t].values
        sc = {w: auc(y, c) for w, c in cols.items()}
        # paired bootstrap, w=1 vs off
        d = []
        for _ in range(2000):
            i = rng.integers(0, 58, 58)
            if 0 < y[i].sum() < 58:
                d.append(auc(y[i], cols[1][i]) - auc(y[i], cols[0][i]))
        d = np.array(d)
        # cross-fitted choice of w among WS + takeover
        cf = []
        keys = list(cols)
        for _ in range(500):
            perm = rng.permutation(58)
            a, b = perm[:29], perm[29:]
            if not (0 < y[a].sum() < 29 and 0 < y[b].sum() < 29):
                continue
            pick = max(keys, key=lambda k: auc(y[a], cols[k][a]))
            cf.append(auc(y[b], cols[pick][b]) - auc(y[b], cols[0][b]))
        cf = np.array(cf)
        gain_fixed[t] = sc[1] - sc[0]
        line = "  ".join(f"w{w}={sc[w]:.4f}" for w in WS) + f"  take={sc['take']:.4f}"
        print(f"{t:17s} {line}")
        print(f"{'':17s} w1 vs off {sc[1] - sc[0]:+.4f} (90% [{np.percentile(d, 5):+.4f}, {np.percentile(d, 95):+.4f}],"
              f" better in {np.mean(d > 0):.0%}) | cross-fitted pick vs off {cf.mean():+.4f}, better in {np.mean(cf > 0):.0%}")
print(f"\nmacro over 12 with every specialist at w=1 on its findings: "
      f"{np.mean([auc(Y[t], ens_sum[t]) for t in T]) + sum(gain_fixed.values()) / 12:.4f} "
      f"({sum(gain_fixed.values()) / 12:+.4f})")

# specialist vs parent, alone, on the 346 clean v5 val studies
v5 = pd.read_csv(RUNS.parent / "data" / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
clean = list(v5.index[v5.source != "audit"])
par, _ = load(PARENT, "val", 5900)
print("\nalone on the 346 clean v5 val (no audit): parent -> specialist")
for r in SPECS:
    sp, _ = load(r, "val", best_step(r))
    for t in sp.columns:
        y = (v5.loc[clean, t].values > 0.5).astype(int)
        a, b = auc(y, par.loc[clean, t].values), auc(y, sp.loc[clean, t].values)
        print(f"  {t:17s} {a:.4f} -> {b:.4f} ({b - a:+.4f})")

spec_of = {t: r for r in SPECS for t in S[r].columns}
plan_cols = {t: ((ens_sum[t].values + PLAN[t] * S[spec_of[t]][t].values) / (14 + PLAN[t]) if t in PLAN else ens_sum[t].values)
             for t in T}
m0 = np.mean([auc(Y[t], ens_sum[t]) for t in T]); m1 = np.mean([auc(Y[t], plan_cols[t]) for t in T])
d = []
for _ in range(2000):
    i = rng.integers(0, 58, 58)
    try:
        d.append(np.mean([auc(Y[t].values[i], plan_cols[t][i]) - auc(Y[t].values[i], ens_sum[t].values[i]) for t in T]))
    except ValueError:
        pass
print(f"\nPLAN {PLAN}: audit-58 macro {m0:.4f} -> {m1:.4f} ({m1 - m0:+.4f}), better in {np.mean(np.array(d) > 0):.0%}")
