"""ensemble_candidates_v5.py - every recent candidate member tried inside the ensemble, several ways, on two surfaces.

SURFACES
  audit58   the batch-2 fourteen (ref 56824782) as split-data analogues, plus SwinV2 v5 swapped in (the planned batch-3 base);
            the only surface with panel labels, but 58 studies (+-0.03 per finding).
  val346    the 346 clean v5 val studies with the SIX v5-split members (convnext_base_v5, dinov3_convnext_base_v5,
            convnext_base_25d_18slices_v5_aug_base, convnext_base_25d_poolattn_v5 @3700, mricore_vitb_cross_attention_v5,
            mricore_vitb_mae_cross_attention_v5) as a proxy; 6x more studies, v3 labels. Candidates on the v5 split are clean
            here (their early stopping used this val: slightly optimistic).
WAYS (12-finding candidates: attributes, duplicate_series, compartment, native_res_420px)
  add       the candidate as one more member
  replace   the candidate in place of its twin convnext_base_25d_18slices_v5_aug_base
  nest      candidate and twin averaged first, then one vote (the family keeps one vote)
WAYS (specialists: votes only on their own findings) at weight 1, 2, 3
Every row: macro AUROC, delta vs the base, paired bootstrap share of resamples where it is better (2,000 on audit, 1,000 on val).

    python runs/others/ensemble_candidates_v5.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ensembling"))
import sweep_new_runs as sw  # noqa: E402
from ten_mii_folds_audit import TEN_SPLIT  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
TWIN = "convnext_base_25d_18slices_v5_aug_base"
FULL = ["convnext_base_25d_v5_attributes", "convnext_base_25d_v5_duplicate_series", "convnext_base_25d_v5_compartment",
        "convnext_base_25d_v5_native_res_420px"]
SPECS = ["convnext_base_25d_v5_spec_lateral", "convnext_base_25d_v5_spec_lateral_half", "convnext_base_25d_v5_spec_pfoa",
         "convnext_base_25d_v5_spec_fluid", "convnext_base_25d_v5_spec_trauma"]
STEP = {"convnext_base_25d_poolattn_v5": 3700, "swinv2_base_in22k_v5_warm": 2000, TWIN: 5900}
V5_SIX = ["convnext_base_v5", "dinov3_convnext_base_v5", TWIN, "convnext_base_25d_poolattn_v5",
          "mricore_vitb_cross_attention_v5", "mricore_vitb_mae_cross_attention_v5"]


def best_step(run):
    v = pd.read_csv(R / run / "results" / "val_log.csv")
    return int(v.loc[v.val_macro_auroc.idxmax(), "step"])


def load(run, split):
    d = R / run / "results" / "predictions"
    s = STEP.get(run)
    if s is None and (R / run / "results" / "val_log.csv").exists() and run.endswith(("_v5", "attributes", "series",
                                                                                      "compartment", "half", "lateral", "420px",
                                                                                      "pfoa", "fluid", "trauma", "warm")):
        s = best_step(run)
    f = next(d.glob(f"*_{split}_s{s}.npz")) if s is not None else Path(sw.best_npz(run, split))
    z = np.load(f, allow_pickle=True)
    cols = [str(t) for t in z["tasks"]] if "tasks" in z.files else T
    return pd.DataFrame(z["logits"], index=[str(x) for x in z["study_ids"]], columns=cols)


aud = pd.read_csv(R.parent / "data" / "05_audit58.csv").set_index("StudyInstanceUID")
v5 = pd.read_csv(R.parent / "data" / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
SURF = {"audit58": (list(aud.index), pd.DataFrame((aud[["official_" + t for t in T]].values >= .5).astype(int),
                                                   index=aud.index, columns=T), 2000),
        "val346": (list(v5.index[v5.source != "audit"]), (v5[T] > .5).astype(int), 1000)}
B2 = {"dinov3_convnext_base": "dinov3_convnext_base_v5", "convnext_base_relabel": "convnext_base_v5",
      "convnext_base_25d_all_knees_same_orientation": TWIN, "convnext_base_25d_poolattn": "convnext_base_25d_poolattn_v5",
      "mricore_vitb_cross_attention": "mricore_vitb_cross_attention_v5",
      "mricore_vitb_mae_cross_attention": "mricore_vitb_mae_cross_attention_v5",
      "swinv2_base_in22k": "swinv2_base_in22k_v5_warm"}
BASE = {"audit58": [B2.get(TEN_SPLIT[n], TEN_SPLIT[n]) for n in TEN_SPLIT if n != "medimageinsight_davit_huge_336_fulldata"]
        + [f"medimageinsight_davit_huge_336_v3_f{k}" for k in range(5)], "val346": V5_SIX}
rng = np.random.default_rng(0)


def macro(Y, P, i=None):
    return np.mean([roc_auc_score(Y[t].values[i] if i is not None else Y[t].values,
                                  P[t].values[i] if i is not None else P[t].values) for t in T])


def boot_macro(Y, P, I):
    """Macro AUROC for every bootstrap resample at once (rows of the index matrix I): Mann-Whitney from ranks,
    ties averaged, identical to roc_auc_score. Fast: one rank pass per finding over all resamples."""
    out = np.zeros(len(I))
    for t in T:
        y, s = Y[t].values[I], P[t].values[I]           # (B, n)
        r = rankdata(s, axis=1)
        n1 = y.sum(1)
        out += ((r * y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * (y.shape[1] - n1))
    return out / len(T)


for surf, (ids, Y, nb) in SURF.items():
    Y = Y.loc[ids]
    members = BASE[surf]
    pred = {m: load(m, "audit" if surf == "audit58" else "val").loc[ids, T] for m in members + FULL}
    spec = {m: load(m, "audit" if surf == "audit58" else "val").loc[ids] for m in SPECS}
    base = sum(pred[m] for m in members) / len(members)
    b0 = macro(Y, base)
    boots = [rng.integers(0, len(ids), len(ids)) for _ in range(nb)]
    boots = [i for i in boots if all(0 < Y[t].values[i].sum() < len(i) for t in T)]
    I = np.array(boots)
    bb = boot_macro(Y, base, I)
    print(f"\n=== {surf}: base of {len(members)} members, macro {b0:.4f}")

    def report(name, P):
        m = macro(Y, P)
        d = boot_macro(Y, P, I) - bb
        print(f"  {name:62s} {m:.4f} ({m - b0:+.4f})  better in {np.mean(d > 0):4.0%}")

    for c in FULL:
        report(f"add      {c}", (base * len(members) + pred[c]) / (len(members) + 1))
        if TWIN in members:
            report(f"replace  twin by {c}", base + (pred[c] - pred[TWIN]) / len(members))
            report(f"nest     twin+{c} as one vote", base + ((pred[c] + pred[TWIN]) / 2 - pred[TWIN]) / len(members))
    report("add      attributes + duplicate_series", (base * len(members) + pred[FULL[0]] + pred[FULL[1]]) / (len(members) + 2))
    if TWIN in members:
        report("nest     twin+attributes+duplicate_series as one vote",
               base + ((pred[TWIN] + pred[FULL[0]] + pred[FULL[1]]) / 3 - pred[TWIN]) / len(members))
    for s in SPECS:
        for w in (1, 2, 3):
            P = base.copy()
            for t in spec[s].columns:
                P[t] = (base[t] * len(members) + w * spec[s][t]) / (len(members) + w)
            report(f"spec     {s.replace('convnext_base_25d_v5_', '')} w{w} on {len(spec[s].columns)} finding(s)", P)


# the combined plan: the specialists that were positive on both surfaces (or positive and not contradicted)
PLAN = {"convnext_base_25d_v5_spec_trauma": 1, "convnext_base_25d_v5_spec_pfoa": 2, "convnext_base_25d_v5_spec_fluid": 1}
for surf, (ids, Y, nb) in SURF.items():
    Y = Y.loc[ids]
    members = BASE[surf]
    split = "audit" if surf == "audit58" else "val"
    base = sum(load(m, split).loc[ids, T] for m in members) / len(members)
    P = base.copy()
    for s, w in PLAN.items():
        sp = load(s, split).loc[ids]
        for t in sp.columns:
            P[t] = (base[t] * len(members) + w * sp[t]) / (len(members) + w)
    I = np.array([i for i in (rng.integers(0, len(ids), len(ids)) for _ in range(nb))
                  if all(0 < Y[t].values[i].sum() < len(i) for t in T)])
    d = boot_macro(Y, P, I) - boot_macro(Y, base, I)
    print(f"PLAN on {surf}: {macro(Y, base):.4f} -> {macro(Y, P):.4f} ({macro(Y, P) - macro(Y, base):+.4f}), better in {np.mean(d > 0):.0%}")
