"""perfinding_v5_swap_121.py (scaffold from old_label_members_ensemble.py): the per-finding v5 swap on audit-58 AND clean-121.
Original doc: old_label_members_ensemble.py - do members trained on the OLD labels (Qwen report labels only: no MedGemma, no teacher)
add the diversity the relabelled ensemble lost? (user, 4 Oct)

Base: the planned batch-3 base = the batch-2 fourteen (split analogues) with SwinV2 v5 (relabelled members correlate with the
MedImageInsight teacher at 0.93-0.95, runs/others/member_correlation_old_vs_new.py).
Pool: every 12-finding member trained on 05_train_qwen24t-med.csv (old split, old labels), not already in the base.
For each candidate, added at weight 1 two ways:
  all12   on all 12 findings
  noMG    only on the 7 findings MedGemma did NOT touch (ACL, MCL, Medial Meniscus, Medial OA, PF OA, Contusion, Fracture);
          excluded on Effusion, Synovitis, Baker's, Lateral Meniscus, Lateral OA, where the new labels carry MedGemma
Surfaces (both unseen by every member involved):
  audit58   panel labels
  clean121  the 58 audit + the 63 v5-val studies outside the old training split (v3 labels for those 63);
            the old members early-stopped on the old val (which holds the 63) and the v5 members on the v5 val: both
            slightly optimistic, equally
Then the candidates positive on both surfaces are added together (each in its better mode).

    python runs/others/old_label_members_ensemble.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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
MG = ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]
NOMG = [t for t in T if t not in MG]
STEP = {"convnext_base_25d_poolattn_v5": 3700, "swinv2_base_in22k_v5_warm": 2000, "convnext_base_25d_perfinding_v5": 3000}
B3 = {"dinov3_convnext_base": "dinov3_convnext_base_v5", "convnext_base_relabel": "convnext_base_v5",
      "convnext_base_25d_all_knees_same_orientation": "convnext_base_25d_18slices_v5_aug_base",
      "convnext_base_25d_poolattn": "convnext_base_25d_poolattn_v5",
      "mricore_vitb_cross_attention": "mricore_vitb_cross_attention_v5",
      "mricore_vitb_mae_cross_attention": "mricore_vitb_mae_cross_attention_v5", "swinv2_base_in22k": "swinv2_base_in22k_v5_warm"}
BASE = [B3.get(TEN_SPLIT[n], TEN_SPLIT[n]) for n in TEN_SPLIT if n != "medimageinsight_davit_huge_336_fulldata"] + \
       [f"medimageinsight_davit_huge_336_v3_f{k}" for k in range(5)]


def load(run, split):
    d = R / run / "results" / "predictions"
    if run in STEP:
        f = next(d.glob(f"*_{split}_s{STEP[run]}.npz"))
    else:
        f = sw.best_npz(run, split)
        if f is None:
            return None
    z = np.load(f, allow_pickle=True)
    cols = [str(t) for t in z["tasks"]] if "tasks" in z.files else None
    if cols is not None and cols != T:
        return None                                          # partial-head branch: not a 12-finding member
    return pd.DataFrame(z["logits"], index=[str(x) for x in z["study_ids"]], columns=T)


def train_csv(run):
    if not (R / run / "config.yaml").exists():
        return ""
    c = yaml.safe_load((R / run / "config.yaml").read_text(encoding="utf-8")) or {}
    return str((c.get("paths") or {}).get("train_csv", ""))


aud = pd.read_csv(R.parent / "data" / "05_audit58.csv").set_index("StudyInstanceUID")
v5 = pd.read_csv(R.parent / "data" / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
old_train = set(pd.read_csv(R.parent / "data" / "05_train_qwen24t-med.csv").StudyInstanceUID)
mii_train = set().union(*[set(pd.read_csv(R.parent / "data" / f"21_fold{k}_train.csv").StudyInstanceUID) for k in range(5)])
c63 = [s for s in v5.index if v5.loc[s, "source"] != "audit" and s not in old_train]
# the MII folds trained on 4/5 of everything, so for the 63 each fold member is replaced by the ONE fold that did not train
# on that study (its out-of-fold prediction), the stand-in the v4 labels were built with
FOLD_OF = {s: k for k in range(5) for s in pd.read_csv(R.parent / "data" / f"21_fold{k}_val.csv").StudyInstanceUID}
assert all(s in FOLD_OF for s in c63 if s in mii_train), "a training study of the MII folds is in no fold's val"
Yaud = pd.DataFrame((aud[["official_" + t for t in T]].values >= .5).astype(int), index=aud.index, columns=T)
Y121 = pd.concat([Yaud, (v5.loc[c63, T] > .5).astype(int)])


def get(run, ids):
    """Predictions on these ids: audit npz for the 58, val npz for the 63 (both the member's best step)."""
    if run.startswith("medimageinsight_davit_huge_336_v3_f"):
        a = load(run, "audit")
        rest = [s for s in ids if s not in a.index]
        oof = {k: load(f"medimageinsight_davit_huge_336_v3_f{k}", "val") for k in {FOLD_OF[s] for s in rest}}
        extra = pd.DataFrame([oof[FOLD_OF[s]].loc[s].values for s in rest], index=rest, columns=T)
        return pd.concat([a, extra]).loc[ids]
    a = load(run, "audit")
    if a is None:
        return None
    parts = [a]
    rest = [s for s in ids if s not in a.index]
    if rest:
        v = load(run, "val")
        if v is None or not set(rest) <= set(v.index):
            return None
        parts.append(v.loc[rest])
    return pd.concat(parts).loc[ids]


SURF = {"audit58": (list(aud.index), Yaud.loc[list(aud.index)]), "clean121": (list(Y121.index), Y121)}

rng = np.random.default_rng(0)
for surf, (ids, Y) in SURF.items():
    I = np.array([i for i in (rng.integers(0, len(ids), len(ids)) for _ in range(3000)) if all(0 < Y[t].values[i].sum() < len(i) for t in T)])
    def bmac(df):
        out = np.zeros(len(I))
        for t in T:
            y, s = Y[t].values[I], df[t].values[I]
            r = rankdata(s, axis=1); n1 = y.sum(1)
            out += ((r * y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * (y.shape[1] - n1))
        return out / len(T)
    mac = lambda df: np.mean([roc_auc_score(Y[t], df[t]) for t in T])
    P = {m: get(m, ids) for m in BASE + ["convnext_base_25d_perfinding_v5"]}
    base = sum(P[m] for m in BASE) / len(BASE)
    old = "convnext_base_25d_perfinding_slots_and_pooling"
    swp = base + (P["convnext_base_25d_perfinding_v5"] - P[old]) / len(BASE)
    add = (base * len(BASE) + P["convnext_base_25d_perfinding_v5"]) / (len(BASE) + 1)
    for name, df in (("swap old -> v5", swp), ("add v5 (keep old)", add)):
        print(f"{surf}: base {mac(base):.4f} | {name}: {mac(df):.4f} ({mac(df) - mac(base):+.4f}), better in {np.mean(bmac(df) - bmac(base) > 0):.0%}")
