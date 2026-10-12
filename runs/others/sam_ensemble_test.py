"""sam_ensemble_test.py - convnext_base_25d_18slices_v5_sam (SAM optimiser twin of aug_base) solo and in the ensemble (8 Oct).

Member "r2" = 5-fold mean logit, each fold at its best v3-graded val step (round2_teacher_audit.best_npz).
  audit-58   5-fold mean (the 58 are in no fold, so every fold is blind to them): what a 5-fold member ships.
  clean-121  audit part as above; the 63 v5-val studies each get the ONE fold that held it out (out-of-fold), since
             the other 4 trained on it. That is a single-fold estimate there, slightly pessimistic for a 5-fold mean.
Bases: the batch-3 scaffold fourteen and the S6 core of the 0.943 submission (zoom_ensemble_test.py's).
Modes: add (all 12), swapP (replaces convnext_base_25d_18slices_v5_aug_base), add w2.

    python runs/others/sam_ensemble_test.py [new_run] [member_swapP_replaces]
"""
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
src = (HERE / "old_label_members_ensemble.py").read_text(encoding="utf-8")
g = {"__file__": str(HERE / "old_label_members_ensemble.py"), "__name__": "olm_scaffold"}
exec(compile(src[:src.index("pool = sorted(")], "old_label_members_ensemble.py", "exec"), g)
T, BASE, SURF, R = g["T"], g["BASE"], g["SURF"], g["R"]
src2 = (HERE / "round2_teacher_audit.py").read_text(encoding="utf-8")
g2 = {"__file__": str(HERE / "round2_teacher_audit.py"), "__name__": "r2_audit"}
exec(compile(src2[:src2.index("\ndef logit(p):")], "round2_teacher_audit.py", "exec"), g2)
best_npz = g2["best_npz"]
PARENT = sys.argv[2] if len(sys.argv) > 2 else "convnext_base_25d_18slices_v5_aug_base"   # the member swapP replaces
RAPTOR = "convnext_base_v5_raptor96_local"
S6 = ["convnext_base_v5", "dinov3_convnext_base_v5", "convnext_base_25d_18slices_v5_aug_base", "convnext_base_25d_poolattn_v5",
      "convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip", "medimageinsight_davit_huge_336_v3_f0"]
rng = np.random.default_rng(0)



NEW = sys.argv[1] if len(sys.argv) > 1 else "convnext_base_25d_18slices_v5_sam"


def auc_cols(y, p, I):
    out = np.zeros(len(I)); cnt = np.zeros(len(I))
    for j in range(y.shape[1]):
        yy, pp = y[I, j], p[I, j]
        n1 = yy.sum(1); n0 = yy.shape[1] - n1
        ok = (n1 > 0) & (n0 > 0)
        r = rankdata(pp, axis=1)
        a = ((r * yy).sum(1) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1)
        out += np.where(ok, a, 0); cnt += ok
    return out / cnt


def full_auc(y, p):
    return auc_cols(y, p, np.arange(len(y))[None])[0]


data = {}
for s, (ids, Y) in SURF.items():
    P = {m: g["get"](m, ids).loc[ids, T].astype(float).to_numpy() for m in dict.fromkeys(BASE + [NEW, PARENT, RAPTOR])}
    y = Y.loc[ids, T].to_numpy().astype(int)
    I = rng.integers(0, len(ids), (2000, len(ids)))
    data[s] = (P, y, I)
    d = auc_cols(y, P[NEW], I) - auc_cols(y, P[PARENT], I)
    print(f"{s}: SAM solo {full_auc(y, P[NEW]):.4f} vs parent {full_auc(y, P[PARENT]):.4f} "
          f"({full_auc(y, P[NEW]) - full_auc(y, P[PARENT]):+.4f}, SAM better in {(d > 0).mean():.0%})")


def blend(P, members):
    return sum(P[m] for m in members) / len(members)


# S6R = the combined_raptor submission's members (raptor96_local in for aug_base)
S6R = [RAPTOR if m == "convnext_base_25d_18slices_v5_aug_base" else m for m in S6 if m in BASE]   # perfinding_fulldata has no held-out preds
for bname, base in (("all 14", BASE), ("S6 (0.943 core)", [m for m in S6 if m in BASE]), ("S6R (combined_raptor)", S6R)):
    print(f"\n=== base: {bname} ({len(base)} members)")
    ref = {s: (full_auc(y, blend(P, base)), auc_cols(y, blend(P, base), I)) for s, (P, y, I) in data.items()}
    print(f"  base: audit {ref['audit58'][0]:.4f} | clean-121 {ref['clean121'][0]:.4f}")
    nb = [m for m in base if m != PARENT]
    for name, mem in {"SAM add": base + [NEW], "SAM swapP": nb + [NEW]}.items():
        r = []
        for s, (P, y, I) in data.items():
            p = blend(P, mem)
            d = auc_cols(y, p, I) - ref[s][1]
            r.append((full_auc(y, p), full_auc(y, p) - ref[s][0], (d > 0).mean()))
        (a, da, pa), (c, dc, pc) = r
        flag = "  <- both up" if da > 0 and dc > 0 else ""
        print(f"  {name:10s} audit {a:.4f} ({da:+.4f}, {pa:4.0%}) | clean {c:.4f} ({dc:+.4f}, {pc:4.0%}){flag}")
