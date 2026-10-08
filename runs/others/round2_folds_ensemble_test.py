"""round2_folds_ensemble_test.py - the 5 round-2 ConvNeXt folds (v4 labels, flip) as ONE ensemble member (8 Oct).

Member "r2" = 5-fold mean logit, each fold at its best v3-graded val step (round2_teacher_audit.best_npz).
  audit-58   5-fold mean (the 58 are in no fold, so every fold is blind to them): what a 5-fold member ships.
  clean-121  audit part as above; the 63 v5-val studies each get the ONE fold that held it out (out-of-fold), since
             the other 4 trained on it. That is a single-fold estimate there, slightly pessimistic for a 5-fold mean.
Bases: the batch-3 scaffold fourteen and the S6 core of the 0.943 submission (zoom_ensemble_test.py's).
Modes: add (all 12), swapP (replaces convnext_base_25d_18slices_v5_aug_base), add w2.

    python runs/others/round2_folds_ensemble_test.py
"""
from pathlib import Path

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
PARENT = "convnext_base_25d_18slices_v5_aug_base"
S6 = ["convnext_base_v5", "dinov3_convnext_base_v5", PARENT, "convnext_base_25d_poolattn_v5",
      "convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip", "medimageinsight_davit_huge_336_v3_f0"]
rng = np.random.default_rng(0)


def npz_df(f):
    z = np.load(f, allow_pickle=True)
    return pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T)


aud = [npz_df(best_npz(f"convnext_base_25d_18slices_v4_flip_f{k}", "audit")) for k in range(5)]
aud_mean = sum(aud) / 5
oof = pd.concat([npz_df(best_npz(f"convnext_base_25d_18slices_v4_flip_f{k}", "val")) for k in range(5)])
assert oof.index.is_unique


def r2(ids):
    out = []
    for s in ids:
        out.append(aud_mean.loc[s].values if s in aud_mean.index else oof.loc[s].values)
    return pd.DataFrame(np.array(out), index=ids, columns=T)


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
    P = {m: g["get"](m, ids).loc[ids, T].astype(float).to_numpy() for m in BASE}
    P["r2"] = r2(ids).loc[ids, T].to_numpy()
    y = Y.loc[ids, T].to_numpy().astype(int)
    data[s] = (P, y, rng.integers(0, len(ids), (2000, len(ids))))
    n_oof = sum(i not in aud_mean.index for i in ids)
    print(f"{s}: {len(ids)} studies, r2 solo {full_auc(y, P['r2']):.4f} ({n_oof} out-of-fold single-fold rows)"
          f" | parent solo {full_auc(y, P[PARENT]):.4f}")


def blend(P, members, w_r2=0.0):
    num = sum(P[m] for m in members) + w_r2 * P["r2"]
    return num / (len(members) + w_r2)


for bname, base in (("all 14", BASE), ("S6 (0.943 core)", [m for m in S6 if m in BASE])):
    print(f"\n=== base: {bname} ({len(base)} members)")
    ref = {s: (full_auc(y, blend(P, base)), auc_cols(y, blend(P, base), I)) for s, (P, y, I) in data.items()}
    print(f"  base: audit {ref['audit58'][0]:.4f} | clean-121 {ref['clean121'][0]:.4f}")
    nb = [m for m in base if m != PARENT]
    for name, (mem, w) in {"r2 add": (base, 1), "r2 add w2": (base, 2), "r2 add w3": (base, 3),
                           "r2 swapP": (nb, 1), "r2 swapP w2": (nb, 2)}.items():
        r = []
        for s, (P, y, I) in data.items():
            p = blend(P, mem, w)
            d = auc_cols(y, p, I) - ref[s][1]
            r.append((full_auc(y, p), full_auc(y, p) - ref[s][0], (d > 0).mean()))
        (a, da, pa), (c, dc, pc) = r
        flag = "  <- both up" if da > 0 and dc > 0 else ""
        print(f"  {name:14s} audit {a:.4f} ({da:+.4f}, {pa:4.0%}) | clean {c:.4f} ({dc:+.4f}, {pc:4.0%}){flag}")
