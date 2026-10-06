"""fit_more_members_test.py - which extra members are worth adding to fit-10? (6 Oct)

Base = fit-10 in scaffold analogues (old_label_members_ensemble.py): convnext_base_v5, dinov3_convnext_base_v5,
convnext_base_25d_18slices_v5_aug_base, convnext_base_25d_poolattn_v5 (split twin of the shipped _full), MII f0 + f4,
mricore_vitb_cross_attention_v5, swinv2_base_in22k_v5_warm (the per-finding member is not in the scaffold), plus the
0.30 mm joint zoom on the 5 joint findings (zoom_ensemble_test.py, step 1500). Candidates added on all 12 findings:
MII f1 / f2 / f3 (singly, all three, and all five folds as ONE averaged vote), MRI-CORE MAE, MII-480.
Paired bootstrap on audit-58 and clean-121.

    python runs/others/fit_more_members_test.py
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE / "zoom_ensemble_test.py").read_text(encoding="utf-8")
g = {"__file__": str(HERE / "zoom_ensemble_test.py"), "__name__": "zet"}
exec(compile(src[:src.index("def variants(base):")], "zoom_ensemble_test.py", "exec"), g)
T, data, blend, auc_cols, full_auc, JM = g["T"], g["data"], g["blend"], g["auc_cols"], g["full_auc"], g["JM"]
MII = [f"medimageinsight_davit_huge_336_v3_f{k}" for k in range(5)]
BASE = ["convnext_base_v5", "dinov3_convnext_base_v5", "convnext_base_25d_18slices_v5_aug_base",
        "convnext_base_25d_poolattn_v5", MII[0], MII[4], "mricore_vitb_cross_attention_v5", "swinv2_base_in22k_v5_warm"]
ZJ = [("z030", 1, JM.astype(float))]
for s, (P, y, I) in data.items():
    P["mii_mean5"] = np.mean([P[m] for m in MII], 0)


def score(members, extra):
    r = {}
    for s, (P, y, I) in data.items():
        p = blend(P, members, extra)
        r[s] = (full_auc(y, p), auc_cols(y, p, I))
    return r


ref = score(BASE, ZJ)
nz = score(BASE, [])
print(f"fit-10 analogue: audit {ref['audit58'][0]:.4f} | clean {ref['clean121'][0]:.4f}   "
      f"(without the zoom: {nz['audit58'][0]:.4f} | {nz['clean121'][0]:.4f})")
cands = {
    "+ MII f1": (BASE + [MII[1]], ZJ), "+ MII f2": (BASE + [MII[2]], ZJ), "+ MII f3": (BASE + [MII[3]], ZJ),
    "+ MII f1 f2 f3 (all 5 flat)": (BASE + MII[1:4], ZJ),
    "MII f0+f4 -> 5-fold mean, 1 vote": ([m for m in BASE if m not in MII] + ["mii_mean5"], ZJ),
    "MII f0+f4 -> 5-fold mean, 2 votes": ([m for m in BASE if m not in MII] + ["mii_mean5"],
                                          ZJ + [("mii_mean5", 1, np.ones(12))]),
    "+ MRI-CORE MAE": (BASE + ["mricore_vitb_mae_cross_attention_v5"], ZJ),
    "+ MII-480": (BASE + ["medimageinsight_davit_huge_perslice_480_warm"], ZJ),
    "+ MAE + MII-480": (BASE + ["mricore_vitb_mae_cross_attention_v5", "medimageinsight_davit_huge_perslice_480_warm"], ZJ),
    "+ MII f1-f3 + MAE + MII-480 (= all 14 + zoom)": (BASE + MII[1:4] + ["mricore_vitb_mae_cross_attention_v5",
                                                                        "medimageinsight_davit_huge_perslice_480_warm"], ZJ),
}
for name, (mem, extra) in cands.items():
    r = score(mem, extra)
    out = []
    for s in ("audit58", "clean121"):
        d = r[s][1] - ref[s][1]
        out.append(f"{r[s][0]:.4f} ({r[s][0] - ref[s][0]:+.4f}, {(d > 0).mean():4.0%})")
    print(f"  {name:45s} audit {out[0]} | clean {out[1]}")
