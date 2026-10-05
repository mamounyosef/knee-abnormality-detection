"""ensemble_time_budget_subsets.py - which subset of the batch-2 fourteen keeps the most score inside a GPU-time budget?
(the combined notebook must fit our ensemble AND the public pipeline in 9 h)

Members: the batch-3 scaffold base (batch-2 split analogues, SwinV2 v5), flat logit mean, as in
old_label_members_ensemble.py. Cost: each member's T4 GPU seconds per study from the combined stress run's timing.json
(runs/others/ensembling/public_side/stress_out/timing.json), matched by architecture. Every subset of the 14 is scored
(macro AUROC, audit-58 and clean-121); per budget the best subset by the MEAN of the two surfaces is printed with both.

    python runs/others/ensemble_time_budget_subsets.py
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
src = (HERE / "old_label_members_ensemble.py").read_text(encoding="utf-8")
g = {"__file__": str(HERE / "old_label_members_ensemble.py"), "__name__": "olm_scaffold"}
exec(compile(src[:src.index("pool = sorted(")], "old_label_members_ensemble.py", "exec"), g)
T, BASE, SURF, get = g["T"], g["BASE"], g["SURF"], g["get"]
tim = json.loads((HERE / "ensembling" / "public_side" / "stress_out" / "timing.json").read_text(encoding="utf-8"))
gpu = {m["name"]: m["gpu_s_per_study"] for m in tim["members"]}


def cost(name):
    # the scaffold uses split analogues; the bundle ships these names
    for key, shipped in (("perslice_480", "medimageinsight_davit_huge_perslice_480_warm"), ("swinv2", "swinv2_base_in22k"),
                         ("mae_cross", "mricore_vitb_mae_cross_attention_v5"), ("cross_attention", "mricore_vitb_cross_attention_v5"),
                         ("dinov3", "dinov3_convnext_base_v5"), ("perfinding", "convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip"),
                         ("poolattn", "convnext_base_25d_poolattn_v5"), ("25d", "convnext_base_25d_18slices_v5_aug_base"),
                         ("336_v3_f", "medimageinsight_davit_huge_336_v3_f0"), ("convnext_base", "convnext_base_v5")):
        if key in name:
            return gpu[shipped]
    raise KeyError(name)


C = np.array([cost(m) for m in BASE])
print("members and GPU s/study:")
for m, c in zip(BASE, C):
    print(f"  {c:5.2f}  {m}")
print(f"  total {C.sum():.2f} s/study (stress wall was {18.9:.1f} s/study incl. decode, both cards)")

# per surface: stacked logits (members x studies x tasks), vectorised rank AUROC per subset
data = {}
for s, (ids, Y) in SURF.items():
    L = np.stack([get(m, ids).loc[ids, T].astype(float).to_numpy() for m in BASE])
    data[s] = (L, Y.loc[ids, T].to_numpy().astype(int))


def macro(L, y, idx):
    p = L[list(idx)].mean(0)
    out = 0.0
    for j in range(len(T)):
        r = rankdata(p[:, j]); pos = y[:, j] == 1; n1 = pos.sum(); n0 = len(pos) - n1
        out += (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out / len(T)


full = tuple(range(len(BASE)))
ref = {s: macro(L, y, full) for s, (L, y) in data.items()}
print(f"\nall 14: audit {ref['audit58']:.4f} | clean-121 {ref['clean121']:.4f}")
rows = []
for k in range(3, len(BASE) + 1):
    for idx in itertools.combinations(range(len(BASE)), k):
        sc = {s: macro(L, y, idx) for s, (L, y) in data.items()}
        rows.append((C[list(idx)].sum(), sc["audit58"], sc["clean121"], idx))
print(f"{len(rows)} subsets scored")
for budget in (6, 8, 10, 12, 14, 16, 18):
    ok = [r for r in rows if r[0] <= budget]
    best = max(ok, key=lambda r: r[1] + r[2])
    print(f"\nGPU <= {budget:2d} s/study: best {best[0]:.1f} s  audit {best[1]:.4f} ({best[1] - ref['audit58']:+.4f})"
          f"  clean-121 {best[2]:.4f} ({best[2] - ref['clean121']:+.4f})  {len(best[3])} members")
    print("    " + ", ".join(BASE[i] for i in best[3]))
    out = [BASE[i] for i in range(len(BASE)) if i not in best[3]]
    print("    dropped: " + ", ".join(out))
