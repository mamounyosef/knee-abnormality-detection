"""ensemble_named_subsets.py - score a few PRINCIPLED cuts of the batch-2 fourteen (chosen by cost, not searched), with a
paired bootstrap against all 14. Companion of ensemble_time_budget_subsets.py, whose best-of-16k picks are selection noise.

    python runs/others/ensemble_named_subsets.py
"""
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
src = (HERE / "ensemble_time_budget_subsets.py").read_text(encoding="utf-8")
g = {"__file__": str(HERE / "ensemble_time_budget_subsets.py"), "__name__": "etb"}
exec(compile(src[:src.index("full = tuple(")], "etb", "exec"), g)
T, BASE, C, data = g["T"], g["BASE"], g["C"], g["data"]
DROP = {
    "drop SwinV2 + MII-480 (the two 5 s members)": ["swinv2", "perslice_480"],
    "drop SwinV2 + MII-480 + both MRI-CORE": ["swinv2", "perslice_480", "mricore"],
    "drop SwinV2 + MII-480 + 3 MII folds": ["swinv2", "perslice_480", "336_v3_f2", "336_v3_f3", "336_v3_f4"],
    "drop SwinV2 + MII-480 + convnext_base_v5": ["swinv2", "perslice_480", "convnext_base_v5"],
    "A: drop MII-480 + MII f2-f4 + MRI-CORE MAE (keep SwinV2)": ["perslice_480", "336_v3_f2", "336_v3_f3", "336_v3_f4", "mae_cross"],
    "B: A + drop MII f1": ["perslice_480", "336_v3_f1", "336_v3_f2", "336_v3_f3", "336_v3_f4", "mae_cross"],
    "D: drop MII-480 + MII f2-f4 (keep SwinV2, MAE)": ["perslice_480", "336_v3_f2", "336_v3_f3", "336_v3_f4"],
    "E: drop MII f1-f4 + MRI-CORE MAE (keep SwinV2, MII-480)": ["336_v3_f1", "336_v3_f2", "336_v3_f3", "336_v3_f4", "mae_cross"],
    "drop MII-480 only": ["perslice_480"],
    "drop SwinV2 only": ["swinv2"],
}
rng = np.random.default_rng(0)


def boot(L, y, idx, I):
    p = L[idx].mean(0)
    out = np.zeros(len(I))
    for j in range(len(T)):
        yy, s = y[I, j], p[I, j]
        r = rankdata(s, axis=1); n1 = yy.sum(1)
        out += ((r * yy).sum(1) - n1 * (n1 + 1) / 2) / (n1 * (yy.shape[1] - n1))
    return out / len(T)


IB = {}
for s, (L, y) in data.items():
    n = len(y)
    IB[s] = np.array([i for i in (rng.integers(0, n, n) for _ in range(3000)) if all(0 < y[i, j].sum() < n for j in range(len(T)))])
full = list(range(len(BASE)))
base = {s: (boot(L, y, full, IB[s]), g["macro"](L, y, full)) for s, (L, y) in data.items()}
print(f"all 14: GPU {C.sum():.1f} s/study | audit {base['audit58'][1]:.4f} | clean-121 {base['clean121'][1]:.4f}")
for name, keys in DROP.items():
    idx = [i for i, m in enumerate(BASE) if not any(k in m for k in keys)]
    parts = []
    for s, (L, y) in data.items():
        v = g["macro"](L, y, idx); b = boot(L, y, idx, IB[s])
        parts.append(f"{s} {v:.4f} ({v - base[s][1]:+.4f}, better in {np.mean(b - base[s][0] > 0):.0%})")
    print(f"{name}: {len(idx)} members, GPU {C[idx].sum():.1f} s/study | " + " | ".join(parts))
