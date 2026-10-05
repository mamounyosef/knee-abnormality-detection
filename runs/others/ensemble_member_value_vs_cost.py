"""ensemble_member_value_vs_cost.py - which batch-2 members are worth their GPU time? (user, 5 Oct: choose what to drop
better than "the most expensive")

1. Leave-one-out: macro AUROC of the 14 without each member, on audit-58 and clean-121 (paired bootstrap), next to its
   T4 GPU seconds per study (combined stress timing.json).
2. Greedy backward elimination: repeatedly drop the member whose removal costs the least score (mean of the two surfaces)
   per GPU second saved, printing every step with both surfaces and the running cost, down to ~6 s/study.
Budget context: public side 4.37 h measured, our safety limit 8.0 h -> ours may use ~3.4 h = ~12.9 s/study wall, about
15 GPU s/study summed over both cards (stress ratio wall/GPU-sum 0.86).

    python runs/others/ensemble_member_value_vs_cost.py
"""
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
src = (HERE / "ensemble_time_budget_subsets.py").read_text(encoding="utf-8")
g = {"__file__": str(HERE / "ensemble_time_budget_subsets.py"), "__name__": "etb"}
exec(compile(src[:src.index("full = tuple(")], "etb", "exec"), g)
T, BASE, C, data, macro = g["T"], g["BASE"], g["C"], g["data"], g["macro"]
rng = np.random.default_rng(0)
IB = {}
for s, (L, y) in data.items():
    n = len(y)
    IB[s] = np.array([i for i in (rng.integers(0, n, n) for _ in range(2000)) if all(0 < y[i, j].sum() < n for j in range(len(T)))])


def boot(s, idx):
    L, y = data[s]
    p = L[list(idx)].mean(0)
    I = IB[s]
    out = np.zeros(len(I))
    for j in range(len(T)):
        yy, sc_ = y[I, j], p[I, j]
        r = rankdata(sc_, axis=1); n1 = yy.sum(1)
        out += ((r * yy).sum(1) - n1 * (n1 + 1) / 2) / (n1 * (yy.shape[1] - n1))
    return out / len(T)


full = list(range(len(BASE)))
ref = {s: macro(*data[s], full) for s in data}
refb = {s: boot(s, full) for s in data}
print(f"all 14: GPU {C.sum():.1f} s | audit {ref['audit58']:.4f} | clean-121 {ref['clean121']:.4f}\n")
print("LEAVE-ONE-OUT (negative = the member helps; 'helps in' = share of bootstraps where removing it loses)")
rows = []
for k in full:
    idx = [i for i in full if i != k]
    d = {s: macro(*data[s], idx) - ref[s] for s in data}
    h = {s: float(np.mean(boot(s, idx) - refb[s] < 0)) for s in data}
    rows.append((k, d, h))
for k, d, h in sorted(rows, key=lambda r: (r[1]["audit58"] + r[1]["clean121"])):
    print(f"  {BASE[k]:48s} GPU {C[k]:4.2f} | without: audit {d['audit58']:+.4f} (helps in {h['audit58']:.0%}) "
          f"| clean {d['clean121']:+.4f} (helps in {h['clean121']:.0%})")

print("\nGREEDY BACKWARD ELIMINATION (least score lost per GPU second saved)")
cur = list(full)
while C[cur].sum() > 6:
    best = None
    base_m = sum(macro(*data[s], cur) for s in data) / 2
    for k in cur:
        idx = [i for i in cur if i != k]
        loss = base_m - sum(macro(*data[s], idx) for s in data) / 2
        score = loss / C[k]
        if best is None or score < best[0]:
            best = (score, k)
    cur.remove(best[1])
    a, c = macro(*data["audit58"], cur), macro(*data["clean121"], cur)
    print(f"  drop {BASE[best[1]]:48s} -> {len(cur):2d} members, GPU {C[cur].sum():5.1f} s | audit {a:.4f} ({a - ref['audit58']:+.4f})"
          f" | clean {c:.4f} ({c - ref['clean121']:+.4f})")
