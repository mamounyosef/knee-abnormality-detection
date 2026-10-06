"""slice_repeat_stats.py - with n_slices = K, how many series get REPEATED slices (pick_slices' short-stack branch:
real slices duplicated, never interpolated) and how many unique real slices does a study actually contribute?

    python runs/others/crop_rois/slice_repeat_stats.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

cfg = sc.load_config(ROOT / "runs" / "convnext_base_25d_v5_joint_zoom_030_30sl", verbose=False)
idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)).set_index("series")
plan = sc.plan_slots(idx.reset_index(), cfg)
margin = cfg["input"].get("slice_margin", 0.0)
rng = np.random.default_rng(0)
for K in (18, 24, 30):
    rep, uniq, frac = [], [], []
    for p in plan.values():
        for sl in cfg["input"]["slots"]:
            s = p.get(sl)
            if not s:
                continue
            n = int(idx.loc[s, "n_slices"])
            pick = sc.pick_slices(n, K, "even", rng, margin)
            u = len(set(pick.tolist()))
            rep.append(u < K); uniq.append(u); frac.append(u / n)
    rep, uniq, frac = np.array(rep), np.array(uniq), np.array(frac)
    print(f"K={K}: series with repeated slices {rep.mean():.0%} | unique real slices per series median {np.median(uniq):.0f} "
          f"| share of the series' slices used: median {np.median(frac):.0%}, p10 {np.percentile(frac, 10):.0%}")
