"""native_spacing_stats.py - how much real detail can a 0.20 mm/px crop add? Native in-plane pixel size of the series
the joint-zoom runs read (their own slot plan), against the 0.40 mm/px the normal models use.

    python runs/others/crop_rois/native_spacing_stats.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

cfg = sc.load_config(ROOT / "runs" / "convnext_base_25d_v5_joint_zoom", verbose=False)
idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
plan = sc.plan_slots(idx, cfg)
sp = idx.set_index("series")["pixel_spacing_c"]
for slot in cfg["input"]["slots"]:
    v = np.array([sp.get(p[slot], np.nan) for p in plan.values() if p.get(slot)], float)
    v = v[np.isfinite(v)]
    print(f"{slot:10s} n={len(v)} median {np.median(v):.3f} mm | <0.25: {np.mean(v < 0.25):.0%} | 0.25-0.40: "
          f"{np.mean((v >= 0.25) & (v < 0.40)):.0%} | >=0.40 (no gain): {np.mean(v >= 0.40):.0%} | "
          f"mean real gain over 0.40: {np.mean(np.clip(0.40 / np.maximum(v, 0.20), 1, 2)):.2f}x")
