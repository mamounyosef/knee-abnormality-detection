"""native_slice_stats.py - how many of each series' real slices do our 18-slice models never see? (slice-zoom potential)

For the series the zoom runs read (their own slot plan): native slice count, slice thickness and spacing, and the
share of slices dropped by an even 18-slice sample. A slice zoom that reads ALL native slices inside the hot half of
the volume gains real data only where the series has more than 18 slices.

    python runs/others/crop_rois/native_slice_stats.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

cfg = sc.load_config(ROOT / "runs" / "convnext_base_25d_v5_joint_zoom_030_ax", verbose=False)
idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)).set_index("series")
plan = sc.plan_slots(idx.reset_index(), cfg)
for slot in ["sag_plain", "sag_fluid", "cor_fluid", "cor_plain", "ax_fluid"]:
    ser = [p[slot] for p in plan.values() if p.get(slot)] if slot in cfg["input"]["slots"] else []
    if not ser:
        c2 = dict(cfg); c2["input"] = dict(cfg["input"], slots=[slot])
        ser = [p[slot] for p in sc.plan_slots(idx.reset_index(), c2).values() if p.get(slot)]
    d = idx.loc[ser]
    n = d.n_slices.to_numpy(float)
    drop = np.clip(1 - 18 / n, 0, 1)
    half = np.clip(n / 2, 0, None)                      # native slices in the hot half
    print(f"{slot:10s} n={len(d)} slices median {np.median(n):.0f} (p10 {np.percentile(n, 10):.0f}, p90 {np.percentile(n, 90):.0f}) | "
          f"thickness {d.slice_thickness.median():.1f} mm, spacing {d.spacing_between.median():.1f} mm | "
          f"18-slice sample drops {drop.mean():.0%} of slices | hot half holds {np.median(half):.0f} native slices "
          f"(> 9 in {np.mean(half > 9):.0%} of series)")
