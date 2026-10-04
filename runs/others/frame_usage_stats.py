"""frame_usage_stats.py - how well does today's crop use its pixels? Measured on the real loader, not estimated.

Today: every series is resampled to 0.40 mm/px and centre-cropped to 336 px (134 mm). For the 404 v5 val studies, per
view (sag_plain, sag_fluid, cor_fluid, ax_fluid, plus cor_plain), on the middle third of the 18 sampled slices:
  * the full resampled image BEFORE the crop: tissue bounding box (rows/columns holding > 5% tissue) in mm, and how
    far the tissue centre sits from the image centre (mm)
  * inside the 336 px crop: share of pixels that are air (background), and how often tissue is CUT by the crop edge
    (left/right and top/bottom)
  * the native pixel size vs our 0.40 mm (how much detail the resample throws away)
Tissue = intensity > 12 of 255 after the shard's windowing (air is ~0).

    python runs/others/frame_usage_stats.py            # ~404 studies, a few minutes on the CPU
    python runs/others/frame_usage_stats.py --limit 100
"""
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RUN = "convnext_base_25d_18slices_v5_aug_base"
lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
spec = importlib.util.spec_from_file_location(RUN, ROOT / "runs" / RUN / "train.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
cfg, sc = copy.deepcopy(m.cfg), m.sc
cfg["input"]["slots"] = ["sag_plain", "sag_fluid", "cor_plain", "cor_fluid", "ax_fluid"]
cfg["input"]["laterality_flip"] = None
idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
plan = sc.plan_slots(idx, cfg)
df = pd.read_csv(sc.resolve(cfg["paths"]["val_csv"], cfg))
if lim:
    df = df.head(lim)
ds = sc.KneeShardDataset(df, cfg, plan, False)
ds._fit = lambda v: v                                   # keep the full resampled image (no crop)
MM, S, THR = float(cfg["input"]["mm_per_px"]), int(cfg["input"]["size"]), 12
geo = idx.set_index("series")
rows = []
rng = np.random.default_rng(0)
for i, st in enumerate(df.StudyInstanceUID):
    for slot, series in plan.get(st, {}).items():
        try:
            v = ds._load(st, series, rng, 18)           # (18*ctx, H, W) at 0.40 mm/px, uncropped
        except Exception:
            continue
        v = v[len(v) // 3: 2 * len(v) // 3]
        tis = (v > THR).mean(0)                          # per-pixel tissue share over the middle slices
        H, W = tis.shape
        r_on = np.where((tis > 0.5).mean(1) > 0.05)[0]
        c_on = np.where((tis > 0.5).mean(0) > 0.05)[0]
        if len(r_on) == 0 or len(c_on) == 0:
            continue
        cy, cx = (r_on[0] + r_on[-1]) / 2, (c_on[0] + c_on[-1]) / 2
        t, l = max((H - S) // 2, 0), max((W - S) // 2, 0)
        crop = tis[t:t + S, l:l + S]
        rows.append({"slot": slot, "img_w_mm": W * MM, "img_h_mm": H * MM,
                     "tis_w_mm": (c_on[-1] - c_on[0] + 1) * MM, "tis_h_mm": (r_on[-1] - r_on[0] + 1) * MM,
                     "off_x_mm": (cx - (W - 1) / 2) * MM, "off_y_mm": (cy - (H - 1) / 2) * MM,
                     "air_in_crop": float((crop < 0.5).mean()),
                     "cut_lr": bool(W > S and (c_on[0] < l or c_on[-1] >= l + S)),
                     "cut_tb": bool(H > S and (r_on[0] < t or r_on[-1] >= t + S)),
                     "native_px": float(geo.loc[series, "pixel_spacing_c"])})
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(df)} studies", flush=True)
r = pd.DataFrame(rows)
q = lambda s: f"{s.median():6.1f} ({s.quantile(.1):5.1f}-{s.quantile(.9):5.1f})"   # noqa: E731
print(f"\n{len(r)} series from {len(df)} studies. Frame today: {S} px x {MM} mm = {S * MM:.0f} mm. median (10%-90%)\n")
print(f"{'view':10s} {'n':>4s} {'image mm':>18s} {'tissue width mm':>18s} {'tissue height mm':>18s} "
      f"{'centre off x mm':>18s} {'centre off y mm':>18s} {'air in crop':>11s} {'cut L/R':>8s} {'cut T/B':>8s}")
for s, g in r.groupby("slot"):
    print(f"{s:10s} {len(g):4d} {q(g.img_w_mm):>18s} {q(g.tis_w_mm):>18s} {q(g.tis_h_mm):>18s} "
          f"{q(g.off_x_mm):>18s} {q(g.off_y_mm):>18s} {g.air_in_crop.mean():11.0%} {g.cut_lr.mean():8.0%} {g.cut_tb.mean():8.0%}")
print(f"\nall views: air in the crop {r.air_in_crop.mean():.0%}; |centre offset| > 10 mm in "
      f"{((r.off_x_mm.abs() > 10) | (r.off_y_mm.abs() > 10)).mean():.0%} of series; native pixel median "
      f"{r.native_px.median():.3f} mm, finer than {MM} mm in {(r.native_px < MM).mean():.0%}")
out = ROOT / "runs" / "others" / "logs" / "frame_usage_stats.csv"
r.to_csv(out, index=False)
print("per-series rows ->", out)
