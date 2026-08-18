"""
08_check_crop_safety.py — does the centre crop ever cut into the knee?

The dataset rescales each slice to a fixed mm/px, then centre-crops or pads to
`size`. That is only safe if the anatomy is (a) centred and (b) small enough to
fit. We measured the anatomy bounding box for every series in the full DICOM
scan, so this can be checked exactly instead of assumed.

    python data_code/08_check_crop_safety.py --size 512 --mm-per-px 0.40
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--mm-per-px", type=float, default=0.40)
args = ap.parse_args()

S = pd.read_csv(ROOT / "data" / "stats" / "series_stats.csv")
S = S.dropna(subset=["tbox_r0", "tbox_r1", "tbox_c0", "tbox_c1",
                     "pixel_spacing_c", "pixel_spacing_r"])
# only trust rows where foreground detection actually worked
S = S[S.fg_ok_frac >= 0.5]

# rescale everything to the target spacing, exactly as the dataset does
fr = S["pixel_spacing_r"] / args.mm_per_px
fc = S["pixel_spacing_c"] / args.mm_per_px
H, W = S["rows"] * fr, S["cols"] * fc
r0, r1 = S["tbox_r0"] * fr, S["tbox_r1"] * fr
c0, c1 = S["tbox_c0"] * fc, S["tbox_c1"] * fc

# the crop window the dataset would take
s = args.size
top = np.where(H > s, ((H - s) // 2), 0)
left = np.where(W > s, ((W - s) // 2), 0)
bot = np.where(H > s, top + s, H)
right = np.where(W > s, left + s, W)

cut_top = np.maximum(0, top - r0)
cut_bot = np.maximum(0, r1 - bot)
cut_left = np.maximum(0, left - c0)
cut_right = np.maximum(0, c1 - right)
cut = cut_top + cut_bot + cut_left + cut_right

anat_h, anat_w = r1 - r0, c1 - c0
lost_frac = ((cut_top + cut_bot) / anat_h.clip(lower=1)
             + (cut_left + cut_right) / anat_w.clip(lower=1)) / 2

print("=" * 70)
print(f"CENTRE-CROP SAFETY @ {args.size}px, {args.mm_per_px} mm/px "
      f"({args.size * args.mm_per_px:.0f} mm field)")
print("=" * 70)
print(f"series checked            : {len(S):,}")
print(f"frames needing any crop   : {int(((H > s) | (W > s)).sum()):,} "
      f"({((H > s) | (W > s)).mean():.1%})")
print(f"series losing ANY anatomy : {int((cut > 0.5).sum()):,} ({(cut > 0.5).mean():.2%})")
print(f"  ...losing >5% of it     : {int((lost_frac > 0.05).sum()):,} "
      f"({(lost_frac > 0.05).mean():.2%})")
print(f"  ...losing >20% of it    : {int((lost_frac > 0.20).sum()):,} "
      f"({(lost_frac > 0.20).mean():.2%})")

print("\nwhere the loss happens (px of anatomy cut, worst edge):")
for name, v in [("top", cut_top), ("bottom", cut_bot),
                ("left", cut_left), ("right", cut_right)]:
    print(f"  {name:7s} mean {v.mean():6.1f}   p99 {np.percentile(v, 99):7.1f}   "
          f"max {v.max():7.1f}")

print("\nanatomy size after rescale, in pixels:")
print(pd.DataFrame({"height": anat_h, "width": anat_w})
      .describe(percentiles=[.5, .95, .99, .999]).round(0).to_string())

print("\nis the anatomy actually centred? (offset of anatomy centre from frame centre, px)")
off_r = ((r0 + r1) / 2 - H / 2).abs()
off_c = ((c0 + c1) / 2 - W / 2).abs()
print(pd.DataFrame({"vertical": off_r, "horizontal": off_c})
      .describe(percentiles=[.5, .95, .99]).round(1).to_string())

print("\nworst offenders:")
bad = S.assign(lost=lost_frac.values, cut=cut.values).nlargest(8, "lost")
print(bad[["series_desc", "rows", "cols", "pixel_spacing_c", "n_slices"]]
      .assign(lost_pct=(bad["lost"] * 100).round(1)).to_string(index=False))

print("\nsize sweep — % of series losing >5% of anatomy:")
for sz in [320, 384, 448, 512, 576, 640]:
    t = np.where(H > sz, ((H - sz) // 2), 0)
    l = np.where(W > sz, ((W - sz) // 2), 0)
    b = np.where(H > sz, t + sz, H)
    rr = np.where(W > sz, l + sz, W)
    lf = ((np.maximum(0, t - r0) + np.maximum(0, r1 - b)) / anat_h.clip(lower=1)
          + (np.maximum(0, l - c0) + np.maximum(0, c1 - rr)) / anat_w.clip(lower=1)) / 2
    print(f"  {sz:4d}px ({sz*args.mm_per_px:5.0f} mm) : {(lf > 0.05).mean():6.2%}")
