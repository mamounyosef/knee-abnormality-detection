"""duplicate_series_stats.py - how many series do we discard? Per slot: studies with more than one candidate series, how many
extra series and slices that is, and how the discarded ones differ from the kept one (the default rule: most coverage, then
finest pixel). Also the series that fit NO slot at all.

    python runs/others/duplicate_series_stats.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent.parent.parent / "data"
idx = pd.read_csv(D / "07_series_index.csv")
SLOTS6 = ["sag_plain", "sag_fluid", "cor_plain", "cor_fluid", "ax_plain", "ax_fluid"]
SLOTS4 = ["sag_plain", "sag_fluid", "cor_fluid", "ax_fluid"]
n_st = idx.study.nunique()
print(f"{n_st} studies, {len(idx)} series, {idx.n_slices.sum():,} slices in total "
      f"(mean {len(idx) / n_st:.1f} series per study)\n")
print(f"{'slot':10s} {'studies w/ slot':>15s} {'w/ >1 series':>13s} {'extra series':>13s} {'extra slices':>13s} "
      f"{'max':>4s}")
tot_x, tot_xs = 0, 0
for s in SLOTS6:
    g = idx[idx.slot == s].groupby("study")
    n = g.size()
    extra = (n - 1).clip(lower=0)
    # slices in the series the default rule discards (keeps max depth_mm, then finest pixel)
    xs = 0
    for _, gg in idx[idx.slot == s].groupby("study"):
        if len(gg) > 1:
            k = gg.sort_values(["depth_mm", "pixel_spacing_c"], ascending=[False, True])
            xs += int(k.iloc[1:].n_slices.sum())
    mark = " (4-slot)" if s in SLOTS4 else ""
    print(f"{s:10s} {len(n):15d} {int((n > 1).sum()):6d} ({(n > 1).mean():4.0%}) {int(extra.sum()):13d} {xs:13,d} "
          f"{int(n.max()):4d}{mark}")
    if s in SLOTS4:
        tot_x, tot_xs = tot_x + int(extra.sum()), tot_xs + xs
used4 = sum(1 for s in SLOTS4 for _ in idx[idx.slot == s].groupby("study"))
print(f"\n4-slot recipe: {used4} series used, {tot_x} competing series discarded ({tot_x / (used4 + tot_x):.0%} of the "
      f"candidates), {tot_xs:,} slices in them")
none = idx[~idx.slot.isin(SLOTS6)]
print(f"series that fit no slot: {len(none)}  (slot values: {none.slot.value_counts().to_dict()})")
multi = idx[idx.slot.isin(SLOTS4)].groupby(["study", "slot"]).size()
print(f"studies with at least one 4-slot duplicate: {multi[multi > 1].reset_index().study.nunique()} of {n_st}")

print("\nhow a discarded series differs from the kept one (4 slots, all duplicate pairs):")
rows = []
for (st, s), gg in idx[idx.slot.isin(SLOTS4)].groupby(["study", "slot"]):
    if len(gg) < 2:
        continue
    k = gg.sort_values(["depth_mm", "pixel_spacing_c"], ascending=[False, True])
    a = k.iloc[0]
    for _, b in k.iloc[1:].iterrows():
        rows.append({"slot": s, "same_desc": a.series_desc == b.series_desc,
                     "thick_ratio": b.slice_thickness / a.slice_thickness if a.slice_thickness else np.nan,
                     "px_ratio": b.pixel_spacing_c / a.pixel_spacing_c, "depth_ratio": b.depth_mm / a.depth_mm,
                     "slices_ratio": b.n_slices / a.n_slices, "desc_kept": a.series_desc, "desc_disc": b.series_desc})
r = pd.DataFrame(rows)
print(f"  pairs {len(r)}; identical description {r.same_desc.mean():.0%}")
for c in ("thick_ratio", "px_ratio", "depth_ratio", "slices_ratio"):
    print(f"  discarded/kept {c[:-6]:8s} median {r[c].median():.2f}  (10%-90%: {r[c].quantile(.1):.2f}-{r[c].quantile(.9):.2f})")
print("  most common discarded descriptions:", r.desc_disc.str.upper().value_counts().head(8).to_dict())
