"""
07_make_series_index.py — one small table the dataset can plan from.
====================================================================

The training dataset has to decide, per study, which series fills each input
slot. Opening a 30 MiB shard just to read its metadata would be absurd, so this
flattens everything the decision needs into a single CSV:

    study, series, plane, fluid, slot,
    n_slices, rows, cols, pixel_spacing_r/c, slice_thickness, spacing_between,
    depth_mm, fov_w_mm

`slot` is the plane+setting name (`sag_plain`, `sag_fluid`, `cor_fluid`,
`ax_fluid`, `cor_plain`, `ax_plain`). Which slots a run actually uses, and how
duplicates are resolved, stay run-time config — this file just describes what
exists.

    python data_code/07_make_series_index.py
Output: data/07_series_index.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "07_series_index.csv"

ts = (pd.read_csv(ROOT / "data" / "raw" / "train_series.csv")
      .rename(columns={"StudyInstanceUID": "study", "SeriesInstanceUID": "series"}))
st = pd.read_csv(ROOT / "data" / "stats" / "series_stats.csv")

df = ts.merge(st, on=["study", "series"], how="left", validate="one_to_one")

df["plane"] = df["Anatomical_Plane"].str.lower()
df["fluid"] = df["Fluid_Sensitive"].astype(int)
df["slot"] = df["plane"].str[:3].map({"sag": "sag", "cor": "cor", "axi": "ax"}) \
             + "_" + df["fluid"].map({1: "fluid", 0: "plain"})

df["depth_mm"] = df["n_slices"] * df["spacing_between"].fillna(df["slice_thickness"])
df["fov_w_mm"] = df["cols"] * df["pixel_spacing_c"]

keep = ["study", "series", "plane", "fluid", "slot", "n_slices", "rows", "cols",
        "pixel_spacing_r", "pixel_spacing_c", "slice_thickness", "spacing_between",
        "depth_mm", "fov_w_mm", "series_desc", "manufacturer"]
df[keep].to_csv(OUT, index=False)

print(f"wrote {OUT}  ({len(df)} series, {df.study.nunique()} studies)")
print(df["slot"].value_counts().to_string())
print(f"\nstudies missing a slot entirely:")
piv = df.pivot_table(index="study", columns="slot", values="n_slices", aggfunc="size").fillna(0)
for c in sorted(piv.columns):
    print(f"  {c:10s} present in {int((piv[c] > 0).sum()):5d} studies "
          f"({(piv[c] > 0).mean():.1%})")
