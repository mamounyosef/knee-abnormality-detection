"""
10_metadata_audit.py - what non-pixel information does this dataset actually carry?

The model currently sees pixels and nothing else. A radiologist reading these scans knows
the patient's age and sex, which scanner made the images and what sequence was used. Some
of that is strongly predictive on its own: osteoarthritis prevalence is driven mostly by
age, and patellofemoral osteoarthritis is markedly more common in women. Three of our twelve
findings are osteoarthritis and two of those are among the four weakest.

Before building anything on that, the metadata has to be shown to EXIST and to VARY. Many
competition datasets strip patient identifiers, age among them, and a field that is present
but constant carries no information at all.

This dumps every DICOM tag found in the local sample, with how often it is filled and how
many distinct values it takes, and then cross-checks the fields already extracted into
data/stats/series_stats.csv across all 24,371 series.

    python data_code/10_metadata_audit.py
    python data_code/10_metadata_audit.py --dicom-dir data/sample_dicoms
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--dicom-dir", default="data/sample_dicoms")
ap.add_argument("--stats", default="data/stats/series_stats.csv")
a = ap.parse_args()

# ---------------------------------------------------------------- 1. raw DICOM tags
try:
    import pydicom
except ImportError:
    raise SystemExit("pydicom is not installed; pip install pydicom")

files = sorted((ROOT / a.dicom_dir).glob("*.dcm"))
print("=" * 78)
print("RAW DICOM TAGS  (%d local sample files)" % len(files))
print("=" * 78)

vals = defaultdict(list)
for f in files:
    ds = pydicom.dcmread(str(f), stop_before_pixels=True)
    for elem in ds:
        if elem.keyword in ("PixelData", ""):
            continue
        v = elem.value
        # sequences and long binary blobs are not usable as model input
        v = "<seq>" if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)) \
            and not isinstance(v, (int, float)) else v
        vals[elem.keyword].append(str(v)[:60])

print("%-34s %6s %7s  %s" % ("tag", "filled", "distinct", "example values"))
print("-" * 78)
for k in sorted(vals, key=lambda k: (-len(set(vals[k])), k)):
    seen = [x for x in vals[k] if x not in ("", "None", "<seq>")]
    d = len(set(seen))
    ex = ", ".join(list(dict.fromkeys(seen))[:3])
    flag = "  <-- USABLE" if d > 1 else ("  (constant)" if d == 1 else "  (EMPTY)")
    print("%-34s %5d/%d %7d  %s%s" % (k, len(seen), len(files), d, ex[:40], flag))

# ---------------------------------------------------------------- 2. the extracted stats
print()
print("=" * 78)
print("ALREADY EXTRACTED, ACROSS EVERY SERIES  (%s)" % a.stats)
print("=" * 78)
S = pd.read_csv(ROOT / a.stats)
print("%d series, %d studies\n" % (len(S), S.study.nunique()))

print("%-22s %8s %9s  %s" % ("column", "non-null", "distinct", "top values / range"))
print("-" * 78)
for c in S.columns:
    if c in ("study", "series"):
        continue
    nn = S[c].notna().sum()
    d = S[c].nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(S[c]) and d > 12:
        desc = "%.3g to %.3g, median %.3g" % (S[c].min(), S[c].max(), S[c].median())
    else:
        top = S[c].value_counts(dropna=True).head(3)
        desc = ", ".join("%s(%d)" % (k, v) for k, v in top.items())
    print("%-22s %8d %9d  %s" % (c, nn, d, desc[:44]))

print("\nWhat matters: a field is only useful if it is filled for most series AND takes")
print("more than one value. Anything constant is a column of zeros to the model.")
