r"""lateral_side_sign_check.py - after the left-knee flip, is the knee's LATERAL side always on the same side of the
coronal and axial images? (The premise of compartment-aware pooling.)

laterality_axis_check.py proved WHICH axis carries left-right (horizontal for coronal and axial). This checks the SIGN:
DICOM ImageOrientationPatient's first vector h is the patient-space direction of increasing image column, and patient
+x points to the patient's LEFT. So h[0] > 0 means "image right = patient left". If every coronal/axial series has the
same sign, then after mirroring left knees (all knees look like right knees) the lateral side (patient right for a
right knee) sits on the same image side everywhere: image LEFT when h[0] > 0.

Also reports, per plane, the share of series with |h[0]| < 0.7 (oblique enough that the side is ambiguous).

    python checks/lateral_side_sign_check.py            # all local shards
    python checks/lateral_side_sign_check.py --limit 500
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
cfg = sc.load_config(ROOT / "runs" / "convnext_base_25d_18slices_v5_aug_base", verbose=False)
shards = sorted(Path(sc.resolve(cfg["paths"]["shards_dir"], cfg)).glob("*.npz"))
if lim:
    shards = shards[:lim]
idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
plane_of = dict(zip(idx["series"].astype(str), idx["plane"]))
rows = []
for f in shards:
    with np.load(f, allow_pickle=True) as z:
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
    for sid, s in (meta.get("series") or {}).items():
        iop = s.get("orientation")
        if iop and len(iop) == 6:
            rows.append((f.stem, str(sid), plane_of.get(str(sid), "?"), float(iop[0])))
df = pd.DataFrame(rows, columns=["study", "series", "plane", "h0"])
print(f"{len(df)} series with orientation from {len(shards)} shards\n")
print(f"{'plane':10s} {'n':>6s} {'h0 > 0.7':>9s} {'h0 < -0.7':>10s} {'|h0| < 0.7':>11s}")
for p, g in df.groupby("plane"):
    print(f"{p:10s} {len(g):6d} {np.mean(g.h0 > 0.7):9.1%} {np.mean(g.h0 < -0.7):10.1%} {np.mean(g.h0.abs() < 0.7):11.1%}")
cor_ax = df[df.plane.isin(["coronal", "axial"])]
bad = cor_ax[cor_ax.h0 < 0]
print(f"\ncoronal + axial series with image-right = patient-RIGHT (h0 < 0): {len(bad)} of {len(cor_ax)} "
      f"({len(bad) / max(len(cor_ax), 1):.2%}), in {bad.study.nunique()} studies")
print("VERDICT:", "lateral side is on a FIXED image side (image left after the flip)" if len(bad) / max(len(cor_ax), 1) < 0.02
      else "NOT consistent: compartment pooling would need a per-series width flip from h0")
