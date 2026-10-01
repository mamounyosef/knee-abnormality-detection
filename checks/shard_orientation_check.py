"""shard_orientation_check.py - is the image / slice orientation consistent across studies in the shards?

The preprocessing keeps each scanner's native pixel layout and orders slices along n = row x col
(from ImageOrientationPatient). If scanners differ in the SIGN of the row/col directions, the
same anatomy lands mirrored or upside down, and the slice order runs the other way. This counts,
per plane, how each image axis points in patient space (DICOM LPS: +x = patient left,
+y = posterior, +z = head), and flags series whose slice count does not fit their extent
(duplicate or interleaved positions, e.g. several echoes or stacks in one series).

    python checks/shard_orientation_check.py [--max 2000]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AX = "xyz"
NAME = {("x", 1): "+x(L)", ("x", -1): "-x(R)", ("y", 1): "+y(P)", ("y", -1): "-y(A)",
        ("z", 1): "+z(H)", ("z", -1): "-z(F)"}


def axis(v):
    i = int(np.argmax(np.abs(v)))
    return NAME[(AX[i], int(np.sign(v[i])))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=2000)
    a = ap.parse_args()
    files = sorted((ROOT / "data/shards").glob("*.npz"))[:a.max]
    combos = {p: Counter() for p in ("sagittal", "coronal", "axial", "other")}
    bad_fit, errs, notes, n_series = [], 0, 0, 0
    for f in files:
        with np.load(f) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        errs += bool(meta.get("errors"))
        notes += bool(meta.get("notes"))
        for sid, m in meta["series"].items():
            o = m.get("orientation")
            if not o:
                continue
            n_series += 1
            r, c = np.array(o[:3]), np.array(o[3:])
            n = np.cross(r, c)
            k = int(np.argmax(np.abs(n)))
            plane = {0: "sagittal", 1: "coronal", 2: "axial"}[k] if abs(n[k]) > 0.8 else "other"
            combos[plane][(axis(r), axis(c), axis(n))] += 1
            sp = m.get("spacing_between") or m.get("slice_thickness")
            if m.get("pos_first") and m.get("pos_last") and sp and m["n_slices"] > 2:
                extent = abs(float(np.dot(np.array(m["pos_last"]) - np.array(m["pos_first"]), n)))
                ratio = extent / ((m["n_slices"] - 1) * sp)
                if not 0.8 < ratio < 1.3:
                    bad_fit.append((f.stem[-12:], m["series_desc"], m["n_slices"], round(ratio, 2)))
    print(f"{len(files)} studies, {n_series} series; studies with decode errors {errs}, "
          f"with mixed-size notes {notes}\n")
    for p, cnt in combos.items():
        if not cnt:
            continue
        tot = sum(cnt.values())
        print(f"{p}: {tot} series  (image right-going, image down-going, slice order)")
        for (r, c, n), v in cnt.most_common(8):
            print(f"    {v:5d} ({100 * v / tot:5.1f}%)  row {r:7s} col {c:7s} slices {n}")
    print(f"\nseries whose slice count does not fit their extent: {len(bad_fit)}")
    for b in sorted(bad_fit, key=lambda b: b[3])[:15]:
        print("   ", b)


if __name__ == "__main__":
    main()
