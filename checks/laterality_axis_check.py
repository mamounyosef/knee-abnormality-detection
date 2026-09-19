r"""laterality_axis_check.py - which image axis actually carries medial-vs-lateral?

WHY THIS EXISTS. convnext_base_25d_left_knees_mirrored normalised laterality by applying a
HORIZONTAL flip to every slot: `v[:, :, ::-1]`. That is correct for coronal and axial, and
WRONG for sagittal, where the medial-lateral direction is not in the image plane at all - it
is the slice axis. Two of that run's four slots are sagittal, so half its input was not being
normalised but corrupted, and its -0.0015 result does not mean what it was reported to mean.

This proves the geometry from the data rather than from an argument. DICOM's
ImageOrientationPatient gives the patient-space direction of the image's horizontal axis
(first three numbers) and vertical axis (last three). The patient x axis points to the
patient's LEFT, so it IS the medial-lateral axis. For each series we ask how much of patient
x lies along the image horizontal, the image vertical, and the slice normal. Whichever wins
is the axis a left/right mirror has to act on.

    python checks/laterality_axis_check.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""))


def axes_of(iop):
    """(horizontal, vertical, slice-normal) unit vectors in patient coordinates."""
    h = np.array(iop[:3], float)
    v = np.array(iop[3:], float)
    n = np.cross(h, v)
    return h, v, n / (np.linalg.norm(n) or 1.0)


def main():
    cfg = sc.load_config(ROOT / "runs" / "convnext_base_25d_18slices", verbose=False)
    shards = Path(sc.resolve(cfg["paths"]["shards_dir"], cfg))
    files = sorted(shards.glob("*.npz"))
    if not files:
        print("no local shards")
        return 1

    idx = __import__("pandas").read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
    plane_of = dict(zip(idx["series"], idx["plane"]))

    rows = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        for sid, s in (meta.get("series") or {}).items():
            iop = s.get("orientation")
            if not iop or len(iop) != 6:
                continue
            h, v, n = axes_of(iop)
            rows.append((plane_of.get(sid, "?"), abs(h[0]), abs(v[0]), abs(n[0])))

    print("  %d series with orientation, from %d shards\n" % (len(rows), len(files)))
    print("  how much of the patient LEFT-RIGHT axis lies along each image axis:")
    print("  %-10s %5s %12s %12s %12s   %s"
          % ("plane", "n", "horizontal", "vertical", "slice axis", "mirror acts on"))

    verdict = {}
    for plane in sorted({r[0] for r in rows}):
        g = [r for r in rows if r[0] == plane]
        hz = float(np.mean([r[1] for r in g]))
        vt = float(np.mean([r[2] for r in g]))
        sl = float(np.mean([r[3] for r in g]))
        which = max((hz, "horizontal"), (vt, "vertical"), (sl, "slice order"))[1]
        verdict[plane] = which
        print("  %-10s %5d %12.3f %12.3f %12.3f   %s"
              % (plane, len(g), hz, vt, sl, which.upper()))

    print()
    # the physical facts this rests on
    for plane, want in (("sagittal", "slice order"), ("coronal", "horizontal"),
                        ("axial", "horizontal")):
        if plane in verdict:
            check("%s: left-right runs along the %s" % (plane, want),
                  verdict[plane] == want, "measured %s" % verdict[plane])

    # what the shipped code actually does
    src = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
    flips_width_everywhere = "v = v[:, :, ::-1].copy()" in src
    check("shared_code no longer flips every slot horizontally",
          not flips_width_everywhere,
          "the old bug was `v[:, :, ::-1]` applied to all four slots")
    check("it mirrors per plane instead", "_mirror_slot(v, slot" in src)
    # and the helper must agree with the measurement, not just exist
    import numpy as _np
    t = _np.arange(4 * 2 * 3).reshape(4, 2, 3).astype(_np.float32)
    check("sagittal mirrors by reversing slice order",
          _np.array_equal(sc._mirror_slot(t, "sag_fluid", 1), t[::-1]))
    check("coronal mirrors by reversing width",
          _np.array_equal(sc._mirror_slot(t, "cor_fluid", 1), t[:, :, ::-1]))
    check("axial mirrors by reversing width",
          _np.array_equal(sc._mirror_slot(t, "ax_fluid", 1), t[:, :, ::-1]))

    sag_wrong = verdict.get("sagittal") != "horizontal" and flips_width_everywhere
    print()
    if sag_wrong:
        print("  => THE SAGITTAL SLOTS ARE BEING MIRRORED ON THE WRONG AXIS.")
        print("     A horizontal flip of a sagittal image reverses ANTERIOR-POSTERIOR, which")
        print("     is not a laterality difference - it is anatomical corruption. The correct")
        print("     operation for sagittal is to reverse the SLICE ORDER.")
        print("     convnext_base_25d_left_knees_mirrored used 4 slots, 2 of them sagittal,")
        print("     so half of its input was corrupted rather than normalised. Its -0.0015")
        print("     does not measure laterality normalisation.")
    else:
        print("  => the flip axis matches the geometry for every plane")

    print("\n  %d passed, %d failed" % (len(OK), len(FAIL)))
    for f in FAIL:
        print("    FAILED: %s" % f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
