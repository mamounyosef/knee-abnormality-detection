r"""extract_laterality.py - which knee is it, left or right, for every study.

WHY IT MATTERS. Four of the twelve findings are side-specific: Medial and Lateral Meniscus,
Medial and Lateral OA. "Medial" means the inner side of the leg - which is the LEFT of the
image for a right knee and the RIGHT of the image for a left knee. The public 0.941 notebook
normalises this, flipping right knees so every study faces the same way. We never have: our
only nod to it is `hflip: {p: 0.0}` in the augmentation, a note saying a flip "would swap
medial <-> lateral".

So our models must learn each side-specific finding twice, once per orientation, from a
dataset split roughly in half. If that is costing us, normalising is worth a retrain; if the
model is equally good on both sides, it is not. This gets the label so we can tell.

NO DICOM RESCAN NEEDED. The shard metadata already carries ImagePositionPatient as
`pos_first`/`pos_last`. In the DICOM patient coordinate system the x axis increases toward
the patient's LEFT, so the sign of x over the knee says which leg it is. Reading just the
`meta` key of each shard is a few KB per study, CPU only.

Writes data/12_laterality.csv: StudyInstanceUID, side, x_mean, n_series, confidence.

    modal run runs/others/extract_laterality.py
"""

import json
import sys
from pathlib import Path

import numpy as np


def _pkg_root() -> Path:
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402


def side_of(meta):
    """Left/right from the x coordinate of the image positions.

    DICOM patient coordinates: +x toward the patient's LEFT, +y posterior, +z superior.
    A knee sits well off the midline, so the mean x over its slices is solidly signed.
    Returns (side, x_mean, n_series_used).
    """
    xs = []
    for s in (meta.get("series") or {}).values():
        for key in ("pos_first", "pos_last"):
            p = s.get(key)
            if p and len(p) == 3 and p[0] is not None:
                xs.append(float(p[0]))
    if not xs:
        return "unknown", float("nan"), 0
    x = float(np.mean(xs))
    return ("left" if x > 0 else "right"), x, len(xs)


def run(shards_dir, out_csv):
    shards = sorted(Path(shards_dir).glob("*.npz"))
    print("  %d shards in %s" % (len(shards), shards_dir), flush=True)
    rows = []
    for i, f in enumerate(shards):
        try:
            with np.load(f, allow_pickle=True) as z:
                meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        except Exception as e:
            print("    %s: %s" % (f.stem, e), flush=True)
            continue
        side, x, n = side_of(meta)
        rows.append((meta.get("study", f.stem), side, x, n))
        if i % 500 == 0:
            print("    %d/%d" % (i, len(shards)), flush=True)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write("StudyInstanceUID,side,x_mean,n_positions\n")
        for sid, side, x, n in rows:
            fh.write("%s,%s,%.3f,%d\n" % (sid, side, x, n))
    from collections import Counter
    c = Counter(r[1] for r in rows)
    print("  wrote %s: %d studies  %s" % (out, len(rows), dict(c)), flush=True)
    return rows


try:
    import modal

    cfg = sc.load_config(PKG_ROOT / "convnext_base_relabel", verbose=False)
    app = modal.App("extract-laterality")
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(cfg["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(image=_image,
                  volumes={cfg["modal"]["shards_mount"]: _shards,
                           cfg["modal"]["runs_mount"]: _runs},
                  cpu=4.0, memory=8192, timeout=3600)          # CPU only: no pixels are read
    def laterality_remote():
        out = Path(cfg["modal"]["runs_mount"]) / "others" / "12_laterality.csv"
        try:
            run(Path(cfg["modal"]["shards_mount"]), out)
        finally:
            _runs.commit()

    @app.local_entrypoint()
    def modal_main():
        laterality_remote.remote()

except ImportError:
    pass


if __name__ == "__main__":
    _cfg = sc.load_config(PKG_ROOT / "convnext_base_relabel", verbose=False)
    run(sc.resolve(_cfg["paths"]["shards_dir"], _cfg),
        Path(_cfg["paths"]["data_dir"]) / "12_laterality.csv")
