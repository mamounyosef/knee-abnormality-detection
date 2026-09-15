"""
modal_preprocess.py — decode the competition zip into training-ready shards.
============================================================================

DESIGN RULE: this stage makes no modelling decisions. It decodes every series of
every study at native resolution, keeps every slice, and stores the result in a
format that is fast to read. Which series to feed a model, how many slices to
sample, what pixel spacing to resample to, what input size to crop to and how to
window intensity are all RUN-TIME choices, made in the dataset class from config
so that every planned ablation stays open.

The single irreversible step is 16-bit -> 8-bit, and it is taken deliberately:

  * No backbone we would use trains on 16-bit input, and MRI's noise floor means
    the extra bits carry almost nothing once windowed.
  * The window used here is intentionally WIDE - the 0.1st to 99.9th percentile
    of each volume - so it clips only noise spikes. Any narrower windowing
    (per-slice, per-volume 1-99, a fixed window) can still be applied at run time
    on the 8-bit data, because that only ever compresses the range further.
  * The full percentile table is stored per series, so a run-time window is
    exactly reproducible.

Output, one .npz per study on the `knee-shards` volume:

    <study>.npz
        <SeriesInstanceUID>   uint8 array, (n_slices, H, W), native H/W
        ...one per series...
        meta                  JSON: per-series spacing, plane, fluid flag,
                              percentiles, slice order, orientation, fallbacks

Slices are ordered by anatomical position (ImagePositionPatient projected onto
the slice normal), not by InstanceNumber or filename, so the stack is
geometrically correct.

Usage (pick the workspace holding the zip — see modal_setup_commands.txt):
    modal run modal_scripts/modal_preprocess.py --limit-studies 20   # smoke test
    modal run --detach modal_scripts/modal_preprocess.py             # everything
    modal run modal_scripts/modal_preprocess.py::verify
"""

import io
import json
import os

import modal

COMP = "rsna-knee-abnormality-detection"
DATA_VOLUME = "knee-data"
SHARD_VOLUME = "knee-shards"
DATA_MOUNT = "/data"
SHARD_MOUNT = "/shards"
ARCHIVE = f"{DATA_MOUNT}/{COMP}.zip"

SHARD_SIZE = 40           # studies per container
MAX_CONTAINERS = 20
N_CORES = 8

# wide window: clip only noise spikes, keep every real grey level
LO_PCT, HI_PCT = 0.1, 99.9

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("pydicom==3.0.1", "pylibjpeg", "pylibjpeg-libjpeg",
                 "pylibjpeg-openjpeg", "numpy", "opencv-python-headless", "pandas")
    # the archive may be a single .zip (older volumes) or part files (everything since the
    # split); archive_parts.open_archive hides the difference behind one file object
    .add_local_python_source("archive_parts")
)

app = modal.App("knee-preprocess")
data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
shard_vol = modal.Volume.from_name(SHARD_VOLUME, create_if_missing=True)


# =============================================================================
# worker
# =============================================================================
def _process_study(job):
    """job = (study, [member paths]) -> (study, npz bytes, meta dict)."""
    import zipfile

    import cv2
    import numpy as np
    import pydicom

    study, members = job
    by_series: dict[str, list[str]] = {}
    for m in members:
        by_series.setdefault(m.split("/")[2], []).append(m)

    arrays: dict[str, np.ndarray] = {}
    meta = {"study": study, "series": {}}

    import archive_parts as ap

    with ap.open_archive(DATA_MOUNT, COMP) as _fh, zipfile.ZipFile(_fh) as zf:
        for series, paths in by_series.items():
            slices = []
            for p in paths:
                try:
                    ds = pydicom.dcmread(io.BytesIO(zf.read(p)))
                    a = ds.pixel_array
                    if a.ndim == 3:                    # multi-frame: split it out
                        for k in range(a.shape[0]):
                            slices.append((ds, a[k], k))
                        continue
                    slices.append((ds, a, None))
                except Exception as exc:               # noqa: BLE001
                    meta.setdefault("errors", []).append(
                        {"member": p, "error": f"{type(exc).__name__}: {exc}"[:150]})

            if not slices:
                continue

            # --- order slices by anatomical position ---------------------------
            # ImagePositionPatient projected onto the slice normal (from
            # ImageOrientationPatient) gives true through-plane order. Fall back
            # to SliceLocation, then InstanceNumber, when geometry is missing.
            def _key(item):
                ds, _, frame = item
                iop = ds.get("ImageOrientationPatient")
                ipp = ds.get("ImagePositionPatient")
                if iop is not None and ipp is not None and len(iop) == 6:
                    r = np.array(iop[:3], dtype=float)
                    c = np.array(iop[3:], dtype=float)
                    n = np.cross(r, c)
                    return float(np.dot(np.array(ipp, dtype=float), n))
                if ds.get("SliceLocation") is not None:
                    return float(ds.SliceLocation)
                return float(getattr(ds, "InstanceNumber", 0) or 0) + (frame or 0)

            slices.sort(key=_key)
            ref = slices[0][0]

            # --- stack, keeping native resolution ------------------------------
            h, w = slices[0][1].shape[:2]
            same = all(s[1].shape[:2] == (h, w) for s in slices)
            if not same:                               # rare; resize to the modal shape
                shapes = [s[1].shape[:2] for s in slices]
                h, w = max(set(shapes), key=shapes.count)
                fixed = []
                for ds, a, f in slices:
                    if a.shape[:2] != (h, w):
                        a = cv2.resize(a.astype(np.float32), (w, h),
                                       interpolation=cv2.INTER_AREA)
                    fixed.append((ds, a, f))
                slices = fixed
                meta.setdefault("notes", []).append(
                    {"series": series, "note": "mixed slice sizes, unified"})

            vol = np.stack([s[1] for s in slices]).astype(np.float32)

            # rescale slope/intercept, so stored values mean what the header says
            slope = float(getattr(ref, "RescaleSlope", 1) or 1)
            inter = float(getattr(ref, "RescaleIntercept", 0) or 0)
            if slope != 1 or inter != 0:
                vol = vol * slope + inter

            # --- the one irreversible step: wide-window to uint8 ---------------
            pcts = np.percentile(vol, [0.1, 1, 50, 99, 99.9]).astype(float)
            lo, hi = float(np.percentile(vol, LO_PCT)), float(np.percentile(vol, HI_PCT))
            if hi <= lo:
                lo, hi = float(vol.min()), float(max(vol.max(), vol.min() + 1))
            u8 = np.clip((vol - lo) / (hi - lo), 0, 1)
            u8 = (u8 * 255.0 + 0.5).astype(np.uint8)
            arrays[series] = u8

            ipp0 = ref.get("ImagePositionPatient")
            ipp1 = slices[-1][0].get("ImagePositionPatient")
            ps = ref.get("PixelSpacing")
            meta["series"][series] = {
                "n_slices": int(u8.shape[0]),
                "rows": int(h), "cols": int(w),
                "pixel_spacing": [float(ps[0]), float(ps[1])] if ps is not None else None,
                "slice_thickness": float(ref.SliceThickness) if ref.get("SliceThickness") else None,
                "spacing_between": float(ref.SpacingBetweenSlices)
                                   if ref.get("SpacingBetweenSlices") else None,
                "orientation": [float(x) for x in ref.ImageOrientationPatient]
                               if ref.get("ImageOrientationPatient") else None,
                "pos_first": [float(x) for x in ipp0] if ipp0 else None,
                "pos_last": [float(x) for x in ipp1] if ipp1 else None,
                "window": {"lo": lo, "hi": hi, "lo_pct": LO_PCT, "hi_pct": HI_PCT},
                "percentiles": {"p0.1": pcts[0], "p1": pcts[1], "p50": pcts[2],
                                "p99": pcts[3], "p99.9": pcts[4],
                                "min": float(vol.min()), "max": float(vol.max())},
                "series_desc": str(ref.get("SeriesDescription", "")),
                "series_number": str(ref.get("SeriesNumber", "")),
                "manufacturer": str(ref.get("Manufacturer", "")),
                "model": str(ref.get("ManufacturerModelName", "")),
                "field_strength": float(ref.MagneticFieldStrength)
                                  if ref.get("MagneticFieldStrength") else None,
                "echo_time": float(ref.EchoTime) if ref.get("EchoTime") else None,
                "repetition_time": float(ref.RepetitionTime) if ref.get("RepetitionTime") else None,
            }

    buf = io.BytesIO()
    np.savez_compressed(buf, meta=np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8), **arrays)
    return study, buf.getvalue(), meta


# =============================================================================
# index
# =============================================================================
@app.function(image=image, volumes={DATA_MOUNT: data_vol}, timeout=3600, memory=4096)
def build_index(limit_studies: int = 0):
    import collections
    import zipfile

    import archive_parts as ap

    with ap.open_archive(DATA_MOUNT, COMP) as _fh, zipfile.ZipFile(_fh) as zf:
        names = [n for n in zf.namelist()
                 if n.startswith("train_series/") and n.endswith(".dcm")]
    by_study = collections.defaultdict(list)
    for n in names:
        by_study[n.split("/")[1]].append(n)
    studies = sorted(by_study)
    if limit_studies:
        studies = studies[:limit_studies]
    print(f"[index] {len(studies)} studies, "
          f"{sum(len(by_study[s]) for s in studies)} slices", flush=True)
    return [(s, sorted(by_study[s])) for s in studies]


# =============================================================================
# shard worker
# =============================================================================
# memory=8192: the first run requested 32 GiB per container but peaked at only
# ~7 GiB each (140 GiB across 20 containers), and memory was ~40% of its $2.10
# bill. 8 GiB leaves headroom over the observed peak without paying for 32.
@app.function(image=image, volumes={DATA_MOUNT: data_vol, SHARD_MOUNT: shard_vol},
              timeout=60 * 60 * 4, cpu=N_CORES, memory=8192,
              max_containers=MAX_CONTAINERS,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def preprocess_shard(shard):
    import time
    from concurrent.futures import ProcessPoolExecutor

    os.makedirs(SHARD_MOUNT, exist_ok=True)
    todo = [(s, m) for s, m in shard
            if not os.path.exists(f"{SHARD_MOUNT}/{s}.npz")]
    skipped = len(shard) - len(todo)
    t0, done, metas = time.time(), 0, []
    print(f"[shard] {len(todo)} studies to do, {skipped} already present", flush=True)

    if todo:
        with ProcessPoolExecutor(max_workers=N_CORES) as pool:
            for study, blob, meta in pool.map(_process_study, todo):
                with open(f"{SHARD_MOUNT}/{study}.npz", "wb") as fh:
                    fh.write(blob)
                metas.append(meta)
                done += 1
                if done % 10 == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"[shard] {done}/{len(todo)}  {el/60:.1f}m  "
                          f"{done/max(el,1e-9):.2f} studies/s  "
                          f"ETA {(len(todo)-done)/max(done/max(el,1e-9),1e-9)/60:.1f}m",
                          flush=True)
        shard_vol.commit()
    return {"done": done, "skipped": skipped,
            "bytes": sum(os.path.getsize(f"{SHARD_MOUNT}/{s}.npz")
                         for s, _ in shard if os.path.exists(f"{SHARD_MOUNT}/{s}.npz")),
            "metas": metas}


# =============================================================================
# verify
# =============================================================================
@app.function(image=image, volumes={SHARD_MOUNT: shard_vol}, timeout=3600, memory=2048)
def verify(sample: int = 20):
    import json as _json

    import numpy as np

    files = sorted(f for f in os.listdir(SHARD_MOUNT) if f.endswith(".npz"))
    total = sum(os.path.getsize(f"{SHARD_MOUNT}/{f}") for f in files)
    print(f"[verify] {len(files)} shards, {total/1024**3:.1f} GiB "
          f"({total/max(len(files),1)/1024**2:.1f} MiB per study)")

    n_series, n_slices = [], []
    for f in files[:sample]:
        with np.load(f"{SHARD_MOUNT}/{f}") as z:
            meta = _json.loads(bytes(z["meta"]).decode("utf-8"))
            keys = [k for k in z.files if k != "meta"]
            n_series.append(len(keys))
            for k in keys:
                n_slices.append(z[k].shape[0])
                assert z[k].dtype == np.uint8, f"{f}:{k} is {z[k].dtype}"
    print(f"[verify] sampled {sample}: series/study {np.mean(n_series):.1f}, "
          f"slices/series {np.mean(n_slices):.1f}, all uint8 OK")
    return {"shards": len(files), "gib": total / 1024**3}


@app.local_entrypoint()
def main(limit_studies: int = 0):
    import time

    t0 = time.time()
    index = build_index.remote(limit_studies)
    shards = [index[i:i + SHARD_SIZE] for i in range(0, len(index), SHARD_SIZE)]
    print(f"[launch] {len(index)} studies in {len(shards)} shards "
          f"x {N_CORES} cores, max {MAX_CONTAINERS} containers")

    done = skipped = nbytes = 0
    for i, res in enumerate(preprocess_shard.map(shards), 1):
        done += res["done"]; skipped += res["skipped"]; nbytes = max(nbytes, res["bytes"])
        el = time.time() - t0
        print(f"[launch] shard {i}/{len(shards)}  done={done} skipped={skipped}  "
              f"elapsed {el/60:.1f}m  ETA {(len(shards)-i)*el/i/60:.1f}m", flush=True)

    print(f"[launch] finished in {(time.time()-t0)/60:.1f}m")
    print(verify.remote())
