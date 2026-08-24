"""
modal_scan_dicoms.py — measure every training DICOM, without extracting anything.
================================================================================

Reads all ~819k slices straight out of the competition zip on the `knee-data`
volume and writes one row per slice: header fields, decoded-pixel statistics, and
the bounding box of non-background signal.

Why every slice rather than a sample: the duplicate-series tie-break needs an
exact slice count for every series, and decoding everything now proves the read
path works for all four transfer syntaxes before the real preprocessing pass
depends on it.

Parallelism is two-level:
  * across containers  — studies are split into shards, one shard per container
  * inside a container — a process pool, one worker per CPU core, each with its
    own zip handle (ZipFile objects are not safe to share)

Outputs, written to /data/stats/ on the volume AND downloaded to ./data/stats/:
    slice_stats.parquet  every slice, every column (compact)
    slice_stats.csv      same, plain CSV
    series_stats.csv     one row per series — what preprocessing consumes
    scan_summary.txt     human-readable summary of the whole scan

Usage (pick the workspace FIRST — see modal_setup_commands.txt):
    modal run modal_scripts/modal_scan_dicoms.py                    # everything
    modal run modal_scripts/modal_scan_dicoms.py --limit-studies 40 # smoke test
    modal run modal_scripts/modal_scan_dicoms.py::fetch_outputs     # re-download only
"""

import io
import os

import modal

COMP = "rsna-knee-abnormality-detection"
VOLUME_NAME = "knee-data"
MOUNT = "/data"
ARCHIVE = f"{MOUNT}/{COMP}.zip"

OUT_DIR = f"{MOUNT}/stats"
OUT_PARQUET = f"{OUT_DIR}/slice_stats.parquet"
OUT_SLICE_CSV = f"{OUT_DIR}/slice_stats.csv"
OUT_SERIES_CSV = f"{OUT_DIR}/series_stats.csv"
OUT_SUMMARY_TXT = f"{OUT_DIR}/scan_summary.txt"

SHARD_SIZE = 60          # studies per container
N_CORES = 8              # CPU cores requested per container == pool workers
MAX_CONTAINERS = 20      # concurrency cap: 20 x 8 = 160 cores peak, not 590
BG_PERCENTILE = 5        # background threshold = this percentile of the slice...
BG_MULT = 2.0            # ...times this; anything above counts as signal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pydicom==3.0.1", "pylibjpeg", "pylibjpeg-libjpeg",
                 "pylibjpeg-openjpeg", "numpy", "scipy", "pandas", "pyarrow")
)

app = modal.App("knee-scan-dicoms")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# =============================================================================
# foreground detection helpers
# =============================================================================
def _otsu(a, nbins=256):
    """Otsu's threshold: the split that minimises within-class intensity variance.

    Standard, parameter-free, and robust to the wildly different intensity scales
    across MRI series — which is exactly why a fixed or noise-derived threshold
    failed here.
    """
    import numpy as np

    lo, hi = float(a.min()), float(np.percentile(a, 99.5))
    if hi <= lo:
        return lo
    hist, edges = np.histogram(np.clip(a, lo, hi), bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    w = np.cumsum(hist)                       # weight of class 0 up to each bin
    centers = (edges[:-1] + edges[1:]) / 2
    m = np.cumsum(hist * centers)
    total_w, total_m = w[-1], m[-1]
    if total_w == 0:
        return lo
    w0 = w[:-1]
    w1 = total_w - w0
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return lo
    mu0 = np.where(w0 > 0, m[:-1] / np.maximum(w0, 1), 0)
    mu1 = np.where(w1 > 0, (total_m - m[:-1]) / np.maximum(w1, 1), 0)
    var_between = w0 * w1 * (mu0 - mu1) ** 2
    var_between[~valid] = -1
    return float(centers[int(np.argmax(var_between))])


def _largest_blob(mask):
    """Keep only the biggest connected region of a boolean mask."""
    import numpy as np
    from scipy import ndimage

    if not mask.any():
        return mask
    # close small holes/gaps first so the knee is one component, not many
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


# =============================================================================
# worker — module level so a process pool can pickle it
# =============================================================================
def _scan_studies(job):
    """job = (worker_id, n_workers, [(study, [members]), ...]) -> list of dicts.

    Runs in its own process with its own ZipFile handle. Prints a progress line
    every 20 seconds so the Modal log shows live throughput per worker.
    """
    import time
    import zipfile

    import numpy as np
    import pydicom

    wid, n_workers, studies = job
    total = sum(len(m) for _, m in studies)
    t0 = time.time()
    last = t0
    rows, done, n_err = [], 0, 0

    def _g(ds, tag, default=None):
        v = ds.get(tag, default)
        return v if v is None or isinstance(v, (int, float, str)) else str(v)

    print(f"[w{wid}/{n_workers}] start: {len(studies)} studies, {total} slices", flush=True)

    with zipfile.ZipFile(ARCHIVE) as zf:
        print(f"[w{wid}/{n_workers}] zip directory ready in {time.time() - t0:.1f}s", flush=True)
        for si, (study, members) in enumerate(studies, 1):
            for name in members:
                parts = name.split("/")
                rec = {"study": study, "series": parts[2],
                       "sop": parts[3][:-4], "member": name, "error": ""}
                try:
                    raw = zf.read(name)
                    rec["file_bytes"] = len(raw)
                    ds = pydicom.dcmread(io.BytesIO(raw))

                    rec.update({
                        "rows": int(getattr(ds, "Rows", 0)),
                        "cols": int(getattr(ds, "Columns", 0)),
                        "bits_stored": int(getattr(ds, "BitsStored", 0)),
                        "bits_alloc": int(getattr(ds, "BitsAllocated", 0)),
                        "pixel_repr": int(getattr(ds, "PixelRepresentation", 0)),
                        "photometric": _g(ds, "PhotometricInterpretation", ""),
                        "transfer_syntax": str(ds.file_meta.TransferSyntaxUID.name),
                        "manufacturer": _g(ds, "Manufacturer", ""),
                        "model": _g(ds, "ManufacturerModelName", ""),
                        "field_strength": _g(ds, "MagneticFieldStrength"),
                        "series_desc": _g(ds, "SeriesDescription", ""),
                        "series_number": _g(ds, "SeriesNumber"),
                        "instance_number": _g(ds, "InstanceNumber"),
                        "slice_thickness": _g(ds, "SliceThickness"),
                        "spacing_between": _g(ds, "SpacingBetweenSlices"),
                        "slice_location": _g(ds, "SliceLocation"),
                        "echo_time": _g(ds, "EchoTime"),
                        "repetition_time": _g(ds, "RepetitionTime"),
                        "scanning_seq": _g(ds, "ScanningSequence", ""),
                        "scan_options": _g(ds, "ScanOptions", ""),
                        "patient_sex": _g(ds, "PatientSex", ""),
                        "body_part": _g(ds, "BodyPartExamined", ""),
                        "rescale_slope": _g(ds, "RescaleSlope"),
                        "rescale_intercept": _g(ds, "RescaleIntercept"),
                        "window_center": _g(ds, "WindowCenter"),
                        "window_width": _g(ds, "WindowWidth"),
                        "n_frames": int(getattr(ds, "NumberOfFrames", 1) or 1),
                    })
                    ps = ds.get("PixelSpacing")
                    if ps is not None and len(ps) == 2:
                        rec["pixel_spacing_r"] = float(ps[0])
                        rec["pixel_spacing_c"] = float(ps[1])
                    ipp = ds.get("ImagePositionPatient")
                    if ipp is not None and len(ipp) == 3:
                        rec["ipp_x"], rec["ipp_y"], rec["ipp_z"] = map(float, ipp)

                    # --- pixels ---
                    a = ds.pixel_array
                    if a.ndim == 3:                       # multi-frame: use middle
                        a = a[a.shape[0] // 2]
                    a = a.astype(np.float32)
                    rec.update({"px_min": float(a.min()), "px_max": float(a.max()),
                                "px_mean": float(a.mean()), "px_std": float(a.std())})
                    for p in (1, 5, 50, 95, 99, 99.9):
                        rec[f"px_p{p}"] = float(np.percentile(a, p))

                    # --- foreground: where is the anatomy in the frame? ---
                    # Otsu's method picks the threshold that best separates the
                    # intensity histogram into two groups (air vs tissue). Then we
                    # keep only the LARGEST connected blob, which discards coil
                    # artefacts and stray bright specks — the failure mode of a
                    # simple noise-floor threshold, which collapsed the box to a
                    # few pixels on some series.
                    rec["fg_ok"] = 0
                    thr = _otsu(a)
                    rec["bg_thr"] = float(thr)
                    fg = a > thr
                    rec["fg_frac_raw"] = float(fg.mean())
                    fg = _largest_blob(fg)
                    rec["fg_frac"] = float(fg.mean())
                    if fg.any():
                        ys, xs = np.where(fg)
                        rec["bbox_r0"], rec["bbox_r1"] = int(ys.min()), int(ys.max())
                        rec["bbox_c0"], rec["bbox_c1"] = int(xs.min()), int(xs.max())
                        rec["tbox_r0"], rec["tbox_r1"] = rec["bbox_r0"], rec["bbox_r1"]
                        rec["tbox_c0"], rec["tbox_c1"] = rec["bbox_c0"], rec["bbox_c1"]
                        # sanity: a real knee spans >30 mm and >15% of the frame
                        span_mm = (rec["bbox_c1"] - rec["bbox_c0"] + 1) * \
                                  rec.get("pixel_spacing_c", 0.3)
                        rec["fg_span_mm"] = float(span_mm)
                        rec["fg_ok"] = int(span_mm > 30 and rec["fg_frac"] > 0.05)
                except Exception as exc:                  # keep going; record the failure
                    rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
                    n_err += 1
                    if n_err <= 3:
                        print(f"[w{wid}] DECODE FAIL {name[-44:]} -> {rec['error']}", flush=True)
                rows.append(rec)
                done += 1

            now = time.time()
            if now - last > 20 or si == len(studies):
                el = now - t0
                rate = done / max(el, 1e-9)
                print(f"[w{wid}/{n_workers}] study {si:3d}/{len(studies)}  "
                      f"slice {done:6d}/{total:6d} ({done / max(total, 1):5.1%})  "
                      f"elapsed {el / 60:5.1f}m  {rate:6.1f} sl/s  "
                      f"ETA {(total - done) / max(rate, 1e-9) / 60:5.1f}m  err {n_err}",
                      flush=True)
                last = now

    el = time.time() - t0
    print(f"[w{wid}/{n_workers}] DONE {done} slices in {el / 60:.1f}m "
          f"({done / max(el, 1e-9):.1f} sl/s), {n_err} errors", flush=True)
    return rows


# =============================================================================
# index: which zip members belong to which study
# =============================================================================
@app.function(image=image, volumes={MOUNT: vol}, timeout=60 * 60, memory=16384)
def build_index(limit_studies: int = 0):
    import collections
    import time
    import zipfile

    t0 = time.time()
    print(f"[index] opening {ARCHIVE} ...", flush=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        names = [n for n in zf.namelist()
                 if n.startswith("train_series/") and n.endswith(".dcm")]
    print(f"[index] central directory read in {time.time() - t0:.1f}s", flush=True)

    by_study = collections.defaultdict(list)
    for n in names:
        by_study[n.split("/")[1]].append(n)

    studies = sorted(by_study)
    if limit_studies:
        studies = studies[:limit_studies]
    n_sl = sum(len(by_study[s]) for s in studies)
    print(f"[index] {len(studies)} studies, {n_sl} slices "
          f"({n_sl / max(len(studies), 1):.1f} per study)", flush=True)
    return [(s, by_study[s]) for s in studies]


# =============================================================================
# shard: one container, N_CORES processes
# =============================================================================
@app.function(image=image, volumes={MOUNT: vol}, timeout=60 * 60 * 4,
              cpu=N_CORES, memory=16384, max_containers=MAX_CONTAINERS,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def scan_shard(shard):
    import time
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    t0 = time.time()
    n_slices = sum(len(m) for _, m in shard)
    tag = shard[0][0][-10:] if shard else "?"
    print(f"\n{'=' * 70}\n[shard {tag}] {len(shard)} studies / {n_slices} slices "
          f"across {N_CORES} cores\n{'=' * 70}", flush=True)

    # round-robin studies across workers so each gets a similar slice count
    jobs = [(w, N_CORES, shard[w::N_CORES]) for w in range(N_CORES)]
    jobs = [j for j in jobs if j[2]]

    rows = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for part in pool.map(_scan_studies, jobs):
            rows.extend(part)

    el = time.time() - t0
    n_err = sum(1 for r in rows if r["error"])
    print(f"[shard {tag}] COMPLETE {len(rows)} slices in {el / 60:.1f}m "
          f"({len(rows) / max(el, 1e-9):.1f} sl/s aggregate), {n_err} errors", flush=True)

    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf, index=False)
    return buf.getvalue()


# =============================================================================
# collect: per-slice + per-series tables and the summary
# =============================================================================
@app.function(image=image, volumes={MOUNT: vol}, timeout=60 * 60 * 2, memory=32768)
def collect(chunks):
    import time

    import pandas as pd

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[collect] concatenating {len(chunks)} shards ...", flush=True)
    df = pd.concat([pd.read_parquet(io.BytesIO(c)) for c in chunks], ignore_index=True)
    print(f"[collect] {len(df):,} rows in {time.time() - t0:.1f}s; writing parquet + csv",
          flush=True)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_SLICE_CSV, index=False)
    vol.commit()
    print(f"[collect] slice tables written ({time.time() - t0:.1f}s)", flush=True)

    ok = df[df["error"] == ""]
    bad = df[df["error"] != ""]

    # --- per-series table -----------------------------------------------------
    def _mode(s):
        m = s.mode()
        return m.iloc[0] if len(m) else None

    g = ok.groupby(["study", "series"])
    series = pd.DataFrame({
        "n_slices":        g.size(),
        "rows":            g["rows"].agg(_mode),
        "cols":            g["cols"].agg(_mode),
        "size_varies":     g.apply(lambda d: d["rows"].nunique() > 1 or d["cols"].nunique() > 1),
        "pixel_spacing_r": g["pixel_spacing_r"].median(),
        "pixel_spacing_c": g["pixel_spacing_c"].median(),
        "slice_thickness": g["slice_thickness"].median(),
        "spacing_between": g["spacing_between"].median(),
        "transfer_syntax": g["transfer_syntax"].agg(_mode),
        "manufacturer":    g["manufacturer"].agg(_mode),
        "model":           g["model"].agg(_mode),
        "field_strength":  g["field_strength"].median(),
        "series_desc":     g["series_desc"].agg(_mode),
        "series_number":   g["series_number"].agg(_mode),
        "echo_time":       g["echo_time"].median(),
        "repetition_time": g["repetition_time"].median(),
        "patient_sex":     g["patient_sex"].agg(_mode),
        "px_p1":           g["px_p1"].median(),
        "px_p50":          g["px_p50"].median(),
        "px_p99":          g["px_p99"].median(),
        "px_max":          g["px_max"].max(),
        "fg_frac":         g["fg_frac"].median(),
        "fg_ok_frac":      g["fg_ok"].mean(),
        "fg_span_mm":      g["fg_span_mm"].median(),
        # bounding box over the slices whose foreground passed the sanity check
        "tbox_r0":         g.apply(lambda d: d.loc[d.fg_ok == 1, "bbox_r0"].min()),
        "tbox_r1":         g.apply(lambda d: d.loc[d.fg_ok == 1, "bbox_r1"].max()),
        "tbox_c0":         g.apply(lambda d: d.loc[d.fg_ok == 1, "bbox_c0"].min()),
        "tbox_c1":         g.apply(lambda d: d.loc[d.fg_ok == 1, "bbox_c1"].max()),
        "total_bytes":     g["file_bytes"].sum(),
    }).reset_index()
    series["crop_h"] = series["tbox_r1"] - series["tbox_r0"] + 1
    series["crop_w"] = series["tbox_c1"] - series["tbox_c0"] + 1
    series["crop_frac"] = (series["crop_h"] * series["crop_w"]) / (series["rows"] * series["cols"])
    series.to_csv(OUT_SERIES_CSV, index=False)
    vol.commit()

    # --- summary --------------------------------------------------------------
    L = []
    def w(line=""):
        L.append(str(line))
        print(line, flush=True)

    w("=" * 78)
    w("RSNA KNEE — FULL DICOM SCAN SUMMARY")
    w("=" * 78)
    w(f"slices scanned      : {len(df):,}")
    w(f"studies             : {df['study'].nunique():,}")
    w(f"series              : {df['series'].nunique():,}")
    w(f"decode failures     : {len(bad):,}  ({len(bad) / max(len(df), 1):.3%})")
    if len(bad):
        w("\nfailure reasons:")
        w(bad["error"].value_counts().head(10).to_string())

    w("\n--- transfer syntaxes (compression formats) ---")
    w(df["transfer_syntax"].value_counts().to_string())

    w("\n--- image size (rows x cols), per slice ---")
    sizes = ok["rows"].astype(str) + "x" + ok["cols"].astype(str)
    w(sizes.value_counts().head(15).to_string())
    w(f"\ndistinct sizes: {sizes.nunique()}")
    w(f"series whose slices are NOT all the same size: {int(series['size_varies'].sum())}")

    w("\n--- slices per series ---")
    w(series["n_slices"].describe(percentiles=[.01, .05, .5, .95, .99]).round(2).to_string())
    w("\nhistogram:")
    w(pd.cut(series["n_slices"], bins=[0, 20, 30, 40, 50, 80, 150, 10**6])
        .value_counts().sort_index().to_string())

    w("\n--- geometry ---")
    for c in ["pixel_spacing_r", "slice_thickness", "spacing_between", "field_strength"]:
        w(f"{c:18s} {series[c].describe(percentiles=[.05, .5, .95]).round(3).to_dict()}")

    w("\n--- intensity (per-series medians) ---")
    for c in ["px_p1", "px_p50", "px_p99", "px_max"]:
        w(f"{c:8s} {series[c].describe(percentiles=[.05, .5, .95]).round(1).to_dict()}")
    w("\nNOTE: MRI has no absolute intensity scale — the spread across series is why")
    w("      per-volume percentile normalization is needed, not a fixed window.")

    w("\n--- foreground detection (Otsu + largest blob) ---")
    w(f"slices passing the sanity check (>30mm span, >5% of frame): "
      f"{ok['fg_ok'].mean():.3%}")
    w(f"series with <90% of slices passing: {int((series['fg_ok_frac'] < 0.9).sum())} "
      f"of {len(series)}")
    w(f"median foreground fraction per slice : {ok['fg_frac'].median():.3f}")
    w("\ndetected anatomy width in mm (should look like a knee, ~90-160mm):")
    w(ok["fg_span_mm"].describe(percentiles=[.01, .05, .5, .95, .99]).round(1).to_string())

    w("\n--- cropping ---")
    w(f"median crop area as fraction of frame: {series['crop_frac'].median():.3f}")
    w(f"  -> cropping to anatomy saves about {1 - series['crop_frac'].median():.1%} of pixels")
    w("crop_h / crop_w percentiles:")
    w(series[["crop_h", "crop_w"]].describe(percentiles=[.05, .5, .95, .99]).round(1).to_string())

    w("\n--- physical scale (the reason cropping matters) ---")
    series["fov_w_mm"] = series["cols"] * series["pixel_spacing_c"]
    series["anat_w_mm"] = series["crop_w"] * series["pixel_spacing_c"]
    w("field of view width, mm:")
    w(series["fov_w_mm"].describe(percentiles=[.01, .05, .5, .95, .99]).round(1).to_string())
    w("\nanatomy width, mm:")
    w(series["anat_w_mm"].describe(percentiles=[.01, .05, .5, .95, .99]).round(1).to_string())
    apparent = 384 * series["crop_w"] / series["cols"]
    w(f"\nif every frame is naively resized to 384px, the anatomy spans "
      f"{apparent.quantile(.05):.0f}-{apparent.quantile(.95):.0f} px "
      f"({apparent.quantile(.95) / max(apparent.quantile(.05), 1):.2f}x variation)")
    w("-> argues for resampling to a fixed mm/px instead of resizing whole frames")
    w("\npixel spacing (mm/px) distribution:")
    w(series["pixel_spacing_c"].describe(percentiles=[.01, .05, .5, .95, .99]).round(3).to_string())

    w("\n--- scanners ---")
    w(series["manufacturer"].value_counts().head(10).to_string())
    w("\ntop models:")
    w(series["model"].value_counts().head(10).to_string())

    w("\n--- storage estimate for preprocessed shards ---")
    px = (series["crop_h"] * series["crop_w"] * series["n_slices"]).sum()
    w(f"total cropped pixels      : {px:,.0f}")
    w(f"  as uint8, uncompressed  : {px / 1024**3:,.1f} GiB")
    w(f"  assuming ~2.5x lossless : {px / 2.5 / 1024**3:,.1f} GiB")

    w("\n--- duplicate series ---")
    per_study = series.groupby("study").size()
    w(f"series per study: mean {per_study.mean():.2f}, max {per_study.max()}")
    w("(tie-break candidates: n_slices, rows*cols, series_number — decided in the notebook)")
    w("=" * 78)
    w(f"collect took {(time.time() - t0) / 60:.1f}m")

    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    vol.commit()

    return {"slices": len(df), "studies": int(df["study"].nunique()),
            "series": int(df["series"].nunique()), "failures": len(bad)}


# =============================================================================
# fetch: hand the artifacts back for local download
# =============================================================================
@app.function(image=image, volumes={MOUNT: vol}, timeout=60 * 30, memory=8192)
def fetch_outputs():
    out = {}
    for fn in sorted(os.listdir(OUT_DIR)):
        path = os.path.join(OUT_DIR, fn)
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                out[fn] = fh.read()
            print(f"[fetch] {fn}  {os.path.getsize(path) / 1024**2:.1f} MiB", flush=True)
    return out


@app.local_entrypoint()
def main(limit_studies: int = 0, local_dir: str = "data/stats"):
    import time
    from pathlib import Path

    t0 = time.time()
    print("[launch] modal")
    index = build_index.remote(limit_studies)
    shards = [index[i:i + SHARD_SIZE] for i in range(0, len(index), SHARD_SIZE)]
    n_slices = sum(len(m) for _, ms in index for m in [ms])
    print(f"[launch] {len(index)} studies, {n_slices} slices, "
          f"{len(shards)} shards x <={SHARD_SIZE} studies x {N_CORES} cores")

    chunks = []
    for i, blob in enumerate(scan_shard.map(shards), 1):
        chunks.append(blob)
        el = time.time() - t0
        eta = el / i * (len(shards) - i)
        print(f"[launch] shard {i:3d}/{len(shards)} back  ({i / len(shards):5.1%})  "
              f"elapsed {el / 60:5.1f}m  ETA {eta / 60:5.1f}m", flush=True)

    print(f"[launch] all shards in after {(time.time() - t0) / 60:.1f}m; collecting")
    print(collect.remote(chunks))

    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for fn, blob in fetch_outputs.remote().items():
        (dest / fn).write_bytes(blob)
        print(f"[saved] {dest / fn}  ({len(blob) / 1024**2:.1f} MiB)")
    print(f"[launch] TOTAL {(time.time() - t0) / 60:.1f}m")
