"""
modal_scan_patient_ids.py — collect study-level identity tags from the DICOMs.
=============================================================================

The splits are currently grouped by report text, which does not catch the case
that matters most for leakage: the SAME PATIENT appearing in more than one
study (both knees scanned, or a follow-up). Those are different
StudyInstanceUIDs with different reports, so text grouping misses them, yet the
two scans share anatomy and usually pathology. One in train and one in
validation inflates the validation score.

The full pixel scan did not record PatientID, so this pass fetches it. It reads
only the FIRST slice of each series, headers only (stop_before_pixels), so it
touches ~24k files instead of 819k and runs in a couple of minutes.

Also collected, because they answer related questions cheaply:
  * Laterality / body part      - is this the left or right knee?
  * StudyDate / AcquisitionDate - are two studies the same session or a follow-up?
  * FrameOfReferenceUID         - shared across series of one physical session
  * PatientSex, PatientAge      - for checking split balance
  * StudyDescription            - free sanity check on protocol

Usage (pick the workspace FIRST — see modal_setup_commands.txt):
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal.toml"
    modal run modal_scripts/modal_scan_patient_ids.py

Outputs (downloaded automatically to ./data/stats/):
    patient_ids.csv   one row per series
"""

import io
import os

import modal

COMP = "rsna-knee-abnormality-detection"
VOLUME_NAME = "knee-data"
MOUNT = "/data"
ARCHIVE = f"{MOUNT}/{COMP}.zip"
OUT_DIR = f"{MOUNT}/stats"
OUT_CSV = f"{OUT_DIR}/patient_ids.csv"

SHARD_SIZE = 400          # studies per container
MAX_CONTAINERS = 10

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pydicom==3.0.1", "pandas", "pyarrow")
)

app = modal.App("knee-scan-patient-ids")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=image, volumes={MOUNT: vol}, timeout=60 * 60, memory=16384)
def build_index():
    """One representative member per (study, series)."""
    import collections
    import zipfile

    with zipfile.ZipFile(ARCHIVE) as zf:
        names = [n for n in zf.namelist()
                 if n.startswith("train_series/") and n.endswith(".dcm")]
    first: dict[tuple[str, str], str] = {}
    for n in names:
        p = n.split("/")
        key = (p[1], p[2])
        if key not in first or n < first[key]:
            first[key] = n
    by_study = collections.defaultdict(list)
    for (study, series), member in first.items():
        by_study[study].append((series, member))
    print(f"[index] {len(by_study)} studies, {len(first)} series", flush=True)
    return sorted(by_study.items())


@app.function(image=image, volumes={MOUNT: vol}, timeout=60 * 60,
              cpu=4, memory=8192, max_containers=MAX_CONTAINERS)
def scan(shard):
    import time
    import zipfile

    import pandas as pd
    import pydicom

    def g(ds, tag):
        v = ds.get(tag, None)
        return None if v is None else str(v)

    rows, t0 = [], time.time()
    with zipfile.ZipFile(ARCHIVE) as zf:
        for study, series_list in shard:
            for series, member in series_list:
                rec = {"study": study, "series": series}
                try:
                    ds = pydicom.dcmread(io.BytesIO(zf.read(member)),
                                         stop_before_pixels=True)
                    rec.update({
                        "patient_id": g(ds, "PatientID"),
                        "patient_sex": g(ds, "PatientSex"),
                        "patient_age": g(ds, "PatientAge"),
                        "laterality": g(ds, "Laterality") or g(ds, "ImageLaterality"),
                        "body_part": g(ds, "BodyPartExamined"),
                        "study_date": g(ds, "StudyDate"),
                        "acq_date": g(ds, "AcquisitionDate"),
                        "study_desc": g(ds, "StudyDescription"),
                        "frame_of_ref": g(ds, "FrameOfReferenceUID"),
                        "manufacturer": g(ds, "Manufacturer"),
                        "model": g(ds, "ManufacturerModelName"),
                        "station": g(ds, "StationName"),
                        "institution": g(ds, "InstitutionName"),
                    })
                except Exception as exc:                  # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {exc}"[:120]
                rows.append(rec)
    print(f"[shard] {len(rows)} series in {time.time()-t0:.0f}s", flush=True)
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf, index=False)
    return buf.getvalue()


@app.function(image=image, volumes={MOUNT: vol}, timeout=60 * 30, memory=16384)
def collect(chunks):
    import pandas as pd

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.concat([pd.read_parquet(io.BytesIO(c)) for c in chunks], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    vol.commit()

    per_study = df.groupby("study")["patient_id"].nunique()
    pid = df.drop_duplicates("study")["patient_id"]
    print(f"[collect] {len(df)} series, {df['study'].nunique()} studies")
    print(f"[collect] distinct patient_id values: {pid.nunique()}")
    print(f"[collect] studies sharing a patient_id: "
          f"{int((pid.value_counts() > 1).sum())} patients cover "
          f"{int(pid.value_counts()[pid.value_counts() > 1].sum())} studies")
    print(f"[collect] studies whose series disagree on patient_id: "
          f"{int((per_study > 1).sum())}")
    print(f"[collect] laterality present: {df['laterality'].notna().mean():.1%}")
    return df.to_csv(index=False)


@app.local_entrypoint()
def main():
    from pathlib import Path

    index = build_index.remote()
    shards = [index[i:i + SHARD_SIZE] for i in range(0, len(index), SHARD_SIZE)]
    print(f"[launch] {len(shards)} shards")
    chunks = list(scan.map(shards))
    csv_text = collect.remote(chunks)

    dest = Path("data/stats")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "patient_ids.csv").write_text(csv_text, encoding="utf-8")
    print(f"[saved] {dest / 'patient_ids.csv'}")
