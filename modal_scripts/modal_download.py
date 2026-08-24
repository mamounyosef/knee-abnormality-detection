"""
modal_download.py — pull the RSNA Knee competition data onto a Modal volume.
===========================================================================

Downloads the whole competition (~570 GB) as ONE zip archive straight onto the
`knee-data` volume, resuming with HTTP Range requests if the connection dies.
The archive is NOT extracted: 820k tiny files perform badly on a network volume,
and training will read from compact preprocessed shards anyway (built later, in
modal_scripts/modal_preprocess.py, once EDA tells us the target resolution).

Kaggle credentials are read from the local ~/.kaggle/kaggle.json at launch time
and injected as an ephemeral secret, so nothing needs pre-creating in each of
the Modal workspace.

Usage (pick the workspace FIRST — see modal_setup_commands.txt):
    modal run --detach modal_scripts/modal_download.py               # csvs + full zip
    modal run modal_scripts/modal_download.py --csvs-only            # quick smoke test
    modal run modal_scripts/modal_download.py::verify                # inspect the volume
"""

import json
import os
from pathlib import Path

import modal

COMP = "rsna-knee-abnormality-detection"
VOLUME_NAME = "knee-data"
MOUNT = "/data"
ARCHIVE = f"{MOUNT}/{COMP}.zip"
CSV_DIR = f"{MOUNT}/csv"
CSVS = ["train.csv", "train_series.csv", "test.csv", "test_series.csv",
        "sample_submission.csv"]

CHUNK = 32 * 1024 * 1024          # 32 MiB per write
COMMIT_EVERY = 8 * 1024**3        # commit the volume every 8 GiB of progress

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests", "kaggle", "tqdm")
)

app = modal.App("knee-download")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _kaggle_secret() -> modal.Secret:
    """Ephemeral secret built from ~/.kaggle/kaggle.json.

    This module is imported both locally (to define the app) and inside the
    container (to run the functions), so the local-only file read is guarded —
    remotely the credentials already live in the environment.
    """
    if not modal.is_local():
        return modal.Secret.from_dict({})
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    return modal.Secret.from_dict({"KAGGLE_USERNAME": creds["username"],
                                   "KAGGLE_KEY": creds["key"]})


KAGGLE_SECRET = _kaggle_secret()


# -----------------------------------------------------------------------------
# small files: the five CSVs
# -----------------------------------------------------------------------------
@app.function(image=image, volumes={MOUNT: vol}, secrets=[KAGGLE_SECRET],
              timeout=60 * 30)
def download_csvs():
    """Fetch the five metadata CSVs into /data/csv/ (a few MB total)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    os.makedirs(CSV_DIR, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    for name in CSVS:
        api.competition_download_file(COMP, name, path=CSV_DIR, force=True, quiet=True)
    vol.commit()
    got = sorted(os.listdir(CSV_DIR))
    print(f"[csv] {len(got)} files in {CSV_DIR}: {got}")
    return got


# -----------------------------------------------------------------------------
# the big one: ~570 GB zip, resumable
# -----------------------------------------------------------------------------
@app.function(image=image, volumes={MOUNT: vol}, secrets=[KAGGLE_SECRET],
              timeout=60 * 60 * 24, cpu=2.0, memory=4096,
              retries=modal.Retries(max_retries=5, backoff_coefficient=1.0,
                                    initial_delay=60.0))
def download_archive():
    """Stream the competition archive onto the volume, resuming if interrupted.

    Kaggle's download-all endpoint 302-redirects to a storage URL. We send
    `Range: bytes=<already_have>-` so a retry picks up where the last attempt
    stopped instead of re-pulling hundreds of gigabytes. If the server answers
    200 instead of 206 it ignored the Range header and we start over.
    """
    import requests

    url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMP}"
    auth = (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])

    have = os.path.getsize(ARCHIVE) if os.path.exists(ARCHIVE) else 0
    print(f"[zip] starting from {have / 1024**3:.2f} GiB", flush=True)

    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, auth=auth, headers=headers, stream=True,
                      timeout=(30, 300), allow_redirects=True) as r:
        r.raise_for_status()
        if have and r.status_code != 206:
            print(f"[zip] server ignored Range (status {r.status_code}); restarting", flush=True)
            have = 0
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have else "wb"
        print(f"[zip] status={r.status_code} total={total / 1024**3:.2f} GiB", flush=True)

        since_commit = 0
        with open(ARCHIVE, mode) as fh:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                have += len(chunk)
                since_commit += len(chunk)
                if since_commit >= COMMIT_EVERY:
                    fh.flush()
                    os.fsync(fh.fileno())
                    vol.commit()
                    since_commit = 0
                    pct = f"{100 * have / total:.1f}%" if total else "?"
                    print(f"[zip] {have / 1024**3:9.2f} GiB  ({pct})", flush=True)

    vol.commit()
    size = os.path.getsize(ARCHIVE)
    print(f"[zip] done: {size / 1024**3:.2f} GiB at {ARCHIVE}", flush=True)
    return size


# -----------------------------------------------------------------------------
# verification: is the archive complete and readable?
# -----------------------------------------------------------------------------
@app.function(image=image, volumes={MOUNT: vol}, timeout=60 * 60, memory=8192)
def verify():
    """Open the zip's central directory and report member counts by top folder."""
    import collections
    import zipfile

    csvs = sorted(os.listdir(CSV_DIR)) if os.path.exists(CSV_DIR) else []
    print(f"[verify] csv dir: {csvs}")

    if not os.path.exists(ARCHIVE):
        print(f"[verify] MISSING {ARCHIVE}")
        return {"ok": False, "csvs": csvs}

    size = os.path.getsize(ARCHIVE)
    print(f"[verify] {ARCHIVE}  {size / 1024**3:.2f} GiB", flush=True)
    try:
        with zipfile.ZipFile(ARCHIVE) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile as exc:
        print(f"[verify] BAD ZIP (truncated or still downloading): {exc}")
        return {"ok": False, "bytes": size, "csvs": csvs}

    tops = collections.Counter(n.split("/")[0] for n in names)
    studies = len({n.split("/")[1] for n in names if n.startswith("train_series/")})
    print(f"[verify] members={len(names)}  top-level={dict(tops)}  train studies={studies}")
    return {"ok": True, "bytes": size, "members": len(names), "studies": studies}


@app.local_entrypoint()
def main(csvs_only: bool = False, skip_csvs: bool = False):
    print("[launch] modal")
    if not skip_csvs:
        download_csvs.remote()
    if not csvs_only:
        download_archive.remote()
