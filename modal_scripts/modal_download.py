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
LOCK = f"{MOUNT}/.download.lock"
DONE = f"{MOUNT}/.download.done"
CSV_DIR = f"{MOUNT}/csv"
CSVS = ["train.csv", "train_series.csv", "test.csv", "test_series.csv",
        "sample_submission.csv"]

CHUNK = 32 * 1024 * 1024          # 32 MiB per write
COMMIT_EVERY = 8 * 1024**3        # commit the volume every 8 GiB of progress
STALE_AFTER_S = 8 * 60       # archive untouched this long = the lock holder is dead. A
                             # live writer commits every 8 GiB, about 3.5 min, so 8 min
                             # is comfortably clear of a healthy run. It was 15, which
                             # was too patient: Modal preempts these workers ("Worker
                             # disappeared"), and each retry that arrived before the
                             # lock expired was REFUSED and burnt, so the download sat
                             # idle for 15-25 min per preemption.

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
              # 20, not 5: these workers get preempted repeatedly on some workspaces and
              # each preemption costs one retry, so 5 does not survive a 247 GiB pull. Every
              # retry resumes by HTTP Range from the committed size, so a retry is cheap.
              retries=modal.Retries(max_retries=20, backoff_coefficient=1.0,
                                    initial_delay=60.0))
def download_archive():
    """Stream the competition archive onto the volume, resuming if interrupted.

    Kaggle's download-all endpoint 302-redirects to a storage URL. We send
    `Range: bytes=<already_have>-` so a retry picks up where the last attempt
    stopped instead of re-pulling hundreds of gigabytes. If the server answers
    200 instead of 206 it ignored the Range header and we start over.
    """
    import time

    import requests

    # lock + done-sentinel: stops two --detach reruns from both appending to the
    # same file at once. That's what silently corrupted an earlier download - the
    # file shrank AFTER a run printed "done", which only happens if a second,
    # still-alive writer touched it afterwards. Refuse to start if a lock is
    # already there, and never resume into an archive already marked complete.
    if os.path.exists(DONE):
        print(f"[zip] {DONE} exists, archive already complete, refusing to touch it")
        return os.path.getsize(ARCHIVE)
    # The lock must be STALE-AWARE. A killed container never runs its finally block, so
    # a plain lock outlives the job holding it and then blocks every retry forever. That
    # is not hypothetical: a client disconnect cancelled one job at 136 GiB and the
    # orphaned lock killed all five of its retries. A live downloader always grows the
    # archive, so "lock present AND the file grew recently" is the real liveness test.
    vol.reload()
    if os.path.exists(LOCK):
        holder = Path(LOCK).read_text().strip()
        age = time.time() - os.path.getmtime(ARCHIVE) if os.path.exists(ARCHIVE) else 1e9
        if age < STALE_AFTER_S:
            raise RuntimeError(
                f"[zip] lock held by {holder} and the archive grew {age:.0f}s ago, so a "
                f"download really is running. Refusing to start a second writer."
            )
        print(f"[zip] stale lock ({holder}); archive untouched for {age / 60:.0f} min, "
              f"taking it over", flush=True)
    Path(LOCK).write_text(f"pid={os.getpid()} started={time.time()}")
    vol.commit()

    try:
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

        # cheap integrity guard: just parsing the central directory (seconds, not the
        # full-archive CRC pass that verify_deep does) catches a truncated/corrupt
        # download right here instead of letting it silently print "done".
        import zipfile
        with zipfile.ZipFile(ARCHIVE) as zf:
            n_members = len(zf.namelist())
        print(f"[zip] done: {size / 1024**3:.2f} GiB, {n_members} members at {ARCHIVE}", flush=True)

        Path(DONE).write_text(f"size={size} members={n_members} finished={time.time()}")
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)
        vol.commit()
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


@app.function(image=image, volumes={MOUNT: vol}, timeout=60 * 60 * 3, memory=8192, cpu=4.0)
def verify_deep(sample: int = 400, seed: int = 0):
    """CRC-check a random SAMPLE of members, which `verify` cannot do.

    verify() only reads the central directory at the tail of the file. That catches a
    truncated download, but not bytes corrupted in the MIDDLE - the exact risk after a
    download that crashed and resumed several times. Every zip member carries its own
    CRC-32, so decompressing one and comparing is a real integrity test.

    Sampling, not testzip(): a full check decompresses all 247 GiB. A few hundred members
    spread across the archive costs minutes and would catch any damage large enough to
    matter, since a resume fault corrupts a contiguous span, not one isolated file.
    """
    import random
    import zipfile

    if not os.path.exists(ARCHIVE):
        print(f"[deep] MISSING {ARCHIVE}")
        return {"ok": False}

    with zipfile.ZipFile(ARCHIVE) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        print(f"[deep] {len(infos)} members, {os.path.getsize(ARCHIVE) / 1024**3:.2f} GiB",
              flush=True)

        # sort by position in the file, then take an EVEN stride, so the sample spans the
        # whole archive instead of clustering wherever random happens to land
        infos.sort(key=lambda i: i.header_offset)
        step = max(1, len(infos) // max(1, sample))
        picked = infos[::step][:sample]
        rnd = random.Random(seed)
        picked += rnd.sample(infos, min(sample // 4, len(infos)))   # plus a random handful

        bad, done = [], 0
        for i in picked:
            try:
                with zf.open(i) as fh:                 # zipfile raises on a CRC mismatch
                    while fh.read(1 << 20):
                        pass
            except Exception as exc:
                bad.append((i.filename, f"{type(exc).__name__}: {exc}"))
            done += 1
            if done % 100 == 0:
                print(f"[deep] {done}/{len(picked)} checked, {len(bad)} bad", flush=True)

    if bad:
        print(f"[deep] FAILED: {len(bad)}/{len(picked)} members are corrupt")
        for n, e in bad[:10]:
            print(f"        {n}  {e}")
        return {"ok": False, "checked": len(picked), "bad": len(bad)}
    print(f"[deep] OK: {len(picked)} members spread across the archive all pass CRC")
    return {"ok": True, "checked": len(picked), "bad": 0}


@app.local_entrypoint()
def main(csvs_only: bool = False, skip_csvs: bool = False):
    print("[launch] modal")
    if not skip_csvs:
        download_csvs.remote()
    if not csvs_only:
        download_archive.remote()
