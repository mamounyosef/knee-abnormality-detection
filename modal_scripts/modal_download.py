"""
modal_download.py — pull the RSNA Knee competition data onto a Modal volume.
===========================================================================

Downloads the whole competition archive onto the `knee-data` volume as 2 GiB PART FILES,
resuming with bounded HTTP Range requests if the connection dies. The archive is NOT
extracted: 820k tiny files perform badly on a network volume, and training reads compact
preprocessed shards anyway (modal_scripts/modal_preprocess.py).

It is split into parts because BOTH failure modes of a single 247 GiB file grow with the
file, so a download that looks healthy for hours collapses near the end:

  * `vol.commit()` re-hashes the whole file to find changed blocks. Measured on a 233 GiB
    archive, one commit ran over 5 MINUTES without returning, and a container stuck in
    commit cannot answer Modal's cancellation signal, so it gets killed.
  * Kaggle stalls an open-ended `Range: bytes=<have>-` covering many GiB: one 32 MiB chunk
    in 1.0 s, then 0.06 GiB in 319 s. The same bytes as a BOUNDED range run at 14.6 MB/s.

Nothing is ever concatenated - archive_parts.open_archive stitches the parts into a
seekable stream that zipfile accepts. Volumes holding a pre-split single `<comp>.zip` keep
working, and a partial one is kept as the BASE with new parts continuing after it.

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
# The part size itself lives in archive_parts.PART_SIZE, because the reader has to agree
# with the writer about it. Both failure modes that made the old single-file download
# unworkable scale with the archive, which is why splitting it is the fix and not a tuning
# knob: see the header of archive_parts.py for the measurements.
STALE_AFTER_S = 90           # no part written for this long = the lock holder is dead. A
                             # live writer finishes a part every minute or two, so 90 s
                             # clears a dead lock fast without ever racing a healthy one.
                             # It was 8 min: a killed container left a lock whose archive
                             # mtime still looked fresh, and the next relaunch refused to
                             # start as "a download really is running".

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests", "kaggle", "tqdm")
    # archive_parts defines the part layout AND the reader that stitches parts back into a
    # zipfile-compatible stream. Modal no longer auto-mounts local modules, so without this
    # the container dies on `import archive_parts` the moment it starts.
    .add_local_python_source("archive_parts")
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
              # 10, Modal's maximum, not the old 5: these workers get preempted repeatedly
              # on some workspaces and each preemption costs one retry. Every retry resumes
              # by HTTP Range from the committed size, so a retry is cheap.
              retries=modal.Retries(max_retries=10, backoff_coefficient=1.0,
                                    initial_delay=60.0))
def download_archive():
    """Fetch the competition archive onto the volume as PART FILES, resuming if interrupted.

    Not one big file. See archive_parts.py for the full reasoning; the short version is that
    both failure modes of the single-file design are unbounded in the archive's size:

      * `vol.commit()` re-hashes the whole file, so it grows from instant to over 5 minutes
        as the download proceeds, and a container stuck in commit cannot answer Modal's
        cancellation signal and gets killed.
      * Kaggle stalls an open-ended `Range: bytes=<have>-` that spans many GiB, delivering
        one chunk and then nothing, while the same bytes as a BOUNDED range run at full speed.

    So: one bounded request per 2 GiB part, one commit per part, and the parts are stitched
    back together at read time by archive_parts.open_archive rather than concatenated.
    """
    import time

    import requests

    import archive_parts as ap

    # lock + done-sentinel: stops two --detach reruns from both writing parts at once.
    # That's what silently corrupted an earlier download - the file shrank AFTER a run
    # printed "done", which only happens if a second, still-alive writer touched it
    # afterwards. Refuse to start if a live lock is there, and never resume into an
    # archive already marked complete.
    if os.path.exists(DONE):
        print(f"[zip] {DONE} exists, archive already complete, refusing to touch it")
        return ap.archive_size(MOUNT, COMP)

    vol.reload()
    pdir = ap.parts_dir(MOUNT, COMP)
    os.makedirs(pdir, exist_ok=True)

    def newest_part_mtime():
        """When a part was last written. This is the liveness signal for the lock.

        The old code watched the single archive's mtime. With parts there is no one file to
        watch, so take the newest part's mtime and treat "no parts yet" as infinitely old.
        """
        parts = ap.list_parts(MOUNT, COMP)
        if not parts:
            return 0.0
        return max(os.path.getmtime(p) for _, p, _ in parts)

    # The lock must be STALE-AWARE. A killed container never runs its finally block, so a
    # plain lock outlives the job holding it and then blocks every retry forever. That is
    # not hypothetical: a client disconnect cancelled one job at 136 GiB and the orphaned
    # lock killed all five of its retries. A live downloader always writes new parts, so
    # "lock present AND a part was written recently" is the real liveness test.
    if os.path.exists(LOCK):
        holder = Path(LOCK).read_text().strip()
        mt = newest_part_mtime()
        age = time.time() - mt if mt else 1e9
        if age < STALE_AFTER_S:
            raise RuntimeError(
                f"[zip] lock held by {holder} and a part was written {age:.0f}s ago, so a "
                f"download really is running. Refusing to start a second writer."
            )
        when = "no part has ever been written" if not mt else f"no part written for {age / 60:.0f} min"
        print(f"[zip] stale lock ({holder}); {when}, taking it over", flush=True)
    Path(LOCK).write_text(f"pid={os.getpid()} started={time.time()}")
    vol.commit()

    t0 = time.time()

    def stamp(msg):
        print(f"[zip] +{time.time() - t0:7.1f}s  {msg}", flush=True)

    try:
        url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMP}"
        auth = (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])

        # A volume that started under the OLD single-file downloader keeps that archive as
        # the BASE and new parts continue after it, so 232 GiB of existing work is not
        # thrown away and never has to be copied.
        base = ap.base_size(MOUNT, COMP)
        if base:
            stamp(f"existing single archive kept as the base: {base / 1024**3:.2f} GiB")

        # Resume from the trustworthy prefix only. contiguous_parts stops at the first gap
        # or short middle part, so an interrupted part is refetched rather than stitched over.
        good = ap.contiguous_parts(MOUNT, COMP)
        # A short part is only valid as the LAST part of a complete archive; mid-download it
        # means "interrupted", so drop it and refetch that index.
        if good and good[-1][2] != ap.PART_SIZE:
            stamp(f"part {good[-1][0]} is short ({good[-1][2]} B), refetching it")
            good = good[:-1]
        next_index = len(good)
        have = base + sum(s for _, _, s in good)
        stamp(f"resuming at part {next_index}, {have / 1024**3:.2f} GiB held")

        total = 0
        stalls = 0
        start_bytes = have
        commit_s = 0.0

        def fetch_part(index, lo, hi):
            """Pull [lo, hi] inclusive into part `index`. Returns (bytes, archive_total).

            Written to a .tmp name and renamed only once complete, so a container killed
            mid-write can never leave a short part that looks finished.
            """
            hdr = {"Range": f"bytes={lo}-{hi}"}
            # read timeout 60 s: a window that goes quiet is a stall, and we want it to raise
            # quickly so the retry re-requests instead of sitting on a dead socket.
            with requests.get(url, auth=auth, headers=hdr, stream=True,
                              timeout=(30, 60), allow_redirects=True) as r:
                r.raise_for_status()
                if r.status_code != 206:
                    raise RuntimeError(f"expected 206 for a bounded range, got {r.status_code}")
                cr = r.headers.get("Content-Range", "")
                tot = int(cr.rsplit("/", 1)[1]) if "/" in cr else 0
                dst = ap.part_path(MOUNT, COMP, index)
                tmp = dst + ".tmp"
                n = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if chunk:
                            fh.write(chunk)
                            n += len(chunk)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, dst)
                return n, tot

        while True:
            # part i covers [base + i*PART_SIZE, ...), so the byte range and the part index
            # stay consistent whether or not this volume has a pre-split base archive
            lo = base + next_index * ap.PART_SIZE
            hi = lo + ap.PART_SIZE - 1
            if total:
                if lo >= total:
                    break
                hi = min(hi, total - 1)
            try:
                n, tot = fetch_part(next_index, lo, hi)
            except Exception as exc:
                stalls += 1
                stamp(f"part {next_index} failed ({type(exc).__name__}: "
                      f"{str(exc)[:120]}), retry {stalls}")
                if stalls > 40:
                    raise
                time.sleep(min(3 * stalls, 30))
                continue
            if tot and not total:
                total = tot
                # parts still to fetch, which is what matters when a base archive means
                # most of the download is already done
                left = -(-(total - base) // ap.PART_SIZE)
                stamp(f"archive total is {total / 1024**3:.2f} GiB; "
                      f"{(total - have) / 1024**3:.2f} GiB still to fetch in {left} parts")
            if n == 0:
                stamp("empty part returned; treating as end of archive")
                break

            # One commit per part. This is the whole point of the split: only the new part
            # is dirty, so the cost stays flat instead of growing with the archive.
            _c = time.time()
            vol.commit()
            _c = time.time() - _c
            commit_s += _c

            have += n
            next_index += 1
            el = time.time() - t0
            got = have - start_bytes
            pct = f"{100 * have / total:.1f}%" if total else "?"
            stamp(f"{have / 1024**3:9.2f} GiB ({pct})  part {next_index - 1} done  "
                  f"+{got / 1024**3:.2f} GiB this run at {got / el / 1024**2:5.1f} MB/s  "
                  f"[commit {_c:.1f}s, {commit_s:.0f}s total, {stalls} retries]")

            if total and have >= total:
                break

        vol.commit()
        size = ap.parts_bytes(MOUNT, COMP)
        if total and size != total:
            raise RuntimeError(f"[zip] finished with {size} bytes but the server says "
                               f"{total}; refusing to mark this complete")

        # cheap integrity guard: just parsing the central directory (seconds, not the
        # full-archive CRC pass that verify_deep does) catches a truncated/corrupt
        # download right here instead of letting it silently print "done".
        import zipfile
        with ap.open_archive(MOUNT, COMP) as fh:
            with zipfile.ZipFile(fh) as zf:
                n_members = len(zf.namelist())
        print(f"[zip] done: {size / 1024**3:.2f} GiB, {n_members} members in "
              f"{len(ap.contiguous_parts(MOUNT, COMP))} parts at {pdir}", flush=True)

        Path(DONE).write_text(f"size={size} members={n_members} finished={time.time()}")
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)
        vol.commit()
    return ap.archive_size(MOUNT, COMP)


# -----------------------------------------------------------------------------
# verification: is the archive complete and readable?
# -----------------------------------------------------------------------------
@app.function(image=image, volumes={MOUNT: vol}, timeout=60 * 60, memory=8192)
def verify():
    """Open the zip's central directory and report member counts by top folder."""
    import collections
    import zipfile

    import archive_parts as ap

    csvs = sorted(os.listdir(CSV_DIR)) if os.path.exists(CSV_DIR) else []
    print(f"[verify] csv dir: {csvs}")

    # open_archive handles BOTH layouts: a single <comp>.zip on volumes filled before the
    # part split, and part files on everything since. Nothing here needs to know which.
    size = ap.archive_size(MOUNT, COMP)
    if not size:
        print(f"[verify] MISSING: no {ARCHIVE} and no parts in {ap.parts_dir(MOUNT, COMP)}")
        return {"ok": False, "csvs": csvs}

    n_parts = len(ap.contiguous_parts(MOUNT, COMP))
    print(f"[verify] {size / 1024**3:.2f} GiB "
          f"({'single archive' if n_parts == 0 else f'{n_parts} parts'})", flush=True)
    try:
        with ap.open_archive(MOUNT, COMP) as fh:
            with zipfile.ZipFile(fh) as zf:
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

    import archive_parts as ap

    size = ap.archive_size(MOUNT, COMP)
    if not size:
        print(f"[deep] MISSING: no {ARCHIVE} and no parts in {ap.parts_dir(MOUNT, COMP)}")
        return {"ok": False}

    with ap.open_archive(MOUNT, COMP) as _fh, zipfile.ZipFile(_fh) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        print(f"[deep] {len(infos)} members, {size / 1024**3:.2f} GiB", flush=True)

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
