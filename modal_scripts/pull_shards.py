"""pull_shards.py - copy the knee-shards Modal volume to data/shards, resumably.

Skips every file already present with the right size, writes each download to a .part file
and renames it only when complete (an interrupted run never leaves a truncated .npz), and
checks the free disk space first. Re-run the same command to resume.

    (select the Modal workspace first)
    python modal_scripts/pull_shards.py [--workers 8]
"""
import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

DEST = Path(__file__).resolve().parent.parent / "data" / "shards"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--volume", default="knee-shards")
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    vol = modal.Volume.from_name(a.volume)
    entries = [e for e in vol.listdir("/") if e.type == modal.volume.FileEntryType.FILE]
    todo = [e for e in entries
            if not (DEST / e.path).exists() or (DEST / e.path).stat().st_size != e.size]
    need = sum(e.size for e in todo)
    free = shutil.disk_usage(DEST).free
    print(f"{len(entries)} files on the volume ({sum(e.size for e in entries) / 1e9:.1f} GB); "
          f"{len(entries) - len(todo)} already here; {len(todo)} to download ({need / 1e9:.1f} GB); "
          f"{free / 1e9:.1f} GB free", flush=True)
    if need > free - 2e9:
        sys.exit(f"not enough space: need {need / 1e9:.1f} GB + 2 GB margin, have {free / 1e9:.1f} GB")
    for p in DEST.glob("*.part"):
        p.unlink()

    def fetch(e):
        tmp = DEST / (e.path + ".part")
        with open(tmp, "wb") as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
        if tmp.stat().st_size != e.size:
            tmp.unlink()
            raise IOError(f"{e.path}: got {tmp.stat().st_size if tmp.exists() else 0} of {e.size} bytes")
        os.replace(tmp, DEST / e.path)
        return e.size

    t0, done, nbytes, errors = time.time(), 0, 0, []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, e): e for e in todo}
        for fu in as_completed(futs):
            try:
                nbytes += fu.result()
                done += 1
            except Exception as err:                                  # noqa: BLE001
                errors.append(f"{futs[fu].path}: {err}")
            if done % 100 == 0 or done == len(todo):
                el = time.time() - t0
                rate = nbytes / max(el, 1e-9)
                print(f"{done}/{len(todo)} files, {nbytes / 1e9:.1f} GB, {rate / 1e6:.0f} MB/s, "
                      f"ETA {(need - nbytes) / max(rate, 1) / 60:.0f} min, errors {len(errors)}", flush=True)
    print(f"DONE: {done} downloaded, {len(errors)} errors" + ("".join("\n  " + x for x in errors[:20])))


if __name__ == "__main__":
    main()
