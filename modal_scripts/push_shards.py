"""push_shards.py - upload data/shards to the knee-shards Modal volume, resumably and verified.

The mirror of pull_shards.py. Every file already on the volume with the SAME size is skipped,
so re-running the same command resumes after any interruption and doubles as a verification
pass. Uploads go in batches (one commit per batch); after each batch the volume is re-listed
and every file of that batch must be there with its exact local size. Also uploads the
pretrained backbone weights the Modal runs read from knee-runs:/pretrained.

    (select the Modal workspace first)
    python modal_scripts/push_shards.py                 # shards + pretrained weights
    python modal_scripts/push_shards.py --limit 5       # rehearsal: 5 shards only
    python modal_scripts/push_shards.py --verify        # list and compare, upload nothing
"""
import argparse
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "shards"
PRETRAINED = ["medimageinsight_davit_huge.safetensors", "MRI_CORE_vitb.pth", "MRI_CORE_vitb_mae.pth"]


def remote_sizes(vol, path="/"):
    try:
        return {Path(e.path).name: e.size for e in vol.listdir(path)
                if e.type == modal.volume.FileEntryType.FILE}
    except Exception:                                   # noqa: BLE001  (path not there yet)
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-gb", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--skip-pretrained", action="store_true")
    a = ap.parse_args()
    files = sorted(SRC.glob("*.npz"))
    if a.limit:
        files = files[:a.limit]
    vol = modal.Volume.from_name("knee-shards", create_if_missing=True)
    have = remote_sizes(vol)
    todo = [f for f in files if have.get(f.name) != f.stat().st_size]
    total = sum(f.stat().st_size for f in todo)
    print(f"{len(files)} local shards ({sum(f.stat().st_size for f in files) / 1e9:.1f} GB); "
          f"{len(files) - len(todo)} already on the volume with the right size; "
          f"{len(todo)} to upload ({total / 1e9:.1f} GB)", flush=True)
    if a.verify:
        sys.exit(0 if not todo else 1)
    t0, done_b, done_n = time.time(), 0, 0
    i = 0
    while i < len(todo):
        batch, size = [], 0
        while i < len(todo) and (not batch or size < a.batch_gb * 1e9):
            batch.append(todo[i]); size += todo[i].stat().st_size; i += 1
        for attempt in range(1, 4):
            try:
                with vol.batch_upload(force=True) as up:
                    for f in batch:
                        up.put_file(str(f), f"/{f.name}")
                break
            except Exception as e:                      # noqa: BLE001
                print(f"  batch upload failed (attempt {attempt}/3): {type(e).__name__}: {e}", flush=True)
                if attempt == 3:
                    sys.exit("giving up; re-run the same command to resume")
                time.sleep(20)
        have = remote_sizes(vol)
        bad = [f.name for f in batch if have.get(f.name) != f.stat().st_size]
        if bad:
            sys.exit(f"verification failed for {len(bad)} file(s), e.g. {bad[:2]}; re-run to resume")
        done_b += size; done_n += len(batch)
        el = time.time() - t0
        rate = done_b / el
        eta = (total - done_b) / rate if rate else 0
        print(f"[{time.strftime('%H:%M')}] {done_n}/{len(todo)} shards, {done_b / 1e9:.1f}/{total / 1e9:.1f} GB, "
              f"{rate / 1e6:.1f} MB/s, ETA {eta / 3600:.1f} h (batch verified)", flush=True)
    final = remote_sizes(vol)
    ok = sum(final.get(f.name) == f.stat().st_size for f in files)
    print(f"SHARDS DONE: {ok}/{len(files)} on the volume with the exact local size", flush=True)
    if not a.skip_pretrained and not a.limit:
        runs = modal.Volume.from_name("knee-runs", create_if_missing=True)
        have_p = remote_sizes(runs, "/pretrained")
        for w in PRETRAINED:
            p = ROOT / "data" / "pretrained" / w
            if have_p.get(w) == p.stat().st_size:
                print(f"  pretrained {w}: already there", flush=True)
                continue
            with runs.batch_upload(force=True) as up:
                up.put_file(str(p), f"/pretrained/{w}")
            print(f"  pretrained {w}: uploaded ({p.stat().st_size / 1e9:.2f} GB)", flush=True)
        have_p = remote_sizes(runs, "/pretrained")
        okp = all(have_p.get(w) == (ROOT / "data/pretrained" / w).stat().st_size for w in PRETRAINED)
        print(f"PRETRAINED {'DONE' if okp else 'INCOMPLETE'}", flush=True)
    sys.exit(0 if ok == len(files) else 1)


if __name__ == "__main__":
    main()
