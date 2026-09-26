"""local_shards_check.py - is data/shards a complete, uncorrupted copy of the knee-shards volume?

1. every file on the volume exists locally with the EXACT byte size
2. every study in the training / validation / audit / full-data CSVs has a shard
3. every local .npz opens and lists its arrays (a truncated download fails the zip index), and
   each array the series index expects for that study is present

    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session4.toml"
    python checks/local_shards_check.py
"""
import zipfile
from pathlib import Path

import modal
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SH = ROOT / "data/shards"
fails = []


def check(ok, msg):
    print(("  OK    " if ok else "  FAIL  ") + msg)
    if not ok:
        fails.append(msg)


vol = modal.Volume.from_name("knee-shards")
remote = {e.path: e.size for e in vol.listdir("/") if e.type == modal.volume.FileEntryType.FILE}
local = {p.name: p.stat().st_size for p in SH.glob("*.npz")}
missing = [n for n in remote if n not in local]
wrong = [n for n in remote if n in local and local[n] != remote[n]]
extra = [n for n in local if n not in remote]
check(not missing, f"all {len(remote)} volume files present locally (missing {len(missing)})")
check(not wrong, f"exact byte sizes match ({len(wrong)} differ)" + (f": {wrong[:3]}" if wrong else ""))
check(not extra, f"no stray local files ({len(extra)})")
check(sum(local.values()) == sum(remote.values()),
      f"total {sum(local.values()) / 1e9:.2f} GB local vs {sum(remote.values()) / 1e9:.2f} GB on the volume")
check(not list(SH.glob("*.part")), "no leftover .part files")

for csv in ("18_train_qwen-mg-oof.csv", "18_val_qwen-mg-oof.csv", "18_train_full_qwen-mg-oof.csv", "05_audit58.csv"):
    ids = pd.read_csv(ROOT / "data" / csv).StudyInstanceUID
    no = [s for s in ids if f"{s}.npz" not in local]
    check(not no, f"{csv}: all {len(ids)} studies have a shard ({len(no)} missing)")

idx = pd.read_csv(ROOT / "data/07_series_index.csv")
series = idx.groupby("study")["series"].apply(lambda s: set(map(str, s))).to_dict() if "series" in idx else {}
bad, miss_series = [], []
for i, n in enumerate(sorted(local)):
    try:
        with zipfile.ZipFile(SH / n) as z:
            names = {x[:-4] for x in z.namelist() if x.endswith(".npy")}
    except Exception as e:                                           # noqa: BLE001
        bad.append(f"{n}: {type(e).__name__}")
        continue
    want = series.get(n[:-4], set())
    if want and not want <= names:
        miss_series.append(n)
    if (i + 1) % 1000 == 0:
        print(f"    opened {i + 1}/{len(local)}", flush=True)
check(not bad, f"all {len(local)} shards open as valid zip archives ({len(bad)} broken)" + (f": {bad[:3]}" if bad else ""))
if series:
    check(not miss_series, f"every shard holds all its indexed series ({len(miss_series)} incomplete)")
print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
