"""epoch_seed_check.py - do DataLoader workers see the epoch counter the main process sets?

Train augmentation is seeded from (seed, epoch, index). The main process sets shared_code._EPOCH
each epoch; workers must read the NEW value, or every epoch replays epoch 1's random slices and
crops. On Linux workers are forked and share the counter; on Windows they are spawned and
re-import shared_code, which froze it at 0 in every local run before 28 Sep.

Uses the REAL training dataset of a run, with persistent workers like the engine:
  1. the same study from a worker differs between epochs 1, 2, 3
  2. a worker's item for epoch e equals the main process's item for epoch e (determinism)

    python checks/epoch_seed_check.py [run]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "convnext_base_25d_poolattn_fulldata_flip_v3_labels"
    sys.path.insert(0, str(ROOT / "runs"))
    spec = importlib.util.spec_from_file_location("train", ROOT / "runs" / run / "train.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    sc, cfg = tm.sc, tm.cfg
    df = pd.read_csv(sc.resolve(cfg["paths"]["train_csv"], cfg)).head(4)
    plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
    ds = sc.KneeShardDataset(df, cfg, plan, train=True)
    dl = DataLoader(Subset(ds, [0, 1]), batch_size=1, num_workers=2, persistent_workers=True,
                    worker_init_fn=sc.seed_worker)
    got, bad = {}, 0
    for ep in (1, 2, 3):
        sc._EPOCH.value = ep
        got[ep] = [b[0][0].numpy() for b in dl]              # drop the loader's batch dim
        ref = ds[0][0].numpy() if hasattr(ds[0][0], "numpy") else np.asarray(ds[0][0])
        same = np.array_equal(got[ep][0], ref)
        print(f"epoch {ep}: worker item == main-process item for this epoch: {same}", flush=True)
        bad += not same
    for a, b in ((1, 2), (2, 3)):
        diff = not np.array_equal(got[a][0], got[b][0])
        print(f"epoch {a} vs {b}: augmentation differs: {diff}", flush=True)
        bad += not diff
    print("ALL OK: workers follow the epoch" if not bad else f"BUG: {bad} check(s) failed", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
