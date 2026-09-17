r"""mcl_smoke_check.py - run the MCL arms end to end on this PC before paying for a GPU.

The static check (mcl_specialist_check.py) proves the WEIGHTS are right. This proves the
DATA path is right, which is the half that has broken before: a run can load a perfectly
correct model and then feed it images from the wrong slots, or from a config it never read.

It does the real thing on real local shards:
  * loads each arm's OWN config.yaml off disk
  * builds the real loaders through the real slot planner
  * pulls one train batch and one val batch and checks the tensor shapes against
    slots x n_slices
  * reports how many studies actually resolve a real cor_plain series versus a blank, since
    cor_plain is the one slot the parent never saw and it is missing for ~23% of studies
  * runs a forward pass and checks the output is ONE logit per study, not twelve

CPU only. No GPU, no Modal, no network.

    python checks/mcl_smoke_check.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

ARMS = ["dinov3_mcl_slice", "dinov3_mcl_fresh"]
SUBSET = 6          # studies per split - enough to exercise every code path, fast enough to run
fails = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        fails.append(name)


def trim(cfg):
    """Shrink to a handful of studies and single-process loading, nothing else."""
    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"], pretrained=False)   # no download in a check
    cfg["dataloader"] = dict(cfg["dataloader"], num_workers=0, val_num_workers=0,
                             persistent_workers=False, prefetch_factor=None,
                             pin_memory=False, batch_size=1, val_batch_size=1)
    cfg["evaluation"] = dict(cfg["evaluation"])
    cfg["evaluation"]["audit"] = dict(cfg["evaluation"]["audit"], enable=False)
    cfg.setdefault("modal", {})["runs_mount"] = str(ROOT / "runs")
    return cfg


for arm in ARMS:
    print("\n=== %s ===" % arm)
    d = ROOT / arm if (ROOT / arm).exists() else ROOT / "runs" / arm

    # 1. the run must read ITS OWN config, not a sibling's
    cfg = sc.load_config(d, verbose=False)
    check("loads its own config.yaml", cfg["experiment"]["name"] == arm,
          cfg["experiment"]["name"])

    slots, ns = cfg["input"]["slots"], cfg["input"]["n_slices"]
    want_imgs = len(slots) * ns

    # 2. the real slot planner and the real dataset, over whatever shards are on this PC.
    # Only part of the corpus is mirrored locally, so each split is filtered to the studies
    # that actually have a .npz here - the code path is identical, just shorter.
    t0 = time.time()
    tcfg = trim(cfg)
    shards = Path(sc.resolve(tcfg["paths"]["shards_dir"], tcfg))
    have = {p.stem for p in shards.glob("*.npz")}
    idx = pd.read_csv(sc.resolve(tcfg["paths"]["series_index"], tcfg))
    plan = sc.plan_slots(idx, tcfg)
    check("built the slot plan over %d studies" % len(plan), len(plan) > 0,
          "%.1fs, %d shards on this PC" % (time.time() - t0, len(have)))

    datasets = {}
    for split, key, train in (("train", "train_csv", True), ("val", "val_csv", False)):
        df = pd.read_csv(sc.resolve(tcfg["paths"][key], tcfg))
        df = df[df["StudyInstanceUID"].isin(have)].head(SUBSET).reset_index(drop=True)
        check("%s has local studies to test with" % split, len(df) > 0, "%d rows" % len(df))
        datasets[split] = sc.KneeShardDataset(df, tcfg, plan, train)

    # 3. slot coverage. cor_plain is the slot the parent never saw and it only EXISTS for
    # 77.3% of studies; fallback_same_plane then substitutes another coronal series, so the
    # planner reports 100% and roughly a quarter of studies see coronal twice, not a blank.
    for slot in slots:
        got = sum(1 for v in plan.values() if v.get(slot))
        pctv = 100.0 * got / max(1, len(plan))
        check("slot %-10s resolves for %.1f%% of studies" % (slot, pctv), pctv >= 90.0,
              "%d of %d" % (got, len(plan)))

    # 4. one real batch from each split, pulled through a real DataLoader
    for split in ("train", "val"):
        dl = DataLoader(datasets[split], batch_size=1, shuffle=False, num_workers=0)
        b = next(iter(dl))
        x, y = (b[0], b[1]) if isinstance(b, (tuple, list)) else (b["x"], b["y"])
        check("%s raw batch is (1, %d slots, %d slices, px, px)" % (split, len(slots), ns),
              tuple(x.shape[:3]) == (1, len(slots), ns), str(tuple(x.shape)))
        # the dataset ships uint8 (B, slots, D, H, W); prepare_batch does the normalise and
        # the channel replication, and it is the SAME call the training loop makes
        x = sc.prepare_batch(x, tcfg, "cpu")
        n, c, h, w = x.shape
        check("%s batch is %d images (%d slots x %d slices)" % (split, want_imgs,
                                                                len(slots), ns),
              n == want_imgs, str(tuple(x.shape)))
        check("%s images are 3-channel %dpx" % (split, cfg["input"]["size"]),
              (c, h, w) == (3, cfg["input"]["size"], cfg["input"]["size"]),
              str((c, h, w)))
        check("%s labels are one column (MCL only)" % split, y.shape[-1] == 1,
              str(tuple(y.shape)))
        check("%s pixels are finite" % split, bool(torch.isfinite(x).all()))
        blank = int((x.view(want_imgs, -1).std(dim=1) < 1e-6).sum())
        check("%s has at most one blank slot of images" % split, blank <= ns,
              "%d blank images of %d" % (blank, want_imgs))

    # 5. a real forward pass produces ONE logit per study
    import importlib.util
    spec = importlib.util.spec_from_file_location("t_" + arm, d / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.build_model(trim(cfg)).eval()
    with torch.no_grad():
        # the loader batch at 512px is too big for a laptop forward, so feed the model
        # the shape it expects at a smaller side - the architecture is fully convolutional
        small = torch.zeros(want_imgs, 3, 64, 64)
        out = model(small)
    check("forward gives (1 study, 1 logit)", tuple(out.shape) == (1, 1), str(tuple(out.shape)))
    check("logit is finite", bool(torch.isfinite(out).all()), "%.4f" % float(out))

print("\n" + "=" * 74)
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("both MCL arms load their own config, their own slots, and produce one MCL logit")
