"""aug_strength_check.py - are the three augmentation-strength arms the v4 twin moved to the v5 split
with the flip, differing from EACH OTHER only in augmentation? Are old runs untouched by the new
augmentation types, do the new types work, and does the new split actually load?

  1. each arm loads its OWN config through its train.py; vs the donor
     (convnext_base_25d_18slices_oof_convnext_mii_labels_w50) it changes ONLY experiment.name, the two
     split files, the knee flip and (not the _base arm) augmentation.*; the arms differ from each
     other ONLY in name and augmentation
  2. v5 split files, flip "left", same GPU / batch / epochs / seed / patience as the donor
  3. OLD RUNS UNCHANGED: with the new types at p 0 (the shared default), _augment gives the same
     output AND leaves the random generator in the same state as an augmentation block without the
     new keys at all (the generator goes on to pick the next slot's slices)
  4. each new type, forced on: output differs, shape kept, values in [0, 255], reproducible from the
     seed; cut-out zeroes one square through every slice
  5. CPU cost per slot volume of each arm vs the donor (the loader is the bottleneck on Modal)

    python checks/aug_strength_check.py
"""
import copy
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(RUNS))
import shared_code as sc  # noqa: E402

DONOR = "convnext_base_25d_18slices_oof_convnext_mii_labels_w50"
ARMS = ["convnext_base_25d_18slices_v5_aug_base", "convnext_base_25d_18slices_v5_aug_strong",
        "convnext_base_25d_18slices_v5_aug_strong_extra"]
SPLIT_KEYS = {"experiment.name", "paths.train_csv", "paths.val_csv", "input.laterality_flip",
              "training.early_stopping.patience"}   # patience 7 -> 6 (user, 3 Oct)
NEW = ("gamma", "noise", "blur", "cutout")
bad = 0


def check(ok, msg):
    global bad
    bad += not ok
    print(f"  {'OK  ' if ok else 'FAIL'}  {msg}")


def load(run):
    spec = importlib.util.spec_from_file_location(run, RUNS / run / "train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.cfg


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


class Fake:
    """just enough of KneeShardDataset for _augment"""
    def __init__(self, aug, size=336, ctx=1):
        self.aug, self.size, self.ctx = aug, size, ctx
    _augment = sc.KneeShardDataset._augment


dcfg = load(DONOR)
for arm in ARMS:
    print(f"\n{arm}")
    cfg = load(arm)
    check(cfg["experiment"]["name"] == arm, "1. loads its own config")
    f, fd = flat(cfg), flat(dcfg)
    diff = sorted(x for x in set(f) | set(fd) if f.get(x) != fd.get(x))
    aug_diff = [x for x in diff if x.startswith("augmentation.")]
    check(set(diff) - set(aug_diff) == SPLIT_KEYS, f"1. vs the donor: name, split files, flip + {len(aug_diff)} augmentation keys")
    check(bool(aug_diff) == (not arm.endswith("_base")), f"1. augmentation {'changed' if aug_diff else 'unchanged (baseline)'}")
    fb = flat(load(ARMS[0]))
    between = sorted(x for x in set(f) | set(fb) if f.get(x) != fb.get(x) and x != "experiment.name")
    check(all(x.startswith("augmentation.") for x in between), f"1. vs the _base arm only augmentation differs ({len(between)} keys)")
    check(f["paths.train_csv"] == "23_train_w50.csv" and f["paths.val_csv"] == "23_val_v3_audit.csv", "2. v5 split files")
    check(f["input.laterality_flip"] == "left", "2. knee flip on (left knees mirrored)")
    check(f["training.early_stopping.patience"] == 6, "2. early-stopping patience 6")
    for k in ("paths.audit_csv", "modal.gpu", "modal.cpu", "dataloader.batch_size", "dataloader.accum_steps",
              "training.epochs", "reproducibility.seed", "training.optimizer.lr"):
        check(f.get(k) == fd.get(k), f"2. {k} = {f.get(k)!r} (donor {fd.get(k)!r})")
    on = [t for t in NEW if cfg["augmentation"][t]["p"] > 0]
    print(f"      augmentation: " + ", ".join(f"{k} p={cfg['augmentation'][k]['p']}" for k in
                                             ("rotation", "translate", "scale", "brightness", "contrast",
                                              "slice_dropout") + NEW))

print("\n3. old runs untouched by the new types")
rng0 = np.random.default_rng(0)
vol = rng0.uniform(0, 255, (18, 336, 336)).astype(np.float32)
aug_now = dcfg["augmentation"]                          # shared defaults incl. the new keys at p 0
aug_old = {k: v for k, v in aug_now.items() if k not in NEW}
check(all(aug_now[t]["p"] == 0 for t in NEW), "the shared defaults have every new type at p 0")
same_out, same_state = True, True
for seed in range(200):
    r1, r2 = np.random.default_rng(seed), np.random.default_rng(seed)
    o1, o2 = Fake(aug_now)._augment(vol.copy(), r1), Fake(aug_old)._augment(vol.copy(), r2)
    same_out &= np.array_equal(o1, o2)
    same_state &= r1.random() == r2.random()
check(same_out, "same output as an augmentation block without the new keys (200 seeds)")
check(same_state, "random generator left in the same state, so later slice picks are unchanged (200 seeds)")

print("\n4. each new type, forced on")
base = copy.deepcopy(aug_old)
for k in ("rotation", "translate", "scale", "brightness", "contrast", "hflip", "slice_dropout"):
    base[k] = dict(base[k], p=0.0)
for t in NEW:
    a = copy.deepcopy(base)
    a.update({x: dict(aug_now[x], p=0.0) for x in NEW})
    a[t] = dict(aug_now[t], p=1.0)
    o = Fake(a)._augment(vol.copy(), np.random.default_rng(1))
    o_again = Fake(a)._augment(vol.copy(), np.random.default_rng(1))
    check(o.shape == vol.shape and o.min() >= 0 and o.max() <= 255 and not np.array_equal(o, vol)
          and np.array_equal(o, o_again), f"{t}: changes the volume, keeps the shape, stays in [0, 255], reproducible")
    if t == "cutout":
        z = (o == 0).all(axis=0)
        side = int(aug_now["cutout"]["frac"] * 336)
        check(z.sum() == side * side, f"cutout: one {side}x{side} square zeroed through all 18 slices ({z.sum()} px)")

print("\n5. CPU cost per slot volume (18 slices of 336 px), mean of 50")
for run, a in [(DONOR, dcfg["augmentation"])] + [(arm, load(arm)["augmentation"]) for arm in ARMS]:
    t0 = time.perf_counter()
    for s in range(50):
        Fake(a)._augment(vol.copy(), np.random.default_rng(s))
    print(f"      {run:52s} {(time.perf_counter() - t0) / 50 * 1000:6.1f} ms")

print("\n6. the v5 split loads through the engine (strong_extra arm, loaders in-process)")
cfg = copy.deepcopy(load(ARMS[2]))
cfg["dataloader"].update(num_workers=0, val_num_workers=0, batch_size=2, val_batch_size=2,
                         persistent_workers=False, pin_memory=False)
L = sc.make_loaders(cfg)
check(len(L["train"].dataset) == 3996 and len(L["val"].dataset) == 404,
      f"train {len(L['train'].dataset)} and val {len(L['val'].dataset)} studies")
xb, yb = next(iter(L["train"]))[:2]
check(np.isfinite(xb.numpy()).all() and yb.shape[-1] == 12, f"a train batch: x {tuple(xb.shape)}, y {tuple(yb.shape)}")
xv, yv = next(iter(L["val"]))[:2]
check(np.isfinite(xv.numpy()).all() and yv.shape[-1] == 12, f"a val batch: x {tuple(xv.shape)}, y {tuple(yv.shape)}")
vdf = L["val"].dataset.df
aud = vdf[vdf["source"] == "audit"]
check(len(aud) == 58 and set(np.unique(aud[cfg["tasks"]].values)) <= {0.0, 1.0},
      "the 58 audit rows in val carry 0/1 panel labels")
check(L["train"].dataset.flip_side == "left" and L["val"].dataset.flip_side == "left",
      "train and val datasets mirror left knees")
check(L["train"].dataset.aug is not None and L["val"].dataset.aug is None, "augmentation on for train, off for val")

print(f"\n{'ALL OK' if not bad else f'{bad} FAIL(S)'}")
sys.exit(1 if bad else 0)
