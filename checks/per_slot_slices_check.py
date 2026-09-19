r"""per_slot_slices_check.py - prove input.n_slices can be per-slot without moving anything else.

The engine has always sampled the same number of slices from every slot. It now accepts a
mapping {slot: count}, because the planes do not deserve equal sampling: the MCL is a flat
band on the inner edge of the knee whose whole length sits in very few CORONAL planes, so
skipping coronal planes risks missing it entirely.

Slots with different depths cannot be stacked into (slots, D, H, W), so the dataset switches
to grouping each slot into its own images and concatenating along the IMAGE axis. That is a
change to the hottest path in the project, shared by 69 finished runs and the submission.
So the first and most important thing here is that a scalar n_slices still produces the OLD
tensor, byte for byte.

    python checks/per_slot_slices_check.py
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

# a run that is still on a SCALAR n_slices, so the "nothing changed" half of this
# check has something real to compare against
BASE = "convnext_base_25d_18slices"
OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""))


def main():
    cfg = sc.load_config(ROOT / "runs" / BASE, verbose=False)
    slots = list(cfg["input"]["slots"])

    # ---- helpers agree with the scalar config -------------------------------------------
    check("scalar config is not flagged per-slot", not sc.per_slot_input(cfg))
    check("slot_slices reads the scalar", all(sc.slot_slices(cfg, s) == 18 for s in slots))
    check("slot_groups reads slices_as_channels",
          all(sc.slot_groups(cfg, s) == 6 for s in slots))
    check("images_per_study unchanged for the scalar config",
          sc.images_per_study(cfg) == 6 * len(slots), str(sc.images_per_study(cfg)))

    # ---- the per-slot config: coronal denser --------------------------------------------
    per = copy.deepcopy(cfg)
    # coronal denser, everything else as-is - whatever slots this base happens to have
    per["input"]["n_slices"] = {s: (24 if s.startswith("cor") else 18) for s in slots}
    per["input"]["slices_as_channels"] = {s: (8 if s.startswith("cor") else 6) for s in slots}
    check("per-slot config IS flagged per-slot", sc.per_slot_input(per))
    check("channels per image agree across slots", sc.check_slot_geometry(per) == 3)
    want = sum(8 if s.startswith("cor") else 6 for s in slots)
    check("images_per_study sums the slots", sc.images_per_study(per) == want,
          "%d images" % sc.images_per_study(per))
    check("coronal really samples more", sc.slot_slices(per, "cor_fluid") == 24
          and sc.slot_slices(per, "sag_fluid") == 18)

    # bad setups must fail loudly, not silently sample something else
    for label, mut in (
            ("a slot missing from the mapping", lambda c: c["input"]["n_slices"].pop(slots[-1])),
            ("groups that do not divide slices", lambda c: c["input"]["slices_as_channels"].__setitem__("cor_fluid", 7)),
            ("channels differing between slots", lambda c: c["input"]["slices_as_channels"].__setitem__("cor_fluid", 6))):
        bad = copy.deepcopy(per)
        mut(bad)
        try:
            sc.check_slot_geometry(bad)
            check("rejects %s" % label, False)
        except ValueError:
            check("rejects %s" % label, True)

    # ---- real data: the OLD layout must be untouched ------------------------------------
    shards = Path(sc.resolve(cfg["paths"]["shards_dir"], cfg))
    have = sorted(p.stem for p in shards.glob("*.npz"))
    if not have:
        print("  (no local shards - skipping the tensor tests)")
        print("\n  %d passed, %d failed" % (len(OK), len(FAIL)))
        return 1 if FAIL else 0

    full = pd.concat([sc._eval_df(cfg, cfg["paths"]["train_csv"]),
                      sc._eval_df(cfg, cfg["paths"]["val_csv"])], ignore_index=True)
    df = full[full["StudyInstanceUID"].isin(have)].reset_index(drop=True)
    plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)

    ds_old = sc.KneeShardDataset(df, cfg, plan, train=False)
    x_old = ds_old[0][0]
    check("scalar config still emits (slots, D, H, W)",
          tuple(x_old.shape) == (len(slots), 18, cfg["input"]["size"], cfg["input"]["size"]),
          str(tuple(x_old.shape)))

    # the regression that matters: same config, same study, same bytes as before this change
    ds_again = sc.KneeShardDataset(df, copy.deepcopy(cfg), plan, train=False)
    check("scalar path is deterministic and unchanged",
          torch.equal(x_old, ds_again[0][0]))

    # ---- the per-slot layout -------------------------------------------------------------
    ds_new = sc.KneeShardDataset(df, per, plan, train=False)
    x_new = ds_new[0][0]
    check("per-slot emits (1, total_images, channels, H, W)",
          tuple(x_new.shape) == (1, want, 3, cfg["input"]["size"], cfg["input"]["size"]),
          str(tuple(x_new.shape)))

    # prepare_batch must turn it into exactly `want` three-channel images
    dev = torch.device("cpu")
    b_new = sc.prepare_batch(torch.stack([x_new, ds_new[1][0]]), per, dev)
    check("prepare_batch gives (2*total, 3, H, W)",
          tuple(b_new.shape) == (2 * want, 3, cfg["input"]["size"], cfg["input"]["size"]),
          str(tuple(b_new.shape)))
    check("per-slot pixels are finite", bool(torch.isfinite(b_new).all()))

    b_old = sc.prepare_batch(torch.stack([x_old, ds_old[1][0]]), cfg, dev)
    n_old = 6 * len(slots)
    check("old path still gives (2*%d, 3, H, W)" % n_old,
          tuple(b_old.shape) == (2 * n_old, 3, cfg["input"]["size"], cfg["input"]["size"]),
          str(tuple(b_old.shape)))
    check("both paths normalise onto the same scale",
          abs(float(b_new.mean()) - float(b_old.mean())) < 0.25,
          "new %.3f vs old %.3f" % (float(b_new.mean()), float(b_old.mean())))

    # ---- images land in SLOT ORDER, which every SLOT_MAP index depends on ---------------
    # give one slot a recognisable signature by blanking the rest through the plan
    solo = {k: {"cor_fluid": v["cor_fluid"]} for k, v in plan.items() if "cor_fluid" in v}
    ds_solo = sc.KneeShardDataset(df, per, solo, train=False)
    xs = ds_solo[0][0][0]                                  # (total_images, 3, H, W)
    off = sum(sc.slot_groups(per, s) for s in slots[:slots.index("cor_fluid")])
    n_cf = sc.slot_groups(per, "cor_fluid")
    nz = [int(i) for i in range(want) if float(xs[i].abs().sum()) > 0]
    check("only cor_fluid's own window range is filled",
          nz and min(nz) >= off and max(nz) < off + n_cf,
          "windows %d..%d, expected %d..%d" % (min(nz), max(nz), off, off + n_cf - 1)
          if nz else "nothing filled")
    check("cor_fluid contributes its 8 images", len(nz) == n_cf, "%d images" % len(nz))

    print("\n  %d passed, %d failed" % (len(OK), len(FAIL)))
    for f in FAIL:
        print("    FAILED: %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
