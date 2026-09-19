r"""laterality_flip_check.py - prove input.laterality_flip does exactly what it claims.

Runs against the real KneeShardDataset and the few shards on this PC, no GPU. What it has
to establish before the flip is trusted on Modal:

  1. OFF BY DEFAULT. Every existing run's config leaves laterality_flip unset, and the
     tensors it produces must be byte-identical to what they were before this existed.
     This is the one that protects 80-odd finished runs and the submission path.
  2. The flip is a genuine left-right mirror of the pixels, not a no-op and not a
     transpose, and it is applied to EVERY slot and EVERY slice of the study.
  3. Only the named side moves. flip="right" must leave left knees untouched, and
     vice versa, and a study with no laterality label must be left alone.
  4. Labels are never touched. Mirroring the image must not reorder y.
  5. A bad value fails loudly rather than silently doing nothing.

    python checks/laterality_flip_check.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""))


def cfg_for(flip, _cfg=[None]):
    """A copy of the run config with laterality_flip set to `flip`."""
    import copy
    c = copy.deepcopy(_cfg[0])
    c["input"]["laterality_flip"] = flip
    return c


def build(cfg_flip, df, cfg, plan):
    c = sc.deep_merge(cfg, {"input": {"laterality_flip": cfg_flip}}) \
        if hasattr(sc, "deep_merge") else None
    if c is None:                       # fall back to an explicit copy
        import copy
        c = copy.deepcopy(cfg)
        c["input"]["laterality_flip"] = cfg_flip
    return sc.KneeShardDataset(df, c, plan, train=False)


def main():
    cfg = sc.load_config(ROOT / "runs" / "convnext_base_25d_18slices", verbose=False)
    shards = Path(sc.resolve(cfg["paths"]["shards_dir"], cfg))
    have = sorted(p.stem for p in shards.glob("*.npz"))
    if not have:
        print("no local shards under %s - cannot run" % shards)
        return 1
    print("  %d local shards" % len(have))

    lat = pd.read_csv(sc.resolve(cfg["paths"]["laterality_csv"], cfg))
    sides = dict(zip(lat["StudyInstanceUID"], lat["side"]))

    # a frame of exactly the studies whose pixels are on this PC, with real labels
    full = sc._eval_df(cfg, cfg["paths"]["train_csv"])
    val = sc._eval_df(cfg, cfg["paths"]["val_csv"])
    full = pd.concat([full, val], ignore_index=True)
    df = full[full["StudyInstanceUID"].isin(have)].reset_index(drop=True)
    if df.empty:
        print("  local shards are not in train/val - cannot run")
        return 1
    df_sides = [sides.get(s, "unknown") for s in df["StudyInstanceUID"]]
    print("  using %d studies: %s" % (len(df), df_sides))

    plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
    cfg_for.__defaults__[0][0] = cfg          # so cfg_for can make copies of it

    ds_off = build(None, df, cfg, plan)
    ds_r = build("right", df, cfg, plan)
    ds_l = build("left", df, cfg, plan)

    check("flip=None loads no laterality table", ds_off.sides == {} and not ds_off.flip_side)
    check("flip='right' loads the table", len(ds_r.sides) > 4000,
          "%d studies" % len(ds_r.sides))

    x_off = [ds_off[i] for i in range(len(df))]
    x_r = [ds_r[i] for i in range(len(df))]
    x_l = [ds_l[i] for i in range(len(df))]

    # 1. off by default == the old behaviour. Compare against a dataset built from a config
    #    that never mentions the key at all, which is literally every finished run.
    import copy
    bare = copy.deepcopy(cfg)
    bare["input"].pop("laterality_flip", None)
    ds_bare = sc.KneeShardDataset(df, bare, plan, train=False)
    same = all(np.array_equal(ds_bare[i][0].numpy(), x_off[i][0].numpy())
               for i in range(len(df)))
    check("absent key == laterality_flip:null, byte for byte", same)

    # 2/3. the right studies moved, and only those
    for i, side in enumerate(df_sides):
        a = x_off[i][0].numpy()
        for mode, xs in (("right", x_r), ("left", x_l)):
            b = xs[i][0].numpy()
            should_move = side == mode
            moved = not np.array_equal(a, b)
            check("study %d (%s), flip=%s: %s" %
                  (i, side, mode, "mirrored" if should_move else "untouched"),
                  moved == should_move)
            if should_move:
                # the mirror axis depends on the PLANE: sagittal reverses slice order,
                # coronal and axial reverse width. a and b are (slots, D, H, W) in
                # cfg["input"]["slots"] order, so each slot is checked against its own rule.
                want = np.stack([sc._mirror_slot(a[si], slot, 1)
                                 for si, slot in enumerate(cfg["input"]["slots"])])
                check("study %d, flip=%s mirrors each plane on its own axis" % (i, mode),
                      np.array_equal(b, want))
                sag = [si for si, s_ in enumerate(cfg["input"]["slots"])
                       if s_.startswith("sag")]
                if sag:
                    si = sag[0]
                    check("study %d, sagittal reversed slices, NOT width" % i,
                          np.array_equal(b[si], a[si][::-1])
                          and not np.array_equal(b[si], a[si][:, :, ::-1]))

    # 3b. THE FLIP MUST APPLY TO EVERY SPLIT, NOT JUST TRAINING.
    # Augmentation in this engine IS train-only (self.aug is gated on `train`), so there is
    # real precedent for a transform that silently skips validation. If the flip did that,
    # the model would train on normalised knees and be scored on un-normalised ones: the
    # val number would measure a train/test mismatch and the whole experiment would be void.
    for train_flag in (True, False):
        ds_t = sc.KneeShardDataset(df, cfg_for("left"), plan, train=train_flag)
        ds_n = sc.KneeShardDataset(df, cfg_for(None), plan, train=train_flag)
        moved = [i for i in range(len(df))
                 if not np.array_equal(ds_t[i][0].numpy(), ds_n[i][0].numpy())]
        want = [i for i, sd in enumerate(df_sides) if sd == "left"]
        check("split=%s: the flip applies (%d of %d studies moved)"
              % ("train" if train_flag else "val/audit", len(moved), len(df)),
              moved == want, "moved %s, expected %s" % (moved, want))
    # and the dataset must not be reading `train` when it decides to mirror
    # the two lines that actually decide a mirror must not mention `train`. A wider slice
    # catches the unrelated `if train and dicom is None` that loads the auxiliary head.
    src = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
    decide = [ln for ln in src.splitlines()
              if "flip_side" in ln or "_mirror_slot(v, slot" in ln]
    check("nothing in the mirror decision consults `train`",
          decide and not any("train" in ln for ln in decide),
          "%d lines checked" % len(decide))

    # 4. labels untouched
    check("labels identical under flip",
          all(np.array_equal(x_off[i][1].numpy(), x_r[i][1].numpy()) for i in range(len(df))))

    # 5. a typo must not silently pass
    try:
        build("Right", df, cfg, plan)
        check("bad laterality_flip value raises", False)
    except ValueError:
        check("bad laterality_flip value raises", True)

    print("\n  %d passed, %d failed" % (len(OK), len(FAIL)))
    for f in FAIL:
        print("    FAILED: %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
