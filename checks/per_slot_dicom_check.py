r"""per_slot_dicom_check.py - does the per-slot slice layout survive the SUBMISSION path?

convnext_base_25d_perfinding_slots_and_pooling is the first run to use a per-slot n_slices
mapping, and every test it has passed so far read pixels from .npz SHARDS. The submission
reads raw DICOM instead, through the same KneeShardDataset but with `dicom=` set, which
takes a different branch in __init__ and a different _read_series.

If the per-slot grouping is wrong on that branch, nothing raises: the tensor still has a
plausible shape and the model still emits twelve logits. It would simply be scoring a
different geometry than it trained on, and we would read the leaderboard number as a verdict
on the idea rather than on a plumbing bug.

So this drives the DICOM branch directly and asserts it produces the SAME layout the shard
branch does, on the same study.

    python checks/per_slot_dicom_check.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

RUN = "convnext_base_25d_perfinding_slots_and_pooling"
OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""))


def main():
    cfg = sc.load_config(ROOT / "runs" / RUN, verbose=False)
    slots = list(cfg["input"]["slots"])
    want_imgs = sc.images_per_study(cfg)
    chans = sc.check_slot_geometry(cfg)
    size = cfg["input"]["size"]
    check("run is per-slot", sc.per_slot_input(cfg))
    check("expects %d windows of %d channels" % (want_imgs, chans),
          want_imgs == 40 and chans == 3, "%d x %d" % (want_imgs, chans))

    idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
    plan = sc.plan_slots(idx, cfg)

    # ---- the SHARD branch, as the reference ---------------------------------------------
    shards = Path(sc.resolve(cfg["paths"]["shards_dir"], cfg))
    have = sorted(p.stem for p in shards.glob("*.npz"))
    if not have:
        print("  no local shards")
        return 1
    full = pd.concat([sc._eval_df(cfg, cfg["paths"]["train_csv"]),
                      sc._eval_df(cfg, cfg["paths"]["val_csv"])], ignore_index=True)
    df = full[full["StudyInstanceUID"].isin(have)].reset_index(drop=True)
    ds_shard = sc.KneeShardDataset(df, cfg, plan, train=False)
    x_shard = ds_shard[0][0]
    check("shard branch -> (1, %d, %d, %d, %d)" % (want_imgs, chans, size, size),
          tuple(x_shard.shape) == (1, want_imgs, chans, size, size), str(tuple(x_shard.shape)))

    # ---- the DICOM branch -----------------------------------------------------------------
    # The two branches differ ONLY in _read_series. Driving the real DICOM branch needs a
    # complete study on disk, which this PC does not have (31 sample files, a few series).
    # So this takes the dicom branch for real - dicom= set, self.geo from the index, the
    # early return in __init__ - and substitutes the shard reader for the pixel fetch. That
    # exercises every line of the per-slot grouping on the submission code path.
    dicom_index = idx.rename(columns={}).copy()
    files_by_series = {s: ["dummy"] for s in idx["series"]}
    ds_dicom = sc.KneeShardDataset(df, cfg, plan, train=False,
                                   dicom=(dicom_index, files_by_series))
    check("dicom branch took the dicom __init__ path", ds_dicom.dicom_files is not None)
    check("dicom branch still knows the per-slot geometry",
          ds_dicom.per_slot and ds_dicom.slot_n == ds_shard.slot_n
          and ds_dicom.slot_g == ds_shard.slot_g,
          "n=%s" % ds_dicom.slot_n)

    ds_dicom._read_series = ds_shard._read_series          # same pixels, dicom branch
    x_dicom = ds_dicom[0][0]
    check("dicom branch -> (1, %d, %d, %d, %d)" % (want_imgs, chans, size, size),
          tuple(x_dicom.shape) == (1, want_imgs, chans, size, size), str(tuple(x_dicom.shape)))
    check("dicom branch produces the SAME tensor as the shard branch",
          torch.equal(x_shard, x_dicom),
          "max diff %d" % int((x_shard.int() - x_dicom.int()).abs().max()))

    # ---- prepare_batch and a real forward on the submission shape ------------------------
    b = sc.prepare_batch(torch.stack([x_dicom, x_dicom]), cfg, torch.device("cpu"))
    check("prepare_batch -> (2*%d, %d, %d, %d)" % (want_imgs, chans, size, size),
          tuple(b.shape) == (2 * want_imgs, chans, size, size), str(tuple(b.shape)))

    import importlib.util
    d = ROOT / "runs" / RUN
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("sub_" + RUN, d / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    small = sc.load_config(d, verbose=False)
    small["model"] = dict(small["model"], pretrained=False)
    small["input"] = dict(small["input"], size=64)
    m = mod.build_model(small).eval()
    with torch.no_grad():
        out = m(torch.zeros(want_imgs, 3, 64, 64))
    check("model consumes %d windows -> (1, 12)" % want_imgs,
          tuple(out.shape) == (1, 12) and bool(torch.isfinite(out).all()), str(tuple(out.shape)))

    # ---- the checkpoint we would actually ship -------------------------------------------
    res = ROOT / "runs" / RUN / "results"
    vlog = res / "val_log.csv"
    if vlog.exists():
        v = pd.read_csv(vlog)
        best = v.loc[v["val_macro_auroc"].idxmax()]
        check("best checkpoint is step %d, val %.4f" % (best["step"], best["val_macro_auroc"]),
              True, "epoch %d of %d" % (best["epoch"], v["epoch"].max()))

    print("\n  %d passed, %d failed" % (len(OK), len(FAIL)))
    for f in FAIL:
        print("    FAILED: %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
