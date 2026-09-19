r"""laterality_run_check.py - prove convnext_base_25d_left_knees_mirrored is a clean A/B.

The run's whole value is that it differs from convnext_base_25d_18slices (val 0.9010) in
exactly one setting. If anything else drifted, the result attributes to nothing. This also
covers the submission path, which is where the flip can silently fail: the test studies are
not in data/12_laterality.csv, so their side has to come from the DICOM headers instead, and
a flip that quietly stops happening at inference would hand the model mirrored-vs-unmirrored
mismatch on every single test study.

    python checks/laterality_run_check.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

import sys as _s
RUN = (_s.argv[1] if len(_s.argv) > 1
       else "convnext_base_25d_all_knees_same_orientation")
PARENT = "convnext_base_25d_18slices"

OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""))


def walk(d, prefix=""):
    """Flatten a config to {dotted.key: value} so two can be diffed key by key."""
    out = {}
    for k, v in d.items():
        p = "%s.%s" % (prefix, k) if prefix else str(k)
        if isinstance(v, dict):
            out.update(walk(v, p))
        else:
            out[p] = v
    return out


def main():
    cfg = sc.load_config(ROOT / "runs" / RUN, verbose=False)
    par = sc.load_config(ROOT / "runs" / PARENT, verbose=False)

    # ---- 1. exactly one setting differs -------------------------------------------------
    a, b = walk(par), walk(cfg)
    keys = set(a) | set(b)
    diff = {k: (a.get(k, "<absent>"), b.get(k, "<absent>")) for k in keys
            if a.get(k, "<absent>") != b.get(k, "<absent>")}
    diff.pop("experiment.name", None)          # the name is allowed to differ
    # the extension block was added AFTER the run finished its 12 epochs; it changes what
    # happens next, not what the first 12 epochs did, so it does not break the A/B
    # modal.* is which machine it runs on, not what it computes: gpu and cpu change
    # throughput and cost, never the gradients, so they do not break the A/B either.
    for k in list(diff):
        if k.startswith("training.extend.") or k.startswith("modal."):
            diff.pop(k)
    expected = {"input.laterality_flip"}
    check("exactly one setting differs from the parent", set(diff) == expected,
          "differs: %s" % sorted(diff))
    check("and it is laterality_flip='left'", cfg["input"].get("laterality_flip") == "left")

    # things that would silently ruin the comparison
    check("same slice count as the parent", cfg["input"]["n_slices"] == par["input"]["n_slices"])
    check("same resolution as the parent", cfg["input"]["size"] == par["input"]["size"])
    check("same slots as the parent", cfg["input"]["slots"] == par["input"]["slots"])
    check("same splits as the parent",
          (cfg["paths"]["train_csv"], cfg["paths"]["val_csv"])
          == (par["paths"]["train_csv"], par["paths"]["val_csv"]))
    check("same epochs and lr as the parent",
          cfg["training"]["epochs"] == par["training"]["epochs"]
          and cfg["training"]["optimizer"]["lr"] == par["training"]["optimizer"]["lr"])
    check("hflip augmentation still 0.0 (it would undo the normalisation)",
          float(cfg["augmentation"]["hflip"]["p"]) == 0.0)
    check("grad_clip is at least 3", float(cfg["training"]["grad_clip"]) >= 3.0,
          str(cfg["training"]["grad_clip"]))
    check("train.py points at this run",
          'EXP_NAME = "%s"' % RUN in (ROOT / "runs" / RUN / "train.py").read_text(encoding="utf-8"))

    # ---- 2. the flip really happens on real data ----------------------------------------
    shards = Path(sc.resolve(cfg["paths"]["shards_dir"], cfg))
    have = sorted(p.stem for p in shards.glob("*.npz"))
    lat = pd.read_csv(sc.resolve(cfg["paths"]["laterality_csv"], cfg))
    sides = dict(zip(lat["StudyInstanceUID"], lat["side"]))

    if have:
        full = pd.concat([sc._eval_df(cfg, cfg["paths"]["train_csv"]),
                          sc._eval_df(cfg, cfg["paths"]["val_csv"])], ignore_index=True)
        df = full[full["StudyInstanceUID"].isin(have)].reset_index(drop=True)
        plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
        import copy
        off = copy.deepcopy(cfg)
        off["input"]["laterality_flip"] = None
        ds_on = sc.KneeShardDataset(df, cfg, plan, train=False)
        ds_off = sc.KneeShardDataset(df, off, plan, train=False)
        n_moved = 0
        for i in range(len(df)):
            side = sides.get(df["StudyInstanceUID"][i], "unknown")
            x_on, x_off = ds_on[i][0].numpy(), ds_off[i][0].numpy()
            moved = not np.array_equal(x_on, x_off)
            check("study %d (%s): %s" % (i, side, "mirrored" if side == "left" else "left as-is"),
                  moved == (side == "left"))
            if moved:
                n_moved += 1
                # per-plane axis: sagittal reverses slice order, coronal/axial reverse width
                want = np.stack([sc._mirror_slot(x_off[si], slot, 1)
                                 for si, slot in enumerate(cfg["input"]["slots"])])
                check("study %d mirrors each plane on its own axis" % i,
                      np.array_equal(x_on, want))
        check("at least one study actually moved in this sample", n_moved > 0,
              "%d of %d" % (n_moved, len(df)))
    else:
        print("  (no local shards - skipping the pixel test)")

    # ---- 3. the SUBMISSION path can still find the side ---------------------------------
    # build_dicom_index must carry ipp_x, and _sides_from_ipp must turn it into sides.
    idx = pd.DataFrame([{"study": "S_left", "series": "a", "ipp_x": 40.0},
                        {"study": "S_left", "series": "b", "ipp_x": 36.0},
                        {"study": "S_right", "series": "c", "ipp_x": -130.0},
                        {"study": "S_unknown", "series": "d", "ipp_x": float("nan")}])
    got = sc._sides_from_ipp(idx)
    check("submission: positive x -> left", got.get("S_left") == "left")
    check("submission: negative x -> right", got.get("S_right") == "right")
    check("submission: unreadable x -> absent, so never flipped", "S_unknown" not in got)
    check("submission: an index with no ipp_x column degrades to {} rather than crashing",
          sc._sides_from_ipp(idx.drop(columns=["ipp_x"])) == {})
    check("build_dicom_index emits ipp_x",
          '"ipp_x"' in (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8"))

    # the rule here must match the rule that built 12_laterality.csv, or train and test
    # disagree about which knee is which
    agree = sum(1 for _, r in lat.iterrows()
                if r["side"] == ("left" if float(r["x_mean"]) > 0 else "right"))
    check("submission rule reproduces 12_laterality.csv exactly", agree == len(lat),
          "%d/%d" % (agree, len(lat)))

    print("\n  %d passed, %d failed" % (len(OK), len(FAIL)))
    for f in FAIL:
        print("    FAILED: %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
