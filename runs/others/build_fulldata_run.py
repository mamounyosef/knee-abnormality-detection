r"""build_fulldata_run.py - make the full-data (+ mirrored) retrain of an existing member.

Training on train+val+audit, and mirroring left knees so every study faces the same way, is
the change that has paid every time it has been tried on the leaderboard: convnext_base_relabel
0.918 -> 0.934, then three members at once 0.935 -> 0.937. This builds that variant of any
finished run without hand-editing a config.

What it changes, and why each one:
  - paths.train_csv -> 08_train_full.csv (every study)
  - training.epochs FIXED at the donor's peak epoch: nothing is held out, so there is no curve
    to early-stop on, and the epoch count is the only guard against overfitting
  - early stopping off, evaluation off (a validation would score its own training data;
    checkpoints fall back to the step counter, which the bundler's staleness check reads)
  - laterality_flip "left" when --flip is given and the donor lacks it

Named "_fulldata" (plus "_flip" when the mirror is added), never "_full": the existing
mricore_vitb_mae_cross_attention_full means full cross-slice FUSION, and that collision
already sent the ensemble sweep's leak filter the wrong way once.

    python runs/others/build_fulldata_run.py convnext_base_25d_perfinding_slots_and_pooling --flip
    python runs/others/build_fulldata_run.py convnext_base_25d_all_knees_same_orientation
"""
import argparse
import io
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

RUNS = Path(__file__).resolve().parent.parent


def peak_epoch(donor: str) -> int:
    d = pd.read_csv(RUNS / donor / "results" / "val_log.csv")
    return int(d.loc[d["val_macro_auroc"].idxmax(), "epoch"])


def build(donor: str, flip: bool, epochs: int | None, why: str) -> str:
    src = RUNS / donor
    has_flip = "laterality_flip: \"left\"" in (src / "config.yaml").read_text(encoding="utf-8")
    name = donor + "_fulldata" + ("_flip" if flip and not has_flip else "")
    dst = RUNS / name
    if dst.exists():
        raise SystemExit("%s already exists; delete it first if you mean to rebuild" % name)
    ep = epochs or peak_epoch(donor)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        "__pycache__", "results", "kaggle", "*.pyc"))
    for py in dst.glob("*.py"):
        t = io.open(py, encoding="utf-8").read().replace(donor, name)
        io.open(py, "w", encoding="utf-8", newline="\n").write(t)

    P = dst / "config.yaml"
    s = io.open(P, encoding="utf-8").read().replace('"%s"' % donor, '"%s"' % name)
    s, n = re.subn(r'train_csv:\s*"[^"]+"[^\n]*',
                   'train_csv: "08_train_full.csv"     # train + val + audit, every study', s,
                   count=1)
    if n == 0:
        s = s.replace("\npaths:\n", '\npaths:\n  train_csv: "08_train_full.csv"     # every study\n', 1)
        n = 1
    assert n == 1, "could not set train_csv"
    s, n = re.subn(r"\n  epochs:\s*\d+",
                   "\n  # FIXED at the donor's peak epoch (%d): nothing is held out, so the epoch\n"
                   "  # count is the only guard against overfitting.\n  epochs: %d" % (ep, ep),
                   s, count=1)
    assert n == 1, "epochs not found"
    s, n = re.subn(r"(early_stopping:\s*\n\s*enable:\s*)[Tt]rue", r"\1false", s, count=1)
    if n == 0 and "early_stopping" not in s:
        s = s.replace("\ntraining:\n", "\ntraining:\n  early_stopping:\n    enable: false\n", 1)
    if flip and not has_flip:
        s, n = re.subn(r'\n(\s*)laterality_flip:\s*\S+', '\n\\1laterality_flip: "left"', s,
                       count=1)
        if n == 0:
            s = s.replace("\ninput:\n", '\ninput:\n  laterality_flip: "left"   # mirror every '
                          "LEFT knee; axis chosen per plane\n", 1)
    if "\nevaluation:" in s:
        s = s.replace("\nevaluation:\n", "\nevaluation:\n  enable: false\n", 1)
    else:
        s = s.rstrip("\n") + ("\n\nevaluation:\n  # OFF: every val and audit study is in the "
                              "training set now.\n  enable: false\n")
    s = "# %s\n#\n# %s\n%s" % (name, why, s)
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("donor")
    ap.add_argument("--flip", action="store_true", help="add the laterality mirror if absent")
    ap.add_argument("--epochs", type=int, default=None, help="default: the donor's peak epoch")
    ap.add_argument("--why", default="Full-data retrain of the donor; see "
                    "runs/others/build_fulldata_run.py for what changes and why.")
    a = ap.parse_args()
    name = build(a.donor, a.flip, a.epochs, a.why)
    print("built", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
