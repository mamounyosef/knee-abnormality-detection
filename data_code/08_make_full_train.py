"""
08_make_full_train.py — one training CSV holding every labelled study we have.

Normal runs train on 3,638 studies and hold out 704 for validation plus 58 for the panel
audit. That held-out 762 is 17% of the data, and it exists to answer "is this model any
good". Once the architecture, the schedule and the member list are settled, that question
is already answered, and the 17% is just data the final model never saw.

This builds the training set for a FINAL model: 3,638 + 704 + 58 = 4,400 studies.

Label sources, in the order they are preferred:

  train (3,638)   qwen-2.4T soft probabilities, exactly as 05_train_qwen24t-med.csv holds them
  val   (704)     the same, from 05_val_qwen24t-med.csv
  audit (58)      the PANEL's official 0/1 labels, not their LLM labels. 05_audit58.csv
                  carries both; the official columns are human ground truth and are strictly
                  better, so they win. Soft-target training takes them as-is - a 0.0 or a
                  1.0 is a perfectly valid probability.

The three splits are verified disjoint here rather than assumed, because a study appearing
twice would be silently upweighted.

What you give up by using this: there is no held-out set left, so the resulting run has NO
validation curve, NO early stopping and NO best-checkpoint selection. It trains for a fixed
number of epochs and you take the last checkpoint. Everything the run prints as "val" is
in-sample and must not be compared against any other run's val score.

    python data_code/08_make_full_train.py

Writes data/08_train_full.csv.
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
ID = "StudyInstanceUID"

TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

OUT = DATA / "08_train_full.csv"


def main():
    tr = pd.read_csv(DATA / "05_train_qwen24t-med.csv")
    va = pd.read_csv(DATA / "05_val_qwen24t-med.csv")
    au = pd.read_csv(DATA / "05_audit58.csv")

    # the audit file carries BOTH the panel's official_<task> columns and its own qwen
    # columns. Drop the qwen ones and promote the official ones - human labels beat LLM
    # labels, and these 58 are the only human labels in the whole corpus.
    off = ["official_%s" % t for t in TASKS]
    missing = [c for c in off if c not in au.columns]
    if missing:
        raise SystemExit("05_audit58.csv is missing %s" % missing)
    au = au.drop(columns=[t for t in TASKS if t in au.columns])
    au = au.rename(columns={"official_%s" % t: t for t in TASKS})

    for name, d in (("train", tr), ("val", va), ("audit", au)):
        if "group" not in d.columns:
            d["group"] = -1              # carried metadata only; the engine never reads it
        d["source"] = name               # so a later run can weight or inspect by origin

    keep = [ID, "lang", "group", "source"] + TASKS
    parts = [d[keep] for d in (tr, va, au)]
    full = pd.concat(parts, ignore_index=True)

    # disjointness is checked, not assumed - a duplicated study is silently upweighted
    dupes = full[ID].duplicated().sum()
    if dupes:
        raise SystemExit("%d duplicated studies across the three splits - refusing to "
                         "write a training set that double-counts them" % dupes)

    n_missing = full[TASKS].isna().sum().sum()
    full.to_csv(OUT, index=False)

    print("wrote %s" % OUT)
    print("  train %d + val %d + audit %d = %d studies" % (len(tr), len(va), len(au), len(full)))
    print("  %+.1f%% more data than the 3,638 every other run trained on"
          % (100 * (len(full) / len(tr) - 1)))
    print("  %d missing label cells" % n_missing)
    print()
    print("  positives per finding (thresholded at 0.5, for a sanity read only):")
    for t in TASKS:
        col = full[t]
        print("    %-18s %4d / %d   (%.1f%%)"
              % (t, int((col > 0.5).sum()), len(full), 100 * (col > 0.5).mean()))


if __name__ == "__main__":
    main()
