"""
13_make_folds.py - build 5 cross-validation folds over every labelled study.

WHY. Today one fixed split trains on 3,638 studies and holds out 704, so those 704 never
contribute a gradient and every model is measured on the same small sample. Five folds fix
both: each member trains on ~80% and the five collectively see 100% of the data, and every
study gets an out-of-fold prediction, giving an honest estimate over ~4,342 studies instead
of 704. The public 0.941 notebook builds 20 of its ~47 members exactly this way.

THREE THINGS THIS HAS TO GET RIGHT.

1. GROUPING. data/05_*.csv carry a `group` column: near-duplicate reports that describe the
   same underlying study. 79 of them share a group with another study. Split those apart and
   the model sees a study's twin in training and is scored on it in validation, which
   silently inflates every fold.

   Patient identity is NOT an additional constraint here, and that is worth recording:
   data/stats/patient_ids.csv has 4,407 studies over 4,407 distinct patient_ids, so no
   patient contributes two studies. The usual both-knees / follow-up leak does not exist in
   this dataset. Checked rather than assumed.

2. STRATIFICATION. The 12 findings are wildly imbalanced (Synovitis and Fracture are rare),
   so a random split can leave a fold with too few positives for a column's AUROC to mean
   anything. sklearn has no multilabel stratifier, so each study gets a single stratum key
   built from its RAREST positive finding - the label whose balance is most fragile - and
   StratifiedGroupKFold balances on that while honouring the groups.

3. REPRODUCIBILITY. Fixed seed, and the fold assignment is written out per study so any run
   can be traced back to exactly which studies it trained on.

Output, consumed directly as paths.train_csv / paths.val_csv in a run's config.yaml:

    data/13_fold{k}_train.csv   ~80%, k = 0..4
    data/13_fold{k}_val.csv     ~20%, the held-out part
    data/13_fold_assignment.csv StudyInstanceUID -> fold, for out-of-fold scoring

    python data_code/13_make_folds.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
N_FOLDS = 5
SEED = 42
SRC = ["05_train_qwen24t-med.csv", "05_val_qwen24t-med.csv"]
TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main():
    parts = []
    for f in SRC:
        p = DATA / f
        if not p.exists():
            raise SystemExit("missing %s" % p)
        parts.append(pd.read_csv(p))
    df = pd.concat(parts, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates("StudyInstanceUID").reset_index(drop=True)
    print("combined %s -> %d studies (%d duplicate rows dropped)"
          % (" + ".join(SRC), len(df), before - len(df)))

    missing = [t for t in TASKS if t not in df.columns]
    if missing:
        raise SystemExit("label columns absent: %s" % missing)

    # groups: fall back to the study id when a row has none, so it is its own group
    if "group" not in df.columns:
        raise SystemExit("no `group` column - cannot split safely")
    groups = df["group"].fillna(pd.Series(df["StudyInstanceUID"], index=df.index))
    shared = groups.value_counts()
    print("groups: %d distinct over %d studies; %d groups hold >1 study (%d studies)"
          % (groups.nunique(), len(df), (shared > 1).sum(), shared[shared > 1].sum()))

    # patient leakage: verified absent, but re-check rather than trust the comment
    pid = DATA / "stats" / "patient_ids.csv"
    if pid.exists():
        pp = pd.read_csv(pid).drop_duplicates("study")
        n = pp.groupby("patient_id").size()
        multi = int((n > 1).sum())
        print("patients: %d studies over %d patient_ids; %d patients have >1 study"
              % (len(pp), len(n), multi))
        if multi:
            print("  !! patients DO repeat - grouping by `group` alone is NOT enough")

    # stratum = the rarest positive finding, or 'none'. Rarest because that is the column
    # whose per-fold balance breaks first, and a fold with no positives scores NaN.
    Y = df[TASKS].to_numpy() > 0.5
    prevalence = Y.mean(axis=0)
    order = np.argsort(prevalence)                    # rarest first
    stratum = np.full(len(df), "none", dtype=object)
    for j in order[::-1]:                             # commonest first, so rarest overwrites
        stratum[Y[:, j]] = TASKS[j]
    print("\nlabel prevalence (rarest first):")
    for j in order:
        print("   %-20s %5.1f%%   (%d positive)" % (TASKS[j], 100 * prevalence[j],
                                                    int(Y[:, j].sum())))

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(df), -1, dtype=int)
    for k, (_, te) in enumerate(skf.split(df, stratum, groups=groups)):
        fold[te] = k
    assert (fold >= 0).all(), "some study was never assigned a fold"

    # no group may straddle two folds
    bad = pd.DataFrame({"g": groups.values, "f": fold}).groupby("g")["f"].nunique()
    assert (bad == 1).all(), "%d groups straddle folds" % int((bad > 1).sum())
    print("\ngroup integrity: every group sits in exactly one fold")

    print("\n%-6s %7s %7s | positives per fold (%% of that fold)" % ("fold", "train", "val"))
    for k in range(N_FOLDS):
        te = fold == k
        tr = ~te
        df[tr].to_csv(DATA / ("13_fold%d_train.csv" % k), index=False)
        df[te].to_csv(DATA / ("13_fold%d_val.csv" % k), index=False)
        rates = "  ".join("%s %.0f%%" % (TASKS[j][:3], 100 * Y[te, j].mean())
                          for j in order[:4])
        print("%-6d %7d %7d | %s" % (k, int(tr.sum()), int(te.sum()), rates))

    pd.DataFrame({"StudyInstanceUID": df["StudyInstanceUID"], "fold": fold}).to_csv(
        DATA / "13_fold_assignment.csv", index=False)

    # the rarest column must have enough positives in EVERY fold or its AUROC is unstable
    rarest = order[0]
    per = [int(Y[fold == k, rarest].sum()) for k in range(N_FOLDS)]
    print("\nrarest finding (%s) positives per fold: %s" % (TASKS[rarest], per))
    if min(per) < 10:
        print("  !! a fold has under 10 positives for it - that column's fold AUROC is noisy")
    print("\nwrote data/13_fold{0..%d}_{train,val}.csv and 13_fold_assignment.csv"
          % (N_FOLDS - 1))


if __name__ == "__main__":
    main()
