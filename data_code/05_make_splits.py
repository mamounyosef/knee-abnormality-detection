"""
05_make_splits.py — build the train / validation / audit splits.
================================================================

Inputs
------
    data/raw/train.csv          the 4,407 studies, 58 of them officially labeled
    data/02_reports_en.csv      English text + detected source language
    data/03_report_labels.csv   12 LLM probabilities per study

Design decisions and why
------------------------
* **The 58 official studies are held out entirely** as an audit set. They are the
  only true ground truth we have, and they were labeled from the IMAGES by an
  expert panel, so they are the closest thing to the competition's own target.
  Training on them would destroy the one honest check we can run.
* **Grouped by report text.** 177 studies share only 46 distinct report texts.
  Identical text means identical LLM labels, so letting a group straddle the
  split leaks the answer from train to validation.
* **Stratified by language and by label pattern.** Language is a near-perfect
  proxy for site (bg is 100% Philips; de/el/fr/nl are 100% Siemens), so an
  unstratified split would measure "does it transfer to Croatian scanners"
  rather than "does it find tears". Prevalence is stratified too, because
  several findings sit under 15%.
* **Soft probabilities are kept as the training target.** The measured label
  noise makes hard 0/1 targets overconfident; soft targets act as built-in label
  smoothing. Hard targets remain a planned ablation.
* **Unusable reports are dropped** (empty templates the labeler flagged); their
  images stay available for later semi-supervised work.

Usage
-----
    python data_code/05_make_splits.py
    python data_code/05_make_splits.py --val-frac 0.15 --seed 42

Outputs
-------
    data/05_train.csv   data/05_val.csv   data/05_audit58.csv
    data/stats/05_splits_log.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
LOG_TXT = ROOT / "data" / "stats" / "05_splits_log.txt"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--labels", default="03_report_labels.csv",
                    help="which LLM label file to split, relative to data/. The "
                         "relabelled corpus lives in 03_report_labels_qwen24t-med.csv.")
    ap.add_argument("--like", default=None,
                    help="reuse the train/val assignment from an existing split, given as "
                         "its tag suffix ('' for the untagged 05_train.csv). Without it, "
                         "new labels reshuffle the split and the run changes two things "
                         "at once.")
    ap.add_argument("--tag", default="",
                    help="isolate the outputs as 05_train_<tag>.csv etc, so a new label "
                         "set never silently overwrites the splits a finished run trained "
                         "on. The SPLIT ITSELF is unchanged - same seed, same grouping - "
                         "only the label columns differ.")
    args = ap.parse_args()
    suf = f"_{args.tag}" if args.tag else ""
    rng = np.random.default_rng(args.seed)

    train = pd.read_csv(RAW / "train.csv")
    en = pd.read_csv(ROOT / "data" / "02_reports_en.csv")[
        ["StudyInstanceUID", "Report_en", "lang"]]
    lab = pd.read_csv(ROOT / "data" / args.labels)
    print(f"[labels] {args.labels}  ({len(lab)} studies)")

    df = (train[["StudyInstanceUID", "Report"] + LABELS]
          .rename(columns={c: f"official_{c}" for c in LABELS})
          .merge(en, on="StudyInstanceUID", how="left")
          .merge(lab, on="StudyInstanceUID", how="left"))

    L = []
    def w(s=""):
        L.append(str(s)); print(s)

    w("=" * 78)
    w("SPLITS")
    w("=" * 78)
    w(f"studies loaded            : {len(df)}")
    w(f"with LLM labels           : {int(df[LABELS[0]].notna().sum())}")

    # --- normalise the language column (early runs returned names, later ISO) --
    NAME2ISO = {"english": "en", "spanish": "es", "turkish": "tr", "croatian": "hr",
                "greek": "el", "german": "de", "bulgarian": "bg", "dutch": "nl",
                "french": "fr"}
    df["lang"] = (df["lang"].astype(str).str.strip().str.lower()
                  .map(lambda v: NAME2ISO.get(v, v)))
    df.loc[~df["lang"].isin(list(NAME2ISO.values())), "lang"] = "other"

    # --- carve out the audit set ---------------------------------------------
    is_official = train[LABELS].notna().all(axis=1).values
    audit = df[is_official].copy()
    pool = df[~is_official].copy()
    w(f"held out as audit set     : {len(audit)}")

    # --- drop unusable and unlabeled -----------------------------------------
    before = len(pool)
    pool = pool[pool[LABELS[0]].notna()]
    n_unusable = int(pool["unusable"].fillna(False).astype(bool).sum())
    pool = pool[~pool["unusable"].fillna(False).astype(bool)]
    w(f"dropped unlabeled         : {before - len(pool) - n_unusable}")
    w(f"dropped unusable reports  : {n_unusable}")
    w(f"pool for train/val        : {len(pool)}")

    # --- group identical report texts ----------------------------------------
    pool["group"] = pool["Report"].fillna("").map(
        {t: i for i, t in enumerate(pool["Report"].fillna("").unique())})
    n_groups = pool["group"].nunique()
    dupe_groups = int((pool.groupby("group").size() > 1).sum())
    w(f"distinct report texts     : {n_groups}  ({dupe_groups} shared by >1 study)")

    # --- stratification key: language + coarse label pattern ------------------
    # A full 12-bit pattern would be too sparse to stratify on, so we bucket by
    # how many findings are positive and whether the knee looks degenerative or
    # traumatic - the two clusters found in the label correlations.
    hard = (pool[LABELS] > 0.5).astype(int)
    degen = hard[["Medial OA", "Lateral OA", "PF OA", "Baker's"]].sum(axis=1)
    trauma = hard[["ACL", "MCL", "Contusion", "Fracture"]].sum(axis=1)
    n_pos = hard.sum(axis=1)
    pool["stratum"] = (pool["lang"] + "|"
                       + pd.cut(n_pos, [-1, 0, 1, 3, 12],
                                labels=["0", "1", "2-3", "4+"]).astype(str) + "|"
                       + (degen > trauma).map({True: "deg", False: "tr"}))

    # one row per group, carrying that group's stratum (groups are duplicates of
    # the same text, so their stratum is identical anyway)
    g = pool.groupby("group").agg(stratum=("stratum", "first"),
                                  size=("group", "size")).reset_index()

    # --- assign groups to val, stratum by stratum -----------------------------
    val_groups: set[int] = set()
    for stratum, sub in g.groupby("stratum"):
        idx = sub.sample(frac=1.0, random_state=args.seed).index
        target = args.val_frac * sub["size"].sum()
        taken = 0
        for i in idx:
            if taken >= target:
                break
            val_groups.add(int(g.loc[i, "group"]))
            taken += int(g.loc[i, "size"])

    pool["split"] = np.where(pool["group"].isin(val_groups), "val", "train")
    if args.like is not None:
        # Reuse an EXISTING split's train/val assignment instead of recomputing it.
        #
        # The split is stratified by label pattern, so changing the labels reshuffles it:
        # relabeling with qwen2.4T moved 567 studies out of val and 546 into it, about 80%
        # of the set. A run trained on that would differ from the baseline in both its
        # targets AND which studies it validates on, which is two variables, not one.
        # With --like, the only thing that changes is the label columns.
        prev_tr = pd.read_csv(ROOT / "data" / f"05_train{args.like}.csv")
        prev_va = pd.read_csv(ROOT / "data" / f"05_val{args.like}.csv")
        assign = ({s: "train" for s in prev_tr.StudyInstanceUID}
                  | {s: "val" for s in prev_va.StudyInstanceUID})
        pool["split"] = pool.StudyInstanceUID.map(assign)
        missing = int(pool["split"].isna().sum())
        if missing:                       # a study the old split never saw: send it to train
            pool["split"] = pool["split"].fillna("train")
        w(f"[like] reused the split from 05_train{args.like}.csv / 05_val{args.like}.csv"
          f"  ({missing} study(s) not in it -> train)")

    tr_df = pool[pool.split == "train"].copy()
    va_df = pool[pool.split == "val"].copy()

    w(f"\ntrain studies             : {len(tr_df)}")
    w(f"val studies               : {len(va_df)}  ({len(va_df)/len(pool):.1%})")
    w(f"group overlap train/val   : "
      f"{len(set(tr_df['group']) & set(va_df['group']))}   (must be 0)")

    # --- did stratification hold? --------------------------------------------
    w("\n--- language balance (%) ---")
    lang_tab = pd.DataFrame({
        "train_%": tr_df["lang"].value_counts(normalize=True).mul(100).round(1),
        "val_%": va_df["lang"].value_counts(normalize=True).mul(100).round(1),
    }).fillna(0)
    lang_tab["diff"] = (lang_tab["val_%"] - lang_tab["train_%"]).round(1)
    w(lang_tab.to_string())

    w("\n--- label prevalence, fraction above 0.5 (%) ---")
    prev = pd.DataFrame({
        "train_%": (tr_df[LABELS] > 0.5).mean().mul(100).round(1),
        "val_%": (va_df[LABELS] > 0.5).mean().mul(100).round(1),
        "audit58_%": audit[[f"official_{c}" for c in LABELS]].mean().mul(100)
                      .round(1).set_axis(LABELS),
    })
    prev["diff"] = (prev["val_%"] - prev["train_%"]).round(1)
    w(prev.to_string())
    w(f"\nlargest prevalence gap between train and val: "
      f"{prev['diff'].abs().max():.1f} percentage points")

    w("\n--- mean soft label (the actual training target) ---")
    w(pd.DataFrame({"train": tr_df[LABELS].mean().round(3),
                    "val": va_df[LABELS].mean().round(3)}).to_string())

    # --- write ----------------------------------------------------------------
    keep = ["StudyInstanceUID", "lang", "group"] + LABELS
    tr_df[keep].to_csv(ROOT / "data" / f"05_train{suf}.csv", index=False)
    va_df[keep].to_csv(ROOT / "data" / f"05_val{suf}.csv", index=False)
    audit_keep = (["StudyInstanceUID", "lang"] + [f"official_{c}" for c in LABELS]
                  + [c for c in LABELS if c in audit.columns])
    # the audit is the PANEL's 0/1 labels, so it never changes with the LLM run -
    # written untagged on purpose, so every run is scored against the same 58
    audit[audit_keep].to_csv(ROOT / "data" / "05_audit58.csv", index=False)

    w("\nwritten: data/05_train.csv  data/05_val.csv  data/05_audit58.csv")
    w("\nNOTE: the label columns are the LLM's soft probabilities and are the")
    w("      training target. official_* in the audit file is the panel's 0/1.")

    LOG_TXT.parent.mkdir(parents=True, exist_ok=True)
    LOG_TXT.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
