"""
06_check_split_leakage.py — what could still leak across the split?
===================================================================

Patient-level grouping is impossible in this dataset: de-identification gave
every study its own PatientID and FrameOfReferenceUID (4,407 distinct values for
4,407 studies) and removed StudyDate, InstitutionName, StationName and
PatientAge entirely. So if one person contributed two studies, no metadata field
reveals it.

The one residual signal is the text. A follow-up scan of the same knee, or the
second knee of the same patient, tends to produce a NEARLY identical report -
which exact-match grouping in 05_make_splits.py does not catch. This script
finds near-duplicate report pairs with TF-IDF cosine similarity and reports how
many straddle the train/validation boundary.

It also checks the other things that can quietly weaken a validation set:
positive counts per finding, scanner-vendor balance, and the effect of the two
untranslated studies.

    python data_code/06_check_split_leakage.py --threshold 0.80
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG_TXT = ROOT / "data" / "stats" / "06_split_leakage.txt"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="cosine similarity above which two reports are 'near-duplicate'")
    args = ap.parse_args()

    tr = pd.read_csv(ROOT / "data" / "05_train.csv")
    va = pd.read_csv(ROOT / "data" / "05_val.csv")
    en = pd.read_csv(ROOT / "data" / "02_reports_en.csv")
    raw = pd.read_csv(ROOT / "data" / "raw" / "train.csv")[["StudyInstanceUID", "Report"]]

    L = []
    def w(s=""):
        L.append(str(s)); print(s)

    w("=" * 78)
    w("SPLIT LEAKAGE AND VALIDATION-QUALITY CHECKS")
    w("=" * 78)
    w(f"train {len(tr)}   val {len(va)}")

    # ---------------------------------------------------------------- 1. text
    w("\n--- 1. near-duplicate reports across the split ---")
    w("Patient identity is unrecoverable (all IDs re-randomised per study), so")
    w("near-identical text is the only remaining trace of a repeat patient.")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    ids = pd.concat([tr[["StudyInstanceUID"]].assign(split="train"),
                     va[["StudyInstanceUID"]].assign(split="val")])
    txt = ids.merge(raw, on="StudyInstanceUID", how="left")
    txt["Report"] = txt["Report"].fillna("")

    vec = TfidfVectorizer(min_df=2, max_features=50_000, ngram_range=(1, 2))
    X = vec.fit_transform(txt["Report"])
    tr_mask = (txt["split"] == "train").values
    Xt, Xv = X[tr_mask], X[~tr_mask]

    # cosine in blocks: val x train, keeping only the best train match per val row
    best_sim = np.zeros(Xv.shape[0])
    best_idx = np.zeros(Xv.shape[0], dtype=int)
    BLOCK = 256
    for i in range(0, Xv.shape[0], BLOCK):
        S = cosine_similarity(Xv[i:i + BLOCK], Xt)
        best_sim[i:i + BLOCK] = S.max(axis=1)
        best_idx[i:i + BLOCK] = S.argmax(axis=1)

    w(f"\nbest train-match similarity for each val study:")
    w(pd.Series(best_sim).describe(percentiles=[.5, .9, .99]).round(3).to_string())
    for t in [0.95, 0.90, 0.85, 0.80, 0.70]:
        n = int((best_sim >= t).sum())
        w(f"  val studies with a train report >= {t:.2f} similar : {n:4d} ({n/len(best_sim):.1%})")

    flagged = np.where(best_sim >= args.threshold)[0]
    w(f"\nexamples at threshold {args.threshold}:")
    tr_txt = txt[tr_mask].reset_index(drop=True)
    va_txt = txt[~tr_mask].reset_index(drop=True)
    for j in flagged[:5]:
        w(f"  sim={best_sim[j]:.3f}")
        w(f"    val  : {va_txt.loc[j, 'Report'][:110]!r}")
        w(f"    train: {tr_txt.loc[best_idx[j], 'Report'][:110]!r}")
    if len(flagged) == 0:
        w("  (none)")

    # ------------------------------------------------------- 2. positive counts
    w("\n--- 2. positives per finding in validation ---")
    w("AUROC on a handful of positives is extremely noisy; this is the practical")
    w("limit on which findings the validation set can actually rank.")
    pos = pd.DataFrame({
        "val_pos_at_0.5": (va[LABELS] > 0.5).sum(),
        "val_pos_at_0.7": (va[LABELS] > 0.7).sum(),
        "train_pos_at_0.5": (tr[LABELS] > 0.5).sum(),
    })
    pos["approx_auroc_se"] = (1 / np.sqrt(np.minimum(
        pos["val_pos_at_0.5"], len(va) - pos["val_pos_at_0.5"]).clip(lower=1))).round(3)
    w(pos.sort_values("val_pos_at_0.5").to_string())

    # ------------------------------------------------------------ 3. scanners
    w("\n--- 3. scanner balance (not stratified for) ---")
    pid = ROOT / "data" / "stats" / "patient_ids.csv"
    if pid.exists():
        meta = pd.read_csv(pid).drop_duplicates("study")[["study", "manufacturer", "model"]]
        meta = meta.rename(columns={"study": "StudyInstanceUID"})
        def vendor(m):
            m = str(m).upper()
            for k in ["SIEMENS", "GE", "PHILIPS", "TOSHIBA", "CANON", "HITACHI"]:
                if k in m:
                    return k.title()
            return "Other"
        meta["vendor"] = meta["manufacturer"].map(vendor)
        a = tr.merge(meta, on="StudyInstanceUID", how="left")
        b = va.merge(meta, on="StudyInstanceUID", how="left")
        tab = pd.DataFrame({
            "train_%": a["vendor"].value_counts(normalize=True).mul(100).round(1),
            "val_%": b["vendor"].value_counts(normalize=True).mul(100).round(1),
        }).fillna(0)
        tab["diff"] = (tab["val_%"] - tab["train_%"]).round(1)
        w(tab.to_string())
        w(f"largest vendor gap: {tab['diff'].abs().max():.1f} pp")
    else:
        w("  data/stats/patient_ids.csv missing — run modal_scan_patient_ids.py")

    # ------------------------------------------------------------ 4. language
    w("\n--- 4. studies with no translation ---")
    miss = en[en["Report_en"].isna()]["StudyInstanceUID"]
    w(f"  untranslated studies: {len(miss)}  "
      f"(in train: {tr.StudyInstanceUID.isin(miss).sum()}, "
      f"in val: {va.StudyInstanceUID.isin(miss).sum()})")

    w("\n" + "=" * 78)
    w("WHAT WE CANNOT CHECK")
    w("  * Whether the same person contributed two studies. Every identifier was")
    w("    re-randomised per study and dates were stripped, so repeat patients")
    w("    are invisible. Near-duplicate text above is the only proxy.")
    w("  * Whether test-set prevalence matches ours; the hosts warn it may not.")
    w("    AUROC is rank-based so this is mostly tolerable.")

    LOG_TXT.parent.mkdir(parents=True, exist_ok=True)
    LOG_TXT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwritten -> {LOG_TXT}")


if __name__ == "__main__":
    main()
