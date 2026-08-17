"""
04_validate_labels.py — how good are the LLM report labels?
===========================================================

Scores data/03_report_labels.csv against the 58 officially labeled studies and
writes a plain-text report.

Two reference points to keep the number honest:
  * the multilingual keyword matcher scores 0.7003 macro AUROC (the floor),
  * the 58 labels were assigned by an expert panel re-reading the IMAGES, so
    perfect agreement with the reports is not attainable, and 58 studies carry
    roughly +/-0.05 of sampling noise.

    python data_code/04_validate_labels.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT_TXT = ROOT / "data" / "stats" / "04_llm_vs_58.txt"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
pred = pd.read_csv(ROOT / "data" / "03_report_labels.csv")

truth = train[train[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")
P = pred.set_index("StudyInstanceUID")
common = truth.index.intersection(P.index)
T, Q = truth.loc[common, LABELS].astype(int), P.loc[common, LABELS]

L = []
def w(s=""):
    L.append(str(s)); print(s)

w("=" * 78)
w("LLM REPORT LABELS vs THE 58 OFFICIAL STUDIES")
w("=" * 78)
w(f"studies scored : {len(common)}")
w(f"labeler        : Qwen3.8-27B, translated-to-English reports, effort=low")
w("")

rows = []
for c in LABELS:
    t, q = T[c].values, Q[c].values
    rows.append({
        "finding": c,
        "pos": int(t.sum()),
        "mean_p_pos": q[t == 1].mean(),
        "mean_p_neg": q[t == 0].mean(),
        "separation": q[t == 1].mean() - q[t == 0].mean(),
        "auroc": roc_auc_score(t, q) if len(set(t)) > 1 else np.nan,
        "ap": average_precision_score(t, q) if len(set(t)) > 1 else np.nan,
    })
R = pd.DataFrame(rows).set_index("finding").round(3)
w(R.sort_values("auroc", ascending=False).to_string())

macro = R["auroc"].mean()
w("")
w(f"MACRO AUROC        = {macro:.4f}")
w(f"keyword floor      = 0.7003")
w(f"improvement        = {macro - 0.7003:+.4f}")
w("")

w("--- how the errors are distributed ---")
w("A 'confident miss' is a study where the model was sure and the panel disagreed.")
for c in LABELS:
    t, q = T[c].values, Q[c].values
    fp = int(((q > 0.8) & (t == 0)).sum())      # model sure yes, panel says no
    fn = int(((q < 0.2) & (t == 1)).sum())      # model sure no, panel says yes
    w(f"  {c:17s} confident false-pos {fp:2d}   confident false-neg {fn:2d}")

w("")
w("--- calibration: does a higher probability mean a higher hit rate? ---")
flat_t = T.values.ravel()
flat_q = Q.values.ravel()
bins = [0, .1, .2, .4, .6, .8, .9, 1.01]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (flat_q >= lo) & (flat_q < hi)
    if m.sum():
        w(f"  p in [{lo:.1f},{hi:.2f})  n={m.sum():4d}  actually positive: {flat_t[m].mean():.1%}")

w("")
w("--- prevalence: what the model predicts over ALL 4,407 vs the 58 ---")
prev = pd.DataFrame({
    "58_panel_%": (T.mean() * 100),
    "58_model_mean_p_%": (Q.mean() * 100),
    "all4407_mean_p_%": (P[LABELS].mean() * 100),
    "all4407_over_0.5_%": ((P[LABELS] > 0.5).mean() * 100),
}).round(1)
w(prev.to_string())
w("")
w("The 58 are enriched for disease, so the full-corpus rates should be lower.")

w("")
w("CAVEATS")
w("  * 58 studies -> roughly +/-0.05 noise on each AUROC.")
w("  * The panel labeled from IMAGES, not reports, so some disagreement is")
w("    irreducible rather than a labeling error.")

OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"\nwritten -> {OUT_TXT}")
