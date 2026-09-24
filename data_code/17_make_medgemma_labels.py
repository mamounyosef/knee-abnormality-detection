"""
17_make_medgemma_labels.py - Qwen labels with MedGemma's image opinion blended into five findings.

WHAT. For Effusion, Synovitis, Baker's, Lateral Meniscus and Lateral OA, the new label is Qwen's
and MedGemma's opinions averaged in logit space, after MedGemma is put on Qwen's scale. The other
seven findings stay Qwen's, byte for byte.

    L_q  = logit(Qwen)                          per study, per finding
    L_m  = logit(MedGemma P(yes))               data/16_medgemma15_corpus_all.csv (all six
                                                views, images only, short question)
    L_m' = (L_m - mean(L_m)) / sd(L_m) * sd(L_q) + mean(L_q)     per finding, over all 4,407
    new  = sigmoid(0.5 * L_q + 0.5 * L_m')

WHY THIS FORM (data_code/label_v2 results, audit-58, 24 Sep):
  - MedGemma alone is far worse than Qwen (0.72 vs 0.91); averaged with Qwen on these five it
    lifts the labels 0.9114 -> 0.9309. Held out (choose on half the studies, score the other,
    300 halvings) +0.0189, positive 100%. Blending any of the other seven findings loses.
  - Its raw P(yes) over-calls (mean 0.36-0.87 on the five vs Qwen 0.20-0.37), so it must be
    rescaled before it becomes a training target. Matching the logit mean and spread to Qwen's
    per finding keeps the full gain and keeps the targets on Qwen's scale. The scale is fitted
    on labels only (no panel data), over the whole corpus.
  - 50/50 is not tuned: tuning the weight on half the studies scored worse on the other half.

Writes, keeping every row, column and order of the Qwen files they replace:
    data/17_train_qwen-mg.csv  <- 05_train_qwen24t-med.csv
    data/17_val_qwen-mg.csv    <- 05_val_qwen24t-med.csv
    data/17_train_full_qwen-mg.csv <- 08_train_full.csv (train + val + audit)

    python data_code/17_make_medgemma_labels.py
    python data_code/17_make_medgemma_labels.py --raw   # NO rescale: same logit average on
                                                        # MedGemma's own scale -> *_qwen-mg-raw.csv
The --raw twin exists to test the rescale itself on the leaderboard (both trained in parallel).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
BLEND = ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]
W = 0.5
EPS = 1e-4
RAW = "--raw" in sys.argv
SUFFIX = "qwen-mg-raw" if RAW else "qwen-mg"
OUT = {"05_train_qwen24t-med.csv": f"17_train_{SUFFIX}.csv",
       "05_val_qwen24t-med.csv": f"17_val_{SUFFIX}.csv",
       "08_train_full.csv": f"17_train_full_{SUFFIX}.csv"}


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    fails = []

    def check(cond, msg):
        print(("  OK    " if cond else "  FAIL  ") + msg)
        if not cond:
            fails.append(msg)

    qwen = pd.read_csv(D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID")
    mg = pd.read_csv(D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID")
    ids = list(qwen.index)
    print(f"Qwen {len(qwen)} studies, MedGemma {len(mg)} studies")
    check(set(ids) == set(mg.index) and len(ids) == len(mg), "same 4,407 studies in both")
    check(not mg[TASKS].isna().any().any() and not qwen[TASKS].isna().any().any(), "no missing values")

    # the scale: per finding, over the whole corpus, from labels only
    Lq = logit(qwen.loc[ids, BLEND].values)
    Lm = logit(mg.loc[ids, BLEND].values)
    mq, sq, mm, sm = Lq.mean(0), Lq.std(0), Lm.mean(0), Lm.std(0)
    Lm_q = Lm if RAW else (Lm - mm) / sm * sq + mq      # --raw: MedGemma's own scale
    print("VARIANT:", "RAW (no rescale)" if RAW else "rescaled to Qwen")
    new = qwen.copy()
    new.loc[ids, BLEND] = sigmoid((1 - W) * Lq + W * Lm_q)
    print("\nscale per finding (logit mean / sd):")
    for j, t in enumerate(BLEND):
        print(f"  {t:17s} Qwen {mq[j]:+.3f}/{sq[j]:.3f}  MedGemma {mm[j]:+.3f}/{sm[j]:.3f}  "
              f"-> corpus mean label Qwen {qwen[t].mean():.3f}, raw MedGemma {mg[t].mean():.3f}, "
              f"new {new[t].mean():.3f}")

    print("\nchecks on the combined table:")
    keep = [t for t in TASKS if t not in BLEND]
    check((new[keep].values == qwen[keep].values).all(), f"the other {len(keep)} findings are Qwen's exactly")
    v = new[BLEND].values
    check(np.isfinite(v).all() and (v > 0).all() and (v < 1).all(), "blended labels finite, inside (0, 1)")
    for t in BLEND:
        r = pd.Series(new[t]).corr(qwen[t], method="spearman")
        check(0.3 < r < 0.999, f"{t}: rank correlation with Qwen {r:.3f} (changed, not replaced)")
    aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
    Y = aud[["official_" + t for t in TASKS]].values
    a_old = np.mean([roc_auc_score(Y[:, j], qwen.loc[aud.index, t]) for j, t in enumerate(TASKS)])
    a_new = np.mean([roc_auc_score(Y[:, j], new.loc[aud.index, t]) for j, t in enumerate(TASKS)])
    check(abs(a_old - 0.9114) < 5e-4, f"audit-58 macro AUROC, Qwen labels {a_old:.4f} (expected 0.9114)")
    check(a_new > a_old + 0.015, f"audit-58 macro AUROC, new labels {a_new:.4f} ({a_new - a_old:+.4f})")
    for j, t in enumerate(TASKS):
        if t in BLEND:
            print(f"      {t:17s} {roc_auc_score(Y[:, j], qwen.loc[aud.index, t]):.3f} -> "
                  f"{roc_auc_score(Y[:, j], new.loc[aud.index, t]):.3f}")

    print("\nwriting the training files (same rows, columns and order as the Qwen ones):")
    for src, dst in OUT.items():
        s = pd.read_csv(D / src)
        o = s.copy()
        u = s["StudyInstanceUID"]
        check(u.isin(new.index).all(), f"{src}: every study has a label")
        # 08_train_full gives the 58 AUDIT studies the PANEL's own 0/1 labels (the best labels
        # there are); those rows are kept exactly as they are, only Qwen rows get the blend
        llm = (s["source"] != "audit").values if "source" in s else np.ones(len(s), bool)
        same = np.allclose(s.loc[llm, TASKS].values, qwen.loc[u[llm], TASKS].values, atol=1e-9)
        check(same, f"{src}: its {llm.sum()} LLM-labelled rows equal 03_report_labels_qwen24t-med exactly")
        o.loc[llm, BLEND] = new.loc[u[llm], BLEND].values
        o.to_csv(D / dst, index=False)
        back = pd.read_csv(D / dst)
        check(list(back.columns) == list(s.columns) and len(back) == len(s)
              and (back["StudyInstanceUID"] == u).all(), f"{dst}: {len(back)} rows, same columns and order")
        check(np.allclose(back[keep].values, s[keep].values), f"{dst}: the other findings unchanged")
        if (~llm).any():
            check(np.array_equal(back.loc[~llm, TASKS].values, s.loc[~llm, TASKS].values),
                  f"{dst}: the {(~llm).sum()} audit rows keep the panel's labels exactly")
        check(np.allclose(back.loc[llm, BLEND].values, new.loc[u[llm], BLEND].values),
              f"{dst}: the five blended findings are the new labels")
    print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
