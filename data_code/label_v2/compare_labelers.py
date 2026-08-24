"""
compare_labelers.py - score every labeler run against the panel, on shared studies.

Each run writes data/03_report_labels_<tag>_validate58.csv. Runs cover different
subsets (--holdout scores 29, a full --validate scores 58, and a model that refuses a
report scores one fewer), so comparing headline numbers across them is wrong. This
intersects the studies first and scores everyone on the same rows.

    python data_code/label_v2/compare_labelers.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def _scores(Y, P):
    """AUROC plus the three metrics that survive a hard labeler.

    AUROC only ranks, so it is not comparable between a soft run and a hard one: two
    distinct values makes almost every positive/negative pair a tie, and a tie is worth
    0.5. Binarising the soft qwen24t labels at 0.5 costs 0.058 AUROC while changing no
    DECISION at all. Balanced accuracy, F1 and Brier read the decisions, so they compare
    a hard run and a soft run on equal terms.
    """
    from sklearn.metrics import f1_score
    m = sc.compute_metrics(Y, P, TASKS)
    H = (P >= 0.5).astype(float)
    sens = (H * Y).sum() / max(Y.sum(), 1)
    spec = ((1 - H) * (1 - Y)).sum() / max((1 - Y).sum(), 1)
    f1 = sum(f1_score(Y[:, i], H[:, i], zero_division=0)
             for i in range(Y.shape[1])) / Y.shape[1]
    return {"auroc": m["macro_auroc"], "bacc": (sens + spec) / 2, "f1": f1,
            "brier": float(((P - Y) ** 2).mean()), "meanp": float(P.mean()),
            "hard": len(np.unique(P)) <= 2}


def main():
    audit = pd.read_csv(ROOT / "data/05_audit58.csv").set_index("StudyInstanceUID")
    runs = {"Qwen3.8-27B (corpus, old prompt)": ROOT / "data/03_report_labels.csv"}
    for f in sorted((ROOT / "data").glob("03_report_labels_*_validate58.csv")):
        runs[f.stem.replace("03_report_labels_", "").replace("_validate58", "")] = f

    frames = {}
    for name, f in runs.items():
        try:
            frames[name] = pd.read_csv(f).set_index("StudyInstanceUID")
        except Exception as e:                               # noqa: BLE001
            print(f"  [skip] {name}: {e}")

    # the smallest run decides the shared rows, so every number below is comparable
    common = None
    for d in frames.values():
        common = set(d.index) if common is None else common & set(d.index)
    common &= set(audit.index)
    ids = [i for i in audit.index if i in common]
    if not ids:
        raise SystemExit("no studies shared by every run")

    Y = audit.loc[ids, [f"official_{t}" for t in TASKS]].values.astype(float)
    sc._banner(f"LABELERS vs THE PANEL  ({len(ids)} shared studies)", "\U0001F9EA")
    rows = []
    for name, d in frames.items():
        P = d.loc[ids, TASKS].values.astype(float)
        rows.append((name, _scores(Y, P)))
    rows.sort(key=lambda r: -r[1]["bacc"])          # ranked on the metric both modes share

    print(f"  panel positive rate: {Y.mean():.3f}")
    print("  ranked by BALANCED ACCURACY, the only column a hard run can win\n")
    print(f"  {'labeler':34s} {'bal.acc':>8s} {'F1':>7s} {'Brier':>7s} "
          f"{'AUROC':>7s} {'mean p':>7s}")
    for name, s in rows:
        flag = " *" if s["hard"] else ""
        print(f"  {name + flag:34s} {s['bacc']:8.4f} {s['f1']:7.4f} {s['brier']:7.4f} "
              f"{s['auroc']:7.4f} {s['meanp']:7.3f}")
    if any(s["hard"] for _, s in rows):
        print("\n  * hard 0/1 labeler. Its AUROC column is not comparable to the soft")
        print("    runs - ties cost it about 0.06 regardless of how good it is.")

    print("\n  per finding (AUROC, soft runs only):")
    soft = [(n, s) for n, s in rows if not s["hard"]][:5]
    head = "  " + " " * 18 + "".join(f"{n[:11]:>12s}" for n, _ in soft)
    print(head)
    for i, t in enumerate(TASKS):
        cells = "".join(
            f"{sc.compute_metrics(Y[:, [i]], frames[n].loc[ids, TASKS].values.astype(float)[:, [i]], [t])['macro_auroc']:12.3f}"
            for n, _ in soft)
        print(f"  {t:18s}{cells}")

if __name__ == "__main__":
    main()
