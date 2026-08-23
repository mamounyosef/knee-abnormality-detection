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

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


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
        m = sc.compute_metrics(Y, P, TASKS)
        rows.append((m["macro_auroc"], name, P.mean(), m))
    rows.sort(reverse=True)

    print(f"  panel positive rate: {Y.mean():.3f}\n")
    print(f"  {'labeler':38s} {'macro AUROC':>11s}  {'mean p':>7s}")
    for score, name, meanp, _ in rows:
        print(f"  {name:38s} {score:11.4f}  {meanp:7.3f}")

    print("\n  per finding:")
    head = "  " + " " * 18 + "".join(f"{n[:11]:>12s}" for _, n, _, _ in rows[:5])
    print(head)
    for i, t in enumerate(TASKS):
        cells = "".join(
            f"{sc.compute_metrics(Y[:, [i]], d.loc[ids, TASKS].values.astype(float)[:, [i]], [t])['macro_auroc']:12.3f}"
            for _, n, _, _ in rows[:5] for d in [frames[n]])
        print(f"  {t:18s}{cells}")


if __name__ == "__main__":
    main()
