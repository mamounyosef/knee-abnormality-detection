"""
confident_learning.py - do the image models find the LLM's labeling mistakes?

The labeler only ever reads the report. The ensemble only ever sees the images. Where
they confidently disagree, one of them is wrong - and 51% of the labeler's residual
error sits on findings the report never mentions, which is exactly the information the
images carry and the text does not.

The audit58 is the only place all three views exist at once: LLM label, ensemble
prediction, and the expert panel's answer. So the question is directly measurable:

    on cells where the ensemble and the LLM disagree, how often is the ENSEMBLE right?

Above 50% the ensemble is finding real label errors, and the same rule can be applied
to the 4,200 studies that have no panel labels. At or below 50% the idea is dead.

Nothing here costs an API call: it reads saved .npz logits, the corpus labels, and the
panel CSV.

    python data_code/label_v2/confident_learning.py
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

# the checkpoint each ensemble member was submitted with, so this measures the same
# blend that scored 0.913 rather than some other combination
MEMBERS = [("convnext_small_4slot", 4200), ("convnext_base_4slot", 3200),
           ("mricore_vitb", 7000), ("radimagenet_resnet50", 5000)]


def _load_member(run: str, step: int):
    """That run's audit logits, keyed by study."""
    hits = sorted((ROOT / "runs" / run / "results" / "predictions").glob(f"*audit_s{step}.npz"))
    if not hits:
        avail = sorted(p.name for p in
                       (ROOT / "runs" / run / "results" / "predictions").glob("*audit*.npz"))
        raise SystemExit(f"{run}: no audit predictions at step {step}. have: {avail}")
    z = np.load(hits[0], allow_pickle=True)
    order = [list(z["tasks"]).index(t) for t in TASKS]      # never trust column order
    return (pd.DataFrame(z["logits"][:, order], index=[str(s) for s in z["study_ids"]],
                         columns=TASKS),
            pd.DataFrame(z["targets"][:, order], index=[str(s) for s in z["study_ids"]],
                         columns=TASKS))


def wilson(k: int, n: int, z: float = 1.96):
    """Binomial confidence interval that stays sane at small n, unlike the normal one."""
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    logits, truth = [], None
    for run, step in MEMBERS:
        lg, tg = _load_member(run, step)
        logits.append(lg)
        truth = tg if truth is None else truth
    llm = pd.read_csv(ROOT / "data" / "03_report_labels.csv").set_index("StudyInstanceUID")
    # one id list, fixed before anything is indexed, so the three views stay aligned
    ids = sorted(set.intersection(*[set(d.index) for d in logits])
                 & set(truth.index) & set(llm.index))
    # logit-space average, matching how the submitted ensemble blends
    M = 1 / (1 + np.exp(-np.mean([d.loc[ids, TASKS].values for d in logits], axis=0)))
    Y = truth.loc[ids, TASKS].values.astype(float)          # the panel
    L = llm.loc[ids, TASKS].values.astype(float)            # the LLM's soft label

    sc._banner(f"DO THE IMAGES FIND THE LABELER'S MISTAKES?  ({len(ids)} panel studies)",
               "\U0001F50D")
    print(f"  ensemble : {' + '.join(r for r, _ in MEMBERS)}")
    print(f"  labels   : data/03_report_labels.csv (the corpus labels the model trained on)")
    print(f"  cells    : {M.size}\n")

    # sanity: on this set, who is closer to the panel overall?
    for n, P in (("LLM label (from the report)", L), ("ensemble (from the images)", M)):
        H = (P >= 0.5).astype(float)
        print(f"  {n:32s} accuracy {(H == Y).mean():.4f}   "
              f"AUROC {sc.compute_metrics(Y, P, TASKS)['macro_auroc']:.4f}")

    print("\n  DISPUTED CELLS - the two disagree, both at least `margin` from 0.5:")
    print(f"  {'margin':>7s} {'disputed':>9s} {'% of all':>9s} {'ensemble right':>15s} "
          f"{'95% CI':>16s}")
    for margin in (0.0, 0.1, 0.2, 0.3, 0.4):
        lo, hi = 0.5 - margin, 0.5 + margin
        conf = ((M <= lo) | (M >= hi)) & ((L <= lo) | (L >= hi))
        disp = conf & ((M >= 0.5) != (L >= 0.5))
        n = int(disp.sum())
        if n == 0:
            print(f"  {margin:7.1f} {0:9d}")
            continue
        win = int((((M >= 0.5).astype(float) == Y) & disp).sum())
        c = wilson(win, n)
        print(f"  {margin:7.1f} {n:9d} {disp.mean():8.1%} {win / n:15.1%} "
              f"  [{c[0]:.2f}, {c[1]:.2f}]")

    # where do the disputes live? the silent findings are the ones with headroom
    margin = 0.2
    lo, hi = 0.5 - margin, 0.5 + margin
    conf = ((M <= lo) | (M >= hi)) & ((L <= lo) | (L >= hi))
    disp = conf & ((M >= 0.5) != (L >= 0.5))
    print(f"\n  per finding at margin {margin} (n = {int(disp.sum())} disputes):")
    print(f"  {'finding':18s} {'disputes':>9s} {'ensemble right':>15s}")
    for i, t in enumerate(TASKS):
        n = int(disp[:, i].sum())
        if not n:
            print(f"  {t:18s} {0:9d}")
            continue
        win = int((((M[:, i] >= 0.5) == Y[:, i]) & disp[:, i]).sum())
        print(f"  {t:18s} {n:9d} {win / n:15.1%}")

    print("\n  reading it: above 50% the ensemble wins its disputes, so its disagreements")
    print("  flag real label errors and the rule generalises to the 4,200 unlabeled")
    print("  studies. A CI straddling 0.50 means 58 studies cannot answer this.")


if __name__ == "__main__":
    main()
