"""
per_class_pooling.py - should each finding pool its slice views differently?

THE IDEA, taken from the public 0.941 notebook. It scores each study on several slice
windows and then pools those per-window predictions with a rule chosen PER FINDING: max for
Fracture, Contusion and Meniscus, top-2 mean for ACL and MCL, plain mean for Synovitis.
Every model here mean-pools all 96 slices for all 12 findings instead.

The motivation is real: a fracture is visible on two or three slices out of ninety-six, so
averaging dilutes it, while synovitis is diffuse and averaging helps. A pooling rule that is
right for one is wrong for the other.

WHAT THIS SCRIPT CAN AND CANNOT TEST. It reuses the slice-variant predictions already in
runs/others/tta/, which are four different SAMPLINGS of the whole stack, not the contiguous
sliding windows the notebook uses. So max over these four is a weaker proxy: each view still
sees the whole knee, so none of them isolates the two slices a fracture lives on. A negative
here therefore does NOT close the idea; a positive would be strong.

Choosing the rule per finding on the same data you score is the selection bias that made
per-class checkpoint routing look like a +0.0128 gain when it was worth ~0. So the rule is
chosen on K-1 folds and applied to the held-out fold, and the baseline is cross-fitted the
same way.

    python runs/others/per_class_pooling.py
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
TTA = ROOT / "runs" / "others" / "tta"
SEEDS = range(10)


def load_views(split="val"):
    """(views, targets, tasks): the slice-sampling variants, aligned on study id."""
    files = sorted(TTA.glob(f"{split}_*.npz"))
    files = [f for f in files if "geom" not in f.stem and "hflip" not in f.stem]
    if not files:
        raise SystemExit(f"no slice-variant predictions under {TTA}")
    ref = np.load(files[0], allow_pickle=True)
    ids = list(ref["study_ids"])
    order = {s: i for i, s in enumerate(ids)}
    Y = ref["targets"].astype(np.float64)
    tasks = [str(t) for t in ref["tasks"]]
    out, names = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        m = np.empty_like(Y)
        m[[order[s] for s in z["study_ids"]]] = z["logits"].astype(np.float64)
        out.append(m)
        names.append(f.stem.replace(f"{split}_", ""))
    return np.stack(out), Y, tasks, names


def auroc(y, p):
    y = (y > 0.5).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, p)


def pool(V, how):
    """(views, n, tasks) -> (n, tasks) under one rule."""
    if how == "mean":
        return V.mean(0)
    if how == "max":
        return V.max(0)
    if how == "min":
        return V.min(0)
    if how == "top2":
        s = np.sort(V, axis=0)
        return s[-2:].mean(0) if V.shape[0] >= 2 else V.mean(0)
    raise ValueError(how)


RULES = ["mean", "max", "top2", "min"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    V, Y, tasks, names = load_views()
    n, t = Y.shape
    print(f"{V.shape[0]} slice views x {n} studies x {t} findings: {', '.join(names)}\n")

    print("one rule for every finding (in-sample, for orientation only):")
    for r in RULES:
        P = pool(V, r)
        print("  %-6s %.4f" % (r, np.nanmean([auroc(Y[:, j], P[:, j]) for j in range(t)])))

    # per-finding best, in-sample. Same trap as per-class checkpoint routing: macro AUROC is
    # a per-column mean, so picking each column's argmax IS the mean of the column maxima.
    per = {r: np.array([auroc(Y[:, j], pool(V, r)[:, j]) for j in range(t)]) for r in RULES}
    stacked = np.vstack([per[r] for r in RULES])
    print("\nper-finding best rule, IN-SAMPLE: %.4f  (not evidence, see the note above)"
          % np.nanmean(stacked.max(0)))
    for j, tk in enumerate(tasks):
        best = RULES[int(np.nanargmax(stacked[:, j]))]
        print("    %-20s %-6s %.4f   (mean %.4f)" % (tk, best, stacked[:, j].max(),
                                                     per["mean"][j]))

    print(f"\ncross-fitted, {a.folds} folds, {len(list(SEEDS))} seeds:")
    router, fixed = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        fold = np.tile(np.arange(a.folds), n // a.folds + 1)[:n]
        rng.shuffle(fold)
        oof_r, oof_f = np.empty_like(Y), np.empty_like(Y)
        for k in range(a.folds):
            tr, te = fold != k, fold == k
            # choose on the training folds only
            tab = np.vstack([[auroc(Y[tr, j], pool(V[:, tr], r)[:, j]) for j in range(t)]
                             for r in RULES])
            best_per = np.nanargmax(np.where(np.isnan(tab), -np.inf, tab), axis=0)
            best_one = int(np.nanargmax(np.nanmean(tab, axis=1)))
            for j in range(t):
                oof_r[te, j] = pool(V[:, te], RULES[best_per[j]])[:, j]
            oof_f[te] = pool(V[:, te], RULES[best_one])
        m = lambda M: float(np.nanmean([auroc(Y[:, j], M[:, j]) for j in range(t)]))
        router.append(m(oof_r))
        fixed.append(m(oof_f))
        print("  seed %d:  per-finding %.4f   single rule %.4f   delta %+.4f"
              % (seed, router[-1], fixed[-1], router[-1] - fixed[-1]))

    r, f = np.array(router), np.array(fixed)
    d = r - f
    print("\n" + "=" * 70)
    print("  per-finding pooling  %.4f +/- %.4f" % (r.mean(), r.std()))
    print("  one rule for all     %.4f +/- %.4f" % (f.mean(), f.std()))
    print("  DELTA                %+.4f +/- %.4f  (%d/%d seeds positive)"
          % (d.mean(), d.std(), (d > 0).sum(), len(d)))
    print("  plain mean baseline  %.4f"
          % np.nanmean([auroc(Y[:, j], pool(V, "mean")[:, j]) for j in range(t)]))
    print("=" * 70)


if __name__ == "__main__":
    main()
