"""
per_class_checkpoint_routing.py - is "one checkpoint per finding" a real gain, or selection bias?

THE IDEA UNDER TEST
A run saves a validation every few hundred steps, and each one wrote its raw per-study
logits. For each of the 12 findings, pick the checkpoint that scores best on THAT finding,
then assemble one prediction matrix whose column j comes from finding j's own best
checkpoint. Score the macro AUROC of that matrix.

WHY IT LOOKS FREE, AND WHY THAT IS THE TRAP
Macro AUROC is the mean of 12 per-column AUROCs, and each column is computed entirely
independently of the others. So routing column j to the checkpoint that maximises column j
makes the macro EXACTLY the mean of the per-column maxima. That is >= any single
checkpoint's macro by construction, always, on any data, even if every checkpoint is pure
noise. The "gain" you read off the validation set is the expected maximum of N noisy
estimates, not skill. There is no interaction between columns to exploit.

SO THE ONLY HONEST QUESTION IS: does the checkpoint that wins a finding on one set of
studies still win it on studies you did not look at? This script answers that by
CROSS-FITTING. The validation set is split into K folds; the per-finding winners are chosen
on K-1 folds and their predictions are recorded for the held-out fold; after all folds every
study has an out-of-fold prediction, and the macro over that matrix is an unbiased estimate.

The baseline is cross-fitted the SAME way (pick one checkpoint by macro on K-1 folds, apply
to the held-out fold), so both sides pay the same selection noise and the comparison is
like for like. Comparing a cross-fitted router against an in-sample single checkpoint would
rig the test in the router's favour.

Everything is read from results/predictions/*.npz. No GPU, no Modal, no network.

    python runs/others/per_class_checkpoint_routing.py
    python runs/others/per_class_checkpoint_routing.py --run convnext_large --folds 10
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

RUNS = Path(__file__).resolve().parent.parent
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def load_checkpoints(runs: str, split: str = "val"):
    """Every cached validation of one or more runs (comma separated).

    Returns (labels, targets, logits stack, tasks) with logits (n_checkpoints, n_studies,
    n_tasks). Checkpoints are aligned on study_ids, so a run that was resumed onto a
    different worker still stacks correctly, and pooling across runs is safe as long as
    they share a validation split (the relabel runs do).
    """
    names = [r.strip() for r in runs.split(",") if r.strip()]
    files = []
    for r in names:
        d = RUNS / r / "results" / "predictions"
        files += sorted(d.glob(f"*_{split}_s*.npz"), key=lambda p: int(p.stem.split("_s")[-1]))
    if not files:
        raise SystemExit(f"no {split} predictions for {names}")

    ref = np.load(files[0], allow_pickle=True)
    ids, tasks = list(ref["study_ids"]), [str(t) for t in ref["tasks"]]
    order = {s: i for i, s in enumerate(ids)}
    Y = ref["targets"].astype(np.float64)

    labels, mats = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if [str(t) for t in z["tasks"]] != tasks:
            print(f"  skipped {f.name}: different task list")
            continue
        idx = [order.get(s, -1) for s in z["study_ids"]]
        if min(idx) < 0 or len(idx) != len(ids):
            print(f"  skipped {f.name}: study set does not match")
            continue
        m = np.empty_like(Y)
        m[idx] = z["logits"].astype(np.float64)
        mats.append(m)
        tag = f.parent.parent.parent.name if len(names) > 1 else ""
        labels.append((tag + ":" if tag else "") + f.stem.replace(f"_{split}", ""))
    return labels, Y, np.stack(mats), tasks


def auroc(y, p):
    """AUROC, or nan when a fold happens to hold only one class of a rare finding."""
    y = (y > 0.5).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, p)


def per_class_table(Y, L):
    """(n_checkpoints, n_tasks) of per-finding AUROC, computed on all studies."""
    return np.array([[auroc(Y[:, j], L[c][:, j]) for j in range(Y.shape[1])]
                     for c in range(L.shape[0])])


def cross_fit(Y, L, folds, seed):
    """Out-of-fold macro AUROC for the router and for a single-checkpoint baseline.

    Both choose on the SAME K-1 training folds and are scored on the SAME held-out fold,
    so the only difference between them is per-finding routing vs one checkpoint for all.
    """
    n, t = Y.shape
    rng = np.random.default_rng(seed)
    fold_of = np.tile(np.arange(folds), n // folds + 1)[:n]
    rng.shuffle(fold_of)

    oof_router = np.empty_like(Y)
    oof_single = np.empty_like(Y)
    picks = np.zeros((folds, t), dtype=int)

    for k in range(folds):
        tr, te = fold_of != k, fold_of == k
        tab = per_class_table(Y[tr], L[:, tr, :])          # choose using training folds only
        with np.errstate(invalid="ignore"):
            best_per_class = np.nanargmax(np.where(np.isnan(tab), -np.inf, tab), axis=0)
            best_single = int(np.nanargmax(np.nanmean(tab, axis=1)))
        picks[k] = best_per_class
        for j in range(t):
            oof_router[te, j] = L[best_per_class[j]][te, j]
        oof_single[te] = L[best_single][te]

    macro = lambda M: float(np.nanmean([auroc(Y[:, j], M[:, j]) for j in range(t)]))
    return macro(oof_router), macro(oof_single), picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="convnext_base_relabel",
                    help="one run, or several comma separated to pool their checkpoints")
    ap.add_argument("--split", default="val")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    labels, Y, L, tasks = load_checkpoints(a.run, a.split)
    n_ck, n, t = L.shape
    print(f"\n{a.run}: {n_ck} cached {a.split} checkpoints x {n} studies x {t} findings\n")

    # ---------- 1. the naive number, and why it is not a result ----------
    tab = per_class_table(Y, L)
    macro_each = np.nanmean(tab, axis=1)
    best_ck = int(np.nanargmax(macro_each))
    naive = float(np.nanmean(np.nanmax(tab, axis=0)))
    print(f"best SINGLE checkpoint      {labels[best_ck]:<20} macro {macro_each[best_ck]:.4f}")
    print(f"per-finding routing, IN-SAMPLE                       macro {naive:.4f}"
          f"   (+{naive - macro_each[best_ck]:.4f})")
    print("  ^ this number is not evidence: macro AUROC is a per-column mean, so routing")
    print("    each column to its own argmax IS the mean of the column maxima, by")
    print("    construction, even for random predictions.\n")

    # how many distinct checkpoints the in-sample router even uses
    winners = np.nanargmax(np.where(np.isnan(tab), -np.inf, tab), axis=0)
    print("  in-sample winner per finding:")
    for j, tk in enumerate(tasks):
        print(f"    {tk:<28} {labels[winners[j]]:<20} {tab[winners[j], j]:.4f}"
              f"   (best single: {tab[best_ck, j]:.4f})")

    # ---------- 2. the honest number ----------
    print(f"\ncross-fitted, {a.folds} folds, {len(SEEDS)} seeds "
          f"(choose on {a.folds - 1}/{a.folds} of the studies, score on the rest):\n")
    r, s = [], []
    for seed in SEEDS:
        ro, si, picks = cross_fit(Y, L, a.folds, seed)
        r.append(ro)
        s.append(si)
        stab = np.mean([len(set(picks[:, j])) == 1 for j in range(t)])
        print(f"  seed {seed}:  router {ro:.4f}   single {si:.4f}   "
              f"delta {ro - si:+.4f}   same pick across folds: {stab:.0%}")

    r, s = np.array(r), np.array(s)
    d = r - s
    print("\n" + "=" * 74)
    print(f"  router (per-finding)   {r.mean():.4f} +/- {r.std():.4f}")
    print(f"  single checkpoint      {s.mean():.4f} +/- {s.std():.4f}")
    print(f"  DELTA                  {d.mean():+.4f} +/- {d.std():.4f}"
          f"   ({(d > 0).sum()}/{len(d)} seeds positive)")
    print(f"  in-sample delta was    {naive - macro_each[best_ck]:+.4f}"
          f"   <- the part that was selection bias: "
          f"{(naive - macro_each[best_ck]) - d.mean():+.4f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
