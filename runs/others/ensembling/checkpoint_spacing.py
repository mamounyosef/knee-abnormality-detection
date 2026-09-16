r"""checkpoint_spacing.py - does SPACING checkpoints out rescue within-run blending?

Blending checkpoints from one run has already been tested here and lost: ten checkpoints of
`mricore_vitb` averaged to 0.8849 against that run's own best single of 0.8935. The obvious
diagnosis is that consecutive checkpoints are nearly the same model, so averaging them buys
noise reduction on an already-converged trajectory and nothing else.

That suggests a different selection rule rather than a different count: take checkpoints that
are FAR APART, so the members differ. This tries several spacing schemes on one fold and
reports each against the honest baseline.

A WARNING ABOUT THE BASELINE. The "best single" is itself chosen on the same val set these
blends are scored on, so it is already optimistically biased. Any blend has to beat a number
that is itself too high. That makes this test conservative in the right direction: a blend
that wins here is winning against a handicap, and a blend that loses by a hair has really
lost by more. Per-class checkpoint routing on this project looked like +0.0128 in-sample and
was worth ~0.0000 cross-fitted, so treat small wins with suspicion.

    python runs/others/ensembling/checkpoint_spacing.py
    python runs/others/ensembling/checkpoint_spacing.py --run convnext_base_25d_18slices_f1
"""

import argparse
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS = ROOT / "runs"


def macro(y, p):
    out = []
    for j in range(y.shape[1]):
        yj = (y[:, j] > 0.5).astype(int)
        if 0 < yj.sum() < len(yj):
            out.append(roc_auc_score(yj, p[:, j]))
    return float(np.nanmean(out)) if out else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="convnext_base_25d_18slices_f1")
    a = ap.parse_args()

    d = RUNS / a.run / "results" / "predictions"
    files = sorted(d.glob("*_val_s*.npz"),
                   key=lambda f: int(re.search(r"_s(\d+)\.npz", f.name).group(1)))
    if not files:
        raise SystemExit("no val predictions in %s" % d)

    steps, L, y = [], [], None
    for f in files:
        z = np.load(f, allow_pickle=True)
        steps.append(int(re.search(r"_s(\d+)\.npz", f.name).group(1)))
        L.append(z["logits"].astype(np.float64))
        if y is None:
            y = z["targets"].astype(np.float64)
    L = np.stack(L)                                    # (n_ckpt, studies, 12)
    solo = np.array([macro(y, x) for x in L])
    n = len(steps)

    print("\n%s: %d checkpoints, steps %d..%d" % (a.run, n, steps[0], steps[-1]))
    print("  best single   %.4f  (step %d)" % (solo.max(), steps[int(solo.argmax())]))
    print("  worst single  %.4f  (step %d)" % (solo.min(), steps[int(solo.argmin())]))
    print("  mean single   %.4f" % solo.mean())

    # rank order: 0 = best checkpoint by val
    order = list(np.argsort(-solo))

    def blend(idx):
        return macro(y, L[list(idx)].mean(axis=0))

    rows = []

    def add(label, idx):
        idx = list(dict.fromkeys(idx))
        if len(idx) < 2:
            return
        rows.append((label, len(idx), blend(idx),
                     ", ".join(str(steps[i]) for i in sorted(idx))))

    # --- the user's schemes: alternate ranks, so members are spread through the ranking
    add("ranks 1,3,5,7 (odd)", order[0:8:2])
    add("ranks 2,4,6,8 (even)", order[1:8:2])
    add("ranks 1,3,5,7,9,11", order[0:12:2])
    add("ranks 2,4,6,8,10,12", order[1:12:2])
    add("ranks 1,4,7,10 (stride 3)", order[0:12:3])

    # --- contiguous top-k, the scheme that already lost, for reference
    for k in (2, 3, 5, 8):
        add("top-%d contiguous" % k, order[:k])

    # --- spacing by STEP rather than by rank: genuinely different points on the trajectory
    for stride in (2, 3, 4):
        add("every %dth checkpoint by step" % stride, range(0, n, stride))
    add("last 8 by step", range(max(0, n - 8), n))
    add("last 12, every 2nd", range(max(0, n - 12), n, 2))
    add("first/middle/last third", [0, n // 2, n - 1])

    # --- widest possible spacing among DECENT checkpoints: top half by score, then spread
    good = sorted([i for i in order[:max(2, n // 2)]])
    add("top half, every 2nd by step", good[::2])
    add("top half, every 3rd by step", good[::3])

    rows.sort(key=lambda r: -r[2])
    best = solo.max()
    print("\n  %-30s %4s %9s %9s" % ("scheme", "n", "AUROC", "vs best"))
    print("  " + "-" * 60)
    for label, k, sc, _ in rows:
        print("  %-30s %4d %9.4f %+9.4f" % (label, k, sc, sc - best))

    win = [r for r in rows if r[2] > best]
    print("\n  %d of %d schemes beat the best single checkpoint" % (len(win), len(rows)))
    if win:
        lab, k, sc, st = win[0]
        print("  best scheme: %s (n=%d) %.4f, +%.4f" % (lab, k, sc, sc - best))
        print("  its steps: %s" % st)
        print("\n  Treat this as a HYPOTHESIS, not a result: both the winning scheme and the")
        print("  baseline were chosen on these same studies. The honest test is whether the")
        print("  SAME fixed rule also wins on another fold.")
    else:
        print("  Nothing beats simply taking the run's own best checkpoint.")


if __name__ == "__main__":
    main()
