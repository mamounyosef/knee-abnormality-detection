r"""val_audit_agreement.py - how far do the 704-study val split and the 58-study audit agree?

The two splits disagree often enough that which one we believe has decided real submissions.
Audit-58 carries HUMAN PANEL labels, the same kind of ground truth the leaderboard uses,
while val carries our LLM labeller's. Over 167 submission pairs audit ordered them correctly
88.6% of the time against val's 85.0%, and on the 14 pairs where the two DISAGREED audit was
right 10 times to val's 4. But that was measured on submissions, not on runs.

This asks the question on the runs themselves, and separates two very different things:

  1. WITHIN a run, across its own training curve. Do the two splits rise and fall together?
     This is what tells you whether audit is usable as a training signal at all.
  2. ACROSS runs, comparing each run's FINAL converged score. This is the question that
     actually matters for choosing ensemble members, and it is NOT the same correlation -
     a pair of splits can track each other beautifully inside a run and still rank two
     finished runs the wrong way round, because the between-run spread is far smaller than
     the within-run spread.

Convergence matters because early training is dominated by "the model is getting better at
everything", which inflates agreement for free. Only the plateau says whether the splits
agree about SKILL rather than about progress.

    python runs/others/val_audit_agreement.py
    python runs/others/val_audit_agreement.py --tail-frac 0.25 --min-evals 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "runs"


def spearman(a, b):
    """Rank correlation, ties averaged. No scipy dependency."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def load_runs(min_evals):
    out = []
    for p in sorted(RUNS.glob("*/results/val_log.csv")):
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if "val_macro_auroc" not in d or "audit_macro_auroc" not in d:
            continue
        d = d.dropna(subset=["val_macro_auroc", "audit_macro_auroc"])
        if len(d) < min_evals:
            continue
        out.append((p.parent.parent.name, d.reset_index(drop=True)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-evals", type=int, default=6,
                    help="skip runs with fewer validations than this")
    ap.add_argument("--tail-frac", type=float, default=0.33,
                    help="the last fraction of a run's evaluations = its converged phase")
    args = ap.parse_args()

    runs = load_runs(args.min_evals)
    if not runs:
        raise SystemExit("no runs with both val and audit columns")

    print("=" * 94)
    print("  VAL (704, LLM labels)  vs  AUDIT-58 (human panel labels)")
    print("  %d runs with >= %d validations each" % (len(runs), args.min_evals))
    print("=" * 94)

    print("\n1. WITHIN each run: do the two splits move together over training?")
    print("   'converged' = the last %.0f%% of that run's validations\n" % (args.tail_frac * 100))
    print("   %-46s %6s %8s %8s %9s" % ("run", "evals", "r(all)", "r(conv)", "audit-val"))
    print("   " + "-" * 82)
    rows = []
    for name, d in runs:
        v, a = d["val_macro_auroc"].to_numpy(), d["audit_macro_auroc"].to_numpy()
        k = max(3, int(round(len(d) * args.tail_frac)))
        r_all, r_tail = pearson(v, a), pearson(v[-k:], a[-k:])
        gap = float(np.mean(a[-k:]) - np.mean(v[-k:]))
        rows.append((name, len(d), r_all, r_tail, gap,
                     float(v.max()), float(a.max()),
                     int(d["step"].iloc[v.argmax()]), int(d["step"].iloc[a.argmax()])))
    for r in sorted(rows, key=lambda x: -(x[3] if np.isfinite(x[3]) else -9)):
        print("   %-46s %6d %8s %8s %+9.4f"
              % (r[0][:46], r[1],
                 "%.3f" % r[2] if np.isfinite(r[2]) else "-",
                 "%.3f" % r[3] if np.isfinite(r[3]) else "-", r[4]))

    allr = np.array([r[2] for r in rows], float)
    tailr = np.array([r[3] for r in rows], float)
    gaps = np.array([r[4] for r in rows], float)
    print("\n   median r over the whole curve   : %+.3f" % np.nanmedian(allr))
    print("   median r over the converged tail: %+.3f" % np.nanmedian(tailr))
    print("   runs where the tail correlation is NEGATIVE: %d of %d"
          % (int(np.sum(tailr < 0)), int(np.sum(np.isfinite(tailr)))))
    print("   audit sits %+.4f above val on average once converged (median %+.4f)"
          % (float(np.nanmean(gaps)), float(np.nanmedian(gaps))))

    print("\n2. THE PEAK: do the two splits peak at the same point in training?")
    same = sum(1 for r in rows if r[7] == r[8])
    dist = [abs(r[7] - r[8]) for r in rows]
    print("   both peak at the SAME step in %d of %d runs (%.0f%%)"
          % (same, len(rows), 100.0 * same / len(rows)))
    print("   median distance between the two peaks: %d steps" % int(np.median(dist)))
    print("   so picking a checkpoint by audit instead of val moves it by that much")

    print("\n3. ACROSS runs: does audit RANK finished runs the way val does?")
    print("   (this is the question that decides ensemble membership)\n")
    best_v = np.array([r[5] for r in rows])
    best_a = np.array([r[6] for r in rows])
    print("   runs compared            : %d" % len(rows))
    print("   Pearson  r (peak val vs peak audit): %+.3f" % pearson(best_v, best_a))
    print("   Spearman r (the ranking itself)    : %+.3f" % spearman(best_v, best_a))
    print("   val spread   : %.4f to %.4f  (range %.4f)"
          % (best_v.min(), best_v.max(), np.ptp(best_v)))
    print("   audit spread : %.4f to %.4f  (range %.4f)"
          % (best_a.min(), best_a.max(), np.ptp(best_a)))

    n = len(rows)
    agree = disagree = 0
    for i in range(n):
        for j in range(i + 1, n):
            dv, da = best_v[i] - best_v[j], best_a[i] - best_a[j]
            if dv == 0 or da == 0:
                continue
            if (dv > 0) == (da > 0):
                agree += 1
            else:
                disagree += 1
    print("\n   pairwise ordering over all %d pairs: agree %d, DISAGREE %d (%.1f%%)"
          % (agree + disagree, agree, disagree,
             100.0 * disagree / max(agree + disagree, 1)))

    # The pairs that matter are the ones where val says the difference is real.
    for thresh in (0.0010, 0.0030, 0.0055):
        ag = dg = 0
        for i in range(n):
            for j in range(i + 1, n):
                dv, da = best_v[i] - best_v[j], best_a[i] - best_a[j]
                if abs(dv) < thresh or da == 0:
                    continue
                ag, dg = (ag + 1, dg) if (dv > 0) == (da > 0) else (ag, dg + 1)
        print("   restricted to pairs val separates by >= %.4f: agree %d, DISAGREE %d (%.1f%%)"
              % (thresh, ag, dg, 100.0 * dg / max(ag + dg, 1)))

    print("\n4. THE BIGGEST DISAGREEMENTS (val rank vs audit rank)\n")
    rv = pd.Series(best_v).rank(ascending=False)
    ra = pd.Series(best_a).rank(ascending=False)
    d = (rv - ra).abs()
    print("   %-46s %8s %8s %6s %6s" % ("run", "val", "audit", "rkV", "rkA"))
    print("   " + "-" * 80)
    for i in d.sort_values(ascending=False).index[:12]:
        print("   %-46s %8.4f %8.4f %6d %6d"
              % (rows[i][0][:46], best_v[i], best_a[i], int(rv[i]), int(ra[i])))

    print("\n" + "=" * 94)
    return 0


if __name__ == "__main__":
    sys.exit(main())
