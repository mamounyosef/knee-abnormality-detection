r"""per_label_scan.py - for each of the 12 findings, what is the BEST thing in the whole repo?

Every run saved raw per-study logits at every validation, so the repo already holds ~1,600
scored checkpoints. This scans all of them, per finding, and asks whether routing a finding
to its own best checkpoint (or to the average of its top 3 / top 5) beats the 0.932 seven on
that column by enough to be real.

WHY THE IN-SAMPLE TABLES ARE NOT THE ANSWER, AND ARE PRINTED ANYWAY
Macro AUROC is a mean of 12 independent per-column AUROCs, so routing column j to whatever
maximises column j makes the macro the mean of the column maxima - larger than any single
model BY CONSTRUCTION, on any data, even for pure noise. With ~1,600 candidates the expected
maximum of that many noisy estimates is large. So the in-sample tables answer "what won" and
NOT "what will win again".

Two guards, and the second is the one that decides:

  MARGIN   a candidate must beat the ensemble on that finding by more than a threshold.
           Default is one bootstrap standard error of the PAIRED difference, computed per
           finding, which is the right scale: both sides score the same studies, so only
           their disagreement carries noise.

  CROSS-FIT   choose the winners on 4/5 of the studies, score them on the 5th, rotate, repeat
           over several seeds. Nothing is ever chosen and scored on the same rows. The
           baseline is cross-fitted the same way, so both sides pay the same selection noise.

TOP-K IS AVERAGED IN RANK SPACE. Different runs put their logits on wildly different scales,
and AUROC only reads order, so averaging ranks is both scale-free and what the metric sees.

LEAKY RUNS ARE EXCLUDED. The five convnext folds, the five dinov2 folds and convnext_base_full
all trained on studies inside this validation set, so their val AUROCs are not comparable to
anything and would win every column for the wrong reason.

    python runs/others/ensembling/per_label_scan.py
    python runs/others/ensembling/per_label_scan.py --margin 0.02 --seeds 20
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS = ROOT / "runs"

TASKS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
         'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

# the 0.932 seven - the baseline every candidate has to beat
SEVEN = ["convnext_base_relabel", "mricore_vitb_cross_attention",
         "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
         "convnext_base_25d_18slices", "swinv2_base_in22k",
         "convnext_base_25d_poolattn"]

# trained on studies inside this val set, so their val scores are meaningless here
LEAKY = ({"convnext_base_25d_18slices_f%d" % i for i in range(5)}
         | {"dinov2_vits14_25d_f%d" % i for i in range(5)}
         | {"convnext_base_full"})


def auc(y, p):
    y = (np.asarray(y) > 0.5).astype(int)
    p = np.asarray(p, float)
    if np.isnan(p).any() or not (0 < y.sum() < len(y)):
        return np.nan
    return roc_auc_score(y, p)


# --------------------------------------------------------------------------------------
# load every scored checkpoint, aligned to one study order
# --------------------------------------------------------------------------------------
def load_all(verbose=True):
    files = sorted(glob.glob(str(RUNS / "*" / "results" / "predictions" / "*_val_s*.npz")))
    ref = None
    for f in files:                                  # a 12-task run fixes the order
        z = np.load(f, allow_pickle=True)
        if len(z["tasks"]) == 12:
            ref = z
            break
    order = {s: i for i, s in enumerate(list(ref["study_ids"]))}
    n = len(order)
    y = np.full((n, 12), np.nan)
    for j, t in enumerate([str(x) for x in ref["tasks"]]):
        y[:, TASKS.index(t)] = ref["targets"][:, j]

    cands, skipped = [], 0
    for f in files:
        run = Path(f).parts[-4]
        if run in LEAKY:
            skipped += 1
            continue
        z = np.load(f, allow_pickle=True)
        ids = [str(s) for s in z["study_ids"]]
        if not set(ids) <= set(order):               # a different split entirely
            skipped += 1
            continue
        rows = [order[s] for s in ids]
        if len(rows) < 0.9 * n:                      # partial coverage, not comparable
            skipped += 1
            continue
        m = np.full((n, 12), np.nan)
        for j, t in enumerate([str(x) for x in z["tasks"]]):
            if t in TASKS:
                m[rows, TASKS.index(t)] = z["logits"][:, j]
        step = re.search(r"_s(\d+)\.npz$", f)
        cands.append(("%s@%s" % (run, step.group(1) if step else "?"), run, m))
    if verbose:
        print("loaded %d checkpoints from %d runs (%d skipped: leaky, or a different split)"
              % (len(cands), len({c[1] for c in cands}), skipped))
    return y, cands, order


def rank_norm(col):
    """Ranks scaled to [0,1]; NaN-safe. AUROC reads only order, so this is lossless."""
    out = np.full(len(col), np.nan)
    ok = ~np.isnan(col)
    if ok.sum():
        out[ok] = rankdata(col[ok]) / ok.sum()
    return out


def combine(mats, j, rows=None):
    """Mean of rank-normalised column j across several checkpoints."""
    stack = []
    for m in mats:
        c = m[:, j] if rows is None else m[rows, j]
        stack.append(rank_norm(c))
    return np.nanmean(np.vstack(stack), axis=0)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=None,
                    help="required gain over the ensemble; default = 1 bootstrap SE per finding")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()

    y, cands, order = load_all()
    n = len(y)
    names = [c[0] for c in cands]
    mats = [c[2] for c in cands]

    # ---- the baseline: the 0.932 seven, best checkpoint of each member -----------------
    base = np.full((n, 12), np.nan)
    for mem in SEVEN:
        idx = [i for i, c in enumerate(cands) if c[1] == mem]
        best_i, best_s = None, -1
        for i in idx:
            s = np.nanmean([auc(y[:, j], mats[i][:, j]) for j in range(12)])
            if s == s and s > best_s:
                best_i, best_s = i, s
        if best_i is None:
            raise SystemExit("no predictions for baseline member %s" % mem)
        for j in range(12):
            col = rank_norm(mats[best_i][:, j])
            base[:, j] = col if np.isnan(base[:, j]).all() else base[:, j] + col
    base /= len(SEVEN)
    bcols = np.array([auc(y[:, j], base[:, j]) for j in range(12)])
    print("\nbaseline = the 0.932 seven, macro %.4f\n" % np.nanmean(bcols))

    # ---- per finding: score every checkpoint ------------------------------------------
    rng = np.random.default_rng(0)
    scores = np.full((len(cands), 12), np.nan)
    for j in range(12):
        for i in range(len(cands)):
            scores[i, j] = auc(y[:, j], mats[i][:, j])

    # paired bootstrap SE of (candidate - ensemble) on each finding, for the margin
    se = np.zeros(12)
    for j in range(12):
        good = np.where(~np.isnan(scores[:, j]))[0]
        top = good[np.argsort(-scores[good, j])[0]]
        d = []
        for _ in range(a.boot):
            b = rng.integers(0, n, n)
            if not (0 < (y[b, j] > 0.5).sum() < n):
                continue
            d.append(auc(y[b, j], mats[top][b, j]) - auc(y[b, j], base[b, j]))
        se[j] = np.nanstd(d) if d else np.nan

    # ---- the three tables -------------------------------------------------------------
    picks = {}
    for k in (1, 3, 5):
        print("=" * 100)
        print("TOP-%d per finding%s   (in-sample: what won, NOT what will win again)"
              % (k, "" if k == 1 else ", averaged in rank space"))
        print("=" * 100)
        print("  %-18s %8s %8s %9s %8s  %s" % ("finding", "ensemble", "routed", "gain",
                                               "vs 1 SE", "source"))
        keep = {}
        for j in range(12):
            good = np.where(~np.isnan(scores[:, j]))[0]
            top = good[np.argsort(-scores[good, j])[:k]]
            p = combine([mats[i] for i in top], j)
            s = auc(y[:, j], p)
            gain = s - bcols[j]
            thr = a.margin if a.margin is not None else se[j]
            ok = gain > thr
            if ok:
                keep[j] = list(top)
            src = ", ".join(names[i] for i in top[:2]) + (" ..." if k > 2 else "")
            print("  %-18s %8.4f %8.4f %+9.4f %8.2f  %s%s"
                  % (TASKS[j], bcols[j], s, gain, gain / thr if thr else np.nan,
                     "KEEP " if ok else "     ", src[:52]))
        routed = bcols.copy()
        for j, top in keep.items():
            routed[j] = auc(y[:, j], combine([mats[i] for i in top], j))
        print("  %-18s %8.4f %8.4f %+9.4f   (%d of 12 findings routed)"
              % ("MACRO", np.nanmean(bcols), np.nanmean(routed),
                 np.nanmean(routed) - np.nanmean(bcols), len(keep)))
        picks[k] = keep
        print()

    # ---- the honest number: choose on 4/5, score on the 5th ---------------------------
    print("=" * 100)
    print("CROSS-FITTED - winners chosen on 4/5 of the studies, scored on the 5th")
    print("nothing is chosen and scored on the same rows; the baseline pays the same cost")
    print("=" * 100)
    print("  %-10s %10s %10s %12s %14s" % ("scheme", "mean", "std", "seeds>0", "vs in-sample"))
    for k in (1, 3, 5):
        deltas = []
        for seed in range(a.seeds):
            r = np.random.default_rng(1000 + seed)
            fold = r.integers(0, a.folds, n)
            oof = base.copy()
            for f in range(a.folds):
                tr = np.where(fold != f)[0]
                te = np.where(fold == f)[0]
                for j in range(12):
                    if not (0 < (y[tr, j] > 0.5).sum() < len(tr)):
                        continue
                    sc = np.array([auc(y[tr, j], m[tr, j]) for m in mats])
                    bl = auc(y[tr, j], base[tr, j])
                    good = np.where(~np.isnan(sc))[0]
                    if not len(good):
                        continue
                    top = good[np.argsort(-sc[good])[:k]]
                    thr = a.margin if a.margin is not None else se[j]
                    if np.nanmean(sc[top]) - bl <= thr:
                        continue                      # not worth routing on this fold
                    oof[te, j] = combine([mats[i] for i in top], j, rows=te)
            m_oof = np.nanmean([auc(y[:, j], oof[:, j]) for j in range(12)])
            deltas.append(m_oof - np.nanmean(bcols))
        d = np.array(deltas)
        ins = np.nanmean([auc(y[:, j], combine([mats[i] for i in picks[k][j]], j))
                          if j in picks[k] else bcols[j] for j in range(12)]) - np.nanmean(bcols)
        print("  top-%-7d %+10.5f %10.5f %9d/%d %+14.5f"
              % (k, d.mean(), d.std(), (d > 0).sum(), len(d), ins))
    print("\n  'vs in-sample' is what the table above claimed. The gap between the two")
    print("  columns is selection bias - the part that does not survive being tested")
    print("  on studies the choice never saw.")


if __name__ == "__main__":
    main()
