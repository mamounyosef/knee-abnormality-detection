r"""synovitis_weight_sweep.py - how much of the Synovitis column should convnext_base_fluid own?

convnext_base_fluid contributes to ONE finding and nothing else. It is the best Synovitis
model we have (0.8767) and beats the whole seven-model ensemble on that column (0.8597), but
it is WORSE than the ensemble on Effusion (-0.0067), the other finding it was fine-tuned on -
so it is restricted to Synovitis alone.

The open question is the weight. w = 0 is the current ensemble; w = 1 is a full takeover where
the other seven models are ignored for that column. Somewhere between is usually better than
either, because the ensemble still carries information the specialist lacks even when the
specialist is better on average.

Both readings are reported: in-sample, and cross-fitted with the weight CHOSEN on K-1 folds
and applied to the held-out fold, which is the only version that cannot flatter itself.

    python runs/others/ensembling/synovitis_weight_sweep.py
"""

import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS = ROOT / "runs"

SPEC = "convnext_base_fluid"
TASK = "Synovitis"
SEVEN = ["convnext_base_relabel", "mricore_vitb_cross_attention",
         "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
         "convnext_base_25d_18slices", "swinv2_base_in22k",
         "convnext_base_25d_poolattn"]
GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def auc(yj, pj):
    yj = (yj > 0.5).astype(int)
    if np.isnan(pj).any() or not (0 < yj.sum() < len(yj)):
        return np.nan
    return roc_auc_score(yj, pj)


def best_val(run):
    best, bs = None, -1.0
    for f in sorted(glob.glob(str(RUNS / run / "results/predictions/*_val_s*.npz"))):
        z = np.load(f, allow_pickle=True)
        s = np.nanmean([auc(z["targets"][:, j], z["logits"][:, j])
                        for j in range(z["logits"].shape[1])])
        if s > bs:
            best, bs = z, s
    return best


ref = best_val(SEVEN[0])
TASKS = [str(t) for t in ref["tasks"]]
order = {s: i for i, s in enumerate(list(ref["study_ids"]))}
y = ref["targets"].astype(np.float64)
n = len(y)
J = TASKS.index(TASK)


def aligned(run):
    z = best_val(run)
    a = np.full((n, 12), np.nan)
    rows = [order[s] for s in list(z["study_ids"])]
    lg = z["logits"].astype(np.float64)
    for j, t in enumerate([str(x) for x in z["tasks"]]):
        if t in TASKS:
            a[rows, TASKS.index(t)] = lg[:, j]
    return a


BASE = np.nanmean([aligned(m) for m in SEVEN], axis=0)
S = aligned(SPEC)

# logits from different models are on different scales, so rank-normalise the column before
# mixing. Without this, w is not a meaningful blend weight - it is dominated by whichever
# model happens to produce larger numbers, and AUROC only cares about order anyway.
def rank01(v):
    r = np.argsort(np.argsort(v)).astype(np.float64)
    return r / (len(r) - 1)


base_r, spec_r = rank01(BASE[:, J]), rank01(S[:, J])
base_macro = np.nanmean([auc(y[:, j], BASE[:, j]) for j in range(12)])
other = [auc(y[:, j], BASE[:, j]) for j in range(12) if j != J]


def macro_at(w, rows=None):
    col = (1 - w) * base_r + w * spec_r
    if rows is None:
        return np.nanmean(other + [auc(y[:, J], col)])
    o = [auc(y[rows][:, j], BASE[rows][:, j]) for j in range(12) if j != J]
    return np.nanmean(o + [auc(y[rows][:, J], col[rows])])


print("\n%s contributes to %s ONLY (and to nothing else)" % (SPEC, TASK))
print("  ensemble alone on %-12s %.4f" % (TASK, auc(y[:, J], BASE[:, J])))
print("  %s alone       %.4f" % (SPEC.replace("convnext_base_", "").ljust(12), auc(y[:, J], S[:, J])))
print("\n  %6s %14s %12s" % ("w", "Synovitis", "macro"))
print("  " + "-" * 34)
for w in GRID:
    col = (1 - w) * base_r + w * spec_r
    print("  %6.1f %14.4f %12.4f%s" % (w, auc(y[:, J], col), macro_at(w),
                                       "   <- current" if w == 0 else ""))

print("\ncross-fitted: weight chosen on 4/5 of the studies, scored on the 5th")
deltas, picks = [], []
for seed in range(10):
    rng = np.random.default_rng(seed)
    fold = np.tile(np.arange(5), n // 5 + 1)[:n]
    rng.shuffle(fold)
    oof_b = np.empty(n)
    oof_r = np.empty(n)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        best_w, best_s = 0.0, -1.0
        for w in GRID:
            s = auc(y[tr, J], ((1 - w) * base_r + w * spec_r)[tr])
            if not np.isnan(s) and s > best_s:
                best_w, best_s = w, s
        picks.append(best_w)
        oof_b[te] = base_r[te]
        oof_r[te] = ((1 - best_w) * base_r + best_w * spec_r)[te]
    a = np.nanmean(other + [auc(y[:, J], oof_r)])
    b = np.nanmean(other + [auc(y[:, J], oof_b)])
    deltas.append(a - b)
d = np.array(deltas)
print("  delta %+.4f +/- %.4f   (%d/10 seeds positive)" % (d.mean(), d.std(), (d > 0).sum()))
print("  weights chosen: %s" % dict(sorted(((w, picks.count(w)) for w in set(picks)),
                                           key=lambda kv: -kv[1])))
print("\n  a-priori w=1.0 (full takeover) is %+.4f of macro" % (macro_at(1.0) - base_macro))
