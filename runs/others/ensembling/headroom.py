r"""headroom.py - macro AUROC is a MEAN over 12 findings, so where is the 0.02 actually lost?

"Get from 0.932 to 0.955" is not a single target, it is twelve. The metric averages the
twelve per-finding AUROCs, so a finding stuck at 0.84 drags the mean down by exactly
(0.93 - 0.84)/12 on its own, and lifting it is worth twelve times less effort than lifting
the whole model. This prints which columns are cheap to fix and what each is worth.

It also prints the prevalence, because the weak columns tend to be the rare ones - and a rare
finding's AUROC is both noisier and more improvable by targeting it directly.

    python runs/others/ensembling/headroom.py
"""

import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS = ROOT / "runs"

# the members of the 0.932 seven
SEVEN = ["convnext_base_relabel", "mricore_vitb_cross_attention",
         "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
         "convnext_base_25d_18slices", "swinv2_base_in22k",
         "convnext_base_25d_poolattn"]


def macro_cols(y, p):
    out = []
    for j in range(y.shape[1]):
        yj = (y[:, j] > 0.5).astype(int)
        out.append(roc_auc_score(yj, p[:, j]) if 0 < yj.sum() < len(yj) else np.nan)
    return np.array(out)


def best_val(run):
    best, bs = None, -1.0
    for f in sorted(glob.glob(str(RUNS / run / "results/predictions/*_val_s*.npz"))):
        z = np.load(f, allow_pickle=True)
        if z["logits"].shape[1] != 12:
            continue
        s = np.nanmean(macro_cols(z["targets"], z["logits"]))
        if s > bs:
            best, bs = z, s
    return best


ref = best_val(SEVEN[0])
tasks = [str(t) for t in ref["tasks"]]
order = {s: i for i, s in enumerate(list(ref["study_ids"]))}
y = ref["targets"].astype(np.float64)

L = []
for m in SEVEN:
    z = best_val(m)
    a = np.empty((len(order), 12))
    a[[order[s] for s in list(z["study_ids"])]] = z["logits"].astype(np.float64)
    L.append(a)
blend = np.mean(L, axis=0)

cols = macro_cols(y, blend)
prev = (y > 0.5).mean(axis=0)
macro = np.nanmean(cols)

print("\nthe 0.932 seven, per finding on the 704-study val set")
print("  %-20s %8s %10s %12s" % ("finding", "AUROC", "prevalence", "cost to macro"))
print("  " + "-" * 56)
best_col = np.nanmax(cols)
for j in np.argsort(cols):
    print("  %-20s %8.4f %9.1f%% %12.4f"
          % (tasks[j], cols[j], 100 * prev[j], (best_col - cols[j]) / 12))
print("  " + "-" * 56)
print("  %-20s %8.4f" % ("MACRO", macro))

print("\nwhat lifting the weak columns is worth (macro is a mean over 12):")
for k in (1, 2, 3, 4):
    worst = np.argsort(cols)[:k]
    for lift in (0.03, 0.05, 0.08):
        new = cols.copy()
        new[worst] = np.minimum(new[worst] + lift, 0.99)
        print("    lift the %d weakest by %.2f -> macro %.4f  (%+.4f)"
              % (k, lift, np.nanmean(new), np.nanmean(new) - macro))
    print()

gap = 0.955 - 0.932
print("  the LB gap to 0.955 is %+.3f." % gap)
print("  On val that is roughly the same size, and the table above says it is NOT")
print("  reachable by lifting one column: even +0.08 on the four weakest buys ~%.4f."
      % (np.nanmean(np.minimum(cols + np.where(np.isin(np.arange(12), np.argsort(cols)[:4]),
                                               0.08, 0.0), 0.99)) - macro))
print("  Every column has to move, which means a better MODEL or better LABELS,")
print("  not a better blend of the models we already have.")
