r"""slot_routed_blend.py - let convnext_base_5slot vote ONLY on the findings its extra plane serves.

THE IDEA. `convnext_base_5slot` adds a `cor_plain` slot (a fifth, non-fluid coronal series).
Coronal plain is the plane radiologists read the MEDIAL COLLATERAL LIGAMENT and the CRUCIATES
on, and the run moved exactly those: MCL +0.0103, ACL +0.0040. The other ten findings lost to
dilution - one model, 20% more input, same capacity - so the run was dropped and its real gain
went with it.

Routing fixes that: give it a vote on MCL and ACL and nothing else.

WHY THIS IS NOT THE ROUTING THAT ALREADY FAILED. Per-class CHECKPOINT routing was tested here
and was pure selection bias: picking each finding's best checkpoint reads as +0.0128 in-sample
and is worth ~0.0000 cross-fitted, because macro AUROC is a per-column mean and routing every
column to its own argmax IS the mean of the column maxima by construction.

The difference is WHERE THE CHOICE COMES FROM. Here the two findings are named in advance
from anatomy - which plane shows which structure - not chosen by reading the val scores. That
makes it a hypothesis with two degrees of freedom, not twelve.

It is still tested cross-fitted below, because a good story is not evidence.

    python runs/others/ensembling/slot_routed_blend.py
"""

import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS = ROOT / "runs"

ROUTED = "convnext_base_5slot"
# named from anatomy BEFORE looking at any score: cor_plain is the MCL/cruciate plane
ROUTE_TASKS = ["MCL", "ACL"]

SEVEN = ["convnext_base_relabel", "mricore_vitb_cross_attention",
         "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
         "convnext_base_25d_18slices", "swinv2_base_in22k",
         "convnext_base_25d_poolattn"]


def cols(y, p):
    out = []
    for j in range(y.shape[1]):
        yj = (y[:, j] > 0.5).astype(int)
        out.append(roc_auc_score(yj, p[:, j]) if 0 < yj.sum() < len(yj) else np.nan)
    return np.array(out)


def macro(y, p, rows=None):
    if rows is not None:
        y, p = y[rows], p[rows]
    return float(np.nanmean(cols(y, p)))


def best_val(run):
    best, bs = None, -1.0
    for f in sorted(glob.glob(str(RUNS / run / "results/predictions/*_val_s*.npz"))):
        z = np.load(f, allow_pickle=True)
        if z["logits"].shape[1] != 12:
            continue
        s = np.nanmean(cols(z["targets"], z["logits"]))
        if s > bs:
            best, bs = z, s
    return best


ref = best_val(SEVEN[0])
TASKS = [str(t) for t in ref["tasks"]]
order = {s: i for i, s in enumerate(list(ref["study_ids"]))}
y = ref["targets"].astype(np.float64)
n = len(y)


def aligned(run):
    z = best_val(run)
    a = np.empty((n, 12))
    a[[order[s] for s in list(z["study_ids"])]] = z["logits"].astype(np.float64)
    return a


A = {m: aligned(m) for m in SEVEN + [ROUTED]}
ridx = [TASKS.index(t) for t in ROUTE_TASKS]

# --- 1. did the extra plane actually move those findings? -------------------------
print("\n1. %s against its parent convnext_base_relabel, per finding" % ROUTED)
p_cols = cols(y, A["convnext_base_relabel"])
s_cols = cols(y, A[ROUTED])
print("  %-20s %9s %9s %9s" % ("finding", "parent", "5slot", "delta"))
for j in np.argsort(-(s_cols - p_cols)):
    mark = "  <- routed" if j in ridx else ""
    print("  %-20s %9.4f %9.4f %+9.4f%s" % (TASKS[j], p_cols[j], s_cols[j],
                                            s_cols[j] - p_cols[j], mark))
print("  %-20s %9.4f %9.4f %+9.4f"
      % ("MACRO", np.nanmean(p_cols), np.nanmean(s_cols),
         np.nanmean(s_cols) - np.nanmean(p_cols)))


def blend(members, routed_in, rows=None):
    """Flat blend; if routed_in, ROUTED joins the vote on ROUTE_TASKS only."""
    base = np.mean([A[m] for m in members], axis=0)
    if not routed_in:
        return base
    out = base.copy()
    k = len(members)
    for j in ridx:
        out[:, j] = (base[:, j] * k + A[ROUTED][:, j]) / (k + 1)
    return out


# --- 2. in-sample ------------------------------------------------------------------
print("\n2. in-sample on all %d val studies (NOT evidence, see 3)" % n)
pairs = [("convnext_base_relabel alone", ["convnext_base_relabel"]),
         ("the 0.932 seven", SEVEN)]
for label, mem in pairs:
    a = macro(y, blend(mem, False))
    b = macro(y, blend(mem, True))
    c = macro(y, np.mean([A[m] for m in mem] + [A[ROUTED]], axis=0))
    print("  %-30s  base %.4f   +5slot ROUTED %.4f (%+.4f)   +5slot on ALL 12 %.4f (%+.4f)"
          % (label, a, b, b - a, c, c - a))

# --- 3. cross-fitted ---------------------------------------------------------------
# The routing rule is fixed in advance, so there is nothing to fit - but the BASELINE is
# still the same studies, so score both on held-out folds and compare like for like.
print("\n3. cross-fitted, 5 folds x 8 seeds (the honest number)")
for label, mem in pairs:
    da = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        fold = np.tile(np.arange(5), n // 5 + 1)[:n]
        rng.shuffle(fold)
        oof_b = np.empty((n, 12))
        oof_r = np.empty((n, 12))
        for k in range(5):
            te = np.where(fold == k)[0]
            oof_b[te] = blend(mem, False)[te]
            oof_r[te] = blend(mem, True)[te]
        da.append(macro(y, oof_r) - macro(y, oof_b))
    da = np.array(da)
    print("  %-30s  delta %+.4f +/- %.4f   (%d/8 seeds positive)"
          % (label, da.mean(), da.std(), (da > 0).sum()))

print("""
  Note what cross-fitting can and cannot do here. The routing rule has no free
  parameters - the two findings were named from anatomy - so there is no fitting to
  undo, and the cross-fitted delta should simply agree with the in-sample one. If it
  does, the gain is real; if it collapses, the anatomy story was a coincidence.""")
