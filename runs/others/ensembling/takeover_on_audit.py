r"""takeover_on_audit.py - test the FIXED routing table on data it was never chosen on.

WHY THIS IS A DIFFERENT QUESTION FROM THE CROSS-FITTED ONE.

Cross-fitted greedy asks "does the PROCEDURE of searching for column owners generalise". It
re-runs the search on every fold, so a column whose winner flips between folds is scored with
a different model each time and the instability itself costs points. That is the right test
if we intended to ship a search.

We do not. We would ship ONE fixed table: this finding -> this model, decided once. The
honest test of a fixed rule is to apply it, unchanged, to data that had no part in choosing
it. Audit-58 is the only such surface we have: 58 studies with PANEL labels, never used to
select anything, and outside the 4,342-study fold universe entirely.

Its weakness is size, not independence. With 58 studies a single finding's AUROC has a
standard error around 0.05-0.10, so this can show a direction and rule out a disaster; it
cannot certify +0.001. Read it for sign and rough magnitude, and note how many of the routed
columns move the same way they did on val - agreement across two independent sets is worth
more than either number alone.

    python runs/others/ensembling/takeover_on_audit.py
"""

import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS = ROOT / "runs"

SEVEN = ["convnext_base_relabel", "mricore_vitb_cross_attention",
         "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
         "convnext_base_25d_18slices", "swinv2_base_in22k",
         "convnext_base_25d_poolattn"]

# the table as chosen on the 704-study val set, fixed and unchanged here
ROUTE = {"Synovitis": "convnext_base_fluid",                      # a priori: built for it
         "Fracture": "mricore_vitb_mae_cross_attention",          # found on val
         "PF OA": "dinov3_convnext_base",                         # found on val
         "Medial OA": "coatnet_rmlp2_384"}                        # found on val
APRIORI = {"Synovitis"}          # the only one named before looking at scores


def auc(yj, pj):
    yj = (yj > 0.5).astype(int)
    if np.isnan(pj).any() or not (0 < yj.sum() < len(yj)):
        return np.nan
    return roc_auc_score(yj, pj)


def picked_step(run):
    """The step with the best VAL score - never chosen on audit."""
    best, bs = None, -1.0
    for f in sorted(glob.glob(str(RUNS / run / "results/predictions/*_val_s*.npz"))):
        z = np.load(f, allow_pickle=True)
        s = np.nanmean([auc(z["targets"][:, j], z["logits"][:, j])
                        for j in range(z["logits"].shape[1])])
        if s > bs:
            best, bs = f, s
    return best.split("_s")[-1].replace(".npz", "") if best else None


def audit_of(run, order, tasks):
    st = picked_step(run)
    fs = [f for f in glob.glob(str(RUNS / run / "results/predictions/*_audit_s*.npz"))
          if f.endswith("_s%s.npz" % st)] or sorted(
        glob.glob(str(RUNS / run / "results/predictions/*_audit_s*.npz")))
    if not fs:
        return None
    z = np.load(fs[0], allow_pickle=True)
    a = np.full((len(order), 12), np.nan)
    rows = [order[s] for s in list(z["study_ids"])]
    lg = z["logits"].astype(np.float64)
    for j, t in enumerate([str(x) for x in z["tasks"]]):
        if t in tasks:
            a[rows, tasks.index(t)] = lg[:, j]
    return a


ref = np.load(sorted(glob.glob(str(RUNS / SEVEN[0] / "results/predictions/*_audit_s*.npz")))[0],
              allow_pickle=True)
TASKS = [str(t) for t in ref["tasks"]]
order = {s: i for i, s in enumerate(list(ref["study_ids"]))}
y = ref["targets"].astype(np.float64)

A = {}
for m in set(SEVEN) | set(ROUTE.values()):
    a = audit_of(m, order, TASKS)
    if a is not None:
        A[m] = a

missing = [m for m in set(SEVEN) | set(ROUTE.values()) if m not in A]
if missing:
    print("  no audit predictions for: %s" % ", ".join(missing))

BASE = np.nanmean([A[m] for m in SEVEN if m in A], axis=0)
bc = np.array([auc(y[:, j], BASE[:, j]) for j in range(12)])

print("\naudit-58: the fixed routing table, applied to data it was never chosen on")
print("  %-18s %10s %10s %9s  %s" % ("finding", "ensemble", "routed", "delta", "owner"))
print("  " + "-" * 72)
take = BASE.copy()
same_sign = 0
VAL_DELTA = {"Synovitis": +0.0171, "Fracture": +0.0168, "PF OA": +0.0083, "Medial OA": +0.0054}
for t, m in ROUTE.items():
    j = TASKS.index(t)
    if m not in A:
        print("  %-18s  (no audit predictions for %s)" % (t, m))
        continue
    s = auc(y[:, j], A[m][:, j])
    take[:, j] = A[m][:, j]
    d = s - bc[j]
    same_sign += d > 0
    print("  %-18s %10.4f %10.4f %+9.4f  %s%s"
          % (t, bc[j], s, d, m.replace("convnext_base_", ""),
             "   (a priori)" if t in APRIORI else ""))

rc = np.array([auc(y[:, j], take[:, j]) for j in range(12)])
print("\n  macro %.4f -> %.4f  (%+.4f)" % (np.nanmean(bc), np.nanmean(rc),
                                           np.nanmean(rc) - np.nanmean(bc)))
print("  %d of %d routed columns moved the SAME direction as on val" % (same_sign, len(ROUTE)))

# the a-priori-only table, for comparison
only = BASE.copy()
for t in APRIORI:
    if ROUTE[t] in A:
        only[:, TASKS.index(t)] = A[ROUTE[t]][:, TASKS.index(t)]
oc = np.array([auc(y[:, j], only[:, j]) for j in range(12)])
print("  Synovitis-only table: macro %.4f  (%+.4f)"
      % (np.nanmean(oc), np.nanmean(oc) - np.nanmean(bc)))

print("""
  58 studies. A single finding's AUROC here has a standard error near 0.05-0.10, so
  no individual delta below that is meaningful on its own. What IS informative is
  whether the columns move the same way they did on the 704 val studies: two
  independent sets agreeing on a direction is much harder to get by chance than one
  set showing a large number.""")
