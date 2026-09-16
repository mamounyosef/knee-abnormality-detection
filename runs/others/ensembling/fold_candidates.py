r"""fold_candidates.py - price and score every candidate fold ensemble against the 9 h cap.

Two constraints decide this, and only one of them is measurable:

  COST is measurable and exact. Decode is paid once per study PER GEOMETRY GROUP, and
  `n_slices` splits a group, so the 18-slice folds do not ride with the 24-slice
  `convnext_base_25d_group3`. The model here is calibrated against the one real measurement
  we have (the 9-member stress run: 3 groups, 2.503 s decode total + 12.699 s GPU = 15.202
  s/study), so it reproduces that number exactly.

  QUALITY is NOT measurable here. The folds cannot be scored on the 704-study val set -
  559-589 of those studies are in each fold's training data. Audit-58 is the only leak-free
  shared surface and its SE is ~0.02, so it can reject a disaster but cannot rank candidates
  separated by 0.00x. Audit is printed for that rejection role only.

So the choice is made on cost plus principle, with audit as a sanity floor.

    python runs/others/ensembling/fold_candidates.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

RUNS = ROOT / "runs"
DECODE_PER_GROUP = 2.503 / 3          # calibrated, see docstring
GPU = {"convnext_base_relabel": 2.045, "convnext_base_trauma": 1.602,
       "convnext_base_wear": 1.605, "convnext_base_fluid": 1.604,
       "mricore_vitb_cross_attention": 1.509,
       "mricore_vitb_mae_cross_attention_full": 1.641,
       "dinov3_convnext_base": 1.616, "convnext_base_25d_group3": 0.559,
       "convnext_base_25d_poolattn": 0.516,
       # from the earlier 7-member stress run; swinv2 is the priciest member
       # we have ever shipped and owns its own decode geometry
       "swinv2_base_in22k": 5.197,
       "mricore_vitb_mae_cross_attention": 1.641,
       "convnext_base_25d_18slices": 0.516,
       # NOT measured yet; ViT-S/14 at 336 px is 22M params against the folds' 88M,
       # and the folds measured 0.220 s/study, so this is a deliberate over-estimate
       "dinov2_vits14_25d": 0.40}
FOLD_GPU = 0.516
FOLDS = ["convnext_base_25d_18slices_f%d" % k for k in range(5)]
DV2 = "dinov2_vits14_25d"
GROUP = ["convnext_base_relabel", "convnext_base_trauma", "convnext_base_wear",
         "convnext_base_fluid"]
STAND = ["mricore_vitb_cross_attention", "mricore_vitb_mae_cross_attention_full",
         "dinov3_convnext_base", "convnext_base_25d_group3", "convnext_base_25d_poolattn"]


def macro(y, p):
    out = []
    for j in range(y.shape[1]):
        yj = (y[:, j] > 0.5).astype(int)
        if 0 < yj.sum() < len(yj):
            out.append(roc_auc_score(yj, p[:, j]))
    return float(np.nanmean(out)) if out else float("nan")


def best_audit(run, order, tasks):
    """The audit prediction at the step with the best VAL score (never chosen on audit)."""
    d = RUNS / run / "results" / "predictions"
    vb, vstep = -1.0, None
    for f in sorted(d.glob("*_val_s*.npz")):
        z = np.load(f, allow_pickle=True)
        s = macro(z["targets"], z["logits"])
        if s > vb:
            vb, vstep = s, f.name.split("_s")[-1].replace(".npz", "")
    m = [f for f in d.glob("*_audit_s*.npz") if f.name.endswith("_s%s.npz" % vstep)]
    if not m:
        m = sorted(d.glob("*_audit_s*.npz"))
    if not m:
        return None
    z = np.load(m[0], allow_pickle=True)
    out = np.full((len(order), 12), np.nan)
    rows = [order[s] for s in list(z["study_ids"])]
    cols = [str(t) for t in z["tasks"]]
    lg = z["logits"].astype(np.float64)
    for j, t in enumerate(cols):
        if t in tasks:
            out[rows, tasks.index(t)] = lg[:, j]
    return out


ref = np.load(sorted((RUNS / FOLDS[0] / "results/predictions").glob("*_audit_s*.npz"))[0],
              allow_pickle=True)
order = {s: i for i, s in enumerate(list(ref["study_ids"]))}
TASKS = [str(t) for t in ref["tasks"]]
ay = ref["targets"].astype(np.float64)

A = {}
for r in GROUP + STAND + FOLDS + [DV2]:
    a = best_audit(r, order, TASKS)
    if a is not None:
        A[r] = a


def geom(run):
    return sc.loader_geometry(sc.load_config(RUNS / run, verbose=False))


def price(members):
    groups = {geom(m) for m in members}
    total = len(groups) * DECODE_PER_GROUP + sum(GPU.get(m, FOLD_GPU) for m in members)
    return total, len(groups), total * 1500 / 3600


def score(votes):
    """votes: list of arrays, each already one VOTE (a sub-ensemble is pre-averaged)."""
    return macro(ay, np.nanmean(votes, axis=0))


grp_vote = np.nanmean([A[m] for m in GROUP], axis=0)
fold_vote = np.nanmean([A[m] for m in FOLDS], axis=0)

# LB 0.932, the best score we have actually achieved (submission 56234763 / 56256423).
# Its 18-slice member is the FOLDS' OWN PARENT, trained on 3,638 of the same studies, so
# keeping it beside its five folds is near-duplicate weight rather than diversity.
SIX = ["convnext_base_relabel", "mricore_vitb_cross_attention",
       "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
       "convnext_base_25d_18slices", "swinv2_base_in22k"]
SEVEN = SIX + ["convnext_base_25d_poolattn"]

CANDIDATES = [
    ("A  5 folds alone", FOLDS, [A[m] for m in FOLDS]),
    ("A+ 5 folds + dinov2 (equal votes)", FOLDS + [DV2],
     [A[m] for m in FOLDS] + [A[DV2]]),
    ("A* folds ONE vote + dinov2 ONE vote", FOLDS + [DV2], [fold_vote, A[DV2]]),
    ("A- dinov2 alone", [DV2], [A[DV2], A[DV2]]),
    ("B  shipped 9 (=0.931 on LB)", GROUP + STAND, [grp_vote] + [A[m] for m in STAND]),
    ("C  shipped 9 + folds as ONE vote", GROUP + STAND + FOLDS,
     [grp_vote] + [A[m] for m in STAND] + [fold_vote]),
    ("D  shipped 9 + folds as FIVE votes", GROUP + STAND + FOLDS,
     [grp_vote] + [A[m] for m in STAND] + [A[m] for m in FOLDS]),
    ("E  drop group3, + folds one vote",
     GROUP + [m for m in STAND if m != "convnext_base_25d_group3"] + FOLDS,
     [grp_vote] + [A[m] for m in STAND if m != "convnext_base_25d_group3"] + [fold_vote]),
    ("F  drop both 25d, + folds one vote",
     GROUP + [m for m in STAND if not m.startswith("convnext_base_25d")] + FOLDS,
     [grp_vote] + [A[m] for m in STAND if not m.startswith("convnext_base_25d")] + [fold_vote]),
    ("G  4 diverse + folds one vote",
     ["convnext_base_relabel"] + STAND[:3] + FOLDS,
     [A["convnext_base_relabel"]] + [A[m] for m in STAND[:3]] + [fold_vote]),
    ("H  the 0.932 SEVEN (flat)", SEVEN, [A[m] for m in SEVEN if m in A]),
    ("I  0.932 seven + folds one vote", SEVEN + FOLDS,
     [A[m] for m in SEVEN if m in A] + [fold_vote]),
    ("I+ SUBMITTED 12 + dinov2", SEVEN + FOLDS + [DV2],
     [A[m] for m in SEVEN if m in A] + [fold_vote, A[DV2]]),
    ("J  0.932 seven, parent OUT, + folds",
     [m for m in SEVEN if m != "convnext_base_25d_18slices"] + FOLDS,
     [A[m] for m in SEVEN if m in A and m != "convnext_base_25d_18slices"] + [fold_vote]),
    ("K2 0.932 seven MINUS SWINV2 ONLY, + folds",
     [m for m in SEVEN if m != "swinv2_base_in22k"] + FOLDS,
     [A[m] for m in SEVEN if m in A and m != "swinv2_base_in22k"] + [fold_vote]),
    ("K  0.932 seven minus swinv2, + folds",
     [m for m in SEVEN if m not in ("swinv2_base_in22k", "convnext_base_25d_18slices")] + FOLDS,
     [A[m] for m in SEVEN if m in A and m not in ("swinv2_base_in22k",
                                                  "convnext_base_25d_18slices")] + [fold_vote]),
]

print("\n%-38s %3s %6s %7s %8s  %s" % ("candidate", "mem", "grp", "s/study", "hours", "audit-58"))
print("-" * 84)
for label, members, votes in CANDIDATES:
    s, g, h = price(members)
    print("%-38s %3d %6d %7.2f %7.2f h  %.4f  %s"
          % (label, len(members), g, s, h, score(votes),
             "OVER CAP" if h >= 9 else ""))

print("""
Audit-58 is 58 studies, SE ~0.02. It is here to reject a disaster, not to rank these.
Do not pick a candidate because its audit is 0.001 higher.""")
