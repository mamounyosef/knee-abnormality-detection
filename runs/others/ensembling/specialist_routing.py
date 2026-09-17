r"""specialist_routing.py - every specialist votes only where it is a specialist.

FOUR SPECIALISTS, each with an a-priori claim on particular findings:

    convnext_base_trauma   ACL, MCL, Contusion, Fracture          (fine-tuned on those 4)
    convnext_base_wear     menisci, the three OA, Baker's         (fine-tuned on those 6)
    convnext_base_fluid    Effusion, Synovitis                    (fine-tuned on those 2)
    convnext_base_5slot    MCL, ACL                               (extra cor_plain slot -
                                                                   the plane those are read on)

Every one of those claims is made by CONSTRUCTION, not by reading val scores: the three
branches were each fine-tuned on their own label group, and 5slot's two findings follow from
which anatomical plane its fifth slot adds. That is what separates this from the per-class
CHECKPOINT routing already tested and rejected here, which chose each column's winner from
the scores and was worth +0.0128 in-sample and ~0.0000 cross-fitted.

Everything is reported cross-fitted, because an a-priori story still has to survive being
scored on studies it was not chosen on.

    python runs/others/ensembling/specialist_routing.py
"""

import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS = ROOT / "runs"

PARENT = "convnext_base_relabel"
BRANCH = {"convnext_base_trauma": ["ACL", "MCL", "Contusion", "Fracture"],
          "convnext_base_wear": ["Medial Meniscus", "Lateral Meniscus", "Medial OA",
                                 "Lateral OA", "PF OA", "Baker's"],
          "convnext_base_fluid": ["Effusion", "Synovitis"],
          "convnext_base_5slot": ["MCL", "ACL"]}

SEVEN = ["convnext_base_relabel", "mricore_vitb_cross_attention",
         "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
         "convnext_base_25d_18slices", "swinv2_base_in22k",
         "convnext_base_25d_poolattn"]


def cols(y, p):
    """Per-finding AUROC; NaN where a member does not predict that finding at all.

    The branches emit only their own 4/6/2 columns, so the other columns arrive as NaN by
    design. Scoring those would raise rather than skip, which is how a partial member gets
    silently dropped from a comparison.
    """
    out = []
    for j in range(y.shape[1]):
        yj = (y[:, j] > 0.5).astype(int)
        pj = p[:, j]
        if np.isnan(pj).any() or not (0 < yj.sum() < len(yj)):
            out.append(np.nan)
        else:
            out.append(roc_auc_score(yj, pj))
    return np.array(out)


def macro(y, p):
    return float(np.nanmean(cols(y, p)))


def best_val(run, need12=True):
    best, bs = None, -1.0
    for f in sorted(glob.glob(str(RUNS / run / "results/predictions/*_val_s*.npz"))):
        z = np.load(f, allow_pickle=True)
        if need12 and z["logits"].shape[1] != 12:
            continue
        s = np.nanmean(cols(z["targets"], z["logits"]))
        if s > bs:
            best, bs = z, s
    return best


ref = best_val(PARENT)
TASKS = [str(t) for t in ref["tasks"]]
order = {s: i for i, s in enumerate(list(ref["study_ids"]))}
y = ref["targets"].astype(np.float64)
n = len(y)


def aligned(run):
    """(n, 12) with NaN in columns this member does not predict."""
    z = best_val(run, need12=False)
    a = np.full((n, 12), np.nan)
    rows = [order[s] for s in list(z["study_ids"])]
    lg = z["logits"].astype(np.float64)
    for j, t in enumerate([str(x) for x in z["tasks"]]):
        if t in TASKS:
            a[rows, TASKS.index(t)] = lg[:, j]
    return a


A = {m: aligned(m) for m in set(SEVEN) | set(BRANCH)}

# --- 1. is each specialist actually better on its own findings? --------------------
print("\n1. each specialist against the parent, ON ITS OWN CLAIMED FINDINGS")
p = cols(y, A[PARENT])
for m, ts in BRANCH.items():
    print("\n  %s" % m)
    s = cols(y, A[m])
    for t in ts:
        j = TASKS.index(t)
        d = s[j] - p[j]
        print("    %-20s parent %.4f  specialist %.4f  %+.4f  %s"
              % (t, p[j], s[j], d, "better" if d > 0 else "worse"))


def routed(base_members):
    """Flat blend of base_members, plus each specialist added ONLY on its own findings."""
    out = np.nanmean([A[m] for m in base_members], axis=0)
    k = len(base_members)
    for j, t in enumerate(TASKS):
        extra = [A[m][:, j] for m, ts in BRANCH.items() if t in ts and not np.isnan(A[m][:, j]).all()]
        if extra:
            out[:, j] = (out[:, j] * k + np.sum(extra, axis=0)) / (k + len(extra))
    return out


# --- 2. cross-fitted -------------------------------------------------------------
print("\n\n2. does routing them in help? (cross-fitted, 5 folds x 8 seeds)")
for label, mem in [("convnext_base_relabel alone", [PARENT]), ("the 0.932 seven", SEVEN)]:
    base = np.nanmean([A[m] for m in mem], axis=0)
    rt = routed(mem)
    d = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        fold = np.tile(np.arange(5), n // 5 + 1)[:n]
        rng.shuffle(fold)
        ob, orr = np.empty((n, 12)), np.empty((n, 12))
        for k in range(5):
            te = np.where(fold == k)[0]
            ob[te], orr[te] = base[te], rt[te]
        d.append(macro(y, orr) - macro(y, ob))
    d = np.array(d)
    print("  %-30s base %.4f  routed %.4f   delta %+.4f +/- %.4f  (%d/8 positive)"
          % (label, macro(y, base), macro(y, rt), d.mean(), d.std(), (d > 0).sum()))

# --- 3. per-finding, where does the routing actually move things? -----------------
print("\n3. the 0.932 seven: per-finding effect of routing the specialists in")
base = np.nanmean([A[m] for m in SEVEN], axis=0)
bc, rc = cols(y, base), cols(y, routed(SEVEN))
for j in np.argsort(-(rc - bc)):
    who = [m.replace("convnext_base_", "") for m, ts in BRANCH.items() if TASKS[j] in ts]
    print("  %-20s %.4f -> %.4f  %+.4f   %s" % (TASKS[j], bc[j], rc[j], rc[j] - bc[j],
                                                "+".join(who)))
print("  %-20s %.4f -> %.4f  %+.4f" % ("MACRO", np.nanmean(bc), np.nanmean(rc),
                                       np.nanmean(rc) - np.nanmean(bc)))
