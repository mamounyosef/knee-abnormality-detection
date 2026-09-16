r"""fold_ensemble.py - evaluate the cross-validation folds HONESTLY, and test them in ensembles.

THE LEAKAGE THAT SHAPES EVERYTHING HERE
---------------------------------------
The folds are cut over all 4,342 studies, while every existing ensemble member was trained on
`05_train` and validated on the fixed 704-study `05_val`. Those overlap badly: **559-589 of
those 704 val studies sit in each fold's TRAINING data**. So a fold model scored on the
standard 704 val set is scoring its own training data, and the number is meaningless - the
same trap that made `convnext_base_full` read 0.9677.

That rules out the comparison we would most like to make. Two honest surfaces remain:

  1. OUT-OF-FOLD (OOF). Score each study with the fold that HELD IT OUT, then concatenate.
     With three folds that covers 836+865+872 = 2,573 studies, each predicted by a model that
     never saw it. This is the honest read on the fold RECIPE, but it cannot be compared to
     the 704-val numbers of other members, because it is a different set of studies.

  2. AUDIT-58. Verified here: **0 of the 58 audit studies appear in any fold's training data**,
     and the audit set lies entirely outside the 4,342-study fold universe. It is therefore
     the ONLY common, clean surface on which fold models and existing members can be compared
     and blended. It is also only 58 studies (SE ~0.02), so it answers "is this catastrophic
     or plausible", never "is this +0.001 better".

Everything below is reported with that caveat attached, because the tempting numbers are the
dishonest ones.

    python runs/others/ensembling/fold_ensemble.py
"""

import csv
import glob
import io
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
RUNS = ROOT / "runs"

FOLDS = ["convnext_base_25d_18slices_f0", "convnext_base_25d_18slices_f1",
         "convnext_base_25d_18slices_f2"]
PARENT = "convnext_base_25d_18slices"
# the members of submission 56258238 (val 0.9167, the 9-model branch group)
SHIPPED = ["convnext_base_relabel", "convnext_base_trauma", "convnext_base_wear",
           "convnext_base_fluid", "mricore_vitb_cross_attention",
           "mricore_vitb_mae_cross_attention_full", "dinov3_convnext_base",
           "convnext_base_25d_group3", "convnext_base_25d_poolattn"]


def macro(y, p):
    out = []
    for j in range(y.shape[1]):
        yj = (y[:, j] > 0.5).astype(int)
        if 0 < yj.sum() < len(yj):
            out.append(roc_auc_score(yj, p[:, j]))
    return float(np.nanmean(out)) if out else float("nan")


def best_pred(run, split):
    """The run's highest-scoring saved prediction for that split, by its own val score."""
    d = RUNS / run / "results" / "predictions"
    files = sorted(d.glob("*_%s_s*.npz" % split))
    if not files:
        return None
    if split == "audit":
        # pick the step whose VAL score is best, not the best audit - choosing on audit
        # would be fitting the 58 studies we are about to report
        vb, vstep = -1.0, None
        for f in sorted(d.glob("*_val_s*.npz")):
            z = np.load(f, allow_pickle=True)
            s = macro(z["targets"], z["logits"])
            if s > vb:
                vb, vstep = s, f.name.split("_s")[-1].replace(".npz", "")
        m = [f for f in files if f.name.endswith("_s%s.npz" % vstep)]
        if m:
            return np.load(m[0], allow_pickle=True)
    best, bs = None, -1.0
    for f in files:
        z = np.load(f, allow_pickle=True)
        s = macro(z["targets"], z["logits"])
        if s > bs:
            best, bs = z, s
    return best


def ids_of(p):
    with io.open(ROOT / "data" / p, encoding="utf-8") as fh:
        return {r["StudyInstanceUID"] for r in csv.DictReader(fh)}


print(__doc__.split("----\n")[0])

# ---------------------------------------------------------------- 1. per fold, own val
print("=" * 78)
print("1. EACH FOLD ON ITS OWN HELD-OUT BLOCK (honest, but a different set each)")
print("=" * 78)
per = {}
for i, run in enumerate(FOLDS):
    z = best_pred(run, "val")
    s = macro(z["targets"], z["logits"])
    per[run] = s
    print("  f%d  %4d studies   macro AUROC %.4f" % (i, len(z["study_ids"]), s))
vals = np.array(list(per.values()))
print("\n  mean %.4f   spread %.4f (min %.4f, max %.4f)"
      % (vals.mean(), vals.max() - vals.min(), vals.min(), vals.max()))
print("  NOTE: these are three DIFFERENT val sets, so the spread mixes split difficulty")
print("        with model variance. It is not comparable to the parent's 704-study 0.9010.")

# ---------------------------------------------------------------- 2. out-of-fold
print("\n" + "=" * 78)
print("2. OUT-OF-FOLD: every study scored by the fold that HELD IT OUT")
print("=" * 78)
oof_l, oof_y, oof_id = [], [], []
for run in FOLDS:
    z = best_pred(run, "val")
    oof_l.append(z["logits"].astype(np.float64))
    oof_y.append(z["targets"].astype(np.float64))
    oof_id.extend(list(z["study_ids"]))
L, Y = np.vstack(oof_l), np.vstack(oof_y)
assert len(set(oof_id)) == len(oof_id), "a study appears in two fold val sets"
print("  %d studies, each predicted by a model that never saw it" % len(oof_id))
print("  OOF macro AUROC  %.4f" % macro(Y, L))
print("\n  This is the honest estimate of the fold RECIPE on unseen data.")
print("  It is NOT comparable to any 704-val number: different studies, harder set.")

# ---------------------------------------------------------------- 3. audit-58
print("\n" + "=" * 78)
print("3. AUDIT-58: the only clean surface shared with the existing members")
print("=" * 78)
aud = ids_of("05_audit58.csv")
leak = 0
for run in FOLDS:
    tr = ids_of("13_fold%s_train.csv" % run[-1])
    leak += len(tr & aud)
print("  leak check: %d of %d audit studies appear in any fold's training data"
      % (leak, len(aud)))
if leak:
    raise SystemExit("  AUDIT IS LEAKY FOR THE FOLDS - stop, this comparison is invalid")

ref = best_pred(FOLDS[0], "audit")
order = {s: i for i, s in enumerate(list(ref["study_ids"]))}
ay = ref["targets"].astype(np.float64)


TASKS = [str(t) for t in ref["tasks"]]


def audit_logits(run):
    """(58, 12) aligned on study id AND task, NaN where a member does not predict a task.

    The task-group branches (trauma / wear / fluid) emit only their own 4-5 findings, so a
    plain assignment misaligns columns - silently producing a blend of the wrong findings.
    Placing by task NAME and leaving NaN elsewhere lets nanmean give each finding exactly the
    members that actually predict it, which is what the branch-group construct means.
    """
    z = best_pred(run, "audit")
    if z is None:
        return None
    ids = list(z["study_ids"])
    if set(ids) != set(order):
        return None
    out = np.full((len(order), 12), np.nan)
    cols = [str(t) for t in z["tasks"]]
    rows = [order[s] for s in ids]
    lg = z["logits"].astype(np.float64)
    for j, t in enumerate(cols):
        if t in TASKS:
            out[rows, TASKS.index(t)] = lg[:, j]
    return out


print("\n  each fold alone:")
fold_a = {}
for i, run in enumerate(FOLDS):
    a = audit_logits(run)
    fold_a[run] = a
    print("    f%d                        %.4f" % (i, macro(ay, a)))

fold_avg = np.nanmean([fold_a[r] for r in FOLDS], axis=0)
print("\n    3-fold average            %.4f" % macro(ay, fold_avg))

pa = audit_logits(PARENT)
if pa is not None:
    print("    parent (single model)     %.4f" % macro(ay, pa))

sh = {}
for m in SHIPPED:
    a = audit_logits(m)
    if a is not None:
        sh[m] = a
print("    shipped members found     %d of %d" % (len(sh), len(SHIPPED)))

if len(sh) == len(SHIPPED):
    # the branch group: parent + 3 task branches share ONE vote, five standalone hold 1/6
    grp = np.nanmean([sh[m] for m in SHIPPED[:4]], axis=0)
    stand = [sh[m] for m in SHIPPED[4:]]
    shipped_blend = np.nanmean([grp] + stand, axis=0)
    print("\n  ensembles on audit-58:")
    print("    shipped 9-model           %.4f" % macro(ay, shipped_blend))
    print("    shipped + 3-fold avg      %.4f"
          % macro(ay, np.nanmean([grp] + stand + [fold_avg], axis=0)))
    print("    shipped + 3 folds as 3    %.4f"
          % macro(ay, np.nanmean([grp] + stand + [fold_a[r] for r in FOLDS], axis=0)))
    print("    folds REPLACE 25d_group3  %.4f"
          % macro(ay, np.nanmean([grp] + [sh[m] for m in SHIPPED[4:] if m != "convnext_base_25d_group3"]
                                 + [fold_avg], axis=0)))
    print("    3-fold avg alone          %.4f" % macro(ay, fold_avg))

print("\n" + "=" * 78)
print("READ THIS BEFORE QUOTING ANY NUMBER ABOVE")
print("=" * 78)
print("""  Audit-58 is 58 studies. Its standard error is around 0.02, which is four times
  the val noise band we already refuse to act on. Differences of 0.00x here mean
  nothing. It is included because it is the only surface where fold models and the
  shipped members can meet without leakage - not because it can rank them.

  The fold models CANNOT be scored on the standard 704-study val set: 559-589 of
  those studies are in each fold's training data. Any such number is inflated.

  What the folds are actually for is the TEST set, where no leakage exists. The
  honest pre-submission evidence is the OOF number in section 2 plus the knowledge
  that a 5-model average of models trained on different 80% subsets is normally
  worth more than any single one of them.""")
