"""
check_partial_head_blend.py - does the Kaggle inference path reproduce the offline score?

predict_dicom_ensemble blends members on the GPU by scattering each one's logits into
the 12-finding column order and taking a nanmean, so that a task-group branch votes only
on the findings it predicts. ensemble.py measures the same thing offline from saved
logits. Those are two independent implementations of one formula, and if they disagree
the submission scores something nobody measured.

This replays the GPU path's exact arithmetic on the saved validation logits and compares
against the offline number.

    python runs/others/ensembling/check_partial_head_blend.py
"""

import glob
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

RUNS = Path(__file__).resolve().parent.parent.parent
MEMBERS = ["convnext_base_relabel", "convnext_base_trauma", "convnext_base_wear",
           "convnext_base_fluid", "mricore_vitb_cross_attention",
           "mricore_vitb_mae_cross_attention_full", "dinov3_convnext_base",
           "convnext_base_25d_group3", "convnext_base_25d_poolattn"]
# the offline construct: the four convnext runs average into ONE vote, then that vote
# blends flat with the five standalone models
GROUP = MEMBERS[:4]
STANDALONE = MEMBERS[4:]
EXPECTED = 0.9167       # was 0.9160: the MAE cross-attention member went 3 -> 6 fusion
                        # points (+0.0004) and convnext_large gave way to
                        # convnext_base_25d_group3 (+0.0003)


def best_npz(run):
    best = None
    for f in sorted(glob.glob(str(RUNS / run / "results" / "predictions" / "*_val_s*.npz"))):
        z = np.load(f, allow_pickle=True)
        sc = float(z["meta"][4]) if str(z["meta"][4]) else -1.0
        if best is None or sc > best[0]:
            best = (sc, z, Path(f).name)
    return best[1], best[2]


ref, refname = best_npz(MEMBERS[0])
tasks = [str(t) for t in ref["tasks"]]
y = ref["targets"]
ids = ref["study_ids"]

print("%-36s%-30s%8s" % ("member", "checkpoint", "heads"))
scattered = {}
for m in MEMBERS:
    z, name = best_npz(m)
    assert np.array_equal(z["study_ids"], ids), m + ": study order differs"
    mt = [str(t) for t in z["tasks"]]
    # exactly what the GPU path does: a full NaN matrix, member logits scattered in
    full = torch.full((len(ids), len(tasks)), float("nan"))
    full[:, [tasks.index(t) for t in mt]] = torch.as_tensor(z["logits"], dtype=torch.float32)
    scattered[m] = full
    print("%-36s%-30s%8d" % (m, name, len(mt)))

# 1) the group: four members, three of them partial, averaged with nanmean
grp = torch.nanmean(torch.stack([scattered[m] for m in GROUP]), dim=0)
assert not torch.isnan(grp).any(), "the four convnext runs do not cover all 12 findings"

# 2) the top level: the group as one vote, plus the five full-width models
top = torch.nanmean(torch.stack([grp] + [scattered[m] for m in STANDALONE]), dim=0)
probs = torch.sigmoid(top).numpy()

# 3) the SAME thing as the flat weight matrix the Kaggle path actually consumes.
#    Nesting exists only offline; the notebook has one weight per member per finding,
#    so branch_group_weights.json must be arithmetically identical to the nesting above.
import json  # noqa: E402

wf = Path(__file__).resolve().parent / "branch_group_weights.json"
if wf.exists():
    fit = json.loads(wf.read_text(encoding="utf-8"))
    assert fit["members"] == MEMBERS, "the weight file lists different members"
    W = torch.zeros(len(MEMBERS), len(tasks))
    for j, t in enumerate(tasks):
        for i, m in enumerate(MEMBERS):
            W[i, j] = fit["weights_per_class"][t][m]
    stack = torch.stack([scattered[m] for m in MEMBERS])
    wtop = torch.nan_to_num(stack, nan=0.0).mul(W[:, None, :]).sum(dim=0)
    gap = float((wtop - top).abs().max())
    print("\nweight-matrix vs nested: max logit gap %.2e  %s"
          % (gap, "identical" if gap < 1e-4 else "DIFFERENT - do not ship"))
    if gap >= 1e-4:
        raise SystemExit(1)
else:
    print("\n(no branch_group_weights.json yet - run make_branch_group_weights.py)")

# the labels are LLM probabilities, not 0/1; ensemble.py binarizes at 0.5 to score
yb = (np.asarray(y) > 0.5).astype(np.float64)
macro = float(np.mean([roc_auc_score(yb[:, i], probs[:, i]) for i in range(len(tasks))]))
print("\ngpu-path blend  %.4f" % macro)
print("ensemble.py     %.4f" % EXPECTED)
d = abs(macro - EXPECTED)
print("difference      %.4f   %s" % (d, "MATCH" if d < 5e-5 else "MISMATCH - do not ship"))
raise SystemExit(0 if d < 5e-5 else 1)
