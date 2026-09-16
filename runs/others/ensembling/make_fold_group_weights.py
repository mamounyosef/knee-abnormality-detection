r"""make_fold_group_weights.py - give the five folds ONE shared vote, not five.

The five folds are the same architecture, the same recipe and the same hyper-parameters; they
differ only in which 80% of the studies they saw. They are far closer to five seeds of one
model than to five distinct models. A flat average over 7 + 5 members would hand that single
model family 5/12 = 42% of the ensemble, drowning the architectural diversity (two MRI-CORE
ViTs, DINOv3, SwinV2) that every measured ensemble gain in this project has come from.

So the folds are fused into one vote, exactly as the convnext task-group branches are:

    each of the 7 standalone members   1/8
    the 5 folds                        1/8 shared, i.e. 1/40 each
                                       ----------------------------
    per finding                        7/8 + 5 x 1/40 = 1

Writes fold_group_weights.json for submit_ensemble.py's BLEND_WEIGHTS.

    python runs/others/ensembling/make_fold_group_weights.py
"""

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent
OUT = HERE / "fold_group_weights.json"

STANDALONE = ["convnext_base_relabel", "mricore_vitb_cross_attention",
              "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
              "convnext_base_25d_18slices", "swinv2_base_in22k",
              "convnext_base_25d_poolattn"]
FOLDS = ["convnext_base_25d_18slices_f%d" % k for k in range(5)]
MEMBERS = STANDALONE + FOLDS          # order MUST match submit_ensemble.py's MEMBERS


def member_tasks(run):
    f = sorted(glob.glob(str(RUNS / run / "results" / "predictions" / "*_val_s*.npz")))[0]
    z = np.load(f, allow_pickle=True)
    return [str(t) for t in z["tasks"]]


tasks = member_tasks(MEMBERS[0])
owns = {m: set(member_tasks(m)) for m in MEMBERS}

n_votes = len(STANDALONE) + 1          # 7 standalone + the fold block
weights = {}
for t in tasks:
    row = {}
    for m in STANDALONE:
        if t not in owns[m]:
            raise SystemExit("%s does not predict %s - it cannot hold a full vote" % (m, t))
        row[m] = 1.0 / n_votes
    present = [m for m in FOLDS if t in owns[m]]
    if len(present) != len(FOLDS):
        raise SystemExit("only %d of %d folds predict %s" % (len(present), len(FOLDS), t))
    for m in FOLDS:
        row[m] = (1.0 / n_votes) / len(FOLDS)
    total = sum(row.values())
    if abs(total - 1.0) > 1e-9:
        raise SystemExit("%s weights sum to %.9f" % (t, total))
    weights[t] = row

OUT.write_text(json.dumps({
    "members": MEMBERS,
    "weights_per_class": weights,
    "note": ("the 0.932 seven at 1/8 each, plus the five CV folds sharing a single 1/8 vote "
             "(1/40 apiece) - the folds are one model family, not five architectures"),
}, indent=1), encoding="utf-8")

print("wrote %s" % OUT)
print("%-38s %s" % ("member", "weight (every finding)"))
for m in MEMBERS:
    print("  %-36s %.5f" % (m, weights[tasks[0]][m]))
print("\n  7 standalone x 1/8 = %.4f   +   5 folds x 1/40 = %.4f   = %.4f"
      % (7 / 8, 5 * (1 / 40), 7 / 8 + 5 * (1 / 40)))
