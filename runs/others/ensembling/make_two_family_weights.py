r"""make_two_family_weights.py - 70/30 across the two fold FAMILIES, not across 10 models.

The ten members are two families of five. A flat average would give each model 1/10 and so
split the vote 50/50 between ConvNeXt and DINOv2 - but the families are not equally strong,
and the out-of-fold measurement says so:

    ConvNeXt folds only                0.8943      (this scored 0.918 on the leaderboard)
    DINOv2 folds only                  0.8823
    both, equal weight (50/50)         0.8972
    both, 70/30                        0.8985      <- best, and the curve is flat
    both, 75/25                        0.8983         from 60/40 to 75/25

So the split is set at the FAMILY level and shared inside each family:

    each ConvNeXt fold   0.70 / 5 = 0.14
    each DINOv2 fold     0.30 / 5 = 0.06
                         ---------------
    per finding          5(0.14) + 5(0.06) = 1.0

That 0.8985 is measured on all 4,342 studies by models that never saw them, which is what
makes it worth acting on where audit-58 numbers are not.

    python runs/others/ensembling/make_two_family_weights.py
"""

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent
OUT = HERE / "two_family_weights.json"

CONV = ["convnext_base_25d_18slices_f%d" % k for k in range(5)]
DINO = ["dinov2_vits14_25d_f%d" % k for k in range(5)]
W_CONV, W_DINO = 0.70, 0.30
MEMBERS = CONV + DINO                  # order MUST match submit_ensemble.py's MEMBERS


def member_tasks(run):
    f = sorted(glob.glob(str(RUNS / run / "results" / "predictions" / "*_val_s*.npz")))[0]
    return [str(t) for t in np.load(f, allow_pickle=True)["tasks"]]


tasks = member_tasks(MEMBERS[0])
owns = {m: set(member_tasks(m)) for m in MEMBERS}

weights = {}
for t in tasks:
    row = {}
    for fam, w in ((CONV, W_CONV), (DINO, W_DINO)):
        have = [m for m in fam if t in owns[m]]
        if len(have) != len(fam):
            raise SystemExit("only %d of %d in a family predict %s" % (len(have), len(fam), t))
        for m in fam:
            row[m] = w / len(fam)
    total = sum(row.values())
    if abs(total - 1.0) > 1e-9:
        raise SystemExit("%s sums to %.9f" % (t, total))
    weights[t] = row

OUT.write_text(json.dumps({
    "members": MEMBERS,
    "weights_per_class": weights,
    "note": ("two fold families on one partition, split 70/30 ConvNeXt/DINOv2 by their "
             "out-of-fold scores (0.8943 vs 0.8823); 0.8985 OOF blended"),
}, indent=1), encoding="utf-8")

print("wrote %s\n" % OUT)
for m in MEMBERS:
    print("  %-38s %.4f" % (m, weights[tasks[0]][m]))
print("\n  ConvNeXt family %.2f   DINOv2 family %.2f   total %.4f"
      % (W_CONV, W_DINO, sum(weights[tasks[0]].values())))
