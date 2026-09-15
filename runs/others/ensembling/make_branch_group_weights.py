"""
make_branch_group_weights.py - write the branch-group construct as a flat weight matrix.

The 0.9160 ensemble is NESTED: convnext_base_relabel and its three task-group branches
average into ONE vote, which then blends flat with the five standalone models. The Kaggle
inference path has no notion of nesting, only a per-class weight per member. The two are
the same thing, so this writes the equivalent flat matrix.

For any finding, exactly two of the four convnext runs predict it: the parent (which
predicts all 12) and whichever branch owns that finding. So the group's 1/6 share splits
1/12 and 1/12, and each standalone model keeps 1/6:

    parent            1/12 on all 12 findings
    trauma            1/12 on ACL, MCL, Contusion, Fracture, else 0
    wear              1/12 on the menisci, the three OA findings and Baker's, else 0
    fluid             1/12 on Effusion and Synovitis, else 0
    the five others   1/6 each, everywhere
                      ------------------------------------------------
    per finding       1/12 + 1/12 + 5 x 1/6  =  1

Writes branch_group_weights.json, which submit_ensemble.py takes as BLEND_WEIGHTS.

    python runs/others/ensembling/make_branch_group_weights.py
"""

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent          # <repo>/runs
OUT = HERE / "branch_group_weights.json"

GROUP = ["convnext_base_relabel", "convnext_base_trauma", "convnext_base_wear",
         "convnext_base_fluid"]
STANDALONE = ["mricore_vitb_cross_attention", "mricore_vitb_mae_cross_attention_full",
              "dinov3_convnext_base", "convnext_base_25d_group3",
              "convnext_base_25d_poolattn"]
# swinv2_base_in22k gave way to convnext_base_25d_poolattn: the construct scores the SAME
# 0.9167 either way, and swinv2 costs 5.197 s/study against poolattn's ~0.4. That is the
# difference between 8.98 h at 1500 studies (two minutes inside the 9 h cap) and ~7 h.
# Two changes from the 0.9160 set, both measured on val:
#   mricore_vitb_mae_cross_attention -> _full  (6 fusion points instead of 3): 0.9160 -> 0.9164
#   convnext_large -> convnext_base_25d_group3 (32 passes/study, not 96):      0.9164 -> 0.9167
# The second is also the cheap one: 0.512 s/study against convnext_large's 3.488 on a T4,
# at the price of a second decode pass because it is 336 px where the rest are 512.
MEMBERS = GROUP + STANDALONE


def member_tasks(run):
    f = sorted(glob.glob(str(RUNS / run / "results" / "predictions" / "*_val_s*.npz")))[0]
    z = np.load(f, allow_pickle=True)
    return [str(t) for t in z["tasks"]]


tasks = member_tasks(MEMBERS[0])
owns = {m: set(member_tasks(m)) for m in MEMBERS}

weights = {}
for t in tasks:
    present = [m for m in GROUP if t in owns[m]]
    if len(present) != 2:
        raise SystemExit("%s is predicted by %d of the group, expected 2 (the parent "
                         "and one branch): %s" % (t, len(present), present))
    share = (1.0 / (len(STANDALONE) + 1)) / len(present)     # the group's vote, split
    row = {m: (share if m in present else 0.0) for m in GROUP}
    for m in STANDALONE:
        row[m] = 1.0 / (len(STANDALONE) + 1)
    total = sum(row.values())
    if abs(total - 1.0) > 1e-9:
        raise SystemExit("%s weights sum to %.6f" % (t, total))
    weights[t] = row

OUT.write_text(json.dumps({
    "members": MEMBERS,
    "weights_per_class": weights,
    "note": ("the nested branch-group construct written flat: the four convnext runs "
             "share one 1/6 vote, each standalone model holds 1/6"),
}, indent=1), encoding="utf-8")
print("wrote %s" % OUT)
print("%-36s%s" % ("member", "  ".join(t[:6] for t in tasks[:6])))
for m in MEMBERS:
    print("%-36s%s" % (m, "  ".join("%.4f" % weights[t][m] for t in tasks[:6])))
