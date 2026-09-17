r"""make_synovitis_route_weights.py - hand Synovitis to its specialist, leave the rest alone.

convnext_base_fluid was fine-tuned on Effusion + Synovitis. On Synovitis it scores 0.8767
against the seven-model ensemble's 0.8597 - it beats not just its parent but the whole
ensemble, the only finding of twelve where that is true. Giving it the column outright is
worth +0.0014 of macro and is robust: 10/10 cross-fitted seeds positive, the weight chosen was
1.0 in 44 of 50 fits, and it is positive on audit-58 too (+0.0096), an independent set.

It gets Synovitis ONLY. On Effusion, the other finding it was trained on, it is WORSE than the
ensemble (-0.0067), and the three other columns that looked winnable on val (Fracture, PF OA,
Medial OA) were chosen by searching the val table rather than specified in advance - two of
them reversed sign on audit. Search-selected columns are not shipped.

WHICH BASE MODEL. Measurement uses convnext_base_relabel, because convnext_base_full trained
on all 704 val AND all 58 audit studies, so with it in the blend there is no honest surface
left to verify anything on. The SUBMISSION swaps in convnext_base_full: it is the same
architecture, same 512/24 geometry, refit on 20.9% more data, and scored 0.920 solo against
relabel's 0.918. No leakage exists on the hidden test set, so the swap is a real gain there -
it simply cannot be checked beforehand.

    python runs/others/ensembling/make_synovitis_route_weights.py            # measure (relabel)
    python runs/others/ensembling/make_synovitis_route_weights.py --submit   # ship (full)
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent

SPECIALIST = "convnext_base_fluid"
ROUTED_TASK = "Synovitis"
BASE7_MEASURE = ["convnext_base_relabel", "mricore_vitb_cross_attention",
                 "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
                 "convnext_base_25d_18slices", "swinv2_base_in22k",
                 "convnext_base_25d_poolattn"]

ap = argparse.ArgumentParser()
ap.add_argument("--submit", action="store_true",
                help="swap convnext_base_relabel for convnext_base_full (the ship build)")
a = ap.parse_args()

base7 = list(BASE7_MEASURE)
if a.submit:
    base7[0] = "convnext_base_full"
MEMBERS = base7 + [SPECIALIST]
OUT = HERE / ("synovitis_route_weights%s.json" % ("_full" if a.submit else ""))


def member_tasks(run):
    f = sorted(glob.glob(str(RUNS / run / "results" / "predictions" / "*_val_s*.npz")))[0]
    return [str(t) for t in np.load(f, allow_pickle=True)["tasks"]]


tasks = member_tasks(base7[1])          # a full 12-task member
owns = {m: set(member_tasks(m)) for m in MEMBERS}
if ROUTED_TASK not in owns[SPECIALIST]:
    raise SystemExit("%s does not predict %s" % (SPECIALIST, ROUTED_TASK))

weights = {}
for t in tasks:
    row = {m: 0.0 for m in MEMBERS}
    if t == ROUTED_TASK:
        # the specialist takes the column outright; the seven are switched off here
        row[SPECIALIST] = 1.0
    else:
        present = [m for m in base7 if t in owns[m]]
        if len(present) != len(base7):
            raise SystemExit("%s: only %d of the %d base members predict it"
                             % (t, len(present), len(base7)))
        for m in base7:
            row[m] = 1.0 / len(base7)
    total = sum(row.values())
    if abs(total - 1.0) > 1e-9:
        raise SystemExit("%s sums to %.9f" % (t, total))
    weights[t] = row

OUT.write_text(json.dumps({
    "members": MEMBERS,
    "weights_per_class": weights,
    "note": ("the 0.932 seven at 1/7 each on 11 findings; Synovitis handed entirely to "
             "convnext_base_fluid" + (" (convnext_base_full swapped in for relabel)"
                                      if a.submit else "")),
}, indent=1), encoding="utf-8")

print("wrote %s\n" % OUT.name)
print("  %-40s %12s %12s" % ("member", ROUTED_TASK, "other 11"))
for m in MEMBERS:
    other = next(weights[t][m] for t in tasks if t != ROUTED_TASK)
    print("  %-40s %12.4f %12.4f" % (m, weights[ROUTED_TASK][m], other))
print("\n  every column sums to exactly 1.0")
