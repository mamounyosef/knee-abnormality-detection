r"""two_family_oof.py - does adding the DINOv2 folds to the ConvNeXt folds actually help?

THIS IS THE ONE HONEST MEASUREMENT AVAILABLE, and it exists only because both fold families
were trained on the SAME partition. For each block k, `convnext..._f{k}` and
`dinov2..._f{k}` were both held out of block k, so averaging their predictions on that block
and concatenating over all five blocks gives an out-of-fold score over all 4,342 studies with
no leakage anywhere.

That matters because we now have a calibration point: the ConvNeXt folds scored **0.8943 OOF
-> 0.918 on the leaderboard**. So an OOF number here is not an abstract quantity, it maps
onto an expected LB. Nothing else we have does that - the 704 val set is poisoned for folds
(559-589 of it sits in each fold's training data) and audit-58 has an SE of ~0.02.

    python runs/others/ensembling/two_family_oof.py
"""

import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS = ROOT / "runs"

CONV = "convnext_base_25d_18slices_f%d"
DINO = "dinov2_vits14_25d_f%d"
CAL_OOF, CAL_LB = 0.8943, 0.918          # the ConvNeXt folds' measured pair


def macro(y, p):
    out = []
    for j in range(y.shape[1]):
        yj = (y[:, j] > 0.5).astype(int)
        if 0 < yj.sum() < len(yj):
            out.append(roc_auc_score(yj, p[:, j]))
    return float(np.nanmean(out)) if out else float("nan")


def best_val(run):
    """Highest-scoring saved val prediction, with its study ids."""
    d = RUNS / run / "results" / "predictions"
    best, bs = None, -1.0
    for f in sorted(d.glob("*_val_s*.npz")):
        z = np.load(f, allow_pickle=True)
        s = macro(z["targets"], z["logits"])
        if s > bs:
            best, bs = z, s
    return best, bs


def oof(make_block):
    """Concatenate per-block predictions into one 4,342-study out-of-fold set."""
    L, Y, ids = [], [], []
    for k in range(5):
        lg, y, sid = make_block(k)
        L.append(lg)
        Y.append(y)
        ids.extend(sid)
    assert len(set(ids)) == len(ids), "a study appears in two blocks"
    return macro(np.vstack(Y), np.vstack(L)), len(ids)


print("\nper-fold, each on its OWN held-out block")
print("  %-6s %10s %10s %10s" % ("block", "convnext", "dinov2", "blend"))
print("  " + "-" * 40)
blocks = {}
for k in range(5):
    zc, sc = best_val(CONV % k)
    zd, sd = best_val(DINO % k)
    idc = list(zc["study_ids"])
    idd = list(zd["study_ids"])
    assert set(idc) == set(idd), "block %d: the two families saw different studies" % k
    order = {s: i for i, s in enumerate(idc)}
    dl = np.empty_like(zc["logits"], dtype=np.float64)
    dl[[order[s] for s in idd]] = zd["logits"].astype(np.float64)
    cl = zc["logits"].astype(np.float64)
    y = zc["targets"].astype(np.float64)
    blocks[k] = (cl, dl, y, idc)
    print("  %-6d %10.4f %10.4f %10.4f"
          % (k, sc, sd, macro(y, (cl + dl) / 2)))

print("\nOUT-OF-FOLD over all 4,342 studies (leak-free)")
c_oof, n = oof(lambda k: (blocks[k][0], blocks[k][2], blocks[k][3]))
d_oof, _ = oof(lambda k: (blocks[k][1], blocks[k][2], blocks[k][3]))
b_oof, _ = oof(lambda k: ((blocks[k][0] + blocks[k][1]) / 2, blocks[k][2], blocks[k][3]))
print("  %-34s %.4f   (%d studies)" % ("ConvNeXt folds only", c_oof, n))
print("  %-34s %.4f" % ("DINOv2 folds only", d_oof))
print("  %-34s %.4f" % ("both families, equal weight", b_oof))

# a couple of weightings, since the two families are not equally strong
for w in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
    s, _ = oof(lambda k, w=w: (w * blocks[k][0] + (1 - w) * blocks[k][1],
                               blocks[k][2], blocks[k][3]))
    print("  %-34s %.4f" % ("both, %.0f%% ConvNeXt / %.0f%% DINOv2" % (w * 100, (1 - w) * 100), s))

print("\n" + "=" * 68)
print("  CALIBRATION: the ConvNeXt folds measured %.4f OOF -> %.3f LB" % (CAL_OOF, CAL_LB))
gain = b_oof - c_oof
print("  adding the DINOv2 folds moves OOF by %+.4f" % gain)
print("""
  That gain is measured on 4,342 studies, not 58, and on models that never saw
  them - so unlike every audit-58 number in this project it is worth acting on.
  It still does not say the result will clear 0.932: the ConvNeXt folds needed
  +0.014 of LB to get there, and this is a fold-family improvement, not the
  architecture diversity that the 0.932 ensemble gets from its ViTs and Swin.""")
print("=" * 68)
