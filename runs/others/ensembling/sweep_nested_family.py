r"""sweep_nested_family.py - give the 2.5D family ONE vote, not five.

WHY THE FLAT SWEEP WAS THE WRONG TEST. sweep_new_runs.py added each new run to the shipped
seven as a full peer and every placement came out flat or negative. But those runs are not
peers: `convnext_base_25d_18slices`, `_left_knees_mirrored`, `_perfinding_pool_features`,
`_perfinding_pool_scores` and `_mcl_top2_pooling` are the SAME architecture on the SAME
geometry, differing in one setting each. Adding them as separate members does not add
diversity, it just gives that one family five votes out of eleven and drowns the genuinely
different members (MRI-CORE, SwinV2, DINOv3, RadImageNet).

The right shape is a SUB-ENSEMBLE: average the family first, then let the average cast ONE
vote alongside the others. ensemble.py already supports this - a list inside the member list
is averaged first - and it has paid here before: nesting convnext_base_seed1337 with its own
parent as one member gave OOF 0.9151 against 0.9140 flat, +0.0011, where adding it flat was
-0.0004.

So this sweeps every subset of the family as a nested member, against the same shipped seven,
and cross-fits the winner.

    python runs/others/ensembling/sweep_nested_family.py
    python runs/others/ensembling/sweep_nested_family.py --set audit
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sweep_new_runs import LEAKY, auroc, best_npz, load  # noqa: E402

RUNS = HERE.parent.parent

PARENT = "convnext_base_25d_18slices"
# every run that is PARENT with one setting changed. Same backbone, same 336px, same 18
# slices, same 4 slots, same splits - so they are one family, not five opinions.
FAMILY = ["convnext_base_25d_left_knees_mirrored",
          "convnext_base_25d_perfinding_pool_features",
          "convnext_base_25d_perfinding_pool_scores",
          "convnext_base_25d_mcl_top2_pooling"]

OTHERS = ["convnext_base_relabel", "mricore_vitb_cross_attention",
          "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
          "swinv2_base_in22k", "convnext_base_25d_poolattn"]
SHIPPED = OTHERS[:4] + [PARENT] + OTHERS[4:]


def vote(member, split):
    """One member's logits. A LIST is a sub-ensemble: averaged first, one vote after."""
    if isinstance(member, (list, tuple)):
        parts = [vote(m, split) for m in member]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        ids = set(parts[0][0])
        for i, _, _ in parts[1:]:
            ids &= set(i)
        ids = np.array(sorted(ids))
        stack, y = [], None
        for i, lg, tg in parts:
            pos = {s: k for k, s in enumerate(i)}
            sel = np.array([pos[s] for s in ids])
            stack.append(lg[sel])
            if y is None:
                y = tg[sel]
        return ids, np.mean(stack, axis=0), y
    v = load(member, split)
    if v is None or v[1].shape[1] != 12:
        return None
    return v


def score(members, split):
    votes = [vote(m, split) for m in members]
    if any(v is None for v in votes):
        return float("nan")
    ids = set(votes[0][0])
    for i, _, _ in votes[1:]:
        ids &= set(i)
    ids = np.array(sorted(ids))
    stack, y = [], None
    for i, lg, tg in votes:
        pos = {s: k for k, s in enumerate(i)}
        sel = np.array([pos[s] for s in ids])
        stack.append(lg[sel])
        if y is None:
            y = tg[sel]
    return auroc(y, np.mean(stack, axis=0))


def per_column(members, split):
    votes = [vote(m, split) for m in members]
    ids = set(votes[0][0])
    for i, _, _ in votes[1:]:
        ids &= set(i)
    ids = np.array(sorted(ids))
    stack, y = [], None
    for i, lg, tg in votes:
        pos = {s: k for k, s in enumerate(i)}
        sel = np.array([pos[s] for s in ids])
        stack.append(lg[sel])
        if y is None:
            y = tg[sel]
    p = np.mean(stack, axis=0)
    return [auroc(y[:, [j]], p[:, [j]]) for j in range(12)]


def cross_fit(cands, split, folds=5, seed=0):
    """Choose on 4/5 of the studies, score the held-out fifth. The honest number."""
    v0 = vote(SHIPPED[0], split)
    ids = v0[0]
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(ids)), folds)
    prepared = []
    for c in cands:
        votes = [vote(m, split) for m in c]
        if any(v is None for v in votes):
            continue
        common = set(votes[0][0])
        for i, _, _ in votes[1:]:
            common &= set(i)
        common = np.array(sorted(common))
        stack, y = [], None
        for i, lg, tg in votes:
            pos = {s: k for k, s in enumerate(i)}
            sel = np.array([pos[s] for s in common])
            stack.append(lg[sel])
            if y is None:
                y = tg[sel]
        prepared.append((c, np.mean(stack, axis=0), y))
    held = []
    for k in range(folds):
        te = parts[k]
        tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        best, bv = None, -1
        for c, p, y in prepared:
            v = auroc(y[tr], p[tr])
            if v > bv:
                best, bv = (c, p, y), v
        if best:
            held.append(auroc(best[2][te], best[1][te]))
    return float(np.mean(held)) if held else float("nan")


def label(m):
    if isinstance(m, (list, tuple)):
        return "[" + " + ".join(x.replace("convnext_base_25d_", "") for x in m) + "]"
    return m.replace("convnext_base_25d_", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="val", choices=["val", "audit"])
    a = ap.parse_args()
    split = a.set

    print("=" * 84)
    print("  NESTED FAMILY SWEEP - the 2.5D family averaged first, then ONE vote (%s)" % split)
    print("=" * 84)

    ship = score(SHIPPED, split)
    print("\n  shipped seven, %s flat: %.4f" % (split, ship))
    print("  family members available: %s"
          % ", ".join(f.replace("convnext_base_25d_", "")
                      for f in FAMILY if load(f, split) is not None))

    print("\n  EVERY SUBSET OF THE FAMILY, nested with the parent as one member")
    print("  %-58s %8s %9s" % ("nested member replacing 18slices", split, "vs shipped"))
    cands, rows = [SHIPPED], []
    for r in range(1, len(FAMILY) + 1):
        for combo in itertools.combinations(FAMILY, r):
            nested = [PARENT] + list(combo)
            c = OTHERS[:4] + [nested] + OTHERS[4:]
            v = score(c, split)
            if np.isnan(v):
                continue
            cands.append(c)
            rows.append((v, nested, c))
            print("  %-58s %8.4f %+9.4f%s"
                  % (label(nested)[:58], v, v - ship,
                     "  <- GAIN" if v > ship + 0.0005 else ""))

    rows.sort(reverse=True, key=lambda x: x[0])
    print("\n  best nested: %s  %.4f (%+.4f)" % (label(rows[0][1]), rows[0][0],
                                                 rows[0][0] - ship))

    # the same family as SEPARATE members, for contrast - this is what the flat sweep did
    flat = OTHERS[:4] + [PARENT] + list(FAMILY) + OTHERS[4:]
    vflat = score(flat, split)
    print("  same runs as separate members (the flat test): %.4f (%+.4f)"
          % (vflat, vflat - ship))
    print("  -> nesting is worth %+.4f over flat" % (rows[0][0] - vflat))

    # does the nested family also help when the OTHER members are nested by family?
    print("\n  ALSO NESTING THE MRI-CORE PAIR (they share a trunk too)")
    mri = ["mricore_vitb_cross_attention", "mricore_vitb_mae_cross_attention"]
    c2 = ["convnext_base_relabel", mri, "dinov3_convnext_base", rows[0][1],
          "swinv2_base_in22k", "convnext_base_25d_poolattn"]
    v2 = score(c2, split)
    cands.append(c2)
    print("  %-58s %8.4f %+9.4f" % ("mricore pair nested too", v2, v2 - ship))

    print("\n  CROSS-FITTED (choose on 4/5 of the studies, score the held-out fifth)")
    cf = cross_fit(cands, split)
    cf_ship = cross_fit([SHIPPED], split)
    print("     best-of-%d in sample : %.4f" % (len(cands), max(score(c, split) for c in cands)))
    print("     same, cross-fitted   : %.4f" % cf)
    print("     shipped, cross-fitted: %.4f" % cf_ship)
    print("     honest delta         : %+.4f" % (cf - cf_ship))

    print("\n  PER-FINDING, best nested against shipped")
    tasks = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
             "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
    a_col = per_column(SHIPPED, split)
    b_col = per_column(rows[0][2], split)
    for t, x, y in zip(tasks, a_col, b_col):
        print("    %-18s %.4f -> %.4f  %+.4f%s" % (t, x, y, y - x, "  <-" if y > x else ""))
    print("\n" + "=" * 84)


if __name__ == "__main__":
    main()
