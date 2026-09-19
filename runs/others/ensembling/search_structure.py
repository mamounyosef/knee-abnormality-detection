r"""search_structure.py - search the ensemble STRUCTURE, not just its membership.

WHAT THE EARLIER SWEEPS GOT WRONG. sweep_new_runs.py added each run flat, one at a time, to a
FIXED seven. sweep_nested_family.py nested subsets of one family into a FIXED slot. Both held
most of the ensemble constant and varied one thing, so neither could find an arrangement that
needs two changes at once - and neither ever removed a member.

This searches the arrangement itself. A structure is a list of members, where a member is
either a run or a GROUP of runs averaged together that then casts one vote. From a starting
point it tries every single move and takes the best, repeating until nothing improves:

    ADD      a run as a new flat member
    INSERT   a run into one of the existing groups
    DROP     a member entirely
    SPLIT    pull a run out of a group into its own flat member
    MERGE    combine two flat members into one group
    SWAP     exchange a member for a run not currently used

That covers "add them one by one", "pull one out and try another", and the nested and flat
shapes in one procedure, and it reaches arrangements no single-change sweep can.

Hill climbing finds a local maximum, so it runs from several starts AND from random restarts.
Scoring goes through fast_ens.Pool, which makes a structure one weighted tensordot (~1.8 ms),
so thousands of restarts are affordable.

EVERY RESULT IS CROSS-FITTED. Searching tens of thousands of structures on 704 studies will
find something that looks good by chance. The held-out number is the one that counts, and it
is computed by re-running the SEARCH inside each fold, not by re-scoring a winner chosen on
all the data - anything less leaks the selection.

    python runs/others/ensembling/search_structure.py
    python runs/others/ensembling/search_structure.py --set audit --restarts 40
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fast_ens import Pool  # noqa: E402

SHIPPED = ["convnext_base_relabel", "mricore_vitb_cross_attention",
           "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
           "convnext_base_25d_18slices", "swinv2_base_in22k",
           "convnext_base_25d_poolattn"]


def moves(structure, pool_names, max_members, max_group):
    """Every one-step change to this structure."""
    used = set()
    for m in structure:
        used.update(m if isinstance(m, (list, tuple)) else [m])
    free = [r for r in pool_names if r not in used]
    out = []
    for r in free:
        if len(structure) < max_members:
            out.append(structure + [r])
        for k, m in enumerate(structure):
            grp = list(m) if isinstance(m, (list, tuple)) else [m]
            if len(grp) < max_group:
                s = copy.deepcopy(structure)
                s[k] = grp + [r]
                out.append(s)
        for k in range(len(structure)):                    # swap
            out.append(structure[:k] + [r] + structure[k + 1:])
    for k, m in enumerate(structure):
        if len(structure) > 2:
            out.append(structure[:k] + structure[k + 1:])
        if isinstance(m, (list, tuple)) and len(m) > 1 and len(structure) < max_members:
            for j in range(len(m)):
                s = copy.deepcopy(structure)
                rest = [x for i, x in enumerate(m) if i != j]
                s[k] = rest[0] if len(rest) == 1 else rest
                out.append(s + [m[j]])
    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            a = structure[i] if isinstance(structure[i], (list, tuple)) else [structure[i]]
            b = structure[j] if isinstance(structure[j], (list, tuple)) else [structure[j]]
            if len(a) + len(b) > max_group:
                continue
            s = [x for k, x in enumerate(structure) if k not in (i, j)]
            out.append(s + [list(a) + list(b)])
    return out


def climb(P, start, max_members, max_group, rows=None, log=None):
    cur = copy.deepcopy(list(start))
    cur_v = P.score(cur, rows)
    tried = 0
    while True:
        best_s, best_v = None, cur_v
        for s in moves(cur, P.names, max_members, max_group):
            tried += 1
            v = P.score(s, rows)
            if log is not None:
                log.append(s)
            if v > best_v + 1e-7:
                best_s, best_v = s, v
        if best_s is None:
            return cur, cur_v, tried
        cur, cur_v = best_s, best_v


def random_start(P, rng, n_members=4, max_group=3):
    picks = list(rng.choice(P.names, size=min(n_members, len(P.names)), replace=False))
    st = []
    for p in picks:
        if st and rng.random() < 0.3:
            k = int(rng.integers(len(st)))
            g = st[k] if isinstance(st[k], list) else [st[k]]
            if len(g) < max_group:
                st[k] = g + [p]
                continue
        st.append(p)
    return st if len(st) >= 2 else picks[:2]


def label(s):
    def one(m):
        if isinstance(m, (list, tuple)):
            return "[" + " + ".join(x.replace("convnext_base_25d_", "25d_") for x in m) + "]"
        return m.replace("convnext_base_25d_", "25d_")
    return "\n        + ".join(one(m) for m in s)


def search(P, starts, max_members, max_group, restarts, seed, rows=None, verbose=True):
    """All the starts plus random restarts. Returns (best_structure, best_value)."""
    rng = np.random.default_rng(seed)
    best, best_v, total = None, -1, 0
    for tag, st in starts:
        s, v, t = climb(P, st, max_members, max_group, rows)
        total += t
        if verbose:
            print("    %-26s -> %.4f  (%d structures)" % (tag, v, t))
        if v > best_v:
            best, best_v = s, v
    for i in range(restarts):
        s, v, t = climb(P, random_start(P, rng), max_members, max_group, rows)
        total += t
        if v > best_v:
            best, best_v = s, v
            if verbose:
                print("    random restart %-11d -> %.4f  (new best)" % (i, v))
    if verbose:
        print("    %d structures evaluated" % total)
    return best, best_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="val", choices=["val", "audit"])
    ap.add_argument("--max-members", type=int, default=9)
    ap.add_argument("--max-group", type=int, default=5)
    ap.add_argument("--restarts", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    P = Pool(a.set)
    print("=" * 84)
    print("  STRUCTURE SEARCH on %s - add / insert / drop / split / merge / swap" % a.set)
    print("=" * 84)
    print("  %d runs, %d studies, max %d members, max %d per group, %d random restarts"
          % (len(P.names), P.n, a.max_members, a.max_group, a.restarts))

    shipped = [s for s in SHIPPED if s in P.index]
    ship_v = P.score(shipped)
    print("  shipped seven: %.4f" % ship_v)

    fam = [r for r in P.names if r.startswith("convnext_base_25d_")]
    starts = [("shipped flat", shipped),
              ("shipped, 2.5D nested",
               shipped[:4] + [[shipped[4], shipped[6]]] + [shipped[5]]),
              ("two members", shipped[:2]),
              ("one 2.5D group + relabel", [fam[:3], "convnext_base_relabel"])]

    print("\n  IN-SAMPLE SEARCH")
    best, best_v = search(P, starts, a.max_members, a.max_group, a.restarts, a.seed)
    print("\n  best in sample: %.4f  (%+.4f vs shipped)" % (best_v, best_v - ship_v))
    print("        %s" % label(best))

    # ---- the honest number -------------------------------------------------------------
    # Re-run the WHOLE SEARCH inside each fold. Scoring a winner that was chosen on all the
    # data would leak the selection, which is the mistake that made per-finding routing look
    # like a gain.
    print("\n  CROSS-FITTED (the search itself re-run inside each fold)")
    rng = np.random.default_rng(a.seed)
    parts = np.array_split(rng.permutation(P.n), a.folds)
    held, held_ship = [], []
    for k in range(a.folds):
        te = parts[k]
        tr = np.concatenate([parts[j] for j in range(a.folds) if j != k])
        s, v = search(P, starts, a.max_members, a.max_group, a.restarts, a.seed + k,
                      rows=tr, verbose=False)
        held.append(P.score(s, te))
        held_ship.append(P.score(shipped, te))
        print("    fold %d: train %.4f -> held-out %.4f   (shipped %.4f)"
              % (k, v, held[-1], held_ship[-1]))
    cf, cfs = float(np.mean(held)), float(np.mean(held_ship))
    print("\n     searched, cross-fitted: %.4f" % cf)
    print("     shipped,  cross-fitted: %.4f" % cfs)
    print("     honest delta          : %+.4f" % (cf - cfs))

    print("\n  PER-FINDING, best against shipped")
    tasks = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
             "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
    for t, x, y in zip(tasks, P.per_column(shipped), P.per_column(best)):
        print("    %-18s %.4f -> %.4f  %+.4f%s" % (t, x, y, y - x, "  <-" if y > x else ""))

    out = HERE / ("search_structure_%s.json" % a.set)
    out.write_text(json.dumps({
        "split": a.set, "shipped": shipped, "shipped_val": ship_v,
        "best_val": best_v, "best_structure": best,
        "cross_fitted": cf, "shipped_cross_fitted": cfs, "honest_delta": cf - cfs,
        "fold_heldout": held, "fold_heldout_shipped": held_ship,
    }, indent=2), encoding="utf-8")
    print("\n  wrote %s   (%.1f s total)" % (out.name, time.time() - t0))
    print("=" * 84)


if __name__ == "__main__":
    main()
