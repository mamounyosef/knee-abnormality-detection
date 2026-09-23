r"""class_routing.py - should each finding be scored by its own SUBSET of the members?

The idea: some members help some findings and hurt others. If a member degrades Fracture,
it should not vote on Fracture. But one member per finding is noisy, which is why we
ensemble in the first place, so every finding keeps a subset of at least MIN_K members.

THE TRAP THIS FILE EXISTS TO AVOID. Pick, per finding, the best of 466 subsets on the same
704 studies you then score it on, and you will always "find" a gain: some subsets win by
chance. This project has been fooled by exactly that before (per-finding routing looked like
+0.0128 in sample and was -0.0007 cross-fitted; it then lost on the leaderboard). So every
number that matters here is CROSS-FITTED: the routing is chosen on 4/5 of the studies and
applied to the held-out fifth, the five held-out fifths are stitched into one out-of-fold
prediction for all 704 studies, and that is scored. Repeated over R random splits.

Part A - the shipped ensemble (the val-clean stand-in for the 0.937 nine).
Part B - the wide pool: every non-leaky, licence-clean run in the repo with 12-column val
         predictions, per-finding forward selection.

    python runs/others/ensembling/class_routing.py
    python runs/others/ensembling/class_routing.py --repeats 20 --min-k 3 --part A
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_new_runs as sw  # noqa: E402

TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


# ---------------------------------------------------------------------------------------
# fast AUROC: many candidate score columns against one label column at once
# ---------------------------------------------------------------------------------------
def auroc_many(scores, y):
    """scores (C, n), y (n,) binary -> (C,) AUROC, ties averaged (Mann-Whitney)."""
    pos = y.astype(bool)
    n1 = int(pos.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(scores.shape[0], np.nan)
    r = rankdata(scores, axis=1)
    return (r[:, pos].sum(1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def macro(pred, Y):
    return float(np.nanmean([auroc_many(pred[:, j][None], Y[:, j])[0]
                             for j in range(Y.shape[1])]))


def per_class(pred, Y):
    return np.array([auroc_many(pred[:, j][None], Y[:, j])[0] for j in range(Y.shape[1])])


# ---------------------------------------------------------------------------------------
# routing rules. Each takes the stack on the TRAINING studies and returns, per finding,
# the list of member indices that vote on it.
# ---------------------------------------------------------------------------------------
def rule_best_subset(st, Y, subsets, sub_means):
    """Exhaustive: per finding, the subset (size >= min_k) with the best training AUROC."""
    out = []
    for j in range(Y.shape[1]):
        a = auroc_many(sub_means[:, :, j], Y[:, j])
        out.append(subsets[int(np.nanargmax(a))])
    return out


def rule_drop_degraders(st, Y, tau, min_k):
    """Per finding, drop a member only if leaving it out raises training AUROC by > tau."""
    M = st.shape[0]
    out = []
    for j in range(Y.shape[1]):
        keep = list(range(M))
        while len(keep) > min_k:
            base = auroc_many(st[keep, :, j].mean(0)[None], Y[:, j])[0]
            cands = np.stack([st[[k for k in keep if k != m], :, j].mean(0) for m in keep])
            gains = auroc_many(cands, Y[:, j]) - base
            b = int(np.nanargmax(gains))
            if gains[b] <= tau:
                break
            keep.pop(b)
        out.append(keep)
    return out


def rule_top_k(st, Y, k):
    """Per finding, the k members with the best SOLO training AUROC."""
    out = []
    for j in range(Y.shape[1]):
        a = auroc_many(st[:, :, j], Y[:, j])
        out.append(list(np.argsort(-a)[:k]))
    return out


def rule_forward(st, Y, min_k, max_k, start=None):
    """Per finding, greedy forward selection (with replacement off), at least min_k."""
    M = st.shape[0]
    out = []
    for j in range(Y.shape[1]):
        chosen = list(start[j]) if start is not None else []
        cur = auroc_many(st[chosen, :, j].mean(0)[None], Y[:, j])[0] if chosen else -1.0
        while len(chosen) < max_k:
            rest = [m for m in range(M) if m not in chosen]
            if not rest:
                break
            cands = np.stack([st[chosen + [m], :, j].mean(0) for m in rest])
            a = auroc_many(cands, Y[:, j])
            b = int(np.nanargmax(a))
            if a[b] <= cur and len(chosen) >= min_k:
                break
            chosen.append(rest[b])
            cur = a[b]
        out.append(chosen)
    return out


def apply(st, routing):
    """Routed prediction: finding j = mean of its own members' logits."""
    return np.stack([st[routing[j], :, j].mean(0) for j in range(len(routing))], axis=1)


def cross_fit(st, Y, rule, repeats, folds=5, seed=0):
    """Out-of-fold routed predictions for every study, scored once per repeat."""
    n = st.shape[1]
    flat = macro(st.mean(0), Y)
    deltas, per_cls = [], []
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        parts = np.array_split(rng.permutation(n), folds)
        oof = np.zeros((n, Y.shape[1]))
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[i] for i in range(folds) if i != k])
            routing = rule(st[:, tr], Y[tr])
            oof[te] = apply(st[:, te], routing)
        deltas.append(macro(oof, Y) - flat)
        per_cls.append(per_class(oof, Y) - per_class(st.mean(0), Y))
    return flat, np.array(deltas), np.array(per_cls)


def report_cf(name, flat, d, pc):
    print("   %-34s cross-fitted %+.4f  (sd %.4f, %d/%d repeats > 0)   -> %.4f"
          % (name, d.mean(), d.std(), int((d > 0).sum()), len(d), flat + d.mean()))
    return pc.mean(0)


# ---------------------------------------------------------------------------------------
def part_a(args):
    members = list(sw.SHIPPED)
    print("=" * 100)
    print("  PART A - routing inside the shipped nine (val-clean stand-in for the 0.937 nine)")
    print("=" * 100)
    ids, st, Yv, miss = sw.align(members, "val")
    if miss:
        raise SystemExit("missing: %s" % miss)
    Y = (Yv > 0.5).astype(int)
    M, N, T = st.shape
    short = [m.replace("convnext_base_", "cnb_").replace("mricore_vitb_", "mri_")
             .replace("_cross_attention", "_xa").replace("medimageinsight_davit_huge", "mii")
             .replace("_same_orientation", "").replace("_slots_and_pooling", "")[:14]
             for m in members]
    flat_pc = per_class(st.mean(0), Y)
    print("\n  %d studies. Positives per finding: %s"
          % (N, ", ".join("%s %d" % (t.split()[0][:6], Y[:, j].sum())
                          for j, t in enumerate(TASKS))))

    print("\n  A1. SOLO AUROC per member per finding (flat nine on the last line)")
    print("   %-16s" % "finding" + "".join("%8s" % s[:7] for s in short) + "   flat9")
    solo = np.stack([per_class(st[m], Y) for m in range(M)])
    for j, t in enumerate(TASKS):
        print("   %-16s" % t[:16] + "".join("%8.3f" % solo[m, j] for m in range(M))
              + "   %.3f" % flat_pc[j])

    print("\n  A2. LEAVE-ONE-OUT per finding: flat nine WITHOUT the member minus WITH it.")
    print("      POSITIVE = the member DEGRADES that finding (removing it helps). In sample.")
    print("   %-16s" % "finding" + "".join("%8s" % s[:7] for s in short))
    loo = np.zeros((M, T))
    for m in range(M):
        rest = [k for k in range(M) if k != m]
        loo[m] = per_class(st[rest].mean(0), Y) - flat_pc
    for j, t in enumerate(TASKS):
        print("   %-16s" % t[:16] + "".join("%+8.4f" % loo[m, j] for m in range(M)))
    print("   %-16s" % "macro" + "".join("%+8.4f" % loo[m].mean() for m in range(M)))
    worst = sorted(((loo[m, j], short[m], TASKS[j]) for m in range(M) for j in range(T)),
                   reverse=True)[:8]
    print("   biggest apparent degraders (in sample): "
          + "; ".join("%s on %s %+.4f" % (s, t, v) for v, s, t in worst))

    subsets = [list(c) for k in range(args.min_k, M + 1)
               for c in itertools.combinations(range(M), k)]
    sub_means = np.stack([st[s].mean(0) for s in subsets])            # (S, N, T)
    print("\n  A3. EXHAUSTIVE best subset per finding (size >= %d, %d subsets), IN SAMPLE:"
          % (args.min_k, len(subsets)))
    routing = rule_best_subset(st, Y, subsets, sub_means)
    ins = per_class(apply(st, routing), Y)
    for j, t in enumerate(TASKS):
        print("   %-16s %.4f -> %.4f (%+.4f)  %d members, drops: %s"
              % (t, flat_pc[j], ins[j], ins[j] - flat_pc[j], len(routing[j]),
                 ", ".join(short[m] for m in range(M) if m not in routing[j]) or "-"))
    print("   IN-SAMPLE macro %.4f -> %.4f (%+.4f)   <- biased upward by construction"
          % (flat_pc.mean(), ins.mean(), ins.mean() - flat_pc.mean()))

    print("\n  A4. CROSS-FITTED (%d repeats x 5 folds) - THE NUMBERS THAT COUNT" % args.repeats)
    rules = [
        ("best subset (>=%d)" % args.min_k,
         lambda s, y: rule_best_subset(s, y, subsets, np.stack([s[q].mean(0) for q in subsets]))),
        ("drop degraders, tau 0", lambda s, y: rule_drop_degraders(s, y, 0.0, args.min_k)),
        ("drop degraders, tau 0.002", lambda s, y: rule_drop_degraders(s, y, 0.002, args.min_k)),
        ("drop degraders, tau 0.005", lambda s, y: rule_drop_degraders(s, y, 0.005, args.min_k)),
        ("top-5 by solo", lambda s, y: rule_top_k(s, y, 5)),
        ("top-7 by solo", lambda s, y: rule_top_k(s, y, 7)),
        ("forward selection (>=%d)" % args.min_k,
         lambda s, y: rule_forward(s, y, args.min_k, M)),
    ]
    best_pc, best_name, best_d = None, None, -9
    for name, rule in rules:
        flat, d, pc = cross_fit(st, Y, rule, args.repeats)
        mpc = report_cf(name, flat, d, pc)
        if d.mean() > best_d:
            best_d, best_name, best_pc = d.mean(), name, mpc
    print("\n   per-finding cross-fitted delta for the best rule (%s):" % best_name)
    for j, t in enumerate(TASKS):
        print("     %-16s %+.4f" % (t, best_pc[j]))

    print("\n  A5. AUDIT-58 check: routing chosen on ALL of val, applied to audit (independent")
    print("      studies, but its resolution is ~0.04, so only a large move means anything)")
    _, sa, Ya, ma = sw.align(members, "audit")
    if not ma:
        Ya = (Ya > 0.5).astype(int)
        for name, rt in [("best subset", routing),
                         ("drop degraders tau 0.002",
                          rule_drop_degraders(st, Y, 0.002, args.min_k))]:
            print("   %-26s flat %.4f -> routed %.4f (%+.4f)"
                  % (name, macro(sa.mean(0), Ya), macro(apply(sa, rt), Ya),
                     macro(apply(sa, rt), Ya) - macro(sa.mean(0), Ya)))


def part_b(args):
    print("\n" + "=" * 100)
    print("  PART B - the WIDE pool: every non-leaky, licence-clean run with 12-column predictions")
    print("=" * 100)
    runs = sorted(p.parent.parent.name for p in sw.RUNS.glob("*/results/predictions")
                  if not sw.LEAKY.search(p.parent.parent.name)
                  and not sw._trains_on_val(p.parent.parent.name)
                  and not sw.BANNED.search(p.parent.parent.name))
    ok = []
    for r in runs:
        v = sw.load(r, "val")
        if v is not None and v[1].shape[1] == 12:
            ok.append(r)
    # align on the shipped set's studies; drop any run missing studies there
    ids0, _, _, _ = sw.align(list(sw.SHIPPED), "val")
    keep = []
    for r in ok:
        ids = set(sw.load(r, "val")[0])
        if set(ids0) <= ids:
            keep.append(r)
    print("   %d runs with val predictions, %d cover all %d shipped-set studies"
          % (len(ok), len(keep), len(ids0)))
    ship_idx = [keep.index(m) for m in sw.SHIPPED if m in keep]
    _, st, Yv, miss = sw.align(keep, "val")
    Y = (Yv > 0.5).astype(int)
    M = st.shape[0]
    flat9 = st[ship_idx].mean(0)
    print("   shipped nine (flat) on this alignment: %.4f" % macro(flat9, Y))

    start9 = [list(ship_idx)] * 12
    rules = [
        ("pool forward from EMPTY (>=%d)" % args.min_k,
         lambda s, y: rule_forward(s, y, args.min_k, 12)),
        ("pool forward from the NINE (add only)",
         lambda s, y: rule_forward(s, y, args.min_k, 14, start=start9)),
    ]
    for name, rule in rules:
        n = st.shape[1]
        deltas = []
        used = {}
        for r in range(args.repeats_b):
            rng = np.random.default_rng(100 + r)
            parts = np.array_split(rng.permutation(n), 5)
            oof = np.zeros((n, 12))
            for k in range(5):
                te = parts[k]
                tr = np.concatenate([parts[i] for i in range(5) if i != k])
                rt = rule(st[:, tr], Y[tr])
                for grp in rt:
                    for m in grp:
                        used[keep[m]] = used.get(keep[m], 0) + 1
                oof[te] = apply(st[:, te], rt)
            deltas.append(macro(oof, Y) - macro(flat9, Y))
        d = np.array(deltas)
        print("\n   %-40s cross-fitted vs the flat nine %+.4f (sd %.4f, %d/%d > 0)"
              % (name, d.mean(), d.std(), int((d > 0).sum()), len(d)))
        top = sorted(used.items(), key=lambda x: -x[1])[:12]
        print("     most-selected members: " + ", ".join("%s(%d)" % (k[:28], v) for k, v in top))

    rt = rule_forward(st, Y, args.min_k, 12)
    distinct = sorted({keep[m] for g in rt for m in g})
    print("\n   in-sample forward-from-empty routing uses %d DISTINCT members (inference cost):"
          % len(distinct))
    for j, t in enumerate(TASKS):
        print("     %-16s %s" % (t, ", ".join(keep[m][:26] for m in rt[j])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--repeats-b", type=int, default=8)
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--part", default="AB")
    args = ap.parse_args()
    if "A" in args.part:
        part_a(args)
    if "B" in args.part:
        part_b(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
