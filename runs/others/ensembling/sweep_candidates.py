r"""sweep_candidates.py - every placement of the new runs against the shipped ensemble.

sweep_new_runs.py only tries "add" and "swap for one fixed parent". A newcomer can earn its
slot by replacing ANY member, and two newcomers can matter together when neither does alone,
so this tries all of it, scores it from cached logits (no GPU), and reports the number that
survives: the CROSS-FITTED one, choose on 4/5 of the studies and score on the held-out fifth.
In-sample maxima over dozens of placements are biased upward by construction.

It also prints each candidate's logit correlation with every shipped member. Diversity is
what an ensemble slot buys: two seeds of one config sit at ~0.952, so a newcomer well below
that is bringing a different opinion, and one above it is a near-duplicate.

Both splits are run, but read audit-58 with care: a bootstrap of a real 58-study macro AUROC
gives SE 0.0195, so audit cannot resolve a difference below about 0.04.

    python runs/others/ensembling/sweep_candidates.py siglip2_base_perslice_384 \
        efficientnetv2_l_384 coatnet_rmlp2_384 medimageinsight_davit_huge_perslice_480_warm
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_new_runs as sw  # noqa: E402  - reuse its alignment, AUROC and cross-fit

SHIPPED = list(sw.SHIPPED)       # the val-clean stand-in for the 0.935 nine


def corr_to_members(cand, split):
    """Mean logit correlation of a candidate with each shipped member."""
    ids, st, _, miss = sw.align(SHIPPED + [cand], split)
    if miss:
        return {}
    c = st[-1].reshape(-1)
    return {m: float(np.corrcoef(c, st[i].reshape(-1))[0, 1])
            for i, m in enumerate(SHIPPED)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cands", nargs="+")
    ap.add_argument("--max-add", type=int, default=3)
    args = ap.parse_args()

    for split in ("val", "audit"):
        ship, miss = sw.score(SHIPPED, split)
        print("=" * 96)
        print("  %s  |  shipped nine: %.4f%s" % (split.upper(), ship,
              "   (missing: %s)" % miss if miss else ""))
        print("=" * 96)

        print("\n  SOLO and DIVERSITY (mean |r| with the shipped members; two seeds ~0.952)")
        for c in args.cands:
            s, _ = sw.score([c], split)
            r = corr_to_members(c, split)
            if not r:
                print("   %-46s  missing predictions" % c)
                continue
            lo = min(r, key=r.get)
            print("   %-46s solo %.4f | r mean %.3f, min %.3f (%s), max %.3f"
                  % (c, s, np.mean(list(r.values())), r[lo], lo[:24], max(r.values())))

        cands_all, rows = [SHIPPED], []

        for c in args.cands:
            v, m = sw.score(SHIPPED + [c], split)
            if not m:
                cands_all.append(SHIPPED + [c])
                rows.append(("+ %s" % c, v))
            for out in SHIPPED:
                combo = [x for x in SHIPPED if x != out] + [c]
                v2, m2 = sw.score(combo, split)
                if not m2:
                    cands_all.append(combo)
                    rows.append(("%s  replaces %s" % (c, out), v2))

        for r in range(2, min(args.max_add, len(args.cands)) + 1):
            for combo in itertools.combinations(args.cands, r):
                v, m = sw.score(SHIPPED + list(combo), split)
                if not m:
                    cands_all.append(SHIPPED + list(combo))
                    rows.append(("+ " + " + ".join(combo), v))

        print("\n  PLACEMENTS (%d tried), best 15 by %s:" % (len(rows), split))
        for name, v in sorted(rows, key=lambda x: -x[1])[:15]:
            print("   %+.4f  %.4f  %s" % (v - ship, v, name))

        print("\n  BEST ADD / BEST SWAP per candidate:")
        for c in args.cands:
            adds = [v for n, v in rows if n == "+ %s" % c]
            swaps = [(v, n) for n, v in rows if n.startswith(c + "  replaces")]
            if adds and swaps:
                bv, bn = max(swaps)
                print("   %-46s add %+.4f | best swap %+.4f (%s)"
                      % (c, adds[0] - ship, bv - ship, bn.split("replaces ")[-1][:30]))

        cf = sw.cross_fit(cands_all, split)
        print("\n  CROSS-FITTED over all %d candidates (incl. no change): %.4f  (%+.4f vs shipped)"
              % (len(cands_all), cf, cf - ship))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
