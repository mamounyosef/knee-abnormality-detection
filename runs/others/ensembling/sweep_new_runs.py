r"""sweep_new_runs.py - does anything from the recent batch of runs earn an ensemble slot?

Five runs closed in the last two days and none of them beat their parent solo. That is not
the same as being useless: `convnext_base_25d_poolattn` has the WORST solo score of its
family and the BEST ensemble contribution, because replacing the mean pool really
decorrelated it. A run that loses solo but predicts differently can still pay.

So this asks the question properly, in three stages of increasing company:

  1. PAIRS     each new run beside its own parent, convnext_base_25d_18slices, alone.
               The cleanest read on whether the run carries anything the parent does not.
  2. SHIPPED   each new run added to, and substituted into, the current best ensemble.
  3. WIDER     the fold runs and the rest of the pool, forward-selected from scratch.

Everything is scored from cached per-study logits, so the whole sweep is a file read and
costs no GPU.

TWO THINGS THAT KEEP THIS HONEST.
  * Leaky members are excluded by name. The 10 fold checkpoints and convnext_base_full
    trained ON the validation studies, so any ensemble containing them scores its own
    training data. They are listed in LEAKY below.
  * Every headline number is also reported CROSS-FITTED: choose the combination on 4/5 of
    the studies, score it on the held-out fifth, repeat. In-sample selection over hundreds
    of combinations is exactly how this project has fooled itself before - per-finding
    checkpoint routing looked like +0.0128 in sample and was -0.0007 cross-fitted.

    python runs/others/ensembling/sweep_new_runs.py
    python runs/others/ensembling/sweep_new_runs.py --set audit
"""

import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RUNS))

PARENT = "convnext_base_relabel"

NEW = ["efficientnetv2_l_384", "coatnet_rmlp2_384"]

# the ensemble currently shipped, LB 0.935. convnext_base_relabel stands in for
# convnext_base_full_all_knees_same_orientation, which is what actually ships: the full-data
# run trained ON the val studies, so scoring it here would score its own training data.
SHIPPED = ["convnext_base_relabel", "mricore_vitb_cross_attention",
           "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
           "convnext_base_25d_all_knees_same_orientation", "swinv2_base_in22k",
           "convnext_base_25d_poolattn",
           "convnext_base_25d_perfinding_slots_and_pooling",
           "medimageinsight_davit_huge_336"]

# trained on the val studies - including one makes every score meaningless
# Leakage is decided by the run's OWN train_csv, not by its name. Matching "_full" was
# wrong in both directions: mricore_vitb_mae_cross_attention_full is NOT a full-data run
# (its "full" means full cross-slice fusion, and it trains on the normal split), while a
# future full-data run named anything else would slip straight through. The fold runs stay
# a name rule because their split is encoded only in the suffix.
FULL_CSVS = ("train_full",)   # 08_train_full and every relabelled NN_train_full_* (e.g. 22_*-w50)
LEAKY = re.compile(r"_f[0-4]$")


def _trains_on_val(run_name: str) -> bool:
    """True if this run's config trains on the validation studies."""
    import yaml
    p = RUNS / run_name / "config.yaml"
    if not p.exists():
        return False
    try:
        csv = str(((yaml.safe_load(p.read_text(encoding="utf-8")) or {})
                   .get("paths") or {}).get("train_csv") or "")
    except Exception:
        return False
    return any(t in csv for t in FULL_CSVS)

# RadImageNet's weights are "personal, non-commercial (e.g., academic) research purposes
# only... any commercial use, sale, or other monetization is strictly prohibited", and a
# cash prize is a commercial use. It cannot ship, so it must not be allowed to win a sweep
# either - it was selected at step 2 here for +0.0055.
BANNED = re.compile(r"^radimagenet")


def auroc(y, p):
    """AUROC per column, averaged. Rank based, so it needs no sklearn."""
    out = []
    for j in range(y.shape[1]):
        t = (y[:, j] > 0.5).astype(int)
        if t.min() == t.max():
            continue
        s = p[:, j]
        r = np.empty(len(s), float)
        order = np.argsort(s, kind="mergesort")
        sr = s[order]
        i = 0
        while i < len(sr):                      # average ranks over ties
            k = i
            while k + 1 < len(sr) and sr[k + 1] == sr[i]:
                k += 1
            r[order[i:k + 1]] = (i + k) / 2.0 + 1
            i = k + 1
        n1 = t.sum()
        n0 = len(t) - n1
        out.append((r[t == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    return float(np.mean(out)) if out else float("nan")


def best_npz(run, split):
    """The run's best-scoring validation file for this split, by its own val_log."""
    d = RUNS / run / "results" / "predictions"
    if not d.exists():
        return None
    files = sorted(d.glob("*_%s_s*.npz" % split))
    if not files:
        return None
    import pandas as pd
    vlog = RUNS / run / "results" / "val_log.csv"
    best_step = None
    if vlog.exists():
        v = pd.read_csv(vlog)
        if "val_macro_auroc" in v.columns and len(v):
            best_step = int(v.loc[v["val_macro_auroc"].idxmax(), "step"])
    if best_step is not None:
        for f in files:
            if f.stem.endswith("_s%d" % best_step):
                return f
    return files[-1]


def load(run, split, cache={}):
    key = (run, split)
    if key in cache:
        return cache[key]
    f = best_npz(run, split)
    if f is None:
        cache[key] = None
        return None
    z = np.load(f, allow_pickle=True)
    ids = np.array([str(s) for s in z["study_ids"]])
    cache[key] = (ids, z["logits"].astype(np.float64), z["targets"].astype(np.float64))
    return cache[key]


def align(members, split):
    """Members on a common study order. Returns (ids, stack (M, N, 12), targets).

    A SINGLE-TASK run (convnext_base_25d_pfoa predicts PF OA alone) has one column, not
    twelve, so it cannot be averaged into a 12-column blend at all - it can only ROUTE its
    own column, which per-finding routing already tested and the leaderboard rejected. Such
    members are reported as missing rather than silently reshaped.
    """
    got = [(m, load(m, split)) for m in members]
    missing = [m for m, v in got if v is None]
    missing += [m for m, v in got if v is not None and v[1].shape[1] != 12]
    got = [(m, v) for m, v in got if v is not None and v[1].shape[1] == 12]
    if not got:
        return None, None, None, missing or list(members)
    if missing:
        return None, None, None, missing
    common = set(got[0][1][0])
    for _, (ids, _, _) in got[1:]:
        common &= set(ids)
    common = np.array(sorted(common))
    stack, y = [], None
    for _, (ids, lg, tg) in got:
        pos = {s: i for i, s in enumerate(ids)}
        sel = np.array([pos[s] for s in common])
        stack.append(lg[sel])
        if y is None:
            y = tg[sel]
    return common, np.stack(stack), y, []


def score(members, split):
    ids, st, y, missing = align(members, split)
    if missing:
        return float("nan"), missing
    return auroc(y, st.mean(axis=0)), []


def cross_fit(cands, split, folds=5, seed=0):
    """Pick the best candidate on 4/5 of the studies, score it on the held-out fifth.

    This is the number that survives. An in-sample maximum over many candidates is biased
    upward by construction, which is how per-finding routing looked like a gain and was not.
    """
    ids, _, _, missing = align(SHIPPED, split)
    if missing:
        return float("nan")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    parts = np.array_split(order, folds)
    held = []
    for k in range(folds):
        te = parts[k]
        tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        best, best_v = None, -1
        for c in cands:
            i2, st, y, miss = align(c, split)
            if miss:
                continue
            v = auroc(y[tr], st.mean(axis=0)[tr])
            if v > best_v:
                best, best_v = c, v
        if best is None:
            continue
        i2, st, y, _ = align(best, split)
        held.append(auroc(y[te], st.mean(axis=0)[te]))
    return float(np.mean(held)) if held else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="val", choices=["val", "audit"])
    ap.add_argument("--max-add", type=int, default=2,
                    help="how many new runs to try adding at once in stage 3")
    a = ap.parse_args()
    split = a.set

    print("=" * 78)
    print("  ENSEMBLE SWEEP over the recent runs, scored on %s" % split)
    print("=" * 78)

    base, _ = score([PARENT], split)
    print("\n  parent %s alone: %.4f" % (PARENT, base))
    solo = {}
    for m in NEW:
        v, miss = score([m], split)
        solo[m] = v
        print("    %-46s solo %.4f  (%+.4f vs parent)" % (m, v, v - base))

    # ---- stage 1: each new run BESIDE the parent, nothing else -------------------------
    print("\n  1. PAIRED WITH THE PARENT (does it add anything the parent lacks?)")
    print("     %-46s %8s %9s" % ("pair", split, "vs parent"))
    pairs = {}
    for m in NEW:
        v, _ = score([PARENT, m], split)
        pairs[m] = v
        flag = "  <- ADDS" if v > base + 0.0005 else ""
        print("     %-46s %8.4f %+9.4f%s" % (PARENT + " + " + m.split("25d_")[-1], v,
                                             v - base, flag))
    best_pair = max(pairs.items(), key=lambda kv: kv[1])
    v3, _ = score([PARENT] + NEW, split)
    print("     %-46s %8.4f %+9.4f" % ("parent + ALL four", v3, v3 - base))

    # ---- stage 2: against the shipped ensemble -----------------------------------------
    ship, miss = score(SHIPPED, split)
    print("\n  2. AGAINST THE SHIPPED ENSEMBLE (%d members, %s %.4f)"
          % (len(SHIPPED), split, ship))
    if miss:
        print("     missing predictions for: %s" % ", ".join(miss))
    cands = [SHIPPED]
    print("     %-52s %8s %9s" % ("combination", split, "vs shipped"))
    for m in NEW:
        c = SHIPPED + [m]
        v, ms = score(c, split)
        if not ms:
            cands.append(c)
            print("     %-52s %8.4f %+9.4f%s" % ("+ " + m.split("25d_")[-1], v, v - ship,
                                                 "  <- GAIN" if v > ship + 0.0005 else ""))
        # and substituted for its own parent, which is the fair swap
        c2 = [x for x in SHIPPED if x != PARENT] + [m]
        v2, ms2 = score(c2, split)
        if not ms2:
            cands.append(c2)
            print("     %-52s %8.4f %+9.4f%s"
                  % ("replace 18slices with " + m.split("25d_")[-1], v2, v2 - ship,
                     "  <- GAIN" if v2 > ship + 0.0005 else ""))
    for r in range(2, min(a.max_add, len(NEW)) + 1):
        for combo in itertools.combinations(NEW, r):
            c = SHIPPED + list(combo)
            v, ms = score(c, split)
            if not ms:
                cands.append(c)
                if v > ship + 0.0005:
                    print("     %-52s %8.4f %+9.4f  <- GAIN"
                          % ("+ " + " + ".join(x.split("25d_")[-1] for x in combo),
                             v, v - ship))

    # ---- stage 3: the wider pool, forward selection ------------------------------------
    print("\n  3. FORWARD SELECTION over the whole non-leaky pool")
    pool = sorted(p.parent.parent.name for p in RUNS.glob("*/results/predictions")
                  if not LEAKY.search(p.parent.parent.name)
                  and not _trains_on_val(p.parent.parent.name)
                  and not BANNED.search(p.parent.parent.name))
    pool = [r for r in pool if best_npz(r, split) is not None]
    print("     %d runs with predictions (leaky excluded)" % len(pool))
    chosen, cur = [], -1
    while len(chosen) < 9:
        best_m, best_v = None, cur
        for m in pool:
            if m in chosen:
                continue
            v, ms = score(chosen + [m], split)
            if not ms and v > best_v:
                best_m, best_v = m, v
        if best_m is None:
            break
        chosen.append(best_m)
        cands.append(list(chosen))
        print("     %d. + %-44s %8.4f  (%+.4f)"
              % (len(chosen), best_m, best_v, best_v - cur if cur > 0 else 0.0))
        cur = best_v

    # ---- the honest number --------------------------------------------------------------
    print("\n  CROSS-FITTED (choose on 4/5 of the studies, score on the held-out fifth)")
    cf = cross_fit(cands, split)
    cf_ship = cross_fit([SHIPPED], split)
    print("     best-of-%d in sample : %.4f" % (len(cands), max(
        score(c, split)[0] for c in cands)))
    print("     same, cross-fitted   : %.4f" % cf)
    print("     shipped, cross-fitted: %.4f" % cf_ship)
    print("     honest delta         : %+.4f" % (cf - cf_ship))
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
