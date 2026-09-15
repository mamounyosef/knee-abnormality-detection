"""
forward_select.py - rebuild the ensemble from scratch, cross-fitted so it cannot lie.

WHY. The shipped six were assembled incrementally over weeks: each new member was tested
against whatever the ensemble happened to be that day, and nothing has ever gone back to ask
whether the EARLY members still earn their slots. mricore_vitb_cross_attention (0.8963 solo)
and dinov3_convnext_base (0.9011) joined before the 2.5D family existed. With ~50 members now
holding saved predictions, greedy forward selection over all of them is a different question
from "does this new member help the current six".

WHY IT MUST BE CROSS-FITTED. Choosing the best of hundreds of subsets on 704 validation
studies fits noise, and this project has already been burned by exactly that: per-class
checkpoint routing looked like +0.0128 in-sample and was worth about ZERO once cross-fitted.
Macro AUROC over 704 studies has a measured noise band of ~0.0055, which is wider than any
gain forward selection has a right to find.

So the protocol is:
  * split the 704 studies into K folds, N times with different seeds
  * run greedy selection using ONLY the K-1 training folds
  * score the chosen ensemble on the held-out fold it never saw
  * report the mean of those held-out scores, and the in-sample number beside it

The GAP between those two numbers is the selection bias, and it is the thing to look at. If
held-out selection does not beat the current six by more than the fold-to-fold spread, the
current six stand.

    python runs/others/ensembling/forward_select.py
    python runs/others/ensembling/forward_select.py --max-members 8 --seeds 10
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS = ROOT / "runs"

# RadImageNet weights are "personal, non-commercial research purposes only" and a cash prize
# is a commercial use, so those members can never ship however well they score.
BANNED = {"radimagenet_resnet50", "radimagenet_resnet50_relabel"}

# the shipped six, for the comparison that actually matters
CURRENT = ["convnext_base_relabel", "mricore_vitb_cross_attention",
           "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
           "convnext_base_25d_18slices", "swinv2_base_in22k"]


def macro(y, p, cols=None):
    out = []
    for j in range(y.shape[1]):
        yj = (y[:, j] > 0.5).astype(int)
        if 0 < yj.sum() < len(yj):
            out.append(roc_auc_score(yj, p[:, j]))
    return float(np.nanmean(out)) if out else float("nan")


def trains_on(run: str, val_ids: set) -> int:
    """How many of the VALIDATION studies this run had in its TRAINING csv.

    Non-zero means its val predictions are in-sample and its score is memorisation.
    convnext_base_full is exactly this: refit on all 4,400 studies including the 704 we
    validate on, so it scores 0.9677 where honest members score 0.90. Greedy selection picks
    it first and every other member is crowded out - and cross-fitting CANNOT catch it,
    because the leak lives inside the member, not in the selection. Checked from the config's
    actual train_csv rather than by name, so a future refit cannot sneak in.
    """
    import csv
    import yaml
    cfg_p = RUNS / run / "config.yaml"
    if not cfg_p.exists():
        return 0
    try:
        c = yaml.safe_load(io.open(cfg_p, encoding="utf-8"))
        name = (c.get("paths") or {}).get("train_csv")
        if not name:
            return 0
        f = ROOT / "data" / name
        if not f.exists():
            return 0
        with io.open(f, encoding="utf-8") as fh:
            ids = {r["StudyInstanceUID"] for r in csv.DictReader(fh)}
        return len(ids & val_ids)
    except Exception:
        return 0


def load_members(min_solo: float):
    """Every run's BEST val prediction, aligned on study id, with matching targets only."""
    ref_ids = ref_y = None
    members = {}
    for d in sorted(RUNS.glob("*/results/predictions")):
        run = d.parent.parent.name
        if run in BANNED:
            continue
        best, bs = None, -1.0
        for f in sorted(d.glob("*val_s*.npz")):
            z = np.load(f, allow_pickle=True)
            if z["logits"].shape[1] != 12:          # task-group branches, not standalone
                continue
            sc_ = macro(z["targets"], z["logits"])
            if sc_ > bs:
                best, bs = f, sc_
        if best is None:
            continue
        z = np.load(best, allow_pickle=True)
        ids = list(z["study_ids"])
        if ref_ids is None:
            ref_ids, ref_y = ids, z["targets"].astype(np.float64)
            order = {s: i for i, s in enumerate(ref_ids)}
        if set(ids) != set(ref_ids):
            print("  skip %-38s different study set" % run)
            continue
        al = np.empty((len(ref_ids), 12), np.float64)
        ay = np.empty((len(ref_ids), 12), np.float64)
        idx = [order[s] for s in ids]
        al[idx] = z["logits"].astype(np.float64)
        ay[idx] = z["targets"].astype(np.float64)
        # a run trained on the OLD labels has different targets on the same studies; blending
        # it in would be scoring two different problems together
        if not np.allclose(ay, ref_y, atol=1e-6):
            print("  skip %-38s targets differ (old label set)" % run)
            continue
        leak = trains_on(run, set(ref_ids))
        if leak:
            print("  skip %-38s LEAK: trained on %d of the %d val studies"
                  % (run, leak, len(ref_ids)))
            continue
        if bs < min_solo:
            continue
        members[run] = al
    return ref_y, members


def greedy(y, M, names, rows, max_members, allow_repeats=True):
    """Forward selection on the given rows only. Returns the chosen member list."""
    chosen, cur = [], None
    best_so_far = -1.0
    while len(chosen) < max_members:
        gain, pick = None, None
        for n in names:
            if not allow_repeats and n in chosen:
                continue
            cand = M[n][rows] if cur is None else (cur * len(chosen) + M[n][rows]) / (len(chosen) + 1)
            s = macro(y[rows], cand)
            if gain is None or s > gain:
                gain, pick = s, n
        if gain is None or gain <= best_so_far + 1e-6:
            break
        chosen.append(pick)
        cur = M[pick][rows] if cur is None else (cur * (len(chosen) - 1) + M[pick][rows]) / len(chosen)
        best_so_far = gain
    return chosen


def blend(M, names, rows):
    return np.mean([M[n][rows] for n in names], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max-members", type=int, default=8)
    ap.add_argument("--min-solo", type=float, default=0.86,
                    help="ignore members below this solo macro AUROC")
    a = ap.parse_args()

    y, M = load_members(a.min_solo)
    names = sorted(M)
    n = len(y)
    print("\n%d members over %d studies (solo >= %.2f, RadImageNet excluded)"
          % (len(names), n, a.min_solo))

    have = [c for c in CURRENT if c in M]
    if len(have) != len(CURRENT):
        print("  NOTE: current six missing %s" % [c for c in CURRENT if c not in M])
    cur_full = macro(y, blend(M, have, np.arange(n)))
    print("the shipped six, in-sample over all %d: %.4f" % (n, cur_full))

    # ---- in-sample greedy, for orientation only ----------------------------------
    allrows = np.arange(n)
    ins = greedy(y, M, names, allrows, a.max_members)
    print("\nIN-SAMPLE greedy (NOT evidence - this is the number that lies):")
    print("   %.4f  %s" % (macro(y, blend(M, ins, allrows)), " + ".join(ins)))

    # ---- cross-fitted ------------------------------------------------------------
    print("\nCROSS-FITTED, %d folds x %d seeds" % (a.folds, a.seeds))
    sel_scores, cur_scores, picks = [], [], {}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        fold = np.tile(np.arange(a.folds), n // a.folds + 1)[:n]
        rng.shuffle(fold)
        oof_sel = np.empty((n, 12))
        oof_cur = np.empty((n, 12))
        for k in range(a.folds):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            ch = greedy(y, M, names, tr, a.max_members)
            for m in ch:
                picks[m] = picks.get(m, 0) + 1
            oof_sel[te] = blend(M, ch, te)
            oof_cur[te] = blend(M, have, te)
        sel_scores.append(macro(y, oof_sel))
        cur_scores.append(macro(y, oof_cur))
        print("  seed %d:  selected %.4f   shipped six %.4f   delta %+.4f"
              % (seed, sel_scores[-1], cur_scores[-1], sel_scores[-1] - cur_scores[-1]))

    s, c = np.array(sel_scores), np.array(cur_scores)
    d = s - c
    print("\n" + "=" * 74)
    print("  forward selection (held out)  %.4f +/- %.4f" % (s.mean(), s.std()))
    print("  the shipped six (held out)    %.4f +/- %.4f" % (c.mean(), c.std()))
    print("  DELTA                         %+.4f +/- %.4f   (%d/%d seeds positive)"
          % (d.mean(), d.std(), (d > 0).sum(), len(d)))
    print("  in-sample greedy              %.4f   <- the selection bias is this minus the above"
          % macro(y, blend(M, ins, allrows)))
    print("=" * 74)

    print("\nhow often each member was picked across all %d fits:" % (a.folds * a.seeds))
    for m, cnt in sorted(picks.items(), key=lambda kv: -kv[1]):
        flag = "  <- in the shipped six" if m in have else ""
        print("   %-42s %3d/%d%s" % (m, cnt, a.folds * a.seeds, flag))
    print("\nshipped members NEVER picked:")
    never = [m for m in have if picks.get(m, 0) == 0]
    print("   %s" % (", ".join(never) if never else "(none - all still earn a slot)"))


if __name__ == "__main__":
    main()
