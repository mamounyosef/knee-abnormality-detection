r"""per_class_takeover_all.py - for EVERY finding, can any single model beat the whole ensemble?

The Synovitis takeover worked: convnext_base_fluid scores 0.8767 there against the seven-model
ensemble's 0.8597, and handing it the column outright is worth +0.0014 of macro, 10/10 seeds.
The obvious next question is whether other findings have the same structure.

A WARNING ABOUT WHERE THE SPECIALIST GAINS CAME FROM. Numbers like "trauma: ACL +0.0074" or
"5slot: MCL +0.0103" are measured against the PARENT, convnext_base_relabel. Beating one model
is a much lower bar than beating seven averaged, and an ensemble is usually better than any of
its members. So a specialist can improve on its parent and still be the wrong choice for that
column. This checks against the ENSEMBLE, which is the only comparison that decides anything.

It also widens the field: every model with a prediction for a finding is a candidate, not just
the four specialists, because if a column really is better served by one model there is no
reason it must be a specialist.

Selection is cross-fitted throughout - choose on K-1 folds, score on the held-out fold - since
picking each column's winner and scoring on the same rows is the exact failure that made
per-class checkpoint routing read +0.0128 in-sample and ~0.0000 honest.

    python runs/others/ensembling/per_class_takeover_all.py
"""

import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS = ROOT / "runs"

SEVEN = ["convnext_base_relabel", "mricore_vitb_cross_attention",
         "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
         "convnext_base_25d_18slices", "swinv2_base_in22k",
         "convnext_base_25d_poolattn"]
# every model that might own a column, specialists included
CANDIDATES = SEVEN + ["convnext_base_fluid", "convnext_base_trauma", "convnext_base_wear",
                      "convnext_base_5slot", "convnext_base_25d_group3",
                      "convnext_base_25d_spatial", "convnext_base_336px", "convnext_large",
                      "dinov2_vits14_25d", "coatnet_rmlp2_384"]
BANNED = {"convnext_base_full"}          # trained on the val set; its scores are memorisation


def auc(yj, pj):
    yj = (yj > 0.5).astype(int)
    if np.isnan(pj).any() or not (0 < yj.sum() < len(yj)):
        return np.nan
    return roc_auc_score(yj, pj)


def best_val(run):
    best, bs = None, -1.0
    for f in sorted(glob.glob(str(RUNS / run / "results/predictions/*_val_s*.npz"))):
        z = np.load(f, allow_pickle=True)
        s = np.nanmean([auc(z["targets"][:, j], z["logits"][:, j])
                        for j in range(z["logits"].shape[1])])
        if s > bs:
            best, bs = z, s
    return best


ref = best_val(SEVEN[0])
TASKS = [str(t) for t in ref["tasks"]]
order = {s: i for i, s in enumerate(list(ref["study_ids"]))}
y = ref["targets"].astype(np.float64)
n = len(y)


def aligned(run):
    z = best_val(run)
    a = np.full((n, 12), np.nan)
    rows = [order[s] for s in list(z["study_ids"])]
    lg = z["logits"].astype(np.float64)
    for j, t in enumerate([str(x) for x in z["tasks"]]):
        if t in TASKS:
            a[rows, TASKS.index(t)] = lg[:, j]
    return a


A = {}
for m in CANDIDATES:
    if m in BANNED or not (RUNS / m / "results" / "predictions").is_dir():
        continue
    A[m] = aligned(m)
BASE = np.nanmean([A[m] for m in SEVEN], axis=0)


def rank01(v):
    return np.argsort(np.argsort(v)).astype(np.float64) / (len(v) - 1)


# --- 1. per finding: ensemble vs the best single model ----------------------------
print("\n1. per finding: the 7-model ensemble against the BEST single model")
print("  %-20s %10s %10s %9s  %s" % ("finding", "ensemble", "best solo", "delta", "which"))
print("  " + "-" * 74)
winners = {}
for j, t in enumerate(TASKS):
    be = auc(y[:, j], BASE[:, j])
    cand = [(auc(y[:, j], A[m][:, j]), m) for m in A if not np.isnan(A[m][:, j]).all()]
    s, m = max(cand)
    winners[t] = (s, m, s - be)
    print("  %-20s %10.4f %10.4f %+9.4f  %s%s"
          % (t, be, s, s - be, m.replace("convnext_base_", ""),
             "   BEATS ENSEMBLE" if s > be else ""))

beat = {t: v for t, v in winners.items() if v[2] > 0}
print("\n  findings where ANY single model beats all seven: %d of 12" % len(beat))
for t, (s, m, d) in sorted(beat.items(), key=lambda kv: -kv[1][2]):
    print("    %-20s %-24s %+.4f -> %+.4f of macro" % (t, m.replace("convnext_base_", ""),
                                                       d, d / 12))

# --- 2. cross-fitted takeover, choosing the owner per column on held-out folds -----
print("\n2. CROSS-FITTED takeover: per column, pick the owner on 4/5, score on the 5th")
deltas, chosen = [], {}
for seed in range(10):
    rng = np.random.default_rng(seed)
    fold = np.tile(np.arange(5), n // 5 + 1)[:n]
    rng.shuffle(fold)
    ob = np.empty((n, 12))
    orr = np.empty((n, 12))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        ob[te] = BASE[te]
        out = BASE[te].copy()
        for j in range(12):
            best_s = auc(y[tr, j], BASE[tr, j])
            best_m = None
            for m in A:
                if np.isnan(A[m][:, j]).all():
                    continue
                s = auc(y[tr, j], A[m][tr, j])
                if not np.isnan(s) and s > best_s:
                    best_s, best_m = s, m
            if best_m:
                out[:, j] = A[best_m][te, j]
                chosen[(TASKS[j], best_m)] = chosen.get((TASKS[j], best_m), 0) + 1
        orr[te] = out
    mb = np.nanmean([auc(y[:, j], ob[:, j]) for j in range(12)])
    mr = np.nanmean([auc(y[:, j], orr[:, j]) for j in range(12)])
    deltas.append(mr - mb)
d = np.array(deltas)
print("  delta %+.4f +/- %.4f   (%d/10 seeds positive)" % (d.mean(), d.std(), (d > 0).sum()))
print("  takeovers chosen (out of 50 fits):")
for (t, m), c in sorted(chosen.items(), key=lambda kv: -kv[1]):
    print("    %-20s %-26s %2d/50%s" % (t, m.replace("convnext_base_", ""), c,
                                        "   stable" if c >= 40 else ""))

# --- 2b. the same, but only take a column when the margin is big enough -----------
# Unrestricted greedy loses because it also seizes columns where the winner leads by
# noise (Lateral Meniscus 20/50, Contusion 15/50, Baker's 12/50). Requiring the candidate
# to beat the ensemble on the TRAINING folds by a real margin filters those out, and the
# margin itself is chosen on training folds, never on the scored rows.
print("\n2b. cross-fitted takeover WITH a margin requirement")
print("  %8s %12s %10s %s" % ("margin", "delta", "seeds +", "columns usually taken"))
for margin in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
    deltas2, chosen2 = [], {}
    for seed in range(10):
        rng = np.random.default_rng(seed)
        fold = np.tile(np.arange(5), n // 5 + 1)[:n]
        rng.shuffle(fold)
        ob = np.empty((n, 12)); orr = np.empty((n, 12))
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            ob[te] = BASE[te]
            out = BASE[te].copy()
            for j in range(12):
                ens = auc(y[tr, j], BASE[tr, j])
                best_s, best_m = ens + margin, None
                for m in A:
                    if np.isnan(A[m][:, j]).all():
                        continue
                    sc = auc(y[tr, j], A[m][tr, j])
                    if not np.isnan(sc) and sc > best_s:
                        best_s, best_m = sc, m
                if best_m:
                    out[:, j] = A[best_m][te, j]
                    chosen2[TASKS[j]] = chosen2.get(TASKS[j], 0) + 1
            orr[te] = out
        mb = np.nanmean([auc(y[:, j], ob[:, j]) for j in range(12)])
        mr = np.nanmean([auc(y[:, j], orr[:, j]) for j in range(12)])
        deltas2.append(mr - mb)
    d2 = np.array(deltas2)
    top = [t for t, c in sorted(chosen2.items(), key=lambda kv: -kv[1]) if c >= 40]
    print("  %8.3f %+12.4f %8d/10  %s" % (margin, d2.mean(), (d2 > 0).sum(), ", ".join(top) or "-"))

# --- 3. the a-priori rule: only the columns that are stable AND beat the ensemble --
STABLE = [t for t, (s, m, dd) in winners.items() if dd > 0]
print("\n3. a-priori takeover of just those %d column(s): %s" % (len(STABLE), STABLE))
take = BASE.copy()
for t in STABLE:
    take[:, TASKS.index(t)] = A[winners[t][1]][:, TASKS.index(t)]
mb = np.nanmean([auc(y[:, j], BASE[:, j]) for j in range(12)])
mr = np.nanmean([auc(y[:, j], take[:, j]) for j in range(12)])
print("  macro %.4f -> %.4f  (%+.4f)" % (mb, mr, mr - mb))
