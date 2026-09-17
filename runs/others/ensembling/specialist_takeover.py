r"""specialist_takeover.py - hand a finding ENTIRELY to its specialist, not a share of the vote.

THE ARGUMENT. Blending a specialist into a seven-model ensemble gives it 1/8 of one column of
a mean over twelve, so a +0.0215 single-finding gain arrives as +0.0001 of macro - the seven
generalists simply outvote it. If the specialist is genuinely BETTER on that finding than the
whole ensemble is, the right move is not to add its opinion but to take everyone else's away.

That is a strictly stronger claim and it has to be checked, because it is only true where the
specialist beats the ENSEMBLE, not merely its own parent. A specialist can improve on the
model it was fine-tuned from and still be worse than seven models averaged.

TWO ROUTING RULES, and the difference matters:

  A-PRIORI: route the findings named in advance from what each specialist was built for.
    Nothing is chosen by reading scores, so there is no selection bias to undo.

  CROSS-FITTED GREEDY: decide per finding on K-1 folds, apply to the held-out fold.
    This is the honest version of "route each class to whoever is best". Per-class CHECKPOINT
    routing was tested here before and was +0.0128 in-sample, ~0.0000 cross-fitted, because
    macro AUROC is a per-column mean and routing every column to its own argmax IS the mean of
    the column maxima by construction. The fix is to never choose and score on the same rows.

    python runs/others/ensembling/specialist_takeover.py
"""

import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS = ROOT / "runs"

SPECIALIST = {"convnext_base_trauma": ["ACL", "MCL", "Contusion", "Fracture"],
              "convnext_base_wear": ["Medial Meniscus", "Lateral Meniscus", "Medial OA",
                                     "Lateral OA", "PF OA", "Baker's"],
              "convnext_base_fluid": ["Effusion", "Synovitis"],
              "convnext_base_5slot": ["MCL", "ACL"],
              # the dedicated MCL run: sliced head off dinov3_convnext_base step5600,
              # coronal-weighted slots, 512px/30 slices. Best checkpoint 0.8837 MCL.
              "dinov3_mcl_slice": ["MCL"]}

SEVEN = ["convnext_base_relabel", "mricore_vitb_cross_attention",
         "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
         "convnext_base_25d_18slices", "swinv2_base_in22k",
         "convnext_base_25d_poolattn"]


def auc(yj, pj):
    yj = (yj > 0.5).astype(int)
    if np.isnan(pj).any() or not (0 < yj.sum() < len(yj)):
        return np.nan
    return roc_auc_score(yj, pj)


def cols(y, p, rows=None):
    if rows is not None:
        y, p = y[rows], p[rows]
    return np.array([auc(y[:, j], p[:, j]) for j in range(y.shape[1])])


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


A = {m: aligned(m) for m in set(SEVEN) | set(SPECIALIST)}
BASE = np.nanmean([A[m] for m in SEVEN], axis=0)

# --- 1. per finding: does any specialist beat the WHOLE ensemble? ------------------
print("\n1. the 0.932 seven vs each specialist, on the specialist's own findings")
print("   (beating the ensemble is a much higher bar than beating the parent)")
bc = cols(y, BASE)
wins = {}
for m, ts in SPECIALIST.items():
    sc = cols(y, A[m])
    for t in ts:
        j = TASKS.index(t)
        d = sc[j] - bc[j]
        if d > 0 and (t not in wins or sc[j] > wins[t][1]):
            wins[t] = (m, sc[j], d)
        print("  %-20s %-24s ensemble %.4f  specialist %.4f  %+.4f %s"
              % (t, m.replace("convnext_base_", ""), bc[j], sc[j], d,
                 "BEATS THE ENSEMBLE" if d > 0 else ""))

print("\n  findings where a specialist beats all seven:")
for t, (m, s, d) in sorted(wins.items(), key=lambda kv: -kv[1][2]):
    print("    %-20s %-22s %+.4f  -> %+.4f of macro" % (t, m.replace("convnext_base_", ""),
                                                        d, d / 12))
if not wins:
    print("    (none)")

# --- 2. A-PRIORI takeover ---------------------------------------------------------
# Synovitis is named in advance: convnext_base_fluid was fine-tuned on Effusion+Synovitis,
# and Synovitis is both our weakest finding and the worst-labelled one (72.4% agreement
# with the panel). Nothing here is chosen by reading the table above.
APRIORI = {"Synovitis": "convnext_base_fluid"}
print("\n\n2. A-PRIORI takeover: %s" % ", ".join("%s -> %s" % (k, v.replace("convnext_base_", ""))
                                                 for k, v in APRIORI.items()))
take = BASE.copy()
for t, m in APRIORI.items():
    take[:, TASKS.index(t)] = A[m][:, TASKS.index(t)]
print("  macro %.4f -> %.4f   (%+.4f)"
      % (np.nanmean(bc), np.nanmean(cols(y, take)), np.nanmean(cols(y, take)) - np.nanmean(bc)))

# --- 3. CROSS-FITTED greedy takeover ---------------------------------------------
print("\n3. CROSS-FITTED greedy takeover (choose per finding on 4/5, score on the 5th)")
cands = {t: [m for m, ts in SPECIALIST.items() if t in ts] for t in TASKS}
deltas = []
chosen_count = {}
for seed in range(10):
    rng = np.random.default_rng(seed)
    fold = np.tile(np.arange(5), n // 5 + 1)[:n]
    rng.shuffle(fold)
    oof_b = np.empty((n, 12))
    oof_r = np.empty((n, 12))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        oof_b[te] = BASE[te]
        out = BASE[te].copy()
        for j, t in enumerate(TASKS):
            best_m, best_s = None, auc(y[tr, j], BASE[tr, j])
            for m in cands[t]:
                s = auc(y[tr, j], A[m][tr, j])
                if not np.isnan(s) and s > best_s:
                    best_m, best_s = m, s
            if best_m:
                out[:, j] = A[best_m][te, j]
                chosen_count[(t, best_m)] = chosen_count.get((t, best_m), 0) + 1
        oof_r[te] = out
    deltas.append(np.nanmean(cols(y, oof_r)) - np.nanmean(cols(y, oof_b)))
d = np.array(deltas)
print("  delta %+.4f +/- %.4f   (%d/10 seeds positive)" % (d.mean(), d.std(), (d > 0).sum()))
print("  how often each takeover was chosen (out of 50 fits):")
for (t, m), c in sorted(chosen_count.items(), key=lambda kv: -kv[1]):
    print("    %-20s %-22s %2d/50" % (t, m.replace("convnext_base_", ""), c))
