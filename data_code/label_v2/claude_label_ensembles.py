"""claude_label_ensembles.py - do the two Claude hand-labellings (Opus 5.5 medium, Sonnet 5.5 high)
improve the audit-58 labels in ANY combination with Qwen, MedGemma 1.5 and each other?

Sources on the 58 audit studies (panel labels = truth):
  Q    Qwen 2.4T                                  (v2 labels)
  M    MedGemma 1.5 corpus labels, all 12 columns (images + report)
  V3   Qwen with MedGemma averaged in on the MedGemma five (the current labels, 0.9311)
  O    Opus 5.5 by hand            S    Sonnet 5.5 by hand       C = O and S averaged

Families (each in logit, probability and rank space):
  1. flat average of every subset of {Q, M, O, S}, MedGemma either on all 12 or only on its five
  2. V3 with X mixed in at w 0.1..0.7, X in {O, S, C, O+S+Q}, on all 12 / the MedGemma five /
     the other seven / everything but fluid
  3. per finding, the best source among all of the above (or keep V3)
Honest numbers: every family's best is cross-fitted (choose on 29 studies, score on the other
29, 300 splits), and a few PRE-SPECIFIED recipes (no search) get a paired bootstrap CI vs V3.

    python data_code/label_v2/claude_label_ensembles.py
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
B5 = ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]
F3 = ["Effusion", "Synovitis", "Baker's"]
OTHER7 = [t for t in T if t not in B5]
NOFLUID9 = [t for t in T if t not in F3]
CLAUDE = {"O": r"C:\temp1\audit58_labeling\labels.csv", "S": r"C:\temp1\audit58_labeling\agent_2\labels.csv"}
SPLITS, BOOT = 300, 2000

aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
U = list(aud.index)
Y = aud.loc[U, ["official_" + t for t in T]].values.astype(int)
N = len(U)


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sg(x):
    return 1 / (1 + np.exp(-x))


def auc_cols(pred, idx):
    y = Y[idx]
    r = rankdata(pred[idx], axis=0)
    n1 = y.sum(0).astype(float)
    n0 = len(idx) - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        out = ((r * y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    out[(n1 == 0) | (n0 == 0)] = np.nan
    return out


def macro(pred, idx=None):
    return float(np.nanmean(auc_cols(pred, np.arange(N) if idx is None else idx)))


def load(f, cols=T):
    return pd.read_csv(f).set_index("StudyInstanceUID").loc[U, cols].values.astype(float)


def avg(arrs, space, w=None):
    w = np.ones(len(arrs)) / len(arrs) if w is None else np.asarray(w, float)
    if space == "logit":
        return sg(sum(wi * lg(a) for wi, a in zip(w, arrs)))
    if space == "prob":
        return sum(wi * a for wi, a in zip(w, arrs))
    return sum(wi * rankdata(a, axis=0) / N for wi, a in zip(w, arrs))


def cols_idx(names):
    return [T.index(t) for t in names]


Q = load(D / "03_report_labels_qwen24t-med.csv")
M = load(D / "16_medgemma15_corpus_all.csv")
V3 = Q.copy()
V3[:, cols_idx(B5)] = sg((lg(Q[:, cols_idx(B5)]) + lg(M[:, cols_idx(B5)])) / 2)
SRC = {"Q": Q, "M": M, "O": load(CLAUDE["O"]), "S": load(CLAUDE["S"])}
SRC["C"] = sg((lg(SRC["O"]) + lg(SRC["S"])) / 2)
print("solo macro AUROC on audit-58:")
for k, v in [("Qwen", Q), ("MedGemma 1.5", M), ("V3 (current)", V3), ("Opus", SRC["O"]),
             ("Sonnet", SRC["S"]), ("Opus+Sonnet", SRC["C"])]:
    print(f"  {k:14s} {macro(v):.4f}")
base = macro(V3)

# ---------------- candidate recipes ----------------
R = {}
for space in ("logit", "prob", "rank"):
    # family 1: flat subsets of Q, M, O, S; M either everywhere or only on its five
    for r in range(1, 5):
        for sub in itertools.combinations("QMOS", r):
            R[("F1", space, "+".join(sub), "M all")] = avg([SRC[s] for s in sub], space)
            if "M" in sub and len(sub) > 1:
                rest = [SRC[s] for s in sub if s != "M"]
                P = avg(rest, space)
                P5 = avg([SRC[s] for s in sub], space)
                P = P.copy()
                P[:, cols_idx(B5)] = P5[:, cols_idx(B5)]
                R[("F1", space, "+".join(sub), "M five")] = P
    # family 2: V3 with X mixed in on a column group
    ref = V3 if space != "rank" else rankdata(V3, axis=0) / N
    for xn, X in [("O", SRC["O"]), ("S", SRC["S"]), ("C", SRC["C"]),
                  ("O+S+Q", avg([SRC["O"], SRC["S"], Q], "logit"))]:
        for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            m = avg([V3, X], space, [1 - w, w])
            for gn, g in [("all12", T), ("five", B5), ("other7", OTHER7), ("nofluid9", NOFLUID9)]:
                P = ref.copy()
                P[:, cols_idx(g)] = m[:, cols_idx(g)]
                R[("F2", space, f"V3+{xn} w{w}", gn)] = P
keys = list(R)
fam = {f: [k for k in keys if k[0] == f] for f in ("F1", "F2")}
print(f"\n{len(keys)} recipes ({len(fam['F1'])} flat subsets, {len(fam['F2'])} V3 mixes)")

allv = {k: macro(P) for k, P in R.items()}
ranked = sorted(allv.items(), key=lambda kv: -kv[1])
print(f"\nTOP 15 IN-SAMPLE (optimistic: best of {len(keys)} on 58 studies) vs V3 {base:.4f}:")
for k, v in ranked[:15]:
    print(f"  {k[0]} {k[1]:5s} {k[2]:18s} {k[3]:8s} {v:.4f} ({v - base:+.4f})")
print(f"  above V3 in-sample: {sum(v > base + 1e-9 for v in allv.values())} of {len(keys)}")
for f in ("F1", "F2"):
    k, v = max(((k, allv[k]) for k in fam[f]), key=lambda kv: kv[1])
    print(f"  best {f}: {k[1]} {k[2]} {k[3]} {v:.4f} ({v - base:+.4f})")

# family 3 (per finding) pool: every recipe's column j, plus the raw sources
colpool = dict(R)
colpool.update({("src", "-", n, "-"): v for n, v in SRC.items()})

# ---------------- cross-fitted ----------------
rng = np.random.default_rng(0)
out = {"F1": [], "F2": [], "any": [], "perfinding": []}
picks = []
done = 0
while done < SPLITS:
    perm = rng.permutation(N)
    A, B = perm[:29], perm[29:]
    if any(Y[A, j].sum() in (0, 29) or Y[B, j].sum() in (0, 29) for j in range(12)):
        continue
    done += 1
    sA = {k: macro(P, A) for k, P in R.items()}
    bB = macro(V3, B)
    for f in ("F1", "F2"):
        k = max(fam[f], key=sA.get)
        out[f].append(macro(R[k], B) - bB)
    k = max(keys, key=sA.get)
    picks.append(k)
    out["any"].append(macro(R[k], B) - bB)
    cA = {k: auc_cols(P, A) for k, P in colpool.items()}
    vA = auc_cols(V3, A)
    P = V3.copy()
    for j in range(12):
        k = max(cA, key=lambda kk: cA[kk][j])
        if cA[k][j] > vA[j]:
            P[:, j] = rankdata(colpool[k][:, j]) / N           # rank: comparable scale per column
    out["perfinding"].append(macro(P, B) - bB)
print(f"\nCROSS-FITTED ({SPLITS} splits: choose on 29 studies, score on the other 29, gain vs V3):")
for k, lab in [("F1", "best flat subset"), ("F2", "best V3 + Claude mix"), ("any", "best recipe overall"),
               ("perfinding", "best source per finding (or keep V3)")]:
    g = np.array(out[k])
    print(f"  {lab:38s} mean {g.mean():+.4f}  median {np.median(g):+.4f}  positive in {100 * (g > 0).mean():.0f}%")
from collections import Counter
print("  most picked overall:", [f"{k[1]} {k[2]} {k[3]} x{n}" for k, n in Counter(picks).most_common(4)])

# ---------------- pre-specified recipes: paired bootstrap vs V3 ----------------
PRE = {
    "V3 + (Opus+Sonnet) 50/50 logit, all 12": avg([V3, SRC["C"]], "logit"),
    "V3 + (Opus+Sonnet) 50/50 logit, other 7 only": None,
    "Qwen+Opus+Sonnet flat logit, MedGemma on its five": R[("F1", "logit", "Q+M+O+S", "M five")],
    "Qwen+MedGemma+Opus+Sonnet flat logit, all 12": R[("F1", "logit", "Q+M+O+S", "M all")],
    "Opus+Sonnet alone": SRC["C"],
}
P = V3.copy()
P[:, cols_idx(OTHER7)] = avg([V3, SRC["C"]], "logit")[:, cols_idx(OTHER7)]
PRE["V3 + (Opus+Sonnet) 50/50 logit, other 7 only"] = P
print(f"\nPRE-SPECIFIED (no search), paired bootstrap over studies ({BOOT} resamples), vs V3 {base:.4f}:")
for name, P in PRE.items():
    g = []
    for _ in range(BOOT):
        idx = rng.integers(0, N, N)
        if any(Y[idx, j].sum() in (0, N) for j in range(12)):
            continue
        g.append(macro(P, idx) - macro(V3, idx))
    g = np.array(g)
    print(f"  {name:52s} {macro(P):.4f} ({macro(P) - base:+.4f})  95% CI [{np.percentile(g, 2.5):+.4f}, "
          f"{np.percentile(g, 97.5):+.4f}]  P(gain>0) {100 * (g > 0).mean():.0f}%")

print("\nPER FINDING (AUROC):")
tab = pd.DataFrame({"V3": auc_cols(V3, np.arange(N)), "Qwen": auc_cols(Q, np.arange(N)),
                    "MedGemma": auc_cols(M, np.arange(N)), "Opus": auc_cols(SRC["O"], np.arange(N)),
                    "Sonnet": auc_cols(SRC["S"], np.arange(N)), "Opus+Sonnet": auc_cols(SRC["C"], np.arange(N)),
                    "V3+C other7": auc_cols(PRE["V3 + (Opus+Sonnet) 50/50 logit, other 7 only"], np.arange(N))},
                   index=T)
print(tab.round(3).to_string())
print("\nagreement (Spearman over all 58 x 12 cells): Opus-Sonnet {:.3f}, Opus-Qwen {:.3f}, Sonnet-Qwen {:.3f}".format(
    *[pd.Series(a.ravel()).corr(pd.Series(b.ravel()), method="spearman")
      for a, b in [(SRC["O"], SRC["S"]), (SRC["O"], Q), (SRC["S"], Q)]]))
