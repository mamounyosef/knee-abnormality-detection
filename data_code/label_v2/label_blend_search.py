"""
label_blend_search.py - does ANY extra labeller improve a base label set on audit-58?

For every source X (every audit-58 labeller file, plus same-model mode mixes) it builds
    base with some columns replaced by (1-w)*base + w*X
for columns in {fluid 3, the MedGemma five, all 12}, w in {0.1..0.5}, and averaging in logit,
probability or rank space. Reports the top recipes in-sample (optimistic: best of hundreds on 58
studies), the best single-column mix per finding, and a CROSS-FITTED estimate (choose on 29
studies, score on the other 29, 200 splits) of both the best whole recipe and a per-finding pick.
Fluid-only sources (3 columns) only ever touch the fluid columns.

    python data_code/label_v2/label_blend_search.py --base current     # Qwen + MedGemma 1.5 on 5
    python data_code/label_v2/label_blend_search.py --base oof         # + 30% out-of-fold, all 12
    python data_code/label_v2/label_blend_search.py --base w50 --all-sources   # the v4 labels (7 Oct), every labeller file
"""
import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "data"
sys.path.insert(0, str(ROOT / "runs/others/ensembling"))
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
F3 = ["Effusion", "Synovitis", "Baker's"]
B5 = ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]
# name -> file; fluid-only files have just the three fluid columns
SOURCES = {
    "M3": "03_report_labels_baichuanm3-avg2_validate58.csv",
    "Nemo": "03_report_labels_nemotron3ultra_validate58.csv",
    "Kimi3": "03_report_labels_kimi3_validate58.csv",
    "Gem31": "03_report_labels_gemma431b-med_validate58.csv",
    "G4img": "03_report_labels_gemma4-img_validate58.csv",
    "G4rep": "03_report_labels_gemma4-img-rep_validate58.csv",
    "G4fl": "03_report_labels_gemma4-fluid_validate58.csv",
    "G4flrep": "03_report_labels_gemma4-fluid-rep_validate58.csv",
    "G27a": "03_report_labels_medgemma27-all_validate58.csv",
    "G27b": "03_report_labels_medgemma27-all36_validate58.csv",
    **{f"H{m}": f"03_report_labels_hulumed14-{m}_validate58.csv"
       for m in ("all180", "all180T", "all96", "all96S", "fluid", "all96R")},
}


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sg(x):
    return 1 / (1 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["current", "oof", "w50"], default="current")
    ap.add_argument("--all-sources", action="store_true", help="every data/03_report_labels_*_validate58.csv")
    ap.add_argument("--splits", type=int, default=200)
    a = ap.parse_args()
    aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
    u = list(aud.index)
    Y = aud.loc[u, ["official_" + t for t in T]].values.astype(int)

    def auc_cols(pred, idx):
        y = Y[idx]
        r = rankdata(pred[idx], axis=0)
        n1 = y.sum(0).astype(float)
        n0 = len(idx) - n1
        with np.errstate(invalid="ignore", divide="ignore"):
            out = ((r * y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
        out[(n1 == 0) | (n0 == 0)] = np.nan
        return out

    def load(f, cols=T):
        return pd.read_csv(D / f).set_index("StudyInstanceUID").loc[u, cols].values.astype(float)

    Q = load("03_report_labels_qwen24t-med.csv")
    M15 = load("16_medgemma15_corpus_all.csv")
    bi = [T.index(t) for t in B5]
    fi = [T.index(t) for t in F3]
    BASE = Q.copy()
    BASE[:, bi] = sg((lg(Q[:, bi]) + lg(M15[:, bi])) / 2)
    if a.base == "oof":
        import sweep_new_runs as sw
        fl = []
        for k in range(5):
            z = np.load(sw.best_npz(f"convnext_base_25d_18slices_f{k}", "audit"), allow_pickle=True)
            fl.append(pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T).loc[u].values)
        BASE = sg(0.7 * lg(BASE) + 0.3 * np.mean(fl, axis=0))
    if a.base == "w50":
        # v4 = the labels every current member trains on (data_code/22_make_convnext_mii_labels.py --w 0.5): the 58
        # are in no fold, so each teacher's audit logit is its 5 fold models' mean (that script's own estimate)
        import sweep_new_runs as sw
        tl = {}
        for fam, run in (("cnx", "convnext_base_25d_18slices_f{k}"), ("mii", "medimageinsight_davit_huge_336_v3_f{k}")):
            fl = []
            for k in range(5):
                z = np.load(sw.best_npz(run.format(k=k), "audit"), allow_pickle=True)
                fl.append(pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T).loc[u].values)
            tl[fam] = np.mean(fl, axis=0)
        BASE = sg(0.5 * lg(BASE) + 0.5 * (0.5 * tl["cnx"] + 0.5 * tl["mii"]))
    ALL = np.arange(len(u))
    base = np.nanmean(auc_cols(BASE, ALL))
    print(f"base = {a.base}: audit-58 macro AUROC {base:.4f}")

    if a.all_sources:
        for f in sorted(D.glob("03_report_labels_*_validate58.csv")):
            n = f.stem[len("03_report_labels_"):-len("_validate58")]
            ids = set(pd.read_csv(f, usecols=["StudyInstanceUID"]).StudyInstanceUID)
            if n != "qwen24t-med" and f.name not in SOURCES.values() and set(u) <= ids:   # partial (hold-out) files skipped
                SOURCES[n] = f.name
    X, fluid_only = {}, set()
    for n, f in SOURCES.items():
        if not (D / f).exists():
            print("  missing, skipped:", f)
            continue
        cols = pd.read_csv(D / f, nrows=1).columns
        if all(t in cols for t in T):
            X[n] = load(f)
        else:
            full = BASE.copy()
            full[:, fi] = load(f, F3)
            X[n] = full
            fluid_only.add(n)
    hs = [k for k in X if k.startswith("H") and k not in ("Hfluid", "Hall96R")]
    for p, q in itertools.combinations([k for k in X if k.startswith("H")], 2):
        X[f"{p}&{q}"] = sg((lg(X[p]) + lg(X[q])) / 2)
    if hs:
        X["Hall"] = sg(np.mean([lg(X[k]) for k in hs], axis=0))
    if "G27a" in X and "G27b" in X:
        X["G27a&G27b"] = sg((lg(X["G27a"]) + lg(X["G27b"])) / 2)
    if "G4fl" in X and "G4img" in X:
        X["G4img&G4fl"] = sg((lg(X["G4img"]) + lg(X["G4fl"])) / 2)
    print(f"{len(X)} sources ({len(fluid_only)} fluid-only): {', '.join(list(X)[:14])} ...")

    def mix(x, w, space):
        if space == "logit":
            return sg((1 - w) * lg(BASE) + w * lg(x))
        if space == "prob":
            return (1 - w) * BASE + w * x
        return (1 - w) * rankdata(BASE, axis=0) / len(u) + w * rankdata(x, axis=0) / len(u)

    colsets = {"F3": F3, "B5": B5, "all12": T}
    recipes, colmix = {}, {}
    for xn, x in X.items():
        for sp in ("logit", "prob", "rank"):
            for w in (0.1, 0.2, 0.3, 0.4, 0.5):
                m = mix(x, w, sp)
                colmix[(xn, sp, w)] = m
                ref = BASE if sp != "rank" else rankdata(BASE, axis=0) / len(u)
                for cn, cols in colsets.items():
                    if xn.split("&")[0] in fluid_only and cn != "F3":
                        continue
                    ci = [T.index(t) for t in cols]
                    P = ref.copy()
                    P[:, ci] = m[:, ci]
                    recipes[(xn, sp, w, cn)] = P
    print(f"{len(recipes)} recipes")
    ranked = sorted(((np.nanmean(auc_cols(P, ALL)), k) for k, P in recipes.items()), reverse=True)
    print(f"\nTop 15 in-sample (optimistic) vs base {base:.4f}:")
    for v, k in ranked[:15]:
        print(f"  {k[0]:20s} {k[1]:5s} w={k[2]:.1f} on {k[3]:5s} {v:.4f} ({v - base:+.4f})")
    print(f"  recipes above base in-sample: {sum(v > base + 1e-9 for v, _ in ranked)} of {len(ranked)}")
    print("\nFLUID 3 only, best mix per source (in-sample), base fluid mean "
          f"{np.nanmean(auc_cols(BASE, ALL)[fi]):.4f}:")
    for xn in X:
        v, k = max((np.nanmean(auc_cols(P, ALL)[fi]), k) for k, P in recipes.items()
                   if k[0] == xn and k[3] == "F3")
        print(f"  {xn:20s} {v:.4f} ({v - np.nanmean(auc_cols(BASE, ALL)[fi]):+.4f}) with {k[1]} w={k[2]}")
    cur = auc_cols(BASE, ALL)
    print("\nper finding, best single-column mix in-sample vs base:")
    for j, t in enumerate(T):
        v, k = max((auc_cols(P, ALL)[j], k) for k, P in colmix.items()
                   if not (k[0].split("&")[0] in fluid_only and t not in F3))
        print(f"  {t:17s} base {cur[j]:.3f} best {v:.3f} ({v - cur[j]:+.3f}) with {k}")

    rng = np.random.default_rng(0)
    keys = list(recipes)
    g_whole, g_fluid, g_pf, picks = [], [], [], []
    for _ in range(a.splits):
        perm = rng.permutation(len(u))
        A, B = perm[:29], perm[29:]
        if any(Y[A, j].sum() in (0, 29) or Y[B, j].sum() in (0, 29) for j in range(12)):
            continue
        best = max(keys, key=lambda k: np.nanmean(auc_cols(recipes[k], A)))
        picks.append(best)
        g_whole.append(np.nanmean(auc_cols(recipes[best], B)) - np.nanmean(auc_cols(BASE, B)))
        fk = [k for k in keys if k[3] == "F3"]
        bf = max(fk, key=lambda k: np.nanmean(auc_cols(recipes[k], A)[fi]))
        g_fluid.append(np.nanmean(auc_cols(recipes[bf], B)[fi]) - np.nanmean(auc_cols(BASE, B)[fi]))
        ca = auc_cols(BASE, A)
        colA = {k: auc_cols(P, A) for k, P in colmix.items()}
        out = BASE.copy()
        for j in range(12):
            ok = [k for k in colA if not (k[0].split("&")[0] in fluid_only and T[j] not in F3)]
            k = max(ok, key=lambda k: colA[k][j])
            if colA[k][j] > ca[j]:
                out[:, j] = colmix[k][:, j]
        g_pf.append(np.nanmean(auc_cols(out, B)) - np.nanmean(auc_cols(BASE, B)))
    g1, g2, g3 = map(np.array, (g_whole, g_fluid, g_pf))
    print(f"\nCROSS-FITTED ({len(g1)} splits: choose on 29, score on the other 29, vs base):")
    print(f"  best whole recipe:                 mean {g1.mean():+.4f}, positive in {100 * (g1 > 0).mean():.0f}%")
    print(f"  best FLUID-only recipe (fluid 3):  mean {g2.mean():+.4f} on the fluid mean, positive in {100 * (g2 > 0).mean():.0f}%")
    print(f"  best mix per finding (or keep):    mean {g3.mean():+.4f}, positive in {100 * (g3 > 0).mean():.0f}%")
    print("  most picked:", Counter(f"{k[0]} {k[1]} w{k[2]} {k[3]}" for k in picks).most_common(5))


if __name__ == "__main__":
    main()
