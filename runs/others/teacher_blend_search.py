"""teacher_blend_search.py - can the ConvNeXt + MII teacher labels be tuned beyond a flat 50/50 at 0.3?

label = sigmoid((1 - w) * logit(v3) + w * (a * MII + (1 - a) * ConvNeXt))     (5-fold mean logits)

Judged on audit-58, the only labels the teachers never saw. 58 studies is small, so every
tuned variant is CROSS-FITTED: the 58 are split into k parts, the setting is chosen on the
other parts and applied to the held-out part, and the pooled held-out predictions are scored.
Repeated over many random splits. A tuned variant must beat the fixed 50/50 at 0.3 out of
sample, not just in sample.

    python runs/others/teacher_blend_search.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mii_vs_convnext_oof as m  # noqa: E402

W_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
A_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def setup():
    import pandas as pd
    aud = pd.read_csv(m.D / "05_audit58.csv").set_index("StudyInstanceUID")
    u = list(aud.index)
    Y = (aud.loc[u, ["official_" + t for t in m.T]].values >= 0.5).astype(int)
    q = pd.read_csv(m.D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID").loc[u, m.T].values
    mg = pd.read_csv(m.D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID").loc[u, m.T].values
    bi = [m.T.index(t) for t in ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]]
    c = q.copy()
    c[:, bi] = m.sig(0.5 * m.lg(q[:, bi]) + 0.5 * m.lg(mg[:, bi]))
    T = {}
    for fam in ("ConvNeXt", "MII"):
        T[fam] = np.mean([m.load(m.FAM[fam].format(k), "audit")[0].loc[u].values for k in range(5)], 0)
    return Y, m.lg(c), T


def lab(L, T, w, a):
    return (1 - w) * L + w * (a * T["MII"] + (1 - a) * T["ConvNeXt"])   # logit scale, rank-equivalent


def auc_col(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def main():
    Y, L, T = setup()
    n = len(Y)
    flat = np.nanmean([auc_col(Y[:, j], lab(L, T, 0.3, 0.5)[:, j]) for j in range(12)])
    print(f"fixed 50/50 at w=0.3 (prespecified): {flat:.4f}\n")

    print("in sample, global (w rows, MII share a columns):")
    print("        " + "  ".join(f"a={a:<5}" for a in A_GRID))
    for w in W_GRID:
        print(f"  w={w:<4} " + "  ".join(
            f"{np.nanmean([auc_col(Y[:, j], lab(L, T, w, a)[:, j]) for j in range(12)]):.4f} " for a in A_GRID))

    # per-class in-sample table: the best (w, a) per finding, for reading only
    print("\nin sample, per finding: best (w, a) and its AUROC vs the fixed 50/50 at 0.3")
    for j, t in enumerate(m.T):
        best = max(((w, a) for w in W_GRID for a in A_GRID), key=lambda p: auc_col(Y[:, j], lab(L, T, *p)[:, j]))
        print(f"  {t:17s} w={best[0]:<4} a={best[1]:<5} {auc_col(Y[:, j], lab(L, T, *best)[:, j]):.4f}"
              f"  vs {auc_col(Y[:, j], lab(L, T, 0.3, 0.5)[:, j]):.4f}")

    # cross-fitted: global (w, a) and per-class (w, a), against the fixed setting
    rng = np.random.default_rng(0)
    res = {"global": [], "per-class": [], "per-class a only (w=0.3)": []}
    for rep in range(200):
        parts = np.array_split(rng.permutation(n), 4)
        out = {k: np.zeros((n, 12)) for k in res}
        for h in parts:
            tr = np.setdiff1d(np.arange(n), h)

            def score(p, cols):
                return np.nanmean([auc_col(Y[tr, j], lab(L[tr], {f: v[tr] for f, v in T.items()}, *p)[:, j])
                                   for j in cols])
            g = max(((w, a) for w in W_GRID for a in A_GRID), key=lambda p: score(p, range(12)))
            out["global"][h] = lab(L[h], {f: v[h] for f, v in T.items()}, *g)
            for j in range(12):
                pc = max(((w, a) for w in W_GRID for a in A_GRID), key=lambda p: score(p, [j]))
                out["per-class"][h, j] = lab(L[h], {f: v[h] for f, v in T.items()}, *pc)[:, j]
                pa = max(((0.3, a) for a in A_GRID), key=lambda p: score(p, [j]))
                out["per-class a only (w=0.3)"][h, j] = lab(L[h], {f: v[h] for f, v in T.items()}, *pa)[:, j]
        for k in res:
            res[k].append(np.nanmean([auc_col(Y[:, j], out[k][:, j]) for j in range(12)]) - flat)
    print("\ncross-fitted (4 parts, 200 random splits), held-out minus the fixed 50/50 at 0.3:")
    for k, d in res.items():
        d = np.array(d)
        print(f"  {k:26s} mean {d.mean():+.4f}  [{np.percentile(d, 5):+.4f}, {np.percentile(d, 95):+.4f}]"
              f"  better in {100 * (d > 0).mean():.0f}% of splits")


if __name__ == "__main__":
    main()
