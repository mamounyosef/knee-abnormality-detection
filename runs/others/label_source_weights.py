"""label_source_weights.py - per-class weights over the FOUR label sources, cross-fitted on audit-58.

Sources, all on the logit scale, per study and finding:
  Qwen      03_report_labels_qwen24t-med        (report LLM, v2)
  MedGemma  16_medgemma15_corpus_all            (report LLM)
  ConvNeXt  convnext_base_25d_18slices_f*       (5-fold mean on the audit studies)
  MII       medimageinsight_davit_huge_336_v3_f* (5-fold mean on the audit studies)
label logit = sum_s w_s * logit_s, weights on a simplex grid (step 0.1, 286 combinations).

Compared, all scored on held-out studies only (4-part splits of the 58, repeated):
  current 22_*     the prespecified recipe (Qwen/MedGemma averaged on 5 findings, then 0.7 / 0.3
                   with the ConvNeXt + MII mean), no fitting at all
  global           one weight vector for all 12 findings, chosen on the other parts
  per-class        one weight vector per finding, chosen on the other parts

    python runs/others/label_source_weights.py [--reps 100]
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mii_vs_convnext_oof as m  # noqa: E402

SRC = ["Qwen", "MedGemma", "ConvNeXt", "MII"]
B5 = ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]


def simplex(k, step):
    n = round(1 / step)
    return np.array([c for c in itertools.product(range(n + 1), repeat=k) if sum(c) == n]) / n


def auc_rows(P, y):
    """AUROC of every row of P (combos x n) against y (n,); nan if y is one class."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(P.shape[0], np.nan)
    r = rankdata(P, axis=1)
    return (r[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    aud = pd.read_csv(m.D / "05_audit58.csv").set_index("StudyInstanceUID")
    u = list(aud.index)
    Y = (aud.loc[u, ["official_" + t for t in m.T]].values >= 0.5).astype(int)
    S = {"Qwen": m.lg(pd.read_csv(m.D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID").loc[u, m.T].values),
         "MedGemma": m.lg(pd.read_csv(m.D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID").loc[u, m.T].values)}
    for fam in ("ConvNeXt", "MII"):
        S[fam] = np.mean([m.load(m.FAM[fam].format(k), "audit")[0].loc[u].values for k in range(5)], 0)
    X = np.stack([S[s] for s in SRC])                      # (4, n, 12)
    G = simplex(len(SRC), 0.1)                             # (286, 4)
    n = len(u)

    # the prespecified 22_* recipe as source weights per finding
    cur = np.zeros((12, 4))
    for j, t in enumerate(m.T):
        q, mg = (0.35, 0.35) if t in B5 else (0.7, 0.0)
        cur[j] = [q, mg, 0.15, 0.15]

    def score(W, idx):                                     # W (12, 4) -> macro AUROC on idx
        return np.nanmean([auc_rows((W[j] @ X[:, idx, j])[None], Y[idx, j])[0] for j in range(12)])

    base = score(cur, np.arange(n))
    print(f"each source alone (audit-58): " + "  ".join(
        f"{s} {np.nanmean([auc_rows(X[i][:, j][None], Y[:, j])[0] for j in range(12)]):.4f}" for i, s in enumerate(SRC)))
    print(f"current 22_* recipe: {base:.4f}\n")

    # in sample, per class: best weights (for reading only, this overfits)
    print("in sample per finding (overfits; for reading only): best weights Qwen/MedGemma/ConvNeXt/MII")
    for j, t in enumerate(m.T):
        A = auc_rows(np.einsum("gs,sn->gn", G, X[:, :, j]), Y[:, j])
        b = int(np.nanargmax(A))
        print(f"  {t:17s} {'/'.join(f'{w:.1f}' for w in G[b])}  {A[b]:.4f}  vs current "
              f"{auc_rows((cur[j] @ X[:, :, j])[None], Y[:, j])[0]:.4f}")

    rng = np.random.default_rng(0)
    d = {"global": [], "per-class": []}
    picks = []
    for rep in range(a.reps):
        parts = np.array_split(rng.permutation(n), 4)
        out = {k: np.zeros((n, 12)) for k in d}
        for h in parts:
            tr = np.setdiff1d(np.arange(n), h)
            A = np.stack([auc_rows(np.einsum("gs,sn->gn", G, X[:, tr, j]), Y[tr, j]) for j in range(12)])  # (12, G)
            g = int(np.nanargmax(np.nanmean(A, 0)))
            for j in range(12):
                pj = int(np.nanargmax(A[j])) if not np.isnan(A[j]).all() else g
                out["global"][h, j] = G[g] @ X[:, h, j]
                out["per-class"][h, j] = G[pj] @ X[:, h, j]
                picks.append((j, pj))
        cur_out = np.stack([cur[j] @ X[:, :, j] for j in range(12)], 1)
        ref = np.nanmean([auc_rows(cur_out[:, j][None], Y[:, j])[0] for j in range(12)])
        for k in d:
            d[k].append(np.nanmean([auc_rows(out[k][:, j][None], Y[:, j])[0] for j in range(12)]) - ref)
    print(f"\ncross-fitted ({a.reps} random 4-part splits), held-out minus the current 22_* recipe:")
    for k, v in d.items():
        v = np.array(v)
        print(f"  {k:10s} mean {v.mean():+.4f}  [{np.percentile(v, 5):+.4f}, {np.percentile(v, 95):+.4f}]"
              f"  better in {100 * (v > 0).mean():.0f}% of splits")


if __name__ == "__main__":
    main()
