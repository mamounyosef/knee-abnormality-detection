"""confidence_weighted_teacher.py - trust the report where it is sure, the image teachers where it is not.

label logit = (1 - w) * v3 + w * teacher, teacher = mean of the ConvNeXt and MII fold logits, where
w depends on how sure the report label is: w_hi when |v3 logit| < t (the report is unsure or silent),
w_lo otherwise. The fixed w = 0.5 (22_*-w50, LB 0.927) is the reference.

Scored on audit-58; every tuned setting is CROSS-FITTED (chosen on 3/4 of the studies, scored on the
held-out quarter, pooled, 200 random splits) so in-sample luck cannot pass as a gain.

    python runs/others/confidence_weighted_teacher.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_source_weights as lw  # noqa: E402
m = lw.m

GRID = [(t, lo, hi) for t, lo, hi in itertools.product([0.5, 1.0, 1.5, 2.0, 3.0], [0.2, 0.3, 0.4, 0.5],
                                                        [0.5, 0.6, 0.7, 0.8, 0.9]) if hi >= lo]


def main():
    aud = pd.read_csv(m.D / "05_audit58.csv").set_index("StudyInstanceUID")
    u = list(aud.index)
    Y = (aud.loc[u, ["official_" + t for t in m.T]].values >= 0.5).astype(int)
    q = pd.read_csv(m.D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID").loc[u, m.T].values
    mg = pd.read_csv(m.D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID").loc[u, m.T].values
    bi = [m.T.index(t) for t in lw.B5]
    v3 = m.lg(q)
    v3[:, bi] = 0.5 * m.lg(q[:, bi]) + 0.5 * m.lg(mg[:, bi])
    C = np.mean([m.load(m.FAM["ConvNeXt"].format(k), "audit")[0].loc[u].values for k in range(5)], 0)
    M = np.mean([m.load(m.FAM["MII"].format(k), "audit")[0].loc[u].values for k in range(5)], 0)
    T = (C + M) / 2

    def lab(idx, t, lo, hi):
        w = np.where(np.abs(v3[idx]) < t, hi, lo)
        return (1 - w) * v3[idx] + w * T[idx]

    def mac(idx, P):
        return np.nanmean([lw.auc_rows(P[:, j][None], Y[idx, j])[0] for j in range(12)])

    allx = np.arange(len(u))
    base = mac(allx, lab(allx, 99, 0.5, 0.5))
    print(f"fixed w=0.5 (w50): {base:.4f}")
    print(f"share of cells where the report is 'unsure' (|logit| < t): " + ", ".join(
        f"t={t}: {100 * (np.abs(v3) < t).mean():.0f}%" for t in (0.5, 1.0, 1.5, 2.0, 3.0)))
    ins = sorted(((mac(allx, lab(allx, *g)), g) for g in GRID), reverse=True)[:5]
    print("best in sample (overfits, for reading only):")
    for s, g in ins:
        print(f"  t={g[0]} w_lo={g[1]} w_hi={g[2]}: {s:.4f} ({s - base:+.4f})")

    rng = np.random.default_rng(0)
    d, picks = [], []
    for _ in range(200):
        parts = np.array_split(rng.permutation(len(u)), 4)
        out = np.zeros_like(v3)
        for h in parts:
            tr = np.setdiff1d(allx, h)
            g = max(GRID, key=lambda g: mac(tr, lab(tr, *g)))
            picks.append(g)
            out[h] = lab(h, *g)
        d.append(mac(allx, out) - base)
    d = np.array(d)
    print(f"\ncross-fitted, held-out minus fixed w=0.5: {d.mean():+.4f} "
          f"[{np.percentile(d, 5):+.4f}, {np.percentile(d, 95):+.4f}], better in {100 * (d > 0).mean():.0f}% of splits")
    from collections import Counter
    print("most often chosen:", Counter(picks).most_common(3))


if __name__ == "__main__":
    main()
